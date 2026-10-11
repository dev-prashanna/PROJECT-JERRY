"""Multimodal Memory Engine — the stable facade over the whole MMS.

AURA components depend on this class only; SQLite, backends and indexes stay
private.  Methods are synchronous to match AURA's threaded pipeline; heavy
work is pushed to :class:`MemoryJobScheduler`.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from config.settings import MemorySettings, get_settings
from src.memory.context_builder import build_memory_block, direct_answer
from src.memory.entity_resolver import EntityResolver
from src.memory.event_models import (
    ConversationEvent,
    Episode,
    EventType,
    RecallResult,
    RecallStatus,
    RetrievedEvent,
    TemporalHint,
)
from src.memory.conversation_recorder import ConversationRecorder
from src.memory.conversation_retriever import ConversationRetriever
from src.memory.episode_builder import EpisodeBuilder
from src.memory.indexing_worker import MemoryIndexingWorker
from src.memory.text_embeddings import (
    HashTextEmbedder,
    TextEmbeddingIndex,
    create_text_embedder,
)
from src.memory.temporal_resolver import parse_temporal
from src.memory.exceptions import (
    ConflictingMemory,
    ConsentRequired,
    FaceNotDetected,
    FaceRecognitionUnavailable,
    InvalidMemoryRequest,
    MemoryEntityNotFound,
    MemoryError,
    MemoryOperationCancelled,
    ObjectNotLocated,
    SnapshotExpired,
)
from src.memory.fact_extractor import (
    StatementClass,
    extract_facts,
)
from src.memory.face_memory import (
    FaceMatchThresholds,
    FaceMemoryService,
    create_face_backend,
)
from src.memory.models import (
    EntityType,
    EntityStatus,
    FactSourceType,
    FactStatus,
    MatchStatus,
    MemoryContext,
    MemoryEntity,
    MemoryFact,
    MemoryOperationResult,
    SnapshotReference,
    new_id,
    utcnow,
)
from src.memory.object_isolation import (
    CenterPriorGrounder,
    create_grounder,
    create_segmenter,
    crop_with_margin,
)
from src.memory.permissions import issue_consent_token
from src.memory.resource_scheduler import MemoryJob, MemoryJobPriority, MemoryJobScheduler
from src.memory.retriever import MemoryRetriever, RetrieverLimits
from src.memory.store import MemoryStore, ensure_private_dir, expand_path
from src.memory.vector_index import (
    ExactNumpyIndex,
    InvalidEmbedding,
    deserialize_vector,
    l2_normalize,
    serialize_vector,
)
from src.memory.visual_embeddings import VisualEmbedder, create_embedder
from src.utils.logger import get_logger

log = get_logger("memory.engine")


@dataclass
class _JobHandler:
    engine: "MultimodalMemoryEngine"

    def __call__(self, job: MemoryJob) -> Any:
        op = job.operation
        payload = job.payload or {}
        if op == "remember_fact":
            return self.engine._remember_fact_payload(payload)
        if op == "learn_object":
            return self.engine._learn_object_payload(payload)
        if op == "embed_event":
            return self.engine._embed_event_payload(payload)
        if op == "build_episode":
            return self.engine._build_episode_payload(payload)
        if op == "rebuild_indexes":
            return {"faces": self.engine.face_service.rebuild_index() if self.engine.face_service else 0}
        raise InvalidMemoryRequest(f"unknown memory job {op}")


class MultimodalMemoryEngine:
    """Narrow public interface used by the orchestrator, tools and GUI."""

    def __init__(self, settings: Optional[MemorySettings] = None, *, llm_client=None):
        self.settings = settings or get_settings().memory
        self.llm_client = llm_client
        self._lock = threading.RLock()
        self._initialized = False
        self.store: Optional[MemoryStore] = None
        self.resolver: Optional[EntityResolver] = None
        self.retriever: Optional[MemoryRetriever] = None
        self.face_service: Optional[FaceMemoryService] = None
        self.embedder: Optional[VisualEmbedder] = None
        self.grounder = None
        self.segmenter = None
        self.object_index: Optional[ExactNumpyIndex] = None
        self.scheduler: Optional[MemoryJobScheduler] = None
        self._last_error: Optional[str] = None
        # -- PECM components ---------------------------------------------
        self.recorder: Optional[ConversationRecorder] = None
        self.episode_builder: Optional[EpisodeBuilder] = None
        self.conversation_retriever: Optional[ConversationRetriever] = None
        self.text_embedder = None
        self.text_index: Optional[TextEmbeddingIndex] = None
        self.index_worker: Optional[MemoryIndexingWorker] = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            if not self.settings.enabled:
                log.info("multimodal memory disabled by configuration")
                self._initialized = True
                return
            started = time.perf_counter()
            ensure_private_dir(expand_path(self.settings.db_path).parent)
            self.store = MemoryStore(self.settings.db_path)
            self.resolver = EntityResolver(self.store)
            limits = RetrieverLimits(
                max_entities=self.settings.max_retrieved_entities,
                max_facts=self.settings.max_retrieved_facts,
                token_budget=self.settings.context_token_budget,
            )
            self.retriever = MemoryRetriever(self.store, limits)

            if self.settings.face_recognition_enabled:
                try:
                    backend = create_face_backend(
                        self.settings.face_backend,
                        model_dir=self.settings.face_model_dir,
                        allow_download=self.settings.face_download_models,
                    )
                    self.face_service = FaceMemoryService(
                        self.store,
                        backend,
                        FaceMatchThresholds(
                            match_distance=self.settings.face_match_threshold,
                            ambiguity_margin=self.settings.face_ambiguity_margin,
                        ),
                        require_confirmation=self.settings.require_enrollment_confirmation,
                    )
                    self.face_service.rebuild_index()
                    self.retriever.face_index = self.face_service._index
                except FaceRecognitionUnavailable as exc:
                    log.warning("face memory unavailable: %s", exc)
                    self.face_service = None
                    self._last_error = str(exc)

            if self.settings.object_memory_enabled:
                self.embedder = create_embedder(
                    self.settings.object_embedder,
                    allow_download=self.settings.clip_allow_download,
                    device=self.settings.object_embedder_device,
                )
                self.grounder = create_grounder(self.settings.object_grounder)
                self.segmenter = create_segmenter(self.settings.object_segmentation_backend)
                self.object_index = ExactNumpyIndex(
                    self.embedder.model_id,
                    self.embedder.embedding_dimension,
                    metric="cosine",
                )
                self._load_object_index()
                self.retriever.object_index = self.object_index

            # -- PECM: conversation recorder / episodes / hybrid search ----
            if self.settings.conversation_memory_enabled:
                self.recorder = ConversationRecorder(self.store, enabled=True)
                self.episode_builder = EpisodeBuilder(self.store, llm_client=self.llm_client)
                if self.settings.memory_semantic_search_enabled:
                    self.text_embedder = create_text_embedder(
                        self.settings.memory_text_embedder,
                        model_name=self.settings.memory_text_embedding_model,
                        allow_download=False,
                    )
                    self._load_text_index()
                self.conversation_retriever = ConversationRetriever(
                    self.store,
                    text_index=self.text_index,
                    embedder=self.text_embedder,
                    max_results=self.settings.memory_max_retrieval_results,
                )

            if self.settings.background_indexing:
                self.scheduler = MemoryJobScheduler(
                    _JobHandler(self),
                    max_pending=self.settings.max_pending_jobs,
                )
                self.scheduler.start()
                self.index_worker = MemoryIndexingWorker(self)
                self.index_worker.start()

            self._initialized = True
            log.info(
                "memory engine ready in %.2fs (db=%s)",
                time.perf_counter() - started,
                self.settings.db_path,
            )

    def shutdown(self) -> None:
        with self._lock:
            if self.index_worker is not None:
                try:
                    self.index_worker.stop(timeout_s=2.0)
                except Exception:  # noqa: BLE001
                    log.debug("index worker stop failed")
                self.index_worker = None
            if self.scheduler is not None:
                self.scheduler.stop(timeout_s=2.0)
                self.scheduler = None
            if self.store is not None:
                try:
                    self.store.close()
                except Exception:  # noqa: BLE001
                    log.exception("memory store close failed")
                self.store = None
            self._initialized = False

    @property
    def enabled(self) -> bool:
        return bool(self.settings.enabled and self._initialized and self.store is not None)

    def _require_store(self) -> MemoryStore:
        if self.store is None:
            raise MemoryError("memory engine is not initialized")
        return self.store

    def _load_object_index(self) -> None:
        assert self.store is not None and self.object_index is not None and self.embedder is not None
        for emb_id, entity_id, blob, dim, model_id, _crop in self.store.all_object_embeddings():
            if model_id != self.embedder.model_id or dim != self.embedder.embedding_dimension:
                continue
            try:
                vec = deserialize_vector(blob, dim)
            except InvalidEmbedding as exc:
                log.warning("skipping corrupt object embedding %s: %s", emb_id, exc)
                continue
            self.object_index.add(emb_id, vec, {"entity_id": entity_id})

    # ------------------------------------------------------------------
    # facts
    # ------------------------------------------------------------------
    def remember_fact(self, request) -> MemoryOperationResult:
        t0 = time.perf_counter()
        try:
            result = self._remember_fact_payload(
                {
                    "transcript": request.transcript,
                    "turn_id": request.turn_id,
                    "use_llm": request.use_llm,
                    "request_id": request.request_id,
                }
            )
            return MemoryOperationResult(
                ok=True,
                operation="remember_fact",
                message=result.get("message", "stored"),
                payload=result,
                duration_s=time.perf_counter() - t0,
            )
        except MemoryError as exc:
            return self._failure("remember_fact", exc, t0)

    def _remember_fact_payload(self, payload: dict) -> dict:
        store = self._require_store()
        assert self.resolver is not None
        transcript = str(payload.get("transcript") or "").strip()
        turn_id = str(payload.get("turn_id") or new_id("turn"))
        if not transcript:
            raise InvalidMemoryRequest("empty transcript")

        extraction = extract_facts(
            transcript,
            client=self.llm_client,
            use_llm=bool(payload.get("use_llm")) and self.settings.use_llm_fact_extraction,
        )

        stored: list[dict] = []
        corrected: list[dict] = []

        for fact in extraction.corrections:
            resolved = self.resolver.resolve_subject(fact.subject)
            target_entity = resolved.entity
            predicate = fact.predicate
            prior = None
            if target_entity is not None:
                candidates = [
                    f
                    for f in store.facts_for_entity(target_entity.entity_id)
                    if f.predicate == predicate
                    or f.predicate.endswith(predicate)
                    or predicate.endswith(f.predicate)
                ]
                prior = candidates[0] if candidates else None
            if prior is None:
                # Search globally: same predicate family, or any recent fact
                # whose stored value differs from the correction.
                lowered_obj = fact.object_value.lower()
                family = [
                    f
                    for f in store.search_facts(predicate=None, limit=100)
                    if f.predicate == predicate
                    or f.predicate.endswith(predicate)
                    or predicate.endswith(f.predicate)
                    or predicate in {"name", "project_name"}
                ]
                prior = next(
                    (f for f in family if f.object_value.lower() != lowered_obj),
                    None,
                )
            if prior is None:
                raise ConflictingMemory(
                    "no prior fact found to correct; store a new fact instead"
                )
            if prior.object_value.lower() == fact.object_value.lower():
                return {
                    "stored": [],
                    "corrected": [],
                    "message": "the stored value already matches",
                    "classification": extraction.classification.value,
                }
            new_fact = MemoryFact(
                subject_entity_id=prior.subject_entity_id,
                predicate=prior.predicate,
                object_value=fact.object_value,
                source_transcript=transcript,
                source_turn_id=turn_id,
                source_type=FactSourceType.USER_ASSERTION,
                extraction_method=extraction.method,
            )
            store.supersede_fact(prior.fact_id, new_fact)
            corrected.append(
                {
                    "fact_id": new_fact.fact_id,
                    "predicate": new_fact.predicate,
                    "object_value": new_fact.object_value,
                    "supersedes": prior.fact_id,
                }
            )

        for fact in extraction.facts:
            entity = None
            if fact.subject in {"current_speaker", "i", "me", "myself"}:
                entity = self.resolver.get_or_create(
                    "current speaker",
                    EntityType.PERSON,
                    provisional=True,
                    extra_aliases=["the user"],
                )
            else:
                resolved = self.resolver.resolve_subject(fact.subject)
                if resolved.entity is not None:
                    entity = resolved.entity
                else:
                    display = fact.subject.replace("_", " ").strip()
                    etype = EntityType.PROJECT if "project" in fact.predicate or "project" in display else (
                        EntityType.OBJECT if display.startswith("my ") else EntityType.CONCEPT
                    )
                    entity = self.resolver.get_or_create(display, etype, provisional=False)

            # De-duplicate: same subject+predicate+value already active
            existing = [
                f
                for f in store.facts_for_entity(entity.entity_id)
                if f.predicate == fact.predicate and f.object_value.lower() == fact.object_value.lower()
            ]
            if existing:
                continue

            # Supersede a conflicting prior value for the same predicate
            conflicting = [
                f for f in store.facts_for_entity(entity.entity_id) if f.predicate == fact.predicate
            ]
            mf = MemoryFact(
                subject_entity_id=entity.entity_id,
                predicate=fact.predicate,
                object_value=fact.object_value,
                source_transcript=transcript,
                source_turn_id=turn_id,
                source_type=FactSourceType.USER_ASSERTION,
                extraction_method=extraction.method,
            )
            if conflicting:
                store.supersede_fact(conflicting[0].fact_id, mf)
                corrected.append(
                    {
                        "fact_id": mf.fact_id,
                        "predicate": mf.predicate,
                        "object_value": mf.object_value,
                        "supersedes": conflicting[0].fact_id,
                    }
                )
            else:
                store.insert_fact(mf)
                stored.append(
                    {
                        "fact_id": mf.fact_id,
                        "predicate": mf.predicate,
                        "object_value": mf.object_value,
                        "subject_entity_id": entity.entity_id,
                        "subject_name": entity.canonical_name,
                    }
                )
            store.touch_entity(entity.entity_id)

        if not stored and not corrected:
            return {
                "stored": [],
                "corrected": [],
                "message": extraction.skipped_reason or "nothing durable to remember",
                "classification": extraction.classification.value,
            }

        parts = []
        if stored:
            parts.append(f"stored {len(stored)} fact(s)")
        if corrected:
            parts.append(f"corrected {len(corrected)} fact(s)")
        return {
            "stored": stored,
            "corrected": corrected,
            "message": ", ".join(parts),
            "classification": extraction.classification.value,
            "method": extraction.method,
        }

    def correct_fact(self, request) -> MemoryOperationResult:
        t0 = time.perf_counter()
        try:
            result = self._remember_fact_payload(
                {
                    "transcript": request.transcript,
                    "turn_id": request.turn_id,
                    "use_llm": True,
                    "request_id": request.request_id,
                }
            )
            if not result.get("corrected") and not result.get("stored"):
                raise ConflictingMemory("I could not find a prior fact to correct")
            return MemoryOperationResult(
                ok=True,
                operation="correct_fact",
                message=result.get("message", "corrected"),
                payload=result,
                duration_s=time.perf_counter() - t0,
            )
        except MemoryError as exc:
            return self._failure("correct_fact", exc, t0)

    # ------------------------------------------------------------------
    # faces
    # ------------------------------------------------------------------
    def enroll_person(self, request) -> MemoryOperationResult:
        t0 = time.perf_counter()
        try:
            if self.face_service is None:
                raise FaceRecognitionUnavailable("face recognition is not enabled")
            if request.frame is None:
                raise InvalidMemoryRequest("no camera frame supplied for enrollment")
            entity, match, face_count = self.face_service.enroll_identity(
                request.entity_name,
                _frame_to_bgr(request.frame),
                transcript=request.consent_context.get("transcript", ""),
                selected_face_id=request.selected_face_id,
                consent_token=request.consent_token,
                frame_id=request.snapshot.frame_id if request.snapshot else "",
                request_id=request.request_id,
            )
            if self.retriever is not None:
                self.retriever.face_index = self.face_service._index
            return MemoryOperationResult(
                ok=True,
                operation="enroll_person",
                message=f"enrolled {entity.canonical_name}",
                payload={
                    "entity_id": entity.entity_id,
                    "name": entity.canonical_name,
                    "faces_seen": face_count,
                    "prior_match": match.status.value,
                },
                duration_s=time.perf_counter() - t0,
            )
        except MemoryError as exc:
            return self._failure("enroll_person", exc, t0)

    def recognize_faces(self, request) -> MemoryOperationResult:
        t0 = time.perf_counter()
        try:
            if self.face_service is None:
                raise FaceRecognitionUnavailable("face recognition is not enabled")
            if request.frame is None:
                raise InvalidMemoryRequest("no camera frame supplied")
            faces = self.face_service.extract_embeddings(_frame_to_bgr(request.frame))
            if not faces:
                raise FaceNotDetected("no face was detected in the snapshot")
            results = []
            best = None
            for face in faces:
                match = self.face_service.match_face(face.embedding)
                results.append(match.model_dump(mode="json"))
                if match.status == MatchStatus.MATCHED and best is None:
                    best = match
            payload: dict[str, Any] = {"matches": results, "face_count": len(faces)}
            if best is not None and best.entity_id:
                facts = self.store.facts_for_entity(best.entity_id) if self.store else []
                payload["entity_id"] = best.entity_id
                payload["entity_name"] = best.entity_name
                payload["facts"] = [f.model_dump(mode="json") for f in facts]
                message = f"I think you are {best.entity_name}, based on the face I have enrolled."
            elif any(r.get("status") == "ambiguous" for r in results):
                message = "I see a face that is close to someone I know, but I am not certain who."
            else:
                message = "I do not recognise this face."
            return MemoryOperationResult(
                ok=True,
                operation="recognize_faces",
                message=message,
                payload=payload,
                duration_s=time.perf_counter() - t0,
            )
        except MemoryError as exc:
            return self._failure("recognize_faces", exc, t0)

    def issue_enrollment_consent(self, subject_description: str) -> MemoryOperationResult:
        t0 = time.perf_counter()
        try:
            store = self._require_store()
            consent = issue_consent_token(store, subject_description)
            return MemoryOperationResult(
                ok=True,
                operation="issue_enrollment_consent",
                message="consent recorded",
                payload={"consent_id": consent.consent_id},
                duration_s=time.perf_counter() - t0,
            )
        except MemoryError as exc:
            return self._failure("issue_enrollment_consent", exc, t0)

    # ------------------------------------------------------------------
    # objects
    # ------------------------------------------------------------------
    def enroll_object(self, request) -> MemoryOperationResult:
        t0 = time.perf_counter()
        try:
            result = self._learn_object_payload(
                {
                    "name": request.object_name,
                    "frame": request.frame,
                    "text_query": request.text_query,
                    "request_id": request.request_id,
                }
            )
            return MemoryOperationResult(
                ok=True,
                operation="enroll_object",
                message=result.get("message", "saved"),
                payload=result,
                duration_s=time.perf_counter() - t0,
            )
        except MemoryError as exc:
            return self._failure("enroll_object", exc, t0)

    def _learn_object_payload(self, payload: dict) -> dict:
        store = self._require_store()
        if self.resolver is None or self.embedder is None:
            raise InvalidMemoryRequest("object memory is unavailable")
        if not self.settings.object_memory_enabled:
            raise InvalidMemoryRequest("object memory is disabled")
        name = str(payload.get("name") or "").strip()
        frame = payload.get("frame")
        if not name:
            raise InvalidMemoryRequest("object name is required")
        if frame is None:
            raise InvalidMemoryRequest("no camera frame supplied")

        image = _frame_to_bgr(frame)
        text_query = str(payload.get("text_query") or name)
        regions = []
        if self.grounder is not None:
            try:
                regions = self.grounder.locate(image, text_query)
            except Exception as exc:  # noqa: BLE001
                log.warning("grounder failed: %s", exc)
                regions = []
        if not regions:
            h, w = image.shape[:2]
            from src.memory.object_isolation import GroundedRegion

            regions = [GroundedRegion(bbox_xyxy=(0, 0, w, h), label=name, source="full-frame")]

        if len(regions) > 1:
            return {
                "needs_disambiguation": True,
                "candidates": [
                    {"bbox_xyxy": list(r.bbox_xyxy), "label": r.label, "source": r.source}
                    for r in regions[:5]
                ],
                "message": "multiple candidate objects; select one",
            }

        region = regions[0]
        assert self.segmenter is not None
        crop = self.segmenter.segment(image, region.bbox_xyxy)
        if min(crop.shape[:2]) < 16:
            raise ObjectNotLocated("isolated region is too small to embed")
        embedding = self.embedder.encode(crop)

        hits = self.object_index.search(embedding, k=3) if self.object_index else []
        prior_entity = None
        best_score = hits[0][1] if hits else 0.0
        if hits and best_score >= self.settings.object_novelty_threshold:
            prior_entity = store.get_entity(hits[0][2].get("entity_id"))

        entity = self.resolver.get_or_create(name, EntityType.OBJECT, aliases=[name])

        # Crops (not full frames) are persisted for learned objects so the
        # GUI can show stored ROI examples; raw frames are never saved.
        asset_path = self._save_crop(crop, entity.entity_id)

        blob, dim, _ = serialize_vector(embedding)
        emb_id = store.insert_object_embedding(
            entity.entity_id,
            blob,
            dim,
            self.embedder.model_id,
            crop_path=str(asset_path),
        )
        if self.object_index is not None:
            self.object_index.add(emb_id, embedding, {"entity_id": entity.entity_id})
        store.touch_entity(entity.entity_id)
        obs_id = new_id("oobs")
        store.insert_observation(
            obs_id,
            frame_id=str(payload.get("request_id") or "unknown"),
            kind="object",
            bbox=region.bbox_xyxy,
            source_model=region.source,
            candidate_label=name,
            request_id=str(payload.get("request_id") or ""),
        )
        store.link_observation(entity.entity_id, obs_id)
        store.record_revision(
            entity_id=entity.entity_id,
            fact_id=None,
            action="enroll_object",
            detail=f"crop saved from {region.source}",
        )

        message = (
            f"saved {name} (similar to {prior_entity.canonical_name})"
            if prior_entity and prior_entity.entity_id != entity.entity_id
            else f"saved {name}"
        )
        return {
            "entity_id": entity.entity_id,
            "name": entity.canonical_name,
            "embedding_id": emb_id,
            "bbox_xyxy": list(region.bbox_xyxy),
            "grounding_source": region.source,
            "crop_path": str(asset_path),
            "prior_similarity": best_score if hits else None,
            "similar_entity_id": prior_entity.entity_id if prior_entity else None,
            "message": message,
        }

    def _save_crop(self, crop: np.ndarray, entity_id: str) -> Path:
        import cv2

        asset_dir = ensure_private_dir(expand_path(self.settings.asset_dir))
        path = asset_dir / f"{entity_id}_{new_id('crop')}.jpg"
        tmp = asset_dir / f".{path.stem}.part.jpg"
        ok = cv2.imwrite(str(tmp), crop)
        if not ok:
            raise ObjectNotLocated("failed to write visual asset")
        tmp.replace(path)
        try:
            import os

            os.chmod(path, 0o600)
        except OSError:  # pragma: no cover
            pass
        return path

    def recognize_objects(self, request) -> MemoryOperationResult:
        t0 = time.perf_counter()
        try:
            if self.embedder is None or request.frame is None:
                raise InvalidMemoryRequest("object memory unavailable or no frame")
            image = _frame_to_bgr(request.frame)
            text_query = request.text_query or "the object in view"
            regions = []
            try:
                regions = self.grounder.locate(image, text_query) if self.grounder else []
            except Exception:  # noqa: BLE001
                regions = []
            from src.memory.object_isolation import GroundedRegion

            if not regions:
                h, w = image.shape[:2]
                regions = [GroundedRegion(bbox_xyxy=(0, 0, w, h), label=text_query, source="full-frame")]
            if self.segmenter is not None:
                crop = self.segmenter.segment(image, regions[0].bbox_xyxy)
            else:
                crop = crop_with_margin(image, regions[0].bbox_xyxy)
            embedding = self.embedder.encode(crop)
            matches = (
                self.retriever.retrieve_by_object(
                    embedding, min_similarity=self.settings.object_novelty_threshold
                )
                if self.retriever
                else []
            )
            strong = [m for m in matches if m.status == MatchStatus.MATCHED]
            if strong and strong[0].entity_name:
                message = (
                    f"This looks similar to the {strong[0].entity_name} you showed me earlier."
                )
            elif matches and matches[0].status == MatchStatus.UNKNOWN:
                message = "I do not remember seeing this object before."
            else:
                message = "I am not sure whether I have seen this object before."
            return MemoryOperationResult(
                ok=True,
                operation="recognize_objects",
                message=message,
                payload={
                    "matches": [m.model_dump(mode="json") for m in matches],
                    "bbox_xyxy": list(regions[0].bbox_xyxy),
                    "grounding_source": regions[0].source,
                },
                duration_s=time.perf_counter() - t0,
            )
        except MemoryError as exc:
            return self._failure("recognize_objects", exc, t0)

    # ------------------------------------------------------------------
    # retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query: str, *, offline: bool = False, request_id: str = "") -> MemoryOperationResult:
        t0 = time.perf_counter()
        try:
            if self.retriever is None:
                raise MemoryError("memory engine is not initialized")
            ctx = self.retriever.retrieve_multimodal(text_query=(query or "").strip())
            block = build_memory_block(
                ctx,
                request_text=query,
                token_budget=self.settings.context_token_budget,
            )
            answer = direct_answer(ctx, offline=offline)
            return MemoryOperationResult(
                ok=True,
                operation="retrieve",
                message=answer,
                payload={
                    "context": ctx.model_dump(mode="json"),
                    "memory_block": block,
                    "direct_answer": answer,
                    "entity_count": len(ctx.matched_entities),
                    "fact_count": len(ctx.retrieved_facts),
                },
                duration_s=time.perf_counter() - t0,
            )
        except MemoryError as exc:
            return self._failure("retrieve", exc, t0)

    def retrieve_context(
        self,
        query: str,
        *,
        face_embedding: Optional[np.ndarray] = None,
        object_embedding: Optional[np.ndarray] = None,
        entity_hint: Optional[str] = None,
    ) -> MemoryContext:
        if self.retriever is None:
            return MemoryContext()
        return self.retriever.retrieve_multimodal(
            text_query=query,
            face_embedding=face_embedding,
            object_embedding=object_embedding,
            entity_hint=entity_hint,
        )

    def build_context_block(self, ctx: MemoryContext, request_text: str = "") -> str:
        return build_memory_block(
            ctx,
            request_text=request_text,
            token_budget=self.settings.context_token_budget,
        )

    # ------------------------------------------------------------------
    # listing / deletion
    # ------------------------------------------------------------------
    def list_memories(self) -> MemoryOperationResult:
        t0 = time.perf_counter()
        try:
            store = self._require_store()
            counts = store.counts()
            entities = [
                {
                    "entity_id": e.entity_id,
                    "name": e.canonical_name,
                    "type": e.entity_type.value,
                    "status": e.status.value,
                }
                for e in store.list_entities(limit=100)
            ]
            facts = [
                {
                    "fact_id": f.fact_id,
                    "predicate": f.predicate,
                    "object_value": f.object_value,
                    "subject_entity_id": f.subject_entity_id,
                }
                for f in store.list_facts(limit=100)
            ]
            enrolled = self.face_service.list_enrolled() if self.face_service else []
            return MemoryOperationResult(
                ok=True,
                operation="list_memories",
                message=(
                    f"{counts.get('entities', 0)} entities, "
                    f"{counts.get('declarative_memory', 0)} facts, "
                    f"{counts.get('face_embeddings', 0)} face embeddings, "
                    f"{counts.get('object_embeddings', 0)} object embeddings"
                ),
                payload={
                    "counts": counts,
                    "entities": entities,
                    "facts": facts,
                    "enrolled_people": enrolled,
                    "database_bytes": store.database_bytes(),
                    "schema_version": store.schema_version,
                },
                duration_s=time.perf_counter() - t0,
            )
        except MemoryError as exc:
            return self._failure("list_memories", exc, t0)

    def delete_memory(self, request) -> MemoryOperationResult:
        t0 = time.perf_counter()
        try:
            store = self._require_store()
            if request.fact_id:
                ok = store.delete_fact(request.fact_id)
                if not ok:
                    raise MemoryEntityNotFound(f"fact {request.fact_id} not found")
                return MemoryOperationResult(
                    ok=True,
                    operation="delete_memory",
                    message="fact deleted",
                    payload={"deleted": "fact", "id": request.fact_id},
                    duration_s=time.perf_counter() - t0,
                )
            entity_id = request.entity_id
            if not entity_id and request.name:
                matches = store.find_entities_by_name(request.name)
                if not matches:
                    raise MemoryEntityNotFound(f"no entity named {request.name!r}")
                entity_id = matches[0].entity_id
            if not entity_id:
                raise InvalidMemoryRequest("specify entity_id, fact_id or name")
            entity = store.get_entity(entity_id)
            if entity is None:
                raise MemoryEntityNotFound(f"entity {entity_id} not found")
            if self.face_service is not None and entity.entity_type == EntityType.PERSON:
                counts = self.face_service.remove_identity(entity_id)
                if self.retriever is not None:
                    self.retriever.face_index = self.face_service._index
            else:
                obj_emb_ids = [
                    emb_id
                    for emb_id, ent, _b, _d, _m, crop in store.all_object_embeddings()
                    if ent == entity_id
                ]
                for _emb_id, _ent, _blob, _dim, _model, crop in store.all_object_embeddings():
                    if _ent == entity_id and crop:
                        from src.memory.store import _safe_unlink

                        _safe_unlink(crop)
                counts = store.delete_entity(entity_id)
                if self.object_index is not None:
                    for emb_id in obj_emb_ids:
                        self.object_index.delete(emb_id)
            return MemoryOperationResult(
                ok=True,
                operation="delete_memory",
                message=f"deleted {entity.canonical_name}",
                payload={"entity_id": entity_id, "deleted": counts},
                duration_s=time.perf_counter() - t0,
            )
        except MemoryError as exc:
            return self._failure("delete_memory", exc, t0)

    # ------------------------------------------------------------------
    # background scheduling
    # ------------------------------------------------------------------
    def submit_background(self, operation: str, payload: dict, *, maintenance: bool = False) -> Optional[MemoryJob]:
        if self.scheduler is None or not self.settings.background_indexing:
            return None
        try:
            return self.scheduler.submit(
                operation,
                payload,
                priority=MemoryJobPriority.MAINTENANCE if maintenance else MemoryJobPriority.BACKGROUND,
            )
        except Exception as exc:  # noqa: BLE001 - queue full etc.
            log.debug("background memory job rejected: %s", exc)
            return None

    def maybe_learn_facts(self, transcript: str, turn_id: str = "") -> Optional[MemoryJob]:
        if not self.settings.auto_learn_facts or not transcript:
            return None
        return self.submit_background(
            "remember_fact",
            {"transcript": transcript, "turn_id": turn_id},
        )

    # ------------------------------------------------------------------
    # PECM: durable event recording
    # ------------------------------------------------------------------
    def record_event(
        self,
        event_type: EventType | str,
        content: str,
        *,
        session_id: str = "",
        turn_id: str = "",
        source_model: str = "",
        source_tool: str = "",
        frame_id: str = "",
        metadata: Optional[dict] = None,
    ) -> dict:
        """Persist one interaction event; returns an ack dict.

        The original event is committed *before* any derived indexing.
        Embedding / episode build jobs are submitted afterwards so fresh
        recall (recent-context + FTS) works even if they never complete.
        """
        if self.recorder is None or self.store is None:
            return {"ok": False, "error": "conversation memory disabled"}
        if isinstance(event_type, str):
            event_type = EventType(event_type)
        turn_id = turn_id or new_id("turn")

        if event_type == EventType.USER_TRANSCRIPT:
            ack = self.recorder.record_user_transcript(session_id or None, turn_id, content)
        elif event_type == EventType.USER_TEXT:
            ack = self.recorder.record_user_text(session_id or None, turn_id, content)
        elif event_type == EventType.VLM_OBSERVATION:
            ack = self.recorder.record_visual_observation(
                session_id or None, turn_id, frame_id, "", content,
                model_id=source_model or "moondream2", metadata=metadata,
            )
        elif event_type == EventType.TOOL_RESULT:
            ack = self.recorder.record_tool_result(
                session_id or None, turn_id, source_tool, content, metadata,
            )
        elif event_type == EventType.ASSISTANT_RESPONSE:
            ack = self.recorder.record_assistant_response(
                session_id or None, turn_id, content, model_id=source_model,
            )
        elif event_type == EventType.MEMORY_CORRECTION:
            ack = self.recorder.record_correction(session_id or None, turn_id, content)
        else:
            return {"ok": False, "error": f"unknown event type {event_type}"}

        if ack.ok and ack.event_id:
            # Derived work: embed + build episode off the hot path.
            from src.memory.event_models import IndexJob

            if self.settings.memory_semantic_search_enabled and self.text_embedder is not None:
                self.store.upsert_index_job(IndexJob(job_type="embed_event", source_id=ack.event_id))
            if self.settings.episodic_memory_enabled:
                self.store.upsert_index_job(IndexJob(job_type="build_episode", source_id=turn_id))
        return {
            "ok": ack.ok,
            "event_id": ack.event_id,
            "session_id": ack.session_id,
            "duplicate": ack.duplicate,
            "error": ack.error,
            "turn_id": turn_id,
        }

    def record_turn(
        self,
        *,
        transcript: str = "",
        observation: str = "",
        response: str = "",
        turn_id: str = "",
        frame_id: str = "",
        session_id: str = "",
        vlm_model: str = "",
    ) -> dict:
        """Convenience: record a full VQA/turn as a sequence of events."""
        turn_id = turn_id or new_id("turn")
        out = {"turn_id": turn_id, "events": []}
        if transcript:
            out["events"].append(self.record_event(
                EventType.USER_TRANSCRIPT, transcript,
                session_id=session_id, turn_id=turn_id,
            ))
        if observation:
            out["events"].append(self.record_event(
                EventType.VLM_OBSERVATION, observation,
                session_id=session_id, turn_id=turn_id,
                frame_id=frame_id, source_model=vlm_model or "moondream2",
                metadata={"user_question": transcript} if transcript else None,
            ))
        if response:
            out["events"].append(self.record_event(
                EventType.ASSISTANT_RESPONSE, response,
                session_id=session_id, turn_id=turn_id,
                source_model=vlm_model,
            ))
        return out

    # ------------------------------------------------------------------
    # PECM: retrieval / recall
    # ------------------------------------------------------------------
    def search_conversations(
        self,
        query: str,
        *,
        session_id: str = "",
        hint: Optional[TemporalHint] = None,
        max_events: Optional[int] = None,
    ) -> MemoryOperationResult:
        t0 = time.perf_counter()
        try:
            if self.conversation_retriever is None or self.store is None:
                raise MemoryError("conversation memory is not enabled")
            result = self.conversation_retriever.retrieve(
                query or "",
                session_id=session_id or None,
                hint=hint,
                max_events=max_events or self.settings.memory_max_retrieval_results,
            )
            return MemoryOperationResult(
                ok=True,
                operation="search_conversations",
                message=self._format_recall(result),
                payload={
                    "status": result.status.value,
                    "events": [
                        {
                            "event_id": r.event.event_id,
                            "event_type": r.event.event_type.value,
                            "content": r.event.content,
                            "frame_id": r.event.frame_id,
                            "timestamp": r.event.timestamp.isoformat(),
                            "score": round(r.score, 4),
                            "source": r.source,
                        }
                        for r in result.events
                    ],
                    "episodes": [
                        {
                            "episode_id": ep.episode.episode_id,
                            "summary": ep.episode.episode_summary,
                            "episode_type": ep.episode.episode_type.value,
                            "frame_id": ep.episode.frame_id,
                            "start": ep.episode.start_timestamp.isoformat(),
                        }
                        for ep in result.episodes
                    ],
                    "notes": result.notes,
                },
                duration_s=time.perf_counter() - t0,
            )
        except MemoryError as exc:
            return self._failure("search_conversations", exc, t0)

    def recall_last_visual_observation(
        self,
        query: str = "",
        *,
        session_id: str = "",
        frame_id: str = "",
    ) -> MemoryOperationResult:
        """Historical visual recall — works with the camera unavailable.

        Looks for the most recent VLM observation, optionally scoped to a
        frame or filtered by the user's question.  Statuses map to the
        spec: FOUND / NOT_FOUND / INDEX_PENDING / STORAGE_UNAVAILABLE.
        """
        t0 = time.perf_counter()
        op = "recall_visual"
        try:
            store = self._require_store()
            if not self.settings.memory_allow_historical_vision_recall:
                raise InvalidMemoryRequest("historical visual recall is disabled")
            if self.conversation_retriever is None:
                raise MemoryError("conversation memory is not enabled")

            # 1) frame_id hit is authoritative
            if frame_id:
                rows = store.recent_events(event_type=EventType.VLM_OBSERVATION, limit=50)
                hit = next((e for e in rows if e.frame_id == frame_id), None)
                if hit is not None:
                    return self._visual_recall_result(hit, "frame_id", t0)

            # 2) hybrid search biased to observations
            hint = parse_temporal(query) if query else TemporalHint(refers_to_history=True)
            hint.refers_to_last_observation = True
            result = self.conversation_retriever.retrieve(
                query or "what did you see",
                session_id=session_id or None,
                hint=hint,
                max_events=self.settings.memory_max_retrieval_results,
            )
            obs = [
                r for r in result.events
                if r.event.event_type == EventType.VLM_OBSERVATION
            ]
            if obs:
                return self._visual_recall_result(obs[0].event, "hybrid", t0)

            # 3) fall back to the most recent observation at all
            recent = store.recent_events(event_type=EventType.VLM_OBSERVATION, limit=1)
            if recent:
                return self._visual_recall_result(recent[0], "recent", t0)

            pending = store.count_pending_index_jobs()
            status = RecallStatus.INDEX_PENDING if pending else RecallStatus.NOT_FOUND
            return MemoryOperationResult(
                ok=True,
                operation=op,
                message=(
                    "I do not have a stored visual observation yet."
                    if status == RecallStatus.NOT_FOUND
                    else "I have not finished indexing recent observations yet."
                ),
                payload={"status": status.value, "events": [], "pending_jobs": pending},
                duration_s=time.perf_counter() - t0,
            )
        except MemoryError as exc:
            return self._failure(op, exc, t0)

    def _visual_recall_result(self, event: ConversationEvent, match_source: str, t0: float) -> MemoryOperationResult:
        question = (event.metadata or {}).get("user_question", "")
        return MemoryOperationResult(
            ok=True,
            operation="recall_visual",
            message=event.content,
            payload={
                "status": RecallStatus.FOUND.value,
                "events": [{
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "content": event.content,
                    "frame_id": event.frame_id,
                    "timestamp": event.timestamp.isoformat(),
                    "user_question": question,
                    "source_model": event.source_model,
                }],
                "match_source": match_source,
                "camera_required": False,
            },
            duration_s=time.perf_counter() - t0,
        )

    def build_rag_context(
        self,
        query: str,
        *,
        session_id: str = "",
        token_budget: Optional[int] = None,
    ) -> str:
        """Format retrieved episodes/events into the RAG context block."""
        if self.conversation_retriever is None:
            return ""
        result = self.conversation_retriever.retrieve(
            query or "", session_id=session_id or None,
            max_events=min(6, self.settings.memory_max_retrieval_results),
            max_episodes=3,
        )
        if not result.found:
            return ""
        budget = token_budget or self.settings.memory_max_context_tokens
        lines: list[str] = []
        used = 0
        for ep in result.episodes[:3]:
            line = f"[{ep.episode.episode_type.value}] {ep.episode.episode_summary}"
            if len(lines) + len(line) > budget:
                break
            lines.append(line)
            used += len(line)
        for r in result.events[:5]:
            label = r.event.event_type.value.replace("_", " ")
            line = f"({label}) {r.event.content}"
            if used + len(line) > budget:
                break
            lines.append(line)
            used += len(line)
        if not lines:
            return ""
        return (
            "Relevant past observations and conversation (local memory):\n"
            + "\n".join(f"- {l}" for l in lines)
        )

    def rebuild_text_index(self) -> dict:
        """Rebuild FTS + re-embed all stored events (maintenance)."""
        t0 = time.perf_counter()
        try:
            store = self._require_store()
            fts_count = store.rebuild_fts()
            emb_count = 0
            if self.text_embedder is not None and self.text_index is not None:
                # rebuild in-memory semantic index from stored embeddings
                self._load_text_index()
                emb_count = len(self.text_index)
            return {
                "ok": True,
                "fts_rows": fts_count,
                "semantic_embeddings": emb_count,
                "duration_s": round(time.perf_counter() - t0, 4),
            }
        except MemoryError as exc:
            return {"ok": False, "error": str(exc)}

    def delete_conversation(
        self,
        *,
        session_id: str = "",
        event_id: str = "",
        all_history: bool = False,
    ) -> MemoryOperationResult:
        t0 = time.perf_counter()
        try:
            store = self._require_store()
            if event_id:
                ok = store.delete_event(event_id)
                if not ok:
                    raise MemoryEntityNotFound(f"event {event_id} not found")
                if self.text_index is not None:
                    try:
                        self.text_index._pos.pop(f"te_{event_id}", None)
                    except Exception:  # noqa: BLE001
                        pass
                return MemoryOperationResult(
                    ok=True, operation="delete_conversation",
                    message="event deleted",
                    payload={"deleted": "event", "id": event_id},
                    duration_s=time.perf_counter() - t0,
                )
            if session_id:
                n = store.delete_session_events(session_id)
                m = store.delete_session_episodes(session_id)
                return MemoryOperationResult(
                    ok=True, operation="delete_conversation",
                    message=f"deleted {n} events and {m} episodes from the session",
                    payload={"events": n, "episodes": m, "session_id": session_id},
                    duration_s=time.perf_counter() - t0,
                )
            if all_history:
                counts = store.clear_all_conversation_memory()
                if self.text_index is not None:
                    self.text_index = TextEmbeddingIndex(
                        self.text_embedder.name if self.text_embedder else "hash-embed-v1",
                        self.text_embedder.dimension if self.text_embedder else 256,
                    )
                    if self.conversation_retriever is not None:
                        self.conversation_retriever.text_index = self.text_index
                return MemoryOperationResult(
                    ok=True, operation="delete_conversation",
                    message="all conversation history deleted",
                    payload={"deleted": counts},
                    duration_s=time.perf_counter() - t0,
                )
            raise InvalidMemoryRequest("specify session_id, event_id or all_history=True")
        except MemoryError as exc:
            return self._failure("delete_conversation", exc, t0)

    # ------------------------------------------------------------------
    # PECM internals
    # ------------------------------------------------------------------
    def _load_text_index(self) -> None:
        assert self.store is not None
        if self.text_embedder is None:
            return
        self.text_index = TextEmbeddingIndex(
            self.text_embedder.name, self.text_embedder.dimension
        )
        for row in self.store.all_text_embeddings(model_id=self.text_embedder.name):
            try:
                vec = deserialize_vector(row["embedding"], row["dimension"])
            except InvalidEmbedding as exc:
                log.warning("skipping corrupt text embedding %s: %s", row.get("embedding_id"), exc)
                continue
            self.text_index.add(
                row["embedding_id"], vec, {"source_id": row["source_id"]}
            )

    def _embed_event_payload(self, payload: dict) -> dict:
        store = self._require_store()
        event_id = str(payload.get("event_id") or "")
        if not event_id:
            raise InvalidMemoryRequest("embed_event requires event_id")
        embedder = self.text_embedder
        if embedder is None:
            return {"embedded": False, "reason": "no text embedder"}
        event = store.get_event(event_id)
        if event is None:
            return {"embedded": False, "reason": "event not found"}
        from src.memory.vector_index import serialize_vector

        vec = embedder.encode(event.content)
        blob, dim, _ = serialize_vector(vec)
        store.insert_text_embedding(
            f"te_{event_id}", event_id, "event", embedder.name, dim, blob
        )
        if self.text_index is not None:
            self.text_index.add(f"te_{event_id}", vec, {"source_id": event_id})
        return {"embedded": True, "event_id": event_id, "dimension": dim}

    def _build_episode_payload(self, payload: dict) -> dict:
        turn_id = str(payload.get("turn_id") or "")
        if self.episode_builder is None:
            return {"built": False, "reason": "episodic memory disabled"}
        episode = self.episode_builder.build_for_turn(turn_id)
        return {
            "built": episode is not None,
            "episode_id": episode.episode_id if episode else None,
            "turn_id": turn_id,
        }

    @staticmethod
    def _format_recall(result: RecallResult) -> str:
        if not result.found:
            return "I could not find anything matching that in our conversation."
        parts = []
        for r in result.events[:3]:
            parts.append(r.event.content)
        for ep in result.episodes[:2]:
            parts.append(ep.episode.episode_summary)
        return " ".join(parts)[:500]

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        if not self.settings.enabled:
            return {"enabled": False, "status": "disabled"}
        if self.store is None:
            return {"enabled": True, "status": "not_initialized", "last_error": self._last_error}
        try:
            counts = self.store.counts()
            try:
                counts.update(self.store.counts_ex())
            except Exception:  # noqa: BLE001
                pass
        except Exception as exc:  # noqa: BLE001
            return {"enabled": True, "status": "error", "last_error": str(exc)}
        return {
            "enabled": True,
            "status": "ready",
            "db_path": str(self.store.db_path),
            "schema_version": self.store.schema_version,
            "counts": counts,
            "database_bytes": self.store.database_bytes(),
            "face_enabled": self.face_service is not None,
            "face_backend": self.face_service.backend.name if self.face_service else None,
            "object_enabled": self.embedder is not None,
            "object_embedder": self.embedder.model_id if self.embedder else None,
            "object_index_size": len(self.object_index) if self.object_index else 0,
            "conversation_enabled": self.recorder is not None,
            "episodic_enabled": self.episode_builder is not None,
            "text_embedder": self.text_embedder.name if self.text_embedder else None,
            "text_index_size": len(self.text_index) if self.text_index else 0,
            "index_worker_running": bool(self.index_worker and self.index_worker.running),
            "background_indexing": self.scheduler.status() if self.scheduler else None,
            "last_error": self._last_error,
        }

    # ------------------------------------------------------------------
    def _failure(self, operation: str, exc: MemoryError, t0: float) -> MemoryOperationResult:
        self._last_error = str(exc)
        d = exc.to_dict() if hasattr(exc, "to_dict") else {}
        log.warning("memory op %s failed: %s", operation, exc)
        return MemoryOperationResult.failure(
            operation,
            d.get("error_code", exc.code if hasattr(exc, "code") else "MemoryError"),
            str(exc),
            retryable=bool(d.get("retryable", getattr(exc, "retryable", False))),
            payload=d.get("detail", {}),
        )


def _frame_to_bgr(frame) -> np.ndarray:
    """Accept a Frame, ndarray, or PIL image and return BGR uint8."""
    import cv2

    image = getattr(frame, "image", frame)
    if image is None:
        raise InvalidMemoryRequest("frame has no image data")
    arr = np.asarray(image)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    # Frame images from OpenCV are already BGR
    return np.ascontiguousarray(arr)
