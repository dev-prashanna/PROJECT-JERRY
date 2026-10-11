"""Face detection, 128-D embeddings and consent-aware biometric memory.

Backends (configurable, auto-detected):

* ``opencv``  – YuNet detector + SFace recognizer (128-D), CPU-friendly.
* ``face_recognition`` – dlib API when that package is installed.

Biometric rules:

* Enrollment is explicit; ordinary observation never enrolls.
* Multiple simultaneous faces block automatic name binding.
* Stored embeddings are sensitive biometric data: local-only, deletable,
  never sent to the cloud LLM.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np

from src.memory.exceptions import (
    FaceNotDetected,
    FaceQualityInsufficient,
    FaceRecognitionUnavailable,
    InvalidEmbedding,
    MultipleFacesAmbiguous,
)
from src.memory.models import FaceObservation, MatchResult, MatchStatus, EntityType, utcnow
from src.memory.permissions import authorize_face_enrollment
from src.memory.store import MemoryStore, ensure_private_dir
from src.memory.vector_index import (
    ExactNumpyIndex,
    deserialize_vector,
    l2_normalize,
    serialize_vector,
)
from src.utils.logger import get_logger

log = get_logger("memory.face")

FACE_DIMENSION = 128
MIN_FACE_PX = 40

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)


@dataclass
class DetectedFace:
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    embedding: Optional[np.ndarray] = None
    observation_id: str = ""

    @property
    def width(self) -> int:
        return self.bbox_xyxy[2] - self.bbox_xyxy[0]

    @property
    def height(self) -> int:
        return self.bbox_xyxy[3] - self.bbox_xyxy[1]


@runtime_checkable
class FaceBackend(Protocol):
    name: str
    dimension: int

    def available(self) -> bool: ...

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]: ...

    def embed(self, image_bgr: np.ndarray, face: DetectedFace) -> np.ndarray: ...


def _download(url: str, dest: Path, timeout_s: float = 60.0) -> Path:
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    ensure_private_dir(dest.parent)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("downloading face model %s", url)
    with urllib.request.urlopen(url, timeout=timeout_s) as resp, open(tmp, "wb") as fh:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            fh.write(chunk)
    tmp.replace(dest)
    return dest


class OpenCVFaceBackend:
    """YuNet detection + SFace recognition (both 128-D, CPU)."""

    name = "opencv-sface"

    def __init__(self, model_dir: Path, *, allow_download: bool = True):
        self.dimension = FACE_DIMENSION
        self.model_dir = Path(model_dir)
        self.allow_download = allow_download
        self._detector = None
        self._recognizer = None
        self._failed: Optional[str] = None

    def _paths(self) -> tuple[Path, Path]:
        return (
            self.model_dir / "face_detection_yunet_2023mar.onnx",
            self.model_dir / "face_recognition_sface_2021dec.onnx",
        )

    def _ensure_models(self) -> tuple[Path, Path]:
        det, rec = self._paths()
        missing = [p for p in (det, rec) if not p.is_file() or p.stat().st_size == 0]
        if missing and not self.allow_download:
            raise FaceRecognitionUnavailable(
                "face models missing and download disabled: "
                + ", ".join(str(p) for p in missing)
            )
        if det.is_file() and det.stat().st_size == 0:
            det.unlink()
        if rec.is_file() and rec.stat().st_size == 0:
            rec.unlink()
        try:
            if not det.is_file():
                _download(YUNET_URL, det)
            if not rec.is_file():
                _download(SFACE_URL, rec)
        except Exception as exc:  # noqa: BLE001
            raise FaceRecognitionUnavailable(f"cannot download face models: {exc}") from exc
        return det, rec

    def _load(self) -> None:
        if self._detector is not None:
            return
        if self._failed:
            raise FaceRecognitionUnavailable(self._failed)
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            self._failed = "opencv not installed"
            raise FaceRecognitionUnavailable(self._failed) from exc
        try:
            det_path, rec_path = self._ensure_models()
            self._detector = cv2.FaceDetectorYN.create(
                str(det_path), "", (320, 320), score_threshold=0.6, nms_threshold=0.3, top_k=5000
            )
            self._recognizer = cv2.FaceRecognizerSF.create(str(rec_path), "")
        except FaceRecognitionUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            self._failed = f"failed to initialise face backend: {exc}"
            raise FaceRecognitionUnavailable(self._failed) from exc

    def available(self) -> bool:
        try:
            self._load()
            return True
        except FaceRecognitionUnavailable:
            return False

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        import cv2

        self._load()
        h, w = image_bgr.shape[:2]
        self._detector.setInputSize((w, h))
        ok, faces = self._detector.detect(image_bgr)
        out: list[DetectedFace] = []
        if not ok or faces is None:
            return out
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = (int(max(0, v)) for v in face[:4])
            conf = float(face[-1])
            x2, y2 = min(x2, w), min(y2, h)
            if x2 - x1 < MIN_FACE_PX or y2 - y1 < MIN_FACE_PX:
                continue
            out.append(DetectedFace(bbox_xyxy=(x1, y1, x2, y2), confidence=conf))
        return out

    def embed(self, image_bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
        import cv2

        self._load()
        x1, y1, x2, y2 = face.bbox_xyxy
        aligned = None
        try:
            row = np.array([[*face.bbox_xyxy, face.confidence]], dtype=np.float32)
            aligned = self._recognizer.alignCrop(image_bgr, row)
        except Exception:  # noqa: BLE001 - fall back to plain crop
            aligned = None
        if aligned is None:
            pad = 2
            aligned = image_bgr[
                max(0, y1 - pad) : min(image_bgr.shape[0], y2 + pad),
                max(0, x1 - pad) : min(image_bgr.shape[1], x2 + pad),
            ]
        feat = self._recognizer.feature(aligned)
        vec = np.asarray(feat, dtype=np.float32).reshape(-1)
        if vec.size != self.dimension:
            raise InvalidEmbedding(f"SFace returned {vec.size}-D, expected {self.dimension}")
        return l2_normalize(vec)


class FaceRecognitionApiBackend:
    """Adapter for the optional ``face_recognition`` (dlib) package."""

    name = "face_recognition-api"

    def __init__(self):
        self.dimension = FACE_DIMENSION
        self._mod = None

    def _load(self):
        if self._mod is None:
            try:
                import face_recognition  # type: ignore
            except ImportError as exc:
                raise FaceRecognitionUnavailable("face_recognition package not installed") from exc
            self._mod = face_recognition
        return self._mod

    def available(self) -> bool:
        try:
            self._load()
            return True
        except FaceRecognitionUnavailable:
            return False

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        mod = self._load()
        import cv2

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        locations = mod.face_locations(rgb, model="hog")
        return [
            DetectedFace(bbox_xyxy=(int(top), int(right), int(bottom), int(left)), confidence=1.0)
            for (top, right, bottom, left) in locations
        ]

    def embed(self, image_bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
        mod = self._load()
        import cv2

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # face_recognition expects (top, right, bottom, left)
        top, right, bottom, left = (
            face.bbox_xyxy[1],
            face.bbox_xyxy[2],
            face.bbox_xyxy[3],
            face.bbox_xyxy[0],
        )
        encodings = mod.face_encodings(rgb, known_face_locations=[(top, right, bottom, left)])
        if not encodings:
            raise FaceQualityInsufficient("dlib could not encode the selected face")
        vec = np.asarray(encodings[0], dtype=np.float32).reshape(-1)
        if vec.size != self.dimension:
            raise InvalidEmbedding(f"face_recognition returned {vec.size}-D")
        return vec  # dlib distances use the raw scale; do not normalize


def create_face_backend(name: str = "auto", model_dir: str | Path = "~/.aura/models/face", allow_download: bool = True) -> FaceBackend:
    model_path = Path(model_dir).expanduser()
    candidates: list[FaceBackend] = []
    if name in {"auto", "face_recognition", "face_recognition-api"}:
        candidates.append(FaceRecognitionApiBackend())
    if name in {"auto", "opencv", "opencv-sface"}:
        candidates.append(OpenCVFaceBackend(model_path, allow_download=allow_download))
    for backend in candidates:
        if backend.available():
            log.info("face backend: %s", backend.name)
            return backend
    raise FaceRecognitionUnavailable("no face-recognition backend is available")


@dataclass
class FaceMatchThresholds:
    match_distance: float = 0.55
    ambiguity_margin: float = 0.05
    min_face_px: int = MIN_FACE_PX


class FaceMemoryService:
    """Detection, embedding, matching and consent-aware enrollment."""

    def __init__(
        self,
        store: MemoryStore,
        backend: Optional[FaceBackend] = None,
        thresholds: Optional[FaceMatchThresholds] = None,
        *,
        require_confirmation: bool = True,
    ):
        self.store = store
        self._backend = backend
        self.thresholds = thresholds or FaceMatchThresholds()
        self.require_confirmation = require_confirmation
        self._index: Optional[ExactNumpyIndex] = None
        self._index_model: str = ""

    @property
    def backend(self) -> FaceBackend:
        if self._backend is None:
            raise FaceRecognitionUnavailable("face backend not configured")
        return self._backend

    def available(self) -> bool:
        if self._backend is None:
            return False
        return self._backend.available()

    def _ensure_index(self) -> Optional[ExactNumpyIndex]:
        try:
            model_id = self.backend.name
        except FaceRecognitionUnavailable:
            return None
        if self._index is not None and self._index_model == model_id:
            return self._index
        index = ExactNumpyIndex(model_id, FACE_DIMENSION, metric="l2")
        for emb_id, entity_id, blob, dim, row_model, _q in self.store.all_face_embeddings():
            if row_model != model_id or dim != FACE_DIMENSION:
                continue
            try:
                vec = deserialize_vector(blob, dim)
            except InvalidEmbedding as exc:
                log.warning("skipping corrupt face embedding %s: %s", emb_id, exc)
                continue
            index.add(emb_id, vec, {"entity_id": entity_id})
        self._index = index
        self._index_model = model_id
        return index

    def rebuild_index(self) -> int:
        self._index = None
        index = self._ensure_index()
        return len(index) if index else 0

    # -- detection / embedding -----------------------------------------
    def detect_faces(self, image_bgr: np.ndarray) -> list[FaceObservation]:
        faces = self.backend.detect(image_bgr)
        out = []
        for face in faces:
            obs_id = f"fobs_{abs(hash((face.bbox_xyxy, id(image_bgr)))) % (10**10)}"
            out.append(
                FaceObservation(
                    observation_id=obs_id,
                    frame_id="",
                    bbox_xyxy=face.bbox_xyxy,
                    detection_quality=face.confidence,
                    source_model=self.backend.name,
                    timestamp=utcnow(),
                )
            )
        return out

    def extract_embeddings(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        faces = self.backend.detect(image_bgr)
        for face in faces:
            face.embedding = self.backend.embed(image_bgr, face)
        return faces

    # -- matching -------------------------------------------------------
    def match_face(self, embedding: np.ndarray) -> MatchResult:
        index = self._ensure_index()
        model_id = self._index_model or getattr(self._backend, "name", "")
        if index is None or len(index) == 0:
            return MatchResult(status=MatchStatus.UNAVAILABLE, embedding_model_id=model_id, notes=["no enrolled faces"])
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vec.size != FACE_DIMENSION:
            raise InvalidEmbedding(f"expected {FACE_DIMENSION}-D face embedding, got {vec.size}")
        hits = index.search(vec, k=2)
        if not hits:
            return MatchResult(status=MatchStatus.UNKNOWN, embedding_model_id=model_id)
        best_id, best_d, best_meta = hits[0]
        second_d = hits[1][1] if len(hits) > 1 else None
        entity_id = best_meta.get("entity_id")
        entity = self.store.get_entity(entity_id) if entity_id else None
        cosine = 1.0 - (best_d ** 2) / 2.0  # for unit-normalized vectors
        notes: list[str] = []
        if best_d > self.thresholds.match_distance:
            return MatchResult(
                entity_id=None,
                status=MatchStatus.UNKNOWN,
                raw_distance=best_d,
                similarity_score=cosine,
                second_best_score=second_d,
                embedding_model_id=model_id,
                evidence_ids=[best_id],
                notes=["best distance above match threshold"],
            )
        if (
            second_d is not None
            and (second_d - best_d) < self.thresholds.ambiguity_margin
        ):
            return MatchResult(
                entity_id=None,
                status=MatchStatus.AMBIGUOUS,
                raw_distance=best_d,
                similarity_score=cosine,
                second_best_score=second_d,
                embedding_model_id=model_id,
                evidence_ids=[best_id],
                notes=["competing candidates are too close to call"],
            )
        return MatchResult(
            entity_id=entity_id,
            status=MatchStatus.MATCHED if entity else MatchStatus.UNKNOWN,
            raw_distance=best_d,
            similarity_score=cosine,
            second_best_score=second_d,
            embedding_model_id=model_id,
            evidence_ids=[best_id],
            entity_name=entity.canonical_name if entity else None,
            notes=notes,
        )

    # -- enrollment -----------------------------------------------------
    def enroll_identity(
        self,
        entity_name: str,
        image_bgr: np.ndarray,
        *,
        transcript: str = "",
        selected_face_id: Optional[str] = None,
        consent_token: Optional[str] = None,
        frame_id: str = "",
        request_id: str = "",
    ) -> tuple[Any, MatchResult, int]:
        """Enroll a consenting person; returns (entity, match, face_count)."""
        faces = self.extract_embeddings(image_bgr)
        if not faces:
            raise FaceNotDetected("no face was detected in the snapshot")
        consent, _mode = authorize_face_enrollment(
            self.store,
            transcript=transcript,
            consent_token=consent_token,
            require_confirmation=self.require_confirmation,
            face_count=len(faces),
        )
        chosen: Optional[DetectedFace] = None
        if len(faces) == 1:
            chosen = faces[0]
        else:
            if not selected_face_id:
                raise MultipleFacesAmbiguous(
                    f"{len(faces)} faces are visible; select the face to enroll"
                )
            for face in faces:
                if face.observation_id == selected_face_id or str(id(face)) == selected_face_id:
                    chosen = face
                    break
            if chosen is None:
                raise MultipleFacesAmbiguous("selected face id does not match any detection")
        if chosen.embedding is None or float(np.linalg.norm(chosen.embedding)) < 1e-6:
            raise FaceQualityInsufficient("face embedding was empty")

        match = self.match_face(chosen.embedding)
        from src.memory.entity_resolver import EntityResolver

        resolver = EntityResolver(self.store)
        if match.status == MatchStatus.MATCHED and match.entity_id:
            entity = self.store.get_entity(match.entity_id)
            if entity is not None and normalize_equal(entity.canonical_name, entity_name):
                blob, dim, _ = serialize_vector(chosen.embedding)
                emb_id = self.store.insert_face_embedding(
                    entity.entity_id,
                    blob,
                    dim,
                    self.backend.name,
                    quality_score=chosen.confidence,
                )
                self._ensure_index().add(emb_id, chosen.embedding, {"entity_id": entity.entity_id})
                self.store.touch_entity(entity.entity_id)
                return entity, match, len(faces)

        entity = resolver.get_or_create(entity_name, EntityType.PERSON, aliases=[entity_name])
        blob, dim, _ = serialize_vector(chosen.embedding)
        emb_id = self.store.insert_face_embedding(
            entity.entity_id,
            blob,
            dim,
            self.backend.name,
            quality_score=chosen.confidence,
        )
        assert self._ensure_index() is not None
        self._index.add(emb_id, chosen.embedding, {"entity_id": entity.entity_id})
        self.store.touch_entity(entity.entity_id)
        self.store.insert_observation(
            chosen.observation_id or f"fobs_{emb_id}",
            frame_id or "unknown",
            "face",
            chosen.bbox_xyxy,
            quality_score=chosen.confidence,
            source_model=self.backend.name,
            request_id=request_id,
        )
        self.store.link_observation(entity.entity_id, chosen.observation_id or f"fobs_{emb_id}")
        if consent is not None:
            self.store.record_identity_binding(entity.entity_id, consent.consent_id, consent.method)
        else:
            self.store.record_identity_binding(entity.entity_id, None, "self_enrollment_utterance")
        self.store.record_revision(
            entity_id=entity.entity_id,
            fact_id=None,
            action="enroll_face",
            detail=f"enrolled from frame {frame_id}",
        )
        return entity, match, len(faces)

    def remove_identity(self, entity_id: str) -> dict[str, int]:
        self.store.revoke_identity_bindings(entity_id)
        removed = self.store.delete_face_embeddings_for(entity_id)
        if self._index is not None:
            for emb_id, ent, *_rest in self.store.all_face_embeddings():
                if ent == entity_id:
                    self._index.delete(emb_id)
        counts = self.store.delete_entity(entity_id)
        counts["face_embeddings_removed"] = removed
        return counts

    def list_enrolled(self) -> list[dict[str, Any]]:
        rows = self.store.execute_fetchall(
            """
            SELECT e.entity_id, e.canonical_name, COUNT(f.embedding_id) AS n
            FROM entities e
            JOIN face_embeddings f ON f.entity_id = e.entity_id
            WHERE e.entity_type = 'person'
            GROUP BY e.entity_id
            ORDER BY e.canonical_name
            """
        )
        return [
            {"entity_id": r[0], "name": r[1], "embeddings": int(r[2])}
            for r in rows
        ]


def normalize_equal(a: str, b: str) -> bool:
    return " ".join(a.lower().split()) == " ".join(b.lower().split())
