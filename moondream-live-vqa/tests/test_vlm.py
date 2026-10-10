"""Moondream engine tests.

Two layers:

* pure unit tests that never touch weights (API detection, prompt building,
  model-path discovery, payload schema);
* an integration test marked ``requires_model`` that loads the real local
  checkpoint and answers a question about a static image.
"""

from __future__ import annotations

import threading

import pytest

from config.settings import (
    InferenceConfig,
    ModelConfig,
    ModelNotFoundError,
    resolve_model_path,
)
from src.vlm.moondream_engine import (
    ModelApiError,
    MoondreamEngine,
    failed_result,
    successful_result,
)
from tests.conftest import requires_model


@pytest.fixture
def isolated_candidates(monkeypatch):
    """Disable the built-in search paths and any MODEL_PATH override."""
    monkeypatch.setattr("config.settings.MODEL_SEARCH_CANDIDATES", ())
    monkeypatch.delenv("MODEL_PATH", raising=False)


class TestModelPathDiscovery:
    def test_finds_the_installed_checkpoint(self):
        path = resolve_model_path()
        assert (path / "config.json").is_file()

    def test_env_override(self, monkeypatch, tmp_path):
        model_dir = tmp_path / "weights"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        (model_dir / "model.safetensors").write_bytes(b"0")
        monkeypatch.setenv("MODEL_PATH", str(model_dir))
        assert resolve_model_path() == model_dir.resolve()

    def test_missing_dir_raises_with_guidance(self, tmp_path, isolated_candidates):
        with pytest.raises(ModelNotFoundError) as excinfo:
            resolve_model_path(str(tmp_path / "nowhere"))
        assert "MODEL_PATH" in str(excinfo.value)

    def test_directory_without_weights_rejected(self, tmp_path, isolated_candidates):
        empty = tmp_path / "incomplete"
        empty.mkdir()
        (empty / "config.json").write_text("{}")
        with pytest.raises(ModelNotFoundError):
            resolve_model_path(str(empty))


class TestApiDetection:
    def test_query_and_encode_image_detected(self):
        class Model:
            def encode_image(self, image): ...

            def query(self, image, question): ...

        assert MoondreamEngine._detect_api(Model()) == "hf_query"

    def test_legacy_answer_question_detected(self):
        class Model:
            def encode_image(self, image): ...

            def answer_question(self, image, question, tokenizer): ...

        assert MoondreamEngine._detect_api(Model()) == "legacy"

    def test_unsupported_interface_raises(self):
        class Model:
            def generate(self, *args, **kwargs): ...

        with pytest.raises(ModelApiError) as excinfo:
            MoondreamEngine._detect_api(Model())
        assert "Unsupported Moondream interface" in str(excinfo.value)

    def test_partial_interface_raises(self):
        class Model:
            def query(self, image, question): ...

        with pytest.raises(ModelApiError):
            MoondreamEngine._detect_api(Model())


class TestDeviceAndDtype:
    def test_auto_device_is_cuda_when_available(self):
        import torch

        device = MoondreamEngine._resolve_device("auto")
        assert device in {"cuda", "cpu"}
        if torch.cuda.is_available():
            assert device == "cuda"

    def test_cuda_request_falls_back_when_unavailable(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert MoondreamEngine._resolve_device("cuda") == "cpu"

    def test_dtype_resolution(self):
        import torch

        assert MoondreamEngine._resolve_dtype("fp32", "cuda") is torch.float32
        assert MoondreamEngine._resolve_dtype("bf16", "cpu") is torch.bfloat16
        assert MoondreamEngine._resolve_dtype("fp16", "cpu") is torch.float16
        assert MoondreamEngine._resolve_dtype("auto", "cpu") is torch.float32


class TestPromptAndPayload:
    def test_grounding_prompt_appended(self):
        engine = MoondreamEngine(
            ModelConfig(),
            InferenceConfig(grounding_prompt=True, grounding_text="Only visible evidence."),
        )
        prompt = engine._build_prompt("What is this?", grounded=True)
        assert prompt.startswith("What is this?")
        assert "Only visible evidence." in prompt

    def test_grounding_disabled(self):
        engine = MoondreamEngine(
            ModelConfig(), InferenceConfig(grounding_prompt=False)
        )
        assert engine._build_prompt("Q?", grounded=False) == "Q?"

    def test_success_payload_schema(self):
        payload = successful_result(answer="hi", device="cuda", frame_id=7)
        for key in (
            "answer",
            "status",
            "error",
            "inference_time_seconds",
            "encode_time_seconds",
            "generate_time_seconds",
            "device",
            "frame_id",
            "question",
            "timestamp",
        ):
            assert key in payload
        assert payload["status"] == "success"
        assert payload["frame_id"] == 7

    def test_failed_payload_schema(self):
        payload = failed_result("boom", frame_id=3)
        assert payload["status"] == "error"
        assert payload["error"] == "boom"
        assert payload["answer"] == ""
        assert payload["frame_id"] == 3


class TestNotLoadedBehaviour:
    @staticmethod
    def _frame():
        import numpy as np

        return np.zeros((8, 8, 3), dtype=np.uint8)

    def test_ask_before_load_returns_error_payload(self, isolated_candidates):
        engine = MoondreamEngine(ModelConfig(model_path="/nonexistent"))
        result = engine.ask("What is this?", frame=self._frame(), frame_id=1)
        assert result["status"] == "error"
        assert "not loaded" in result["error"]

    def test_empty_question_rejected(self):
        engine = MoondreamEngine(ModelConfig())
        engine._model = object()  # simulate loaded state without weights
        result = engine.ask("   ")
        assert result["status"] == "error"
        assert "empty" in result["error"]

    def test_missing_image_rejected(self):
        engine = MoondreamEngine(ModelConfig())
        engine._model = object()
        result = engine.ask("What?")
        assert result["status"] == "error"
        assert "no image" in result["error"]

    def test_overlong_question_rejected(self):
        engine = MoondreamEngine(ModelConfig())
        engine._model = object()
        result = engine.ask("x" * 4001, frame=self._frame())
        assert result["status"] == "error"
        assert "too long" in result["error"]


@requires_model
class TestRealModelIntegration:
    @pytest.fixture(scope="class")
    def engine(self):
        config = build_engine_config()
        engine = MoondreamEngine(config[0], config[1])
        engine.load()
        yield engine
        engine.unload()

    def test_load_reports_expected_info(self, engine):
        info = engine.info()
        assert info is not None
        assert info.api == "hf_query"
        assert info.parameter_count > 1_000_000_000
        assert info.model_path.endswith("moondream2")
        assert engine.device in {"cuda", "cpu"}

    def test_answers_a_question_about_a_static_image(self, engine, sample_image):
        result = engine.ask(
            "What objects are visible in this image?",
            image=sample_image,
            frame_id=1,
        )
        assert result["status"] == "success", result.get("error")
        assert result["answer"].strip()
        assert result["inference_time_seconds"] > 0
        assert result["device"] == engine.device
        assert result["frame_id"] == 1

    def test_encoding_cache_reused_for_same_key(self, engine, sample_image):
        first = engine.ask("Describe this image.", image=sample_image, cache_key="k1")
        second = engine.ask("How many objects?", image=sample_image, cache_key="k1")
        assert first["status"] == "success"
        assert second["status"] == "success"
        assert second["encoding_reused"] is True
        assert second["encode_time_seconds"] == 0.0
        assert second["inference_time_seconds"] <= first["inference_time_seconds"]

    def test_cache_invalidated_on_new_key(self, engine, sample_image):
        engine.ask("q1", image=sample_image, cache_key="frame:1")
        result = engine.ask("q2", image=sample_image, cache_key="frame:2")
        assert result["encoding_reused"] is False
        assert result["encode_time_seconds"] > 0

    def test_concurrent_questions_are_serialised(self, engine, sample_image):
        results = []

        def worker(question: str) -> None:
            results.append(engine.ask(question, image=sample_image, cache_key=None))

        threads = [
            threading.Thread(target=worker, args=(f"Question {i}?",))
            for i in range(3)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        assert len(results) == 3
        assert all(r["status"] == "success" for r in results)


def build_engine_config():
    model = ModelConfig()
    inference = InferenceConfig(grounding_prompt=True)
    return model, inference
