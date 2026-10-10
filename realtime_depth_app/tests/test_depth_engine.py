from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config import ConfigError, validate_config
from src.depth_engine import DepthEngine, resolve_device


def test_device_resolution():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda"}


def test_model_reports_relative_type(engine):
    assert engine.estimation_type == "relative"
    assert engine.meta.convention == "inverse"
    assert engine.meta.unit == "relative"


def test_prediction_shape_matches_frame(engine, sample_image):
    result = engine.infer(sample_image, frame_id=7, t_capture=123.0)
    assert result.depth.shape == sample_image.shape[:2]
    assert result.depth.dtype == np.float32
    assert result.frame_id == 7
    assert result.t_capture == 123.0


def test_prediction_is_finite_and_non_trivial(engine, sample_image):
    depth = engine.infer(sample_image).depth
    assert np.isfinite(depth).all()
    assert depth.max() > depth.min() + 1e-3
    # the head is ReLU so raw values are non-negative; bicubic upsampling can undershoot
    # by a tiny amount at strong edges
    assert depth.min() > -0.1
    assert (depth >= 0).mean() > 0.99


def test_inverse_depth_convention_on_sample(engine, sample_image):
    """The sample frame is near at the bottom (robot/floor) and far at the top (back wall).

    An inverse-depth model must therefore produce larger values at the bottom.
    """
    depth = engine.infer(sample_image).depth
    h = depth.shape[0]
    near = float(np.median(depth[int(h * 0.75):, :]))
    far = float(np.median(depth[: int(h * 0.25), :]))
    assert near > far, "expected inverse depth (higher = closer) on the sample scene"


def test_inference_timings_recorded(engine, sample_image):
    result = engine.infer(sample_image)
    assert result.inference_ms >= 0
    assert result.preprocess_ms >= 0
    assert result.postprocess_ms >= 0
    assert result.t_end >= result.t_start


def test_input_size_multiple_of_14_required(cfg):
    cfg.model.input_size = 500
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_half_precision_requires_cuda(cfg):
    if torch.cuda.is_available():
        pytest.skip("cuda available - the guard only triggers on cpu-only hosts")
    cfg.model.device = "cpu"
    cfg.model.precision = "fp16"
    with pytest.raises(ConfigError, match="CUDA"):
        DepthEngine(cfg)


def test_device_override(cfg):
    engine = DepthEngine(cfg)
    assert str(engine.device) in {"cuda", "cpu"}


def test_gpu_memory_reporting(engine):
    mb = engine.gpu_memory_mb()
    if str(engine.device).startswith("cuda"):
        assert mb is not None and mb > 0
    else:
        assert mb is None
