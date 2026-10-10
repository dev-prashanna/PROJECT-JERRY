from __future__ import annotations

import pytest

from src.config import ConfigError, load_config, validate_config, AppConfig, ModelConfig, MeasurementConfig


def test_default_config_loads(cfg):
    assert cfg.source.index >= 0
    assert cfg.model.input_size % 14 == 0
    assert cfg.model.local_files_only is True


def test_invalid_input_size_rejected(cfg):
    cfg.model.input_size = 500
    with pytest.raises(ConfigError, match="multiple of 14"):
        validate_config(cfg)


def test_invalid_precision_rejected(cfg):
    cfg.model.precision = "int8"
    with pytest.raises(ConfigError, match="precision"):
        validate_config(cfg)


def test_invalid_region_size_rejected(cfg):
    cfg.measurement.region_size = 4
    with pytest.raises(ConfigError, match="odd"):
        validate_config(cfg)


def test_invalid_alpha_rejected(cfg):
    cfg.depth.smoothing.alpha = 0.0
    with pytest.raises(ConfigError, match="alpha"):
        validate_config(cfg)


def test_camera_source_requires_index(cfg):
    cfg.source.index = -1
    with pytest.raises(ConfigError, match="index"):
        validate_config(cfg)


def test_unknown_key_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("model:\n  not_a_key: 1\n")
    with pytest.raises(ConfigError, match="Unknown key"):
        load_config(bad)


def test_missing_file_reports_path(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_defaults_are_valid():
    validate_config(AppConfig())
