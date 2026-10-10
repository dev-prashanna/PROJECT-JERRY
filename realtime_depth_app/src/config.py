"""Configuration loading and validation for the realtime depth application."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = APP_ROOT / "config.yaml"

VALID_PRECISIONS = {"fp32", "fp16", "bf16", "autocast_fp16", "autocast_bf16"}
VALID_DEVICES = {"auto", "cuda", "cpu"}
VALID_RANGE_MODES = {"auto", "fixed"}
VALID_COLORMAPS = {
    "TURBO",
    "JET",
    "INFERNO",
    "MAGMA",
    "VIRIDIS",
    "CIVIDIS",
    "BONE",
    "PARULA",
}
VALID_ESTIMATION_TYPES = {"auto", "relative", "metric"}


class ConfigError(ValueError):
    """Raised when the configuration file is missing, malformed or invalid."""


@dataclass
class SourceConfig:
    type: str = "camera"
    index: int = 0
    path: str | None = None
    width: int = 640
    height: int = 480
    fps: int = 30
    buffer_size: int = 1


@dataclass
class ReconnectConfig:
    enabled: bool = True
    interval_s: float = 2.0
    max_failures: int = 30


@dataclass
class ModelConfig:
    repo_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    local_dir: str | None = None
    local_files_only: bool = True
    input_size: int = 518
    precision: str = "fp32"
    device: str = "auto"
    warmup_iters: int = 3
    sync_cuda_timing: bool = True


@dataclass
class SmoothingConfig:
    enabled: bool = True
    alpha: float = 0.45
    median_window: int = 1


@dataclass
class VisualizationConfig:
    colormap: str = "TURBO"
    grayscale: bool = False
    range_mode: str = "auto"
    range_fixed: list[float] = field(default_factory=lambda: [0.0, 1.0])
    range_smooth: float = 0.15


@dataclass
class DepthConfig:
    estimation_type: str = "auto"
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


@dataclass
class MeasurementConfig:
    region_size: int = 7
    max_points: int = 8
    show_center: bool = True
    point_smoothing: float = 0.5


@dataclass
class CalibrationConfig:
    profile: str = "calibration_data/profile.json"
    min_range_m: float = 0.3
    max_range_m: float = 15.0
    warn_extrapolation: bool = True


@dataclass
class GeometryConfig:
    intrinsics: str = "calibration_data/intrinsics.json"
    undistort: bool = False


@dataclass
class AnchorConfig:
    enabled: bool = False
    marker_length_m: float = 0.12
    dictionary: str = "DICT_4X4_50"
    validate_only: bool = True


@dataclass
class DisplayConfig:
    window_name: str = "Depth Anything V2 - Live"
    window_scale: float = 1.0
    show_perf: bool = True
    show_legend: bool = True
    screenshot_dir: str = "outputs"
    depth_age_warning_ms: int = 250


@dataclass
class RuntimeConfig:
    frame_skip: int = 1
    headless_frames: int = 0
    debug: bool = False


@dataclass
class AppConfig:
    source: SourceConfig = field(default_factory=SourceConfig)
    camera_reconnect: ReconnectConfig = field(default_factory=ReconnectConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    anchor: AnchorConfig = field(default_factory=AnchorConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    config_path: Path = DEFAULT_CONFIG_PATH
    base_dir: Path = APP_ROOT

    def resolve(self, relative: str | Path | None) -> Path | None:
        """Resolve a config-relative path against the application root."""
        if relative is None:
            return None
        p = Path(relative)
        return p if p.is_absolute() else (self.base_dir / p)


def _build(cls: type, data: Any, path: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"'{path}' must be a mapping, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(f"Unknown key(s) in '{path}': {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        f = known[name]
        if isinstance(value, dict):
            nested = _NESTED.get((cls.__name__, name))
            if nested is None:
                raise ConfigError(f"Unexpected mapping for '{path}.{name}'")
            kwargs[name] = _build(nested, value, f"{path}.{name}")
        else:
            kwargs[name] = value
    return cls(**kwargs)


_NESTED: dict[tuple[str, str], type] = {
    ("AppConfig", "source"): SourceConfig,
    ("AppConfig", "camera_reconnect"): ReconnectConfig,
    ("AppConfig", "model"): ModelConfig,
    ("AppConfig", "depth"): DepthConfig,
    ("AppConfig", "measurement"): MeasurementConfig,
    ("AppConfig", "calibration"): CalibrationConfig,
    ("AppConfig", "geometry"): GeometryConfig,
    ("AppConfig", "anchor"): AnchorConfig,
    ("AppConfig", "display"): DisplayConfig,
    ("AppConfig", "runtime"): RuntimeConfig,
    ("DepthConfig", "smoothing"): SmoothingConfig,
    ("DepthConfig", "visualization"): VisualizationConfig,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def validate_config(cfg: AppConfig) -> AppConfig:
    _require(cfg.source.type in {"camera", "video", "image"}, "source.type must be camera|video|image")
    if cfg.source.type == "camera":
        _require(cfg.source.index >= 0, "source.index must be >= 0")
    else:
        _require(bool(cfg.source.path), "source.path is required when source.type is not 'camera'")
    _require(cfg.source.width > 0 and cfg.source.height > 0, "source width/height must be > 0")
    _require(cfg.source.buffer_size >= 1, "source.buffer_size must be >= 1")

    _require(cfg.model.input_size >= 126, "model.input_size must be >= 126")
    _require(cfg.model.input_size % 14 == 0, "model.input_size must be a multiple of 14 (ViT patch size)")
    _require(cfg.model.precision in VALID_PRECISIONS, f"model.precision must be one of {sorted(VALID_PRECISIONS)}")
    _require(cfg.model.device in VALID_DEVICES, f"model.device must be one of {sorted(VALID_DEVICES)}")
    _require(cfg.model.warmup_iters >= 0, "model.warmup_iters must be >= 0")

    _require(cfg.depth.estimation_type in VALID_ESTIMATION_TYPES, "depth.estimation_type must be auto|relative|metric")
    s = cfg.depth.smoothing
    _require(0.0 < s.alpha <= 1.0, "depth.smoothing.alpha must be in (0, 1]")
    _require(s.median_window >= 1, "depth.smoothing.median_window must be >= 1")
    v = cfg.depth.visualization
    _require(v.colormap.upper() in VALID_COLORMAPS, f"depth.visualization.colormap must be one of {sorted(VALID_COLORMAPS)}")
    _require(v.range_mode in VALID_RANGE_MODES, "depth.visualization.range_mode must be auto|fixed")
    _require(len(v.range_fixed) == 2 and v.range_fixed[1] > v.range_fixed[0], "range_fixed must be [lo, hi] with lo < hi")
    _require(0.0 <= v.range_smooth <= 1.0, "range_smooth must be in [0, 1]")

    m = cfg.measurement
    _require(m.region_size >= 1 and m.region_size % 2 == 1, "measurement.region_size must be a positive odd number")
    _require(m.max_points >= 1, "measurement.max_points must be >= 1")
    _require(0.0 <= m.point_smoothing <= 1.0, "measurement.point_smoothing must be in [0, 1]")

    c = cfg.calibration
    _require(c.min_range_m > 0 and c.max_range_m > c.min_range_m, "calibration range must satisfy 0 < min < max")

    _require(cfg.runtime.frame_skip >= 1, "runtime.frame_skip must be >= 1")
    _require(cfg.display.window_scale > 0, "display.window_scale must be > 0")
    _require(cfg.anchor.marker_length_m > 0, "anchor.marker_length_m must be > 0")
    return cfg


def load_config(path: str | Path | None = None) -> AppConfig:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise ConfigError(f"Configuration file not found: {cfg_path}")
    try:
        raw = yaml.safe_load(cfg_path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {cfg_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Top level of {cfg_path} must be a mapping")
    cfg = _build(AppConfig, raw, "config")
    cfg.config_path = cfg_path.resolve()
    cfg.base_dir = cfg.config_path.parent
    return validate_config(cfg)
