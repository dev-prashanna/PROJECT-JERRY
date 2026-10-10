"""Configuration package for the Moondream live VQA application."""

from .settings import (  # noqa: F401
    APP_ROOT,
    OFFLINE_MODE,
    WORKSPACE_ROOT,
    CameraConfig,
    InferenceConfig,
    ModelConfig,
    ModelNotFoundError,
    AppConfig,
    build_config,
    discover_model,
    load_env_file,
    resolve_model_path,
    supported_resolutions,
)
