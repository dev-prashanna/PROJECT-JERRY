"""Logging helpers: one console handler plus a rotating file handler."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from config.settings import LOG_DIR

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%H:%M:%S"

_configured = False


def configure_logging(
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
    force: bool = False,
) -> Path:
    """Configure the root logger.

    Parameters
    ----------
    level:
        Root level name (``DEBUG``, ``INFO``, ...).
    log_file:
        Explicit log path; defaults to ``<app>/logs/app.log``.
    force:
        Re-configure even if this function already ran (used by tests).

    Returns
    -------
    Path
        The file that logs are written to.
    """
    global _configured
    root = logging.getLogger()
    numeric = getattr(logging, str(level).upper(), logging.INFO)

    if _configured and not force:
        root.setLevel(numeric)
        return Path(log_file) if log_file else LOG_DIR / "app.log"

    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(numeric)
    root.addHandler(console)

    target = Path(log_file) if log_file else LOG_DIR / "app.log"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            target, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        file_handler.setLevel(numeric)
        root.addHandler(file_handler)
    except OSError as exc:  # pragma: no cover - log directory not writable
        root.warning("File logging disabled (%s)", exc)

    root.setLevel(numeric)
    # Third-party chatter is noisy at DEBUG.
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    _configured = True
    return target


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger."""
    return logging.getLogger(name)
