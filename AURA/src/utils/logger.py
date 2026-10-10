"""Central logging: console + rotating file, secrets never written."""

from __future__ import annotations

import logging
import os
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from config.settings import APP_ROOT, LOG_DIR

_SECRET_PATTERNS = (
    re.compile(r"gsk_[A-Za-z0-9\-_]+"),
    re.compile(r"(?i)\b(api[_-]?key|authorization|bearer|token)\b\s*[=:]\s*\S+"),
)


class SecretFilter(logging.Filter):
    """Redact anything that looks like an API credential."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        redacted = msg
        for pat in _SECRET_PATTERNS:
            redacted = pat.sub("<redacted>", redacted)
        if redacted != msg:
            record.msg = redacted
            record.args = ()
        return True


_configured = False


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> logging.Logger:
    global _configured
    root = logging.getLogger("aura")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if _configured:
        return root
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S"
    )
    secret = SecretFilter()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    console.addFilter(secret)
    root.addHandler(console)

    path = log_file or (LOG_DIR / "aura.log")
    try:
        fileh = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        fileh.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        fileh.addFilter(secret)
        root.addHandler(fileh)
    except OSError:
        root.warning("cannot open log file %s", path)

    _configured = True
    return root


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"aura.{name}")
    logger.addFilter(SecretFilter())
    if not _configured:
        setup_logging(os.environ.get("AURA_LOG_LEVEL", "INFO"))
    return logger
