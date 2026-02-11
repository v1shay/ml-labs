from __future__ import annotations

import logging
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(cfg: LoggingConfig) -> None:
    """Configure Python logging.

    No global logger instances are kept; callers should use logging.getLogger.
    """

    level = getattr(logging, cfg.level.upper(), logging.INFO)
    logging.basicConfig(level=level, format=cfg.format)
