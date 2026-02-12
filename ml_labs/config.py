# ml_labs/config.py

from __future__ import annotations

import logging
from dataclasses import dataclass


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class AppConfig:
    log_level: str = "INFO"

    def normalize_path(self, path: str):
        from pathlib import Path
        return Path(path).resolve()


def configure_logging(cfg: LoggingConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, cfg.level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
