from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Configuration for Vishay's ML Labs execution engine.

    This module is deliberately small and dependency-free so it can be imported
    by future service layers (e.g., FastAPI) without side effects.
    """

    workspace_dir: Path = Path.cwd()
    log_level: str = "INFO"

    @staticmethod
    def normalize_path(path: str | Path) -> Path:
        p = Path(path).expanduser()
        try:
            return p.resolve()
        except FileNotFoundError:
            # resolve() fails if parts of the path don't exist; keep best-effort.
            return p.absolute()
