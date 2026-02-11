from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from ml_labs.config import AppConfig
from ml_labs.logging_config import LoggingConfig, configure_logging
from ml_labs.core.orchestrator import ForgeOrchestrator


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(prog="ml-labs")
    parser.add_argument("dataset_path", type=str, help="Path to a CSV dataset")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    args = parser.parse_args()

    configure_logging(LoggingConfig(level=args.log_level))

    cfg = AppConfig(log_level=args.log_level)
    orchestrator = ForgeOrchestrator(config=cfg)

    state = orchestrator.run_project(args.dataset_path)

    print(json.dumps(_to_jsonable(state), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
