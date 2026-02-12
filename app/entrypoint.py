# app/entrypoint.py

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, is_dataclass
from enum import Enum
from datetime import datetime
from typing import Any

from ml_labs.config import AppConfig, LoggingConfig, configure_logging
from ml_labs.core.orchestrator import ForgeOrchestrator


def _to_jsonable(obj: Any) -> Any:
    """
    Safely serialize:
    - dataclasses
    - enums
    - datetime
    """

    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}

    if isinstance(obj, Enum):
        return obj.value

    if isinstance(obj, datetime):
        return obj.isoformat()

    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]

    return obj


def main() -> int:
    parser = argparse.ArgumentParser(prog="ml-labs")

    parser.add_argument(
        "--data",
        dest="dataset_path",
        type=str,
        help="Path to dataset",
    )

    parser.add_argument(
        "--target",
        dest="target_column",
        type=str,
        default=None,
        help="Target column",
    )

    parser.add_argument(
        "dataset_path_pos",
        nargs="?",
        type=str,
        help="Path to dataset (positional)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    dataset_path = args.dataset_path or args.dataset_path_pos
    if not dataset_path:
        parser.error("Provide dataset path via --data or positional argument")

    configure_logging(LoggingConfig(level=args.log_level))

    try:
        cfg = AppConfig(log_level=args.log_level)
        orchestrator = ForgeOrchestrator(config=cfg)

        state = orchestrator.run_project(
            dataset_path=dataset_path,
            target_column=args.target_column,
        )

        print(json.dumps(_to_jsonable(state), indent=2, sort_keys=True))
        return 0

    except Exception as e:
        logging.exception("Fatal error")
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    main()
