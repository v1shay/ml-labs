from __future__ import annotations

import logging
import uuid
from dataclasses import replace

from ml_labs.config import AppConfig
from ml_labs.core.ingest import IngestionError, ingest_csv
from ml_labs.core.llm_inference import infer_strategy
from ml_labs.core.profiler import profile_dataset
from ml_labs.core.project import ProjectState, new_project_state
from ml_labs.core.types import ProjectStatus


class ForgeOrchestrator:
    """Orchestrates project execution state.

    This class is designed as a service-layer boundary that can be mounted into
    future FastAPI endpoints without refactoring core logic.
    """

    def __init__(self, *, config: AppConfig | None = None, logger: logging.Logger | None = None) -> None:
        self._config = config or AppConfig()
        self._logger = logger or logging.getLogger(self.__class__.__name__)

    def run_project(self, dataset_path: str, *, target_column: str | None = None) -> ProjectState:
        project_id = str(uuid.uuid4())
        state = new_project_state(
            project_id=project_id,
            dataset_path=str(self._config.normalize_path(dataset_path)),
        )

        try:
            dataset = ingest_csv(dataset_path, config=self._config)
            state = replace(state, status=ProjectStatus.INGESTED)

            profile = profile_dataset(dataset.dataframe)
            state = replace(state, profile=profile, status=ProjectStatus.PROFILED)

            strategy = infer_strategy(profile, target_column_override=target_column)
            state = replace(
                state,
                strategy=strategy,
                status=ProjectStatus.STRATEGY_INFERRED,
            )

            return state

        except IngestionError:
            self._logger.exception("Ingestion failed")
            return replace(state, status=ProjectStatus.FAILED)
        except Exception:
            self._logger.exception("Project run failed")
            return replace(state, status=ProjectStatus.FAILED)
