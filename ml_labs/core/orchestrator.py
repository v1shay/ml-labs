# ml_labs/core/orchestrator.py

from __future__ import annotations

import uuid
from typing import Optional

from ml_labs.config import AppConfig
from ml_labs.core.types import (
    ProjectState,
    ExecutionPhase,
    ProblemType,
    DatasetModality,
)
from ml_labs.core.ingest import load_dataset
from ml_labs.core.profiler import profile_dataset
from ml_labs.core.llm_inference import infer_strategy
from ml_labs.execution.tabular_trainer import train_tabular_model
from ml_labs.execution.cv_trainer import train_cv_model


class ForgeOrchestrator:
    """
    Deterministic lifecycle controller for Forge.

    Phase-driven execution state machine.
    """

    def __init__(self, config: AppConfig | None = None):
        self.config = config or AppConfig()

    def run_project(
        self,
        dataset_path: str,
        target_column: Optional[str] = None,
    ) -> ProjectState:

        state = ProjectState(
            project_id=str(uuid.uuid4()),
            dataset_path=dataset_path,
        )

        try:
            # -----------------------------
            # Phase 1: Ingestion
            # -----------------------------
            dataset = load_dataset(dataset_path, config=self.config)
            state.dataset = dataset  # ✅ FIX: Persist dataset in state
            state.phase = ExecutionPhase.INGESTED

            # -----------------------------
            # Phase 2: Profiling
            # -----------------------------
            profile = profile_dataset(dataset)
            state.profile = profile
            state.phase = ExecutionPhase.PROFILED

            # -----------------------------
            # Phase 3: Strategy Inference
            # -----------------------------
            strategy = infer_strategy(
                profile,
                target_column_override=target_column,
            )
            state.strategy = strategy
            state.phase = ExecutionPhase.STRATEGY_INFERRED

            # -----------------------------
            # Phase 4: Deterministic Transition
            # -----------------------------
            self._determine_next_phase(state)

            # -----------------------------
            # Phase 5: Execution
            # -----------------------------
            if state.phase == ExecutionPhase.READY_FOR_EXECUTION:
                result = self._execute(state)
                state.model_result = result
                state.phase = ExecutionPhase.COMPLETED

        except Exception as e:
            state.phase = ExecutionPhase.FAILED
            state.errors.append(str(e))

        return state

    # -------------------------------------
    # Execution Layer Bridge
    # -------------------------------------

    def _execute(self, state: ProjectState):
        """
        Deterministic execution router.
        """

        action = state.next_actions[0]

        if action == "train_tabular_model":
            return train_tabular_model(
                df=state.dataset.data,  # ✅ FIXED
                target=state.strategy.target_column,
                problem_type=state.strategy.problem_type.value,
                metric=state.strategy.metric,
            )

        elif action == "train_cv_model":
            return train_cv_model(
                dataset_path=state.dataset_path,
            )

        else:
            raise ValueError(f"Unknown execution action: {action}")

    # -------------------------------------
    # Deterministic Transition Logic
    # -------------------------------------

    def _determine_next_phase(self, state: ProjectState) -> None:

        if state.strategy is None:
            state.phase = ExecutionPhase.FAILED
            state.errors.append("Strategy inference failed.")
            return

        if state.strategy.requires_user_input:
            state.phase = ExecutionPhase.AWAITING_USER_INPUT
            state.next_actions = state.strategy.next_actions
            state.requires_user_input = True
            return

        problem_type = state.strategy.problem_type
        modality = state.strategy.modality

        if not state.strategy.next_actions:
            state.phase = ExecutionPhase.FAILED
            state.errors.append("Strategy has no next actions defined.")
            return

        if modality == DatasetModality.TABULAR:
            if problem_type in (
                ProblemType.CLASSIFICATION,
                ProblemType.REGRESSION,
                ProblemType.UNSUPERVISED,
            ):
                state.phase = ExecutionPhase.READY_FOR_EXECUTION
                state.next_actions = state.strategy.next_actions
                state.requires_user_input = False
                return

        elif modality in (DatasetModality.IMAGE, DatasetModality.AUDIO):
            if problem_type == ProblemType.CLASSIFICATION:
                state.phase = ExecutionPhase.READY_FOR_EXECUTION
                state.next_actions = state.strategy.next_actions
                state.requires_user_input = False
                return

        if state.strategy.next_actions:
            state.phase = ExecutionPhase.READY_FOR_EXECUTION
            state.next_actions = state.strategy.next_actions
            state.requires_user_input = False
            return

        state.phase = ExecutionPhase.FAILED
        state.errors.append(
            f"Unknown problem type or modality: {problem_type}, {modality}"
        )
