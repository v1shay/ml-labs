# ml_labs/core/orchestrator.py

from __future__ import annotations

import uuid
from typing import Optional

from ml_labs.config import AppConfig
from ml_labs.core.types import (
    ProjectState,
    ExecutionPhase,
    ProblemType,
)
from ml_labs.core.ingest import load_dataset
from ml_labs.core.profiler import profile_dataset
from ml_labs.core.llm_inference import infer_strategy


class ForgeOrchestrator:
    """
    Deterministic lifecycle controller for Forge.

    This is a phase-driven state machine.
    """

    def __init__(self, config: AppConfig | None = None):
        self.config = config or AppConfig()

    def run_project(
        self,
        dataset_path: str,
        target_override: Optional[str] = None,
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
                target_column_override=target_override,
            )
            state.strategy = strategy
            state.phase = ExecutionPhase.STRATEGY_INFERRED

            # -----------------------------
            # Phase 4: Deterministic Transition
            # -----------------------------
            self._determine_next_phase(state)

        except Exception as e:
            state.phase = ExecutionPhase.FAILED
            state.errors.append(str(e))

        return state

    # -------------------------------------
    # Deterministic Transition Logic
    # -------------------------------------

    def _determine_next_phase(self, state: ProjectState) -> None:
        """
        Converts strategy output into execution phase.
        LLM does NOT control lifecycle.
        """

        if state.strategy is None:
            state.phase = ExecutionPhase.FAILED
            state.errors.append("Strategy inference failed.")
            return

        # Case 1: Needs user input
        if state.strategy.requires_user_input:
            state.phase = ExecutionPhase.AWAITING_USER_INPUT
            state.next_actions = state.strategy.next_actions
            state.requires_user_input = True
            return

        problem_type = state.strategy.problem_type

        # Case 2: Supervised ML
        if problem_type in (
            ProblemType.CLASSIFICATION,
            ProblemType.REGRESSION,
        ):
            state.phase = ExecutionPhase.READY_FOR_EXECUTION
            state.next_actions = ["train_supervised_model"]
            state.requires_user_input = False
            return

        # Case 3: Unsupervised ML
        if problem_type == ProblemType.UNSUPERVISED:
            state.phase = ExecutionPhase.READY_FOR_EXECUTION
            state.next_actions = ["run_unsupervised_analysis"]
            state.requires_user_input = False
            return

        # Case 4: Unknown
        state.phase = ExecutionPhase.FAILED
        state.errors.append("Unknown problem type.")
