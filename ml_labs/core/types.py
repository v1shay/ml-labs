from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

import pandas as pd


class ProjectStatus(str, Enum):
    CREATED = "created"
    INGESTED = "ingested"
    PROFILED = "profiled"
    STRATEGY_INFERRED = "strategy_inferred"
    FAILED = "failed"


class ProblemType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    UNSUPERVISED = "unsupervised"
    UNKNOWN = "unknown"


SemanticType = Literal[
    "numeric",
    "categorical",
    "boolean",
    "datetime",
    "text",
    "unknown",
]


@dataclass(frozen=True, slots=True)
class Dataset:
    dataset_id: str
    path: str
    loaded_at: datetime
    dataframe: pd.DataFrame = field(repr=False)


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    name: str
    pandas_dtype: str
    semantic_type: SemanticType
    missing_pct: float
    unique_count: int
    unique_pct: float
    example_values: list[str]


@dataclass(frozen=True, slots=True)
class TargetCandidate:
    column: str
    score: float
    reasons: list[str]


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    row_count: int
    column_count: int
    columns: dict[str, ColumnProfile]
    missing_by_column_pct: dict[str, float]
    target_candidates: list[TargetCandidate]
    class_balance: dict[str, dict[str, float]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True, slots=True)
class StrategySpec:
    problem_type: ProblemType
    target_column: str | None
    metric: str
    recommended_models: list[str]
    preprocessing: list[str]
    reasoning: dict[str, Any]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)

# ml_labs/core/types.py

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any


class ExecutionPhase(str, Enum):
    INITIALIZED = "initialized"
    INGESTED = "ingested"
    PROFILED = "profiled"
    STRATEGY_INFERRED = "strategy_inferred"
    AWAITING_USER_INPUT = "awaiting_user_input"
    READY_FOR_EXECUTION = "ready_for_execution"
    FAILED = "failed"


@dataclass
class DatasetProfile:
    row_count: int
    column_count: int
    column_types: Dict[str, str]
    missing_percentages: Dict[str, float]
    candidate_targets: List[str]
    class_balance: Optional[Dict[str, float]] = None


@dataclass
class StrategySpec:
    problem_type: str  # classification | regression | unsupervised
    selected_target_column: Optional[str]
    metric: Optional[str]
    recommended_models: List[str]
    preprocessing: List[str]
    reasoning: Dict[str, Any]

    # NEW PRODUCTION FIELDS
    next_actions: List[str] = field(default_factory=list)
    requires_user_input: bool = False


@dataclass
class ProjectState:
    project_id: str
    dataset_path: str

    phase: ExecutionPhase = ExecutionPhase.INITIALIZED
    profile: Optional[DatasetProfile] = None
    strategy: Optional[StrategySpec] = None

    next_actions: List[str] = field(default_factory=list)
    requires_user_input: bool = False
    errors: List[str] = field(default_factory=list)
