from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Literal

import pandas as pd


# =====================================================
# ENUMS
# =====================================================

class ProblemType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    UNSUPERVISED = "unsupervised"
    UNKNOWN = "unknown"


class DatasetModality(str, Enum):
    TABULAR = "tabular"
    IMAGE = "image"
    AUDIO = "audio"
    TEXT = "text"


class ExecutionPhase(str, Enum):
    INITIALIZED = "initialized"
    INGESTED = "ingested"
    PROFILED = "profiled"
    STRATEGY_INFERRED = "strategy_inferred"
    AWAITING_USER_INPUT = "awaiting_user_input"
    READY_FOR_EXECUTION = "ready_for_execution"
    FAILED = "failed"


SemanticType = Literal[
    "numeric",
    "categorical",
    "boolean",
    "datetime",
    "text",
    "unknown",
]


# =====================================================
# CORE DATA STRUCTURES
# =====================================================

@dataclass(frozen=True, slots=True)
class Dataset:
    dataset_id: str
    path: str
    loaded_at: datetime
    modality: DatasetModality
    dataframe: Optional[pd.DataFrame] = field(default=None, repr=False)
    file_paths: Optional[List[str]] = field(default=None)


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    name: str
    pandas_dtype: str
    semantic_type: SemanticType
    missing_pct: float
    unique_count: int
    unique_pct: float
    example_values: List[str]


@dataclass(frozen=True, slots=True)
class TargetCandidate:
    column: str
    score: float
    reasons: List[str]


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    modality: DatasetModality
    row_count: int
    column_count: int
    columns: Dict[str, ColumnProfile]
    missing_by_column_pct: Dict[str, float]
    target_candidates: List[TargetCandidate]
    class_balance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    modality_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =====================================================
# STRATEGY
# =====================================================

@dataclass(frozen=True, slots=True)
class StrategySpec:
    modality: DatasetModality
    problem_type: ProblemType
    target_column: Optional[str]
    metric: Optional[str]
    recommended_models: List[str]
    preprocessing: List[str]
    reasoning: Dict[str, Any]

    # Production lifecycle controls
    next_actions: List[str] = field(default_factory=list)
    requires_user_input: bool = False


# =====================================================
# PROJECT STATE
# =====================================================

@dataclass(slots=True)
class ProjectState:
    project_id: str
    dataset_path: str

    phase: ExecutionPhase = ExecutionPhase.INITIALIZED
    profile: Optional[DatasetProfile] = None
    strategy: Optional[StrategySpec] = None

    next_actions: List[str] = field(default_factory=list)
    requires_user_input: bool = False
    errors: List[str] = field(default_factory=list)


# =====================================================
# UTIL
# =====================================================

def utcnow() -> datetime:
    return datetime.now(timezone.utc)
