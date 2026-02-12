from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from ml_labs.core.types import (
    ColumnProfile,
    DatasetProfile,
    SemanticType,
    TargetCandidate,
    Dataset,
)


# -------------------------------------------------
# Semantic Type Inference
# -------------------------------------------------

def _infer_semantic_type(series: pd.Series) -> SemanticType:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        non_null = series.dropna()
        if len(non_null) == 0:
            return "unknown"
        sample = non_null.astype(str).head(200)
        avg_len = sample.map(len).mean()
        return "text" if avg_len >= 30 else "categorical"
    return "unknown"


def _example_values(series: pd.Series, max_examples: int = 5) -> list[str]:
    non_null = series.dropna()
    if len(non_null) == 0:
        return []
    values = non_null.astype(str)
    counts = Counter(values)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [v for v, _ in ordered[:max_examples]]


# -------------------------------------------------
# Target Heuristics
# -------------------------------------------------

def _target_name_hint(col: str) -> float:
    c = col.strip().lower()
    hints = {
        "target": 1.0,
        "label": 0.95,
        "y": 0.8,
        "class": 0.75,
        "outcome": 0.7,
        "response": 0.65,
    }
    if c in hints:
        return hints[c]
    for k, v in hints.items():
        if c.endswith(f"_{k}") or c.startswith(f"{k}_"):
            return v * 0.9
    return 0.0


def _candidate_score(
    *,
    col: str,
    series: pd.Series,
    semantic_type: SemanticType,
    missing_pct: float,
    unique_count: int,
    row_count: int,
    is_last_column: bool,
) -> TargetCandidate:

    reasons: list[str] = []
    score = 0.0

    name_score = _target_name_hint(col)
    if name_score > 0:
        score += 3.0 * name_score
        reasons.append("column name matches common target naming")

    if is_last_column:
        score += 0.35
        reasons.append("last column convention")

    if missing_pct <= 5.0:
        score += 0.6
        reasons.append("low missingness")
    elif missing_pct <= 20.0:
        score += 0.2
        reasons.append("moderate missingness")
    else:
        score -= 0.4
        reasons.append("high missingness")

    unique_ratio = unique_count / max(row_count, 1) if row_count > 0 else 0.0

    if unique_ratio >= 0.98 and row_count > 0:
        score -= 1.2
        reasons.append("near-unique values suggest identifier")

    if semantic_type in ("numeric", "categorical", "boolean"):
        score += 0.7
        reasons.append("plausible target semantic type")
    elif semantic_type == "text":
        score += 0.2
        reasons.append("text targets possible")
    else:
        score -= 0.2
        reasons.append("unclear semantic type")

    if semantic_type in ("categorical", "text") and row_count > 0:
        if unique_ratio >= 0.5:
            score -= 0.6
            reasons.append("very high cardinality")

    return TargetCandidate(
        column=col,
        score=round(score, 4),
        reasons=reasons,
    )


# -------------------------------------------------
# MAIN PROFILE FUNCTION (V3 FIXED)
# -------------------------------------------------

def profile_dataset(dataset: Dataset) -> DatasetProfile:
    """
    Profile a Dataset object (v3 architecture).

    Extracts dataframe safely.
    Future-proof for modality expansion.
    """

    if not hasattr(dataset, "dataframe"):
        raise ValueError("Dataset does not contain a dataframe attribute")

    df: pd.DataFrame = dataset.dataframe

    row_count = int(df.shape[0])
    col_names = [str(c) for c in df.columns]

    columns: dict[str, ColumnProfile] = {}
    missing_by_col: dict[str, float] = {}

    for col in col_names:
        s = df[col]
        missing_pct = float(s.isna().mean() * 100.0)
        missing_by_col[col] = round(missing_pct, 4)

        non_null = s.dropna()
        unique_count = int(non_null.nunique(dropna=True))
        unique_pct = (
            float((unique_count / max(len(non_null), 1)) * 100.0)
            if len(non_null)
            else 0.0
        )

        semantic_type = _infer_semantic_type(s)

        columns[col] = ColumnProfile(
            name=col,
            pandas_dtype=str(s.dtype),
            semantic_type=semantic_type,
            missing_pct=round(missing_pct, 4),
            unique_count=unique_count,
            unique_pct=round(unique_pct, 4),
            example_values=_example_values(s),
        )

    # -------------------------------------------------
    # Target Candidates
    # -------------------------------------------------

    candidates: list[TargetCandidate] = []

    for i, col in enumerate(col_names):
        cp = columns[col]
        non_null = df[col].dropna()
        unique_count = int(non_null.nunique(dropna=True))

        candidates.append(
            _candidate_score(
                col=col,
                series=df[col],
                semantic_type=cp.semantic_type,
                missing_pct=cp.missing_pct,
                unique_count=unique_count,
                row_count=row_count,
                is_last_column=(i == len(col_names) - 1),
            )
        )

    candidates_sorted = sorted(candidates, key=lambda c: (-c.score, c.column))

    # -------------------------------------------------
    # Class Balance
    # -------------------------------------------------

    class_balance: dict[str, dict[str, float]] = {}

    if candidates_sorted:
        top = candidates_sorted[0]
        s = df[top.column]
        non_null = s.dropna()

        unique = int(non_null.nunique(dropna=True))

        if row_count > 0 and unique > 0 and unique <= min(20, max(2, int(row_count * 0.05) or 2)):
            counts = non_null.astype(str).value_counts(dropna=True)
            dist = {
                k: float(v / max(len(non_null), 1))
                for k, v in counts.items()
            }
            class_balance[top.column] = {
                k: round(dist[k], 6)
                for k in sorted(dist.keys())
            }

    return DatasetProfile(
        row_count=row_count,
        column_count=int(df.shape[1]),
        columns=columns,
        missing_by_column_pct=missing_by_col,
        target_candidates=candidates_sorted,
        class_balance=class_balance,
    )


# -------------------------------------------------
# LLM Prompt Representation
# -------------------------------------------------

def profile_to_prompt_dict(profile: DatasetProfile) -> dict[str, Any]:

    cols = []
    for name in sorted(profile.columns.keys()):
        cp = profile.columns[name]
        cols.append(
            {
                "name": cp.name,
                "dtype": cp.pandas_dtype,
                "semantic_type": cp.semantic_type,
                "missing_pct": cp.missing_pct,
                "unique_count": cp.unique_count,
                "unique_pct": cp.unique_pct,
                "examples": cp.example_values,
            }
        )

    candidates = [
        {
            "column": c.column,
            "score": c.score,
            "reasons": list(c.reasons),
        }
        for c in profile.target_candidates[:10]
    ]

    return {
        "row_count": profile.row_count,
        "column_count": profile.column_count,
        "columns": cols,
        "target_candidates": candidates,
        "class_balance": profile.class_balance,
    }
