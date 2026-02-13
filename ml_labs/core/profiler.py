from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from pathlib import Path
from collections import Counter

from ml_labs.core.types import (
    ColumnProfile,
    DatasetProfile,
    DatasetModality,
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
# TABULAR PROFILING
# -------------------------------------------------

def profile_tabular(dataset: Dataset) -> DatasetProfile:
    """
    Profile a tabular Dataset object.

    Extracts dataframe safely.
    Keeps all existing tabular profiling logic unchanged.
    """

    if dataset.dataframe is None:
        raise ValueError("Tabular dataset must have a dataframe")

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
        modality=DatasetModality.TABULAR,
        row_count=row_count,
        column_count=int(df.shape[1]),
        columns=columns,
        missing_by_column_pct=missing_by_col,
        target_candidates=candidates_sorted,
        class_balance=class_balance,
        modality_metadata={},
    )


# -------------------------------------------------
# IMAGE PROFILING
# -------------------------------------------------

def profile_image(dataset: Dataset) -> DatasetProfile:
    """
    Profile an image Dataset object.

    Counts images, infers class labels from subfolder names,
    computes class distribution, and file extension stats.
    """
    if dataset.file_paths is None:
        raise ValueError("Image dataset must have file_paths")

    file_paths = dataset.file_paths
    image_count = len(file_paths)

    # Infer class labels from subfolder names
    # Assumes structure: root/class_name/image.jpg
    class_labels: dict[str, int] = {}
    extension_counts: dict[str, int] = {}

    for file_path in file_paths:
        path_obj = Path(file_path)
        extension = path_obj.suffix.lower()
        extension_counts[extension] = extension_counts.get(extension, 0) + 1

        # Try to infer class from parent directory name
        # Look for a subdirectory that's not the root
        parent = path_obj.parent
        if parent.name and parent.name != Path(dataset.path).name:
            class_labels[parent.name] = class_labels.get(parent.name, 0) + 1

    # If no subfolder structure, treat as single class
    if not class_labels:
        class_labels["default"] = image_count

    # Compute class distribution
    total = sum(class_labels.values())
    class_distribution = {
        label: round(count / total, 6) if total > 0 else 0.0
        for label, count in class_labels.items()
    }

    # Create empty column structures for compatibility
    columns: dict[str, ColumnProfile] = {}
    missing_by_col: dict[str, float] = {}
    target_candidates: list[TargetCandidate] = []
    class_balance: dict[str, dict[str, float]] = {}

    # Store class distribution in class_balance for consistency
    if class_distribution:
        class_balance["inferred_class"] = class_distribution

    modality_metadata: dict[str, Any] = {
        "image_count": image_count,
        "class_labels": list(class_labels.keys()),
        "class_distribution": class_distribution,
        "file_extensions": extension_counts,
        "inferred_from_subfolders": len(class_labels) > 1 or (len(class_labels) == 1 and "default" not in class_labels),
    }

    return DatasetProfile(
        modality=DatasetModality.IMAGE,
        row_count=image_count,
        column_count=0,  # No columns for image data
        columns=columns,
        missing_by_column_pct=missing_by_col,
        target_candidates=target_candidates,
        class_balance=class_balance,
        modality_metadata=modality_metadata,
    )


# -------------------------------------------------
# AUDIO PROFILING
# -------------------------------------------------

def profile_audio(dataset: Dataset) -> DatasetProfile:
    """
    Profile an audio Dataset object.

    Counts audio files, infers class labels from subfolder names,
    computes class distribution, and file type stats.
    """
    if dataset.file_paths is None:
        raise ValueError("Audio dataset must have file_paths")

    file_paths = dataset.file_paths
    audio_count = len(file_paths)

    # Infer class labels from subfolder names
    class_labels: dict[str, int] = {}
    extension_counts: dict[str, int] = {}

    for file_path in file_paths:
        path_obj = Path(file_path)
        extension = path_obj.suffix.lower()
        extension_counts[extension] = extension_counts.get(extension, 0) + 1

        # Try to infer class from parent directory name
        parent = path_obj.parent
        if parent.name and parent.name != Path(dataset.path).name:
            class_labels[parent.name] = class_labels.get(parent.name, 0) + 1

    # If no subfolder structure, treat as single class
    if not class_labels:
        class_labels["default"] = audio_count

    # Compute class distribution
    total = sum(class_labels.values())
    class_distribution = {
        label: round(count / total, 6) if total > 0 else 0.0
        for label, count in class_labels.items()
    }

    # Create empty column structures for compatibility
    columns: dict[str, ColumnProfile] = {}
    missing_by_col: dict[str, float] = {}
    target_candidates: list[TargetCandidate] = []
    class_balance: dict[str, dict[str, float]] = {}

    # Store class distribution in class_balance for consistency
    if class_distribution:
        class_balance["inferred_class"] = class_distribution

    modality_metadata: dict[str, Any] = {
        "audio_count": audio_count,
        "class_labels": list(class_labels.keys()),
        "class_distribution": class_distribution,
        "file_extensions": extension_counts,
        "inferred_from_subfolders": len(class_labels) > 1 or (len(class_labels) == 1 and "default" not in class_labels),
    }

    return DatasetProfile(
        modality=DatasetModality.AUDIO,
        row_count=audio_count,
        column_count=0,  # No columns for audio data
        columns=columns,
        missing_by_column_pct=missing_by_col,
        target_candidates=target_candidates,
        class_balance=class_balance,
        modality_metadata=modality_metadata,
    )


# -------------------------------------------------
# MAIN PROFILE FUNCTION (MODALITY-AWARE)
# -------------------------------------------------

def profile_dataset(dataset: Dataset) -> DatasetProfile:
    """
    Profile a Dataset object with modality-aware routing.

    Routes to appropriate profiling function based on dataset modality.
    """
    if dataset.modality == DatasetModality.TABULAR:
        return profile_tabular(dataset)
    elif dataset.modality == DatasetModality.IMAGE:
        return profile_image(dataset)
    elif dataset.modality == DatasetModality.AUDIO:
        return profile_audio(dataset)
    else:
        raise ValueError(f"Unsupported modality for profiling: {dataset.modality}")


# -------------------------------------------------
# LLM Prompt Representation
# -------------------------------------------------

def profile_to_prompt_dict(profile: DatasetProfile) -> dict[str, Any]:
    """Convert DatasetProfile to a dictionary suitable for LLM prompts."""

    base_dict: dict[str, Any] = {
        "modality": profile.modality.value,
        "row_count": profile.row_count,
        "column_count": profile.column_count,
        "class_balance": profile.class_balance,
        "modality_metadata": profile.modality_metadata,
    }

    # Add tabular-specific fields
    if profile.modality == DatasetModality.TABULAR:
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

        base_dict["columns"] = cols
        base_dict["target_candidates"] = candidates

    return base_dict
