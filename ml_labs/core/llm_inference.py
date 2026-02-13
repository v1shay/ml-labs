from __future__ import annotations

from typing import Any, Optional

from ml_labs.core.profiler import profile_to_prompt_dict
from ml_labs.core.types import DatasetModality, DatasetProfile, ProblemType, StrategySpec
from ml_labs.core.llm_client import LLMClient


MIN_TARGET_SCORE = 0.8  # safer threshold than hard-coded 1.0

# Global LLM client instance (can be configured via environment variable in future)
_llm_client: Optional[LLMClient] = None


def _get_llm_client() -> Optional[LLMClient]:
    """Get LLM client instance (lazy initialization)."""
    global _llm_client
    if _llm_client is None:
        # In future, can read from environment variable
        # api_key = os.getenv("LLM_API_KEY")
        _llm_client = LLMClient(api_key=None)  # Stub mode
    return _llm_client


def _choose_metric(problem_type: ProblemType) -> str:
    if problem_type == ProblemType.CLASSIFICATION:
        return "f1"
    if problem_type == ProblemType.REGRESSION:
        return "rmse"
    if problem_type == ProblemType.UNSUPERVISED:
        return "silhouette"
    return "n/a"


def _recommended_models_tabular(problem_type: ProblemType) -> list[str]:
    """Recommended models for tabular data."""
    if problem_type == ProblemType.CLASSIFICATION:
        return [
            "logistic_regression",
            "random_forest",
            "xgboost_or_lightgbm",
        ]
    if problem_type == ProblemType.REGRESSION:
        return [
            "linear_regression",
            "random_forest_regressor",
            "xgboost_or_lightgbm_regressor",
        ]
    if problem_type == ProblemType.UNSUPERVISED:
        return [
            "kmeans",
            "hdbscan",
            "pca_for_visualization",
        ]
    return ["manual_review"]


def _recommended_models_image() -> list[str]:
    """Recommended models for image data."""
    return [
        "resnet",
        "efficientnet",
        "vit",
    ]


def _recommended_models_audio() -> list[str]:
    """Recommended models for audio data."""
    return [
        "cnn_spectrogram",
        "wav2vec",
    ]


def _infer_problem_type(profile: DatasetProfile, target_column: str | None) -> ProblemType:
    if not target_column:
        return ProblemType.UNSUPERVISED

    cp = profile.columns.get(target_column)
    if cp is None:
        return ProblemType.UNKNOWN

    if cp.semantic_type in ("categorical", "boolean"):
        return ProblemType.CLASSIFICATION

    if cp.semantic_type == "numeric":
        unique = cp.unique_count
        if unique <= min(20, max(2, int(profile.row_count * 0.05) or 2)):
            return ProblemType.CLASSIFICATION
        return ProblemType.REGRESSION

    if cp.semantic_type == "text":
        return ProblemType.CLASSIFICATION

    return ProblemType.UNKNOWN


def _call_llm_strategy(prompt_payload: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    Call LLM to infer strategy from prompt payload.

    Returns:
        Optional dictionary with strategy specification, or None if LLM unavailable.
    """
    llm_client = _get_llm_client()
    if llm_client is None:
        return None

    try:
        response = llm_client.infer_strategy(prompt_payload)
        # Validate response schema here if needed
        return response
    except Exception:
        # Fall back to deterministic logic on any error
        return None


def _infer_strategy_tabular(
    profile: DatasetProfile,
    *,
    target_column_override: str | None = None,
) -> StrategySpec:
    """
    Infer strategy for tabular data.

    Uses existing deterministic logic with optional LLM enhancement.
    """
    prompt_payload = profile_to_prompt_dict(profile)

    # Try LLM enhancement first (if available)
    llm_response = _call_llm_strategy(prompt_payload)
    if llm_response is not None:
        # Validate and use LLM response
        # For now, fall through to deterministic logic
        pass

    top_candidate = profile.target_candidates[0] if profile.target_candidates else None

    # -----------------------------------------
    # Target Selection Logic
    # -----------------------------------------

    if target_column_override and target_column_override in profile.columns:
        target_column = target_column_override
        override_reason = "target override provided and exists in dataset"

    elif target_column_override and target_column_override not in profile.columns:
        return StrategySpec(
            modality=DatasetModality.TABULAR,
            problem_type=ProblemType.UNKNOWN,
            target_column=None,
            metric=None,
            recommended_models=[],
            preprocessing=[],
            reasoning={
                "error": "Provided target override not found in dataset.",
                "prompt_payload": prompt_payload,
            },
            next_actions=["request_valid_target"],
            requires_user_input=True,
        )

    else:
        target_column = (
            top_candidate.column
            if (top_candidate and top_candidate.score >= MIN_TARGET_SCORE)
            else None
        )
        override_reason = "no target override provided"

    # -----------------------------------------
    # Problem Type Inference
    # -----------------------------------------

    problem_type = _infer_problem_type(profile, target_column)

    # -----------------------------------------
    # Lifecycle Action Planning
    # -----------------------------------------

    if problem_type in (ProblemType.CLASSIFICATION, ProblemType.REGRESSION):
        next_actions = ["train_tabular_model"]
        requires_user_input = False

    elif problem_type == ProblemType.UNSUPERVISED:
        next_actions = ["run_unsupervised_analysis"]
        requires_user_input = False

    else:
        return StrategySpec(
            modality=DatasetModality.TABULAR,
            problem_type=ProblemType.UNKNOWN,
            target_column=None,
            metric=None,
            recommended_models=[],
            preprocessing=[],
            reasoning={
                "error": "Unable to infer valid problem type.",
                "prompt_payload": prompt_payload,
            },
            next_actions=["manual_review_required"],
            requires_user_input=True,
        )

    # -----------------------------------------
    # Preprocessing Plan
    # -----------------------------------------

    preprocessing: list[str] = [
        "train_test_split",
        "handle_missing_values",
        "encode_categoricals",
        "scale_numeric_features_if_needed",
    ]

    # -----------------------------------------
    # Reasoning Object
    # -----------------------------------------

    reasoning: dict[str, Any] = {
        "prompt_payload": prompt_payload,
        "decision": {
            "selected_target_column": target_column,
            "target_override": target_column_override,
            "target_override_reason": override_reason,
            "problem_type": problem_type.value,
            "why_target": (top_candidate.reasons if top_candidate else []),
        },
        "notes": [
            "This strategy was inferred deterministically from the dataset profile.",
            "LLM enhancement attempted but fell back to deterministic logic.",
        ],
    }

    return StrategySpec(
        modality=DatasetModality.TABULAR,
        problem_type=problem_type,
        target_column=target_column,
        metric=_choose_metric(problem_type),
        recommended_models=_recommended_models_tabular(problem_type),
        preprocessing=preprocessing,
        reasoning=reasoning,
        next_actions=next_actions,
        requires_user_input=requires_user_input,
    )


def _infer_strategy_image(profile: DatasetProfile) -> StrategySpec:
    """
    Infer strategy for image data.

    Defaults to classification with CV models.
    """
    prompt_payload = profile_to_prompt_dict(profile)

    # Extract class information from modality metadata
    class_labels = profile.modality_metadata.get("class_labels", [])
    class_distribution = profile.modality_metadata.get("class_distribution", {})

    reasoning: dict[str, Any] = {
        "prompt_payload": prompt_payload,
        "decision": {
            "problem_type": "classification",
            "inferred_from": "image dataset structure",
            "class_labels": class_labels,
            "class_distribution": class_distribution,
        },
        "notes": [
            "Image datasets default to classification tasks.",
            "Class labels inferred from subfolder structure.",
        ],
    }

    return StrategySpec(
        modality=DatasetModality.IMAGE,
        problem_type=ProblemType.CLASSIFICATION,
        target_column=None,  # No column for image data
        metric="accuracy",
        recommended_models=_recommended_models_image(),
        preprocessing=[
            "resize",
            "normalize",
            "train_val_split",
        ],
        reasoning=reasoning,
        next_actions=["train_cv_model"],
        requires_user_input=False,
    )


def _infer_strategy_audio(profile: DatasetProfile) -> StrategySpec:
    """
    Infer strategy for audio data.

    Defaults to classification with audio models.
    """
    prompt_payload = profile_to_prompt_dict(profile)

    # Extract class information from modality metadata
    class_labels = profile.modality_metadata.get("class_labels", [])
    class_distribution = profile.modality_metadata.get("class_distribution", {})

    reasoning: dict[str, Any] = {
        "prompt_payload": prompt_payload,
        "decision": {
            "problem_type": "classification",
            "inferred_from": "audio dataset structure",
            "class_labels": class_labels,
            "class_distribution": class_distribution,
        },
        "notes": [
            "Audio datasets default to classification tasks.",
            "Class labels inferred from subfolder structure.",
        ],
    }

    return StrategySpec(
        modality=DatasetModality.AUDIO,
        problem_type=ProblemType.CLASSIFICATION,
        target_column=None,  # No column for audio data
        metric="accuracy",
        recommended_models=_recommended_models_audio(),
        preprocessing=[
            "resample",
            "mel_spectrogram",
        ],
        reasoning=reasoning,
        next_actions=["train_audio_model"],
        requires_user_input=False,
    )


def infer_strategy(
    profile: DatasetProfile,
    *,
    target_column_override: str | None = None,
) -> StrategySpec:
    """
    Infer an ML strategy from a DatasetProfile (modality-aware).

    Routes to appropriate inference function based on dataset modality:
    - TABULAR: Uses existing deterministic logic with optional LLM enhancement
    - IMAGE: Classification with CV models
    - AUDIO: Classification with audio models
    """
    if profile.modality == DatasetModality.TABULAR:
        return _infer_strategy_tabular(profile, target_column_override=target_column_override)
    elif profile.modality == DatasetModality.IMAGE:
        return _infer_strategy_image(profile)
    elif profile.modality == DatasetModality.AUDIO:
        return _infer_strategy_audio(profile)
    else:
        # Fallback for unknown modalities
        return StrategySpec(
            modality=profile.modality,
            problem_type=ProblemType.UNKNOWN,
            target_column=None,
            metric=None,
            recommended_models=[],
            preprocessing=[],
            reasoning={
                "error": f"Unsupported modality: {profile.modality}",
            },
            next_actions=["manual_review_required"],
            requires_user_input=True,
        )

