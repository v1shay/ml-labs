from __future__ import annotations

from typing import Any

from ml_labs.core.profiler import profile_to_prompt_dict
from ml_labs.core.types import DatasetProfile, ProblemType, StrategySpec


MIN_TARGET_SCORE = 0.8  # safer threshold than hard-coded 1.0


def _choose_metric(problem_type: ProblemType) -> str:
    if problem_type == ProblemType.CLASSIFICATION:
        return "f1"
    if problem_type == ProblemType.REGRESSION:
        return "rmse"
    if problem_type == ProblemType.UNSUPERVISED:
        return "silhouette"
    return "n/a"


def _recommended_models(problem_type: ProblemType) -> list[str]:
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


def infer_strategy(
    profile: DatasetProfile,
    *,
    target_column_override: str | None = None,
) -> StrategySpec:
    """
    Infer an ML strategy from a DatasetProfile.

    Deterministic LLM-ready stub with lifecycle-aware action planning.
    """

    prompt_payload = profile_to_prompt_dict(profile)

    top_candidate = profile.target_candidates[0] if profile.target_candidates else None

    # -----------------------------------------
    # Target Selection Logic
    # -----------------------------------------

    if target_column_override and target_column_override in profile.columns:
        target_column = target_column_override
        override_reason = "target override provided and exists in dataset"

    elif target_column_override and target_column_override not in profile.columns:
        return StrategySpec(
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
        next_actions = ["train_supervised_model"]
        requires_user_input = False

    elif problem_type == ProblemType.UNSUPERVISED:
        next_actions = ["run_unsupervised_analysis"]
        requires_user_input = False

    else:
        return StrategySpec(
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
            "A future implementation should replace heuristic inference with an LLM call.",
        ],
    }

    # REAL LLM CALL boundary:
    # response = llm_client.complete(messages=[...], response_schema=StrategySpec)
    # return response

    return StrategySpec(
        problem_type=problem_type,
        target_column=target_column,
        metric=_choose_metric(problem_type),
        recommended_models=_recommended_models(problem_type),
        preprocessing=preprocessing,
        reasoning=reasoning,
        next_actions=next_actions,
        requires_user_input=requires_user_input,
    )
