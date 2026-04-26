#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from time import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42
MODEL_BUNDLE_FILE = "model.joblib"
RUNTIME_METADATA_FILE = "runtime-metadata.json"


def parse_args() -> argparse.Namespace:
    argv = sys.argv[1:]
    if not argv or argv[0].startswith("--"):
        argv = ["train", *argv]

    parser = argparse.ArgumentParser(description="Run the ML-Labs experiment engine.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train", help="Train a lab run and emit JSON results.")
    train_parser.add_argument("--csv", required=True, help="Path to the CSV file.")
    train_parser.add_argument("--target", required=True, help="Name of the target column.")
    train_parser.add_argument("--intent", required=False, help="Narrative prompt for the run.")
    train_parser.add_argument("--run-id", required=True, help="Run identifier for bundle storage.")
    train_parser.add_argument("--bundle-dir", required=True, help="Directory for runtime model artifacts.")

    predict_parser = subparsers.add_parser("predict", help="Score one new row from a saved runtime bundle.")
    predict_parser.add_argument("--bundle-dir", required=True, help="Directory containing the runtime bundle.")
    predict_parser.add_argument("--run-id", required=True, help="Run identifier for the prediction response.")
    predict_parser.add_argument("--input-json", required=True, help="JSON object representing one input row.")

    return parser.parse_args(argv)


def detect_problem_type(target_series: pd.Series) -> str:
    numeric_target = pd.api.types.is_numeric_dtype(target_series)
    unique_count = target_series.nunique(dropna=True)
    unique_ratio = unique_count / max(len(target_series), 1)

    if numeric_target and unique_count > 10 and unique_ratio > 0.05:
        return "regression"
    return "classification"


def build_target_summary(target_series: pd.Series, problem_type: str) -> str:
    if problem_type == "regression":
        return (
            f"mean={target_series.mean():.2f}, median={target_series.median():.2f}, "
            f"min={target_series.min():.2f}, max={target_series.max():.2f}"
        )

    top_counts = target_series.value_counts(dropna=False).head(3)
    parts = [f"{idx}: {count}" for idx, count in top_counts.items()]
    return "Top classes -> " + ", ".join(parts)


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", make_one_hot_encoder()),
                    ]
                ),
                categorical_columns,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor, numeric_columns, categorical_columns


def model_specs(problem_type: str) -> list[tuple[str, str, Any]]:
    if problem_type == "regression":
        return [
            ("Mean Regressor", "Baseline", DummyRegressor(strategy="mean")),
            ("Linear Regression", "Linear Model", LinearRegression()),
            (
                "Random Forest Regressor",
                "Tree Ensemble",
                RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
            ),
            (
                "Gradient Boosting Regressor",
                "Boosted Trees",
                GradientBoostingRegressor(random_state=RANDOM_STATE),
            ),
        ]

    return [
        ("Majority Classifier", "Baseline", DummyClassifier(strategy="most_frequent")),
        ("Logistic Regression", "Linear Model", LogisticRegression(max_iter=1000)),
        (
            "Random Forest Classifier",
            "Tree Ensemble",
            RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        ),
        (
            "Gradient Boosting Classifier",
            "Boosted Trees",
            GradientBoostingClassifier(random_state=RANDOM_STATE),
        ),
    ]


def target_split(y: pd.Series, problem_type: str):
    if problem_type == "classification" and y.nunique(dropna=True) > 1:
        class_counts = y.value_counts()
        if class_counts.min() >= 2:
            return y
    return None


def build_dataset_profile(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, Any]:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "targetColumn": target_column,
        "problemType": problem_type,
        "numericColumns": numeric_columns,
        "categoricalColumns": categorical_columns,
        "missingValues": {column: int(value) for column, value in df.isna().sum().to_dict().items()},
        "targetSummary": build_target_summary(df[target_column], problem_type),
    }


def build_prediction_input_schema(X: pd.DataFrame, problem_type: str, target_column: str) -> dict[str, Any]:
    fields: list[dict[str, Any]] = []
    for column in X.columns:
        series = X[column]
        if pd.api.types.is_bool_dtype(series):
            fields.append(
                {
                    "name": column,
                    "label": humanize_label(column),
                    "kind": "boolean",
                    "required": True,
                    "example": bool(series.dropna().mode().iloc[0]) if not series.dropna().empty else False,
                }
            )
            continue

        if pd.api.types.is_numeric_dtype(series):
            example = float(series.dropna().median()) if not series.dropna().empty else 0.0
            if float(example).is_integer():
                example = int(example)
            fields.append(
                {
                    "name": column,
                    "label": humanize_label(column),
                    "kind": "number",
                    "required": True,
                    "example": example,
                }
            )
            continue

        options = sorted(series.dropna().astype(str).unique().tolist())
        example = options[0] if options else ""
        fields.append(
            {
                "name": column,
                "label": humanize_label(column),
                "kind": "string",
                "required": True,
                "options": options,
                "example": example,
            }
        )

    return {
        "targetColumn": target_column,
        "problemType": problem_type,
        "fields": fields,
    }


def build_critic_report(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    leaderboard: list[dict[str, Any]],
) -> dict[str, list[str]]:
    warnings: list[str] = []
    failure_modes: list[str] = []
    next_experiments: list[str] = []
    limitations: list[str] = []

    if len(df) < 500:
        warnings.append("Dataset is small, so leaderboard order may be sensitive to the train-test split.")
        limitations.append("Single-split validation on a small dataset is not enough for production confidence.")

    missing_ratios = (df.isna().sum() / max(len(df), 1)).sort_values(ascending=False)
    high_missing = missing_ratios[missing_ratios > 0.15]
    if not high_missing.empty:
        columns = ", ".join(high_missing.index.tolist())
        warnings.append(f"High missing-value columns detected: {columns}.")
        next_experiments.append("Try targeted feature cleaning or domain-aware imputers for sparse columns.")

    best_model = leaderboard[0]
    if best_model.get("trainScore") is not None and best_model.get("testScore") is not None:
        gap = float(best_model["trainScore"]) - float(best_model["testScore"])
        if gap > 0.12:
            warnings.append("The top model shows a noticeable train-test gap and may be overfitting.")
            next_experiments.append("Tune regularization or reduce ensemble depth to improve generalization.")

    if problem_type == "classification":
        class_ratio = df[target_column].value_counts(normalize=True, dropna=False).max()
        if class_ratio > 0.7:
            warnings.append("The target distribution is imbalanced, so accuracy may overstate real performance.")
            next_experiments.append("Add F1, class weights, or resampling to stress-test the minority class.")
        if float(best_model["score"]) < 0.7:
            warnings.append("Held-out classification quality is modest for a production claim.")
    else:
        if float(best_model["score"]) < 0.5:
            warnings.append("Regression fit is weak, so predictions may have limited business value.")
        next_experiments.append("Run cross-validation to confirm the regression winner is not split-specific.")

    failure_modes.append("Distribution shift can reduce performance when new data differs from the training slice.")
    failure_modes.append("Important causal drivers may still be missing from the available columns.")

    limitations.append("This MVP does not include fairness, calibration, or drift monitoring.")
    limitations.append("Model search is intentionally limited to lightweight families for hackathon reliability.")

    if not next_experiments:
        next_experiments.append("Expand feature engineering and compare the winner under cross-validation.")

    return {
        "warnings": warnings,
        "failureModes": failure_modes,
        "nextExperiments": next_experiments,
        "limitations": limitations,
    }


def build_visualizations(
    df: pd.DataFrame,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    problem_type: str,
    best_pipeline: Pipeline,
    leaderboard: list[dict[str, Any]],
    positive_label: Any | None,
    categorical_columns: list[str],
) -> list[dict[str, Any]]:
    visualizations: list[dict[str, Any]] = [
        {
            "id": "experiment-graph",
            "stageId": "evaluation",
            "type": "experiment_graph",
            "title": "ML-Labs experiment graph",
            "data": {
                "nodes": [
                    "Data Profile",
                    "Schema Validation",
                    "Target Analysis",
                    "Preprocessing",
                    "Baseline",
                    "Linear Model",
                    "Tree Model",
                    "Best Model",
                    "Critic",
                    "Report",
                ],
                "edges": [
                    ["Data Profile", "Schema Validation"],
                    ["Schema Validation", "Target Analysis"],
                    ["Target Analysis", "Preprocessing"],
                    ["Preprocessing", "Baseline"],
                    ["Baseline", "Linear Model"],
                    ["Linear Model", "Tree Model"],
                    ["Tree Model", "Best Model"],
                    ["Best Model", "Critic"],
                    ["Critic", "Report"],
                ],
            },
        },
        build_missingness_visualization(df),
        build_correlation_visualization(X_full),
        build_model_comparison_visualization(leaderboard),
    ]

    if problem_type == "classification":
        visualizations.append(build_class_balance_visualization(y_full))
    else:
        visualizations.append(build_actual_vs_predicted_visualization(best_pipeline, X_eval, y_eval))
        visualizations.append(build_residual_visualization(best_pipeline, X_eval, y_eval))

    if problem_type == "classification" and positive_label is not None:
        probabilistic_visualizations = build_classification_curve_visualizations(
            best_pipeline,
            X_eval,
            y_eval,
            positive_label,
        )
        visualizations.extend(probabilistic_visualizations)
        visualizations.append(build_confusion_matrix_visualization(best_pipeline, X_eval, y_eval))
    elif problem_type == "classification":
        visualizations.append(build_confusion_matrix_visualization(best_pipeline, X_eval, y_eval))

    feature_importance = extract_feature_importance(best_pipeline, categorical_columns)
    if feature_importance:
        visualizations.append(
            {
                "id": "feature-importance",
                "stageId": "evaluation",
                "type": "feature_importance",
                "title": "Winning model feature importance",
                "data": feature_importance[:12],
            }
        )

    return [visualization for visualization in visualizations if visualization is not None]


def build_missingness_visualization(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "id": "missingness-summary",
        "stageId": "profiling",
        "type": "missingness_summary",
        "title": "Missing value summary",
        "data": [
            {
                "column": column,
                "missingCount": int(count),
                "missingRatio": round(float(count / max(len(df), 1)), 4),
            }
            for column, count in df.isna().sum().to_dict().items()
        ],
    }


def build_class_balance_visualization(y: pd.Series) -> dict[str, Any]:
    counts = y.astype(str).value_counts(dropna=False)
    return {
        "id": "class-balance",
        "stageId": "profiling",
        "type": "class_balance",
        "title": "Target class balance",
        "data": [
            {
                "label": label,
                "count": int(count),
                "ratio": round(float(count / max(len(y), 1)), 4),
            }
            for label, count in counts.items()
        ],
    }


def build_correlation_visualization(X: pd.DataFrame) -> dict[str, Any]:
    numeric = X.select_dtypes(include=[np.number])
    columns = numeric.columns.tolist()
    if not columns:
        return {
            "id": "correlation-heatmap",
            "stageId": "profiling",
            "type": "correlation_heatmap",
            "title": "Numeric feature correlation heatmap",
            "data": {"columns": [], "matrix": []},
        }

    correlation_matrix = numeric.corr().fillna(0).round(3)
    return {
        "id": "correlation-heatmap",
        "stageId": "profiling",
        "type": "correlation_heatmap",
        "title": "Numeric feature correlation heatmap",
        "data": {
            "columns": columns,
            "matrix": correlation_matrix.values.tolist(),
        },
    }


def build_model_comparison_visualization(leaderboard: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "model-comparison",
        "stageId": "evaluation",
        "type": "model_comparison",
        "title": "Model comparison summary",
        "data": leaderboard,
    }


def build_actual_vs_predicted_visualization(
    best_pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, Any]:
    predictions = best_pipeline.predict(X)
    rows = []
    for actual, predicted in list(zip(y.tolist(), predictions.tolist()))[:60]:
        rows.append(
            {
                "actual": round(float(actual), 3),
                "predicted": round(float(predicted), 3),
            }
        )

    return {
        "id": "actual-vs-predicted",
        "stageId": "evaluation",
        "type": "actual_vs_predicted",
        "title": "Actual vs predicted comparison",
        "data": rows,
    }


def build_residual_visualization(
    best_pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, Any]:
    predictions = best_pipeline.predict(X)
    residual_rows = []
    for actual, predicted in list(zip(y.tolist(), predictions.tolist()))[:60]:
        residual_rows.append(
            {
                "actual": round(float(actual), 3),
                "predicted": round(float(predicted), 3),
                "residual": round(float(actual - predicted), 3),
            }
        )

    return {
        "id": "residual-plot",
        "stageId": "evaluation",
        "type": "residual_plot",
        "title": "Residual behavior on scored data",
        "data": residual_rows,
    }


def build_classification_curve_visualizations(
    best_pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    positive_label: Any,
) -> list[dict[str, Any]]:
    if not hasattr(best_pipeline, "predict_proba"):
        return []

    probabilities = best_pipeline.predict_proba(X)
    classes = getattr(best_pipeline.named_steps["model"], "classes_", None)
    if classes is None or len(classes) != 2:
        return []

    positive_index = list(classes).index(positive_label)
    positive_scores = probabilities[:, positive_index]
    y_binary = (y == positive_label).astype(int)

    fpr, tpr, roc_thresholds = roc_curve(y_binary, positive_scores)
    precision, recall, pr_thresholds = precision_recall_curve(y_binary, positive_scores)

    roc_auc = roc_auc_score(y_binary, positive_scores)

    return [
        {
            "id": "roc-curve",
            "stageId": "evaluation",
            "type": "roc_curve",
            "title": f"ROC curve (AUC={roc_auc:.3f})",
            "data": {
                "fpr": [round(float(value), 4) for value in fpr.tolist()],
                "tpr": [round(float(value), 4) for value in tpr.tolist()],
                "thresholds": [round(float(value), 4) for value in roc_thresholds.tolist()],
            },
        },
        {
            "id": "pr-curve",
            "stageId": "evaluation",
            "type": "pr_curve",
            "title": "Precision-recall curve",
            "data": {
                "precision": [round(float(value), 4) for value in precision.tolist()],
                "recall": [round(float(value), 4) for value in recall.tolist()],
                "thresholds": [round(float(value), 4) for value in pr_thresholds.tolist()],
            },
        },
    ]


def build_confusion_matrix_visualization(
    best_pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, Any]:
    predictions = best_pipeline.predict(X)
    labels = sorted(pd.Series(y).dropna().astype(str).unique().tolist())
    matrix = confusion_matrix(y.astype(str), pd.Series(predictions).astype(str), labels=labels)

    return {
        "id": "confusion-matrix",
        "stageId": "evaluation",
        "type": "confusion_matrix",
        "title": "Confusion matrix on scored data",
        "data": {"labels": labels, "matrix": matrix.tolist()},
    }


def extract_feature_importance(
    best_pipeline: Pipeline,
    categorical_columns: list[str],
) -> list[dict[str, Any]]:
    model = best_pipeline.named_steps["model"]
    preprocessor = best_pipeline.named_steps["preprocessor"]
    if not hasattr(preprocessor, "get_feature_names_out"):
        return []

    feature_names = preprocessor.get_feature_names_out().tolist()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if isinstance(coef, np.ndarray) and coef.ndim > 1:
            importances = np.mean(np.abs(coef), axis=0)
        else:
            importances = np.abs(coef)
    else:
        return []

    rows = []
    for feature, importance in zip(feature_names, importances):
        parsed = parse_feature_name(feature, categorical_columns)
        rows.append(
            {
                "feature": parsed["display"],
                "sourceColumn": parsed["source"],
                "importance": round(float(importance), 4),
            }
        )

    rows.sort(key=lambda row: row["importance"], reverse=True)
    return rows


def parse_feature_name(raw_feature_name: str, categorical_columns: list[str]) -> dict[str, str]:
    if raw_feature_name.startswith("num__"):
        feature_name = raw_feature_name.replace("num__", "", 1)
        return {"display": feature_name, "source": feature_name}

    if raw_feature_name.startswith("cat__"):
        remainder = raw_feature_name.replace("cat__", "", 1)
        for column in sorted(categorical_columns, key=len, reverse=True):
            prefix = f"{column}_"
            if remainder.startswith(prefix):
                option = remainder.replace(prefix, "", 1)
                return {
                    "display": f"{column}={option}",
                    "source": column,
                }

        return {"display": remainder.replace("_", " "), "source": remainder.split("_")[0]}

    return {"display": raw_feature_name, "source": raw_feature_name}


def resolve_positive_label(y: pd.Series) -> Any | None:
    labels = pd.Series(y).dropna().unique().tolist()
    if len(labels) != 2:
        return None

    preferred_labels = ["yes", "true", "1", "churn", "positive"]
    lowered = {str(label).lower(): label for label in labels}
    for preferred in preferred_labels:
        if preferred in lowered:
            return lowered[preferred]

    return sorted(labels, key=lambda item: str(item))[-1]


def train_mode(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.csv)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' was not found in the uploaded CSV.")

    df = df.dropna(subset=[args.target]).copy()
    if df.empty:
        raise ValueError("The target column contains only missing values after cleanup.")

    target_series = df[args.target]
    problem_type = detect_problem_type(target_series)

    X = df.drop(columns=[args.target])
    if X.shape[1] == 0:
        raise ValueError("The dataset must contain at least one feature column besides the target.")

    y = df[args.target]
    preprocessor, numeric_columns, categorical_columns = make_preprocessor(X)
    prediction_input_schema = build_prediction_input_schema(X, problem_type, args.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target_split(y, problem_type),
    )

    leaderboard: list[dict[str, Any]] = []
    model_failures: list[str] = []
    best_pipeline: Pipeline | None = None
    best_score = -float("inf")
    positive_label = resolve_positive_label(y_train if problem_type == "classification" else y)

    for model_name, family, estimator in model_specs(problem_type):
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        try:
            pipeline.fit(X_train, y_train)
            train_predictions = pipeline.predict(X_train)
            test_predictions = pipeline.predict(X_test)

            entry = build_leaderboard_entry(
                family=family,
                model_name=model_name,
                pipeline=pipeline,
                positive_label=positive_label,
                problem_type=problem_type,
                X_test=X_test,
                train_predictions=train_predictions,
                test_predictions=test_predictions,
                y_test=y_test,
                y_train=y_train,
            )
            leaderboard.append(entry)

            if entry["score"] > best_score:
                best_score = entry["score"]
                best_pipeline = pipeline
        except Exception as exc:
            model_failures.append(f"{model_name} failed and was skipped: {exc}")

    if not leaderboard or best_pipeline is None:
        raise RuntimeError("All candidate models failed during the experiment sweep.")

    leaderboard.sort(key=lambda item: item["score"], reverse=True)
    best_model_name = str(leaderboard[0]["modelName"])
    feature_importance = extract_feature_importance(best_pipeline, categorical_columns)

    save_runtime_bundle(
        bundle_dir=Path(args.bundle_dir),
        pipeline=best_pipeline,
        metadata={
            "createdAtEpochMs": int(time() * 1000),
            "modelName": best_model_name,
            "predictionInputSchema": prediction_input_schema,
            "problemType": problem_type,
            "targetColumn": args.target,
            "topFeatures": feature_importance[:8],
        },
    )

    result = {
        "datasetProfile": build_dataset_profile(
            df,
            args.target,
            problem_type,
            numeric_columns,
            categorical_columns,
        ),
        "leaderboard": leaderboard,
        "criticReport": build_critic_report(df, args.target, problem_type, leaderboard),
        "visualizations": build_visualizations(
            df,
            X,
            y,
            X_test,
            y_test,
            problem_type,
            best_pipeline,
            leaderboard,
            positive_label,
            categorical_columns,
        ),
        "predictionInputSchema": prediction_input_schema,
        "metadata": {
            "targetMean": round(float(pd.to_numeric(y, errors="coerce").dropna().mean()), 3)
            if problem_type == "regression"
            else None,
            "targetStd": round(float(pd.to_numeric(y, errors="coerce").dropna().std()), 3)
            if problem_type == "regression"
            else None,
            "modelFailures": model_failures,
            "intentPrompt": args.intent,
            "trainingNote": f"Best model selected from {len(leaderboard)} successful candidates.",
        },
    }

    print(json.dumps(sanitize_for_json(result), allow_nan=False))


def build_leaderboard_entry(
    family: str,
    model_name: str,
    pipeline: Pipeline,
    positive_label: Any | None,
    problem_type: str,
    X_test: pd.DataFrame,
    train_predictions: np.ndarray,
    test_predictions: np.ndarray,
    y_test: pd.Series,
    y_train: pd.Series,
) -> dict[str, Any]:
    if problem_type == "regression":
        train_score = r2_score(y_train, train_predictions)
        test_score = r2_score(y_test, test_predictions)
        rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        note = f"RMSE={rmse:.2f}"
        metric_name = "R2"
    else:
        train_score = accuracy_score(y_train, train_predictions)
        test_score = accuracy_score(y_test, test_predictions)
        metric_name = "Accuracy"
        note_parts = ["Primary metric uses held-out accuracy."]

        if positive_label is not None and hasattr(pipeline, "predict_proba"):
            classes = getattr(pipeline.named_steps["model"], "classes_", None)
            if classes is not None and len(classes) == 2:
                positive_index = list(classes).index(positive_label)
                probabilities = pipeline.predict_proba(X_test)[:, positive_index]
                y_binary = (y_test == positive_label).astype(int)
                roc_auc = roc_auc_score(y_binary, probabilities)
                note_parts.append(f"ROC-AUC={roc_auc:.3f}")
        note = " ".join(note_parts)

    entry = {
        "modelName": model_name,
        "family": family,
        "metricName": metric_name,
        "score": round(float(test_score), 3),
        "trainScore": round(float(train_score), 3),
        "testScore": round(float(test_score), 3),
        "notes": note,
    }

    return entry


def save_runtime_bundle(bundle_dir: Path, pipeline: Pipeline, metadata: dict[str, Any]) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline}, bundle_dir / MODEL_BUNDLE_FILE)
    (bundle_dir / RUNTIME_METADATA_FILE).write_text(
        json.dumps(sanitize_for_json(metadata), allow_nan=False)
    )


def predict_mode(args: argparse.Namespace) -> None:
    bundle_dir = Path(args.bundle_dir)
    bundle = joblib.load(bundle_dir / MODEL_BUNDLE_FILE)
    metadata = json.loads((bundle_dir / RUNTIME_METADATA_FILE).read_text())
    payload = json.loads(args.input_json)

    if not isinstance(payload, dict):
        raise ValueError("Prediction input must be a JSON object.")

    row = coerce_prediction_input(payload, metadata["predictionInputSchema"])
    input_frame = pd.DataFrame([row], columns=[field["name"] for field in metadata["predictionInputSchema"]["fields"]])
    pipeline: Pipeline = bundle["pipeline"]

    prediction = pipeline.predict(input_frame)[0]
    response: dict[str, Any] = {
        "runId": args.run_id,
        "problemType": metadata["problemType"],
        "prediction": to_jsonable(prediction),
        "explanation": build_prediction_explanation(
            metadata["problemType"],
            metadata.get("modelName", "Saved model"),
            prediction,
            None,
        ),
        "topFactors": build_prediction_factors(metadata.get("topFeatures", []), row),
    }

    if metadata["problemType"] == "classification" and hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(input_frame)
        classes = getattr(pipeline.named_steps["model"], "classes_", None)
        if classes is not None:
            predicted_index = list(classes).index(prediction)
            probability = float(probabilities[0][predicted_index])
            response["probability"] = round(probability, 4)
            response["explanation"] = build_prediction_explanation(
                metadata["problemType"],
                metadata.get("modelName", "Saved model"),
                prediction,
                probability,
            )

    if metadata["problemType"] == "regression" and str(metadata.get("targetColumn", "")).lower() == "charges":
        response["unit"] = "USD / year"

    print(json.dumps(sanitize_for_json(response), allow_nan=False))


def coerce_prediction_input(payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for field in schema["fields"]:
        name = field["name"]
        kind = field["kind"]

        if name not in payload:
            raise ValueError(f"Prediction input is missing required field '{name}'.")

        value = payload[name]
        if kind == "number":
            row[name] = float(value)
        elif kind == "boolean":
            if isinstance(value, bool):
                row[name] = value
            elif isinstance(value, str) and value.lower() in {"true", "false"}:
                row[name] = value.lower() == "true"
            else:
                raise ValueError(f"Field '{name}' must be boolean.")
        else:
            row[name] = str(value)
            options = field.get("options") or []
            if options and row[name] not in options:
                raise ValueError(f"Field '{name}' must be one of: {', '.join(options)}.")

    return row


def build_prediction_explanation(
    problem_type: str,
    model_name: str,
    prediction: Any,
    probability: float | None,
) -> str:
    if problem_type == "classification":
        if probability is None:
            return f"The saved {model_name} predicted class '{prediction}' for this input row."
        return f"The saved {model_name} assigned {probability:.1%} confidence to class '{prediction}'."

    return f"The saved {model_name} estimated {float(prediction):.2f} for the provided feature set."


def build_prediction_factors(top_features: list[dict[str, Any]], row: dict[str, Any]) -> list[str]:
    factors: list[str] = []
    for feature_info in top_features[:3]:
        source = feature_info.get("sourceColumn") or feature_info.get("feature") or "feature"
        feature_label = feature_info.get("feature") or source
        if source in row:
            factors.append(f"{feature_label} was influential; provided value = {row[source]}.")
        else:
            factors.append(f"{feature_label} is one of the strongest learned drivers in the saved model.")
    return factors


def humanize_label(name: str) -> str:
    return name.replace("_", " ").strip().title()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return round(float(value), 4)
    return value


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_for_json(inner_value) for key, inner_value in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric_value = float(value)
        if not np.isfinite(numeric_value):
            return None
        return numeric_value
    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    return value


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        train_mode(args)
        return

    if args.mode == "predict":
        predict_mode(args)
        return

    raise RuntimeError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
