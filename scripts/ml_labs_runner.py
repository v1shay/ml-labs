#!/usr/bin/env python3

import argparse
import json
from typing import Any

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
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal ML-Labs experiment sweep.")
    parser.add_argument("--csv", required=True, help="Path to the CSV file.")
    parser.add_argument("--target", required=True, help="Name of the target column.")
    parser.add_argument("--intent", required=False, help="Narrative prompt for the run.")
    return parser.parse_args()


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
            ("Random Forest Regressor", "Tree Ensemble", RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)),
            ("Gradient Boosting Regressor", "Boosted Trees", GradientBoostingRegressor(random_state=RANDOM_STATE)),
        ]

    return [
        ("Majority Classifier", "Baseline", DummyClassifier(strategy="most_frequent")),
        ("Logistic Regression", "Linear Model", LogisticRegression(max_iter=1000)),
        ("Random Forest Classifier", "Tree Ensemble", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
        ("Gradient Boosting Classifier", "Boosted Trees", GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ]


def score_model(problem_type: str, y_true: pd.Series, y_pred: np.ndarray) -> tuple[str, float, str]:
    if problem_type == "regression":
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return "R2", round(float(r2), 3), f"RMSE={rmse:.2f}"

    accuracy = accuracy_score(y_true, y_pred)
    return "Accuracy", round(float(accuracy), 3), "Primary metric uses held-out accuracy."


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
    gap = 0.0
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
    problem_type: str,
    best_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> list[dict[str, Any]]:
    visualizations: list[dict[str, Any]] = [
        {
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
        }
    ]

    predictions = best_pipeline.predict(X_test)
    model = best_pipeline.named_steps["model"]

    if problem_type == "regression":
        residual_rows = []
        for actual, predicted in list(zip(y_test.tolist(), predictions.tolist()))[:40]:
            residual_rows.append(
                {
                    "actual": round(float(actual), 3),
                    "predicted": round(float(predicted), 3),
                    "residual": round(float(actual - predicted), 3),
                }
            )
        visualizations.append(
            {
                "type": "residual_plot",
                "title": "Residual behavior on held-out data",
                "data": residual_rows,
            }
        )
    else:
        labels = sorted(pd.Series(y_test).dropna().unique().tolist())
        matrix = confusion_matrix(y_test, predictions, labels=labels)
        visualizations.append(
            {
                "type": "confusion_matrix",
                "title": "Confusion matrix on held-out data",
                "data": {"labels": labels, "matrix": matrix.tolist()},
            }
        )

    feature_importance = extract_feature_importance(best_pipeline)
    if feature_importance:
        visualizations.append(
            {
                "type": "feature_importance",
                "title": "Winning model feature importance",
                "data": feature_importance[:12],
            }
        )

    return visualizations


def extract_feature_importance(best_pipeline: Pipeline) -> list[dict[str, Any]]:
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

    rows = [
        {"feature": feature, "importance": round(float(importance), 4)}
        for feature, importance in zip(feature_names, importances)
    ]
    rows.sort(key=lambda row: row["importance"], reverse=True)
    return rows


def main() -> None:
    args = parse_args()
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

            metric_name, test_score, note = score_model(problem_type, y_test, test_predictions)
            _, train_score, _ = score_model(problem_type, y_train, train_predictions)

            entry = {
                "modelName": model_name,
                "family": family,
                "metricName": metric_name,
                "score": test_score,
                "trainScore": train_score,
                "testScore": test_score,
                "notes": note,
            }
            leaderboard.append(entry)

            if test_score > best_score:
                best_score = test_score
                best_pipeline = pipeline
        except Exception as exc:
            model_failures.append(f"{model_name} failed and was skipped: {exc}")

    if not leaderboard or best_pipeline is None:
        raise RuntimeError("All candidate models failed during the experiment sweep.")

    leaderboard.sort(key=lambda item: item["score"], reverse=True)

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
        "visualizations": build_visualizations(problem_type, best_pipeline, X_test, y_test),
        "metadata": {
            "targetMean": round(float(pd.to_numeric(y, errors="coerce").dropna().mean()), 3)
            if problem_type == "regression"
            else None,
            "targetStd": round(float(pd.to_numeric(y, errors="coerce").dropna().std()), 3)
            if problem_type == "regression"
            else None,
            "modelFailures": model_failures,
            "intentPrompt": args.intent,
        },
    }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
