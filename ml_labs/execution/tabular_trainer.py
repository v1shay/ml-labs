# ml_labs/execution/tabular_trainer.py

import os
import joblib
import numpy as np
import pandas as pd

from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

from ml_labs.core.model_result import ModelResult


ARTIFACT_DIR = "artifacts"


def train_tabular_model(
    df: pd.DataFrame,
    target: str,
    problem_type: str,
    metric: str,
) -> ModelResult:

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if problem_type == "classification":
        model = RandomForestClassifier(random_state=42)
    elif problem_type == "regression":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(random_state=42)
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")

    pipeline = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("model", model),
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    if problem_type == "classification":
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "f1": f1_score(y_test, predictions, average="weighted"),
        }
    else:
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        metrics = {"rmse": rmse}

    artifact_path = os.path.join(ARTIFACT_DIR, "tabular_model.joblib")
    joblib.dump(pipeline, artifact_path)

    return ModelResult(
        model_name=model.__class__.__name__,
        modality="tabular",
        problem_type=problem_type,
        metrics=metrics,
        artifact_path=artifact_path,
        metadata={"rows": len(df), "features": X.shape[1]},
    )
