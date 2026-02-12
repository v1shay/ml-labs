from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd

from ml_labs.config import AppConfig
from ml_labs.core.types import Dataset, utcnow


class IngestionError(RuntimeError):
    pass


def ingest_csv(dataset_path: str, *, config: AppConfig | None = None) -> Dataset:
    """Load a CSV dataset from disk.

    Args:
        dataset_path: Path to a CSV file.
        config: Optional AppConfig for path normalization.

    Returns:
        Dataset: Structured dataset wrapper containing the DataFrame.

    Raises:
        IngestionError: If the dataset doesn't exist or can't be parsed.
    """

    cfg = config or AppConfig()
    path = cfg.normalize_path(dataset_path)

    if not path.exists():
        raise IngestionError(f"Dataset not found: {path}")

    if not path.is_file():
        raise IngestionError(f"Dataset path is not a file: {path}")

    if path.suffix.lower() != ".csv":
        raise IngestionError(f"Unsupported dataset type (expected .csv): {path}")

    try:
        df = pd.read_csv(path)
    except Exception as e:  # pragma: no cover
        raise IngestionError(f"Failed to read CSV: {path}. Error: {e}") from e

    if df.shape[1] == 0:
        raise IngestionError(f"CSV has no columns: {path}")

    return Dataset(
        dataset_id=str(uuid.uuid4()),
        path=str(path),
        loaded_at=utcnow(),
        dataframe=df,
    )


# ------------------------------------------------------------------
# PRODUCTION WRAPPER
# ------------------------------------------------------------------

def load_dataset(dataset_path: str, *, config: AppConfig | None = None) -> Dataset:
    """
    Public ingestion entrypoint used by orchestrator.

    This wrapper exists to:
    - Keep ingest_csv as the CSV-specific loader
    - Provide a future extension point for other formats (Parquet, JSON, images)
    - Maintain backward compatibility with orchestrator imports
    """

    path = Path(dataset_path)

    if path.suffix.lower() == ".csv":
        return ingest_csv(dataset_path, config=config)

    raise IngestionError(f"Unsupported dataset type: {path.suffix}")
