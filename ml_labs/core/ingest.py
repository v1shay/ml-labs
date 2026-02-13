from __future__ import annotations

import uuid
from pathlib import Path
from typing import List

import pandas as pd

from ml_labs.config import AppConfig
from ml_labs.core.types import Dataset, DatasetModality, utcnow
from ml_labs.core.modality_utils import detect_modality, is_image_file, is_audio_file


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
        modality=DatasetModality.TABULAR,
        dataframe=df,
        file_paths=None,
    )


def ingest_image_folder(dataset_path: str, *, config: AppConfig | None = None) -> Dataset:
    """Load an image dataset from a folder.

    Args:
        dataset_path: Path to a folder containing images.
        config: Optional AppConfig for path normalization.

    Returns:
        Dataset: Structured dataset wrapper containing file paths.

    Raises:
        IngestionError: If the folder doesn't exist or contains no images.
    """
    cfg = config or AppConfig()
    path = cfg.normalize_path(dataset_path)

    if not path.exists():
        raise IngestionError(f"Dataset not found: {path}")

    if not path.is_dir():
        raise IngestionError(f"Dataset path is not a directory: {path}")

    # Collect all image files recursively
    image_files: List[str] = []
    for file_path in path.rglob("*"):
        if file_path.is_file() and is_image_file(file_path):
            image_files.append(str(file_path))

    if not image_files:
        raise IngestionError(f"No image files found in: {path}")

    return Dataset(
        dataset_id=str(uuid.uuid4()),
        path=str(path),
        loaded_at=utcnow(),
        modality=DatasetModality.IMAGE,
        dataframe=None,
        file_paths=sorted(image_files),
    )


def ingest_audio_folder(dataset_path: str, *, config: AppConfig | None = None) -> Dataset:
    """Load an audio dataset from a folder.

    Args:
        dataset_path: Path to a folder containing audio files.
        config: Optional AppConfig for path normalization.

    Returns:
        Dataset: Structured dataset wrapper containing file paths.

    Raises:
        IngestionError: If the folder doesn't exist or contains no audio files.
    """
    cfg = config or AppConfig()
    path = cfg.normalize_path(dataset_path)

    if not path.exists():
        raise IngestionError(f"Dataset not found: {path}")

    if not path.is_dir():
        raise IngestionError(f"Dataset path is not a directory: {path}")

    # Collect all audio files recursively
    audio_files: List[str] = []
    for file_path in path.rglob("*"):
        if file_path.is_file() and is_audio_file(file_path):
            audio_files.append(str(file_path))

    if not audio_files:
        raise IngestionError(f"No audio files found in: {path}")

    return Dataset(
        dataset_id=str(uuid.uuid4()),
        path=str(path),
        loaded_at=utcnow(),
        modality=DatasetModality.AUDIO,
        dataframe=None,
        file_paths=sorted(audio_files),
    )


# ------------------------------------------------------------------
# PRODUCTION WRAPPER
# ------------------------------------------------------------------

def load_dataset(dataset_path: str, *, config: AppConfig | None = None) -> Dataset:
    """
    Public ingestion entrypoint used by orchestrator.

    Auto-detects modality and routes to appropriate ingestion function:
    - .csv → TABULAR (ingest_csv)
    - Folder with images → IMAGE (ingest_image_folder)
    - Folder with audio → AUDIO (ingest_audio_folder)

    Maintains backward compatibility with existing CSV functionality.
    """

    cfg = config or AppConfig()
    path = cfg.normalize_path(dataset_path)

    if not path.exists():
        raise IngestionError(f"Dataset not found: {path}")

    # Auto-detect modality
    modality = detect_modality(path)

    if modality == DatasetModality.TABULAR:
        return ingest_csv(dataset_path, config=config)
    elif modality == DatasetModality.IMAGE:
        return ingest_image_folder(dataset_path, config=config)
    elif modality == DatasetModality.AUDIO:
        return ingest_audio_folder(dataset_path, config=config)
    else:
        raise IngestionError(f"Unsupported modality: {modality}")
