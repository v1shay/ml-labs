from __future__ import annotations

from pathlib import Path
from typing import List

from ml_labs.core.types import DatasetModality


def is_image_file(path: Path | str) -> bool:
    """Check if a file path is an image file."""
    path_obj = Path(path) if isinstance(path, str) else path
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    return path_obj.suffix.lower() in image_extensions


def is_audio_file(path: Path | str) -> bool:
    """Check if a file path is an audio file."""
    path_obj = Path(path) if isinstance(path, str) else path
    audio_extensions = {".wav", ".mp3", ".flac", ".aac", ".ogg", ".m4a", ".wma"}
    return path_obj.suffix.lower() in audio_extensions


def detect_modality(path: str | Path) -> DatasetModality:
    """
    Auto-detect dataset modality from path.
    
    Rules:
    - .csv → TABULAR
    - Folder with images (.jpg/.png/.jpeg) → IMAGE
    - Folder with audio (.wav/.mp3/.flac) → AUDIO
    - Default: TABULAR (for backward compatibility)
    """
    path_obj = Path(path) if isinstance(path, str) else path
    
    # CSV file → TABULAR
    if path_obj.is_file() and path_obj.suffix.lower() == ".csv":
        return DatasetModality.TABULAR
    
    # Directory → check contents
    if path_obj.is_dir():
        # Collect all files
        all_files = list(path_obj.rglob("*"))
        image_files = [f for f in all_files if f.is_file() and is_image_file(f)]
        audio_files = [f for f in all_files if f.is_file() and is_audio_file(f)]
        
        # If directory has images, it's IMAGE
        if image_files:
            return DatasetModality.IMAGE
        
        # If directory has audio, it's AUDIO
        if audio_files:
            return DatasetModality.AUDIO
        
        # Empty or unrecognized directory defaults to TABULAR
        return DatasetModality.TABULAR
    
    # Single file that's not CSV → try to infer from extension
    if path_obj.is_file():
        if is_image_file(path_obj):
            return DatasetModality.IMAGE
        if is_audio_file(path_obj):
            return DatasetModality.AUDIO
    
    # Default to TABULAR for backward compatibility
    return DatasetModality.TABULAR
