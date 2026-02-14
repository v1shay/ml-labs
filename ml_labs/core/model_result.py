# ml_labs/core/model_result.py

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class ModelResult:
    model_name: str
    modality: str
    problem_type: str
    metrics: Dict[str, float]
    artifact_path: Optional[str]
    metadata: Dict[str, Any]
