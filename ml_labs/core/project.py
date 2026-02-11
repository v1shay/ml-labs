from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from ml_labs.core.types import DatasetProfile, ProjectStatus, StrategySpec


@dataclass(frozen=True, slots=True)
class ProjectState:
    project_id: str
    dataset_path: str
    created_at: datetime
    strategy: StrategySpec | None
    profile: DatasetProfile | None
    status: ProjectStatus


def new_project_state(*, project_id: str, dataset_path: str) -> ProjectState:
    return ProjectState(
        project_id=project_id,
        dataset_path=dataset_path,
        created_at=datetime.now(timezone.utc),
        strategy=None,
        profile=None,
        status=ProjectStatus.CREATED,
    )
