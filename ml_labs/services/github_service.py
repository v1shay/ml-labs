from __future__ import annotations

from pathlib import Path

from ml_labs.core.project import ProjectState


class GitHubService:
    """Abstraction layer for future GitHub-backed project workflows.

    Current scope:
    - scaffold a local project directory structure that can later be synced
      to a GitHub repository.

    Out of scope (intentionally):
    - OAuth
    - pushing commits
    - interacting with GitHub APIs
    """

    def __init__(self, *, repo_url: str) -> None:
        self._repo_url = repo_url

    def scaffold_local_project(self, *, project: ProjectState, workspace_dir: str | Path) -> Path:
        root = Path(workspace_dir).expanduser().resolve()
        project_dir = root / project.project_id

        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "data").mkdir(exist_ok=True)
        (project_dir / "artifacts").mkdir(exist_ok=True)
        (project_dir / "reports").mkdir(exist_ok=True)
        (project_dir / "src").mkdir(exist_ok=True)

        self._write_gitignore(project_dir)
        self._write_readme(project_dir, project)

        return project_dir

    def _write_gitignore(self, project_dir: Path) -> None:
        gitignore = project_dir / ".gitignore"
        if gitignore.exists():
            return
        gitignore.write_text(
            """__pycache__/\n*.pyc\n.venv/\n.env\nartifacts/\nreports/\n""",
            encoding="utf-8",
        )

    def _write_readme(self, project_dir: Path, project: ProjectState) -> None:
        readme = project_dir / "README.md"
        if readme.exists():
            return

        readme.write_text(
            """# ML Labs Project\n\n"""
            f"This directory was scaffolded by Vishay's ML Labs.\n\n"
            f"Upstream repository (future sync target): {self._repo_url}\n\n"
            "## Project Metadata\n\n"
            f"- Project ID: {project.project_id}\n"
            f"- Dataset path: {project.dataset_path}\n",
            encoding="utf-8",
        )
