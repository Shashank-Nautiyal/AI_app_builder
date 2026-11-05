"""Central project paths.

Every runtime path is resolved from the repository root so the application
behaves the same regardless of the directory it was started from.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
GENERATED_PROJECTS_DIR = PROJECT_ROOT / "generated_project"
DOCS_DIR = PROJECT_ROOT / "docs"

ENV_FILE = PROJECT_ROOT / ".env"
PROJECTS_DB = PROJECT_ROOT / "projects.db"
CHECKPOINT_DB = PROJECT_ROOT / "pipeline_checkpoints.db"


def ensure_runtime_dirs() -> None:
    """Create directories the server expects to exist at startup."""
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
