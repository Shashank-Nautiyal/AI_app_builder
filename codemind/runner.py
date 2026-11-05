"""Run the CodeMind API server.

Kept separate from the package entry points so ``run.py`` and
``python -m codemind`` share the exact same configuration.
"""

from __future__ import annotations

import os

import uvicorn

from codemind.paths import PROJECT_ROOT, ensure_runtime_dirs

RELOAD_EXCLUDES = [
    "generated_project",
    "project_env",
    "templates/**/node_modules",
    "node_modules",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ai_studio_index",
    ".project_snapshots",
    "tests/**",
    "*.db",
    "*.db-*",
    "*.sqlite",
    "*.sqlite3",
    "pipeline_checkpoints.db*",
]


def main() -> None:
    ensure_runtime_dirs()
    # Keep relative paths used by subprocesses and third-party libraries
    # anchored to the repository root.
    os.chdir(PROJECT_ROOT)

    reload_enabled = os.getenv("UVICORN_RELOAD", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    uvicorn.run(
        "codemind.api.server:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=reload_enabled,
        reload_dirs=[str(PROJECT_ROOT)] if reload_enabled else None,
        reload_excludes=RELOAD_EXCLUDES if reload_enabled else None,
    )
