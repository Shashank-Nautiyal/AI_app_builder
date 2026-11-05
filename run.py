import os

import uvicorn

RELOAD_EXCLUDES = [
    "generated_project",
    "project_env",
    "templates/nextjs-app/node_modules",
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

RELOAD_ENABLED = os.getenv("UVICORN_RELOAD", "0").strip().lower() in {"1", "true", "yes", "on"}

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=RELOAD_ENABLED,
        reload_dirs=["."] if RELOAD_ENABLED else None,
        reload_excludes=RELOAD_EXCLUDES if RELOAD_ENABLED else None,
    )
