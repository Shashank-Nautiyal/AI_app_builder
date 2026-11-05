# CodeMind

CodeMind is an AI project generator: describe an application, and a LangGraph
pipeline plans it, writes the files, validates the result, and serves a live
preview. Projects can be edited, re-run, downloaded, and iterated on from the
web UI.

## Features

- Multi-agent generation pipeline (clarifier, planner, architect, coder, integrator, verifier)
- Optional GitHub repo discovery and adaptation
- Templates for React/Vite, Next.js, FastAPI, Streamlit, Tkinter and Electron
- Deterministic verification plus runtime health checks
- Live file stream, terminal, project chat, snapshots and rollback
- Automatic `.env` provisioning for generated projects

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Node.js 18+ (only needed for JavaScript/TypeScript projects)

## Setup

```powershell
uv sync
Copy-Item .env.example .env
# edit .env and add the key for the provider you want to use
```

Set `LLM_PROVIDER` to `openrouter`, `groq`, `cerebras`, `mistral`, `nvidia`,
`deepinfra`, `felo` or `fireworks`, then fill in the matching API key.

## Run

```powershell
uv run python run.py
# or
uv run python -m codemind
```

The UI is served at <http://127.0.0.1:8000>.

Set `UVICORN_RELOAD=1` for auto-reload during development. `HOST` and `PORT`
are also configurable.

## Tests

```powershell
uv run --with pytest python -m pytest -q
```

## Project layout

```text
codemind/
|-- api/                 FastAPI app and UI event schemas
|   |-- server.py
|   `-- ui_events.py
|-- core/                Shared models and low-level helpers
|   |-- database.py
|   |-- runtime_detection.py
|   |-- schemas.py
|   |-- structured_output.py
|   `-- template_registry.py
|-- pipeline/            LangGraph agents and prompts
|   |-- agent_tools.py
|   |-- graph.py
|   `-- prompt.py
|-- services/            Reusable application services
|   |-- env_sync.py
|   |-- github_scout.py
|   |-- pipeline_validation.py
|   |-- project_context.py
|   |-- project_ops.py
|   `-- project_runner.py
|-- paths.py             Central filesystem paths
|-- runner.py            Uvicorn launcher
`-- __main__.py          Enables `python -m codemind`

docs/                    Workflow and architecture notes
frontend/                React control panel source
scripts/                 Maintenance scripts
static/                  Prebuilt UI served by FastAPI
templates/               Starter project blueprints
tests/                   Unit and contract tests
run.py                   Main development entry point
```

Runtime output is written to `generated_project/`; databases, logs, virtual
environments and `.env` files are ignored by git.
