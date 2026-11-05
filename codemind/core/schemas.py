from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class ClarifiedRequest(BaseModel):
    """Output of the CLARIFIER agent to resolve upfront ambiguity."""
    refined_prompt: str = Field(description="The unambiguous, fully-resolved project request.")

    model_config = ConfigDict(
        json_schema_extra={"example": {"refined_prompt": "Build a React calculator app using SQLite."}}
    )

class File(BaseModel):
    path: str = Field(
        description="Relative file path from project root, e.g. 'src/app.py'",
    )
    purpose: str = Field(
        description="One-sentence description of what this file does",
    )

    @field_validator("path")
    @classmethod
    def must_be_relative(cls, v: str) -> str:
        if v.startswith("/"):
            raise ValueError("File path must be relative, not absolute.")
        return v.strip()

    model_config = ConfigDict(
        json_schema_extra={"example": {"path": "src/app.py", "purpose": "Main application entry point"}}
    )


class Plan(BaseModel):
    """High-level project plan — output of the PLANNER agent."""

    name: str = Field(description="Short application name based on the user's request", min_length=2, max_length=80)
    description: str = Field(description="One-sentence description of the application", min_length=10)
    techstack: str = Field(description="Comma-separated technologies, e.g. 'Python, Flask, HTML, CSS'")
    project_type: str = Field(description="The primary project architecture type, e.g. 'react-vite', 'nextjs-app', 'fastapi', 'fastapi-sqlite', 'python-tkinter-sqlite', 'streamlit-app', 'electron-react-app'.")
    features: list[str] = Field(description="Concrete user-facing features", min_length=1)
    files: list[File] = Field(description="All files to create, in dependency order", min_length=1)

    @field_validator("features")
    @classmethod
    def non_empty_features(cls, v: list[str]) -> list[str]:
        cleaned = [f.strip() for f in v if f.strip()]
        if not cleaned:
            raise ValueError("features must contain at least one non-empty string.")
        return cleaned

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "My App",
                "description": "A web application as described by the user request",
                "techstack": "React, Vite, FastAPI, SQLite",
                "project_type": "fullstack",
                "features": ["Feature A from user request", "Feature B from user request"],
                "files": [
                    {"path": "src/App.jsx", "purpose": "Main React component"},
                    {"path": "src/main.jsx", "purpose": "React entry point"},
                    {"path": "backend/main.py", "purpose": "FastAPI server"},
                ],
            }
        }
    )




class ImplementationTask(BaseModel):
    """One atomic coding task — one file."""

    filepath: str = Field(description="Relative path of the file to create or modify")
    action: str = Field(
        default="create",
        description="'create' for new files, 'rewrite' to completely replace an existing template file (e.g., to connect components)."
    )
    task_description: str = Field(
        description=(
            "Detailed, self-contained description of what to implement. "
            "Include function names, class names, imports, and integration points."
        ),
        min_length=20,
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Relative paths of files that must be implemented before this one",
    )

    @field_validator("filepath")
    @classmethod
    def must_be_relative(cls, v: str) -> str:
        if v.startswith("/"):
            raise ValueError("filepath must be relative.")
        return v.strip()

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "filepath": "src/App.jsx",
                "action": "rewrite",
                "task_description": (
                    "Rewrite App.jsx to import and render all components. "
                    "Set up routing with react-router-dom for all pages."
                ),
                "dependencies": ["src/components/Header.jsx", "src/components/Home.jsx"],
            }
        }
    )


class TaskPlan(BaseModel):
    """Ordered list of implementation tasks — output of the ARCHITECT agent."""

    implementation_steps: list[ImplementationTask] = Field(
        description="Tasks ordered so that dependencies are implemented first.",
        min_length=1,
    )

    @model_validator(mode="after")
    def merge_duplicate_filepaths(self) -> "TaskPlan":
        unique_tasks = {}
        for t in self.implementation_steps:
            if t.filepath in unique_tasks:
                unique_tasks[t.filepath].task_description += f"\n\nAdditional Task:\n{t.task_description}"
                # Extend dependencies but keep unique
                unique_tasks[t.filepath].dependencies = list(set(unique_tasks[t.filepath].dependencies + t.dependencies))
            else:
                unique_tasks[t.filepath] = t
        self.implementation_steps = list(unique_tasks.values())
        return self

    model_config = ConfigDict(extra="allow")



class RequirementsSpec(BaseModel):
    """
    pip-installable packages needed to run the generated project.

    The requirements_agent writes these to requirements.txt so the executor
    can call `pip install -r requirements.txt` before running the project.
    Prefer flexible dependency specs unless an exact pin is truly required.
    Stdlib-only projects should return an empty list.
    """

    package_manager: str = Field(
        default="pip",
        description="The package manager to use: 'pip' for Python, 'npm' for Node.js."
    )
    packages: list[str] = Field(
        default_factory=list,
        description=(
            "dependencies to install. For pip: e.g. ['flask', 'requests>=2.31']. "
            "For npm: e.g. ['express@^4.21.2', 'react@^18.3.1']. "
            "Return an empty list for projects that need no third-party packages."
        ),
    )

    @field_validator("packages")
    @classmethod
    def clean_packages(cls, v: list[str]) -> list[str]:
        """Strip whitespace and drop blank entries."""
        return [p.strip() for p in v if p.strip()]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "package_manager": "pip",
                "packages": ["flask", "requests>=2.31.0"],
            }
        }
    )


class RetrievedContext(BaseModel):
    task_filepath:     str
    relevant_snippets: list[str] = Field(default_factory=list)
    related_files:     list[str] = Field(default_factory=list)
    completed_files:   list[str] = Field(default_factory=list)
    pending_files:     list[str] = Field(default_factory=list)
    progress:          str       = Field(default="")




class CoderState(BaseModel):
    """Internal tracking state for the iterative CODER loop (not LLM-parsed)."""

    task_plan: TaskPlan
    current_step_idx: int = Field(0, ge=0)
    current_file_content: Optional[str] = Field(None)
    rag_context: Optional[RetrievedContext] = Field(None)


class MemoryEntry(BaseModel):
    """Compact record created after each file is implemented — LLM-parsed."""

    filepath: str = Field(description="Relative path of the implemented file")
    summary: str = Field(description="1-2 sentence plain-English description", min_length=10)
    exports: list[str] = Field(
        default_factory=list,
        description="All public function / class / variable names defined in this file",
    )
    key_decisions: list[str] = Field(
        default_factory=list,
        description="Important design choices, e.g. 'uses SQLite for portability'",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "filepath": "src/calculator.py",
                "summary": "Pure-Python arithmetic module.",
                "exports": ["add", "subtract", "multiply", "divide"],
                "key_decisions": ["Raises ZeroDivisionError on divide-by-zero"],
            }
        }
    )


class GeneratedFileInfo(BaseModel):
    """Deterministic import/export facts for a generated source file."""

    filepath: str = Field(description="Relative path from the generated project root")
    imports: list[str] = Field(default_factory=list, description="Import specifiers referenced by the file")
    exports: list[str] = Field(default_factory=list, description="Exported functions/classes/variables/components")
    kind: str = Field(default="source", description="Coarse source kind, e.g. javascript, python, or other")



class ExecutionResult(BaseModel):
    """Structured summary of a project run — LLM-parsed from executor output."""

    command: str = Field(description="Exact shell command that was run")
    returncode: int = Field(description="Process exit code (0 = success)", ge=-1)
    stdout: str = Field(default="", description="Captured stdout (≤ 2000 chars)")
    stderr: str = Field(default="", description="Captured stderr (≤ 2000 chars)")
    success: bool = Field(description="True only when returncode == 0 and no traceback in stderr")
    error_type: Optional[str] = Field(
        None,
        description="Python exception class if a traceback was found, else null",
    )

    @model_validator(mode="after")
    def sync_success(self) -> "ExecutionResult":
        if self.returncode != 0:
            self.success = False
        return self

    @field_validator("stdout", "stderr")
    @classmethod
    def trim(cls, v: str) -> str:
        return v[:2000]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "command": "python src/app.py",
                "returncode": 1,
                "stdout": "",
                "stderr": "ModuleNotFoundError: No module named 'flask'",
                "success": False,
                "error_type": "ModuleNotFoundError",
            }
        }
    )


class TestCase(BaseModel):
    """One deterministic verification check result."""

    name: str = Field(description="Stable check identifier")
    description: str = Field(default="", description="Human-readable check description")
    command: str = Field(default="", description="Command or guard that produced the result")
    success: bool = Field(description="Whether this check passed")
    output: str = Field(default="", description="Combined stdout/stderr or guard message")


class TestReport(BaseModel):
    """Summary returned by the deterministic verify node."""

    test_cases: list[TestCase] = Field(default_factory=list)
    summary: str = Field(default="")


class DebugResult(BaseModel):
    """Compatibility record for older checkpoints that still contain debugger output."""

    error_summary: str = Field(default="")
    root_cause: str = Field(default="")
    fix_description: str = Field(default="")
    files_modified: list[str] = Field(default_factory=list)
    resolved: bool = Field(default=False)







class MemoryStore(BaseModel):
    """
    In-memory store of MemoryEntry records, keyed by filepath.

    Serialization note
    ──────────────────
    LangGraph's SqliteSaver checkpointer serialises the entire PipelineState to
    JSON via its own encoder.  Because MemoryStore is a Pydantic BaseModel nested
    inside a TypedDict, we expose two helpers so the checkpointer can round-trip
    it without a registered custom type:

      • to_dict()   — called by the checkpointer when saving a snapshot
      • from_dict() — called when rehydrating a snapshot
    """

    entries: list[MemoryEntry] = Field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a plain dict that JSON-serialises cleanly."""
        return {"entries": [e.model_dump() for e in self.entries]}

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryStore":
        """Rehydrate from the dict produced by to_dict()."""
        if not isinstance(data, dict):
            return cls()
        return cls(entries=[MemoryEntry(**e) for e in data.get("entries", [])])

    def get_context_for(self, filepath: str) -> str:
        lines = [
            f"• {e.filepath}\n"
            f"  {e.summary}\n"
            f"  exports  : {', '.join(e.exports) or 'none'}\n"
            f"  decisions: {'; '.join(e.key_decisions) or 'none'}\n"
            for e in self.entries
            if e.filepath != filepath
        ]
        return "\n".join(lines) if lines else "No prior context."

    def add(self, entry: MemoryEntry) -> None:
        self.entries = [e for e in self.entries if e.filepath != entry.filepath]
        self.entries.append(entry)




def _merge_memory_store(
    left: Union[MemoryStore, dict, None],
    right: Union[MemoryStore, dict, None],
) -> MemoryStore:
    """
    LangGraph reducer for the memory_store field.

    Accepts both MemoryStore instances and raw dicts (the form SqliteSaver
    produces when deserialising a checkpoint snapshot).  The right-hand value
    (from the returning node) always wins.
    """
    def _coerce(v) -> MemoryStore:
        if isinstance(v, MemoryStore):
            return v
        if isinstance(v, dict):
            return MemoryStore.from_dict(v)
        return MemoryStore()

    return _coerce(right) if right is not None else _coerce(left)


def _merge_generated_file_manifest(
    left: Optional[dict[str, Union[GeneratedFileInfo, dict]]],
    right: Optional[dict[str, Union[GeneratedFileInfo, dict]]],
) -> dict[str, GeneratedFileInfo]:
    """LangGraph reducer for generated file manifest updates."""

    def _coerce_map(value) -> dict[str, GeneratedFileInfo]:
        if not isinstance(value, dict):
            return {}
        result: dict[str, GeneratedFileInfo] = {}
        for key, item in value.items():
            try:
                info = item if isinstance(item, GeneratedFileInfo) else GeneratedFileInfo(**item)
                result[str(key)] = info
            except Exception:
                continue
        return result

    merged = _coerce_map(left)
    merged.update(_coerce_map(right))
    return merged



from langgraph.graph.message import MessagesState





class DetectedService(BaseModel):
    """A database / auth / storage / email / payments service found in the downloaded repo."""
    service_type: str = Field(
        description="One of: 'database', 'auth', 'storage', 'email', 'payments'"
    )
    original_provider: str = Field(
        description="The library/service detected, e.g. 'mongodb', 'prisma', 'firebase-auth'"
    )
    files_affected: list[str] = Field(
        default_factory=list,
        description="Relative file paths that contain this provider's code",
    )
    replacement: str = Field(
        description="The service to swap in, e.g. 'supabase', 'neon', 'clerk', 'supabase-storage', 'cloudinary', 'resend', 'stripe'"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "service_type": "database",
                "original_provider": "mongodb",
                "files_affected": ["backend/db.js", "backend/models/User.js"],
                "replacement": "supabase",
            }
        }
    )


class RepoAdaptationPlan(BaseModel):
    """Full plan produced by the GitHub Scout after downloading & analysing a repo."""
    repo_url: str = Field(description="GitHub HTML URL of the downloaded repo")
    repo_name: str = Field(description="owner/repo string, e.g. 'vercel/next.js'")
    detected_services: list[DetectedService] = Field(
        default_factory=list,
        description="All DB / auth / storage / email / payments services found in the repo",
    )
    files_to_delete: list[str] = Field(
        default_factory=list,
        description="Provider-specific config files that should be removed",
    )
    adaptation_summary: str = Field(
        default="",
        description="Human-readable summary of what was found and what will change",
    )
    framework: str = Field(
        default="",
        description="Detected framework: 'nextjs', 'react-vite', 'fastapi', 'express', etc.",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Important dependency and script signals detected in the repository.",
    )
    env_vars: list[str] = Field(
        default_factory=list,
        description="Environment variables referenced by the repository.",
    )
    key_files: list[str] = Field(
        default_factory=list,
        description="Important files discovered in the repository for planner context.",
    )
    external_connections: list[str] = Field(
        default_factory=list,
        description="External services the project uses or should be connected to.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "repo_url": "https://github.com/owner/repo",
                "repo_name": "owner/repo",
                "detected_services": [],
                "files_to_delete": [],
                "adaptation_summary": "No services detected.",
                "framework": "react-vite",
            }
        }
    )


class ToolTranscriptEntry(BaseModel):
    """A record of a tool execution for the 'live tool context' pattern."""
    tool_name: str
    input: dict = Field(default_factory=dict)
    output: str = Field(default="")
    status: str = Field("success", description="success, error, or in_progress")
    timestamp: float = Field(default_factory=lambda: 0.0)

class SupervisorDecision(BaseModel):
    """The structured output of the thinking supervisor."""
    rationale: str = Field(description="Internal reasoning for the next step")
    next_node: str = Field(description="The identifier of the next node to execute")
    instructions: Optional[str] = Field(None, description="Specific instructions for the next agent")

class PipelineState(MessagesState, total=False):
    """
    Top-level state object threaded through every LangGraph node.
    """

    # Input (set once, never changed)
    user_prompt: str
    original_user_prompt: str

    # Stage outputs
    plan:              Optional[Plan]
    task_plan:         Optional[TaskPlan]
    requirements:      Optional[RequirementsSpec]
    coder_state:       Optional[CoderState]

    memory_store:      Annotated[MemoryStore, _merge_memory_store]
    generated_file_manifest: Annotated[dict[str, GeneratedFileInfo], _merge_generated_file_manifest]

    # GitHub Scout — set after download & analysis, consumed by adapter_agent
    repo_adaptation_plan: Optional[RepoAdaptationPlan]
    github_selected_repo: Optional[dict[str, Any]]
    github_search_completed: bool

    # List fields with append-reducer
    rag_contexts:      Annotated[list[RetrievedContext], operator.add]
    execution_results: Annotated[list[ExecutionResult],  operator.add]

    tool_transcript:   Annotated[list[ToolTranscriptEntry], operator.add]
    env_sync_missing_keys: list[str]
    env_sync_written_files: list[str]

    status:             str

    supervisor_turns:   int
    max_supervisor_turns: int
    integrator_attempts: int   # how many consecutive integrator→update_coder cycles without progress
    thinking_rationale: Optional[str] # From supervisor
    rollback_snapshot_path: Optional[str]
    rollback_applied: bool
    rollback_reason: Optional[str]
