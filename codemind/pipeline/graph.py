

from __future__ import annotations

import json
import os
import sys
import sqlite3
import re
import tempfile
import uuid
import warnings
warnings.filterwarnings("ignore", message=r".*not known to support tools.*", category=UserWarning)
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any, Callable
from contextlib import contextmanager

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
import langchain_core.globals
from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq
from langchain_fireworks import ChatFireworks
from langgraph.graph import StateGraph, START, END
try:
    from langchain_mistralai import ChatMistralAI
except ModuleNotFoundError:
    ChatMistralAI = None

from langchain_core.messages import HumanMessage

_LANGCHAIN_DEBUG = os.getenv("LANGCHAIN_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
langchain_core.globals.set_verbose(_LANGCHAIN_DEBUG)
langchain_core.globals.set_debug(_LANGCHAIN_DEBUG)
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.runnables import RunnableConfig
from langchain.agents import create_agent
from langchain_cerebras import ChatCerebras
from codemind.pipeline.prompt import (
    clarifier_prompt, planner_existing_repo_prompt, planner_scratch_prompt,
    architect_prompt, requirements_prompt, retriever_prompt,
    memory_prompt, coder_system_prompt, coder_user_prompt, PLACEHOLDER_PATTERNS,
    supervisor_prompt, integrator_system_prompt, integrator_user_prompt,
    github_scout_adapter_system_prompt, github_scout_adapter_user_prompt,
)
from codemind.core.schemas import (
    PipelineState, Plan, TaskPlan, ImplementationTask, RequirementsSpec, CoderState, RetrievedContext,
    ExecutionResult, DebugResult, TestReport, TestCase, MemoryEntry, MemoryStore, ToolTranscriptEntry, SupervisorDecision,
    ClarifiedRequest, RepoAdaptationPlan, DetectedService, GeneratedFileInfo, File,
)
from codemind.services.project_ops import create_project_snapshot, restore_project_snapshot
from codemind.core.structured_output import repair_structured_json
from codemind.services.project_context import build_project_context, ProjectContext
from codemind.core.template_registry import resolve_template, resolve_template_from_plan, is_frontend_template, is_backend_template, get_coder_model_tier, get_all_template_ids
from codemind.services.env_sync import auto_fill_project_env
from codemind.core.runtime_detection import build_run_plan
from codemind.services.pipeline_validation import (
    build_generated_file_manifest,
    env_placeholder_failure_result,
    format_generated_manifest,
    placeholder_failure_result,
    run_runtime_health_check,
    scan_runtime_env_placeholders,
    scan_placeholders,
    validate_project_imports,
)
from codemind.pipeline.agent_tools import (
    write_file, read_file, list_files, list_project_tree, tree_graph,
    get_current_directory, run_cmd, ddg_search, delete_file, replace_file_content, apply_patch_block, validate_imports
)
from codemind.paths import CHECKPOINT_DB, ENV_FILE, GENERATED_PROJECTS_DIR

load_dotenv(ENV_FILE, override=True)

_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "4096"))

def _hf(repo_id: str) -> ChatHuggingFace:
    return ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            repo_id=repo_id, task="text-generation", huggingfacehub_api_token=os.getenv("HF_TOKEN") or os.getenv("HF_token"),
            max_new_tokens=_MAX_OUTPUT_TOKENS, temperature=0.1, timeout=120,
        )
    )
try:
    from langchain_openrouter import ChatOpenRouter
except ModuleNotFoundError:
    ChatOpenRouter = None


def _openrouter(model: str, *, temperature: float = 0.4) -> Any:
    """OpenRouter via its OpenAI-compatible endpoint.

    Using ChatOpenAI (rather than ChatOpenRouter) keeps retry/timeout behaviour
    consistent with the DeepInfra/Felo adapters and avoids client-side hangs.
    """
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
        max_tokens=_MAX_OUTPUT_TOKENS,
        timeout=120,
        max_retries=1,
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "CodeMind",
        },
    )


def _groq(model: str, *, temperature: float = 0.25) -> ChatGroq:
    return ChatGroq(
        model=model,
        timeout=120,
        max_tokens=_MAX_OUTPUT_TOKENS,
        temperature=temperature,
    )


def _fireworks(model: str, *, temperature: float = 0.25) -> ChatFireworks:
    return ChatFireworks(
        model=model,
        timeout=120,
        max_tokens=_MAX_OUTPUT_TOKENS,
        temperature=temperature,
    )


def _mistral(model: str, *, temperature: float = 0.25) -> Any:
    if ChatMistralAI is None:
        raise RuntimeError("langchain_mistralai is not installed; Mistral models are unavailable.")
    return ChatMistralAI(
        model=model,
        timeout=120,
        max_tokens=_MAX_OUTPUT_TOKENS,
        temperature=temperature,
    )


def _cerebras(model: str, *, temperature: float = 0.25) -> ChatCerebras:
    return ChatCerebras(
        model=model,
        max_tokens=_MAX_OUTPUT_TOKENS,
        temperature=temperature,
    )


def _nvidia(model: str, *, temperature: float = 0.25) -> Any:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*(moonshotai/kimi-k2\.6|qwen/qwen3-coder-480b-a35b-instruct).*type is unknown and inference may fail.*",
            category=UserWarning,
        )
        return ChatNVIDIA(
            model=model,
            temperature=temperature,
            max_completion_tokens=_MAX_OUTPUT_TOKENS,
        )


def _deepinfra(model: str, *, temperature: float = 0.25) -> Any:
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("DEEPINFRA_API_TOKEN") or os.getenv("DEEPINFRA_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPINFRA_API_TOKEN is not configured")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://api.deepinfra.com/v1/openai",
        temperature=temperature,
        max_tokens=_MAX_OUTPUT_TOKENS,
        timeout=120,
        max_retries=1,
    )


def _felo(model: str, *, temperature: float = 0.25) -> Any:
    """Felo API Platform (OpenAI-compatible chat completions)."""
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("FELO_API_KEY") or os.getenv("felo_api_key")
    if not api_key:
        raise RuntimeError("FELO_API_KEY is not configured")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://openapi.felo.ai/api/v1",
        temperature=temperature,
        max_tokens=_MAX_OUTPUT_TOKENS,
        timeout=120,
        max_retries=1,
    )


class _LazyModel:
    def __init__(self, label: str, factory: Callable[[], Any]):
        self.label = label
        self._factory = factory
        self._instance: Any | None = None

    def get(self) -> Any:
        if self._instance is None:
            self._instance = self._factory()
        return self._instance


def _resolve_model_candidate(candidate: Any) -> Any:
    if isinstance(candidate, _LazyModel):
        return candidate.get()
    return candidate


OPENROUTER_MINIMAX_LLM = _LazyModel(
    os.getenv("OPENROUTER_PLANNER_MODEL", "minimax/minimax-m2.5"),
    lambda: _openrouter(os.getenv("OPENROUTER_PLANNER_MODEL", "minimax/minimax-m2.5"), temperature=0.35),
)
OPENROUTER_MINIMAX_FREE_LLM = _LazyModel(
    os.getenv("OPENROUTER_PLANNER_FALLBACK_MODEL", "minimax/minimax-m2.5:free"),
    lambda: _openrouter(os.getenv("OPENROUTER_PLANNER_FALLBACK_MODEL", "minimax/minimax-m2.5:free"), temperature=0.35),
)
OPENROUTER_QWEN_CODER_LLM = _LazyModel(
    os.getenv("OPENROUTER_CODER_MODEL", "qwen/qwen3-coder"),
    lambda: _openrouter(os.getenv("OPENROUTER_CODER_MODEL", "qwen/qwen3-coder"), temperature=0.2),
)
OPENROUTER_QWEN_CODER_FREE_LLM = _LazyModel(
    os.getenv("OPENROUTER_CODER_FREE_MODEL", "qwen/qwen3-coder:free"),
    lambda: _openrouter(os.getenv("OPENROUTER_CODER_FREE_MODEL", "qwen/qwen3-coder:free"), temperature=0.2),
)
OPENROUTER_GLM_LLM = _LazyModel(
    os.getenv("OPENROUTER_GLM_MODEL", "zhipuai/glm-5-2"),
    lambda: _openrouter(os.getenv("OPENROUTER_GLM_MODEL", "zhipuai/glm-5-2"), temperature=0.2),
)
GROQ_FAST_LLM = _LazyModel(
    os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b"),
    lambda: _groq(os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b"), temperature=0.2),
)
GROQ_BALANCED_LLM = _LazyModel(
    os.getenv("GROQ_BALANCED_MODEL", "openai/gpt-oss-120b"),
    lambda: _groq(os.getenv("GROQ_BALANCED_MODEL", "openai/gpt-oss-120b"), temperature=0.25),
)
GROQ_REASONING_LLM = _LazyModel(
    os.getenv("GROQ_REASONING_MODEL", "openai/gpt-oss-120b"),
    lambda: _groq(os.getenv("GROQ_REASONING_MODEL", "openai/gpt-oss-120b"), temperature=0.2),
)
MISTRAL_SMALL_LLM = _LazyModel(
    os.getenv("MISTRAL_SMALL_MODEL", "mistral-small-latest"),
    lambda: _mistral(os.getenv("MISTRAL_SMALL_MODEL", "mistral-small-latest"), temperature=0.2),
)
MISTRAL_MEDIUM_LLM = _LazyModel(
    os.getenv("MISTRAL_MEDIUM_MODEL", "mistral-medium-3-5"),
    lambda: _mistral(os.getenv("MISTRAL_MEDIUM_MODEL", "mistral-medium-3-5"), temperature=0.25),
)
MISTRAL_DEVSTRAL_LLM = _LazyModel(
    os.getenv("MISTRAL_DEVSTRAL_MODEL", "devstral-2512"),
    lambda: _mistral(os.getenv("MISTRAL_DEVSTRAL_MODEL", "devstral-2512"), temperature=0.2),
)
FIREWORKS_GPT_OSS_LLM = _LazyModel(
    os.getenv("FIREWORKS_DEBUG_MODEL", "accounts/fireworks/models/gpt-oss-120b"),
    lambda: _fireworks(os.getenv("FIREWORKS_DEBUG_MODEL", "accounts/fireworks/models/gpt-oss-120b"), temperature=0.2),
)
FIREWORKS_KIMI_LLM = _LazyModel(
    os.getenv("FIREWORKS_AGENT_MODEL", "accounts/fireworks/models/kimi-k2p6"),
    lambda: _fireworks(os.getenv("FIREWORKS_AGENT_MODEL", "accounts/fireworks/models/kimi-k2p6"), temperature=0.2),
)
FIREWORKS_DEEPSEEK_LLM = _LazyModel(
    os.getenv("FIREWORKS_UPDATE_MODEL", "accounts/fireworks/models/deepseek-v4-pro"),
    lambda: _fireworks(os.getenv("FIREWORKS_UPDATE_MODEL", "accounts/fireworks/models/deepseek-v4-pro"), temperature=0.2),
)
NVIDIA_KIMI_LLM = _LazyModel(
    os.getenv("NVIDIA_AGENT_MODEL", "moonshotai/kimi-k2.6"),
    lambda: _nvidia(os.getenv("NVIDIA_AGENT_MODEL", "moonshotai/kimi-k2.6"), temperature=0.2),
)
NVIDIA_QWEN_CODER_LLM = _LazyModel(
    os.getenv("NVIDIA_CODER_MODEL", "qwen/qwen3-coder-480b-a35b-instruct"),
    lambda: _nvidia(os.getenv("NVIDIA_CODER_MODEL", "qwen/qwen3-coder-480b-a35b-instruct"), temperature=0.2),
)
DEEPINFRA_PLANNER_LLM = _LazyModel(
    os.getenv("DEEPINFRA_PLANNER_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash"),
    lambda: _deepinfra(os.getenv("DEEPINFRA_PLANNER_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash"), temperature=0.2),
)
DEEPINFRA_CODER_LLM = _LazyModel(
    os.getenv("DEEPINFRA_CODER_MODEL", "Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo"),
    lambda: _deepinfra(os.getenv("DEEPINFRA_CODER_MODEL", "Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo"), temperature=0.2),
)
DEEPINFRA_ARCHITECT_LLM = _LazyModel(
    os.getenv("DEEPINFRA_ARCHITECT_MODEL", "openai/gpt-oss-120b"),
    lambda: _deepinfra(os.getenv("DEEPINFRA_ARCHITECT_MODEL", "openai/gpt-oss-120b"), temperature=0.2),
)
FELO_FAST_LLM = _LazyModel(
    os.getenv("FELO_FAST_MODEL", "gpt-5.6-terra"),
    lambda: _felo(os.getenv("FELO_FAST_MODEL", "gpt-5.6-terra"), temperature=0.2),
)
FELO_CODER_LLM = _LazyModel(
    os.getenv("FELO_CODER_MODEL", "gpt-5.6-sol"),
    lambda: _felo(os.getenv("FELO_CODER_MODEL", "gpt-5.6-sol"), temperature=0.2),
)
FELO_ARCHITECT_LLM = _LazyModel(
    os.getenv("FELO_ARCHITECT_MODEL", "kimi-k3"),
    lambda: _felo(os.getenv("FELO_ARCHITECT_MODEL", "kimi-k3"), temperature=0.2),
)
CEREBRAS_GPT_OSS_LLM = _LazyModel(
    os.getenv("CEREBRAS_REASONING_MODEL", "gpt-oss-120b"),
    lambda: _cerebras(os.getenv("CEREBRAS_REASONING_MODEL", "gpt-oss-120b"), temperature=0.2),
)
CEREBRAS_LLAMA_FAST_LLM = _LazyModel(
    os.getenv("CEREBRAS_FAST_MODEL", "llama3.1-8b"),
    lambda: _cerebras(os.getenv("CEREBRAS_FAST_MODEL", "llama3.1-8b"), temperature=0.2),
)
CEREBRAS_QWEN_235B_LLM = _LazyModel(
    os.getenv("CEREBRAS_LEGACY_ARCHITECT_MODEL", "qwen-3-235b-a22b-instruct-2507"),
    lambda: _cerebras(
        os.getenv("CEREBRAS_LEGACY_ARCHITECT_MODEL", "qwen-3-235b-a22b-instruct-2507"),
        temperature=0.2,
    ),
)


llm1 = OPENROUTER_MINIMAX_FREE_LLM
llm2 = OPENROUTER_QWEN_CODER_LLM

CLARIFIER_LLM = GROQ_FAST_LLM              
PLAN_LLM = GROQ_BALANCED_LLM              
ARCHITECT_LLM = CEREBRAS_GPT_OSS_LLM      
SUPERVISOR_LLM = GROQ_BALANCED_LLM        
RETRIEVER_LLM = GROQ_FAST_LLM             
CODER_LLM = FIREWORKS_DEEPSEEK_LLM        
UPDATE_CODER_LLM = MISTRAL_DEVSTRAL_LLM   
INTEGRATOR_LLM = NVIDIA_KIMI_LLM          
EXECUTOR_LLM = GROQ_FAST_LLM            

# --- Per-domain coder routing ------------------------------------------------
# The coder_agent and parallel_coder_agent no longer use one model for every
# file. The router below picks the model based on what kind of file is being
# written, since models have measurably different strengths by domain.
#
# To swap models later, change only these three constants — nothing else needs
# to change.
FRONTEND_CODER_LLM  = NVIDIA_KIMI_LLM       # Kimi 2.6 — UI / React / CSS
BACKEND_CODER_LLM   = OPENROUTER_GLM_LLM    # GLM-5.2  — Python / API / SQL
REASONING_CODER_LLM = CODER_LLM             # DeepSeek — fallback for ambiguous files

_FRONTEND_CODER_EXTS  = {".tsx", ".jsx", ".css", ".scss", ".sass", ".html", ".vue", ".svelte"}
_BACKEND_CODER_EXTS   = {".py", ".sql"}


def _coder_model_for_task(task: "ImplementationTask", plan: "Plan | None") -> Any:
    """Return the correct model to use when writing a specific file.

    Routing priority:
      1. File extension  — most reliable signal, used for all non-.js files.
      2. Project type    — used as a tiebreaker for .js (could be a Vite config
                           OR an Express server depending on the project).
      3. Fallback        — REASONING_CODER_LLM for everything else (JSON,
                           TOML, Markdown, lock files, configs, etc.).

    This function only selects the model. It does not change which graph node
    runs, which tools are available, or anything else about the coder pipeline.
    """
    ext = Path(task.filepath).suffix.lower()
    project_type = (getattr(plan, "project_type", "") or "").lower()

    if ext in _FRONTEND_CODER_EXTS:
        return FRONTEND_CODER_LLM

    if ext in _BACKEND_CODER_EXTS:
        return BACKEND_CODER_LLM

    # .js is ambiguous: disambiguate using project type
    if ext == ".js":
        if is_frontend_template(project_type):
            return FRONTEND_CODER_LLM
        if is_backend_template(project_type):
            return BACKEND_CODER_LLM

    tier = get_coder_model_tier(project_type, ext)
    if tier == "frontend":
        return FRONTEND_CODER_LLM
    elif tier == "backend":
        return BACKEND_CODER_LLM
        
    return REASONING_CODER_LLM


def prewarm_llm_models() -> None:
    """Optionally instantiate and lightly touch primary LLMs at server startup."""
    models = [CLARIFIER_LLM, PLAN_LLM, ARCHITECT_LLM, CODER_LLM, UPDATE_CODER_LLM, INTEGRATOR_LLM]
    seen: set[str] = set()
    for model_ref in models:
        label = _model_label(model_ref)
        if label in seen:
            continue
        seen.add(label)
        try:
            model = _resolve_model_candidate(model_ref)
            _with_timeout(lambda: model.invoke("ping"), min(15, _CHAIN_TIMEOUT), f"prewarm:{label}")
            print(f"[prewarm] warmed {label}")
        except Exception as exc:
            print(f"[prewarm] skipped {label}: {exc}")

_HF_FALLBACK_REPO    = os.getenv("HF_FALLBACK_REPO", "Qwen/Qwen2.5-Coder-7B-Instruct")
_GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")
_AUTO_MODEL_FALLBACK_DEFAULT = os.getenv("AUTO_MODEL_FALLBACK_ON_LIMIT", "1").strip().lower() not in {"0", "false", "no", "off"}

_hf_fallback_instance: ChatHuggingFace | None = None
_groq_fallback_instance: ChatGroq | None = None

def _get_hf_fallback() -> ChatHuggingFace:
    global _hf_fallback_instance
    if _hf_fallback_instance is None:
        _hf_fallback_instance = _hf(_HF_FALLBACK_REPO)
    return _hf_fallback_instance

def _get_groq_fallback() -> ChatGroq:
    global _groq_fallback_instance
    if _groq_fallback_instance is None:
        _groq_fallback_instance = ChatGroq(model=_GROQ_FALLBACK_MODEL, timeout=120, max_tokens=_MAX_OUTPUT_TOKENS)
    return _groq_fallback_instance

_NO_FALLBACK_RESULT = object()

_LIMIT_FALLBACK_ROUTES: dict[str, list[Any]] = {
    
    "clarifier": [GROQ_FAST_LLM, MISTRAL_SMALL_LLM, CEREBRAS_LLAMA_FAST_LLM, OPENROUTER_MINIMAX_LLM, OPENROUTER_MINIMAX_FREE_LLM],
    "planner": [GROQ_BALANCED_LLM, MISTRAL_MEDIUM_LLM, CEREBRAS_GPT_OSS_LLM, OPENROUTER_MINIMAX_LLM, OPENROUTER_MINIMAX_FREE_LLM],
    "supervisor": [GROQ_BALANCED_LLM, MISTRAL_SMALL_LLM, CEREBRAS_GPT_OSS_LLM, OPENROUTER_MINIMAX_LLM, OPENROUTER_MINIMAX_FREE_LLM],
    "requirements_search": [GROQ_FAST_LLM, MISTRAL_SMALL_LLM, CEREBRAS_LLAMA_FAST_LLM, OPENROUTER_MINIMAX_LLM, OPENROUTER_MINIMAX_FREE_LLM],
    "requirements": [GROQ_FAST_LLM, MISTRAL_SMALL_LLM, CEREBRAS_LLAMA_FAST_LLM, OPENROUTER_MINIMAX_LLM, OPENROUTER_MINIMAX_FREE_LLM],
    "retriever": [GROQ_FAST_LLM, MISTRAL_SMALL_LLM, CEREBRAS_LLAMA_FAST_LLM, OPENROUTER_MINIMAX_LLM, OPENROUTER_MINIMAX_FREE_LLM],
    "memory": [GROQ_FAST_LLM, MISTRAL_SMALL_LLM, CEREBRAS_LLAMA_FAST_LLM, OPENROUTER_MINIMAX_LLM, OPENROUTER_MINIMAX_FREE_LLM],
   
    "architect": [CEREBRAS_GPT_OSS_LLM, MISTRAL_DEVSTRAL_LLM, GROQ_REASONING_LLM, NVIDIA_QWEN_CODER_LLM, OPENROUTER_QWEN_CODER_LLM],
    "coder": [FIREWORKS_DEEPSEEK_LLM, MISTRAL_DEVSTRAL_LLM, NVIDIA_QWEN_CODER_LLM, GROQ_REASONING_LLM, CEREBRAS_GPT_OSS_LLM, OPENROUTER_QWEN_CODER_LLM, OPENROUTER_QWEN_CODER_FREE_LLM],
    "parallel_coder": [FIREWORKS_DEEPSEEK_LLM, MISTRAL_DEVSTRAL_LLM, NVIDIA_QWEN_CODER_LLM, GROQ_REASONING_LLM, CEREBRAS_GPT_OSS_LLM, OPENROUTER_QWEN_CODER_LLM, OPENROUTER_QWEN_CODER_FREE_LLM],
    "update_coder": [MISTRAL_DEVSTRAL_LLM, FIREWORKS_DEEPSEEK_LLM, NVIDIA_QWEN_CODER_LLM, GROQ_REASONING_LLM, CEREBRAS_GPT_OSS_LLM, OPENROUTER_QWEN_CODER_LLM, OPENROUTER_QWEN_CODER_FREE_LLM],
    "integrator": [NVIDIA_KIMI_LLM, MISTRAL_MEDIUM_LLM, CEREBRAS_GPT_OSS_LLM, GROQ_REASONING_LLM, OPENROUTER_QWEN_CODER_LLM],
    "executor": [GROQ_FAST_LLM, MISTRAL_SMALL_LLM, CEREBRAS_LLAMA_FAST_LLM, OPENROUTER_MINIMAX_LLM, OPENROUTER_MINIMAX_FREE_LLM],
    "executor_summary": [GROQ_FAST_LLM, MISTRAL_SMALL_LLM, CEREBRAS_LLAMA_FAST_LLM, OPENROUTER_MINIMAX_LLM, OPENROUTER_MINIMAX_FREE_LLM],
}

# --- Provider selection ------------------------------------------------------
# Set LLM_PROVIDER in .env to route every role to one provider whose key is
# valid. Allowed: auto | groq | nvidia | deepinfra | openrouter | mistral |
# cerebras | fireworks. "auto" keeps the built-in per-role defaults.
_LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "auto").strip().lower()

_PROVIDER_MODELS: dict[str, tuple[Any, Any, Any]] = {
    # provider: (fast/planner, coder/reasoning, frontend/architect)
    "groq": (GROQ_BALANCED_LLM, GROQ_REASONING_LLM, GROQ_FAST_LLM),
    "nvidia": (NVIDIA_KIMI_LLM, NVIDIA_QWEN_CODER_LLM, NVIDIA_KIMI_LLM),
    "deepinfra": (DEEPINFRA_PLANNER_LLM, DEEPINFRA_CODER_LLM, DEEPINFRA_ARCHITECT_LLM),
    "felo": (FELO_FAST_LLM, FELO_CODER_LLM, FELO_ARCHITECT_LLM),
    "openrouter": (OPENROUTER_MINIMAX_FREE_LLM, OPENROUTER_QWEN_CODER_LLM, OPENROUTER_GLM_LLM),
    "mistral": (MISTRAL_MEDIUM_LLM, MISTRAL_DEVSTRAL_LLM, MISTRAL_SMALL_LLM),
    "cerebras": (CEREBRAS_GPT_OSS_LLM, CEREBRAS_QWEN_235B_LLM, CEREBRAS_LLAMA_FAST_LLM),
    "fireworks": (FIREWORKS_GPT_OSS_LLM, FIREWORKS_DEEPSEEK_LLM, FIREWORKS_KIMI_LLM),
}

if _LLM_PROVIDER in _PROVIDER_MODELS:
    _provider_fast, _provider_coder, _provider_frontend = _PROVIDER_MODELS[_LLM_PROVIDER]
    CLARIFIER_LLM = _provider_fast
    PLAN_LLM = _provider_fast
    ARCHITECT_LLM = _provider_coder
    SUPERVISOR_LLM = _provider_fast
    RETRIEVER_LLM = _provider_fast
    EXECUTOR_LLM = _provider_fast
    CODER_LLM = _provider_coder
    UPDATE_CODER_LLM = _provider_coder
    INTEGRATOR_LLM = _provider_coder
    FRONTEND_CODER_LLM = _provider_frontend
    BACKEND_CODER_LLM = _provider_coder
    REASONING_CODER_LLM = _provider_coder
    llm1 = _provider_fast
    llm2 = _provider_coder

    _provider_pool = []
    for _candidate in (_provider_fast, _provider_coder, _provider_frontend):
        if _candidate not in _provider_pool:
            _provider_pool.append(_candidate)
    for _role, _route in list(_LIMIT_FALLBACK_ROUTES.items()):
        _merged = list(_provider_pool)
        for _candidate in _route:
            if _candidate not in _merged:
                _merged.append(_candidate)
        _LIMIT_FALLBACK_ROUTES[_role] = _merged
    print(f"[graph] LLM_PROVIDER={_LLM_PROVIDER}: all roles will try this provider first.")

# Tool kits for the remaining active roles.
WRITE_TOOLS  = [write_file, read_file, list_files, list_project_tree, tree_graph, get_current_directory, ddg_search]
EXEC_TOOLS   = [read_file, list_files, list_project_tree, tree_graph, get_current_directory, run_cmd, validate_imports]
UPDATE_TOOLS = [write_file, read_file, list_files, list_project_tree, tree_graph, get_current_directory, ddg_search, delete_file, replace_file_content, apply_patch_block, validate_imports]

# Structured-output calls (plan, architect, supervisor) must finish fast.
# Agent calls (coder, update_coder) are writing full files so need more time.
_CHAIN_TIMEOUT = int(os.getenv("GRAPH_CHAIN_TIMEOUT", "60"))   # fast: structured calls only
_AGENT_TIMEOUT = int(os.getenv("GRAPH_AGENT_TIMEOUT", "240"))  # coder/update_coder tool calls
_thread_pool = ThreadPoolExecutor(max_workers=4)
_PROMPT_CLIP_CHARS = int(os.getenv("GRAPH_PROMPT_CLIP_CHARS", "6000"))
_UPDATE_REQUEST_CLIP_CHARS = int(os.getenv("UPDATE_REQUEST_CLIP_CHARS", "12000"))
_UPDATE_SNAPSHOT_FILE_CHARS = int(os.getenv("UPDATE_SNAPSHOT_FILE_CHARS", "1800"))
_UPDATE_SNAPSHOT_TOTAL_CHARS = int(os.getenv("UPDATE_SNAPSHOT_TOTAL_CHARS", "16000"))
_UPDATE_SNAPSHOT_MAX_FILES = int(os.getenv("UPDATE_SNAPSHOT_MAX_FILES", "16"))
_UPDATE_FILE_OVERVIEW_CHARS = int(os.getenv("UPDATE_FILE_OVERVIEW_CHARS", "2500"))
_UPDATE_DELETE_MAX_ALLOWED = int(os.getenv("UPDATE_DELETE_MAX_ALLOWED", "3"))
_NODE_MAX_RETRIES = int(os.getenv("GRAPH_NODE_MAX_RETRIES", "1"))
_LLM_MAX_RETRIES = int(os.getenv("GRAPH_LLM_MAX_RETRIES", "2"))
_LLM_RETRY_BASE_DELAY = float(os.getenv("GRAPH_LLM_RETRY_BASE_DELAY", "1.0"))
_MAX_DEBUG_ATTEMPTS_DEFAULT = int(os.getenv("GRAPH_MAX_DEBUG_ATTEMPTS", "1"))
_MAX_SUPERVISOR_TURNS_DEFAULT = int(os.getenv("GRAPH_MAX_SUPERVISOR_TURNS", "40"))
_MAX_INTEGRATOR_ATTEMPTS = int(os.getenv("GRAPH_MAX_INTEGRATOR_ATTEMPTS", "3"))  # break integratorâ†’update loop
_VERIFICATION_TIMEOUT = int(os.getenv("GRAPH_VERIFICATION_TIMEOUT", "90"))

def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off"}

_FAST_PATH_DEFAULT = _env_flag("GRAPH_FAST_PATH", "1")
_SKIP_CLARIFIER_DEFAULT = _env_flag("GRAPH_SKIP_CLARIFIER", "1")
_ENABLE_INTEGRATOR_DEFAULT = _env_flag("GRAPH_ENABLE_INTEGRATOR", "0")
_ENABLE_REQUIREMENTS_LLM_DEFAULT = _env_flag("GRAPH_REQUIREMENTS_LLM", "0")
_ENABLE_MEMORY_SUMMARY_DEFAULT = _env_flag("GRAPH_MEMORY_SUMMARY", "0")
_ENABLE_RUNTIME_HEALTH_CHECK_DEFAULT = _env_flag("GRAPH_RUNTIME_HEALTH_CHECK", "1")
_ENABLE_PARALLEL_CODER_DEFAULT = _env_flag("GRAPH_PARALLEL_CODER", "0")
_ENABLE_GITHUB_SCOUT_DEFAULT = _env_flag("GRAPH_ENABLE_GITHUB_SCOUT", "0")
_GRAPH_DEBUG_DEFAULT = _env_flag("GRAPH_DEBUG", "0")

def _config_bool(config: RunnableConfig | None, key: str, default: bool) -> bool:
    configurable = (config or {}).get("configurable", {})
    value = configurable.get(key)
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)

def _fast_path_enabled(config: RunnableConfig | None) -> bool:
    return _config_bool(config, "fast_path", _FAST_PATH_DEFAULT)

def _skip_clarifier_enabled(config: RunnableConfig | None) -> bool:
    return _config_bool(config, "skip_clarifier", _SKIP_CLARIFIER_DEFAULT)

def _integrator_enabled(config: RunnableConfig | None) -> bool:
    return _config_bool(config, "enable_integrator", _ENABLE_INTEGRATOR_DEFAULT)

def _requirements_llm_enabled(config: RunnableConfig | None) -> bool:
    return _config_bool(config, "requirements_llm", _ENABLE_REQUIREMENTS_LLM_DEFAULT)

def _memory_summary_enabled(config: RunnableConfig | None) -> bool:
    return _config_bool(config, "memory_summary", _ENABLE_MEMORY_SUMMARY_DEFAULT)

def _runtime_health_check_enabled(config: RunnableConfig | None) -> bool:
    return _config_bool(config, "runtime_health_check", _ENABLE_RUNTIME_HEALTH_CHECK_DEFAULT)

def _parallel_coder_enabled(config: RunnableConfig | None) -> bool:
    return _config_bool(config, "parallel_coder", _ENABLE_PARALLEL_CODER_DEFAULT)

def _github_scout_enabled(config: RunnableConfig | None) -> bool:
    return _config_bool(config, "github_scout", _ENABLE_GITHUB_SCOUT_DEFAULT)

def _supports_structured_output(llm: Any) -> bool:
    llm = _resolve_model_candidate(llm)
    return not isinstance(llm, ChatHuggingFace)

def _get_fallback(llm: Any) -> Any:
    if _supports_structured_output(llm):
        return _get_groq_fallback()
    return _get_hf_fallback()

def _fallback_enabled(config: RunnableConfig | None) -> bool:
    configurable = (config or {}).get("configurable", {})
    value = configurable.get("auto_model_fallback_on_limit")
    if value is None:
        return _AUTO_MODEL_FALLBACK_DEFAULT
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)

def _role_from_label(label: str) -> str:
    return (label or "unknown").split(":", 1)[0]

def _is_limit_error(exc: Exception) -> bool:
    err = str(exc).lower()
    limit_markers = (
        "401",
        "402",
        "403",
        "404",
        "412",
        "429",
        "access denied",
        "billing",
        "bad record mac",
        "capacity",
        "connection",
        "connecterror",
        "context length",
        "context window",
        "credits",
        "depleted your monthly included credits",
        "does not exist",
        "forbidden",
        "insufficient credits",
        "invalid model",
        "maximum context length",
        "model is overloaded",
        "model not found",
        "model_not_found",
        "not found",
        "not installed",
        "not supported",
        "payment required",
        "permission",
        "pre-paid credits",
        "purchase pre-paid credits",
        "quota",
        "rate limit",
        "readerror",
        "remote protocol",
        "requests per min",
        "resource exhausted",
        "ssl",
        "sslv3_alert_bad_record_mac",
        "token limit",
        "timeout",
        "timed out",
        "too many requests",
        "tokens per min",
        "exceeded",
    )
    return any(marker in err for marker in limit_markers)

def _model_label(model: Any) -> str:
    if isinstance(model, _LazyModel):
        return model.label
    direct_attrs = ("model_name", "model", "repo_id")
    for attr in direct_attrs:
        value = getattr(model, attr, None)
        if value:
            return str(value)
    nested_llm = getattr(model, "llm", None)
    for attr in direct_attrs:
        value = getattr(nested_llm, attr, None)
        if value:
            return str(value)
    return model.__class__.__name__

def _limit_fallback_models(primary: Any, label: str, config: RunnableConfig | None) -> list[Any]:
    if not _fallback_enabled(config):
        return []
    configured = _LIMIT_FALLBACK_ROUTES.get(_role_from_label(label), [llm1, llm2])
    unique: list[Any] = []
    seen = {id(primary)}
    seen_labels = {_model_label(primary)}
    for candidate in configured:
        candidate_label = _model_label(candidate)
        if id(candidate) in seen or candidate_label in seen_labels:
            continue
        try:
            model = _resolve_model_candidate(candidate)
        except Exception as exc:
            print(f"[{label}] skipping unavailable fallback {candidate_label}: {exc}")
            continue
        if model is None or id(model) in seen:
            continue
        unique.append(model)
        seen.add(id(model))
        seen_labels.add(candidate_label)
    return unique

def _try_limit_fallbacks(
    *,
    primary_error: Exception,
    primary_model: Any,
    label: str,
    config: RunnableConfig | None,
    invoke_with_model: Callable[[Any], Any],
) -> Any:
    if not _is_limit_error(primary_error):
        return _NO_FALLBACK_RESULT

    fallback_models = _limit_fallback_models(primary_model, label, config)
    if not fallback_models:
        return _NO_FALLBACK_RESULT

    fallback_labels = ", ".join(_model_label(model) for model in fallback_models)
    print(f"[{label}] limit reached on {_model_label(primary_model)}; auto-switching to {fallback_labels}")
    for model in fallback_models:
        try:
            return invoke_with_model(model)
        except Exception as exc:
            print(f"[{label}] fallback {_model_label(model)} failed: {exc!r}")
    return _NO_FALLBACK_RESULT


def _llm_invoke_with_fallback(
    primary_model: Any,
    *,
    label: str,
    config: RunnableConfig | None,
    invoke_with_model: Callable[[Any, str], Any],
    generic_fallback: Any | None = None,
    default: Any = None,
) -> Any:
    """Shared retry/fallback wrapper for structured and agent LLM calls."""
    try:
        return _retry_llm_call(lambda: invoke_with_model(primary_model, label), label=label)
    except Exception as primary_error:
        fallback_result = _try_limit_fallbacks(
            primary_error=primary_error,
            primary_model=primary_model,
            label=label,
            config=config,
            invoke_with_model=lambda model: _retry_llm_call(
                lambda: invoke_with_model(model, f"{label}:fallback"),
                label=f"{label}:fallback",
            ),
        )
        if fallback_result is not _NO_FALLBACK_RESULT:
            return fallback_result
        if generic_fallback is None:
            print(f"[{label}] LLM call failed: {primary_error!r}")
            return default
        print(f"[graph] Primary failed ({label}): {primary_error!r} -> trying generic fallback")
        try:
            return _retry_llm_call(
                lambda: invoke_with_model(generic_fallback, f"{label}:fallback"),
                label=f"{label}:fallback",
            )
        except Exception as fallback_error:
            print(f"[graph] Fallback also failed ({label}): {fallback_error!r}")
            return default

def _with_timeout(fn: Callable, timeout: int, label: str) -> Any:
    future = _thread_pool.submit(fn)
    try:
        return future.result(timeout=timeout)
    except FuturesTimeoutError:
        future.cancel()
        raise TimeoutError(f"[timeout] {label} exceeded {timeout}s")

def _retry_llm_call(
    fn: Callable,
    max_retries: int = _LLM_MAX_RETRIES,
    base_delay: float = _LLM_RETRY_BASE_DELAY,
    label: str = "llm",
) -> Any:
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            err_str = str(exc).lower()
            if "rate limit" in err_str or "429" in err_str or "too many requests" in err_str:
                if attempt == max_retries - 1:
                    raise
                sleep_time = base_delay * (2 ** attempt)
                print(f"[{label}] âš ï¸ Rate limit hit. Retrying in {sleep_time}s... ({attempt + 1}/{max_retries})")
                time.sleep(sleep_time)
            elif "tool_use_failed" in err_str or "failed_generation" in err_str or "parse" in err_str or "validation" in err_str or "badrequest" in err_str:
                if attempt == max_retries - 1:
                    raise
                print(f"[{label}] âš ï¸ Generation/Parsing error. Retrying... ({attempt + 1}/{max_retries})")
            else:
                raise

def _invoke_structured(
    llm: Any,
    prompt_fn: Callable[..., str],
    prompt_kwargs: dict,
    schema: type,
    label: str = "structured",
    config: RunnableConfig | None = None,
) -> Any:
    def _call(model: Any) -> Any:
        if _supports_structured_output(model):
            prompt = prompt_fn(**prompt_kwargs, format_instructions="")
            return model.with_structured_output(schema).invoke(prompt)
        else:
            parser = PydanticOutputParser(pydantic_object=schema)
            prompt = prompt_fn(**prompt_kwargs, format_instructions=parser.get_format_instructions())
            raw = model.invoke(prompt)
            content = raw.content if hasattr(raw, "content") else str(raw)
            try:
                return parser.parse(content)
            except Exception:
                repaired_content = repair_structured_json(content)
                if repaired_content != content:
                    print(f"[graph] Repaired malformed structured output for {label}.")
                return parser.parse(repaired_content)

    def _execute(model: Any):
        return _with_timeout(lambda: _call(_resolve_model_candidate(model)), _CHAIN_TIMEOUT, label)

    return _llm_invoke_with_fallback(
        llm,
        label=label,
        config=config,
        invoke_with_model=lambda model, _active_label: _execute(model),
        generic_fallback=_get_fallback(llm),
        default=None,
    )

def _react(llm: Any, tools: list, system_prompt: str):
    model = _resolve_model_candidate(llm)
    # NVIDIA's current LangChain catalog has not caught up with Kimi K2.6's
    # documented tool support, so keep the real errors but silence that stale warning.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*(moonshotai/kimi-k2\.6|qwen/qwen3-coder-480b-a35b-instruct).*not known to support tools\..*",
            category=UserWarning,
        )
        return create_agent(model=model, tools=tools, system_prompt=system_prompt)

def _run(agent: Any, user_input: str, label: str = "agent", config: RunnableConfig = None) -> str:
    """Single, canonical agent runner: timeout + rate-limit retry + message extraction."""
    print(f"[{label}] -> Running agent...")
    def _invoke():
        result = _with_timeout(
            lambda: agent.invoke({"messages": [HumanMessage(content=user_input)]}, config=config),
            _AGENT_TIMEOUT, label,
        )
        msgs = result.get("messages", []) if isinstance(result, dict) else []
        return msgs[-1].content if msgs and hasattr(msgs[-1], "content") else str(result)
    try:
        return _retry_llm_call(_invoke, label=label)
    except Exception as e:
        print(f"[{label}] Agent failed: {e}")
        return f"ERROR: {e}"

def _run_agent_with_model_fallback(
    llm: Any,
    tools: list,
    system_prompt: str,
    user_input: str,
    *,
    label: str = "agent",
    config: RunnableConfig | None = None,
) -> str:
    print(f"[{label}] -> Running agent...")

    def _invoke_with_model(model: Any, active_label: str) -> str:
        agent = _react(model, tools, system_prompt)

        def _call(payload: str):
            return _with_timeout(
                lambda: agent.invoke({"messages": [HumanMessage(content=payload)]}, config=config),
                _AGENT_TIMEOUT,
                active_label,
            )

        try:
            result = _call(user_input)
        except Exception as exc:
            message = str(exc)
            # Some OpenAI-compatible backends (notably Groq's gpt-oss models)
            # reject the whole request when the model emits a tool name that is
            # not in request.tools (e.g. abbreviating ddg_search to "search").
            # Retry once with an explicit, exhaustive tool list instead of
            # letting the verify/repair loop spin.
            if "not in request.tools" not in message:
                raise
            tool_names = ", ".join(
                getattr(tool, "name", None) or getattr(tool, "__name__", str(tool))
                for tool in tools
            )
            corrected = (
                f"{user_input}\n\n"
                "IMPORTANT: You may ONLY call the following tool names, exactly as written: "
                f"{tool_names}. Do not invent, shorten, or abbreviate tool names. In "
                "particular, the web search tool is named `ddg_search`, never `search`."
            )
            print(f"[{active_label}] Invalid tool name from model; retrying with explicit tool list.")
            result = _call(corrected)

        msgs = result.get("messages", []) if isinstance(result, dict) else []
        return msgs[-1].content if msgs and hasattr(msgs[-1], "content") else str(result)

    result = _llm_invoke_with_fallback(
        llm,
        label=label,
        config=config,
        invoke_with_model=_invoke_with_model,
        default=None,
    )
    return result if result is not None else f"ERROR: {label} failed"


def _log_tool(state: PipelineState, tool_name: str, input: dict, output: str, status: str = "success") -> dict:
    import time
    entry = ToolTranscriptEntry(
        tool_name=tool_name,
        input=input,
        output=str(output)[:1000], # Cap output in transcript
        status=status,
        timestamp=time.time()
    )
    return {"tool_transcript": [entry]}

def _invoke_tool_with_logging(tool: Any, args: Any, state: PipelineState, config: RunnableConfig) -> tuple[str, dict]:
    try:
        res = tool.invoke(args, config=config)
        return str(res), _log_tool(state, tool.name, args if isinstance(args, dict) else {"arg": args}, res)
    except Exception as e:
        err = f"ERROR: {e}"
        return err, _log_tool(state, tool.name, args if isinstance(args, dict) else {"arg": args}, err, status="error")


def _clip_text(text: str | None, limit: int = _PROMPT_CLIP_CHARS, *, empty_fallback: str = "(empty)") -> str:
    if text is None:
        return empty_fallback
    text = str(text).strip()
    if not text:
        return empty_fallback
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... (truncated)"


_STATIC_ANALYSIS_SKIP_PARTS = {
    ".git",
    ".next",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}


def _resolve_project_root(config: RunnableConfig) -> Path:
    configurable = (config or {}).get("configurable", {})
    explicit_root = configurable.get("project_root")
    if explicit_root:
        return Path(explicit_root).resolve()
    thread_id = configurable.get("thread_id", "default")
    return (GENERATED_PROJECTS_DIR / thread_id).resolve()


def _auto_fill_generated_env(root: Path, label: str) -> dict:
    try:
        result = auto_fill_project_env(root)
    except Exception as exc:
        print(f"[env-sync] {label}: skipped ({exc})")
        return {}

    if result.written_files:
        print(
            f"[env-sync] {label}: filled {len(result.filled_keys)} env key(s) "
            f"across {len(result.written_files)} file(s): {', '.join(result.written_files)}"
        )
    elif result.filled_keys:
        print(f"[env-sync] {label}: generated env files already had usable values")

    if result.missing_keys:
        print(
            f"[env-sync] {label}: {len(result.missing_keys)} referenced key(s) "
            "were not available in the main .env/process environment"
        )
    return {
        "env_sync_missing_keys": result.missing_keys,
        "env_sync_written_files": result.written_files,
    }


def _refresh_generated_file_manifest(root: Path) -> dict[str, GeneratedFileInfo]:
    try:
        return build_generated_file_manifest(root)
    except Exception as exc:
        print(f"[manifest] refresh skipped: {exc}")
        return {}


def _manifest_state_update(config: RunnableConfig) -> dict:
    root = _resolve_project_root(config)
    manifest = _refresh_generated_file_manifest(root)
    return {"generated_file_manifest": manifest} if manifest else {}


def _parse_run_cmd_result(result: str) -> tuple[int, str, str]:
    match = re.match(r"exit_code:\s*(-?\d+)\nstdout:\n(.*)\nstderr:\n(.*)\Z", result, re.S)
    if not match:
        return -1, "", result
    return int(match.group(1)), match.group(2), match.group(3)


def _make_execution_result(command: str, exit_code: int, stdout: str, stderr: str, *, error_type: str | None = None) -> ExecutionResult:
    return ExecutionResult(
        command=command,
        returncode=exit_code,
        stdout=stdout,
        stderr=stderr,
        success=exit_code == 0,
        error_type=error_type,
    )


def _iter_source_paths(config: RunnableConfig, suffixes: tuple[str, ...]) -> list[str]:
    files_listing = list_files.invoke({"directory": "."}, config=config)
    if not files_listing or files_listing == "No files found." or files_listing.startswith("ERROR:"):
        return []

    filepaths = []
    for raw_line in files_listing.splitlines():
        path = raw_line.strip().replace("\\", "/")
        if not path:
            continue
        parts = set(path.split("/"))
        if parts & _STATIC_ANALYSIS_SKIP_PARTS:
            continue
        if path.lower().endswith(suffixes):
            filepaths.append(path)
    return sorted(filepaths)


def _run_static_analysis(config: RunnableConfig) -> ExecutionResult | None:
    python_files = _iter_source_paths(config, (".py",))
    for path in python_files:
        command = f'python -m py_compile "{path}"'
        raw_result = run_cmd.invoke({"cmd": command, "cwd": ".", "timeout": 30}, config=config)
        exit_code, stdout, stderr = _parse_run_cmd_result(raw_result)
        if exit_code != 0:
            return _make_execution_result(command, exit_code, stdout, stderr or f"Static analysis failed for {path}", error_type="StaticAnalysisError")

    node_check = run_cmd.invoke({"cmd": "node --version", "cwd": ".", "timeout": 10}, config=config)
    node_exit_code, _, _ = _parse_run_cmd_result(node_check)
    if node_exit_code == 0:
        javascript_files = _iter_source_paths(config, (".js", ".mjs", ".cjs"))
        for path in javascript_files:
            command = f'node --check "{path}"'
            raw_result = run_cmd.invoke({"cmd": command, "cwd": ".", "timeout": 30}, config=config)
            exit_code, stdout, stderr = _parse_run_cmd_result(raw_result)
            if exit_code != 0:
                return _make_execution_result(command, exit_code, stdout, stderr or f"Static analysis failed for {path}", error_type="StaticAnalysisError")

    return None


def _command_label(command: str, cwd: str = ".") -> str:
    if not cwd or cwd == ".":
        return command
    return f"(cd {cwd} && {command})"


def _run_checked_command(
    command: str,
    config: RunnableConfig,
    *,
    cwd: str = ".",
    timeout: int = _VERIFICATION_TIMEOUT,
    error_type: str | None = None,
) -> ExecutionResult:
    raw_result = run_cmd.invoke({"cmd": command, "cwd": cwd, "timeout": timeout}, config=config)
    exit_code, stdout, stderr = _parse_run_cmd_result(raw_result)
    return _make_execution_result(
        _command_label(command, cwd),
        exit_code,
        stdout,
        stderr,
        error_type=error_type,
    )


def _rel_dir(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    return "." if rel == "." else rel


def _iter_project_file_paths(root: Path, filename: str) -> list[Path]:
    matches: list[Path] = []
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name for name in dirnames
            if name not in _STATIC_ANALYSIS_SKIP_PARTS and not name.startswith(".")
        ]
        if filename in filenames:
            matches.append(Path(current_root) / filename)
    return sorted(matches, key=lambda path: (len(path.parts), path.as_posix()))


def _load_json_file(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _find_tsconfigs(package_dir: Path) -> list[Path]:
    configs: list[Path] = []
    for current_root, dirnames, filenames in os.walk(package_dir):
        dirnames[:] = [
            name for name in dirnames
            if name not in _STATIC_ANALYSIS_SKIP_PARTS and not name.startswith(".")
        ]
        if "tsconfig.json" in filenames:
            configs.append(Path(current_root) / "tsconfig.json")
    return sorted(configs, key=lambda path: (len(path.parts), path.as_posix()))


def _package_has_typescript(package_dir: Path, package_data: dict[str, Any]) -> bool:
    deps = {
        *(package_data.get("dependencies") or {}).keys(),
        *(package_data.get("devDependencies") or {}).keys(),
    }
    return "typescript" in deps or bool(_find_tsconfigs(package_dir))


_TAILWIND_DEPENDENCIES = ("tailwindcss", "postcss", "autoprefixer")
_TAILWIND_CONFIG_NAMES = ("tailwind.config.js", "tailwind.config.cjs", "tailwind.config.mjs", "tailwind.config.ts")
_POSTCSS_CONFIG_NAMES = ("postcss.config.js", "postcss.config.cjs", "postcss.config.mjs")


def _package_dependency_names(package_data: dict[str, Any]) -> set[str]:
    return {
        *(package_data.get("dependencies") or {}).keys(),
        *(package_data.get("devDependencies") or {}).keys(),
    }


def _has_any_file(package_dir: Path, filenames: tuple[str, ...]) -> bool:
    return any((package_dir / filename).exists() for filename in filenames)


def _project_mentions_tailwind(package_dir: Path) -> bool:
    text_files = ("PRD.md", "Architecture.md", "README.md")
    for filename in text_files:
        path = package_dir / filename
        if path.exists():
            try:
                if "tailwind" in path.read_text(encoding="utf-8", errors="ignore").lower():
                    return True
            except OSError:
                continue
    return False


def _project_has_tailwind_source(package_dir: Path) -> bool:
    for current_root, dirnames, filenames in os.walk(package_dir):
        dirnames[:] = [
            name for name in dirnames
            if name not in _STATIC_ANALYSIS_SKIP_PARTS and not name.startswith(".")
        ]
        for filename in filenames:
            if not filename.endswith((".css", ".js", ".jsx", ".ts", ".tsx")):
                continue
            path = Path(current_root) / filename
            try:
                content = path.read_text(encoding="utf-8", errors="ignore").lower()
            except OSError:
                continue
            if "@tailwind" in content or "@apply" in content or "tailwind css" in content or "tailwindcss" in content:
                return True
    return False


def _package_uses_tailwind(package_dir: Path, package_data: dict[str, Any]) -> bool:
    return (
        "tailwindcss" in _package_dependency_names(package_data)
        or _project_mentions_tailwind(package_dir)
        or _project_has_tailwind_source(package_dir)
    )


def _raw_tailwind_output_failure(root: Path) -> ExecutionResult | None:
    hits: list[str] = []
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name for name in dirnames
            if name not in _STATIC_ANALYSIS_SKIP_PARTS - {"dist", "build"} and not name.startswith(".")
        ]
        if not any(part in {"dist", "build"} for part in Path(current_root).parts):
            continue
        for filename in filenames:
            if not filename.endswith(".css"):
                continue
            path = Path(current_root) / filename
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if "@tailwind" in content or "@apply" in content:
                rel = path.relative_to(root).as_posix()
                hits.append(rel)
    if not hits:
        return None
    return _make_execution_result(
        "tailwind output validation",
        1,
        "",
        "Tailwind CSS was not processed before build output was produced. "
        f"Raw @tailwind/@apply directives remain in: {', '.join(hits[:8])}",
        error_type="TailwindBuildError",
    )


def _tailwind_setup_failure(root: Path) -> ExecutionResult | None:
    for package_json in _iter_project_file_paths(root, "package.json"):
        package_dir = package_json.parent
        package_data = _load_json_file(package_json)
        if not package_data or not _package_uses_tailwind(package_dir, package_data):
            continue

        dependency_names = _package_dependency_names(package_data)
        missing = [name for name in _TAILWIND_DEPENDENCIES if name not in dependency_names]
        if not _has_any_file(package_dir, _TAILWIND_CONFIG_NAMES):
            missing.append("tailwind.config.js")
        if not _has_any_file(package_dir, _POSTCSS_CONFIG_NAMES):
            missing.append("postcss.config.js")

        if missing:
            cwd = _rel_dir(package_dir, root)
            return _make_execution_result(
                _command_label("tailwind setup validation", cwd),
                1,
                "",
                "Tailwind CSS was requested or used, but setup is incomplete. "
                f"Missing: {', '.join(dict.fromkeys(missing))}. "
                "Add Tailwind/PostCSS dependencies and config before verification.",
                error_type="TailwindSetupError",
            )

    return _raw_tailwind_output_failure(root)


def _node_install_command(package_dir: Path) -> str:
    if (package_dir / "package-lock.json").exists():
        return "npm install --no-audit --no-fund --progress=false"
    return "npm install --no-audit --no-fund --progress=false"


def _needs_node_install(package_dir: Path) -> bool:
    node_modules = package_dir / "node_modules"
    if not node_modules.exists():
        return True
    try:
        node_modules_mtime = node_modules.stat().st_mtime
        package_mtime = max(
            path.stat().st_mtime
            for path in (package_dir / "package.json", package_dir / "package-lock.json")
            if path.exists()
        )
    except OSError:
        return True
    return package_mtime > node_modules_mtime


def _run_node_project_checks(config: RunnableConfig) -> list[ExecutionResult]:
    root = _resolve_project_root(config)
    results: list[ExecutionResult] = []

    for package_json in _iter_project_file_paths(root, "package.json"):
        package_dir = package_json.parent
        cwd = _rel_dir(package_dir, root)
        package_data = _load_json_file(package_json)
        if package_data is None:
            results.append(
                _make_execution_result(
                    _command_label("parse package.json", cwd),
                    1,
                    "",
                    f"Invalid package.json at {package_json.relative_to(root).as_posix()}",
                    error_type="PackageJsonError",
                )
            )
            return results

        if _needs_node_install(package_dir):
            install_result = _run_checked_command(
                _node_install_command(package_dir),
                config,
                cwd=cwd,
                timeout=_VERIFICATION_TIMEOUT,
                error_type="DependencyInstallError",
            )
            results.append(install_result)
            if not install_result.success:
                return results

        scripts = package_data.get("scripts") or {}
        check_commands: list[tuple[str, str]] = []
        if isinstance(scripts.get("lint"), str) and scripts["lint"].strip():
            check_commands.append(("npm run lint", "LintError"))
        if isinstance(scripts.get("build"), str) and scripts["build"].strip():
            check_commands.append(("npm run build", "BuildError"))
        elif isinstance(scripts.get("typecheck"), str) and scripts["typecheck"].strip():
            check_commands.append(("npm run typecheck", "TypeCheckError"))
        elif _package_has_typescript(package_dir, package_data):
            for tsconfig in _find_tsconfigs(package_dir):
                rel_tsconfig = tsconfig.relative_to(package_dir).as_posix()
                check_commands.append((f'npx tsc -p "{rel_tsconfig}" --noEmit', "TypeCheckError"))

        for command, error_type in check_commands:
            result = _run_checked_command(
                command,
                config,
                cwd=cwd,
                timeout=_VERIFICATION_TIMEOUT,
                error_type=error_type,
            )
            results.append(result)
            if not result.success:
                return results

    return results


def _has_python_sources(path: Path) -> bool:
    for current_root, dirnames, filenames in os.walk(path):
        dirnames[:] = [
            name for name in dirnames
            if name not in _STATIC_ANALYSIS_SKIP_PARTS and not name.startswith(".")
        ]
        if any(filename.endswith(".py") for filename in filenames):
            return True
    return False


def _run_python_dependency_checks(config: RunnableConfig) -> list[ExecutionResult]:
    root = _resolve_project_root(config)
    results: list[ExecutionResult] = []

    for requirements_path in _iter_project_file_paths(root, "requirements.txt"):
        package_dir = requirements_path.parent
        if not _has_python_sources(package_dir):
            continue
        cwd = _rel_dir(package_dir, root)
        result = _run_checked_command(
            "pip install -r requirements.txt",
            config,
            cwd=cwd,
            timeout=_VERIFICATION_TIMEOUT,
            error_type="DependencyInstallError",
        )
        results.append(result)
        if not result.success:
            return results

    return results


def _success_result(message: str = "Deterministic verification completed successfully.") -> ExecutionResult:
    return ExecutionResult(
        command="deterministic_verification",
        returncode=0,
        stdout=message,
        stderr="",
        success=True,
    )


def _run_project_verification(config: RunnableConfig, state: PipelineState | None = None) -> list[ExecutionResult]:
    results: list[ExecutionResult] = []
    root = _resolve_project_root(config)

    template = resolve_template_from_plan(state.get("plan")) if state and state.get("plan") else None
    if template:
        from codemind.services.pipeline_validation import validate_template_constraints
        template_failure = validate_template_constraints(root, template)
        if template_failure is not None:
            return [template_failure]

    placeholder_failure = placeholder_failure_result(scan_placeholders(root, PLACEHOLDER_PATTERNS))
    if placeholder_failure is not None:
        return [placeholder_failure]

    env_placeholder_failure = env_placeholder_failure_result(scan_runtime_env_placeholders(root))
    if env_placeholder_failure is not None:
        return [env_placeholder_failure]

    import_failure = validate_project_imports(root)
    if import_failure is not None:
        return [import_failure]

    static_failure = _run_static_analysis(config)
    if static_failure is not None:
        return [static_failure]

    results.extend(_run_node_project_checks(config))
    if results and not results[-1].success:
        return results

    tailwind_failure = _tailwind_setup_failure(root)
    if tailwind_failure is not None:
        return [tailwind_failure]

    results.extend(_run_python_dependency_checks(config))
    if results and not results[-1].success:
        return results

    if _runtime_health_check_enabled(config) and _runtime_health_check_needed(root):
        runtime_result = run_runtime_health_check(root)
        if runtime_result is not None:
            results.append(runtime_result)
            if not runtime_result.success:
                return results

    return results or [_success_result()]


def _results_success(results: list[ExecutionResult]) -> bool:
    return bool(results) and all(result.success for result in results)


def _test_report_from_execution_results(results: list[ExecutionResult]) -> TestReport:
    cases: list[TestCase] = []
    for idx, result in enumerate(results, start=1):
        output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
        cases.append(
            TestCase(
                name=f"check_{idx}",
                description=result.command,
                success=result.success,
                output=_clip_text(output, 1000, empty_fallback=""),
            )
        )

    failed = [result for result in results if not result.success]
    if failed:
        summary = f"Deterministic verification failed at: {failed[0].command}"
    else:
        summary = "Deterministic verification passed."
    return TestReport(test_cases=cases, summary=summary)


def _rollback_to_snapshot(state: PipelineState, config: RunnableConfig) -> dict:
    snapshot_path = state.get("rollback_snapshot_path")
    if not snapshot_path:
        return {
            "rollback_applied": False,
            "rollback_reason": "Automated fixes failed, but no rollback snapshot was available.",
        }

    try:
        project_root = _resolve_project_root(config)
        restore_project_snapshot(project_root, snapshot_path)
        return {
            "rollback_applied": True,
            "rollback_reason": "Automated fixes failed repeatedly, so the project was restored to the last stable snapshot.",
        }
    except Exception as exc:
        return {
            "rollback_applied": False,
            "rollback_reason": f"Automated fixes failed and rollback also failed: {exc}",
        }


def _request_keywords(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9_./-]+", text.lower())
    keywords = []
    seen = set()
    stopwords = {
        "the", "and", "for", "with", "this", "that", "from", "into", "then", "than", "have",
        "has", "had", "your", "their", "there", "what", "when", "where", "which", "will",
        "would", "should", "could", "make", "need", "needs", "want", "wants", "update",
        "change", "changes", "feature", "features", "file", "files", "code", "project",
    }
    for token in tokens:
        if len(token) < 3 or token in stopwords:
            continue
        if token not in seen:
            seen.add(token)
            keywords.append(token)
    return keywords


_REQUEST_PATH_PATTERN = re.compile(
    r"(?<![:\w/@])(?:[A-Za-z0-9_.-]+[\\/])*[A-Za-z0-9_.-]+\."
    r"(?:py|js|jsx|ts|tsx|css|html|json|md|txt|env|local|yml|yaml|toml|ini)\b",
    re.IGNORECASE,
)


def _extract_request_paths(text: str) -> list[str]:
    """Return file paths explicitly named by the user, normalized to POSIX style."""
    paths: list[str] = []
    seen: set[str] = set()
    skip_parts = {"node_modules", ".git", "dist", "build", "__pycache__"}

    for match in _REQUEST_PATH_PATTERN.finditer(text or ""):
        raw = match.group(0).strip("`'\".,;:()[]{}")
        if "://" in raw or raw.lower().startswith(("http", "www.")):
            continue

        path = raw.replace("\\", "/").lstrip("./")
        parts = [part for part in path.split("/") if part and part not in {".", ".."}]
        if not parts or any(part in skip_parts for part in parts):
            continue

        normalized = "/".join(parts)
        key = normalized.lower()
        if key not in seen:
            seen.add(key)
            paths.append(normalized)
    return paths


def _list_project_files(config: RunnableConfig) -> set[str]:
    files_listing = list_files.invoke({"directory": "."}, config=config)
    if not files_listing or files_listing == "No files found." or files_listing.startswith("ERROR:"):
        return set()
    return {
        line.strip().replace("\\", "/")
        for line in files_listing.splitlines()
        if line.strip()
    }


_FRONTEND_PROJECT_TYPES = {"react", "nextjs", "vue", "svelte", "fullstack"}
_FRONTEND_ENTRY_FILES = ("src/App.jsx", "src/App.tsx", "src/App.js")
_FRONTEND_BOOTSTRAP_FILES = ("src/main.jsx", "src/main.tsx", "src/main.js", "src/main.ts")
_VITE_TEMPLATE_MARKERS = (
    "Get started",
    "Edit <code>src/App",
    "Count is",
    "reactLogo",
    "viteLogo",
    "HMR",
)


def _frontend_entry_files(config: RunnableConfig) -> tuple[str, ...]:
    root = _resolve_project_root(config)
    ordered: list[str] = []

    for bootstrap in _FRONTEND_BOOTSTRAP_FILES:
        bootstrap_path = root / bootstrap
        if not bootstrap_path.exists():
            continue
        try:
            content = bootstrap_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        imported_paths = re.findall(
            r"""(?:from\s+|import\s*)["'](\./App(?:\.[jt]sx?)?)["']""",
            content,
        )
        for imported_path in imported_paths:
            rel_path = (Path(bootstrap).parent / imported_path).as_posix()
            if rel_path not in ordered:
                ordered.append(rel_path)

    for rel_path in _FRONTEND_ENTRY_FILES:
        if rel_path not in ordered:
            ordered.append(rel_path)
    return tuple(ordered)


def _is_frontend_project(state: PipelineState, config: RunnableConfig) -> bool:
    plan = state.get("plan")
    project_type = getattr(plan, "project_type", "").lower() if plan else ""
    if project_type in _FRONTEND_PROJECT_TYPES:
        return True

    root = _resolve_project_root(config)
    return (root / "package.json").exists() and any(
        (root / name).exists()
        for name in ("vite.config.js", "vite.config.ts", "next.config.js", "next.config.mjs")
    )


def _frontend_entry_has_template_boilerplate(config: RunnableConfig) -> bool:
    return bool(_frontend_template_entry_paths(config))


def _frontend_template_entry_paths(config: RunnableConfig) -> list[str]:
    root = _resolve_project_root(config)
    paths: list[str] = []
    for rel_path in _frontend_entry_files(config):
        path = root / rel_path
        if not path.exists():
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        marker_count = sum(1 for marker in _VITE_TEMPLATE_MARKERS if marker in content)
        if marker_count >= 2:
            paths.append(rel_path)
    return paths


def _project_title_from_state(state: PipelineState) -> str:
    plan = state.get("plan")
    title = (getattr(plan, "name", "") if plan else "") or state.get("original_user_prompt") or state.get("user_prompt") or "Generated Application"
    title = re.sub(r"\s+", " ", str(title)).strip()
    return title[:80] or "Generated Application"


def _project_description_from_state(state: PipelineState) -> str:
    plan = state.get("plan")
    description = (getattr(plan, "description", "") if plan else "") or state.get("original_user_prompt") or state.get("user_prompt") or ""
    description = re.sub(r"\s+", " ", str(description)).strip()
    return description[:180] or "A generated application assembled from the requested project plan."


def _project_features_from_state(state: PipelineState) -> list[str]:
    plan = state.get("plan")
    features = [str(feature).strip() for feature in (getattr(plan, "features", []) if plan else []) if str(feature).strip()]
    if not features:
        features = ["Core workflow", "User dashboard", "Data management", "System status"]
    return features[:6]


def _package_slug(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "generated-application"


def _html_title(title: str) -> str:
    return title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_MEDIA_CAPTURE_TERMS = (
    "camera",
    "webcam",
    "photo",
    "picture",
    "image",
    "scan",
    "scanner",
    "snapshot",
)
_FACE_CAPTURE_TERMS = ("face", "facial", "biometric", "recognition")
_ATTENDANCE_TERMS = ("attendance", "check-in", "check in", "checkin")
_MEDIA_CAPTURE_IMPLEMENTATION_PATTERNS = (
    r"navigator\.mediadevices",
    r"getusermedia",
    r"<video\b",
    r"<canvas\b",
    r"canvas\.todataurl",
    r"type\s*=\s*['\"]file['\"]",
    r"accept\s*=\s*['\"][^'\"]*image",
    r"capture\s*=",
    r"filereader",
    r"createobjecturl",
)


def _state_text_for_feature_detection(state: PipelineState) -> str:
    chunks: list[str] = [
        str(state.get("original_user_prompt") or ""),
        str(state.get("user_prompt") or ""),
    ]
    plan = state.get("plan")
    if plan:
        chunks.extend(
            [
                str(getattr(plan, "name", "")),
                str(getattr(plan, "description", "")),
                str(getattr(plan, "techstack", "")),
                str(getattr(plan, "project_type", "")),
                " ".join(str(feature) for feature in (getattr(plan, "features", []) or [])),
            ]
        )
        for file in getattr(plan, "files", []) or []:
            chunks.append(str(getattr(file, "path", "")))
            chunks.append(str(getattr(file, "purpose", "")))
    return re.sub(r"\s+", " ", " ".join(chunks)).lower()


def _request_needs_media_capture(state: PipelineState) -> bool:
    text = _state_text_for_feature_detection(state)
    has_face = any(term in text for term in _FACE_CAPTURE_TERMS)
    has_media = any(term in text for term in _MEDIA_CAPTURE_TERMS)
    has_attendance = any(term in text for term in _ATTENDANCE_TERMS)
    if "face detection" in text or "face recognition" in text or "facial recognition" in text:
        return True
    if has_face and (has_attendance or "detection" in text or "verify" in text or "match" in text):
        return True
    return has_media and (has_face or has_attendance or "capture" in text or "upload" in text)


def _frontend_has_media_capture(config: RunnableConfig) -> bool:
    root = _resolve_project_root(config)
    suffixes = {".js", ".jsx", ".ts", ".tsx", ".html"}
    skip_dirs = {"node_modules", "dist", "build", ".next", ".git", "coverage"}
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in skip_dirs and not name.startswith(".cache")]
        for filename in filenames:
            path = Path(current_root) / filename
            if path.suffix.lower() not in suffixes:
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="ignore").lower()
            except Exception:
                continue
            if any(re.search(pattern, content) for pattern in _MEDIA_CAPTURE_IMPLEMENTATION_PATTERNS):
                return True
    return False


def _render_frontend_repair_app_jsx(state: PipelineState) -> str:
    title = _project_title_from_state(state)
    description = _project_description_from_state(state)
    features = _project_features_from_state(state)
    needs_media_capture = _request_needs_media_capture(state)
    import_line = "import { useMemo, useRef, useState } from 'react'" if needs_media_capture else "import { useMemo, useState } from 'react'"
    media_state = """
  const videoRef = useRef(null)
  const [capturedImage, setCapturedImage] = useState('')
  const [cameraStatus, setCameraStatus] = useState('Camera idle')

  async function startCamera() {
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        setCameraStatus('Camera unavailable')
        return
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user' },
        audio: false,
      })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      setCameraStatus('Camera ready')
    } catch (error) {
      setCameraStatus('Camera permission denied')
    }
  }

  function captureFrame() {
    const video = videoRef.current
    if (!video || video.readyState < 2) {
      setCameraStatus('Start camera before capture')
      return
    }

    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth || 640
    canvas.height = video.videoHeight || 480
    const context = canvas.getContext('2d')
    context.drawImage(video, 0, 0, canvas.width, canvas.height)
    setCapturedImage(canvas.toDataURL('image/png'))
    setCameraStatus('Snapshot captured')
  }

  function handleImageUpload(event) {
    const file = event.target.files?.[0]
    if (!file) return
    setCapturedImage(URL.createObjectURL(file))
    setCameraStatus(`Loaded ${file.name}`)
  }
""" if needs_media_capture else ""
    media_panel = """
        <article className="capture-panel">
          <div className="panel-heading">
            <p className="eyebrow">Live scan</p>
            <h2>Camera check-in</h2>
          </div>
          <video ref={videoRef} className="camera-preview" autoPlay playsInline muted aria-label="Live camera preview" />
          <div className="capture-actions">
            <button type="button" onClick={startCamera}>Start camera</button>
            <button type="button" onClick={captureFrame}>Capture</button>
            <label className="upload-control">
              Upload image
              <input type="file" accept="image/*" capture="environment" onChange={handleImageUpload} />
            </label>
          </div>
          <div className="capture-result" aria-live="polite">
            {capturedImage ? (
              <img src={capturedImage} alt="Captured attendance verification" />
            ) : (
              <span>Awaiting verification image</span>
            )}
          </div>
          <p className="capture-status">{cameraStatus}</p>
        </article>
""" if needs_media_capture else ""
    return f"""{import_line}
import './App.css'

const appName = {json.dumps(title)}
const appDescription = {json.dumps(description)}
const featureList = {json.dumps(features)}

const initialRecords = featureList.map((feature, index) => ({{
  id: `TASK-${{String(index + 1).padStart(2, '0')}}`,
  name: feature,
  status: index === 0 ? 'active' : index === 1 ? 'ready' : 'queued',
  owner: index % 2 === 0 ? 'Operations' : 'Product',
}}))

function App() {{
  const [records, setRecords] = useState(initialRecords)
  const [selectedId, setSelectedId] = useState(initialRecords[0]?.id || '')
  const selected = records.find((record) => record.id === selectedId) || records[0]
{media_state}

  const metrics = useMemo(() => [
    {{ label: 'Features', value: records.length }},
    {{ label: 'Active', value: records.filter((record) => record.status === 'active').length }},
    {{ label: 'Ready', value: records.filter((record) => record.status === 'ready').length }},
  ], [records])

  function advanceSelected() {{
    if (!selected) return
    setRecords((current) =>
      current.map((record) =>
        record.id === selected.id
          ? {{ ...record, status: record.status === 'active' ? 'ready' : 'active' }}
          : record,
      ),
    )
  }}

  return (
    <main className="generated-app-shell">
      <header className="app-hero">
        <div>
          <p className="eyebrow">Generated project</p>
          <h1>{{appName}}</h1>
          <p className="hero-copy">{{appDescription}}</p>
        </div>
        <button type="button" onClick={{advanceSelected}}>
          Update selected
        </button>
      </header>

      <section className="metric-grid" aria-label="Project metrics">
        {{metrics.map((metric) => (
          <article className="metric-card" key={{metric.label}}>
            <span>{{metric.label}}</span>
            <strong>{{metric.value}}</strong>
          </article>
        ))}}
      </section>

      <section className="work-grid">
        <article className="feature-panel">
          <div className="panel-heading">
            <p className="eyebrow">Workflow</p>
            <h2>Planned feature flow</h2>
          </div>
          <div className="feature-list">
            {{records.map((record) => (
              <button
                type="button"
                className={{record.id === selected?.id ? 'feature-row selected' : 'feature-row'}}
                key={{record.id}}
                onClick={{() => setSelectedId(record.id)}}
              >
                <span>{{record.name}}</span>
                <strong>{{record.status}}</strong>
              </button>
            ))}}
          </div>
        </article>

{media_panel}
        <article className="detail-panel">
          <p className="eyebrow">Current focus</p>
          <h2>{{selected?.name || 'Project setup'}}</h2>
          <p>
            This entry file has been connected to the generated application plan so the first screen
            reflects the requested product instead of a starter scaffold.
          </p>
          <dl>
            <div>
              <dt>Status</dt>
              <dd>{{selected?.status || 'ready'}}</dd>
            </div>
            <div>
              <dt>Owner</dt>
              <dd>{{selected?.owner || 'Application'}}</dd>
            </div>
          </dl>
        </article>
      </section>
    </main>
  )
}}

export default App
"""


def _render_frontend_repair_css() -> str:
    return """:root {
  color: #172026;
  background: #eef3f7;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-width: 320px;
  min-height: 100vh;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

button {
  font: inherit;
}

.generated-app-shell {
  min-height: 100vh;
  padding: 32px;
  color: #172026;
}

.app-hero,
.panel-heading,
.metric-grid,
.work-grid {
  display: grid;
  gap: 18px;
}

.app-hero {
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: end;
  margin-bottom: 22px;
}

.eyebrow {
  margin: 0 0 8px;
  color: #59656f;
  font-size: 0.78rem;
  font-weight: 800;
  letter-spacing: 0;
  text-transform: uppercase;
}

h1,
h2,
p {
  margin: 0;
}

h1 {
  max-width: 760px;
  font-size: 3.2rem;
  line-height: 1;
}

h2 {
  font-size: 1.2rem;
}

.hero-copy {
  max-width: 760px;
  margin-top: 14px;
  color: #53616b;
  font-size: 1.05rem;
}

.app-hero button,
.feature-row {
  min-height: 44px;
  border: 1px solid #cbd8e2;
  border-radius: 8px;
  background: #ffffff;
  color: #172026;
  font-weight: 800;
  cursor: pointer;
}

.app-hero button {
  padding: 0 16px;
}

.metric-grid {
  grid-template-columns: repeat(3, minmax(0, 1fr));
  margin-bottom: 18px;
}

.metric-card,
.feature-panel,
.detail-panel {
  border: 1px solid #dbe3ea;
  border-radius: 8px;
  background: #ffffff;
  box-shadow: 0 18px 45px rgba(31, 54, 72, 0.08);
}

.metric-card {
  padding: 18px;
}

.metric-card span {
  display: block;
  color: #65727d;
  font-weight: 800;
}

.metric-card strong {
  display: block;
  margin-top: 10px;
  font-size: 2.3rem;
}

.work-grid {
  grid-template-columns: minmax(320px, 0.95fr) minmax(320px, 1.05fr);
}

.feature-panel,
.detail-panel {
  padding: 22px;
}

.feature-list {
  display: grid;
  gap: 10px;
}

.feature-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  width: 100%;
  padding: 0 14px;
  text-align: left;
}

.feature-row.selected {
  border-color: #2474b5;
  background: #edf6ff;
}

.feature-row strong {
  color: #2474b5;
  text-transform: capitalize;
}

.detail-panel p {
  margin-top: 12px;
  color: #53616b;
}

.capture-panel {
  display: grid;
  gap: 14px;
  padding: 22px;
  border: 1px solid #dbe3ea;
  border-radius: 8px;
  background: #ffffff;
  box-shadow: 0 18px 45px rgba(31, 54, 72, 0.08);
}

.camera-preview,
.capture-result {
  width: 100%;
  aspect-ratio: 4 / 3;
  border: 1px solid #dbe3ea;
  border-radius: 8px;
  background: #172026;
  object-fit: cover;
}

.capture-result {
  display: grid;
  place-items: center;
  overflow: hidden;
  background: #edf3f7;
  color: #59656f;
  font-weight: 800;
}

.capture-result img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.capture-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.capture-actions button,
.upload-control {
  min-height: 42px;
  border: 1px solid #cbd8e2;
  border-radius: 8px;
  background: #ffffff;
  color: #172026;
  font-weight: 800;
  cursor: pointer;
}

.capture-actions button {
  padding: 0 14px;
}

.upload-control {
  display: inline-grid;
  place-items: center;
  padding: 0 14px;
}

.upload-control input {
  position: absolute;
  width: 1px;
  height: 1px;
  opacity: 0;
  pointer-events: none;
}

.capture-status {
  color: #53616b;
  font-weight: 800;
}

dl {
  display: grid;
  gap: 12px;
  margin: 22px 0 0;
}

dl div {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  padding: 12px 0;
  border-top: 1px solid #e4ebf1;
}

dt {
  color: #65727d;
  font-weight: 800;
}

dd {
  margin: 0;
  text-transform: capitalize;
}

@media (max-width: 820px) {
  .generated-app-shell {
    padding: 20px;
  }

  .app-hero,
  .metric-grid,
  .work-grid {
    grid-template-columns: 1fr;
  }

  .app-hero button {
    width: fit-content;
  }

  h1 {
    font-size: 2rem;
  }
}
"""


def _repair_frontend_template_entry(state: PipelineState, config: RunnableConfig) -> bool:
    template_paths = _frontend_template_entry_paths(config)
    if not template_paths:
        return False

    root = _resolve_project_root(config)
    target_rel = template_paths[0]
    target = root / target_rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_render_frontend_repair_app_jsx(state), encoding="utf-8")

    css_path = target.parent / "App.css"
    css_path.write_text(_render_frontend_repair_css(), encoding="utf-8")

    title = _project_title_from_state(state)
    index_path = root / "index.html"
    if index_path.exists():
        index_html = index_path.read_text(encoding="utf-8", errors="ignore")
        safe_title = _html_title(title)
        if re.search(r"<title>.*?</title>", index_html, flags=re.I | re.S):
            index_html = re.sub(r"<title>.*?</title>", f"<title>{safe_title}</title>", index_html, flags=re.I | re.S)
        else:
            index_html = index_html.replace("</head>", f"    <title>{safe_title}</title>\n  </head>")
        index_path.write_text(index_html, encoding="utf-8")

    package_path = root / "package.json"
    package_data = _load_json_file(package_path) if package_path.exists() else None
    if package_data and package_data.get("name") in {"react-vite", "vite-react", "vite-react-app"}:
        package_data["name"] = _package_slug(title)
        package_path.write_text(json.dumps(package_data, indent=2) + "\n", encoding="utf-8")

    print(f"[completion_guard] Repaired starter frontend entry: {target_rel}")
    return True


def _active_task_plan(state: PipelineState) -> TaskPlan | None:
    coder_state = state.get("coder_state")
    if coder_state and coder_state.task_plan:
        return coder_state.task_plan
    return state.get("task_plan")


def _missing_planned_files(state: PipelineState, config: RunnableConfig) -> list[str]:
    task_plan = _active_task_plan(state)
    if not task_plan:
        return []
    planned_files = [
        task.filepath.replace("\\", "/")
        for task in task_plan.implementation_steps
        if getattr(task, "filepath", None)
    ]
    existing_files = _list_project_files(config)
    return [path for path in planned_files if path not in existing_files]


def _completion_guard_failure(state: PipelineState, config: RunnableConfig) -> ExecutionResult | None:
    if _coder_progress_incomplete(state):
        return ExecutionResult(
            command="completion_guard",
            returncode=1,
            stdout="",
            stderr="Project is not complete: coder still has pending implementation steps.",
            success=False,
            error_type="IncompleteProjectError",
        )

    missing_files = _missing_planned_files(state, config)
    if missing_files:
        preview = ", ".join(missing_files[:12])
        if len(missing_files) > 12:
            preview += f", ... (+{len(missing_files) - 12} more)"
        return ExecutionResult(
            command="completion_guard",
            returncode=1,
            stdout="",
            stderr=f"Project is not complete: planned files are missing: {preview}",
            success=False,
            error_type="IncompleteProjectError",
        )

    if _is_frontend_project(state, config) and _frontend_entry_has_template_boilerplate(config):
        if _repair_frontend_template_entry(state, config) and not _frontend_entry_has_template_boilerplate(config):
            return None
        return ExecutionResult(
            command="completion_guard",
            returncode=1,
            stdout="",
            stderr=(
                "Project is not complete: the frontend entry file still contains "
                "the default Vite/React starter UI instead of the requested app."
            ),
            success=False,
            error_type="TemplateBoilerplateError",
        )

    if (
        _is_frontend_project(state, config)
        and _request_needs_media_capture(state)
        and not _frontend_has_media_capture(config)
    ):
        return ExecutionResult(
            command="completion_guard",
            returncode=1,
            stdout="",
            stderr=(
                "Project is not complete: the requested camera/photo workflow has no "
                "real camera or image upload capture path."
            ),
            success=False,
            error_type="MediaCaptureMissingError",
        )

    return None


def _coder_steps_finished(state: PipelineState) -> bool:
    coder_state = state.get("coder_state")
    if not coder_state:
        return False
    return coder_state.current_step_idx >= len(coder_state.task_plan.implementation_steps)


def _completion_guard_should_use_update_coder(state: PipelineState, failure: ExecutionResult | None) -> bool:
    if failure is None or failure.command != "completion_guard":
        return False
    if "pending implementation steps" in (failure.stderr or ""):
        return False
    if state.get("coder_state"):
        return _coder_steps_finished(state)
    return bool(_github_repo_found(state))


def _completion_guard_repair_prompt(failure: ExecutionResult) -> str:
    return (
        "Fix integration issues found by the completion guard:\n"
        f"- {failure.stderr.strip()}\n\n"
        "Create missing planned files at the exact requested paths, rewrite any remaining "
        "starter/template entry content, and wire the existing app together. Keep the fix scoped."
    )


def _verification_failure_repair_prompt(failure: ExecutionResult) -> str:
    details = "\n".join(
        part
        for part in (
            f"Command: {failure.command}",
            f"Error type: {failure.error_type or 'VerificationError'}",
            f"STDOUT:\n{_clip_text(failure.stdout, 2000, empty_fallback='(empty)')}",
            f"STDERR:\n{_clip_text(failure.stderr, 3000, empty_fallback='(empty)')}",
        )
        if part
    )
    if failure.error_type in {"TailwindSetupError", "TailwindBuildError"}:
        return (
            "Fix the Tailwind verification failure below. Ensure the project has "
            "`tailwindcss`, `postcss`, and `autoprefixer` installed, plus "
            "`tailwind.config.js` and `postcss.config.js` at the package root. "
            "If build output already exists, rebuild it so no raw `@tailwind` or "
            "`@apply` directives remain in generated CSS.\n\n"
            f"{details}"
        )
    return (
        "Fix the deterministic verification failure below. Keep the change scoped, "
        "read the files you edit, and do not delete unrelated files.\n\n"
        f"{details}"
    )


def _route_completion_guard_to_update_coder(
    failure: ExecutionResult,
    *,
    turn_count: int | None = None,
    rationale: str = "Completion guard found a concrete integration issue; routing directly to update_coder.",
) -> dict:
    update: dict[str, Any] = {
        "status": "delegating_update_coder",
        "user_prompt": _completion_guard_repair_prompt(failure),
        "thinking_rationale": rationale,
        "execution_results": [failure],
        "debug_attempts": 0,
    }
    if turn_count is not None:
        update["supervisor_turns"] = turn_count
    return update


def _supervisor_turn_limit(state: PipelineState) -> int:
    configured_limit = int(state.get("max_supervisor_turns") or _MAX_SUPERVISOR_TURNS_DEFAULT)
    configured_limit = min(configured_limit, _MAX_SUPERVISOR_TURNS_DEFAULT)
    task_plan = _active_task_plan(state)
    if not task_plan:
        return configured_limit

    # Each implementation task normally needs one coder turn plus a supervisor turn,
    # then a few integration/testing passes. Keep this bounded so failed repairs do not
    # spend many minutes looping through agents that cannot change the outcome.
    return max(configured_limit, len(task_plan.implementation_steps) * 2 + 20)



def _coder_progress_incomplete(state: PipelineState) -> bool:
    coder_state = state.get("coder_state")
    if not coder_state:
        return False
    steps = coder_state.task_plan.implementation_steps
    return coder_state.current_step_idx < len(steps)


def _request_explicitly_allows_file_deletion(update_request: str) -> bool:
    text = (update_request or "").lower()
    delete_word = r"\b(delete|remove|rm|discard|drop)\b"
    file_word = r"\b(file|files|folder|directory|path)\b"
    path_like = r"[\w./-]+\.(py|js|jsx|ts|tsx|css|html|json|md|txt|env|yml|yaml)"
    return bool(
        re.search(delete_word, text)
        and (re.search(file_word, text) or re.search(path_like, text))
    )


def _update_deletions_allowed(update_request: str, deleted_paths: list[str]) -> bool:
    if not deleted_paths:
        return True
    if not _request_explicitly_allows_file_deletion(update_request):
        return False
    if len(deleted_paths) > _UPDATE_DELETE_MAX_ALLOWED:
        return False

    request_text = (update_request or "").lower().replace("\\", "/")
    for deleted_path in deleted_paths:
        normalized = deleted_path.lower().replace("\\", "/")
        basename = normalized.rsplit("/", 1)[-1]
        if normalized not in request_text and basename not in request_text:
            return False
    return True


def _score_update_path(path: str, keywords: list[str]) -> int:
    normalized = path.lower()
    basename = os.path.basename(normalized)
    score = 0

    for keyword in keywords:
        if keyword == normalized:
            score += 12
        if keyword in basename:
            score += 8
        elif keyword in normalized:
            score += 4

    priority_names = (
        "package.json", "requirements.txt", "readme.md", "server.py", "app.py",
        "main.py", "index.html", "vite.config", "tsconfig.json",
    )
    if basename in priority_names:
        score += 3
    if normalized.count("/") <= 1:
        score += 1
    return score


def _select_update_paths(filepaths: list[str], update_request: str, max_files: int = _UPDATE_SNAPSHOT_MAX_FILES) -> list[str]:
    if not filepaths:
        return []

    normalized_lookup = {path.replace("\\", "/").lower(): path for path in filepaths}
    selected: list[str] = []
    for request_path in _extract_request_paths(update_request):
        existing_path = normalized_lookup.get(request_path.lower())
        if existing_path and existing_path not in selected:
            selected.append(existing_path)
        if len(selected) >= max_files:
            return selected

    keywords = _request_keywords(update_request)
    scored = sorted(
        filepaths,
        key=lambda path: (_score_update_path(path, keywords), -len(path)),
        reverse=True,
    )
    for path in scored:
        if path in selected:
            continue
        if _score_update_path(path, keywords) > 0:
            selected.append(path)
        if len(selected) >= max_files:
            return selected

    if len(selected) < max_files:
        for fallback in scored:
            if fallback not in selected:
                selected.append(fallback)
            if len(selected) >= max_files:
                break

    return selected


def _build_file_overview(filepaths: list[str], highlighted_paths: list[str], max_chars: int = _UPDATE_FILE_OVERVIEW_CHARS) -> str:
    if not filepaths:
        return "(project is empty - no files found yet)"

    directories = sorted({os.path.dirname(path) or "." for path in filepaths})
    dir_preview = ", ".join(directories[:12])
    lines = [
        f"Total files: {len(filepaths)}",
        f"Directories: {dir_preview or '.'}",
    ]
    if len(directories) > 12:
        lines.append(f"Additional directories omitted: {len(directories) - 12}")
    if highlighted_paths:
        lines.append("Most relevant files:")
        lines.extend(f"- {path}" for path in highlighted_paths)

    preview_paths = [path for path in filepaths if path not in highlighted_paths][:20]
    if preview_paths:
        lines.append("Other files (sample):")
        lines.extend(f"- {path}" for path in preview_paths)
    if len(filepaths) > len(highlighted_paths) + len(preview_paths):
        lines.append(f"... and {len(filepaths) - len(highlighted_paths) - len(preview_paths)} more files")

    return _clip_text("\n".join(lines), max_chars, empty_fallback="(project is empty - no files found yet)")


def live_tool_context(transcript: list[ToolTranscriptEntry]) -> str:
    """Summarizes recent tool activity for the supervisor's context."""
    if not transcript:
        return "No recent tool activity."
    
    # Sort by timestamp (if available) and take last 5
    recent = sorted(transcript, key=lambda x: x.timestamp)[-5:]
    
    lines = ["Recent Tool Activity:"]
    for entry in recent:
        # Truncate output to avoid prompt bloat
        out_preview = str(entry.output)[:200] + "..." if len(str(entry.output)) > 200 else str(entry.output)
        lines.append(f"- {entry.tool_name}({json.dumps(entry.input)}): {entry.status}\n  Output: {out_preview}")
        
    return "\n".join(lines)


def _fallback_debug_result(output: str, last_exec: ExecutionResult | None = None) -> DebugResult:
    error_summary = "Unknown error occurred during debugging."
    if last_exec and last_exec.stderr:
        first_line = next((line.strip() for line in last_exec.stderr.splitlines() if line.strip()), "")
        if first_line:
            error_summary = _clip_text(first_line, 120, empty_fallback=error_summary)

    root_cause = "Unknown root cause in the current project state."
    if last_exec and last_exec.error_type:
        root_cause = f"{last_exec.error_type} was reported, but the debugger could not return a structured diagnosis."

    fix_description = _clip_text(output, 300, empty_fallback="Fix description unavailable after debugger failure.")
    if len(fix_description) < 10:
        fix_description = "Fix description unavailable after debugger failure."

    return DebugResult(
        error_summary=error_summary,
        root_cause=root_cause,
        fix_description=fix_description,
        files_modified=[],
        resolved=False,
    )

def _retry_node(label: str, fn: Callable[[], dict]) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, _NODE_MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            print(f"[{label}] attempt {attempt}/{_NODE_MAX_RETRIES} failed: {exc!r}")
    raise RuntimeError(f"{label} failed after {_NODE_MAX_RETRIES} attempts: {last_error}")

def _check_placeholders(filepath: str, content: str) -> None:
    lower = content.lower()
    hits = [p for p in PLACEHOLDER_PATTERNS if p.lower() in lower]
    if hits:
        print(f"[coder] âš  PLACEHOLDER WARNING in {filepath}: {len(hits)} suspect pattern(s): {hits[:5]}")

def _build_update_project_snapshot(
    config: RunnableConfig,
    *,
    update_request: str = "",
    max_chars_per_file: int = _UPDATE_SNAPSHOT_FILE_CHARS,
    max_total_chars: int = _UPDATE_SNAPSHOT_TOTAL_CHARS,
    max_files: int = _UPDATE_SNAPSHOT_MAX_FILES,
) -> str:
    files_listing = list_files.invoke({"directory": "."}, config=config)
    if not files_listing or files_listing == "No files found.":
        return "(project is empty - no files found yet)"
    if files_listing.startswith("ERROR:"):
        return f"(unable to list project files: {files_listing})"

    filepaths = [line.strip() for line in files_listing.splitlines() if line.strip()]
    selected_paths = _select_update_paths(filepaths, update_request, max_files=max_files)
    parts: list[str] = []
    total_chars = 0

    for filepath in selected_paths:
        try:
            content = read_file.invoke(filepath, config=config)
        except UnicodeDecodeError:
            content = "(binary file omitted)"
        except Exception as exc:
            content = f"(unable to read file: {exc})"

        if len(content) > max_chars_per_file:
            content = content[:max_chars_per_file] + "\n... (truncated)"

        remaining_chars = max_total_chars - total_chars
        if remaining_chars <= 0:
            parts.append("(... remaining files omitted for brevity)")
            break
        if len(content) > remaining_chars:
            content = content[:remaining_chars] + "\n... (truncated)"

        parts.append(f"### {filepath}\n```\n{content}\n```")
        total_chars += len(content)

    omitted = max(0, len(filepaths) - len(selected_paths))
    if omitted:
        parts.append(f"(snapshot limited to {len(selected_paths)} relevant files; {omitted} additional files omitted)")

    return "\n\n".join(parts)

def _repo_adaptation_plan_from_dict(plan_dict: dict[str, Any]) -> RepoAdaptationPlan:
    detected_services = [
        DetectedService(
            service_type=d.get("service_type", ""),
            original_provider=d.get("original_provider", ""),
            files_affected=d.get("files_affected", []),
            replacement=d.get("replacement", ""),
        )
        for d in plan_dict.get("detected_services", [])
        if isinstance(d, dict)
    ]
    return RepoAdaptationPlan(
        repo_url=plan_dict.get("repo_url", ""),
        repo_name=plan_dict.get("repo_name", ""),
        detected_services=detected_services,
        files_to_delete=plan_dict.get("files_to_delete", []),
        adaptation_summary=plan_dict.get("adaptation_summary", ""),
        framework=plan_dict.get("framework", ""),
        dependencies=plan_dict.get("dependencies", []),
        env_vars=plan_dict.get("env_vars", []),
        key_files=plan_dict.get("key_files", []),
        external_connections=plan_dict.get("external_connections", []),
    )

def _selected_repo_payload(repo: dict | None) -> dict[str, Any]:
    if not repo:
        return {}
    return {
        "full_name": repo.get("full_name", ""),
        "html_url": repo.get("html_url", ""),
        "default_branch": repo.get("default_branch", "main"),
        "description": repo.get("description") or "",
        "language": repo.get("language") or "",
        "stargazers_count": repo.get("stargazers_count", 0),
        "pushed_at": repo.get("pushed_at", ""),
    }

def _github_repo_found(state: PipelineState) -> bool:
    repo_plan = state.get("repo_adaptation_plan")
    selected_repo = state.get("github_selected_repo") or {}
    return bool(
        (repo_plan and getattr(repo_plan, "repo_name", ""))
        or selected_repo.get("full_name")
    )

def _planner_github_context(state: PipelineState) -> str:
    repo_plan = state.get("repo_adaptation_plan")
    selected_repo = state.get("github_selected_repo") or {}

    if not _github_repo_found(state):
        summary = getattr(repo_plan, "adaptation_summary", "") if repo_plan else ""
        return summary or "No matching GitHub repository was selected. Build the project from scratch."

    lines = [
        f"Repo: {selected_repo.get('full_name') or getattr(repo_plan, 'repo_name', '')}",
        f"URL: {selected_repo.get('html_url') or getattr(repo_plan, 'repo_url', '')}",
        f"Description: {selected_repo.get('description', '') or '(none)'}",
        f"Language: {selected_repo.get('language', '') or '(unknown)'}",
        f"Stars: {selected_repo.get('stargazers_count', 0)}",
    ]
    if repo_plan:
        if repo_plan.framework:
            lines.append(f"Detected framework: {repo_plan.framework}")
        if repo_plan.adaptation_summary:
            lines.append(f"Scout summary: {repo_plan.adaptation_summary}")
        if repo_plan.dependencies:
            lines.append("Dependency/script signals:")
            lines.extend(f"- {item}" for item in repo_plan.dependencies[:20])
        if repo_plan.env_vars:
            lines.append("Environment variables referenced:")
            lines.extend(f"- {item}" for item in repo_plan.env_vars[:30])
        if repo_plan.key_files:
            lines.append("Key files:")
            lines.extend(f"- {item}" for item in repo_plan.key_files[:30])
        if repo_plan.external_connections:
            lines.append("External connections to account for:")
            lines.extend(f"- {item}" for item in repo_plan.external_connections[:20])
        elif repo_plan.detected_services:
            lines.append("Detected services to adapt:")
            for svc in repo_plan.detected_services:
                lines.append(
                    f"- {svc.service_type}: {svc.original_provider} -> {svc.replacement} "
                    f"({', '.join(svc.files_affected[:5])})"
                )

    return _clip_text("\n".join(lines), 6000)


def clarifier_agent(state: PipelineState, config: RunnableConfig) -> dict:
    if state.get("status") == "delegating_update_coder":
        print("[clarifier] Update run detected; preserving exact user request.")
        return {"status": "delegating_update_coder"}

    if _skip_clarifier_enabled(config):
        print("[clarifier] Fast path enabled; using original request without an extra LLM pass.")
        return {
            "original_user_prompt": state.get("original_user_prompt") or state["user_prompt"],
            "status": "delegating_github_search",
        }

    print(f"[clarifier] Refining request: {state['user_prompt'][:100]}...")
    clarified: ClarifiedRequest | None = _invoke_structured(
        CLARIFIER_LLM,
        clarifier_prompt,
        {"user_prompt": state["user_prompt"]},
        ClarifiedRequest,
        "clarifier",
        config,
    )
    if clarified:
        print(f"[clarifier] âœ“ Refined: {clarified.refined_prompt[:100]}...")
        return {
            "original_user_prompt": state.get("original_user_prompt") or state["user_prompt"],
            "user_prompt": clarified.refined_prompt,
            "status": "delegating_github_search",
        }
    return {
        "original_user_prompt": state.get("original_user_prompt") or state["user_prompt"],
        "status": "delegating_github_search",
    }

def github_search_agent(state: PipelineState, config: RunnableConfig) -> dict:
    """
    Pre-planning scout. It searches GitHub using the original user wording,
    analyzes the selected repo in a temporary folder, then gives planner a
    clean fork in the road: adapt this repo, or build from scratch.
    """
    search_prompt = state.get("original_user_prompt") or state.get("user_prompt", "")
    if not _github_scout_enabled(config):
        empty_plan = RepoAdaptationPlan(
            repo_url="",
            repo_name="",
            detected_services=[],
            files_to_delete=[],
            adaptation_summary="GitHub scouting is disabled; planner should build from scratch.",
            framework="",
        )
        transcript_update = _log_tool(
            state,
            "github_search",
            {"prompt": search_prompt, "enabled": False},
            empty_plan.adaptation_summary,
        )
        return {
            "repo_adaptation_plan": empty_plan,
            "github_selected_repo": {},
            "github_search_completed": True,
            "status": "delegating_planner",
            **transcript_update,
        }

    from codemind.services.github_scout import (
        search_github_repos, rank_and_select_repo,
        download_repo_zip, analyze_repo_files, generate_adaptation_plan,
    )

    print(f"[github_search] Searching GitHub before planning: {search_prompt[:100]}...")
    repos = search_github_repos(search_prompt, "", max_results=5)
    best = rank_and_select_repo(repos, search_prompt)

    if best is None:
        empty_plan = RepoAdaptationPlan(
            repo_url="",
            repo_name="",
            detected_services=[],
            files_to_delete=[],
            adaptation_summary="No matching GitHub repo found before planning; planner should build from scratch.",
            framework="",
        )
        transcript_update = _log_tool(
            state,
            "github_search",
            {"prompt": search_prompt},
            empty_plan.adaptation_summary,
        )
        return {
            "repo_adaptation_plan": empty_plan,
            "github_selected_repo": {},
            "github_search_completed": True,
            "status": "delegating_planner",
            **transcript_update,
        }

    selected_repo = _selected_repo_payload(best)
    plan_dict = {
        "repo_url": best.get("html_url", ""),
        "repo_name": best.get("full_name", ""),
        "detected_services": [],
        "files_to_delete": [],
        "adaptation_summary": (
            f"Selected '{best.get('full_name', '')}' from GitHub before planning. "
            "Temporary analysis could not be completed, so planner should rely on repo metadata."
        ),
        "framework": "",
        "dependencies": [],
        "env_vars": [],
        "key_files": [],
        "external_connections": [],
    }

    with tempfile.TemporaryDirectory(prefix="github-scout-") as tmpdir:
        tmp_root = Path(tmpdir)
        if download_repo_zip(best, tmp_root):
            analysis = analyze_repo_files(tmp_root)
            plan_dict = generate_adaptation_plan(analysis, best)
        else:
            print("[github_search] Temporary repo download failed; planner will use repo metadata only.")

    adaptation_plan = _repo_adaptation_plan_from_dict(plan_dict)
    print(f"[github_search] Selected repo context ready: {adaptation_plan.repo_name or selected_repo.get('full_name')}")
    transcript_update = _log_tool(
        state,
        "github_search",
        {"prompt": search_prompt, "repo": selected_repo.get("full_name", "")},
        adaptation_plan.adaptation_summary,
    )
    return {
        "repo_adaptation_plan": adaptation_plan,
        "github_selected_repo": selected_repo,
        "github_search_completed": True,
        "status": "delegating_github_scout",
        **transcript_update,
    }

def planner_agent(state: PipelineState, config: RunnableConfig) -> dict:
    github_context = _planner_github_context(state)
    repo_found = _github_repo_found(state)
    prompt_fn = planner_existing_repo_prompt if repo_found else planner_scratch_prompt
    prompt_kwargs = {"user_prompt": state["user_prompt"], "github_context": github_context}
    mode = "existing GitHub repo" if repo_found else "from scratch"
    print(f"[planner] Thinking about: {state['user_prompt'][:100]}... ({mode})")
    plan: Plan | None = _invoke_structured(
        PLAN_LLM,
        prompt_fn,
        prompt_kwargs,
        Plan,
        "planner",
        config,
    )
    if plan is None: raise ValueError("[planner] failed to produce a valid Plan.")
    write_file.invoke({"path": "PRD.md", "content": plan.model_dump_json(indent=2)}, config=config)
    print(f"[planner] âœ“ {plan.name} | {len(plan.files)} files | {plan.techstack}")
    return {"plan": plan, "status": "planning_architecture"}

def _fallback_task_plan_from_plan(plan: Plan) -> TaskPlan:
    """Build a usable task plan from the planner output when architect parsing fails."""
    feature_summary = "; ".join(str(feature) for feature in (getattr(plan, "features", []) or []))
    if not feature_summary:
        feature_summary = getattr(plan, "description", "the requested application")

    template_rewrite_paths = {
        "index.html",
        "package.json",
        "src/App.jsx",
        "src/App.tsx",
        "src/App.js",
        "src/main.jsx",
        "src/main.tsx",
        "src/main.js",
        "src/index.css",
        "src/App.css",
        "tailwind.config.js",
        "postcss.config.js",
        "vite.config.js",
    }
    implemented_paths: list[str] = []
    tasks: list[ImplementationTask] = []

    for planned_file in getattr(plan, "files", []) or []:
        filepath = str(getattr(planned_file, "path", "")).strip().replace("\\", "/")
        if not filepath:
            continue

        purpose = str(getattr(planned_file, "purpose", "") or "Implement this project file").strip()
        action = "rewrite" if filepath in template_rewrite_paths else "create"
        task_description = (
            f"Implement {filepath}. Purpose: {purpose}. "
            f"Integrate it into {plan.name} using {plan.techstack}. "
            f"Support these user-facing features: {feature_summary}. "
            "Use complete production-ready code, wire imports to the exact relative paths in this plan, "
            "and do not leave placeholders, TODOs, or unfinished sections."
        )
        tasks.append(
            ImplementationTask(
                filepath=filepath,
                action=action,
                task_description=task_description,
                dependencies=list(implemented_paths),
            )
        )
        implemented_paths.append(filepath)

    if not tasks:
        tasks.append(
            ImplementationTask(
                filepath="src/App.jsx",
                action="rewrite",
                task_description=(
                    f"Implement the complete {plan.name} React application in src/App.jsx. "
                    f"Use {plan.techstack}, include all requested features ({feature_summary}), "
                    "and provide complete runnable code with no placeholders."
                ),
                dependencies=[],
            )
        )

    return TaskPlan(implementation_steps=tasks)


def architect_agent(state: PipelineState, config: RunnableConfig) -> dict:
    task_plan: TaskPlan | None = _invoke_structured(
        ARCHITECT_LLM,
        architect_prompt,
        {"plan_json": state["plan"].model_dump_json(indent=2)},
        TaskPlan,
        "architect",
        config,
    )
    if task_plan is None:
        task_plan = _fallback_task_plan_from_plan(state["plan"])
        print("[architect] Structured output failed; using planner-file fallback task plan.")
    write_file.invoke({"path": "Architecture.md", "content": task_plan.model_dump_json(indent=2)}, config=config)
    print(f"[architect] âœ“ {len(task_plan.implementation_steps)} tasks")
    return {"task_plan": task_plan, "status": "delegating_github_scout"}

def _template_name_for_plan(plan: Plan | None) -> str | None:
    if not plan:
        return None
    project_type = (getattr(plan, "project_type", "") or "").lower().strip()
    # Exact template id / alias wins over keyword heuristics. Without this,
    # a plan with project_type="react-vite" can fall through to an unrelated
    # template (e.g. python-tkinter-sqlite) when the description mentions
    # "desktop", producing a hybrid project that cannot build.
    template = resolve_template(project_type)
    if template:
        return template.template_id

    template = resolve_template_from_plan(plan)
    if template:
        return template.template_id

    print(
        f"[_template_name_for_plan] WARNING: could not resolve a template. "
        f"project_type={project_type!r} did not match any known value. "
        f"Generation will proceed without a starter template."
    )
    return None

def _normalized_scout_framework(framework: str | None) -> str:
    normalized = str(framework or "").strip().lower()
    return "" if normalized in {"", "unknown", "none", "n/a"} else normalized

def _scout_framework_matches_plan(plan: Plan | None, framework: str | None) -> bool:
    """Only accept a downloaded repo as the base when its framework matches the planned scaffold."""
    detected = _normalized_scout_framework(framework)
    if not detected:
        return False

    template_name = _template_name_for_plan(plan)
    if template_name:
        template = resolve_template(template_name)
        if template:
            if "react" in template_name or "vite" in template_name:
                return detected in {"react-vite", "vite", "react"}
            if "nextjs" in template_name:
                return detected in {"nextjs", "nextjs-app"}
            if "fastapi" in template_name:
                return detected == "fastapi"

    return True

def _scout_downloaded_repo(state: PipelineState) -> bool:
    """Return True when GitHub Scout accepted a compatible repo as the project base."""
    repo_plan = state.get("repo_adaptation_plan")
    framework = getattr(repo_plan, "framework", "") if repo_plan else ""
    return _scout_framework_matches_plan(state.get("plan"), framework)


def _project_type_from_repo_framework(framework: str) -> str:
    normalized = _normalized_scout_framework(framework)
    if normalized in {"nextjs", "nextjs-app"}:
        return "nextjs"
    if normalized in {"react", "react-vite", "vite"}:
        return "react"
    if normalized in {"fastapi", "flask"}:
        return "fastapi"
    if normalized in {"express", "node", "node-api"}:
        return "node-api"
    return "fullstack"


def _repo_plan_from_state(state: PipelineState) -> Plan:
    repo_plan = state.get("repo_adaptation_plan")
    selected_repo = state.get("github_selected_repo") or {}
    repo_name = (
        getattr(repo_plan, "repo_name", "") if repo_plan else ""
    ) or selected_repo.get("full_name", "")
    description = (
        selected_repo.get("description")
        or (getattr(repo_plan, "adaptation_summary", "") if repo_plan else "")
        or state.get("original_user_prompt")
        or state.get("user_prompt")
        or "Adapted GitHub project"
    )
    framework = getattr(repo_plan, "framework", "") if repo_plan else ""
    services = [svc.service_type for svc in (getattr(repo_plan, "detected_services", []) if repo_plan else [])]
    features = ["Existing GitHub project adaptation"]
    if "database" in services:
        features.append("Supabase database integration")
    if "auth" in services:
        features.append("Clerk authentication")
    if "storage" in services:
        features.append("Supabase Storage integration")
    if "email" in services:
        features.append("Resend transactional email")
    if "payments" in services:
        features.append("Stripe payments")
    if len(features) == 1:
        features.append("Repository verification")

    key_files = list(getattr(repo_plan, "key_files", []) if repo_plan else [])[:12]
    if not key_files:
        key_files = ["package.json", "requirements.txt", "README.md"]
    files = [File(path=path, purpose="Existing repository file") for path in key_files]

    title = repo_name.split("/")[-1].replace("-", " ").replace("_", " ").title() if repo_name else _project_title_from_state(state)
    tech_parts = [framework] if framework else []
    tech_parts.extend(getattr(repo_plan, "dependencies", [])[:8] if repo_plan else [])
    techstack = ", ".join(part for part in tech_parts if part) or "Existing GitHub repository"

    return Plan(
        name=title,
        description=str(description)[:180] or "Adapted GitHub project",
        techstack=techstack,
        project_type=_project_type_from_repo_framework(framework),
        features=features,
        files=files,
    )


def _required_packages_for_repo_services(state: PipelineState, root: Path) -> list[str]:
    repo_plan = state.get("repo_adaptation_plan")
    if not repo_plan:
        return []

    is_next = (root / "next.config.js").exists() or (root / "next.config.mjs").exists()
    has_package_json = (root / "package.json").exists()
    packages: list[str] = []

    def add(package: str) -> None:
        if package not in packages:
            packages.append(package)

    for service in repo_plan.detected_services:
        if service.service_type in {"database", "storage"} and has_package_json:
            add("@supabase/supabase-js@^2")
        elif service.service_type == "auth" and has_package_json:
            add("@clerk/nextjs@^6" if is_next else "@clerk/clerk-react@^5")
        elif service.service_type == "email" and has_package_json:
            add("resend@^4")
        elif service.service_type == "payments" and has_package_json:
            add("stripe@^17")
    return packages


def _ensure_repo_env_example(root: Path, state: PipelineState) -> None:
    repo_plan = state.get("repo_adaptation_plan")
    if not repo_plan:
        return

    env_lines = {
        "VITE_SUPABASE_URL=https://your-project.supabase.co",
        "VITE_SUPABASE_ANON_KEY=your-supabase-anon-key",
        "SUPABASE_URL=https://your-project.supabase.co",
        "SUPABASE_KEY=your-service-role-or-anon-key",
    }
    for service in repo_plan.detected_services:
        if service.service_type == "auth":
            env_lines.add("VITE_CLERK_PUBLISHABLE_KEY=pk_test_your-clerk-key")
            env_lines.add("CLERK_SECRET_KEY=sk_test_your-clerk-secret")
        elif service.service_type == "email":
            env_lines.add("RESEND_API_KEY=re_your-resend-key")
            env_lines.add("EMAIL_FROM=Attendance <attendance@example.com>")
        elif service.service_type == "payments":
            env_lines.add("VITE_STRIPE_PUBLISHABLE_KEY=pk_test_your-stripe-key")
            env_lines.add("STRIPE_SECRET_KEY=sk_test_your-stripe-secret")
            env_lines.add("STRIPE_WEBHOOK_SECRET=whsec_your-webhook-secret")

    env_path = root / ".env.example"
    existing = env_path.read_text(encoding="utf-8", errors="ignore") if env_path.exists() else ""
    additions = [line for line in sorted(env_lines) if line.split("=", 1)[0] not in existing]
    if additions:
        prefix = existing.rstrip() + "\n" if existing.strip() else ""
        env_path.write_text(prefix + "\n".join(additions) + "\n", encoding="utf-8")

def _fallback_requirements_spec(plan: Plan, root: Path, template_name: str | None) -> RequirementsSpec:
    """Deterministic package fallback when structured requirements generation fails."""
    techstack = getattr(plan, "techstack", "").lower()
    features = " ".join(getattr(plan, "features", []) or []).lower()
    text = f"{techstack} {features}"
    has_package_json = (root / "package.json").exists()
    
    template = resolve_template(template_name) if template_name else None
    
    is_node = has_package_json or (template and is_frontend_template(template.template_id)) or any(
        token in text for token in ("react", "vite", "node", "express", "javascript", "typescript")
    )
    if not is_node:
        pip_packages: list[str] = []
        if "streamlit" in text or template_name == "streamlit-app":
            pip_packages.append("streamlit")
        if "fastapi" in text or template_name in {"fastapi", "fastapi-sqlite"}:
            pip_packages.extend(["fastapi", "uvicorn"])
        if "flask" in text:
            pip_packages.append("flask")
        if "supabase" in text:
            pip_packages.append("supabase")
        if "resend" in text or "email" in text:
            pip_packages.append("resend")
        return RequirementsSpec(package_manager="pip", packages=list(dict.fromkeys(pip_packages)))

    packages: list[str] = []

    def add(package: str) -> None:
        if package not in packages:
            packages.append(package)

    if "react" in text or template_name in {"react-vite", "nextjs-app"}:
        add("react@^18.3.1")
        add("react-dom@^18.3.1")
        add("react-router-dom@^6.30.0")
    if "vite" in text or template_name == "react-vite":
        add("vite@^5.4.0")
        add("@vitejs/plugin-react@^4.3.4")
    if "tailwind" in text:
        add("tailwindcss@^3.4.19")
        add("postcss@^8.5.15")
        add("autoprefixer@^10.5.0")
    if "lucide" in text:
        add("lucide-react@^0.511.0")
    if "express" in text or "node" in text:
        add("express@^4.18.2")
        add("cors@^2.8.5")
        add("dotenv@^16.4.5")
    if "supabase" in text:
        add("@supabase/supabase-js@^2")
    if "clerk" in text:
        add("@clerk/clerk-react@^5")
    if "resend" in text or "email" in text:
        add("resend@^4")
    if "face" in text or "facial" in text or "recognition" in text or "biometric" in text:
        add("face-api.js@^0.22.2")
    if "stripe" in text or "payment" in text:
        add("stripe@^17")

    return RequirementsSpec(package_manager="npm", packages=packages)


def _npm_dependency_spec_parts(spec: str) -> tuple[str, str]:
    spec = spec.strip()
    if not spec:
        return "", ""
    if spec.startswith("@"):
        version_at = spec.find("@", 1)
        if version_at == -1:
            return spec, "latest"
        return spec[:version_at], spec[version_at + 1 :] or "latest"
    if "@" in spec:
        name, version = spec.split("@", 1)
        return name.strip(), version.strip() or "latest"
    return spec, "latest"


def _merge_package_json_dependencies(root: Path, packages: list[str]) -> None:
    package_path = root / "package.json"
    package_data = _load_json_file(package_path)
    if not package_data:
        return

    dependencies = package_data.get("dependencies")
    if not isinstance(dependencies, dict):
        dependencies = {}
        package_data["dependencies"] = dependencies

    dev_dependencies = package_data.get("devDependencies")
    if not isinstance(dev_dependencies, dict):
        dev_dependencies = {}
        package_data["devDependencies"] = dev_dependencies

    dev_names = {"vite", "@vitejs/plugin-react", "typescript", "eslint", "tailwindcss", "postcss", "autoprefixer"}
    changed = False
    for spec in packages:
        name, version = _npm_dependency_spec_parts(spec)
        if not name:
            continue
        target = dev_dependencies if name in dev_names or name.startswith("@types/") else dependencies
        other = dependencies if target is dev_dependencies else dev_dependencies
        if name in target or name in other:
            continue
        target[name] = version
        changed = True

    if changed:
        package_path.write_text(json.dumps(package_data, indent=2) + "\n", encoding="utf-8")
        print(f"[requirements] Merged npm dependencies into package.json: {packages}")


def _expected_package_manager(plan: Plan, root: Path, template_name: str | None) -> str:
    project_type = getattr(plan, "project_type", "").lower()
    techstack = getattr(plan, "techstack", "").lower()
    text = f"{project_type} {techstack}"
    
    template = resolve_template(template_name) if template_name else None
    if (root / "package.json").exists() or (template and is_frontend_template(template.template_id)):
        return "npm"
    if any(token in text for token in ("react", "vite", "next", "node", "express", "javascript", "typescript", "fullstack")):
        return "npm"
    return "pip"


def _plan_uses_tailwind(plan: Plan) -> bool:
    parts: list[str] = [
        getattr(plan, "project_type", ""),
        getattr(plan, "techstack", ""),
        getattr(plan, "description", ""),
        " ".join(getattr(plan, "features", []) or []),
    ]
    for file_info in getattr(plan, "files", []) or []:
        parts.append(getattr(file_info, "path", ""))
        parts.append(getattr(file_info, "purpose", ""))
    return "tailwind" in " ".join(parts).lower()


def _write_if_missing(path: Path, content: str) -> bool:
    if path.exists():
        return False
    path.write_text(content, encoding="utf-8")
    return True


def _ensure_tailwind_setup(root: Path, plan: Plan) -> None:
    if not _plan_uses_tailwind(plan):
        return

    package_dirs = [path.parent for path in _iter_project_file_paths(root, "package.json")]
    if not package_dirs and (root / "package.json").exists():
        package_dirs = [root]

    for package_dir in package_dirs:
        _merge_package_json_dependencies(
            package_dir,
            [
                "tailwindcss@^3.4.19",
                "postcss@^8.5.15",
                "autoprefixer@^10.5.0",
            ],
        )
        wrote_tailwind = _write_if_missing(
            package_dir / "tailwind.config.js",
            "/** @type {import('tailwindcss').Config} */\n"
            "export default {\n"
            "  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],\n"
            "  theme: {\n"
            "    extend: {},\n"
            "  },\n"
            "  plugins: [],\n"
            "}\n",
        )
        wrote_postcss = _write_if_missing(
            package_dir / "postcss.config.js",
            "export default {\n"
            "  plugins: {\n"
            "    tailwindcss: {},\n"
            "    autoprefixer: {},\n"
            "  },\n"
            "}\n",
        )
        if wrote_tailwind or wrote_postcss:
            rel = _rel_dir(package_dir, root)
            print(f"[requirements] Ensured Tailwind config in {rel}")


def _plan_needs_provider_env(plan: Plan) -> bool:
    text = " ".join(
        str(part)
        for part in [
            getattr(plan, "project_type", ""),
            getattr(plan, "techstack", ""),
            getattr(plan, "description", ""),
            " ".join(getattr(plan, "features", []) or []),
        ]
    ).lower()
    if any(
        phrase in text
        for phrase in (
            "no backend",
            "no database",
            "no external api",
            "no external apis",
            "localstorage",
            "local storage only",
        )
    ):
        return False
    return any(
        token in text
        for token in (
            "api",
            "auth",
            "backend",
            "clerk",
            "database",
            "login",
            "signup",
            "file storage",
            "supabase",
            "upload",
        )
    )


def _runtime_health_check_needed(root: Path) -> bool:
    run_plan = build_run_plan(root)
    commands = run_plan.get("commands") or []
    if not commands:
        return False

    names = " ".join(str(command.get("name", "")).lower() for command in commands)
    return "backend" in names or "server" in names


def requirements_agent(state: PipelineState, config: RunnableConfig) -> dict:
    from codemind.services.project_runner import ProjectRunner, validate_dependency, copy_template
    from codemind.pipeline.agent_tools import _get_project_root
    
    plan = state["plan"]
    task_plan = state["task_plan"]
    coder_state = state.get("coder_state") or CoderState(task_plan=task_plan, current_step_idx=0)
    
    root = _get_project_root(config)
    runner = ProjectRunner(root)
    
    template_name = _template_name_for_plan(plan)
        
    scout_repo_available = _scout_downloaded_repo(state)
    template_copied = False
    if template_name and not scout_repo_available:
        template_copied = bool(copy_template(template_name, root))
        if template_copied:
            print(f"[requirements] Copied template: {template_name}")
        else:
            print(f"[requirements] Template {template_name} was unavailable; continuing without template copy")
        if _plan_needs_provider_env(plan):
            runner.provision_backend(plan.name)
        else:
            print("[requirements] Skipping provider env provisioning; plan does not require managed services")
        
        # Auto-install dependencies after template copy
        pkg_json = root / "package.json"
        req_txt = root / "requirements.txt"
        if pkg_json.exists():
            import subprocess
            print("[requirements] Running npm install...")
            result = subprocess.run(
                ["npm", "install"],
                cwd=str(root),
                capture_output=True, text=True, timeout=600,
                shell=True,
            )
            if result.returncode == 0:
                print(f"[requirements] npm install succeeded")
            else:
                print(f"[requirements] npm install failed: {result.stderr[:500]}")
        elif req_txt.exists():
            import subprocess
            print("[requirements] Running pip install...")
            subprocess.run(
                ["pip", "install", "-r", "requirements.txt"],
                cwd=str(root),
                capture_output=True, text=True, timeout=600,
            )
    elif template_name and scout_repo_available:
        scout_plan = state.get("repo_adaptation_plan")
        repo_name = getattr(scout_plan, "repo_name", "") or "downloaded repo"
        print(f"[requirements] GitHub Scout found {repo_name}; skipping {template_name} template copy")
        
    tasks_block = "\n".join(f"  [{i+1}] {t.filepath}: {t.task_description[:200]}" for i, t in enumerate(task_plan.implementation_steps))
    
    if _requirements_llm_enabled(config):
        search_prompt = "You are a RESEARCH agent. Suggest standard pip/npm packages for the features required. Only output package names."
        search_results = _run_agent_with_model_fallback(
            RETRIEVER_LLM,
            [],
            search_prompt,
            f"Suggest exact package names needed for: {plan.techstack}. Features: {plan.features}",
            label="requirements_search",
            config=config,
        )
        
        spec: RequirementsSpec | None = _invoke_structured(
            RETRIEVER_LLM,
            requirements_prompt,
            {"name": plan.name, "techstack": plan.techstack, "tasks_block": tasks_block, "search_results": search_results},
            RequirementsSpec,
            "requirements",
            config,
        )
    else:
        spec = _fallback_requirements_spec(plan, root, template_name)
        print(f"[requirements] Using deterministic fallback dependencies (fast path): {spec.package_manager} {spec.packages}")

    if spec is None:
        spec = _fallback_requirements_spec(plan, root, template_name)
        print(f"[requirements] Using deterministic fallback dependencies: {spec.package_manager} {spec.packages}")
    else:
        expected_package_manager = _expected_package_manager(plan, root, template_name)
        if spec.package_manager.lower() != expected_package_manager:
            print(
                f"[requirements] Overriding mismatched package manager "
                f"{spec.package_manager!r} -> {expected_package_manager!r} for {plan.techstack}"
            )
            spec = _fallback_requirements_spec(plan, root, template_name)
            print(f"[requirements] Using deterministic fallback dependencies: {spec.package_manager} {spec.packages}")
    
    # Controlled fallback validation
    valid_packages = []
    # Always include predefined features map if we can guess them
    for feature in plan.features:
        feature_lower = feature.lower()
        for key in runner.resolve_dependencies(feature_lower, spec.package_manager.lower()):
            if key not in valid_packages:
                valid_packages.append(key)

    for pkg in spec.packages:
        if pkg not in valid_packages:
            if validate_dependency(pkg, spec.package_manager.lower()):
                valid_packages.append(pkg)
            else:
                print(f"[requirements] Rejected suspicious or unknown dependency: {pkg}")
    spec.packages = valid_packages

    if spec.package_manager.lower() == "npm":
        _merge_package_json_dependencies(root, spec.packages)
        if spec.packages:
            for task in coder_state.task_plan.implementation_steps:
                if "package.json" in task.filepath.lower():
                    task.task_description += f"\n\nCRITICAL: You MUST include the following dependency specs in the package.json:\n" + "\n".join(spec.packages)
                    break
    else:
        req_content = "\n".join(spec.packages) + ("\n" if spec.packages else "")
        if req_content.strip():
            req_path = root / "requirements.txt"
            if req_path.exists():
                existing = req_path.read_text(encoding="utf-8")
                req_path.write_text(existing + "\n" + req_content, encoding="utf-8")
            else:
                write_file.invoke({"path": "requirements.txt", "content": req_content}, config=config)

    _ensure_tailwind_setup(root, plan)
    
    project_type = getattr(plan, "project_type", "").lower()
    is_frontend = is_frontend_template(project_type)
    existing_filepaths = {t.filepath.replace("\\", "/") for t in coder_state.task_plan.implementation_steps}

    if is_frontend and template_copied:
        # For frontend projects: rewrite the exact component imported by src/main.*.
        # The template renders App.jsx, while planners often invent App.tsx; if we
        # harden the invented file, the browser still shows the untouched starter UI.
        ordered_entry_files = _frontend_entry_files(config)
        entry_files = set(ordered_entry_files)
        target_entry = next(
            (rel_path for rel_path in ordered_entry_files if (root / rel_path).exists()),
            "src/App.jsx",
        )
        if target_entry not in existing_filepaths:
            # Build a description from the plan features
            features_list = ", ".join(plan.features[:8]) if plan.features else "the requested features"
            all_component_files = [
                t.filepath.replace("\\", "/")
                for t in coder_state.task_plan.implementation_steps
                if t.filepath.replace("\\", "/").startswith("src/")
                and t.filepath.replace("\\", "/") not in (entry_files | set(_FRONTEND_BOOTSTRAP_FILES))
                and t.filepath.replace("\\", "/").endswith((".jsx", ".tsx"))
                and any(part in t.filepath.replace("\\", "/") for part in ("/components/", "/pages/", "/layouts/"))
            ]
            imports_hint = "\n".join(f"- Import and render: {f}" for f in all_component_files) if all_component_files else ""
            
            entry_task = ImplementationTask(
                filepath=target_entry,
                action="rewrite",
                task_description=(
                    f"REWRITE this file completely. The current content is Vite boilerplate and MUST be replaced.\n"
                    f"Build the main application UI that implements: {features_list}.\n"
                    f"{'Import all component files and wire them together:' if imports_hint else ''}\n"
                    f"{imports_hint}\n"
                    f"Set up routing with react-router-dom if multiple pages are needed.\n"
                    f"Remove ALL references to the default Vite template (logos, counter, 'Get started', 'HMR', etc)."
                ),
                dependencies=all_component_files,
            )
            coder_state.task_plan.implementation_steps.append(entry_task)
            print(f"[requirements] Auto-injected {target_entry} rewrite task (architect missed rendered entry)")
        else:
            for task in coder_state.task_plan.implementation_steps:
                normalized_path = task.filepath.replace("\\", "/")
                if normalized_path == target_entry:
                    task.action = "rewrite"
                    if "default Vite template" not in task.task_description:
                        task.task_description += (
                            "\n\nCRITICAL ENTRYPOINT REQUIREMENT: REWRITE this file completely. "
                            "The current content may be Vite boilerplate and MUST be replaced with "
                            "the requested application UI. Remove ALL references to the default "
                            "Vite template, including logos, counter, 'Get started', and 'HMR'."
                        )
                    print(f"[requirements] Hardened {normalized_path} as required frontend entry rewrite")
                    break
    elif not is_frontend and template_copied:
        # For backend projects: ensure main.py / app.py is in the task plan
        backend_entries = {"main.py", "app.py", "server.py"}
        if not (backend_entries & existing_filepaths):
            entry_task = ImplementationTask(
                filepath="main.py",
                action="rewrite",
                task_description=(
                    f"REWRITE this file completely. Wire up all routes, models, and services.\n"
                    f"Implement: {', '.join(plan.features[:8]) if plan.features else 'the requested features'}."
                ),
                dependencies=[t.filepath for t in coder_state.task_plan.implementation_steps 
                              if t.filepath.endswith((".py",)) and t.filepath != "main.py"],
            )
            coder_state.task_plan.implementation_steps.append(entry_task)
            print(f"[requirements] âœ… Auto-injected main.py rewrite task (architect missed it)")

    if template_copied:
        protected_files = {"src/main.jsx", "src/main.tsx", "vite.config.js", "vite.config.ts", "index.html", "public/index.html"}
        original_count = len(coder_state.task_plan.implementation_steps)
        coder_state.task_plan.implementation_steps = [
            t for t in coder_state.task_plan.implementation_steps
            if t.filepath not in protected_files
        ]
        removed = original_count - len(coder_state.task_plan.implementation_steps)
        if removed:
            print(f"[requirements] ðŸ›¡ Removed {removed} task(s) targeting protected template files: {protected_files & existing_filepaths}")

    env_update = _auto_fill_generated_env(root, "requirements")

    return {"requirements": spec, "coder_state": coder_state, "status": "delegating_coder", **env_update, **_manifest_state_update(config)}


def github_scout_agent(state: PipelineState, config: RunnableConfig) -> dict:
    """
    Downloads the best matching GitHub repo for the user's project prompt,
    analyses it for DB/auth/storage/email/payments patterns, and stores the RepoAdaptationPlan in state.
    Always runs â€” there is no skip/scratch option.
    """
    from codemind.services.github_scout import (
        search_github_repos, rank_and_select_repo,
        download_repo_zip, analyze_repo_files, generate_adaptation_plan,
    )
    from codemind.pipeline.agent_tools import _get_project_root

    plan = state.get("plan")
    user_prompt = state.get("original_user_prompt") or state.get("user_prompt", "")
    project_root = _get_project_root(config)
    existing_repo_plan = state.get("repo_adaptation_plan")
    selected_repo = state.get("github_selected_repo") or {}

    if state.get("github_search_completed") and not _github_repo_found(state):
        print("[github_scout] Pre-planning search found no repo; skipping later GitHub search.")
        empty_plan = existing_repo_plan or RepoAdaptationPlan(
            repo_url="",
            repo_name="",
            detected_services=[],
            files_to_delete=[],
            adaptation_summary="No matching GitHub repo found; project will be generated from scratch.",
            framework="",
        )
        return {"repo_adaptation_plan": empty_plan, "status": "delegating_adapter"}

    print("[github_scout] Preparing GitHub base repo...")
    if selected_repo.get("full_name"):
        print(f"[github_scout] Reusing pre-planning selection: {selected_repo['full_name']}")
        best = {
            "full_name": selected_repo["full_name"],
            "html_url": selected_repo.get("html_url", ""),
            "default_branch": selected_repo.get("default_branch", "main"),
            "description": selected_repo.get("description", ""),
            "language": selected_repo.get("language", ""),
            "stargazers_count": selected_repo.get("stargazers_count", 0),
            "pushed_at": selected_repo.get("pushed_at", ""),
        }
    else:
        tech_stack = getattr(plan, "techstack", "") if plan else ""
        repos = search_github_repos(user_prompt, tech_stack, max_results=5)
        best = rank_and_select_repo(repos, user_prompt)

    if best is None:
        print("[github_scout] âš  No suitable repo found â€” proceeding without a base repo.")
        # Create an empty adaptation plan so adapter_agent has something to work with
        empty_plan = RepoAdaptationPlan(
            repo_url="",
            repo_name="",
            detected_services=[],
            files_to_delete=[],
            adaptation_summary="No matching GitHub repo found; project will be generated from scratch by the coder.",
            framework="",
        )
        return {"repo_adaptation_plan": empty_plan, "status": "delegating_adapter"}

    scout_snapshot = create_project_snapshot(project_root)
    success = download_repo_zip(best, project_root)
    if not success:
        print("[github_scout] âš  Download failed â€” proceeding without a base repo.")
        if scout_snapshot:
            restore_project_snapshot(project_root, scout_snapshot)
        empty_plan = RepoAdaptationPlan(
            repo_url=best.get("html_url", ""),
            repo_name=best.get("full_name", ""),
            detected_services=[],
            files_to_delete=[],
            adaptation_summary="Repo download failed; project will be generated from scratch by the coder.",
            framework="",
        )
        return {"repo_adaptation_plan": empty_plan, "status": "delegating_adapter"}

    analysis = analyze_repo_files(project_root)
    if not _scout_framework_matches_plan(plan, analysis.get("framework", "")):
        framework = analysis.get("framework", "") or "unknown"
        print(
            f"[github_scout] Downloaded repo framework '{framework}' is not compatible "
            f"with planned template '{_template_name_for_plan(plan) or 'none'}'; restoring pre-scout project."
        )
        if scout_snapshot:
            restore_project_snapshot(project_root, scout_snapshot)
        empty_plan = RepoAdaptationPlan(
            repo_url=best.get("html_url", ""),
            repo_name=best.get("full_name", ""),
            detected_services=[],
            files_to_delete=[],
            adaptation_summary=(
                f"Downloaded repo '{best.get('full_name', '')}' was not compatible with the planned stack; "
                "project will be generated from the matching template instead."
            ),
            framework="",
        )
        return {"repo_adaptation_plan": empty_plan, "status": "delegating_adapter"}

    plan_dict = generate_adaptation_plan(analysis, best)
    adaptation_plan = _repo_adaptation_plan_from_dict(plan_dict)

    print(f"[github_scout] âœ… Adaptation plan ready: {adaptation_plan.adaptation_summary}")
    return {"repo_adaptation_plan": adaptation_plan, "status": "delegating_adapter"}


def adapter_agent(state: PipelineState, config: RunnableConfig) -> dict:
    """
    ReAct agent that reads the RepoAdaptationPlan and surgically replaces
    every DB / auth / storage / email / payments provider with the configured managed stack.
    Skips gracefully if the plan has no detections (nothing to swap).
    """
    plan = state.get("repo_adaptation_plan")
    transcript = state.get("tool_transcript", [])
    formatted_transcript = live_tool_context(transcript)

    if plan is None or not plan.detected_services:
        print("[adapter] No services to swap â€” passing through to coder.")
        transcript_update = _log_tool(
            state, "adapter_thought",
            {"request": "no-op"},
            "No detected services. Skipping adaptation.",
        )
        next_status = "delegating_repo_finalize" if _scout_downloaded_repo(state) and not state.get("task_plan") else "delegating_coder"
        return {"status": next_status, **transcript_update}

    # Serialize the plan to a plain dict for the prompt builder
    plan_dict = {
        "repo_url": plan.repo_url,
        "repo_name": plan.repo_name,
        "framework": plan.framework,
        "adaptation_summary": plan.adaptation_summary,
        "files_to_delete": plan.files_to_delete,
        "dependencies": plan.dependencies,
        "env_vars": plan.env_vars,
        "key_files": plan.key_files,
        "external_connections": plan.external_connections,
        "detected_services": [
            {
                "service_type": svc.service_type,
                "original_provider": svc.original_provider,
                "files_affected": svc.files_affected,
                "replacement": svc.replacement,
            }
            for svc in plan.detected_services
        ],
    }

    system  = github_scout_adapter_system_prompt(formatted_transcript)
    user_msg = github_scout_adapter_user_prompt(plan_dict)

    print(f"[adapter] ðŸ”§ Swapping {len(plan.detected_services)} service(s) to the managed stack...")
    output = _run_agent_with_model_fallback(
        UPDATE_CODER_LLM,       
        UPDATE_TOOLS,
        system,
        user_msg,
        label="adapter",
        config=config,
    )

    adapter_failed = str(output).lstrip().startswith("ERROR:")
    transcript_update = _log_tool(
        state,
        "adapter_thought",
        {"request": "swap services"},
        output,
        status="error" if adapter_failed else "success",
    )
    if adapter_failed:
        print("[adapter] Adaptation failed; coder will continue with the recorded adapter error.")
        next_status = "delegating_repo_finalize" if _scout_downloaded_repo(state) and not state.get("task_plan") else "delegating_coder"
        return {"status": next_status, **transcript_update}
    print(f"[adapter] âœ… Adaptation complete.")
    next_status = "delegating_repo_finalize" if _scout_downloaded_repo(state) and not state.get("task_plan") else "delegating_coder"
    return {"status": next_status, **transcript_update}


def repo_finalize_agent(state: PipelineState, config: RunnableConfig) -> dict:
    """Deterministic final prep for a downloaded GitHub repo before verification."""
    root = _resolve_project_root(config)
    repo_plan = state.get("repo_adaptation_plan")
    plan = state.get("plan") or _repo_plan_from_state(state)

    packages = _required_packages_for_repo_services(state, root)
    if packages:
        _merge_package_json_dependencies(root, packages)
    _ensure_repo_env_example(root, state)
    env_update = _auto_fill_generated_env(root, "repo_finalize")

    summary = (
        f"Prepared downloaded repo {getattr(repo_plan, 'repo_name', '') or '(unknown repo)'} "
        "for deterministic verification."
    )
    transcript_update = _log_tool(
        state,
        "repo_finalize",
        {"repo": getattr(repo_plan, "repo_name", "") if repo_plan else ""},
        summary,
    )
    return {"plan": plan, "status": "delegating_verify", **transcript_update, **env_update, **_manifest_state_update(config)}


def retriever_agent(state: PipelineState, config: RunnableConfig) -> dict:
    coder_state = state.get("coder_state")
    steps = coder_state.task_plan.implementation_steps
    idx = coder_state.current_step_idx
    task = steps[idx]
    completed_tasks = [f"[{i+1}] {s.filepath}" for i, s in enumerate(steps[:idx])]
    pending_tasks = [f"[{i+idx+2}] {s.filepath}" for i, s in enumerate(steps[idx+1:])]
    memory_store = state.get("memory_store") or MemoryStore()
    
    ctx: RetrievedContext | None = _invoke_structured(
        RETRIEVER_LLM,
        retriever_prompt,
        {
            "task_description": task.task_description,
            "existing_files": list_files.invoke({"directory": "."}, config=config),
            "memory_context": memory_store.get_context_for(task.filepath),
            "filepath": task.filepath,
            "completed_tasks": completed_tasks,
            "pending_tasks": pending_tasks,
            "progress": f"Step {idx+1} of {len(steps)}",
        },
        RetrievedContext,
        f"retriever:{task.filepath}",
        config,
    )
    
    if ctx is None: ctx = RetrievedContext(task_filepath=task.filepath)
    return {"rag_contexts": [ctx], "coder_state": coder_state}

def coder_agent(state: PipelineState, config: RunnableConfig) -> dict:
    coder_state = state["coder_state"]
    steps = coder_state.task_plan.implementation_steps
    if coder_state.current_step_idx >= len(steps):
        final_status = "delegating_integrator" if _integrator_enabled(config) else "delegating_verify"
        return {"status": final_status}

    task = steps[coder_state.current_step_idx]
    # Use rag_contexts already in state â€” no extra LLM call per file
    rag_ctx = next((r for r in reversed(state.get("rag_contexts", [])) if r.task_filepath == task.filepath), None)
    memory_store = state.get("memory_store") or MemoryStore()
    transcript = state.get("tool_transcript", [])
    formatted_transcript = live_tool_context(transcript)

    print(f"[coder] ðŸ‘¨â€ðŸ’» Implementing {task.filepath} ({coder_state.current_step_idx + 1}/{len(steps)})")
    
    file_structure_list = "\n".join([f"- {s.filepath}" for s in steps])
    enriched_task_description = f"{task.task_description}\n\nPROJECT FILE STRUCTURE (For Imports/Links):\n{file_structure_list}"
    
    user_input = coder_user_prompt(
        enriched_task_description,
        task.filepath,
        read_file.invoke(task.filepath, config=config),
        memory_store.get_context_for(task.filepath),
        rag_ctx.relevant_snippets if rag_ctx else [],
        action=getattr(task, "action", "create")
    )

    
    output = _run_agent_with_model_fallback(
        _coder_model_for_task(task, state.get("plan")),
        WRITE_TOOLS,
        coder_system_prompt(formatted_transcript),
        user_input,
        label=f"coder:{task.filepath}",
        config=config,
    )
    
    transcript_update = _log_tool(state, f"coder_thought:{task.filepath}", {"task": task.task_description}, output)
    
    new_coder_state = coder_state.model_copy(update={"current_step_idx": coder_state.current_step_idx + 1})
    final_status = "delegating_integrator" if _integrator_enabled(config) else "delegating_verify"
    next_status = "delegating_coder" if new_coder_state.current_step_idx < len(steps) else final_status
    updates = {"coder_state": new_coder_state, "status": next_status, **transcript_update}
    
    # Check for placeholders â€” only call memory LLM if the file was actually written
    new_content = read_file.invoke(task.filepath, config=config)
    if new_content and not new_content.startswith("ERROR:"):
        _check_placeholders(task.filepath, new_content)
        # Skip the memory LLM call for trivial config/lock files to save time
        _MEMORY_SKIP_EXTENSIONS = (".json", ".lock", ".toml", ".yaml", ".yml", ".env", ".md", ".txt", ".gitignore")
        _skip_memory = task.filepath.endswith(_MEMORY_SKIP_EXTENSIONS)
        if _memory_summary_enabled(config) and not _skip_memory:
            mem_entry: MemoryEntry | None = _invoke_structured(
                RETRIEVER_LLM,
                memory_prompt,
                {"filepath": task.filepath, "file_content": new_content[:3000]},
                MemoryEntry,
                f"memory:{task.filepath}",
                config,
            )
            if mem_entry:
                memory_store.add(mem_entry)
                updates["memory_store"] = memory_store

    updates.update(_manifest_state_update(config))

    return updates


def _task_dependency_layers(tasks: list[ImplementationTask]) -> list[list[ImplementationTask]]:
    by_path = {task.filepath.replace("\\", "/"): task for task in tasks}
    remaining = dict(by_path)
    completed: set[str] = set()
    layers: list[list[ImplementationTask]] = []

    while remaining:
        ready = [
            task
            for path, task in remaining.items()
            if all(dep.replace("\\", "/") not in by_path or dep.replace("\\", "/") in completed for dep in task.dependencies)
        ]
        if not ready:
            ready = [next(iter(remaining.values()))]
        layers.append(ready)
        for task in ready:
            path = task.filepath.replace("\\", "/")
            completed.add(path)
            remaining.pop(path, None)
    return layers


def _run_parallel_coder_task(
    task: ImplementationTask,
    task_plan: TaskPlan,
    memory_store: MemoryStore,
    transcript: list[ToolTranscriptEntry],
    config: RunnableConfig,
    plan: "Plan | None" = None,
) -> str:
    file_structure_list = "\n".join([f"- {s.filepath}" for s in task_plan.implementation_steps])
    enriched_task_description = f"{task.task_description}\n\nPROJECT FILE STRUCTURE (For Imports/Links):\n{file_structure_list}"
    user_input = coder_user_prompt(
        enriched_task_description,
        task.filepath,
        read_file.invoke(task.filepath, config=config),
        memory_store.get_context_for(task.filepath),
        [],
        action=getattr(task, "action", "create"),
    )
    return _run_agent_with_model_fallback(
        _coder_model_for_task(task, plan),
        WRITE_TOOLS,
        coder_system_prompt(live_tool_context(transcript)),
        user_input,
        label=f"parallel_coder:{task.filepath}",
        config=config,
    )


def parallel_coder_agent(state: PipelineState, config: RunnableConfig) -> dict:
    """Opt-in layered coder fan-out. Default pipeline still uses the safer sequential coder."""
    task_plan = state["task_plan"]
    plan = state.get("plan")
    memory_store = state.get("memory_store") or MemoryStore()
    transcript = state.get("tool_transcript", [])
    layers = _task_dependency_layers(task_plan.implementation_steps)
    outputs: list[str] = []

    for layer_index, layer in enumerate(layers, start=1):
        print(f"[parallel_coder] Running layer {layer_index}/{len(layers)} with {len(layer)} task(s)")
        with ThreadPoolExecutor(max_workers=min(len(layer), int(os.getenv("GRAPH_PARALLEL_CODER_WORKERS", "4")))) as pool:
            futures = [
                pool.submit(_run_parallel_coder_task, task, task_plan, memory_store, transcript, config, plan)
                for task in layer
            ]
            for task, future in zip(layer, futures):
                try:
                    outputs.append(f"{task.filepath}: {future.result()[:400]}")
                except Exception as exc:
                    outputs.append(f"{task.filepath}: ERROR: {exc}")

    coder_state = CoderState(task_plan=task_plan, current_step_idx=len(task_plan.implementation_steps))
    transcript_update = _log_tool(
        state,
        "parallel_coder_thought",
        {"layers": len(layers), "tasks": len(task_plan.implementation_steps)},
        "\n".join(outputs),
    )
    final_status = "delegating_integrator" if _integrator_enabled(config) else "delegating_verify"
    return {"coder_state": coder_state, "status": final_status, **transcript_update, **_manifest_state_update(config)}

def executor_agent(state: PipelineState, config: RunnableConfig) -> dict:
    print("[executor] Running deterministic verification checks...")
    results = _run_project_verification(config, state)
    return {
        "execution_results": results,
        "status": "done" if _results_success(results) else "execution_failed",
    }





    try:
        exec_res = _retry_node("executor", lambda: executor_agent(state, config))
    except Exception as exc:
        fail_exec = ExecutionResult(
            command="executor",
            returncode=1,
            stdout="",
            stderr=f"Executor failed: {exc}",
            success=False,
        )
        report = _test_report_from_execution_results([fail_exec])
        return {"test_report": report, "execution_results": [fail_exec], "status": "execution_failed"}

    exec_results = exec_res["execution_results"]
    report = _test_report_from_execution_results(exec_results)
    status = "done" if _results_success(exec_results) else "execution_failed"
    return {"test_report": report, "execution_results": exec_results, "status": status}


def verify_agent(state: PipelineState, config: RunnableConfig) -> dict:
    """Deterministic verification node; replaces LLM tester/debugger in the normal path."""
    env_update = _auto_fill_generated_env(_resolve_project_root(config), "verify")
    manifest_update = _manifest_state_update(config)
    completion_failure = _completion_guard_failure(state, config)
    if completion_failure is not None:
        results = [completion_failure]
    else:
        try:
            results = _retry_node("executor", lambda: executor_agent(state, config))["execution_results"]
        except Exception as exc:
            results = [
                ExecutionResult(
                    command="executor",
                    returncode=1,
                    stdout="",
                    stderr=f"Executor failed: {exc}",
                    success=False,
                    error_type="ExecutorError",
                )
            ]

    report = _test_report_from_execution_results(results)
    if _results_success(results):
        return {"test_report": report, "execution_results": results, "status": "done", **env_update, **manifest_update}

    attempts = int(state.get("debug_attempts") or 0)
    max_attempts = int(state.get("max_debug_attempts") or _MAX_DEBUG_ATTEMPTS_DEFAULT)
    first_failure = next((result for result in results if not result.success), results[-1])
    if attempts < max_attempts:
        repair_prompt = (
            _completion_guard_repair_prompt(first_failure)
            if _completion_guard_should_use_update_coder(state, first_failure)
            else _verification_failure_repair_prompt(first_failure)
        )
        return {
            "test_report": report,
            "execution_results": results,
            "status": "delegating_update_coder",
            "user_prompt": repair_prompt,
            "debug_attempts": attempts + 1,
            "thinking_rationale": "Deterministic verification failed; routing once to update_coder for a scoped fix.",
            **env_update,
            **manifest_update,
        }

    return {"test_report": report, "execution_results": results, "status": "verification_failed", **env_update, **manifest_update}


def summary_agent(state: PipelineState, config: RunnableConfig) -> dict:
    from codemind.pipeline.agent_tools import _get_project_root
    import subprocess as _sp
    
    root = _get_project_root(config)
    env_update = _auto_fill_generated_env(root, "summary")
    
    # Auto-install dependencies after all code has been written
    pkg_json = root / "package.json"
    req_txt = root / "requirements.txt"
    if pkg_json.exists():
        print("[summary] Running final npm install to ensure all dependencies are ready...")
        result = _sp.run(
            ["npm", "install"], cwd=str(root),
            capture_output=True, text=True, timeout=180, shell=True,
        )
        if result.returncode == 0:
            print("[summary] âœ… npm install succeeded")
        else:
            print(f"[summary] âš  npm install failed: {result.stderr[:500]}")
    elif req_txt.exists():
        print("[summary] Running final pip install...")
        _sp.run(
            ["pip", "install", "-r", "requirements.txt"], cwd=str(root),
            capture_output=True, text=True, timeout=180,
        )
    
    print("\n" + "=" * 50 + "\n  âœ…  PIPELINE COMPLETE\n" + "=" * 50)
    return {"status": "done", **env_update, **_manifest_state_update(config)}

def update_coder_agent(state: PipelineState, config: RunnableConfig) -> dict:
    """Specialized coder for updates â€” reads all existing files, then makes surgical changes."""
    update_request = state["user_prompt"]
    original_filepaths = _list_project_files(config)
    filepaths = sorted(original_filepaths)
    referenced_paths = _extract_request_paths(update_request)
    missing_referenced_paths = [path for path in referenced_paths if path not in original_filepaths]
    relevant_paths = _select_update_paths(filepaths, update_request)
    project_overview = _build_file_overview(filepaths, relevant_paths)
    project_snapshot = _build_update_project_snapshot(config, update_request=update_request)
    transcript = state.get("tool_transcript", [])
    formatted_transcript = live_tool_context(transcript)

    system = (
        "You are the UPDATE CODER agent. Your job is to modify an EXISTING project.\n"
        "Available tools: write_file, read_file, list_files, list_project_tree, tree_graph, get_current_directory, ddg_search, delete_file, replace_file_content, apply_patch_block.\n\n"
        f"RECENT TOOL ACTIVITY:\n{formatted_transcript}\n\n"
        "CRITICAL RULES:\n"
        "1. Use the provided project overview and targeted snapshot as your starting map of the codebase.\n"
        "2. FIRST, call list_project_tree or list_files when you need the full tree, then call read_file on every file you plan to edit.\n"
        "3. Do NOT delete files during normal updates. Words like update, change, replace, fix, redesign, or rebuild are not permission to delete files.\n"
        "4. Only call delete_file for a single file when the user's latest request explicitly names that file/path and asks for that file to be deleted/removed. Include a concrete reason in the delete_file reason argument.\n"
        "5. NEVER call delete_file for '.', the project root, a directory, src/, dist/, public/, node_modules/, or a whole project tree. If a replacement is needed, patch or rewrite the existing files instead.\n"
        "6. Before deleting a named file, read it and check imports/references. If you are not certain deletion is safe, leave the file and explain why.\n"
        "7. Prefer apply_patch_block for surgical edits. Use SEARCH/REPLACE blocks in this format:\n"
        "   <<<<<<< SEARCH\n"
        "   original lines\n"
        "   =======\n"
        "   replacement lines\n"
        "   >>>>>>> REPLACE\n"
        "8. Use replace_file_content only when the exact target block is trivially reliable.\n"
        "9. ONLY use write_file if you are creating a brand new file, or if the file needs a complete rewrite. When using write_file, provide the COMPLETE file content.\n"
        "10. If an existing entry file (like App.jsx or main.py) needs to be modified to connect new components, you MUST modify or completely rewrite it to ensure everything is properly integrated.\n"
        "11. After all changes, confirm what you changed, created, and deleted.\n"
        "12. FRONTEND CONSISTENCY: For Vite React projects, `index.html` MUST stay at the project root (not inside public/) and must point to the actual Vite entry file already in the scaffold, normally `<script type=\"module\" src=\"/src/main.jsx\"></script>`. Do NOT change it to `/src/index.tsx` unless you also intentionally create that matching entry file.\n"
        "13. SERVICE STACK DEFAULTS: If the update involves login/auth, use Clerk only. If it involves a database, use Supabase by default or Neon only when explicitly configured. If it involves file uploads, use Supabase Storage by default or Cloudinary for media-heavy uploads. If it involves email, use Resend. If it involves payments or subscriptions, use Stripe.\n"
        "14. SERVICE SAFETY: Never hardcode secrets. Add env placeholders for Clerk, Supabase/Neon, Cloudinary, Resend, and Stripe as needed, and do not implement custom auth, local-only upload storage, custom SMTP, or custom card handling.\n"
    )

    user_msg = (
        f"The user wants this change applied to the existing project:\n\n"
        f"{_clip_text(update_request, _UPDATE_REQUEST_CLIP_CHARS)}\n\n"
        f"Paths explicitly named by the user:\n"
        f"{_clip_text(chr(10).join(f'- {path}' for path in referenced_paths), 3000, empty_fallback='(none)')}\n\n"
        f"Named paths that do not exist yet and may need to be created:\n"
        f"{_clip_text(chr(10).join(f'- {path}' for path in missing_referenced_paths), 3000, empty_fallback='(none)')}\n\n"
        f"Project overview:\n{project_overview}\n\n"
        f"Targeted project snapshot:\n{project_snapshot}\n\n"
        "Read every existing named path before editing it. Create named missing paths when the request asks for them."
    )

    output = _run_agent_with_model_fallback(
        UPDATE_CODER_LLM,
        UPDATE_TOOLS,
        system,
        user_msg,
        label="update_coder",
        config=config,
    )

    current_filepaths = _list_project_files(config)
    raw_deleted = sorted(original_filepaths - current_filepaths)
    
    ignored_prefixes = ("node_modules/", ".git/", ".venv/", "venv/", "__pycache__/", ".next/", "dist/", "build/")
    deleted_paths = [
        p for p in raw_deleted 
        if not any(p.startswith(prefix) or f"/{prefix}" in p for prefix in ignored_prefixes)
    ]

    if deleted_paths and not _update_deletions_allowed(update_request, deleted_paths):
        rollback_update = _rollback_to_snapshot(state, config)
        preview = ", ".join(deleted_paths[:10])
        if len(deleted_paths) > 10:
            preview += f", ... (+{len(deleted_paths) - 10} more)"
        message = (
            "Unsafe update stopped: the update agent deleted file(s) that were "
            f"not explicitly requested by path: {preview}. "
            f"{rollback_update['rollback_reason']}"
        )
        print(f"[update_coder] {message}")
        raise RuntimeError(message)
    
    transcript_update = _log_tool(state, "update_coder_thought", {"request": update_request}, output)
    
    return {
        "status": "delegating_verify",
        "thinking_rationale": f"Update coder finished: {output[:200]}...",
        **transcript_update,
        **_manifest_state_update(config),
    }

def integrator_agent(state: PipelineState, config: RunnableConfig) -> dict:
    plan = state.get("plan")
    task_plan = state.get("task_plan")
    
    transcript = state.get("tool_transcript", [])
    formatted_transcript = live_tool_context(transcript)
    
    system = integrator_system_prompt(formatted_transcript)
    user_msg = integrator_user_prompt(
        plan.model_dump_json(indent=2) if plan else "{}",
        task_plan.model_dump_json(indent=2) if task_plan else "{}",
        format_generated_manifest(state.get("generated_file_manifest")),
    )
    
    output = _run_agent_with_model_fallback(
        INTEGRATOR_LLM,
        UPDATE_TOOLS,
        system,
        user_msg,
        label="integrator",
        config=config,
    )
    
    transcript_update = _log_tool(state, "integrator_thought", {"request": "integrate features"}, output)
    
    # Dynamic feature and dependency validation
    issues = []
    expected_files = [task.filepath for task in task_plan.implementation_steps] if task_plan else []
    
    files_listing = list_files.invoke({"directory": "."}, config=config)
    existing_files = [line.strip().replace("\\", "/") for line in files_listing.splitlines() if line.strip() and not line.startswith("ERROR:")] if files_listing else []
    
    missing_files = [f for f in expected_files if f not in existing_files]
    if missing_files:
        issues.append(f"Missing files that were planned: {', '.join(missing_files)}")
        
    def is_file_used(filepath: str) -> bool:
        basename = os.path.basename(filepath)
        name = os.path.splitext(basename)[0]
        for f in existing_files:
            if f == filepath: continue
            try:
                content = read_file.invoke(f, config=config)
                if name in content:
                    return True
            except Exception:
                pass
        return False
        
    project_type = getattr(plan, "project_type", "").lower() if plan else ""
    is_frontend = is_frontend_template(project_type)
    
    unused_files = []
    
    for f in expected_files[:6]:
        if f in existing_files and not is_file_used(f):
            is_entry = False
            if is_frontend and any(f.endswith(ext) for ext in ["App.jsx", "App.tsx", "index.js", "index.html", "vite.config.js", "next.config.js", "layout.tsx", "page.tsx"]):
                is_entry = True
            elif not is_frontend and any(f.endswith(ext) for ext in ["main.py", "app.py", "server.js", "index.js"]):
                is_entry = True
            
            if not is_entry:
                unused_files.append(f)
            
    if unused_files:
        issues.append(f"Unused files (not imported anywhere): {', '.join(unused_files)}")
        
    if plan and plan.features:
        for feature in plan.features:
            feature_words = _request_keywords(feature)
            found = False
            for f in existing_files:
                if f.endswith((".py", ".js", ".jsx", ".ts", ".tsx", ".html")):
                    try:
                        content = read_file.invoke(f, config=config).lower()
                        if any(w in content for w in feature_words if len(w) > 3):
                            found = True
                            break
                    except Exception:
                        pass
            if not found:
                issues.append(f"Feature '{feature}' appears unimplemented in code.")
                
    if issues:
        integrator_attempts = (state.get("integrator_attempts") or 0) + 1
        prompt = "Fix integration issues:\n" + "\n".join(f"- {iss}" for iss in issues)
        print(f"[integrator] Validation failed (attempt {integrator_attempts}/{_MAX_INTEGRATOR_ATTEMPTS}). Routing to update_coder. Issues: {issues}")
        if integrator_attempts >= _MAX_INTEGRATOR_ATTEMPTS:
           
            print(f"[integrator] âš  Max integrator attempts ({_MAX_INTEGRATOR_ATTEMPTS}) reached. Breaking loop â†’ tester.")
            return {"status": "delegating_verify", "integrator_attempts": integrator_attempts,
                    "thinking_rationale": f"Integrator gave up after {integrator_attempts} attempts; unresolved: {issues[:2]}", **transcript_update}
        return {"status": "delegating_update_coder", "user_prompt": prompt, "integrator_attempts": integrator_attempts,
                "thinking_rationale": "Validation failed, sending to update_coder", **transcript_update}

    return {"status": "delegating_verify", "integrator_attempts": 0,
            "thinking_rationale": f"Integrator finished successfully: {output[:200]}...", **transcript_update}

def supervisor_agent(state: PipelineState, config: RunnableConfig) -> dict:
    
    state_for_llm = {
        "user_prompt": state.get("user_prompt"),
        "status": state.get("status"),
        "has_plan": bool(state.get("plan")),
        "has_task_plan": bool(state.get("task_plan")),
        "github_search_completed": bool(state.get("github_search_completed")),
        "github_repo_found": _github_repo_found(state),
        "github_context": _planner_github_context(state)[:1200] if state.get("github_search_completed") else "",
        "coder_progress": f"{state['coder_state'].current_step_idx}/{len(state['coder_state'].task_plan.implementation_steps)}" if state.get("coder_state") else "N/A",
        "debug_attempts": state.get("debug_attempts", 0),
        "last_execution_success": state["execution_results"][-1].success if state.get("execution_results") else None,
        "recent_tool_activity": live_tool_context(state.get("tool_transcript", [])),
    }
    
    turn_count = (state.get("supervisor_turns") or 0) + 1
    max_turns = _supervisor_turn_limit(state)
    
    max_debug_attempts = state.get("max_debug_attempts")
    if max_debug_attempts is None:
        max_debug_attempts = _MAX_DEBUG_ATTEMPTS_DEFAULT
    if state.get("debug_attempts", 0) >= max_debug_attempts:
        completion_failure = _completion_guard_failure(state, config)
        if completion_failure is not None:
            if _completion_guard_should_use_update_coder(state, completion_failure):
                return _route_completion_guard_to_update_coder(
                    completion_failure,
                    turn_count=turn_count,
                    rationale="Debug attempts reached the limit, but the remaining issue is a concrete completion-guard repair.",
                )
            return {
                "status": "delegating_coder" if _coder_progress_incomplete(state) else "delegating_update_coder",
                "user_prompt": _completion_guard_repair_prompt(completion_failure),
                "thinking_rationale": "Debug attempts reached the limit, but completion guard still found an incomplete project.",
                "execution_results": [completion_failure],
                "debug_attempts": 0,
                "supervisor_turns": turn_count,
            }
        rollback_update = _rollback_to_snapshot(state, config)
        return {
            "status": "done",
            "thinking_rationale": rollback_update["rollback_reason"],
            "supervisor_turns": turn_count,
            **rollback_update,
        }

    if turn_count >= max_turns:
        if _coder_progress_incomplete(state):
            return {
                "status": "delegating_coder",
                "thinking_rationale": f"Supervisor turn limit reached ({max_turns}), but implementation steps are still pending.",
                "supervisor_turns": turn_count,
            }
        completion_failure = _completion_guard_failure(state, config)
        if _completion_guard_should_use_update_coder(state, completion_failure):
            return _route_completion_guard_to_update_coder(
                completion_failure,
                turn_count=turn_count,
                rationale=f"Supervisor turn limit reached ({max_turns}), but the remaining issue is repairable by update_coder.",
            )
        # Check if we're stuck in an integrator loop at turn limit
        integrator_attempts = state.get("integrator_attempts") or 0
        if integrator_attempts >= _MAX_INTEGRATOR_ATTEMPTS:
            print(f"[supervisor] Integrator loop detected at turn limit. Forcing deterministic verification.")
            return {"status": "delegating_verify", "supervisor_turns": turn_count,
                    "thinking_rationale": "Turn limit + integrator loop -> forwarding to deterministic verification with what exists."}
        return {
            "status": "done",
            "thinking_rationale": f"Supervisor turn limit reached ({max_turns}); stopping to avoid loop.",
            "supervisor_turns": turn_count,
        }

   
    def _has_code_files(cfg: RunnableConfig) -> bool:
        """Return True only if real code files (not just .md) exist in the project."""
        try:
            listing = list_files.invoke({"directory": "."}, cfg)
            if not listing or listing.startswith("ERROR:") or listing == "No files found.":
                return False
            code_exts = {".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".vue", ".svelte"}
            for line in listing.splitlines():
                f = line.strip()
                if f and any(f.endswith(ext) for ext in code_exts):
                   
                    return True
            return False
        except Exception:
            return False

    has_code = _has_code_files(config)

    completion_failure = _completion_guard_failure(state, config)
    if _completion_guard_should_use_update_coder(state, completion_failure):
        return _route_completion_guard_to_update_coder(completion_failure, turn_count=turn_count)

    decision: SupervisorDecision | None = _invoke_structured(
        SUPERVISOR_LLM, 
        supervisor_prompt, 
        {"state_json": json.dumps(state_for_llm, indent=2)}, 
        SupervisorDecision, 
        "supervisor",
        config,
    )
    
    if not decision:
       
        if not state.get("plan") and not state.get("github_search_completed"):
            return {"status": "delegating_github_search", "supervisor_turns": turn_count}
        if not state.get("plan"):
            return {"status": "delegating_planner", "supervisor_turns": turn_count}
        if not state.get("task_plan"):
            return {"status": "delegating_architect", "supervisor_turns": turn_count}
        if _coder_progress_incomplete(state):
            print(f"[supervisor] LLM fallback: Coder has pending implementation steps, routing to coder")
            return {"status": "delegating_coder", "supervisor_turns": turn_count}
        completion_failure = _completion_guard_failure(state, config)
        if _completion_guard_should_use_update_coder(state, completion_failure):
            print("[supervisor] LLM fallback: Completion guard issue needs a scoped update")
            return _route_completion_guard_to_update_coder(completion_failure, turn_count=turn_count)
        if not has_code:
            print(f"[supervisor] âš  LLM fallback: No code files exist yet, routing to requirements")
            return {"status": "delegating_requirements", "supervisor_turns": turn_count}
        return {"status": "done", "supervisor_turns": turn_count}
    
    print(f"[supervisor] ðŸ¤” Rationale: {decision.rationale}")
    print(f"[supervisor] ðŸŽ¯ Next: {decision.next_node}")
    
    next_node = decision.next_node
    valid_nodes = {"github_search", "planner", "architect", "requirements", "github_scout", "adapter", "repo_finalize", "coder", "parallel_coder", "update_coder", "verify", "summary", "integrator"}
    if next_node not in valid_nodes:
        next_node = "summary"

    if not state.get("plan") and not state.get("github_search_completed") and next_node != "github_search":
        print(f"[supervisor] BLOCKED {next_node}: GitHub search must run before initial planning")
        next_node = "github_search"

    last_exec_failed = bool(state.get("execution_results")) and not state["execution_results"][-1].success
    coder_incomplete = _coder_progress_incomplete(state)
    if coder_incomplete and not last_exec_failed and next_node != "coder":
        print(f"[supervisor] BLOCKED {next_node}: coder still has pending implementation steps, routing to coder")
        next_node = "coder"

    if next_node == "summary":
        completion_failure = _completion_guard_failure(state, config)
        if _completion_guard_should_use_update_coder(state, completion_failure):
            print("[supervisor] BLOCKED summary: completion guard still has a concrete repair")
            next_node = "update_coder"
        elif completion_failure is not None:
            print("[supervisor] BLOCKED summary: completion guard says implementation is incomplete")
            next_node = "coder" if state.get("coder_state") else "requirements"

   
    if next_node == "summary" and not has_code:
        if not state.get("plan"):
            print(f"[supervisor] ðŸš« BLOCKED summary: No plan exists, routing to planner")
            next_node = "planner"
        elif not state.get("task_plan"):
            print(f"[supervisor] ðŸš« BLOCKED summary: No task plan exists, routing to architect")
            next_node = "architect"
        elif not state.get("coder_state"):
            print(f"[supervisor] ðŸš« BLOCKED summary: No coder state, routing to requirements")
            next_node = "requirements"
        else:
            print(f"[supervisor] ðŸš« BLOCKED summary: No code files exist yet, routing to coder")
            next_node = "coder"

    
    if next_node != "update_coder":
        if not state.get("plan") and next_node not in ["github_search", "github_scout", "adapter", "repo_finalize", "verify", "planner", "summary"]:
            print(f"[supervisor] âš  Override: No plan exists, routing to planner instead of {next_node}")
            next_node = "planner"
        elif state.get("plan") and not state.get("task_plan") and next_node not in ["planner", "architect", "repo_finalize", "verify", "summary"]:
            print(f"[supervisor] âš  Override: No task plan exists, routing to architect instead of {next_node}")
            next_node = "architect"
        elif next_node == "coder" and not state.get("coder_state"):
            print(f"[supervisor] âš  Override: Coder state missing, routing to requirements instead of {next_node}")
            next_node = "planner" if not state.get("plan") else "requirements"

    last_exec = state["execution_results"][-1] if state.get("execution_results") else None
    if last_exec_failed and _completion_guard_should_use_update_coder(state, last_exec):
        print("[supervisor] Completion guard failure is a scoped update, routing to update_coder")
        next_node = "update_coder"
        last_exec_failed = False

    if last_exec_failed and state.get("debug_attempts", 0) < max_debug_attempts and next_node != "update_coder":
        print(f"[supervisor] âš  Override: Latest execution failed, routing to reviewer instead of {next_node}")
        next_node = "update_coder"

    status_map = {
        "github_search": "delegating_github_search",
        "planner": "delegating_planner",
        "architect": "delegating_architect", 
        "requirements": "delegating_requirements",
        "github_scout": "delegating_github_scout",
        "adapter": "delegating_adapter",
        "repo_finalize": "delegating_repo_finalize",
        "coder": "delegating_coder",
        "parallel_coder": "delegating_parallel_coder",
        "update_coder": "delegating_update_coder",
        "integrator": "delegating_integrator",
        "verify": "delegating_verify",
        "summary": "done"
    }
    
    return {
        "status": status_map.get(next_node, "done"),
        "thinking_rationale": decision.rationale,
        "supervisor_turns": turn_count,
    }




def route_after_clarifier(state: PipelineState):
    status = state.get("status")
    if status == "delegating_update_coder":
        return "update_coder"
    if not _FAST_PATH_DEFAULT:
        return "supervisor"
    return "github_search"


def route_after_github_search(state: PipelineState):
    return "github_scout" if _github_repo_found(state) else "planner"


def route_after_github_scout(state: PipelineState):
    return "adapter" if _scout_downloaded_repo(state) else "planner"


def route_after_coder(state: PipelineState):
    status = state.get("status")
    if status == "delegating_coder":
        return "coder"
    if status == "delegating_integrator":
        return "integrator"
    return "verify"


def route_after_parallel_coder(state: PipelineState):
    return "integrator" if state.get("status") == "delegating_integrator" else "verify"


def route_after_verify(state: PipelineState):
    if state.get("status") == "delegating_update_coder":
        return "update_coder"
    return "summary"


def route_after_supervisor(state: PipelineState):
    status = state.get("status")
    if status == "delegating_github_search": return "github_search"
    if status == "delegating_planner": return "planner"
    if status == "delegating_architect": return "architect"
    if status == "delegating_requirements": return "requirements"
    if status == "delegating_github_scout": return "github_scout"
    if status == "delegating_adapter": return "adapter"
    if status == "delegating_repo_finalize": return "repo_finalize"
    if status == "delegating_update_coder": return "update_coder"
    if status == "delegating_coder": return "coder"
    if status == "delegating_parallel_coder": return "parallel_coder"
    if status == "delegating_integrator": return "integrator"
    if status in {"delegating_verify"}: return "verify"
    return "summary"


_DEFAULT_DB = Path(os.getenv("DB_PATH", str(CHECKPOINT_DB)))

_CHECKPOINT_ALLOWED_MODULES = [
    ("schemas", "Plan"),
    ("schemas", "File"),
    ("schemas", "TaskPlan"),
    ("schemas", "ImplementationTask"),
    ("schemas", "RequirementsSpec"),
    ("schemas", "CoderState"),
    ("schemas", "MemoryStore"),
    ("schemas", "MemoryEntry"),
    ("schemas", "GeneratedFileInfo"),
    ("schemas", "ExecutionResult"),
    ("schemas", "DebugResult"),
    ("schemas", "TestReport"),
    ("schemas", "TestCase"),
    ("schemas", "RetrievedContext"),
    ("schemas", "ToolTranscriptEntry"),
    ("schemas", "SupervisorDecision"),
    ("schemas", "ClarifiedRequest"),
    ("schemas", "DetectedService"),
    ("schemas", "RepoAdaptationPlan"),
]


def _make_jsonplus_serializer() -> JsonPlusSerializer:
    try:
        return JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_ALLOWED_MODULES)
    except TypeError:
        return JsonPlusSerializer(allowed_json_modules=_CHECKPOINT_ALLOWED_MODULES)

@contextmanager
def get_checkpointer(db_path: str | Path = _DEFAULT_DB):
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    try:
        yield SqliteSaver(conn, serde=_make_jsonplus_serializer())
    finally:
        conn.close()

def build_graph(checkpointer: SqliteSaver):
    g = StateGraph(PipelineState)
    g.add_node("clarifier", clarifier_agent)
    g.add_node("supervisor", supervisor_agent)
    g.add_node("github_search", github_search_agent)
    g.add_node("planner", planner_agent)
    g.add_node("architect", architect_agent)
    g.add_node("requirements", requirements_agent)
    g.add_node("github_scout", github_scout_agent)
    g.add_node("adapter", adapter_agent)
    g.add_node("repo_finalize", repo_finalize_agent)
    g.add_node("coder", coder_agent)
    g.add_node("parallel_coder", parallel_coder_agent)
    g.add_node("update_coder", update_coder_agent)
    g.add_node("integrator", integrator_agent)
    g.add_node("verify", verify_agent)
    g.add_node("summary", summary_agent)

    g.add_edge(START, "clarifier")
    g.add_conditional_edges(
        "clarifier", route_after_clarifier,
        ["github_search", "update_coder", "supervisor"],
    )
    g.add_conditional_edges(
        "supervisor", route_after_supervisor,
        ["github_search", "planner", "architect", "requirements", "github_scout", "adapter",
         "repo_finalize", "update_coder", "coder", "parallel_coder", "integrator", "verify", "summary"],
    )

    g.add_conditional_edges("github_search", route_after_github_search, ["github_scout", "planner"])
    g.add_conditional_edges("github_scout", route_after_github_scout, ["adapter", "planner"])
    g.add_edge("adapter", "repo_finalize")
    g.add_edge("repo_finalize", "verify")

    g.add_edge("planner", "architect")
    # From-scratch flow only runs after GitHub search fails to find a suitable repo.
    g.add_edge("architect", "requirements")
    if _ENABLE_PARALLEL_CODER_DEFAULT:
        g.add_edge("requirements", "parallel_coder")
    else:
        g.add_edge("requirements", "coder")

    g.add_conditional_edges("coder", route_after_coder, ["coder", "integrator", "verify"])
    g.add_conditional_edges("parallel_coder", route_after_parallel_coder, ["integrator", "verify"])
    g.add_edge("update_coder", "verify")
    g.add_edge("integrator", "verify")
    g.add_conditional_edges("verify", route_after_verify, ["summary", "update_coder"])
    g.add_edge("summary", END)

    return g.compile(checkpointer=checkpointer, debug=_GRAPH_DEBUG_DEFAULT)

def _config(
    thread_id: str,
    project_root: str = None,
    auto_model_fallback_on_limit: bool | None = None,
) -> dict:
    cfg = {"thread_id": thread_id}
    if project_root:
        cfg["project_root"] = project_root
    if auto_model_fallback_on_limit is not None:
        cfg["auto_model_fallback_on_limit"] = bool(auto_model_fallback_on_limit)
    return {"configurable": cfg}

def _blank_state(user_prompt: str) -> PipelineState:
    return {
        "original_user_prompt": user_prompt,
        "user_prompt": user_prompt, 
        "rag_contexts": [], 
        "execution_results": [], 
        "debug_results": [], 
        "tool_transcript": [],
        "debug_attempts": 0, 
        "max_debug_attempts": _MAX_DEBUG_ATTEMPTS_DEFAULT, 
        "supervisor_turns": 0,
        "max_supervisor_turns": _MAX_SUPERVISOR_TURNS_DEFAULT,
        "integrator_attempts": 0,
        "memory_store": MemoryStore(), 
        "generated_file_manifest": {},
        "env_sync_missing_keys": [],
        "env_sync_written_files": [],
        "status": "pending",
        "github_search_completed": False,
        "rollback_applied": False,
    }

def _update_state(user_prompt: str, rollback_snapshot_path: str | None = None) -> PipelineState:
    """Creates a state that skips planning and routes directly to the update_coder."""
    return {
        "original_user_prompt": user_prompt,
        "user_prompt": user_prompt,
        "rag_contexts": [],
        "execution_results": [],
        "debug_results": [],
        "debug_attempts": 0,
        "max_debug_attempts": _MAX_DEBUG_ATTEMPTS_DEFAULT,
        "supervisor_turns": 0,
        "max_supervisor_turns": _MAX_SUPERVISOR_TURNS_DEFAULT,
        "integrator_attempts": 0,
        "memory_store": MemoryStore(),
        "generated_file_manifest": {},
        "env_sync_missing_keys": [],
        "env_sync_written_files": [],
        "status": "delegating_update_coder",
        "github_search_completed": True,
        "rollback_snapshot_path": rollback_snapshot_path,
        "rollback_applied": False,
    }

_SESSION_FILE = Path(os.getenv("SESSION_FILE", "run_session.json"))
def _save_session(thread_id: str, user_prompt: str, status: str) -> None:
    _SESSION_FILE.write_text(json.dumps({"thread_id": thread_id, "user_prompt": user_prompt, "status": status}, indent=2), encoding="utf-8")

def _set_project_status(project_id: str | None, status: str) -> None:
    if not project_id:
        return
    try:
        from codemind.core.database import ProjectGeneration, transactional_session
        with transactional_session() as session:
            db_proj = session.query(ProjectGeneration).filter_by(id=project_id).first()
            if db_proj:
                db_proj.status = status
    except Exception as exc:
        print(f"[graph] Failed to update project status for {project_id}: {exc}")


def run_pipeline(user_prompt: str, *, thread_id: str | None = None, db_path=None, project_id: str | None = None) -> str:
    thread_id = thread_id or f"run-{uuid.uuid4().hex[:12]}"
    project_id = project_id or thread_id
    _save_session(thread_id, user_prompt, "running")
    _set_project_status(project_id, "running")
    try:
        with get_checkpointer(db_path or _DEFAULT_DB) as checkpointer:
            workflow = build_graph(checkpointer)
            workflow.invoke(_blank_state(user_prompt), config=_config(thread_id))
    except Exception:
        _save_session(thread_id, user_prompt, "failed")
        _set_project_status(project_id, "failed")
        raise
    _save_session(thread_id, user_prompt, "done")
    _set_project_status(project_id, "done")
    return thread_id


def resume_pipeline(thread_id: str, *, db_path=None, project_id: str | None = None) -> str:
    project_id = project_id or thread_id
    _set_project_status(project_id, "running")
    try:
        with get_checkpointer(db_path or _DEFAULT_DB) as checkpointer:
            workflow = build_graph(checkpointer)
            workflow.invoke(None, config=_config(thread_id))
    except Exception:
        _set_project_status(project_id, "failed")
        raise
    _set_project_status(project_id, "done")
    return thread_id

def get_run_state(thread_id: str, *, db_path=None):
    with get_checkpointer(db_path or _DEFAULT_DB) as checkpointer:
        return build_graph(checkpointer).get_state(_config(thread_id))
