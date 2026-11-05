from __future__ import annotations

import sys

# Windows redirects stdout/stderr to the console code page (cp1252) by default,
# which raises UnicodeEncodeError on emoji in log messages. Force UTF-8 so a
# log line can never kill a background pipeline.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import uuid
import json
import asyncio
import subprocess
import socket
import io
import zipfile
import hashlib
import logging
import time
import warnings
from typing import Any, Callable
from datetime import datetime, timezone
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends, Request, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import threading
from database import get_db, ProjectGeneration, Base, engine, init_database, verify_database_health, transactional_session
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage as LCHumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from project_ops import create_project_snapshot, extract_project_zip, import_project_files
from runtime_detection import build_run_plan
from ui_events import build_ui_event
from env_sync import auto_fill_project_env

_graph_module = None


def _get_graph():
    global _graph_module
    if _graph_module is None:
        import graph as graph_module

        _graph_module = graph_module
    return _graph_module

try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
except ModuleNotFoundError:
    ChatNVIDIA = None

warnings.filterwarnings(
    "ignore",
    message=r"Found moonshotai/kimi-k2\.6 in available_models, but type is unknown and inference may fail\.",
    category=UserWarning,
    module=r"langchain_nvidia_ai_endpoints\._common",
)

def _make_chat_llm(kwargs: dict[str, Any]) -> Any:
    if ChatNVIDIA is None:
        raise RuntimeError("langchain_nvidia_ai_endpoints is not installed; project chat is unavailable.")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*moonshotai/kimi-k2\.6.*type is unknown and inference may fail.*",
            category=UserWarning,
        )
        return ChatNVIDIA(**kwargs)


_nvidia_chat_kwargs: dict[str, Any] = {
    "model": os.getenv("NVIDIA_CHAT_MODEL", "moonshotai/kimi-k2.6"),
    "temperature": float(os.getenv("NVIDIA_CHAT_TEMPERATURE", "0.7")),
    "max_completion_tokens": int(os.getenv("NVIDIA_CHAT_MAX_COMPLETION_TOKENS", "2048")),
}
_nvidia_api_key = os.getenv("NVIDIA_API_KEY")
if _nvidia_api_key:
    _nvidia_chat_kwargs["api_key"] = _nvidia_api_key

_chat_llm_instance = None
def get_chat_llm():
    global _chat_llm_instance
    if _chat_llm_instance is None:
        _chat_llm_instance = _make_chat_llm(_nvidia_chat_kwargs)
    return _chat_llm_instance

chat_histories: dict[str, list[dict]] = {}  # thread_id -> [{role, content}]
chat_turn_locks: dict[str, asyncio.Lock] = {}
chat_tool_activity: dict[str, list[dict]] = {}
pipeline_run_lock = threading.Lock()
active_pipeline_threads: set[str] = set()

CHAT_HISTORY_LIMIT = int(os.getenv("CHAT_HISTORY_LIMIT", "40"))
CHAT_TOOL_ACTIVITY_LIMIT = int(os.getenv("CHAT_TOOL_ACTIVITY_LIMIT", "24"))
CHAT_TOOL_ACTIVITY_CONTEXT_LIMIT = int(os.getenv("CHAT_TOOL_ACTIVITY_CONTEXT_LIMIT", "6"))
CHAT_LLM_TIMEOUT_SECONDS = int(os.getenv("CHAT_LLM_TIMEOUT_SECONDS", "90"))
FINAL_STATUS_EVENT_DELAY_SECONDS = float(os.getenv("FINAL_STATUS_EVENT_DELAY_SECONDS", "0.75"))

logger = logging.getLogger("ai_studio.server")
if not logger.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

state_lock = threading.RLock()


def _chat_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text is not None:
                    parts.append(str(text))
        if parts:
            return "".join(parts)
    return str(content)


def _chat_exception_message(exc: BaseException) -> str:
    if isinstance(exc, asyncio.TimeoutError):
        return "The chat model timed out before it could respond. Please try again."
    message = str(exc).strip()
    if message:
        return message
    return f"{exc.__class__.__name__}: {exc!r}"


async def _invoke_chat_llm(model: Any, messages: list[Any]) -> Any:
    return await asyncio.wait_for(
        asyncio.to_thread(model.invoke, messages),
        timeout=CHAT_LLM_TIMEOUT_SECONDS,
    )


def _bind_chat_tools(model: Any, tools: list[Any]) -> Any:
    # NVIDIA documents Kimi K2.6 tool support, but the LangChain capability
    # registry can lag new NIM releases. Silence only this known false positive.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*moonshotai/kimi-k2\.6.*not known to support tools\..*",
            category=UserWarning,
        )
        return model.bind_tools(tools)


def _get_chat_lock(thread_id: str) -> asyncio.Lock:
    with state_lock:
        lock = chat_turn_locks.get(thread_id)
        if lock is None:
            lock = asyncio.Lock()
            chat_turn_locks[thread_id] = lock
        return lock


def _trim_chat_history(history: list[dict]) -> None:
    overflow = len(history) - CHAT_HISTORY_LIMIT
    if overflow > 0:
        del history[:overflow]


def _clip_chat_text(text: str | None, limit: int = 240, *, empty_fallback: str = "(empty)") -> str:
    if text is None:
        return empty_fallback
    text = str(text).strip()
    if not text:
        return empty_fallback
    if len(text) <= limit:
        return text
    return text[:limit] + "... (truncated)"


def _record_chat_tool_activity(thread_id: str, tool_name: str, tool_args: dict, result: str) -> None:
    with state_lock:
        entries = chat_tool_activity.setdefault(thread_id, [])
        entries.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tool_name": tool_name,
                "tool_args": json.dumps(tool_args, ensure_ascii=False, sort_keys=True, default=str),
                "result_preview": _clip_chat_text(result, 260),
            }
        )
        overflow = len(entries) - CHAT_TOOL_ACTIVITY_LIMIT
        if overflow > 0:
            del entries[:overflow]


def _format_recent_tool_activity(thread_id: str) -> str:
    with state_lock:
        entries = list(chat_tool_activity.get(thread_id) or [])
    if not entries:
        return "(none yet)"

    lines = []
    for entry in entries[-CHAT_TOOL_ACTIVITY_CONTEXT_LIMIT:]:
        timestamp = entry.get("timestamp", "")
        if timestamp:
            timestamp = timestamp.split(".")[0].replace("+00:00", "Z")
        lines.append(
            f"- {timestamp} {entry.get('tool_name', 'tool')}({entry.get('tool_args', '{}')})"
            f" -> {entry.get('result_preview', '(empty)')}"
        )
    return "\n".join(lines)


def _reserve_pipeline_thread(thread_id: str) -> bool:
    with pipeline_run_lock:
        if thread_id in active_pipeline_threads:
            return False
        active_pipeline_threads.add(thread_id)
        return True


def _is_pipeline_running(thread_id: str) -> bool:
    with pipeline_run_lock:
        return thread_id in active_pipeline_threads


def _release_pipeline_thread(thread_id: str) -> None:
    with pipeline_run_lock:
        active_pipeline_threads.discard(thread_id)

Base.metadata.create_all(bind=engine)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Fine for a college project local demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)


@app.get("/health")
async def health():
    return {"status": "ok"}


def _verify_checkpoint_db(path: str = "pipeline_checkpoints.db") -> bool:
    import sqlite3
    try:
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA integrity_check;").fetchone()
        conn.close()
        return True
    except Exception:
        return False


@app.on_event("startup")
async def startup_health_checks():
    init_database()
    if not verify_database_health():
        logger.warning("projects.db health check failed at startup")
    if not _verify_checkpoint_db():
        logger.warning("pipeline_checkpoints.db integrity check failed at startup")
    if os.getenv("PREWARM_LLM_MODELS", "0").strip().lower() not in {"0", "false", "no", "off"}:
        def _prewarm():
            try:
                _get_graph().prewarm_llm_models()
            except Exception as exc:
                logger.warning("LLM prewarm skipped: %s", exc)
        threading.Thread(target=_prewarm, daemon=True).start()


def api_success(data: dict | None = None, *, message: str | None = None) -> dict:
    payload = {"ok": True}
    if message:
        payload["message"] = message
    if data:
        payload["data"] = data
    return payload


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        {"ok": False, "error": {"code": "http_error", "message": str(exc.detail)}},
        status_code=exc.status_code,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled server error", exc_info=exc)
    return JSONResponse(
        {"ok": False, "error": {"code": "internal_error", "message": "Internal server error"}},
        status_code=500,
    )

@app.get("/")
def serve_root():
    """Serve the main UI at the root URL."""
    react_index = os.path.join("static", "react", "index.html")
    if not os.path.exists(react_index):
        raise HTTPException(
            status_code=404,
            detail="UI build not found. Run `npm install && npm run build` in frontend/.",
        )
    return FileResponse(react_index)

app.mount("/static", StaticFiles(directory="static"), name="static")

stream_queues = {}
running_apps = {}

PROJECT_CONTEXT_CHUNK_SIZE = int(os.getenv("PROJECT_CONTEXT_CHUNK_SIZE", "500"))
PROJECT_CONTEXT_CHUNK_OVERLAP = int(os.getenv("PROJECT_CONTEXT_CHUNK_OVERLAP", "100"))
PROJECT_CONTEXT_TOP_K = int(os.getenv("PROJECT_CONTEXT_TOP_K", "5"))
PROJECT_CONTEXT_MAX_FILE_BYTES = int(os.getenv("PROJECT_CONTEXT_MAX_FILE_BYTES", "200000"))
PROJECT_CONTEXT_MAX_CHARS = int(os.getenv("PROJECT_CONTEXT_MAX_CHARS", "6000"))
PROJECT_CONTEXT_MANIFEST_VERSION = 1
PROJECT_CONTEXT_INDEX_ROOT = ".ai_studio_index"

_project_embedder = None
np = None
chromadb = None
HashingVectorizer = None


def _get_numpy():
    global np
    if np is None:
        import numpy as numpy_module

        np = numpy_module
    return np


def _get_chromadb():
    global chromadb
    if chromadb is None:
        import chromadb as chromadb_module

        chromadb = chromadb_module
    return chromadb


def _get_hashing_vectorizer():
    global HashingVectorizer
    if HashingVectorizer is None:
        from sklearn.feature_extraction.text import HashingVectorizer as hashing_vectorizer_cls

        HashingVectorizer = hashing_vectorizer_cls
    return HashingVectorizer


class HashingCodeEmbedder:
    """Offline-safe fallback embedder for semantic-ish code retrieval."""

    def __init__(self):
        self.provider_key = "hashing-charwb-3-5-v1"
        vectorizer_cls = _get_hashing_vectorizer()
        self.vectorizer = vectorizer_cls(
            n_features=2048,
            analyzer="char_wb",
            ngram_range=(3, 5),
            alternate_sign=False,
            norm=None,
        )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        np_mod = _get_numpy()
        matrix = self.vectorizer.transform(texts).astype(np_mod.float32)
        dense = matrix.toarray()
        norms = np_mod.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return dense / norms


class OpenAIProjectEmbedder:
    def __init__(self, model_name: str):
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.provider_key = f"openai::{model_name}"

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        np_mod = _get_numpy()
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        dense = np_mod.array([item.embedding for item in response.data], dtype=np_mod.float32)
        norms = np_mod.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return dense / norms


class SentenceTransformerProjectEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, local_files_only=True)
        self.provider_key = f"sentence-transformers::{model_name}"

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        np_mod = _get_numpy()
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np_mod.float32)


def _get_project_index_dir(thread_id: str) -> str:
    return os.path.abspath(os.path.join("generated_project", PROJECT_CONTEXT_INDEX_ROOT, thread_id))


def _get_project_manifest_path(thread_id: str) -> str:
    return os.path.join(_get_project_index_dir(thread_id), "manifest.json")


def _get_project_chroma_dir(thread_id: str) -> str:
    return os.path.join(_get_project_index_dir(thread_id), "chroma")


def _recover_project_state(thread_id: str) -> dict:
    """Best-effort recovery for broken semantic index state."""
    import shutil

    index_dir = _get_project_index_dir(thread_id)
    if os.path.exists(index_dir):
        try:
            shutil.rmtree(index_dir)
        except OSError as exc:
            return {"status": "failed", "reason": f"index cleanup failed: {exc}"}
    try:
        _sync_project_index(thread_id)
        return {"status": "recovered"}
    except Exception as exc:
        return {"status": "failed", "reason": str(exc)}


def _get_project_embedder():
    global _project_embedder
    if _project_embedder is not None:
        return _project_embedder

    provider = os.getenv("PROJECT_CONTEXT_EMBEDDER", "auto").strip().lower()
    openai_model = os.getenv("PROJECT_CONTEXT_OPENAI_MODEL", "text-embedding-3-small")
    local_model = os.getenv("PROJECT_CONTEXT_LOCAL_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    candidates = []

    if provider in {"auto", "openai"} and os.getenv("OPENAI_API_KEY"):
        candidates.append(lambda: OpenAIProjectEmbedder(openai_model))
    if provider in {"auto", "local", "sentence-transformers"}:
        candidates.append(lambda: SentenceTransformerProjectEmbedder(local_model))
    candidates.append(HashingCodeEmbedder)

    for factory in candidates:
        try:
            _project_embedder = factory()
            return _project_embedder
        except Exception:
            continue

    _project_embedder = HashingCodeEmbedder()
    return _project_embedder


def _load_project_manifest(thread_id: str) -> dict:
    manifest_path = _get_project_manifest_path(thread_id)
    if not os.path.exists(manifest_path):
        return {
            "version": PROJECT_CONTEXT_MANIFEST_VERSION,
            "embedder_key": None,
            "collection_name": None,
            "files": {},
        }

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {
            "version": PROJECT_CONTEXT_MANIFEST_VERSION,
            "embedder_key": None,
            "collection_name": None,
            "files": {},
        }


def _save_project_manifest(thread_id: str, manifest: dict) -> None:
    index_dir = _get_project_index_dir(thread_id)
    os.makedirs(index_dir, exist_ok=True)
    with open(_get_project_manifest_path(thread_id), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _get_collection_name(embedder_key: str) -> str:
    suffix = hashlib.sha1(embedder_key.encode("utf-8")).hexdigest()[:12]
    return f"project_context_{suffix}"


def _iter_project_text_files(project_dir: str):
    for root, dirs, filenames in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for filename in filenames:
            if filename in SKIP_FILES_SET:
                continue
            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, project_dir).replace("\\", "/")
            try:
                if os.path.getsize(full_path) > PROJECT_CONTEXT_MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield rel_path, full_path


def _read_text_file_for_index(full_path: str) -> str | None:
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except (UnicodeDecodeError, PermissionError, OSError):
        return None


def _chunk_text_for_context(text: str, chunk_size: int = PROJECT_CONTEXT_CHUNK_SIZE, overlap: int = PROJECT_CONTEXT_CHUNK_OVERLAP) -> list[dict]:
    if not text:
        return []

    text = text.replace("\r\n", "\n")
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            newline = text.rfind("\n", start + int(chunk_size * 0.6), end)
            if newline > start:
                end = newline + 1

        snippet = text[start:end].strip()
        if snippet:
            chunks.append({"start": start, "end": end, "text": snippet})

        if end >= text_len:
            break
        next_start = max(end - overlap, start + 1)
        start = next_start

    return chunks


def _batched(values: list, size: int = 64):
    for idx in range(0, len(values), size):
        yield values[idx:idx + size]


def _sync_project_index(thread_id: str) -> dict:
    project_dir = _get_project_dir(thread_id)
    if not os.path.exists(project_dir):
        return {"status": "missing"}

    embedder = _get_project_embedder()
    collection_name = _get_collection_name(embedder.provider_key)
    index_dir = _get_project_index_dir(thread_id)
    chroma_dir = _get_project_chroma_dir(thread_id)
    os.makedirs(index_dir, exist_ok=True)
    os.makedirs(chroma_dir, exist_ok=True)

    manifest = _load_project_manifest(thread_id)
    previous_files = manifest.get("files", {})
    current_files = {}
    changed_paths = []

    chroma = _get_chromadb()
    client = chroma.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    if manifest.get("embedder_key") != embedder.provider_key or manifest.get("collection_name") != collection_name:
        previous_files = {}
        manifest = {
            "version": PROJECT_CONTEXT_MANIFEST_VERSION,
            "embedder_key": embedder.provider_key,
            "collection_name": collection_name,
            "files": {},
        }

    if collection.count() == 0 and previous_files:
        previous_files = {}

    for rel_path, full_path in _iter_project_text_files(project_dir):
        try:
            stat = os.stat(full_path)
        except OSError:
            continue

        current_state = {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}
        current_files[rel_path] = current_state
        old_state = previous_files.get(rel_path)
        if not old_state or old_state.get("mtime_ns") != stat.st_mtime_ns or old_state.get("size") != stat.st_size:
            changed_paths.append(rel_path)

    deleted_paths = [rel_path for rel_path in previous_files if rel_path not in current_files]

    for rel_path in deleted_paths + changed_paths:
        old_state = previous_files.get(rel_path, {})
        chunk_ids = old_state.get("chunk_ids", [])
        for chunk_batch in _batched(chunk_ids):
            if chunk_batch:
                collection.delete(ids=chunk_batch)

    next_files = {}
    for rel_path, current_state in current_files.items():
        if rel_path not in changed_paths and rel_path in previous_files:
            next_files[rel_path] = previous_files[rel_path]
            continue

        full_path = os.path.join(project_dir, rel_path.replace("/", os.sep))
        content = _read_text_file_for_index(full_path)
        if not content or not content.strip():
            continue

        chunks = _chunk_text_for_context(content)
        if not chunks:
            continue

        chunk_ids = []
        documents = []
        metadatas = []
        for idx, chunk in enumerate(chunks):
            digest = hashlib.sha1(f"{rel_path}:{idx}:{chunk['text']}".encode("utf-8")).hexdigest()[:12]
            chunk_id = f"{rel_path}::{idx}::{digest}"
            chunk_ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append(
                {
                    "path": rel_path,
                    "chunk_index": idx,
                    "start_offset": chunk["start"],
                    "end_offset": chunk["end"],
                }
            )

        embeddings = embedder.embed_texts(documents)
        for ids_batch, docs_batch, meta_batch, embed_batch in zip(
            _batched(chunk_ids),
            _batched(documents),
            _batched(metadatas),
            _batched(embeddings.tolist()),
        ):
            collection.upsert(
                ids=ids_batch,
                documents=docs_batch,
                metadatas=meta_batch,
                embeddings=embed_batch,
            )

        next_files[rel_path] = {
            "mtime_ns": current_state["mtime_ns"],
            "size": current_state["size"],
            "chunk_ids": chunk_ids,
        }

    for rel_path, old_state in previous_files.items():
        if rel_path in current_files and rel_path not in changed_paths:
            next_files.setdefault(rel_path, old_state)

    manifest["version"] = PROJECT_CONTEXT_MANIFEST_VERSION
    manifest["embedder_key"] = embedder.provider_key
    manifest["collection_name"] = collection_name
    manifest["files"] = next_files
    _save_project_manifest(thread_id, manifest)

    return {
        "status": "ok",
        "indexed_files": len(changed_paths),
        "deleted_files": len(deleted_paths),
        "total_files": len(next_files),
        "collection_name": collection_name,
    }


def _retrieve_project_chunks(thread_id: str, query: str, top_k: int = PROJECT_CONTEXT_TOP_K) -> list[dict]:
    manifest = _load_project_manifest(thread_id)
    if not manifest.get("files"):
        return []

    embedder = _get_project_embedder()
    if manifest.get("embedder_key") != embedder.provider_key:
        return []

    chroma_dir = _get_project_chroma_dir(thread_id)
    chroma = _get_chromadb()
    client = chroma.PersistentClient(path=chroma_dir)
    collection_name = manifest.get("collection_name") or _get_collection_name(embedder.provider_key)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    collection_count = collection.count()
    if collection_count == 0:
        return []

    query_embedding = embedder.embed_texts([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection_count),
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    retrieved = []
    seen = set()

    for document, metadata, distance in zip(documents, metadatas, distances):
        if not document or not metadata:
            continue
        dedupe_key = (metadata.get("path"), metadata.get("chunk_index"))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        retrieved.append(
            {
                "path": metadata.get("path", "unknown"),
                "chunk_index": metadata.get("chunk_index", 0),
                "distance": distance,
                "content": document,
            }
        )

    return retrieved


def _build_retrieval_query(user_message: str, history: list[dict] | None = None) -> str:
    lines = []
    if history:
        for msg in history[-4:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                lines.append(f"{role}: {content}")
    lines.append(f"user: {user_message}")
    return "\n".join(lines)

def run_project_generation(
    thread_id: str,
    prompt: str,
    loop: asyncio.AbstractEventLoop,
    is_update: bool = False,
    rollback_snapshot_path: str | None = None,
    auto_model_fallback_on_limit: bool = True,
    is_resume: bool = False,
):
    db_path = "pipeline_checkpoints.db"
    final_status = "done"
    final_message = None
    final_done_payload = {
        "status": "done",
        "message": None,
        "run_plan": {"commands": [], "html_file": None, "html_path": None, "serve_port": None},
        "structure": [],
    }
    live_updates_stop = threading.Event()
    live_updates_thread = None
    
    def emit_event(event_type, data, agent: str | None = None):
        if thread_id in stream_queues and loop:
            try:
                event = build_ui_event(event_type, data, agent=agent)
            except Exception as exc:
                logger.warning("Dropping invalid UI event %s for %s: %s", event_type, thread_id, exc)
                event = build_ui_event("status", f"Skipped malformed {event_type} event.")
            for q in list(stream_queues[thread_id]):
                loop.call_soon_threadsafe(q.put_nowait, event)

    print(f"[server] 🚀 Starting pipeline for thread {thread_id}...")
    emit_event("status", "Starting pipeline update..." if is_update else "Starting pipeline...")
    
    try:
        graph = _get_graph()
        live_updates_thread = threading.Thread(
            target=_stream_live_project_files,
            args=(thread_id, emit_event, live_updates_stop),
            kwargs={"seed_existing": is_update},
            daemon=True,
        )
        live_updates_thread.start()

        with graph.get_checkpointer(db_path) as checkpointer:
            workflow = graph.build_graph(checkpointer)
            graph._save_session(thread_id, prompt, "running")
            config = graph._config(thread_id, auto_model_fallback_on_limit=auto_model_fallback_on_limit)
            
            if is_resume:
                state = None
            elif is_update:
                state = graph._update_state(prompt, rollback_snapshot_path=rollback_snapshot_path)
                original_project_dir = os.path.abspath(os.path.join("generated_project", thread_id))
                config = graph._config(
                    f"{thread_id}-update-{uuid.uuid4().hex[:6]}",
                    project_root=original_project_dir,
                    auto_model_fallback_on_limit=auto_model_fallback_on_limit,
                )
            else:
                state = graph._blank_state(prompt)
            
            for event in workflow.stream(state, config=config, stream_mode=["messages", "updates"]):
                if len(event) == 2:
                    event_type, event_data = event[0], event[1]
                else:
                    event_type, event_data = "updates", event
                    
                if event_type == "messages":
                    msg_chunk = event_data[0]
                    if hasattr(msg_chunk, "content") and msg_chunk.content:
                        emit_event("thoughts", msg_chunk.content)
                    if hasattr(msg_chunk, "tool_calls") and msg_chunk.tool_calls:
                        emit_event("tool_calls", msg_chunk.tool_calls)
                        
                elif event_type == "updates":
                    for node_name, node_state in event_data.items():
                        emit_event("node_finished", {"node": node_name}, agent=node_name)
                        missing_env_keys = node_state.get("env_sync_missing_keys") if isinstance(node_state, dict) else None
                        if missing_env_keys:
                            preview = ", ".join(str(key) for key in missing_env_keys[:12])
                            extra = f" (+{len(missing_env_keys) - 12} more)" if len(missing_env_keys) > 12 else ""
                            emit_event(
                                "status",
                                f"Missing generated-project env values: {preview}{extra}. Add them to the main .env before running provider-backed features.",
                                agent=node_name,
                            )
                        
                        if node_name == "coder":
                            cs = node_state.get("coder_state")
                            if cs and hasattr(cs, "current_step_idx"):
                                idx = cs.current_step_idx - 1
                                if idx >= 0 and idx < len(cs.task_plan.implementation_steps):
                                    task = cs.task_plan.implementation_steps[idx]
                                    filepath = os.path.join("generated_project", thread_id, task.filepath)
                                    if os.path.exists(filepath):
                                        with open(filepath, "r", encoding="utf-8") as f:
                                            emit_event("file_update", {"filepath": task.filepath, "content": f.read()}, agent=node_name)
                                        
                        if node_name == "reviewer":
                            dbg_res_list = node_state.get("debug_results")
                            if dbg_res_list and len(dbg_res_list) > 0:
                                last_dbg = dbg_res_list[-1]
                                for modified_file in last_dbg.files_modified:
                                    filepath = os.path.join("generated_project", thread_id, modified_file)
                                    if os.path.exists(filepath):
                                        with open(filepath, "r", encoding="utf-8") as f:
                                            emit_event("file_update", {"filepath": modified_file, "content": f.read()}, agent=node_name)
                        
                        if node_name == "update_coder":
                            project_dir = os.path.abspath(os.path.join("generated_project", thread_id))
                            if os.path.exists(project_dir):
                                for root_dir, _, filenames in os.walk(project_dir):
                                    for filename in filenames:
                                        fpath = os.path.join(root_dir, filename)
                                        rel_path = os.path.relpath(fpath, project_dir).replace("\\", "/")
                                        if "node_modules" in rel_path or ".git" in rel_path:
                                            continue
                                        try:
                                            with open(fpath, "r", encoding="utf-8") as f:
                                                emit_event("file_update", {"filepath": rel_path, "content": f.read()}, agent=node_name)
                                        except (UnicodeDecodeError, PermissionError):
                                            pass

                        if node_name in ("executor", "debugger", "tester", "reviewer"):
                            exec_res = node_state.get("execution_results")
                            if exec_res and len(exec_res) > 0:
                                last_exec = exec_res[-1]
                                output = f"$ {last_exec.command}\n{last_exec.stdout}\n{last_exec.stderr}"
                                emit_event("terminal_output", output, agent=node_name)

                        if node_state.get("rollback_applied"):
                            final_status = "rolled_back"
                            final_message = node_state.get("rollback_reason")
                            emit_event("status", final_message or "Project restored to the last stable snapshot.", agent=node_name)
                                
        final_project_dir = _get_project_dir(thread_id)
        if os.path.exists(final_project_dir):
            try:
                env_sync_result = auto_fill_project_env(final_project_dir)
                if env_sync_result.written_files:
                    emit_event(
                        "status",
                        f"Auto-filled environment values in {len(env_sync_result.written_files)} generated env file(s).",
                    )
                elif env_sync_result.filled_keys:
                    emit_event("status", "Generated env files already had usable provider values.")
                if env_sync_result.missing_keys:
                    preview = ", ".join(env_sync_result.missing_keys[:12])
                    extra = f" (+{len(env_sync_result.missing_keys) - 12} more)" if len(env_sync_result.missing_keys) > 12 else ""
                    emit_event("status", f"Missing generated-project env values: {preview}{extra}.")
            except Exception as env_sync_error:
                logger.warning("Environment auto-fill skipped for %s: %s", thread_id, env_sync_error)

        try:
            emit_event("status", "Refreshing project semantic index...")
            _sync_project_index(thread_id)
        except Exception as index_error:
            emit_event("status", f"Semantic index refresh skipped: {index_error}")

        if os.path.exists(final_project_dir):
            try:
                final_done_payload["structure"] = _build_project_structure(final_project_dir)
                final_done_payload["run_plan"] = build_run_plan(final_project_dir)
            except Exception as metadata_error:
                logger.warning("Final project metadata build failed for %s: %s", thread_id, metadata_error)

        print(f"[server] ✅ Pipeline finished for thread {thread_id}.")
        emit_event("status", "Pipeline finished.")
        with transactional_session() as session:
            db_proj = session.query(ProjectGeneration).filter_by(id=thread_id).first()
            if db_proj:
                db_proj.status = final_status
        time.sleep(FINAL_STATUS_EVENT_DELAY_SECONDS)
                
    except Exception as e:
        final_status = "failed"
        final_message = str(e)
        print(f"[server] ❌ Pipeline failed for thread {thread_id}: {e}")
        traceback.print_exc()
        emit_event("status", f"Pipeline failed: {e}")
        with transactional_session() as session:
            db_proj = session.query(ProjectGeneration).filter_by(id=thread_id).first()
            if db_proj:
                db_proj.status = "failed"
        time.sleep(FINAL_STATUS_EVENT_DELAY_SECONDS)
    finally:
        live_updates_stop.set()
        if live_updates_thread and live_updates_thread.is_alive():
            live_updates_thread.join(timeout=max(1.0, LIVE_FILE_STREAM_INTERVAL_SECONDS * 3))

        final_project_dir = _get_project_dir(thread_id)
        final_done_payload["status"] = final_status
        final_done_payload["message"] = final_message
        if os.path.exists(final_project_dir):
            try:
                if not final_done_payload.get("structure"):
                    final_done_payload["structure"] = _build_project_structure(final_project_dir)
                run_plan = final_done_payload.get("run_plan") or {}
                if not run_plan.get("commands"):
                    final_done_payload["run_plan"] = build_run_plan(final_project_dir)
            except Exception as metadata_error:
                logger.warning("Completion payload refresh failed for %s: %s", thread_id, metadata_error)

        _release_pipeline_thread(thread_id)
        emit_event("done", final_done_payload)
        with state_lock:
            stream_queues.pop(thread_id, None)

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=5, max_length=20000)
    auto_model_fallback_on_limit: bool = True

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)


class SaveFileRequest(BaseModel):
    filepath: str = Field(min_length=1, max_length=400)
    content: str

@app.post("/api/generate")
async def start_generation(req: GenerateRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    thread_id = f"run-{uuid.uuid4().hex[:12]}"
    if not _reserve_pipeline_thread(thread_id):
        raise HTTPException(status_code=409, detail="A pipeline is already running for this project.")

    try:
        db_proj = ProjectGeneration(id=thread_id, prompt=req.prompt, status="running")
        db.add(db_proj)
        db.commit()

        with state_lock:
            stream_queues[thread_id] = []

        loop = asyncio.get_running_loop()
        background_tasks.add_task(
            run_project_generation,
            thread_id,
            req.prompt,
            loop,
            False,
            None,
            req.auto_model_fallback_on_limit,
        )

        return api_success({"thread_id": thread_id})
    except Exception:
        _release_pipeline_thread(thread_id)
        raise

@app.post("/api/projects/{thread_id}/update")
async def update_project(thread_id: str, req: GenerateRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    db_proj = db.query(ProjectGeneration).filter_by(id=thread_id).first()
    if not db_proj:
        raise HTTPException(status_code=404, detail="Project not found")
    if not _reserve_pipeline_thread(thread_id):
        raise HTTPException(status_code=409, detail="This project already has a pipeline in progress.")

    try:
        project_dir = _get_project_dir(thread_id)
        if not os.path.exists(project_dir):
            raise HTTPException(status_code=404, detail="Project directory not found")
        rollback_snapshot_path = create_project_snapshot(project_dir)

        db_proj.status = "running"
        db_proj.prompt += f"\n\n[UPDATE]: {req.prompt}"
        db.commit()

        with state_lock:
            stream_queues[thread_id] = []

        loop = asyncio.get_running_loop()
        background_tasks.add_task(
            run_project_generation,
            thread_id,
            req.prompt,
            loop,
            True,
            rollback_snapshot_path,
            req.auto_model_fallback_on_limit,
        )

        return api_success({"thread_id": thread_id})
    except Exception:
        _release_pipeline_thread(thread_id)
        raise

@app.post("/api/projects/{thread_id}/resume")
async def resume_project(thread_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    db_proj = db.query(ProjectGeneration).filter_by(id=thread_id).first()
    if not db_proj:
        raise HTTPException(status_code=404, detail="Project not found")
    if not _reserve_pipeline_thread(thread_id):
        raise HTTPException(status_code=409, detail="This project already has a pipeline in progress.")

    try:
        db_proj.status = "running"
        db.commit()

        with state_lock:
            stream_queues[thread_id] = []

        loop = asyncio.get_running_loop()
        background_tasks.add_task(
            run_project_generation,
            thread_id,
            db_proj.prompt,
            loop,
            False,
            None,
            True,
            True,
        )

        return api_success({"thread_id": thread_id})
    except Exception:
        _release_pipeline_thread(thread_id)
        raise


@app.post("/api/projects/upload")
async def upload_project_archive(
    archive: UploadFile | None = File(default=None),
    files: list[UploadFile] | None = File(default=None),
    relative_paths: list[str] | None = Form(default=None),
    project_name: str | None = Form(default=None),
    db: Session = Depends(get_db),
):
    import os as _os
    import tempfile

    uploaded_files = files or []
    if archive is None and not uploaded_files:
        raise HTTPException(status_code=400, detail="Choose a zip archive, files, or a project folder to import.")
    if archive is not None and uploaded_files:
        raise HTTPException(status_code=400, detail="Upload either a zip archive or files/folder, not both.")

    thread_id = f"run-{uuid.uuid4().hex[:12]}"
    temp_path = None
    try:
        project_dir = _get_project_dir(thread_id)
        if archive is not None:
            filename = archive.filename or "project.zip"
            if not filename.lower().endswith(".zip"):
                raise HTTPException(status_code=400, detail="Please upload a .zip file containing your project.")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
                temp_path = temp_file.name
                while True:
                    chunk = await archive.read(1024 * 1024)
                    if not chunk:
                        break
                    temp_file.write(chunk)

            imported_files = extract_project_zip(temp_path, project_dir)
            default_name = os.path.splitext(filename)[0]
        else:
            if relative_paths and len(relative_paths) != len(uploaded_files):
                raise HTTPException(status_code=400, detail="Uploaded file paths did not match the selected files.")

            file_entries: list[tuple[str, bytes]] = []
            for index, uploaded_file in enumerate(uploaded_files):
                relative_path = (
                    relative_paths[index]
                    if relative_paths and index < len(relative_paths)
                    else (uploaded_file.filename or f"file-{index + 1}")
                )
                file_entries.append((relative_path, await uploaded_file.read()))

            imported_files = import_project_files(file_entries, project_dir)
            if relative_paths and len(relative_paths) > 0:
                normalized = relative_paths[0].replace("\\", "/").strip("/")
                default_name = normalized.split("/", 1)[0] or "files"
            else:
                default_name = os.path.splitext(uploaded_files[0].filename or "files")[0]

        prompt = project_name.strip() if project_name and project_name.strip() else f"Imported project: {default_name}"

        db_proj = ProjectGeneration(id=thread_id, prompt=prompt, status="done")
        db.add(db_proj)
        db.commit()

        try:
            _sync_project_index(thread_id)
        except Exception as exc:
            logger.warning("Project index refresh failed after upload for %s: %s", thread_id, exc)

        return api_success({"thread_id": thread_id, "files_imported": len(imported_files)}, message="Project imported")
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid zip archive.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        if archive is not None:
            await archive.close()
        for uploaded_file in uploaded_files:
            await uploaded_file.close()
        if temp_path and _os.path.exists(temp_path):
            _os.unlink(temp_path)


def _get_project_dir(thread_id: str) -> str:
    return os.path.abspath(os.path.join("generated_project", thread_id))


def _build_project_structure(project_dir: str) -> list[dict]:
    if not os.path.exists(project_dir):
        return []

    structure: list[dict] = []
    for current_root, dirs, filenames in os.walk(project_dir):
        dirs[:] = [d for d in sorted(dirs) if d not in SKIP_DIRS]
        filenames = [f for f in sorted(filenames) if f not in SKIP_FILES_SET]
        relative_root = os.path.relpath(current_root, project_dir).replace("\\", "/")

        if relative_root != ".":
            structure.append(
                {
                    "type": "directory",
                    "path": relative_root,
                    "name": os.path.basename(current_root),
                    "depth": relative_root.count("/"),
                }
            )

        for filename in filenames:
            relative_path = os.path.relpath(os.path.join(current_root, filename), project_dir).replace("\\", "/")
            structure.append(
                {
                    "type": "file",
                    "path": relative_path,
                    "name": filename,
                    "depth": relative_path.count("/"),
                }
            )

    return structure


def _iter_streamable_project_files(project_dir: str):
    if not os.path.exists(project_dir):
        return

    for root, dirs, filenames in os.walk(project_dir):
        dirs[:] = [d for d in sorted(dirs) if d not in SKIP_DIRS]
        for filename in sorted(filenames):
            if filename in SKIP_FILES_SET:
                continue

            filepath = os.path.join(root, filename)
            rel_path = os.path.relpath(filepath, project_dir).replace("\\", "/")
            try:
                file_size = os.path.getsize(filepath)
            except OSError:
                continue

            if file_size > PROJECT_FILE_LOAD_MAX_BYTES:
                continue

            yield rel_path, filepath


def _snapshot_project_files(project_dir: str) -> dict[str, tuple[int, int]]:
    snapshots: dict[str, tuple[int, int]] = {}
    for rel_path, filepath in _iter_streamable_project_files(project_dir) or []:
        try:
            stat = os.stat(filepath)
        except OSError:
            continue
        snapshots[rel_path] = (stat.st_mtime_ns, stat.st_size)
    return snapshots


def _stream_live_project_files(
    thread_id: str,
    emit_event: Callable[[str, Any, str | None], None],
    stop_event: threading.Event,
    *,
    seed_existing: bool = False,
) -> None:
    project_dir = _get_project_dir(thread_id)
    known_files = _snapshot_project_files(project_dir) if seed_existing else {}

    while not stop_event.is_set():
        try:
            current_files: dict[str, tuple[int, int]] = {}
            for rel_path, filepath in _iter_streamable_project_files(project_dir) or []:
                try:
                    stat = os.stat(filepath)
                except OSError:
                    continue

                signature = (stat.st_mtime_ns, stat.st_size)
                current_files[rel_path] = signature
                if known_files.get(rel_path) == signature:
                    continue

                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                except (OSError, UnicodeDecodeError, PermissionError):
                    continue

                known_files[rel_path] = signature
                emit_event("file_update", {"filepath": rel_path, "content": content}, agent="watcher")

            for rel_path in list(known_files.keys()):
                if rel_path not in current_files:
                    known_files.pop(rel_path, None)
        except Exception as exc:
            logger.debug("Live file watcher skipped a poll for %s: %s", thread_id, exc)

        stop_event.wait(LIVE_FILE_STREAM_INTERVAL_SECONDS)

def _build_project_context_legacy(thread_id: str, max_chars_per_file: int = 1500) -> str:
    """Build a compact summary of the project's files for the chat system prompt."""
    project_dir = _get_project_dir(thread_id)
    if not os.path.exists(project_dir):
        return "(No project files found yet.)"

    SKIP_DIRS = {"node_modules", ".git", "__pycache__", ".venv", "venv", ".next", "dist", "build"}
    SKIP_FILES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml"}
    parts = []
    total_chars = 0
    MAX_TOTAL = 15000  # Cap total context to avoid blowing token limits

    for root, dirs, filenames in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for filename in filenames:
            if filename in SKIP_FILES:
                continue
            filepath = os.path.join(root, filename)
            rel_path = os.path.relpath(filepath, project_dir).replace("\\", "/")
            try:
                file_size = os.path.getsize(filepath)
                if file_size > 50_000:
                    parts.append(f"### {rel_path}\n(Large file — {file_size} bytes, skipped)")
                    continue
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read(max_chars_per_file)
                if len(content) >= max_chars_per_file:
                    content += "\n... (truncated)"
                parts.append(f"### {rel_path}\n```\n{content}\n```")
                total_chars += len(content)
                if total_chars > MAX_TOTAL:
                    parts.append("\n(... remaining files omitted for brevity)")
                    break
            except (UnicodeDecodeError, PermissionError):
                parts.append(f"### {rel_path}\n(Binary file, skipped)")
        if total_chars > MAX_TOTAL:
            break

    return "\n\n".join(parts) if parts else "(No project files found yet.)"

SKIP_DIRS = {"node_modules", ".git", "__pycache__", ".venv", "venv", ".next", "dist", "build"}
SKIP_FILES_SET = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml"}
CHAT_TOOL_READ_MAX_BYTES = int(os.getenv("CHAT_TOOL_READ_MAX_BYTES", "100000"))
CHAT_TOOL_TREE_MAX_ENTRIES = int(os.getenv("CHAT_TOOL_TREE_MAX_ENTRIES", "2000"))
PROJECT_FILE_LOAD_MAX_BYTES = int(os.getenv("PROJECT_FILE_LOAD_MAX_BYTES", "100000"))
LIVE_FILE_STREAM_INTERVAL_SECONDS = float(os.getenv("LIVE_FILE_STREAM_INTERVAL_SECONDS", "0.35"))


def _build_fallback_project_context(thread_id: str, max_chars_per_file: int = 800) -> str:
    project_dir = _get_project_dir(thread_id)
    if not os.path.exists(project_dir):
        return "(No project files found yet.)"

    parts = []
    total_chars = 0
    max_total = min(PROJECT_CONTEXT_MAX_CHARS, 8000)

    for root, dirs, filenames in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for filename in filenames:
            if filename in SKIP_FILES_SET:
                continue
            filepath = os.path.join(root, filename)
            rel_path = os.path.relpath(filepath, project_dir).replace("\\", "/")
            try:
                file_size = os.path.getsize(filepath)
                if file_size > PROJECT_CONTEXT_MAX_FILE_BYTES:
                    parts.append(f"### {rel_path}\n(Large file - {file_size} bytes, skipped)")
                    continue
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read(max_chars_per_file)
                if len(content) >= max_chars_per_file:
                    content += "\n... (truncated)"
                parts.append(f"### {rel_path}\n```\n{content}\n```")
                total_chars += len(content)
                if total_chars > max_total:
                    parts.append("\n(... remaining files omitted for brevity)")
                    break
            except (UnicodeDecodeError, PermissionError):
                parts.append(f"### {rel_path}\n(Binary file, skipped)")
        if total_chars > max_total:
            break

    return "\n\n".join(parts) if parts else "(No project files found yet.)"


def _build_project_context(thread_id: str, user_message: str, history: list[dict] | None = None) -> str:
    project_dir = _get_project_dir(thread_id)
    if not os.path.exists(project_dir):
        return "(No project files found yet.)"

    try:
        _sync_project_index(thread_id)
        retrieval_query = _build_retrieval_query(user_message, history)
        retrieved_chunks = _retrieve_project_chunks(thread_id, retrieval_query)
        if not retrieved_chunks:
            return _build_fallback_project_context(thread_id)

        parts = []
        total_chars = 0
        for chunk in retrieved_chunks:
            snippet = chunk["content"]
            remaining = PROJECT_CONTEXT_MAX_CHARS - total_chars
            if remaining <= 0:
                break
            if len(snippet) > remaining:
                snippet = snippet[:remaining] + "\n... (truncated)"

            parts.append(f"### {chunk['path']} [chunk {chunk['chunk_index']}]")
            parts.append(f"```\n{snippet}\n```")
            total_chars += len(snippet)

        return "\n\n".join(parts) if parts else _build_fallback_project_context(thread_id)
    except Exception as exc:
        logger.warning("Project context retrieval failed for %s: %s", thread_id, exc)
        recovery = _recover_project_state(thread_id)
        if recovery.get("status") != "recovered":
            logger.warning("Project context recovery failed for %s: %s", thread_id, recovery.get("reason", "unknown"))
        return _build_fallback_project_context(thread_id)


def _resolve_chat_tool_target(project_dir: str, path: str | None) -> tuple[str | None, str | None]:
    requested_path = path or "."
    project_root = os.path.abspath(project_dir)
    target = os.path.abspath(os.path.join(project_root, requested_path))
    try:
        if os.path.commonpath([project_root, target]) != project_root:
            return None, "Error: Access denied - path is outside the project directory."
    except ValueError:
        return None, "Error: Access denied - path is outside the project directory."
    return target, None


def _is_chat_tool_visible(name: str, is_dir: bool) -> bool:
    if is_dir:
        return name not in SKIP_DIRS
    return name not in SKIP_FILES_SET

def _legacy_get_chat_tools(project_dir: str):
    
    @tool
    def list_directory(path: str) -> str:
        """List all files and subdirectories at a given path inside the project. Use '.' for the project root."""
        target = os.path.normpath(os.path.join(project_dir, path))
        if not target.startswith(project_dir):
            return "Error: Access denied — path is outside the project directory."
        if not os.path.exists(target):
            return f"Error: Path '{path}' does not exist."
        if not os.path.isdir(target):
            return f"Error: '{path}' is a file, not a directory. Use read_file to read it."

        entries = []
        try:
            for entry in sorted(os.listdir(target)):
                if entry in SKIP_DIRS or entry.startswith('.'):
                    continue
                full = os.path.join(target, entry)
                rel = os.path.relpath(full, project_dir).replace("\\", "/")
                if os.path.isdir(full):
                    entries.append(f"📁 {rel}/")
                else:
                    size = os.path.getsize(full)
                    entries.append(f"📄 {rel}  ({size} bytes)")
        except PermissionError:
            return f"Error: Permission denied for '{path}'."
        return "\n".join(entries) if entries else "(empty directory)"


    @tool
    def read_file(path: str) -> str:
        """Read the full content of a specific file in the project. Always use list_directory first to discover available files."""
        target = os.path.normpath(os.path.join(project_dir, path))
        if not target.startswith(project_dir):
            return "Error: Access denied — path is outside the project directory."
        if not os.path.exists(target):
            return f"Error: File '{path}' does not exist."
        if os.path.isdir(target):
            return f"Error: '{path}' is a directory. Use list_directory to list it."
        try:
            size = os.path.getsize(target)
            if size > 100_000:
                return f"Error: File is too large ({size} bytes). Only files under 100KB can be read."
            with open(target, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            return f"Error: '{path}' is a binary file and cannot be read as text."
        except PermissionError:
            return f"Error: Permission denied for '{path}'."

    @tool
    def trigger_update(update_prompt: str) -> str:
        """Request user approval to run the project update pipeline with a precise update prompt."""
        return "Update approval requested. Once the user approves, the UI should start the update pipeline with this prompt."

    return [list_directory, read_file, trigger_update]


def get_chat_tools(project_dir: str):
    project_root = os.path.abspath(project_dir)

    @tool
    def list_directory(path: str) -> str:
        """List all visible files and subdirectories at a given path inside the project. Use '.' for the project root."""
        target, error = _resolve_chat_tool_target(project_root, path)
        if error:
            return error
        if not os.path.exists(target):
            return f"Error: Path '{path}' does not exist."
        if not os.path.isdir(target):
            return f"Error: '{path}' is a file, not a directory. Use read_file to read it."

        entries = []
        try:
            for entry in sorted(os.listdir(target)):
                full = os.path.join(target, entry)
                if not _is_chat_tool_visible(entry, os.path.isdir(full)):
                    continue
                rel = os.path.relpath(full, project_root).replace("\\", "/")
                if os.path.isdir(full):
                    entries.append(f"[DIR] {rel}/")
                else:
                    size = os.path.getsize(full)
                    entries.append(f"[FILE] {rel} ({size} bytes)")
        except PermissionError:
            return f"Error: Permission denied for '{path}'."

        listing = "\n".join(entries) if entries else "(empty directory)"
        return f"{listing}\n\nTotal entries: {len(entries)}"

    @tool
    def list_project_tree(path: str = ".") -> str:
        """Recursively list the visible directories and files under a path. Use this when you need the whole project tree in one call."""
        target, error = _resolve_chat_tool_target(project_root, path)
        if error:
            return error
        if not os.path.exists(target):
            return f"Error: Path '{path}' does not exist."
        if not os.path.isdir(target):
            return f"Error: '{path}' is a file, not a directory. Use read_file to read it."

        display_root = os.path.relpath(target, project_root).replace("\\", "/")
        if display_root == ".":
            display_root = "."

        lines = [f"[DIR] {display_root}/"]
        directory_count = 1
        file_count = 0
        shown_entries = 1
        truncated = False

        for current_root, dirs, filenames in os.walk(target):
            dirs[:] = [d for d in sorted(dirs) if _is_chat_tool_visible(d, True)]
            filenames = [f for f in sorted(filenames) if _is_chat_tool_visible(f, False)]
            relative_root = os.path.relpath(current_root, target).replace("\\", "/")
            depth = 0 if relative_root == "." else relative_root.count("/") + 1

            if current_root != target:
                lines.append(f"{'  ' * depth}[DIR] {os.path.basename(current_root)}/")
                directory_count += 1
                shown_entries += 1
                if shown_entries >= CHAT_TOOL_TREE_MAX_ENTRIES:
                    truncated = True
                    break

            for filename in filenames:
                full_path = os.path.join(current_root, filename)
                try:
                    size = os.path.getsize(full_path)
                except OSError:
                    size = None
                size_suffix = f" ({size} bytes)" if size is not None else ""
                lines.append(f"{'  ' * (depth + 1)}[FILE] {filename}{size_suffix}")
                file_count += 1
                shown_entries += 1
                if shown_entries >= CHAT_TOOL_TREE_MAX_ENTRIES:
                    truncated = True
                    break

            if truncated:
                break

        summary = f"Directories: {directory_count}, files: {file_count}"
        if truncated:
            summary += f" (truncated after {shown_entries} entries)"
        return "\n".join(lines + ["", summary])

    @tool
    def read_file(path: str) -> str:
        """Read the full content of a specific file in the project. Use list_project_tree or list_directory first to discover available files."""
        target, error = _resolve_chat_tool_target(project_root, path)
        if error:
            return error
        if not os.path.exists(target):
            return f"Error: File '{path}' does not exist."
        if os.path.isdir(target):
            return f"Error: '{path}' is a directory. Use list_directory to list it."
        try:
            size = os.path.getsize(target)
            if size > CHAT_TOOL_READ_MAX_BYTES:
                return f"Error: File is too large ({size} bytes). Only files under {CHAT_TOOL_READ_MAX_BYTES} bytes can be read."
            with open(target, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            return f"Error: '{path}' is a binary file and cannot be read as text."
        except PermissionError:
            return f"Error: Permission denied for '{path}'."

    @tool
    def trigger_update(update_prompt: str) -> str:
        """Request user approval to run the project update pipeline with a precise update prompt."""
        return "Update approval requested. Once the user approves, the UI should start the update pipeline with this prompt."

    return [list_project_tree, list_directory, read_file, trigger_update]

CHAT_SYSTEM_TEMPLATE = """You are an expert AI coding assistant embedded in an IDE called AI Studio.
You are helping the user with their project. Be helpful, concise, and precise.

CAPABILITIES:
- You are provided with the most relevant retrieved project context below
- You are provided with recent tool activity from earlier chat turns so you can keep continuity
- You can ALSO browse the project deeper by calling list_project_tree, list_directory, and read_file tools
- You can explain code, debug errors, suggest improvements, and answer questions
- You can ask for approval to run project updates using the trigger_update tool.

RULES:
- Be conversational and friendly
- Reference specific files and line numbers when discussing code
- Format code snippets with markdown code blocks
- If the user asks you to modify/create/delete files or fix a bug, DO NOT just tell them to do it. Invoke the `trigger_update` tool with a highly specific `update_prompt` describing EXACTLY what code changes need to be made.
- After calling `trigger_update`, briefly explain what will change and ask the user to approve the inline update action.
- Keep responses focused — don't dump entire files unless asked
- Use `list_project_tree` when the user asks for all files, the full structure, or a project-wide scan
- Recent tool activity is only a summary; if exact code matters, call read_file again
- ALWAYS use your tools to read files if the context provided is truncated or missing details

## Recent Tool Activity
{recent_tool_activity}

## Retrieved Project Context
{project_context}
"""

@app.post("/api/projects/{thread_id}/chat")
async def chat_with_project(thread_id: str, req: ChatRequest, db: Session = Depends(get_db)):
    db_proj = db.query(ProjectGeneration).filter_by(id=thread_id).first()
    if not db_proj:
        raise HTTPException(status_code=404, detail="Project not found")

    project_dir = _get_project_dir(thread_id)
    lock = _get_chat_lock(thread_id)

    async def generate_stream():
        full_response = ""
        lock_acquired = False
        max_tool_rounds = 8  # Safety limit to prevent infinite loops

        try:
            if lock.locked():
                yield f"data: {json.dumps({'type': 'status', 'content': 'Waiting for the previous chat turn to finish...'})}\n\n"

            await lock.acquire()
            lock_acquired = True
            history = chat_histories.setdefault(thread_id, [])
            project_context = _build_project_context(thread_id, req.message, history)
            recent_tool_activity = _format_recent_tool_activity(thread_id)
            system_msg = CHAT_SYSTEM_TEMPLATE.format(
                recent_tool_activity=recent_tool_activity,
                project_context=project_context,
            )
            messages = [SystemMessage(content=system_msg)]
            for msg in history[-20:]:
                if msg["role"] == "user":
                    messages.append(LCHumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
            messages.append(LCHumanMessage(content=req.message))
            history.append({"role": "user", "content": req.message})
            _trim_chat_history(history)

            chat_tools_list = get_chat_tools(project_dir)
            tools_by_name = {t.name: t for t in chat_tools_list}
            chat_llm_with_tools = _bind_chat_tools(get_chat_llm(), chat_tools_list)
            current_messages = list(messages)
            update_action_prompt: str | None = None

            for _round in range(max_tool_rounds):
                response = await _invoke_chat_llm(chat_llm_with_tools, current_messages)

                # Check if the LLM wants to call tools
                if getattr(response, "tool_calls", None):
                    tool_results = []
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call.get("args", {}) or {}
                        tool_id = tool_call.get("id", "")
                        yield f"data: {json.dumps({'type': 'tool_call', 'name': tool_name, 'args': tool_args, 'round': _round + 1})}\n\n"

                        # Execute the tool
                        if tool_name == "list_project_tree":
                            path = tool_args.get("path", ".")
                            yield f"data: {json.dumps({'type': 'status', 'content': f'Scanning project tree {path}/'})}\n\n"
                            result = tools_by_name["list_project_tree"].invoke({"path": path})
                        elif tool_name == "list_directory":
                            path = tool_args.get("path", ".")
                            yield f"data: {json.dumps({'type': 'status', 'content': f'📁 Browsing {path}/'})}\n\n"
                            result = tools_by_name["list_directory"].invoke({"path": path})
                        elif tool_name == "read_file":
                            path = tool_args.get("path", "")
                            yield f"data: {json.dumps({'type': 'status', 'content': f'📄 Reading {path}'})}\n\n"
                            result = tools_by_name["read_file"].invoke({"path": path})
                        elif tool_name == "trigger_update":
                            update_prompt = tool_args.get("update_prompt", "")
                            update_action_prompt = update_prompt
                            yield f"data: {json.dumps({'type': 'action', 'action_type': 'trigger_update', 'update_prompt': update_prompt})}\n\n"
                            result = tools_by_name["trigger_update"].invoke({"update_prompt": update_prompt})
                        else:
                            result = f"Unknown tool: {tool_name}"

                        _record_chat_tool_activity(thread_id, tool_name, tool_args, str(result))
                        yield f"data: {json.dumps({'type': 'tool_result', 'name': tool_name, 'preview': _clip_chat_text(str(result), 200)})}\n\n"

                        tool_results.append({
                            "tool_call_id": tool_id,
                            "content": result
                        })

                    # Add assistant message with tool calls and tool results
                    current_messages.append(response)
                    for tr in tool_results:
                        current_messages.append(ToolMessage(
                            content=str(tr["content"]),
                            tool_call_id=tr["tool_call_id"]
                        ))
                    if update_action_prompt is not None:
                        response_text = (
                            "I prepared an update for this project. Approve the inline update action "
                            "and I will run the pipeline with the exact requested changes."
                        )
                        full_response += response_text
                        yield f"data: {json.dumps({'type': 'token', 'content': response_text})}\n\n"
                        break
                    continue  # Loop back for another LLM call

                # No tool calls; send the final text through the existing SSE token event.
                response_text = _chat_content_to_text(getattr(response, "content", ""))
                if not response_text:
                    final_response = await _invoke_chat_llm(get_chat_llm(), current_messages)
                    response_text = _chat_content_to_text(getattr(final_response, "content", ""))
                if response_text:
                    full_response += response_text
                    yield f"data: {json.dumps({'type': 'token', 'content': response_text})}\n\n"
                break  # Done

            # Save assistant response to history
            history.append({"role": "assistant", "content": full_response})
            _trim_chat_history(history)
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            logger.exception("Project chat failed for %s", thread_id, exc_info=e)
            yield f"data: {json.dumps({'type': 'error', 'content': _chat_exception_message(e)})}\n\n"
        finally:
            if lock_acquired and lock.locked():
                lock.release()

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


@app.get("/api/projects")
def get_projects(db: Session = Depends(get_db)):
    """Fetches all past project generations for the UI history list."""
    projs = db.query(ProjectGeneration).order_by(ProjectGeneration.created_at.desc()).all()
    return [
        {
            "id": p.id, 
            "prompt": p.prompt, 
            "status": p.status, 
            "created_at": str(p.created_at)
        } 
        for p in projs
    ]

@app.delete("/api/projects/{thread_id}")
async def delete_project(thread_id: str, db: Session = Depends(get_db)):
    """Deletes a project and all its associated data."""
    db_proj = db.query(ProjectGeneration).filter_by(id=thread_id).first()
    if not db_proj:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if _is_pipeline_running(thread_id):
        raise HTTPException(status_code=409, detail="Cannot delete a project that is currently running.")
    
    db.delete(db_proj)
    db.commit()
    
    import shutil
    project_dir = _get_project_dir(thread_id)
    if os.path.exists(project_dir):
        try:
            shutil.rmtree(project_dir)
        except Exception as e:
            print(f"Error deleting project files for {thread_id}: {e}")
        
    index_dir = _get_project_index_dir(thread_id)
    if os.path.exists(index_dir):
        try:
            shutil.rmtree(index_dir)
        except Exception as e:
            print(f"Error deleting index files for {thread_id}: {e}")
        
    import sqlite3
    try:
        conn = sqlite3.connect("pipeline_checkpoints.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        cursor.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error deleting checkpoints for {thread_id}: {e}")

    chat_histories.pop(thread_id, None)
    chat_turn_locks.pop(thread_id, None)
    chat_tool_activity.pop(thread_id, None)
    
    return {"status": "success", "message": f"Project {thread_id} deleted."}

@app.get("/api/projects/{thread_id}")
def get_project(thread_id: str, db: Session = Depends(get_db)):
    db_proj = db.query(ProjectGeneration).filter_by(id=thread_id).first()
    if not db_proj:
        raise HTTPException(status_code=404, detail="Project not found")
    
    status = db_proj.status
    files = []
    project_dir = os.path.abspath(os.path.join("generated_project", thread_id))
    # Directories to skip — these make loading extremely slow
    if os.path.exists(project_dir):
        for root, dirs, filenames in os.walk(project_dir):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for filename in filenames:
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, project_dir).replace("\\", "/")
                # Skip lock files and other generated artifacts
                if filename in ("package-lock.json", "yarn.lock", "pnpm-lock.yaml"):
                    continue
                try:
                    file_size = os.path.getsize(filepath)
                    if file_size > PROJECT_FILE_LOAD_MAX_BYTES:
                        continue  # skip very large files
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                    files.append({"filepath": rel_path, "content": content})
                except (UnicodeDecodeError, PermissionError):
                    pass  # ignore binary / locked files

    terminal_output = ""
    try:
        import concurrent.futures
        # get_run_state rebuilds the graph + loads checkpoint — can be very slow.
        # Use a 2s timeout so project loading is never blocked by this.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            graph = _get_graph()
            future = pool.submit(graph.get_run_state, thread_id)
            run_state = future.result(timeout=2)
            exec_res = run_state.values.get("execution_results")
            if exec_res and len(exec_res) > 0:
                last_exec = exec_res[-1]
                terminal_output = f"$ {last_exec.command}\n{last_exec.stdout}\n{last_exec.stderr}"
    except Exception:
        pass  # terminal output is nice-to-have, not a blocker

    structure = _build_project_structure(project_dir)
    run_plan = build_run_plan(project_dir) if os.path.exists(project_dir) else {"commands": [], "html_file": None, "html_path": None, "serve_port": None}

    return {
        "id": thread_id,
        "prompt": db_proj.prompt,
        "status": status,
        "files": files,
        "terminal_output": terminal_output,
        "structure": structure,
        "run_plan": run_plan,
    }

@app.post("/api/projects/{thread_id}/file")
async def save_file(thread_id: str, payload: SaveFileRequest):
    filepath = payload.filepath
    content = payload.content
        
    project_dir = os.path.abspath(os.path.join("generated_project", thread_id))
    full_path = os.path.abspath(os.path.join(project_dir, filepath))
    
    if not full_path.startswith(project_dir):
        raise HTTPException(status_code=400, detail="Invalid path")
        
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)

    index_warning = None
    try:
        _sync_project_index(thread_id)
    except Exception as exc:
        index_warning = str(exc)

    response = api_success({"thread_id": thread_id}, message="File saved")
    if index_warning:
        response["index_warning"] = index_warning
    return response

@app.get("/api/projects/{thread_id}/download")
def download_project(thread_id: str):
    project_dir = os.path.abspath(os.path.join("generated_project", thread_id))
    if not os.path.exists(project_dir):
        raise HTTPException(status_code=404, detail="Project not found")
        
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, filenames in os.walk(project_dir):
            for filename in filenames:
                filepath = os.path.join(root, filename)
                arcname = os.path.relpath(filepath, project_dir)
                zf.write(filepath, arcname)
    
    memory_file.seek(0)
    return StreamingResponse(
        memory_file,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={thread_id}.zip"}
    )

class PtyProcess:
    def __init__(self, websocket: WebSocket, cwd: str):
        self.websocket = websocket
        self.cwd = cwd
        self.process = None
        self.running = False
        self._flush_interval_seconds = 0.03
        self._max_chunk_bytes = 256
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            
    def start(self):
        cmd = ["cmd.exe", "/Q"] if os.name == 'nt' else ["bash"]
        self.process = subprocess.Popen(
            cmd,
            cwd=self.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        )
        self.running = True
        threading.Thread(target=self._read_stdout, daemon=True).start()

    def _send_output(self, text: str) -> None:
        if not text:
            return
        try:
            asyncio.run_coroutine_threadsafe(self.websocket.send_text(text), self.loop)
        except RuntimeError:
            pass
        
    def _read_stdout(self):
        pending = bytearray()
        last_flush = time.monotonic()

        while self.running and self.process.poll() is None:
            data = self.process.stdout.read(1)
            if not data:
                break

            pending.extend(data)
            now = time.monotonic()
            should_flush = (
                data in (b"\n", b"\r")
                or len(pending) >= self._max_chunk_bytes
                or (now - last_flush) >= self._flush_interval_seconds
            )
            if should_flush:
                self._send_output(pending.decode('utf-8', errors='replace'))
                pending.clear()
                last_flush = now

        if pending:
            self._send_output(pending.decode('utf-8', errors='replace'))

        exit_code = self.process.poll() if self.process else None
        if self.running and exit_code is not None:
            self._send_output(f"\r\n\x1b[31m[Process exited with code {exit_code}]\x1b[0m\r\n")
        self.running = False
            
    def write(self, text: str):
        if self.running and self.process.poll() is None:
            self.process.stdin.write(text.encode('utf-8'))
            self.process.stdin.flush()
            
    def stop(self):
        self.running = False
        if self.process:
            try:
                if os.name == 'nt':
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.process.pid)], capture_output=True)
                else:
                    self.process.terminate()
            except Exception:
                pass

active_terminals = {}

@app.websocket("/ws/terminal/{thread_id}")
@app.websocket("/ws/terminal/{thread_id}/{term_id}")
async def terminal_websocket(websocket: WebSocket, thread_id: str, term_id: str = "default"):
    await websocket.accept()
    project_dir = os.path.abspath(os.path.join("generated_project", thread_id))
    if not os.path.exists(project_dir):
        await websocket.send_text("Project not found.\r\n")
        await websocket.close()
        return

    # If an old terminal exists for this thread, we could stop it, but here we just create a new one.
    pty = PtyProcess(websocket, project_dir)
    pty.start()
    
    try:
        while True:
            data = await websocket.receive_text()
            pty.write(data)
    except WebSocketDisconnect:
        pass
    finally:
        pty.stop()

def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

@app.post("/api/projects/{thread_id}/run")
def run_project_app(thread_id: str):
    project_dir = os.path.abspath(os.path.join("generated_project", thread_id))
    if not os.path.exists(project_dir):
        raise HTTPException(status_code=404, detail="Project directory not found")
    return build_run_plan(project_dir)

    has_package_json = os.path.exists(os.path.join(project_dir, "package.json"))
    has_requirements = os.path.exists(os.path.join(project_dir, "requirements.txt"))

    SKIP_DIRS = {"node_modules", ".git", "__pycache__", ".venv", "venv"}
    index_html_path = None   # Best match: index.html
    fallback_html_path = None  # Any other .html file
    index_html_dir = None
    fallback_html_dir = None

    for root, dirs, files in os.walk(project_dir):
        # Prune directories we never want to search
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if f.lower() == "index.html" and index_html_path is None:
                index_html_path = f
                index_html_dir = root
            elif f.endswith('.html') and fallback_html_path is None:
                fallback_html_path = f
                fallback_html_dir = root
        # Stop early if we found index.html
        if index_html_path:
            break

    # Prefer index.html, fall back to any .html
    html_file = index_html_path or fallback_html_path
    html_dir = index_html_dir or fallback_html_dir
    has_html = html_file is not None

    # Build cd command prefix - terminals start in project_dir so this is relative
    html_rel_dir = None
    html_rel_path = None  # relative path from project root to the HTML file
    if html_dir:
        html_rel_dir = os.path.relpath(html_dir, project_dir).replace("\\", "/")
        html_rel_path = (html_rel_dir + "/" + html_file) if html_rel_dir != "." else html_file
        if html_rel_dir == ".":
            html_rel_dir = None  # already in project root, no cd needed

    serve_port = 3000

    def html_serve_cmd(port=3000):
        """Return a command that cds into the HTML directory (if needed) and serves it."""
        if html_rel_dir:
            return f"cd {html_rel_dir} && python -m http.server {port}\r\n"
        return f"python -m http.server {port}\r\n"

    commands = []

    if has_package_json and has_html:
        # Full-stack: Node backend + static HTML frontend
        commands.append({"name": "Backend", "cmd": "npm install && npm start\r\n"})
        commands.append({"name": "Frontend", "cmd": html_serve_cmd(serve_port)})
    elif has_package_json:
        # Node only
        commands.append({"name": "Backend", "cmd": "npm install && npm start\r\n"})
    elif has_html:
        # Pure static site
        commands.append({"name": "Frontend", "cmd": html_serve_cmd(serve_port)})
    elif has_requirements:
        # Python backend
        app_file = "app.py" if os.path.exists(os.path.join(project_dir, "app.py")) else "main.py"
        commands.append({"name": "Backend", "cmd": f"pip install -r requirements.txt && python {app_file}\r\n"})
    else:
        # Fallback: serve from project root
        commands.append({"name": "Server", "cmd": f"python -m http.server {serve_port}\r\n"})

    return {
        "commands": commands,
        "html_file": html_file,               # e.g. "index.html"
        "html_path": html_rel_path,            # e.g. "client/public/index.html"
        "serve_port": serve_port if has_html else None,
    }

@app.post("/api/projects/{thread_id}/stop")
def stop_project_app(thread_id: str):
    """Deprecated: Terminals are now managed explicitly by the user."""
    return {"status": "deprecated"}

@app.websocket("/ws/stream/{thread_id}")
async def websocket_endpoint(websocket: WebSocket, thread_id: str):
    await websocket.accept()
    with state_lock:
        if thread_id not in stream_queues:
            stream_queues[thread_id] = []
        
    q = asyncio.Queue()
    with state_lock:
        stream_queues[thread_id].append(q)
    
    try:
        while True:
            event = await q.get()
            await websocket.send_json(event)
            if event["type"] == "done": break
    except WebSocketDisconnect:
        pass
    finally:
        with state_lock:
            if thread_id in stream_queues and q in stream_queues[thread_id]:
                stream_queues[thread_id].remove(q)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
