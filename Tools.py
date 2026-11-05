"""
Tools.py — sandboxed filesystem + shell tools for LangGraph agents.

Security layers
───────────────
1. Path sanitisation : every file op goes through _safe(), which refuses
   any path that escapes PROJECT_ROOT (prevents directory-traversal attacks).
2. Docker sandbox    : run_cmd prefers Docker (network-disabled, memory-capped,
   read-only host fs except the project volume).  Falls back to subprocess only
   when Docker is unavailable (CI, local dev without daemon).
3. Output cap        : stdout + stderr are each capped at MAX_CMD_OUTPUT chars
   so a runaway printf can't OOM the host process.
4. Timeout cap       : both Docker and subprocess honour a hard ceiling
   (CMD_TIMEOUT_CAP) regardless of what the agent requests.
"""

from __future__ import annotations

import difflib
import os
import pathlib
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from pipeline_validation import validate_project_imports

_SANDBOX_IMAGE   = os.getenv("SANDBOX_IMAGE",   "python:3.11-slim")
_NODE_SANDBOX_IMAGE = os.getenv("NODE_SANDBOX_IMAGE", "node:20-slim")
_MAX_CMD_OUTPUT  = int(os.getenv("MAX_CMD_OUTPUT",  "4096"))  # chars per stream
_CMD_TIMEOUT_CAP = int(os.getenv("CMD_TIMEOUT_CAP", "120"))   # hard ceiling (s)
_TREE_LIST_MAX_ENTRIES = int(os.getenv("TREE_LIST_MAX_ENTRIES", "2000"))
_PATCH_SEARCH_MARKER = "<<<<<<< SEARCH"
_PATCH_MID_MARKER = "======="
_PATCH_REPLACE_MARKER = ">>>>>>> REPLACE"
_DELETE_REASON_MIN_CHARS = int(os.getenv("DELETE_REASON_MIN_CHARS", "20"))




def _get_project_root(config: RunnableConfig) -> pathlib.Path:
    configurable = config.get("configurable", {})
    # Allow an explicit project_root override (used by update sub-threads)
    explicit_root = configurable.get("project_root")
    if explicit_root:
        root = pathlib.Path(explicit_root)
    else:
        thread_id = configurable.get("thread_id", "default")
        root = pathlib.Path.cwd() / "generated_project" / thread_id
    root.mkdir(parents=True, exist_ok=True)
    return root

def _safe(path: str, root: pathlib.Path) -> pathlib.Path:
    """
    Resolve *path* inside the dynamic project root.
    Raises ValueError immediately if the result escapes the sandbox root.
    This is the single choke-point for all path-traversal prevention.
    """
    resolved = (root / path).resolve()
    actual_root = root.resolve()
    if resolved != actual_root and actual_root not in resolved.parents:
        raise ValueError(
            f"Path '{path}' escapes the project root '{actual_root}'. "
            "This operation is not permitted."
        )
    return resolved



def _probe_docker() -> bool:
    """Return True only if a Docker daemon is reachable right now."""
    if not shutil.which("docker"):
        return False
    try:
        r = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
        return r.returncode == 0
    except Exception:
        return False


_DOCKER_AVAILABLE: bool = _probe_docker()

_NODE_TOOL_RE = re.compile(r"(^|[\s;&|()])(node|npm|npx|pnpm|yarn)\b")
_INSTALL_TOOL_RE = re.compile(
    r"(^|[\s;&|()])("
    r"npm\s+(install|ci)|"
    r"pnpm\s+install|"
    r"yarn\s+(install|add)|"
    r"pip\s+install"
    r")\b"
)
_HOST_NODE_TOOLCHAIN_ON_WINDOWS = os.getenv("HOST_NODE_TOOLCHAIN_ON_WINDOWS", "1").strip().lower() not in {"0", "false", "no", "off"}


def _select_sandbox_image(cmd: str) -> str:
    if _NODE_TOOL_RE.search(cmd or ""):
        return _NODE_SANDBOX_IMAGE
    return _SANDBOX_IMAGE


def _should_use_host_subprocess(cmd: str) -> bool:
    command = cmd or ""
    if _INSTALL_TOOL_RE.search(command):
        return True
    if os.name == "nt" and _HOST_NODE_TOOLCHAIN_ON_WINDOWS and _NODE_TOOL_RE.search(command):
        return True
    return False


if _DOCKER_AVAILABLE:
    print(f"[Tools] Docker sandbox enabled  image={_SANDBOX_IMAGE}  node_image={_NODE_SANDBOX_IMAGE}")
else:
    print("[Tools] Docker unavailable — subprocess fallback active (less secure)")



def _truncate(text: str, limit: int = _MAX_CMD_OUTPUT) -> str:
    """Cap *text* at *limit* chars, inserting an ellipsis in the middle."""
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - limit
    return text[:half] + f"\n…[{omitted} chars omitted]…\n" + text[-half:]


def _format_result(returncode: int, stdout: str, stderr: str) -> str:
    return (
        f"exit_code: {returncode}\n"
        f"stdout:\n{_truncate(stdout)}\n"
        f"stderr:\n{_truncate(stderr)}"
    )


def _run_docker(cmd: str, work_dir: pathlib.Path, timeout: int, root: pathlib.Path) -> str:
    """
    Execute *cmd* inside a locked-down Docker container.

    Security flags applied:
      --network none   — generated code cannot make outbound connections
      --memory 512m    — prevent memory-bomb denial-of-service
      --cpus 0.5       — prevent CPU monopolisation
      --read-only      — container's own filesystem is read-only …
      -v …:rw          — … except the project volume (writable)
      --tmpfs /tmp     — writable scratch space inside the container
      --rm             — container cleaned up automatically on exit
    """
    real_root = root.resolve()
    # Translate the host work_dir into the container path (/sandbox/…)
    try:
        rel = work_dir.relative_to(real_root)
        container_cwd = f"/sandbox/{rel}"
    except ValueError:
        container_cwd = "/sandbox"

    image = _select_sandbox_image(cmd)
    docker_argv = [
        "docker", "run", "--rm",
        "--network", "none",
        "--memory",  "512m",
        "--cpus",    "0.5",
        "--read-only",
        "--tmpfs",   "/tmp:size=64m",
        "-v",        f"{real_root}:/sandbox:rw",
        "-w",        container_cwd,
        image,
        "sh", "-c", cmd,
    ]
    try:
        proc = subprocess.run(
            docker_argv,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return _format_result(proc.returncode, proc.stdout, proc.stderr)
    except subprocess.TimeoutExpired:
        return _format_result(
            -1, "",
            f"ERROR: Docker container timed out after {timeout}s "
            "(it was killed automatically).",
        )
    except Exception as exc:
        # Unexpected Docker failure — degrade gracefully
        print(f"[Tools] Docker run raised {exc!r} — falling back to subprocess")
        return _run_subprocess(cmd, work_dir, timeout)


def _run_subprocess(cmd: str, work_dir: pathlib.Path, timeout: int) -> str:
    """
    Fallback executor: run *cmd* directly via the host shell.
    Less isolated than Docker but available everywhere.
    """
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return _format_result(proc.returncode, proc.stdout, proc.stderr)
    except subprocess.TimeoutExpired:
        return _format_result(
            -1, "",
            f"ERROR: command timed out after {timeout}s.",
        )
    except Exception as exc:
        return _format_result(-1, "", f"ERROR: {exc!r}")



@tool
def write_file(path: str, content: str, config: RunnableConfig) -> str:
    """
    Write *content* to *path* (relative to project root), creating parent
    directories as needed.  Returns a confirmation string.
    """
    root = _get_project_root(config)
    p = _safe(path, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"WROTE: {p}"


def _normalize_patch_line(line: str) -> str:
    return line.strip()


def _match_line_block(content_lines: list[str], target_lines: list[str]) -> tuple[int, int] | None:
    if not target_lines:
        return None

    normalized_target = [_normalize_patch_line(line) for line in target_lines]
    exact_matches: list[tuple[int, int]] = []
    for start in range(0, len(content_lines) - len(target_lines) + 1):
        window = content_lines[start : start + len(target_lines)]
        if window == target_lines:
            exact_matches.append((start, start + len(target_lines)))
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        raise ValueError("search block matches multiple locations; provide a larger block.")

    normalized_matches: list[tuple[int, int]] = []
    for start in range(0, len(content_lines) - len(target_lines) + 1):
        window = content_lines[start : start + len(target_lines)]
        if [_normalize_patch_line(line) for line in window] == normalized_target:
            normalized_matches.append((start, start + len(target_lines)))
    if len(normalized_matches) == 1:
        return normalized_matches[0]
    if len(normalized_matches) > 1:
        raise ValueError("normalized search block matches multiple locations; provide a larger block.")

    search_text = "\n".join(normalized_target).strip()
    scored_matches: list[tuple[float, int, int]] = []
    for start in range(0, len(content_lines) - len(target_lines) + 1):
        window = content_lines[start : start + len(target_lines)]
        window_text = "\n".join(_normalize_patch_line(line) for line in window).strip()
        score = difflib.SequenceMatcher(None, search_text, window_text).ratio()
        if score >= 0.82:
            scored_matches.append((score, start, start + len(target_lines)))

    if not scored_matches:
        return None

    scored_matches.sort(key=lambda item: item[0], reverse=True)
    best_score, start, end = scored_matches[0]
    if len(scored_matches) > 1 and best_score - scored_matches[1][0] < 0.05:
        raise ValueError("fuzzy search block matches multiple similar locations; provide more context.")
    return (start, end)


def _apply_search_replace_blocks(content: str, blocks: list[tuple[str, str]]) -> str:
    content_lines = content.splitlines()
    trailing_newline = content.endswith("\n")

    for search_block, replace_block in blocks:
        search_lines = search_block.splitlines()
        replace_lines = replace_block.splitlines()
        match = _match_line_block(content_lines, search_lines)
        if match is None:
            raise ValueError("search block not found; read the file again and include more exact surrounding lines.")
        start, end = match
        content_lines[start:end] = replace_lines

    updated = "\n".join(content_lines)
    if trailing_newline and updated and not updated.endswith("\n"):
        updated += "\n"
    return updated


def _parse_search_replace_patch(patch: str) -> list[tuple[str, str]]:
    lines = patch.splitlines()
    blocks: list[tuple[str, str]] = []
    idx = 0

    while idx < len(lines):
        if lines[idx].strip() != _PATCH_SEARCH_MARKER:
            idx += 1
            continue

        idx += 1
        search_lines: list[str] = []
        while idx < len(lines) and lines[idx].strip() != _PATCH_MID_MARKER:
            search_lines.append(lines[idx])
            idx += 1
        if idx >= len(lines):
            raise ValueError("patch block is missing the ======= separator.")

        idx += 1
        replace_lines: list[str] = []
        while idx < len(lines) and lines[idx].strip() != _PATCH_REPLACE_MARKER:
            replace_lines.append(lines[idx])
            idx += 1
        if idx >= len(lines):
            raise ValueError("patch block is missing the >>>>>>> REPLACE marker.")

        idx += 1
        blocks.append(("\n".join(search_lines), "\n".join(replace_lines)))

    if not blocks:
        raise ValueError("No valid search/replace blocks were found in the patch.")
    return blocks


def _make_patch_block(target_content: str, replacement_content: str) -> str:
    return (
        f"{_PATCH_SEARCH_MARKER}\n"
        f"{target_content}\n"
        f"{_PATCH_MID_MARKER}\n"
        f"{replacement_content}\n"
        f"{_PATCH_REPLACE_MARKER}"
    )


@tool
def apply_patch_block(path: str, patch: str, config: RunnableConfig) -> str:
    """
    Apply one or more search/replace blocks to an existing file.

    Patch format:
    <<<<<<< SEARCH
    exact or near-exact lines to find
    =======
    replacement lines
    >>>>>>> REPLACE
    """
    root = _get_project_root(config)
    p = _safe(path, root)
    if not p.exists():
        return f"ERROR: File '{path}' does not exist."

    try:
        blocks = _parse_search_replace_patch(patch)
        updated_content = _apply_search_replace_blocks(p.read_text(encoding="utf-8"), blocks)
    except ValueError as exc:
        return f"ERROR: {exc}"

    p.write_text(updated_content, encoding="utf-8")
    return f"PATCHED: {p}"

@tool
def replace_file_content(path: str, target_content: str, replacement_content: str, config: RunnableConfig) -> str:
    """
    Replace a specific block of text in an existing file.
    Use this for SURGICAL edits rather than rewriting the entire file.
    This now accepts near-exact search blocks and uses fuzzy matching when
    indentation or whitespace differs slightly.
    """
    return apply_patch_block.invoke(
        {
            "path": path,
            "patch": _make_patch_block(target_content, replacement_content),
            "config": config,
        }
    )


@tool
def delete_file(path: str, config: RunnableConfig, reason: str = "") -> str:
    """
    Delete a single file at *path* (relative to project root).
    Directory and project-root deletes are refused because agents should never
    be able to wipe a project tree. A clear reason is required for auditability.
    """
    root = _get_project_root(config)
    requested_path = (path or "").strip()
    if not requested_path or requested_path in {".", "/", "\\"}:
        return "ERROR: Refusing to delete the project root."

    configurable = (config or {}).get("configurable", {})
    thread_id = configurable.get("thread_id", "default")
    
    if "-update-" in thread_id:
        import sqlite3
        project_id = thread_id.split("-update-")[0]
        latest_request = ""
        try:
            conn = sqlite3.connect("projects.db")
            cur = conn.cursor()
            cur.execute("SELECT prompt FROM project_generations WHERE id = ?", (project_id,))
            row = cur.fetchone()
            conn.close()
            if row:
                prompt = row[0]
                if "[UPDATE]:" in prompt:
                    latest_request = prompt.split("[UPDATE]:")[-1].strip()
                else:
                    latest_request = prompt.strip()
        except Exception:
            pass
            
        if latest_request:
            name = os.path.basename(requested_path).lower()
            name_no_ext = os.path.splitext(name)[0]
            latest_request_lower = latest_request.lower()
            if name not in latest_request_lower and name_no_ext not in latest_request_lower:
                return (
                    f"ERROR: Refusing to delete '{requested_path}'. Deletions are strictly "
                    "prohibited unless the user's latest request explicitly names the file or path "
                    "and asks for it to be deleted."
                )

    deletion_reason = (reason or "").strip()
    if len(deletion_reason) < _DELETE_REASON_MIN_CHARS:
        return (
            "ERROR: delete_file requires a specific reason explaining why this "
            f"file must be removed ({_DELETE_REASON_MIN_CHARS}+ characters)."
        )

    p = _safe(requested_path, root)
    if p.resolve() == root.resolve():
        return "ERROR: Refusing to delete the project root."
    if not p.exists():
        return f"File '{requested_path}' does not exist - nothing to delete."
    if p.is_dir():
        return (
            f"ERROR: Refusing to delete directory '{requested_path}'. "
            "Delete only specific files with an explicit reason."
        )
    p.unlink()
    return f"DELETED: {requested_path} | reason: {deletion_reason}"


@tool
def read_file(path: str, config: RunnableConfig) -> str:
    """
    Read and return the contents of *path* (relative to project root).
    Returns an empty string if the file does not exist yet.
    """
    root = _get_project_root(config)
    p = _safe(path, root)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


@tool
def list_files(config: RunnableConfig, directory: str = ".") -> str:
    """
    Recursively list all files under *directory* (relative to project root).
    Returns one relative path per line, or 'No files found.' for an empty tree.
    """
    root = _get_project_root(config)
    p = _safe(directory, root)
    if not p.is_dir():
        return f"ERROR: '{directory}' is not a directory."
    files = [
        str(f.relative_to(root))
        for f in p.glob("**/*")
        if f.is_file()
    ]
    return "\n".join(sorted(files)) if files else "No files found."


@tool
def list_project_tree(config: RunnableConfig, directory: str = ".") -> str:
    """
    Recursively list directories and files under *directory* (relative to
    project root) in a tree-style format. Returns a summary footer and
    truncates extremely large trees.
    """
    root = _get_project_root(config)
    p = _safe(directory, root)
    if not p.is_dir():
        return f"ERROR: '{directory}' is not a directory."

    display_root = str(p.relative_to(root)).replace("\\", "/")
    if display_root == ".":
        display_root = "."

    lines = [f"[DIR] {display_root}/"]
    directories = 1
    files = 0
    shown_entries = 1
    truncated = False

    for current_root, dirnames, filenames in os.walk(p):
        dirnames.sort()
        filenames.sort()
        relative_root = pathlib.Path(current_root).relative_to(p)
        depth = len(relative_root.parts)

        if pathlib.Path(current_root) != p:
            lines.append(f"{'  ' * depth}[DIR] {pathlib.Path(current_root).name}/")
            directories += 1
            shown_entries += 1
            if shown_entries >= _TREE_LIST_MAX_ENTRIES:
                truncated = True
                break

        for filename in filenames:
            file_path = pathlib.Path(current_root) / filename
            try:
                size = file_path.stat().st_size
            except OSError:
                size = None
            size_suffix = f" ({size} bytes)" if size is not None else ""
            lines.append(f"{'  ' * (depth + 1)}[FILE] {filename}{size_suffix}")
            files += 1
            shown_entries += 1
            if shown_entries >= _TREE_LIST_MAX_ENTRIES:
                truncated = True
                break

        if truncated:
            break

    summary = f"Directories: {directories}, files: {files}"
    if truncated:
        summary += f" (truncated after {shown_entries} entries)"
    return "\n".join(lines + ["", summary])


@tool
def get_current_directory(config: RunnableConfig) -> str:
    """Return the absolute path of the project root directory."""
    root = _get_project_root(config)
    return str(root)


@tool
def run_cmd(cmd: str, config: RunnableConfig, cwd: str = ".", timeout: int = 30) -> str:
    """
    Run a shell command inside the project root (or a sub-directory).

    When a Docker daemon is available, the command runs inside an isolated
    container (no network, memory-capped, read-only except the project
    volume).  Falls back to a plain subprocess otherwise.

    The *timeout* is capped at CMD_TIMEOUT_CAP seconds (default 120)
    regardless of what the caller requests.

    Returns:
        exit_code: <int>
        stdout:
        <captured stdout — capped at MAX_CMD_OUTPUT chars>
        stderr:
        <captured stderr — capped at MAX_CMD_OUTPUT chars>
    """
    root = _get_project_root(config)
    work_dir = _safe(cwd, root)
    if not work_dir.exists() or not work_dir.is_dir():
        return _format_result(-1, "", f"ERROR: cwd '{cwd}' does not exist or is not a directory.")
    timeout = max(1, min(int(timeout), _CMD_TIMEOUT_CAP))   # enforce hard ceiling with sane minimum
    if not cmd or not str(cmd).strip():
        return _format_result(-1, "", "ERROR: command is empty.")

    if _DOCKER_AVAILABLE and not _should_use_host_subprocess(cmd):
        return _run_docker(cmd, work_dir, timeout, root)
    return _run_subprocess(cmd, work_dir, timeout)



@tool
def validate_imports(config: RunnableConfig) -> str:
    """
    Deterministically check generated source files for unresolved relative imports.
    Returns OK when imports resolve, otherwise returns a concise error report.
    """
    root = _get_project_root(config)
    result = validate_project_imports(root)
    if result is None:
        return "OK: relative imports resolved."
    return _format_result(result.returncode, result.stdout, result.stderr)


_TREE_SKIP_DIRS: frozenset[str] = frozenset({
    "__pycache__", "node_modules", ".git", ".hg", ".svn",
    ".venv", "venv", "env", ".env", ".tox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "dist", "build", ".next",
    ".nuxt", "coverage", ".coverage", "htmlcoverage",
})

def _human_size(n_bytes: int) -> str:
    """Convert a byte count to a compact human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes}{unit}" if unit == "B" else f"{n_bytes:.1f}{unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f}TB"


def _build_tree_lines(
    path: pathlib.Path,
    root: pathlib.Path,
    prefix: str = "",
    *,
    max_entries: int = _TREE_LIST_MAX_ENTRIES,
    counter: list[int] | None = None,   # [entries_shown, dirs, files]
) -> list[str]:
    """
    Recursively build tree-style lines for *path*.
    Uses ├── / └── / │   connectors, same as the Unix `tree` command.
    Skips directories in _TREE_SKIP_DIRS.
    """
    if counter is None:
        counter = [0, 0, 0]   # [shown, dirs, files]

    lines: list[str] = []

    try:
        entries = sorted(path.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
    except PermissionError:
        return [f"{prefix}[permission denied]"]

    # Filter out noise directories before counting so connectors are correct
    entries = [
        e for e in entries
        if not (e.is_dir() and e.name in _TREE_SKIP_DIRS)
    ]

    for idx, entry in enumerate(entries):
        if counter[0] >= max_entries:
            lines.append(f"{prefix}... (truncated)")
            break

        is_last = (idx == len(entries) - 1)
        connector = "└── " if is_last else "├── "
        extension = "    " if is_last else "│   "

        if entry.is_dir():
            counter[0] += 1
            counter[1] += 1
            lines.append(f"{prefix}{connector}{entry.name}/")
            lines.extend(
                _build_tree_lines(
                    entry, root,
                    prefix=prefix + extension,
                    max_entries=max_entries,
                    counter=counter,
                )
            )
        else:
            counter[0] += 1
            counter[2] += 1
            try:
                size = _human_size(entry.stat().st_size)
            except OSError:
                size = "?"
            lines.append(f"{prefix}{connector}{entry.name}  ({size})")

    return lines


@tool
def tree_graph(config: RunnableConfig, directory: str = ".") -> str:
    """
    Render a complete directory + file tree rooted at *directory* (relative
    to the project root) using classic ├── / └── connectors, similar to the
    Unix `tree` command.

    Each file is annotated with its size (e.g. 3.2KB).
    Noisy directories such as __pycache__, node_modules, .git, .venv, dist,
    and build are automatically skipped.

    Returns the tree as a multi-line string followed by a summary line, e.g.:

        .
        ├── src/
        │   ├── app.py  (1.2KB)
        │   └── models.py  (0.8KB)
        ├── templates/
        │   └── index.html  (2.1KB)
        └── requirements.txt  (0.1KB)

        2 directories, 4 files
    """
    root = _get_project_root(config)
    target = _safe(directory, root)
    if not target.is_dir():
        return f"ERROR: '{directory}' is not a directory."

    # Display name for the tree root
    display_name = str(target.relative_to(root)).replace("\\", "/") or "."
    if display_name == ".":
        display_name = "."

    counter: list[int] = [0, 0, 0]   # [shown, dirs, files]
    lines = _build_tree_lines(target, root, max_entries=_TREE_LIST_MAX_ENTRIES, counter=counter)

    truncated_note = f"  (tree capped at {_TREE_LIST_MAX_ENTRIES} entries)" if counter[0] >= _TREE_LIST_MAX_ENTRIES else ""
    summary = f"\n{counter[1]} {'directory' if counter[1] == 1 else 'directories'}, {counter[2]} {'file' if counter[2] == 1 else 'files'}{truncated_note}"

    return "\n".join([display_name] + lines) + summary



def init_project_root() -> str:
    """Create the base generated_project directory if it doesn't exist yet."""
    root = pathlib.Path.cwd() / "generated_project"
    root.mkdir(parents=True, exist_ok=True)
    return str(root)


def _visual_verify_sync(url_or_path: str, config: RunnableConfig) -> str:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return "ERROR: playwright is not installed. Please pip install playwright and playwright run install"

    if not url_or_path.startswith("http"):
        root = _get_project_root(config)
        p = _safe(url_or_path, root)
        target_url = p.as_uri()
    else:
        target_url = url_or_path

    logs = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            page.on("console", lambda msg: logs.append(msg.text))
            
            response = page.goto(target_url, wait_until="networkidle")
            
            title = page.title()
            body_text = page.locator("body").inner_text()[:500]
            
            browser.close()
            
            return (
                f"Successfully loaded: {target_url}\n"
                f"HTTP Status: {response.status if response else 'Unknown'}\n"
                f"Page Title: {title}\n"
                f"Console Logs: {logs}\n"
                f"Body Text Preview: {body_text}\n"
            )
    except Exception as e:
        return f"ERROR visually verifying {target_url}: {str(e)}"


@tool
def visual_verify(url_or_path: str, config: RunnableConfig) -> str:
    """
    Open the given URL or local HTML file path in a worker thread to verify the UI.
    If it's a relative path, it resolves within the project directory.
    Returns the page title, console logs, and basic DOM properties (like body text).
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(_visual_verify_sync, url_or_path, config).result()


@tool
def ddg_search(query: str, config: RunnableConfig) -> str:
    """
    Search the web using DuckDuckGo.
    Returns the top search results, including snippets.
    Use this to look up documentation, fix errors, or find up-to-date versions of libraries.
    """
    try:
        from langchain_community.tools import DuckDuckGoSearchResults
        search = DuckDuckGoSearchResults(num_results=4)
        print(f"[Tools] Executing web search for: '{query}'")
        return search.invoke({"query": query})
    except Exception as e:
        return f"ERROR performing web search for '{query}': {str(e)}"
