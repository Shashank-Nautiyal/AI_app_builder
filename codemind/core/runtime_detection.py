from __future__ import annotations

import json
import os
from pathlib import Path


SKIP_DIRS = {
    ".git",
    ".next",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}

FRONTEND_DIR_HINTS = {"client", "frontend", "public", "ui", "web"}
FRONTEND_DEP_HINTS = {
    "@angular/core",
    "next",
    "react",
    "react-dom",
    "react-scripts",
    "svelte",
    "vite",
    "vue",
}
FRONTEND_SCRIPT_HINTS = (
    "next dev",
    "react-scripts",
    "vite",
    "vue-cli-service serve",
    "webpack serve",
)
FRONTEND_SCRIPT_PORT_DEFAULTS = (
    ("vite preview", 4173),
    ("vite", 5173),
    ("next", 3000),
    ("react-scripts", 3000),
    ("vue-cli-service serve", 3000),
    ("webpack serve", 3000),
)

PYTHON_ENTRY_FILES = ("app.py", "main.py", "server.py", "api.py", "run.py", "wsgi.py")
NODE_ENTRY_FILES = ("server.js", "app.js", "index.js", "main.js", "api.js")

PYTHON_FRAMEWORK_MARKERS = (
    "FastAPI(",
    "Flask(__name__",
    "Quart(__name__",
    "app.run(",
    "uvicorn.run(",
)
NODE_SERVER_MARKERS = (
    "app.listen(",
    "createServer(",
    "express(",
    "fastify(",
    "listen(",
)


def _relative_posix(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    return rel or "."


def _iter_files(project_root: Path, *, filenames: set[str] | None = None, suffixes: tuple[str, ...] | None = None):
    for root, dirs, files in os.walk(project_root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        root_path = Path(root)
        for filename in files:
            if filenames and filename not in filenames:
                continue
            if suffixes and not filename.lower().endswith(suffixes):
                continue
            yield root_path / filename


def _safe_read_text(path: Path, limit: int = 200_000) -> str:
    try:
        return path.read_text(encoding="utf-8")[:limit]
    except (OSError, UnicodeDecodeError):
        return ""


def _command_in_dir(rel_dir: str | None, command: str) -> str:
    if not rel_dir or rel_dir == ".":
        return f"{command}\r\n"
    escaped = rel_dir.replace('"', '""')
    return f'cd /d "{escaped}" && {command}\r\n'


def _format_name(base_name: str, rel_dir: str | None) -> str:
    if not rel_dir or rel_dir == ".":
        return base_name
    return f"{base_name} ({rel_dir})"


def _dedupe_command_names(commands: list[dict]) -> list[dict]:
    seen: dict[str, int] = {}
    for command in commands:
        original_name = command["name"]
        count = seen.get(original_name, 0) + 1
        seen[original_name] = count
        if count > 1:
            command["name"] = f"{original_name} {count}"
    return commands


def _load_package_json(package_json_path: Path) -> dict:
    try:
        return json.loads(package_json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def _node_install_prefix(project_root: Path) -> str:
    if (project_root / "node_modules").exists():
        return ""
    return "npm install && "


def _is_frontend_package(rel_dir: str, package_data: dict) -> bool:
    dir_parts = {part.lower() for part in rel_dir.split("/") if part}
    if dir_parts & FRONTEND_DIR_HINTS:
        return True

    deps = {
        *(package_data.get("dependencies") or {}).keys(),
        *(package_data.get("devDependencies") or {}).keys(),
    }
    if any(dep in FRONTEND_DEP_HINTS for dep in deps):
        return True

    scripts = " ".join(str(value).lower() for value in (package_data.get("scripts") or {}).values())
    return any(hint in scripts for hint in FRONTEND_SCRIPT_HINTS)


def _infer_script_port(script_name: str, script_command: str) -> int | None:
    lowered = str(script_command or "").lower()
    for token in ("--port", "-p"):
        marker = f"{token} "
        if marker in lowered:
            fragment = lowered.split(marker, 1)[1].strip()
            digits = []
            for char in fragment:
                if char.isdigit():
                    digits.append(char)
                else:
                    break
            if digits:
                return int("".join(digits))

    for hint, port in FRONTEND_SCRIPT_PORT_DEFAULTS:
        if hint in lowered:
            return port

    if script_name == "preview":
        return 4173
    if script_name in {"dev", "start", "serve"}:
        return 3000
    return None


def _detect_frontend_command(package_data: dict, install_prefix: str) -> tuple[str | None, int | None]:
    scripts = package_data.get("scripts") or {}
    script_candidates = ("start", "dev", "serve", "preview")

    for script_name in script_candidates:
        script_command = scripts.get(script_name)
        if not isinstance(script_command, str) or not script_command.strip():
            continue
        npm_command = "npm start" if script_name == "start" else f"npm run {script_name}"
        return f"{install_prefix}{npm_command}", _infer_script_port(script_name, script_command)

    return None, None


def _detect_node_launch(project_root: Path, *, rel_dir: str = ".") -> tuple[str | None, int | None, bool]:
    package_json_path = project_root / "package.json"
    if not package_json_path.exists():
        return None, None, False

    package_data = _load_package_json(package_json_path)
    install_prefix = _node_install_prefix(project_root)
    is_frontend = _is_frontend_package(rel_dir, package_data)

    if is_frontend:
        frontend_command, frontend_port = _detect_frontend_command(package_data, install_prefix)
        
        # Check for explicit port in vite.config
        import re
        for ext in ("js", "ts", "mjs", "cjs"):
            vite_cfg = project_root / f"vite.config.{ext}"
            if vite_cfg.exists():
                content = _safe_read_text(vite_cfg, limit=20_000)
                match = re.search(r"port\s*:\s*(\d+)", content)
                if match:
                    frontend_port = int(match.group(1))
                    break
                        
        if frontend_command:
            return frontend_command, frontend_port, True
        return None, None, True

    scripts = package_data.get("scripts") or {}
    start_script = scripts.get("start")
    if isinstance(start_script, str) and start_script.strip():
        return f"{install_prefix}npm start", None, False

    main_file = package_data.get("main")
    if isinstance(main_file, str) and main_file.strip():
        main_path = (project_root / main_file).resolve()
        try:
            main_path.relative_to(project_root)
        except ValueError:
            main_path = None
        if main_path and main_path.exists():
            return f"{install_prefix}node {Path(main_file).as_posix()}", None, False

    detected_entry = detect_node_entry_file(project_root)
    if detected_entry:
        return f"{install_prefix}node {detected_entry}", None, False

    return None, None, False


def detect_python_app_file(project_dir: str | Path) -> str | None:
    project_root = Path(project_dir).resolve()
    scored_candidates: list[tuple[int, int, str]] = []
    fallback_candidates: list[str] = []

    for file_path in _iter_files(project_root, suffixes=(".py",)):
        rel_path = _relative_posix(file_path, project_root)
        fallback_candidates.append(rel_path)
        content = _safe_read_text(file_path, limit=80_000)
        filename = file_path.name.lower()
        score = 0

        if filename in PYTHON_ENTRY_FILES:
            score += 3
        if any(marker in content for marker in PYTHON_FRAMEWORK_MARKERS):
            score += 6
        if 'if __name__ == "__main__"' in content or "if __name__ == '__main__'" in content:
            score += 3
        if filename == "__main__.py":
            score += 4
        if "FastAPI" in content or "Flask" in content:
            score += 2

        if score > 0:
            scored_candidates.append((score, len(rel_path), rel_path))

    if scored_candidates:
        scored_candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        return scored_candidates[0][2]

    for preferred in PYTHON_ENTRY_FILES:
        for rel_path in fallback_candidates:
            if rel_path.endswith(preferred):
                return rel_path

    if not fallback_candidates:
        return None

    return sorted(fallback_candidates, key=lambda rel: (len(rel), rel))[0]


def detect_node_entry_file(project_dir: str | Path) -> str | None:
    project_root = Path(project_dir).resolve()
    candidates: list[tuple[int, int, str]] = []

    for file_path in _iter_files(project_root, suffixes=(".js", ".mjs", ".cjs")):
        rel_path = _relative_posix(file_path, project_root)
        content = _safe_read_text(file_path, limit=80_000)
        filename = file_path.name.lower()
        score = 0

        if filename in NODE_ENTRY_FILES:
            score += 3
        if any(marker in content for marker in NODE_SERVER_MARKERS):
            score += 6
        if "process.env.PORT" in content:
            score += 1
        if "import express" in content or 'require("express")' in content or "require('express')" in content:
            score += 2

        if score > 0:
            candidates.append((score, len(rel_path), rel_path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return candidates[0][2]


def detect_node_start_command(project_dir: str | Path) -> str | None:
    project_root = Path(project_dir).resolve()
    command, _, _ = _detect_node_launch(project_root)
    return command


def _discover_node_commands(project_root: Path) -> list[dict]:
    commands: list[dict] = []
    for package_json_path in sorted(_iter_files(project_root, filenames={"package.json"}), key=lambda path: len(path.parts)):
        package_dir = package_json_path.parent
        rel_dir = _relative_posix(package_dir, project_root)
        command_body, serve_port, is_frontend = _detect_node_launch(package_dir, rel_dir=rel_dir)
        if not command_body:
            continue

        package_data = _load_package_json(package_json_path)
        is_frontend = is_frontend or _is_frontend_package(rel_dir, package_data)
        commands.append(
            {
                "name": _format_name("Frontend" if is_frontend else "Backend", rel_dir),
                "cmd": _command_in_dir(rel_dir, command_body),
                "kind": "frontend" if is_frontend else "backend",
                "serve_port": serve_port if is_frontend else None,
            }
        )

    return commands


def _discover_python_commands(project_root: Path) -> list[dict]:
    commands: list[dict] = []

    for requirements_path in sorted(_iter_files(project_root, filenames={"requirements.txt"}), key=lambda path: len(path.parts)):
        project_dir = requirements_path.parent.resolve()
        rel_dir = _relative_posix(project_dir, project_root)
        app_file = detect_python_app_file(project_dir)
        if not app_file:
            continue
        commands.append(
            {
                "name": _format_name("Backend", rel_dir),
                "cmd": _command_in_dir(rel_dir, f"pip install -r requirements.txt && python {app_file}"),
                "kind": "backend",
            }
        )

    if commands:
        return commands

    app_file = detect_python_app_file(project_root)
    if app_file:
        commands.append({"name": "Backend", "cmd": _command_in_dir(".", f"python {app_file}"), "kind": "backend"})

    return commands


def _find_static_html(project_root: Path) -> dict | None:
    candidates: list[tuple[int, int, str, Path]] = []

    for html_path in _iter_files(project_root, suffixes=(".html",)):
        rel_path = _relative_posix(html_path, project_root)
        parts = {part.lower() for part in Path(rel_path).parts}
        score = 0
        if html_path.name.lower() == "index.html":
            score += 4
        if parts & FRONTEND_DIR_HINTS:
            score += 3
        if "templates" in parts:
            score -= 1
        candidates.append((score, len(rel_path), rel_path, html_path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    _, _, rel_path, html_path = candidates[0]
    rel_dir = _relative_posix(html_path.parent, project_root)
    return {
        "html_file": html_path.name,
        "html_path": rel_path,
        "html_dir": None if rel_dir == "." else rel_dir,
        "serve_port": 3000,
    }


def build_run_plan(project_dir: str | Path) -> dict:
    project_root = Path(project_dir).resolve()
    node_commands = _discover_node_commands(project_root)
    python_commands = _discover_python_commands(project_root)
    html_plan = None if any(cmd["kind"] == "frontend" for cmd in node_commands) else _find_static_html(project_root)
    frontend_node_port = next((cmd.get("serve_port") for cmd in node_commands if cmd.get("kind") == "frontend" and cmd.get("serve_port")), None)

    commands: list[dict] = [
        {"name": cmd["name"], "cmd": cmd["cmd"]}
        for cmd in [*node_commands, *python_commands]
    ]

    if html_plan:
        commands.append(
            {
                "name": _format_name("Frontend", html_plan["html_dir"]),
                "cmd": _command_in_dir(html_plan["html_dir"], f"python -m http.server {html_plan['serve_port']}"),
            }
        )

    if not commands:
        serve_port = 3000
        commands.append({"name": "Server", "cmd": f"python -m http.server {serve_port}\r\n"})
        return {"commands": _dedupe_command_names(commands), "html_file": None, "html_path": None, "serve_port": serve_port}

    return {
        "commands": _dedupe_command_names(commands),
        "html_file": html_plan["html_file"] if html_plan else None,
        "html_path": html_plan["html_path"] if html_plan else None,
        "serve_port": html_plan["serve_port"] if html_plan else frontend_node_port,
    }
