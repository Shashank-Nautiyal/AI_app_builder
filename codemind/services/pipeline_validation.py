from __future__ import annotations

import ast
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request

from codemind.core.runtime_detection import build_run_plan
from codemind.core.schemas import ExecutionResult, GeneratedFileInfo
from codemind.core.template_registry import TemplateConfig


SKIP_DIRS = {
    ".ai_studio_index",
    ".git",
    ".next",
    ".project_snapshots",
    ".venv",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "venv",
}
SOURCE_SUFFIXES = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".py")
JS_SUFFIXES = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")
PY_SUFFIXES = (".py",)
RUNTIME_ENV_FILENAMES = {
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    ".env.test",
}
ENV_ASSIGNMENT_RE = re.compile(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=(.*)$")
ENV_PLACEHOLDER_RE = re.compile(
    r"(^$|your[-_\s]|placeholder|change-?me|example|<[^>]+>|\bxxx+\b|\btodo\b)",
    re.IGNORECASE,
)
RESOLVABLE_SUFFIXES = (
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".json",
    ".css", ".scss", ".sass", ".less", ".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp",
)

JS_IMPORT_RE = re.compile(
    r"(?:import\s+(?:[^'\"\n]+?\s+from\s+)?|export\s+[^'\"\n]+?\s+from\s+|require\(|import\()\s*['\"]([^'\"]+)['\"]",
    re.MULTILINE,
)
JS_EXPORT_PATTERNS = (
    re.compile(r"\bexport\s+default\s+(?:function|class)?\s*([A-Za-z_$][\w$]*)?"),
    re.compile(r"\bexport\s+(?:async\s+)?(?:function|class|const|let|var)\s+([A-Za-z_$][\w$]*)"),
    re.compile(r"\bexport\s*\{([^}]+)\}"),
    re.compile(r"\bexports\.([A-Za-z_$][\w$]*)\s*="),
)
COMMONJS_OBJECT_RE = re.compile(r"\bmodule\.exports\s*=\s*\{([^}]+)\}", re.S)


def _relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _safe_read(path: Path, limit: int = 250_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:limit]
    except OSError:
        return ""


def iter_source_files(project_root: str | Path):
    root = Path(project_root).resolve()
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS and not name.startswith(".cache")]
        current = Path(current_root)
        for filename in filenames:
            if filename.lower().endswith(SOURCE_SUFFIXES):
                yield current / filename


def _clean_export_name(name: str) -> str | None:
    name = name.strip()
    if not name:
        return None
    if " as " in name:
        name = name.rsplit(" as ", 1)[-1].strip()
    name = name.split(":", 1)[0].strip()
    match = re.match(r"[A-Za-z_$][\w$]*", name)
    return match.group(0) if match else None


def _extract_js_exports(content: str) -> list[str]:
    exports: set[str] = set()
    if re.search(r"\bexport\s+default\b", content):
        exports.add("default")
    for pattern in JS_EXPORT_PATTERNS:
        for match in pattern.finditer(content):
            value = match.group(1) if match.groups() else ""
            if pattern.pattern.startswith("\\bexport\\s*\\{"):
                for part in value.split(","):
                    name = _clean_export_name(part)
                    if name:
                        exports.add(name)
            else:
                exports.add(value or "default")

    for match in COMMONJS_OBJECT_RE.finditer(content):
        for part in match.group(1).split(","):
            name = _clean_export_name(part)
            if name:
                exports.add(name)
    if "module.exports =" in content and not exports:
        exports.add("default")
    return sorted(exports)


def _extract_python_facts(content: str) -> tuple[list[str], list[str]]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return [], []

    imports: set[str] = set()
    exports: set[str] = set()
    explicit_all: list[str] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                exports.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            else:
                prefix = "." * node.level
                imports.add(prefix + (node.module or ""))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    try:
                        value = ast.literal_eval(node.value)
                    except Exception:
                        value = None
                    if isinstance(value, (list, tuple)):
                        explicit_all = [item for item in value if isinstance(item, str)]

    if explicit_all:
        exports = set(explicit_all)
    return sorted(imports), sorted(exports)


def extract_file_info(filepath: str, content: str) -> GeneratedFileInfo:
    normalized = filepath.replace("\\", "/")
    suffix = Path(normalized).suffix.lower()
    if suffix in JS_SUFFIXES:
        imports = sorted(set(JS_IMPORT_RE.findall(content)))
        exports = _extract_js_exports(content)
        kind = "javascript"
    elif suffix in PY_SUFFIXES:
        imports, exports = _extract_python_facts(content)
        kind = "python"
    else:
        imports, exports, kind = [], [], "other"
    return GeneratedFileInfo(filepath=normalized, imports=imports, exports=exports, kind=kind)


def build_generated_file_manifest(project_root: str | Path) -> dict[str, GeneratedFileInfo]:
    root = Path(project_root).resolve()
    manifest: dict[str, GeneratedFileInfo] = {}
    for source_path in iter_source_files(root):
        rel = _relative_posix(source_path, root)
        manifest[rel] = extract_file_info(rel, _safe_read(source_path))
    return manifest


def format_generated_manifest(manifest: dict[str, GeneratedFileInfo] | None, limit: int = 12000) -> str:
    if not manifest:
        return "No generated file manifest is available yet."
    lines: list[str] = []
    for filepath in sorted(manifest):
        info = manifest[filepath]
        lines.append(
            f"- {filepath} ({info.kind})\n"
            f"  imports: {', '.join(info.imports) or 'none'}\n"
            f"  exports: {', '.join(info.exports) or 'none'}"
        )
    text = "\n".join(lines)
    return text[:limit] + ("\n... (truncated)" if len(text) > limit else "")


def validate_template_constraints(project_root: str | Path, template: TemplateConfig) -> ExecutionResult | None:
    """Verifies that the generated project adheres to the template's static constraints."""
    root = Path(project_root).resolve()
    manifest = build_generated_file_manifest(root)
    errors: list[str] = []

    rules = template.verifier_rules

    # 1. Required Structure
    if rules.required_structure:
        for required in rules.required_structure:
            req_path = root / required
            # If it ends with '/' it's a directory check, otherwise file/dir exists check
            if not req_path.exists():
                errors.append(f"Missing required structure: {required}")

    # 2. Forbidden Imports & Patterns
    if rules.forbidden_imports or rules.forbidden_patterns:
        forbidden_imports_set = set(rules.forbidden_imports)
        for filepath, info in manifest.items():
            # Check parsed imports
            if forbidden_imports_set:
                violations = [imp for imp in info.imports if imp in forbidden_imports_set]
                if violations:
                    errors.append(f"{filepath} uses forbidden imports: {', '.join(violations)}")
            
            # Check raw patterns
            if rules.forbidden_patterns:
                abs_path = root / filepath
                content = _safe_read(abs_path).lower()
                for pattern in rules.forbidden_patterns:
                    if pattern.lower() in content:
                        errors.append(f"{filepath} contains forbidden pattern: {pattern!r}")

    if not errors:
        return None

    return ExecutionResult(
        command="template_constraint_scan",
        returncode=1,
        stdout="",
        stderr="\n".join(errors),
        success=False,
        error_type="TemplateConstraintError",
    )


def scan_placeholders(project_root: str | Path, patterns: tuple[str, ...]) -> list[dict[str, str | int]]:
    hits: list[dict[str, str | int]] = []
    lowered_patterns = [(pattern, pattern.lower()) for pattern in patterns]
    root = Path(project_root).resolve()
    for source_path in iter_source_files(root):
        rel = _relative_posix(source_path, root)
        for line_number, line in enumerate(_safe_read(source_path).splitlines(), start=1):
            lowered = line.lower()
            for original, lowered_pattern in lowered_patterns:
                if lowered_pattern == "placeholder" and (
                    re.search(r"\bplaceholder\s*=", line, re.I) or "::placeholder" in lowered
                ):
                    continue
                if lowered_pattern in lowered:
                    hits.append(
                        {
                            "filepath": rel,
                            "line_number": line_number,
                            "pattern": original,
                            "line": line.strip()[:240],
                        }
                    )
                    break
    return hits


def placeholder_failure_result(hits: list[dict[str, str | int]]) -> ExecutionResult | None:
    if not hits:
        return None
    preview = "\n".join(
        f"{hit['filepath']}:{hit['line_number']} contains {hit['pattern']!r}: {hit['line']}"
        for hit in hits[:20]
    )
    if len(hits) > 20:
        preview += f"\n... +{len(hits) - 20} more placeholder hit(s)"
    return ExecutionResult(
        command="placeholder_scan",
        returncode=1,
        stdout="",
        stderr=preview,
        success=False,
        error_type="PlaceholderError",
    )


def _strip_env_value(raw_value: str) -> str:
    value = raw_value.strip()
    if (
        len(value) >= 2
        and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'"))
    ):
        value = value[1:-1]
    return value.strip()


def scan_runtime_env_placeholders(project_root: str | Path) -> list[dict[str, str | int]]:
    """Detect fake values in runtime env files.

    Placeholder values are fine in .env.example files. They are dangerous in
    .env/.env.local because the generated app may treat them as live provider
    credentials and fail later in confusing ways.
    """

    hits: list[dict[str, str | int]] = []
    root = Path(project_root).resolve()
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS and not name.startswith(".cache")]
        current = Path(current_root)
        for filename in filenames:
            lower_name = filename.lower()
            if lower_name.endswith((".example", ".sample")):
                continue
            if lower_name not in RUNTIME_ENV_FILENAMES:
                continue

            path = current / filename
            rel = _relative_posix(path, root)
            for line_number, line in enumerate(_safe_read(path).splitlines(), start=1):
                match = ENV_ASSIGNMENT_RE.match(line)
                if not match:
                    continue
                key, raw_value = match.groups()
                value = _strip_env_value(raw_value)
                if ENV_PLACEHOLDER_RE.search(value):
                    hits.append(
                        {
                            "filepath": rel,
                            "line_number": line_number,
                            "pattern": "runtime env placeholder",
                            "line": f"{key}=<placeholder-like value>",
                        }
                    )
    return hits


def env_placeholder_failure_result(hits: list[dict[str, str | int]]) -> ExecutionResult | None:
    if not hits:
        return None
    preview = "\n".join(
        f"{hit['filepath']}:{hit['line_number']} contains {hit['line']}"
        for hit in hits[:20]
    )
    if len(hits) > 20:
        preview += f"\n... +{len(hits) - 20} more runtime env placeholder hit(s)"
    return ExecutionResult(
        command="runtime_env_placeholder_scan",
        returncode=1,
        stdout="",
        stderr=preview,
        success=False,
        error_type="RuntimeEnvPlaceholderError",
    )


def _relative_import_candidates(importer: Path, specifier: str) -> list[Path]:
    raw_base = (importer.parent / specifier).resolve()
    candidates = [raw_base]
    if raw_base.suffix:
        return candidates
    candidates.extend(Path(f"{raw_base}{suffix}") for suffix in RESOLVABLE_SUFFIXES)
    candidates.extend(raw_base / f"index{suffix}" for suffix in RESOLVABLE_SUFFIXES)
    return candidates


def _relative_python_candidates(importer: Path, specifier: str) -> list[Path]:
    dots = len(specifier) - len(specifier.lstrip("."))
    module = specifier[dots:]
    base = importer.parent
    for _ in range(max(0, dots - 1)):
        base = base.parent
    if module:
        base = base / module.replace(".", os.sep)
    return [base.with_suffix(".py"), base / "__init__.py"]


def validate_project_imports(project_root: str | Path) -> ExecutionResult | None:
    root = Path(project_root).resolve()
    issues: list[str] = []
    manifest = build_generated_file_manifest(root)

    for rel, info in manifest.items():
        importer = (root / rel).resolve()
        for specifier in info.imports:
            if info.kind == "javascript" and specifier.startswith("."):
                if not any(candidate.exists() for candidate in _relative_import_candidates(importer, specifier)):
                    issues.append(f"{rel}: unresolved relative import {specifier!r}")
            elif info.kind == "python" and specifier.startswith("."):
                if not any(candidate.exists() for candidate in _relative_python_candidates(importer, specifier)):
                    issues.append(f"{rel}: unresolved relative import {specifier!r}")

    if not issues:
        return None
    return ExecutionResult(
        command="validate_imports",
        returncode=1,
        stdout="",
        stderr="\n".join(issues[:40]),
        success=False,
        error_type="ImportValidationError",
    )


def _health_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/"


def _stop_process(proc: subprocess.Popen) -> tuple[str, str]:
    stdout = stderr = ""
    try:
        proc.terminate()
        stdout, stderr = proc.communicate(timeout=3)
    except Exception:
        try:
            proc.kill()
            stdout, stderr = proc.communicate(timeout=3)
        except Exception as exc:
            stderr = f"Failed to collect server output after stopping process: {exc}"
    return stdout or "", stderr or ""


def run_runtime_health_check(project_root: str | Path, timeout_seconds: int = 15) -> ExecutionResult | None:
    root = Path(project_root).resolve()
    run_plan = build_run_plan(root)
    port = run_plan.get("serve_port")
    commands = run_plan.get("commands") or []
    if not port or not commands:
        return None

    command = commands[0].get("cmd", "").strip()
    frontend_command = next((cmd for cmd in commands if "frontend" in str(cmd.get("name", "")).lower()), None)
    if frontend_command and frontend_command.get("cmd"):
        command = frontend_command["cmd"].strip()
    if not command:
        return None

    proc = None
    try:
        proc = subprocess.Popen(
            command,
            cwd=str(root),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.time() + timeout_seconds
        last_error = ""
        while time.time() < deadline:
            if proc.poll() is not None:
                stdout, stderr = proc.communicate(timeout=2)
                return ExecutionResult(
                    command=f"start_server_detached: {command}",
                    returncode=proc.returncode or 1,
                    stdout=stdout[-2000:],
                    stderr=stderr[-2000:] or f"Server exited before health check reached {_health_url(int(port))}",
                    success=False,
                    error_type="RuntimeStartError",
                )
            try:
                with urllib.request.urlopen(_health_url(int(port)), timeout=1.5) as response:
                    if 200 <= response.status < 500:
                        return ExecutionResult(
                            command=f"health_check_loop: {_health_url(int(port))}",
                            returncode=0,
                            stdout=f"Server responded with HTTP {response.status}.",
                            stderr="",
                            success=True,
                        )
            except Exception as exc:
                last_error = str(exc)
                time.sleep(0.5)

        stdout, stderr = _stop_process(proc)
        return ExecutionResult(
            command=f"health_check_loop: {_health_url(int(port))}",
            returncode=1,
            stdout=(stdout or "")[-2000:],
            stderr=((stderr or "") + f"\nHealth check timed out: {last_error}").strip()[-2000:],
            success=False,
            error_type="RuntimeHealthCheckError",
        )
    finally:
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
