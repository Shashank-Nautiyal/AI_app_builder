"""Copy known provider environment values into generated projects.

The generator asks agents to use managed providers such as Supabase and Clerk,
but those agents should never hardcode secrets.  This module bridges that gap by
copying matching values from the main project environment into local env files
inside each generated project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Iterable


_ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_ENV_ASSIGNMENT_RE = re.compile(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=(.*)$")
_ENV_REFERENCE_PATTERNS = (
    re.compile(r"process\.env\.([A-Z][A-Z0-9_]*)"),
    re.compile(r"import\.meta\.env\.([A-Z][A-Z0-9_]*)"),
    re.compile(r"os\.environ(?:\.get)?\(\s*['\"]([A-Z][A-Z0-9_]*)['\"]"),
    re.compile(r"os\.environ\[\s*['\"]([A-Z][A-Z0-9_]*)['\"]\s*\]"),
    re.compile(r"Deno\.env\.get\(\s*['\"]([A-Z][A-Z0-9_]*)['\"]"),
)

_SKIP_DIRS = {
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
_SKIP_FILES = {
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
}
_SCAN_SUFFIXES = {
    ".cjs",
    ".env.example",
    ".env.local.example",
    ".env.sample",
    ".html",
    ".js",
    ".jsx",
    ".mjs",
    ".py",
    ".ts",
    ".tsx",
}

_PLACEHOLDER_RE = re.compile(
    r"(^$|"
    r"your[-_\s]|"
    r"placeholder|"
    r"changeme|"
    r"change-me|"
    r"example|"
    r"<[^>]+>|"
    r"\bxxx+\b|"
    r"\btodo\b)",
    re.IGNORECASE,
)

_PUBLIC_PREFIXES = ("VITE_", "NEXT_PUBLIC_", "PUBLIC_")
_SENSITIVE_MARKERS = (
    "API_KEY",
    "API_SECRET",
    "DATABASE_URL",
    "PASSWORD",
    "POSTGRES_URL",
    "PRIVATE",
    "SECRET",
    "SERVICE_ROLE",
    "TOKEN",
    "WEBHOOK",
)

_ALIAS_GROUPS = (
    (
        "SUPABASE_URL",
        "VITE_SUPABASE_URL",
        "NEXT_PUBLIC_SUPABASE_URL",
    ),
    (
        "SUPABASE_ANON_KEY",
        "VITE_SUPABASE_ANON_KEY",
        "NEXT_PUBLIC_SUPABASE_ANON_KEY",
    ),
    (
        "SUPABASE_SERVICE_ROLE_KEY",
        "SUPABASE_SERVICE_KEY",
    ),
    (
        "CLERK_PUBLISHABLE_KEY",
        "VITE_CLERK_PUBLISHABLE_KEY",
        "NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY",
    ),
    (
        "CLERK_SECRET_KEY",
        "CLERK_API_KEY",
    ),
    (
        "STRIPE_PUBLISHABLE_KEY",
        "VITE_STRIPE_PUBLISHABLE_KEY",
        "NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY",
    ),
    (
        "STRIPE_SECRET_KEY",
        "STRIPE_API_KEY",
    ),
    (
        "RESEND_API_KEY",
        "VITE_RESEND_API_KEY",
    ),
    (
        "RESEND_FROM_EMAIL",
        "EMAIL_FROM",
        "VITE_EMAIL_FROM",
    ),
    (
        "CLOUDINARY_CLOUD_NAME",
        "VITE_CLOUDINARY_CLOUD_NAME",
        "NEXT_PUBLIC_CLOUDINARY_CLOUD_NAME",
    ),
    (
        "CLOUDINARY_UPLOAD_PRESET",
        "VITE_CLOUDINARY_UPLOAD_PRESET",
        "NEXT_PUBLIC_CLOUDINARY_UPLOAD_PRESET",
    ),
    (
        "CLOUDINARY_API_KEY",
        "VITE_CLOUDINARY_API_KEY",
        "NEXT_PUBLIC_CLOUDINARY_API_KEY",
    ),
    (
        "CLOUDINARY_API_SECRET",
    ),
    (
        "DATABASE_URL",
        "POSTGRES_URL",
        "NEON_DATABASE_URL",
    ),
)
_ALIAS_TO_GROUP = {name: group for group in _ALIAS_GROUPS for name in group}

_SERVICE_DEFAULT_KEYS = {
    "supabase": {
        "SUPABASE_URL",
        "SUPABASE_ANON_KEY",
        "VITE_SUPABASE_URL",
        "VITE_SUPABASE_ANON_KEY",
        "NEXT_PUBLIC_SUPABASE_URL",
        "NEXT_PUBLIC_SUPABASE_ANON_KEY",
    },
    "clerk": {
        "CLERK_PUBLISHABLE_KEY",
        "VITE_CLERK_PUBLISHABLE_KEY",
        "NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY",
    },
    "resend": {"RESEND_API_KEY", "RESEND_FROM_EMAIL", "EMAIL_FROM"},
    "stripe": {
        "STRIPE_PUBLISHABLE_KEY",
        "VITE_STRIPE_PUBLISHABLE_KEY",
        "NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY",
        "STRIPE_SECRET_KEY",
        "STRIPE_WEBHOOK_SECRET",
    },
    "cloudinary": {
        "CLOUDINARY_CLOUD_NAME",
        "CLOUDINARY_UPLOAD_PRESET",
        "CLOUDINARY_API_KEY",
        "CLOUDINARY_API_SECRET",
        "VITE_CLOUDINARY_CLOUD_NAME",
        "VITE_CLOUDINARY_UPLOAD_PRESET",
    },
    "neon": {"DATABASE_URL", "POSTGRES_URL", "NEON_DATABASE_URL"},
}

_SERVICE_TOKENS = {
    "supabase": ("supabase", "@supabase/supabase-js"),
    "clerk": ("clerk", "@clerk/"),
    "resend": ("resend",),
    "stripe": ("stripe",),
    "cloudinary": ("cloudinary",),
    "neon": ("neon", "@neondatabase/serverless"),
}


@dataclass(slots=True)
class EnvSyncResult:
    written_files: list[str] = field(default_factory=list)
    filled_keys: list[str] = field(default_factory=list)
    missing_keys: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _ScanResult:
    referenced_keys: set[str] = field(default_factory=set)
    services: set[str] = field(default_factory=set)
    process_env_dirs: set[Path] = field(default_factory=set)
    python_env_dirs: set[Path] = field(default_factory=set)
    package_dirs: set[Path] = field(default_factory=set)
    frontend_dirs: set[Path] = field(default_factory=set)
    server_dirs: set[Path] = field(default_factory=set)


def _strip_env_value(raw_value: str) -> str:
    value = raw_value.strip()
    if (
        len(value) >= 2
        and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'"))
    ):
        value = value[1:-1]
    return value.strip()


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return values

    for line in lines:
        match = _ENV_ASSIGNMENT_RE.match(line)
        if not match:
            continue
        key, raw_value = match.groups()
        values[key] = _strip_env_value(raw_value)
    return values


def _is_placeholder(value: str | None) -> bool:
    return value is None or bool(_PLACEHOLDER_RE.search(value.strip()))


def _is_usable_value(value: str | None) -> bool:
    return value is not None and not _is_placeholder(value)


def _is_public_key(name: str) -> bool:
    return (
        name.startswith(_PUBLIC_PREFIXES)
        or "PUBLISHABLE" in name
        or "ANON_KEY" in name
        or name.endswith("_URL")
        or name.endswith("_CLOUD_NAME")
        or name.endswith("_UPLOAD_PRESET")
    )


def _is_sensitive_key(name: str) -> bool:
    if "PUBLISHABLE" in name or "ANON_KEY" in name:
        return False
    return any(marker in name for marker in _SENSITIVE_MARKERS)


def _normalize_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _iter_scannable_files(project_root: Path) -> Iterable[Path]:
    for current_root, dirnames, filenames in os.walk(project_root):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in _SKIP_DIRS and not name.startswith(".cache")
        ]
        current = Path(current_root)
        for filename in filenames:
            if filename in _SKIP_FILES:
                continue
            path = current / filename
            lower_name = filename.lower()
            lower_path = path.as_posix().lower()
            if lower_name in {".env", ".env.local"}:
                continue
            if lower_name == "package.json" or any(lower_path.endswith(suffix) for suffix in _SCAN_SUFFIXES):
                yield path


def _scan_project(project_root: Path) -> _ScanResult:
    result = _ScanResult()

    for package_path in project_root.rglob("package.json"):
        if any(part in _SKIP_DIRS for part in package_path.parts):
            continue
        result.package_dirs.add(package_path.parent)
        try:
            text = package_path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        if any(token in text for token in ("vite", "react", "next", "@vitejs/plugin-react")):
            result.frontend_dirs.add(package_path.parent)
        if any(token in text for token in ("express", "fastify", "hono", "nestjs", "dotenv", "server")):
            result.server_dirs.add(package_path.parent)

    for path in _iter_scannable_files(project_root):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        lower = text.lower()
        for service, tokens in _SERVICE_TOKENS.items():
            if any(token in lower for token in tokens):
                result.services.add(service)

        for pattern in _ENV_REFERENCE_PATTERNS:
            for match in pattern.finditer(text):
                result.referenced_keys.add(match.group(1))

        if path.name.lower() in {".env.example", ".env.local.example", ".env.sample"}:
            for line in text.splitlines():
                match = _ENV_ASSIGNMENT_RE.match(line)
                if match:
                    result.referenced_keys.add(match.group(1))

        normalized = path.as_posix().lower()
        if "process.env" in text:
            result.process_env_dirs.add(path.parent)
        if "os.environ" in text:
            result.python_env_dirs.add(path.parent)
        if "import.meta.env" in text or "next_public_" in lower:
            result.frontend_dirs.add(path.parent)
        if any(part in normalized.split("/") for part in ("backend", "server", "api")):
            if path.suffix.lower() in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".py"}:
                result.server_dirs.add(path.parent)

    return result


def _source_values(source_env_path: Path | None) -> dict[str, str]:
    values = dict(os.environ)
    if source_env_path is not None:
        values.update(_read_env_file(source_env_path))
    return values


def _resolve_env_value(key: str, values: dict[str, str]) -> str | None:
    direct_value = values.get(key)
    if _is_usable_value(direct_value):
        return direct_value

    group = _ALIAS_TO_GROUP.get(key)
    if group:
        for alias in group:
            candidate = values.get(alias)
            if _is_usable_value(candidate):
                return candidate
    return None


def _expand_needed_keys(scan: _ScanResult, values: dict[str, str]) -> set[str]:
    needed = {key for key in scan.referenced_keys if _ENV_NAME_RE.match(key)}

    for service in scan.services:
        for key in _SERVICE_DEFAULT_KEYS.get(service, set()):
            if _resolve_env_value(key, values):
                needed.add(key)

    for key in list(needed):
        group = _ALIAS_TO_GROUP.get(key)
        if not group:
            continue
        for alias in group:
            if alias in scan.referenced_keys or alias.startswith(_PUBLIC_PREFIXES):
                needed.add(alias)
        if key.startswith("SUPABASE") or "SUPABASE" in key:
            needed.update({"SUPABASE_URL", "SUPABASE_ANON_KEY", "VITE_SUPABASE_URL", "VITE_SUPABASE_ANON_KEY"})
        if "CLERK_PUBLISHABLE_KEY" in group:
            needed.update({"CLERK_PUBLISHABLE_KEY", "VITE_CLERK_PUBLISHABLE_KEY", "NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY"})

    return needed


def _nearest_package_or_root(directory: Path, project_root: Path) -> Path:
    current = directory
    while project_root == current or project_root in current.parents:
        if (current / "package.json").exists() or (current / "requirements.txt").exists():
            return current
        if current == project_root:
            return project_root
        current = current.parent
    return project_root


def _target_dirs(scan: _ScanResult, project_root: Path) -> tuple[set[Path], set[Path]]:
    frontend_dirs = {_nearest_package_or_root(path, project_root) for path in scan.frontend_dirs}
    server_dirs = {_nearest_package_or_root(path, project_root) for path in scan.server_dirs}
    server_dirs.update(_nearest_package_or_root(path, project_root) for path in scan.process_env_dirs)
    server_dirs.update(_nearest_package_or_root(path, project_root) for path in scan.python_env_dirs)

    for package_dir in scan.package_dirs:
        name = package_dir.name.lower()
        if name in {"backend", "server", "api"}:
            server_dirs.add(package_dir)
        if name in {"frontend", "client", "web", "ui"}:
            frontend_dirs.add(package_dir)

    if not frontend_dirs and not server_dirs:
        frontend_dirs.add(project_root)

    return frontend_dirs, server_dirs


def _merge_env_file(path: Path, additions: dict[str, str]) -> bool:
    existing = _read_env_file(path)
    merged = dict(existing)
    changed = False

    for key, value in sorted(additions.items()):
        existing_value = existing.get(key)
        if _is_usable_value(existing_value):
            continue
        if existing_value != value:
            merged[key] = value
            changed = True

    if not changed and path.exists():
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(f"{key}={value}" for key, value in sorted(merged.items())) + "\n"
    path.write_text(content, encoding="utf-8")
    return True


def _ensure_gitignore(project_root: Path) -> None:
    gitignore = project_root / ".gitignore"
    existing = gitignore.read_text(encoding="utf-8", errors="ignore") if gitignore.exists() else ""
    lines = existing.splitlines()
    wanted = [".env", ".env.*", "!.env.example", "!.env.*.example"]
    additions = [line for line in wanted if line not in lines]
    if not additions:
        return
    prefix = existing.rstrip() + "\n" if existing.strip() else ""
    gitignore.write_text(prefix + "\n".join(additions) + "\n", encoding="utf-8")


def auto_fill_project_env(
    project_root: str | Path,
    *,
    source_env_path: str | Path | None = None,
) -> EnvSyncResult:
    """Copy available provider env values into a generated project.

    Values are read from the main project's `.env` by default plus the process
    environment.  Placeholder-looking values are ignored.
    """
    root = Path(project_root).resolve()
    result = EnvSyncResult()
    if not root.exists():
        return result
    if os.getenv("AUTO_FILL_GENERATED_ENV", "1").strip().lower() in {"0", "false", "no", "off"}:
        return result

    source_path = Path(source_env_path).resolve() if source_env_path else Path(__file__).with_name(".env")
    values = _source_values(source_path)
    scan = _scan_project(root)
    needed_keys = _expand_needed_keys(scan, values)
    if not needed_keys:
        return result

    resolved: dict[str, str] = {}
    for key in sorted(needed_keys):
        value = _resolve_env_value(key, values)
        if value is None:
            result.missing_keys.append(key)
        else:
            resolved[key] = value

    if not resolved:
        result.missing_keys = sorted(set(result.missing_keys))
        return result

    frontend_dirs, server_dirs = _target_dirs(scan, root)
    files_to_entries: dict[Path, dict[str, str]] = {}

    public_entries = {key: value for key, value in resolved.items() if _is_public_key(key) and not _is_sensitive_key(key)}
    server_entries = dict(resolved)

    for directory in frontend_dirs:
        entries = server_entries if directory in server_dirs and (directory / "next.config.js").exists() else public_entries
        if entries:
            files_to_entries.setdefault(directory / ".env.local", {}).update(entries)

    for directory in server_dirs:
        if server_entries:
            files_to_entries.setdefault(directory / ".env.local", {}).update(server_entries)
            files_to_entries.setdefault(directory / ".env", {}).update(server_entries)

    if not files_to_entries and public_entries:
        files_to_entries[root / ".env.local"] = public_entries

    for path, entries in sorted(files_to_entries.items(), key=lambda item: item[0].as_posix()):
        if _merge_env_file(path, entries):
            result.written_files.append(_normalize_path(path, root))
        result.filled_keys.extend(entries.keys())

    if result.written_files:
        _ensure_gitignore(root)

    result.written_files = sorted(set(result.written_files))
    result.filled_keys = sorted(set(result.filled_keys))
    result.missing_keys = sorted(set(result.missing_keys) - set(result.filled_keys))
    return result
