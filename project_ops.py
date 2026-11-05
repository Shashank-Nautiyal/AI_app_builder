from __future__ import annotations

import shutil
import uuid
import zipfile
from pathlib import Path, PurePosixPath


SNAPSHOT_ROOT_NAME = ".project_snapshots"
SNAPSHOT_IGNORE_NAMES = {
    ".git",
    ".next",
    ".venv",
    "__pycache__",
    "node_modules",
    "venv",
}


def reset_directory(path: str | Path) -> Path:
    target = Path(path).resolve()
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _safe_target_path(root: Path, relative_parts: tuple[str, ...]) -> Path:
    candidate = (root / Path(*relative_parts)).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Archive entry escapes the project root.")
    return candidate


def _normalize_relative_parts(relative_path: str) -> tuple[str, ...]:
    path_str = str(relative_path or "").replace("\\", "/")
    parts = tuple(part for part in PurePosixPath(path_str).parts if part not in ("", "."))
    if any(part == ".." for part in parts):
        raise ValueError("Uploaded file path escapes the project root.")
    return parts


def _strip_common_root_folder(relative_paths: list[str]) -> str | None:
    top_levels = set()
    eligible_paths = 0

    for relative_path in relative_paths:
        parts = _normalize_relative_parts(relative_path)
        if not parts or parts[0] == "__MACOSX":
            continue
        if len(parts) < 2:
            return None
        top_levels.add(parts[0])
        eligible_paths += 1

    return next(iter(top_levels)) if eligible_paths > 0 and len(top_levels) == 1 else None


def _strip_single_root_folder(members: list[zipfile.ZipInfo]) -> str | None:
    top_levels = set()
    for member in members:
        parts = PurePosixPath(member.filename).parts
        if not parts:
            continue
        if parts[0] == "__MACOSX":
            continue
        top_levels.add(parts[0])
    return next(iter(top_levels)) if len(top_levels) == 1 else None


def extract_project_zip(archive_path: str | Path, target_dir: str | Path) -> list[str]:
    archive = Path(archive_path).resolve()
    destination = reset_directory(target_dir)
    extracted_files: list[str] = []

    with zipfile.ZipFile(archive) as zf:
        file_members = [member for member in zf.infolist() if not member.is_dir()]
        strip_root = _strip_single_root_folder(file_members)

        for member in file_members:
            parts = list(PurePosixPath(member.filename).parts)
            if not parts or parts[0] == "__MACOSX":
                continue
            if strip_root and parts and parts[0] == strip_root:
                parts = parts[1:]
            if not parts:
                continue

            target_path = _safe_target_path(destination, tuple(parts))
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as source, target_path.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            extracted_files.append(target_path.relative_to(destination).as_posix())

    if not extracted_files:
        raise ValueError("The uploaded zip file did not contain any project files.")

    return extracted_files


def import_project_files(file_entries: list[tuple[str, bytes]], target_dir: str | Path) -> list[str]:
    destination = reset_directory(target_dir)
    imported_files: list[str] = []
    strip_root = _strip_common_root_folder([relative_path for relative_path, _ in file_entries])

    for relative_path, content in file_entries:
        parts = list(_normalize_relative_parts(relative_path))
        if not parts or parts[0] == "__MACOSX":
            continue
        if strip_root and parts[0] == strip_root:
            parts = parts[1:]
        if not parts:
            continue

        target_path = _safe_target_path(destination, tuple(parts))
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(content)
        imported_files.append(target_path.relative_to(destination).as_posix())

    if not imported_files:
        raise ValueError("The uploaded files did not contain any project files.")

    return imported_files


def create_project_snapshot(project_dir: str | Path) -> str | None:
    source = Path(project_dir).resolve()
    if not source.exists():
        return None

    snapshot_root = source.parent / SNAPSHOT_ROOT_NAME / source.name
    snapshot_root.mkdir(parents=True, exist_ok=True)
    snapshot_dir = snapshot_root / f"snapshot-{uuid.uuid4().hex[:10]}"

    def _ignore(path: str, names: list[str]) -> set[str]:
        return {name for name in names if name in SNAPSHOT_IGNORE_NAMES}

    shutil.copytree(source, snapshot_dir, ignore=_ignore)
    return str(snapshot_dir)


def _copytree_robust(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            _copytree_robust(item, target)
        else:
            try:
                shutil.copy2(item, target)
            except Exception:
                try:
                    shutil.copyfile(item, target)
                except Exception as e:
                    print(f"[restore] Failed to copy {item} to {target}: {e}")


def restore_project_snapshot(project_dir: str | Path, snapshot_dir: str | Path) -> str:
    project_path = Path(project_dir).resolve()
    snapshot_path = Path(snapshot_dir).resolve()
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    # Selective cleanup: skip deleting anything in SNAPSHOT_IGNORE_NAMES (like node_modules, .git, etc.)
    if project_path.exists():
        for item in project_path.iterdir():
            if item.name in SNAPSHOT_IGNORE_NAMES:
                continue
            try:
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
            except Exception:
                pass

    _copytree_robust(snapshot_path, project_path)
    return str(project_path)
