"""
github_scout.py — GitHub-first project bootstrap for the pipeline.

Flow:
  1. search_github_repos()     — find top matching repos via GitHub Search API
  2. rank_and_select_repo()    — score by stars, recency, language match
  3. download_repo_zip()       — download + extract into project root
  4. analyze_repo_files()      — detect DB/auth patterns in every file
  5. generate_adaptation_plan() — produce a RepoAdaptationPlan

All GitHub API calls use GITHUB_TOKEN from .env when available.
"""

from __future__ import annotations

import io
import os
import re
import zipfile
import urllib.request
import urllib.parse
import urllib.error
import json
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from codemind.paths import ENV_FILE

load_dotenv(ENV_FILE, override=True)

_GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
_GITHUB_SEARCH_QUERY_LIMIT = max(1, int(os.getenv("GITHUB_SEARCH_QUERY_LIMIT", "4")))
_GITHUB_SEARCH_LLM_ENABLED = os.getenv("GITHUB_SEARCH_LLM_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
_GITHUB_SEARCH_LLM_TIMEOUT_SECONDS = float(os.getenv("GITHUB_SEARCH_LLM_TIMEOUT_SECONDS", "15"))
_PREFERRED_AUTH = os.getenv("PREFERRED_AUTH_PROVIDER", "clerk").strip().lower()
_PREFERRED_DB = os.getenv("PREFERRED_DB_PROVIDER", "supabase").strip().lower()
_PREFERRED_STORAGE = os.getenv("PREFERRED_STORAGE_PROVIDER", "supabase-storage").strip().lower()
_PREFERRED_EMAIL = os.getenv("PREFERRED_EMAIL_PROVIDER", "resend").strip().lower()
_PREFERRED_PAYMENTS = os.getenv("PREFERRED_PAYMENTS_PROVIDER", "stripe").strip().lower()

if _PREFERRED_AUTH != "clerk":
    _PREFERRED_AUTH = "clerk"
if _PREFERRED_DB not in {"supabase", "neon"}:
    _PREFERRED_DB = "supabase"
if _PREFERRED_STORAGE not in {"supabase-storage", "cloudinary"}:
    _PREFERRED_STORAGE = "supabase-storage"
if _PREFERRED_EMAIL != "resend":
    _PREFERRED_EMAIL = "resend"
if _PREFERRED_PAYMENTS != "stripe":
    _PREFERRED_PAYMENTS = "stripe"


# Each entry: (pattern_regex, service_type, original_provider, replacement)
_DETECTION_RULES: list[tuple[str, str, str, str]] = [
    (r"mongoose|MongoClient|mongodb\+srv|from ['\"]mongoose['\"]",
     "database", "mongodb", "supabase"),
    (r"import.*from ['\"]pg['\"]|new Pool\(|new Client\(|require\(['\"]pg['\"]\)",
     "database", "postgresql", "supabase"),
    (r"mysql2|mysql\.createConnection|createPool.*host",
     "database", "mysql", "supabase"),
    (r"PrismaClient|from ['\"]@prisma/client['\"]|prisma\.\w+\.\w+",
     "database", "prisma", "supabase"),
    (r"from ['\"]firebase/firestore['\"]|getFirestore|collection\(db,",
     "database", "firebase-firestore", "supabase"),
    (r"from ['\"]firebase-admin['\"]|admin\.firestore\(\)",
     "database", "firebase-admin", "supabase"),
    (r"Sequelize|DataTypes\.|sequelize\.define",
     "database", "sequelize", "supabase"),
    (r"TypeORM|@Entity\(\)|@Column\(\)|createConnection\(",
     "database", "typeorm", "supabase"),
    (r"knex\(|from ['\"]knex['\"]",
     "database", "knex", "supabase"),
    (r"better-sqlite3|sqlite3\.Database|import sqlite",
     "database", "sqlite3", "supabase"),
    (r"from ['\"]@neondatabase/serverless['\"]|neon\(|NEON_DATABASE_URL",
     "database", "neon", "supabase"),
    (r"from ['\"]drizzle-orm|drizzle\(",
     "database", "drizzle", "supabase"),
    (r"from ['\"]passport['\"]|passport\.use\(|passport\.authenticate",
     "auth", "passport", "clerk"),
    (r"jwt\.sign\(|jwt\.verify\(|from ['\"]jsonwebtoken['\"]",
     "auth", "jwt", "clerk"),
    (r"bcrypt\.hash\(|bcrypt\.compare\(|from ['\"]bcrypt",
     "auth", "bcrypt", "clerk"),
    (r"express-session|session\(\{.*secret",
     "auth", "express-session", "clerk"),
    (r"from ['\"]next-auth['\"]|NextAuth\(|getSession\(",
     "auth", "next-auth", "clerk"),
    (r"from ['\"]firebase/auth['\"]|signInWithEmailAndPassword|getAuth\(\)",
     "auth", "firebase-auth", "clerk"),
    (r"from ['\"]@auth/core['\"]|from ['\"]auth\.js['\"]",
     "auth", "auth.js", "clerk"),
    # Custom auth patterns (login forms with manual password hashing)
    (r"password.*hash|hashPassword|comparePassword|LOGIN_SECRET|AUTH_SECRET",
     "auth", "custom-auth", "clerk"),
    (r"from ['\"]firebase/storage['\"]|getStorage\(\)|ref\(storage",
     "storage", "firebase-storage", "supabase-storage"),
    (r"aws-sdk.*S3|S3Client|PutObjectCommand",
     "storage", "aws-s3", "supabase-storage"),
    (r"cloudinary\.v2|cloudinary\.uploader",
     "storage", "cloudinary", "supabase-storage"),
    (r"from ['\"]uploadthing|UTApi|createUploadthing",
     "storage", "uploadthing", "supabase-storage"),
    (r"multer|diskStorage\(|memoryStorage\(",
     "storage", "local-upload", "supabase-storage"),
    # Email
    (r"nodemailer|createTransport\(",
     "email", "nodemailer", "resend"),
    (r"@sendgrid/mail|sendgrid",
     "email", "sendgrid", "resend"),
    (r"mailgun-js|mailgun\.client|api\.mailgun\.net",
     "email", "mailgun", "resend"),
    (r"postmark|ServerClient\(",
     "email", "postmark", "resend"),
    (r"from ['\"]resend['\"]|new Resend\(",
     "email", "resend", "resend"),
    # Payments
    (r"from ['\"]stripe['\"]|require\(['\"]stripe['\"]\)|new Stripe\(",
     "payments", "stripe", "stripe"),
    (r"paypal|@paypal|braintree|razorpay|paddle",
     "payments", "non-stripe-payments", "stripe"),
]

# Files/directories we never analyze or copy
_SKIP_DIRS = {
    ".git", "node_modules", ".next", "dist", "build", "__pycache__",
    ".venv", "venv", ".cache", "coverage", ".nyc_output",
}
_SKIP_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".woff", ".woff2",
    ".ttf", ".eot", ".mp4", ".mp3", ".pdf", ".zip", ".tar", ".gz",
    ".lock",  # package-lock.json is fine but yarn.lock / poetry.lock aren't source
}



def _gh_headers() -> dict[str, str]:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if _GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {_GITHUB_TOKEN}"
    return h


def _gh_get(url: str, params: dict | None = None, *, retries: int = 3) -> dict | list | None:
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_gh_headers())
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 403:
                # Rate limit — wait and retry
                reset = int(e.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(1, reset - int(time.time()))
                print(f"[github_scout] Rate limited. Waiting {min(wait, 30)}s...")
                time.sleep(min(wait, 30))
            elif e.code == 422:
                print(f"[github_scout] GitHub 422 (bad query). Skipping.")
                return None
            else:
                print(f"[github_scout] HTTP {e.code} on attempt {attempt+1}: {e.reason}")
                time.sleep(2 ** attempt)
        except Exception as exc:
            print(f"[github_scout] Request error on attempt {attempt+1}: {exc}")
            time.sleep(2 ** attempt)
    return None



def search_github_repos(prompt: str, tech_stack: str = "", max_results: int = 5) -> list[dict]:
    """
    Search GitHub for repos matching the prompt + tech stack.
    Uses a hybrid query strategy: exact user keywords first, then LLM/synonym
    phrase expansions. Returns deduped repo dicts from the GitHub API.
    """
    queries = _build_github_search_queries(
        prompt,
        tech_stack,
        max_queries=_GITHUB_SEARCH_QUERY_LIMIT,
    )
    if not queries:
        print("[github_scout] No usable GitHub search query could be built.")
        return []

    repos_by_id: dict[int | str, dict] = {}
    for idx, query in enumerate(queries, start=1):
        print(f"[github_scout] Searching GitHub ({idx}/{len(queries)}): {query!r}")
        data = _gh_get(
            "https://api.github.com/search/repositories",
            {"q": query, "sort": "stars", "order": "desc", "per_page": max_results},
        )
        if not data or "items" not in data:
            print("[github_scout] No results for this GitHub search query.")
            continue

        for repo in data["items"]:
            key = repo.get("id") or repo.get("full_name")
            if key and key not in repos_by_id:
                repos_by_id[key] = repo

    if not repos_by_id:
        print("[github_scout] No results from GitHub search.")
        return []

    repos = list(repos_by_id.values())
    print(f"[github_scout] Found {len(repos)} unique candidate repo(s) across {len(queries)} query/queries.")
    return repos


_REPO_RELEVANCE_STOPWORDS = {
    "a", "an", "and", "app", "application", "based", "build", "create", "for", "from",
    "generate", "have", "make", "management", "mark", "project", "system", "that",
    "the", "this", "tool", "use", "uses", "using", "with", "website",
}


def _repo_prompt_terms(prompt: str) -> set[str]:
    terms: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", prompt.lower()):
        token = {"attandance": "attendance", "attendence": "attendance"}.get(token, token)
        if len(token) < 3 or token in _REPO_RELEVANCE_STOPWORDS:
            continue
        terms.add(token)
    return terms


def _repo_search_text(repo: dict) -> str:
    topics = repo.get("topics") or []
    if not isinstance(topics, list):
        topics = []
    return " ".join(
        str(part or "")
        for part in (
            repo.get("full_name", ""),
            repo.get("name", ""),
            repo.get("description", ""),
            repo.get("language", ""),
            " ".join(str(topic) for topic in topics),
        )
    ).lower()


def _repo_relevance_score(repo: dict, prompt_terms: set[str]) -> float:
    if not prompt_terms:
        return 1.0

    full_name = str(repo.get("full_name") or repo.get("name") or "").lower()
    description = str(repo.get("description") or "").lower()
    search_text = _repo_search_text(repo)
    score = 0.0
    for term in prompt_terms:
        if term in full_name:
            score += 3.0
        elif term in description:
            score += 2.0
        elif term in search_text:
            score += 1.0

    if {"face", "attendance"} <= prompt_terms and "face" in search_text and "attendance" in search_text:
        score += 8.0
    if "attendance" in prompt_terms and any(term in search_text for term in ("attendance", "attendence")):
        score += 4.0
    if "face" in prompt_terms and any(term in search_text for term in ("face", "facial", "recognition")):
        score += 4.0
    return score


def rank_and_select_repo(repos: list[dict], prompt: str) -> dict | None:
    """
    Score each repo and return the best match.
    Scoring: prompt relevance first, then stars, recency, and repo quality.
    """
    if not repos:
        return None

    import math

    prompt_terms = _repo_prompt_terms(prompt)
    best_repo: dict | None = None
    best_score: float = -1.0

    for repo in repos:
        score = 0.0
        relevance = _repo_relevance_score(repo, prompt_terms)
        if prompt_terms and relevance <= 0:
            continue
        score += relevance * 20

        # Stars matter heavily once a repo is relevant.
        stars = repo.get("stargazers_count", 0)
        score += math.log1p(stars) * 8

        # Recency: pushed within last 2 years gets a bonus
        pushed = repo.get("pushed_at", "")
        if pushed:
            try:
                from datetime import datetime, timezone
                age_days = (datetime.now(timezone.utc) -
                            datetime.fromisoformat(pushed.replace("Z", "+00:00"))).days
                if age_days < 730:
                    score += max(0, 10 - age_days / 100)
            except Exception:
                pass

        # Has a README
        if repo.get("has_wiki") or repo.get("has_pages"):
            score += 1

        # Not archived / not a fork (prefer originals)
        if not repo.get("archived", False):
            score += 3
        if not repo.get("fork", False):
            score += 5

        # Minimum quality gate: must have at least 5 stars
        if stars < 5:
            continue

        if score > best_score:
            best_score = score
            best_repo = repo

    if best_repo:
        print(f"[github_scout] Selected: {best_repo['full_name']} "
              f"(stars={best_repo.get('stargazers_count', 0)}, score={best_score:.1f})")
    return best_repo


def download_repo_zip(repo: dict, project_root: Path) -> bool:
    """
    Download the default branch of `repo` as a zip and extract into `project_root`.
    Existing files in project_root are NOT removed — the repo is overlaid on top.
    Returns True on success.
    """
    full_name   = repo["full_name"]
    default_branch = repo.get("default_branch", "main")
    zip_url = f"https://github.com/{full_name}/archive/refs/heads/{default_branch}.zip"

    print(f"[github_scout] Downloading {zip_url} ...")
    try:
        req = urllib.request.Request(zip_url, headers=_gh_headers())
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
    except Exception as exc:
        print(f"[github_scout] Download failed: {exc}")
        return False

    print(f"[github_scout] Downloaded {len(raw) // 1024} KB. Extracting...")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            # The zip always has a single top-level folder like "repo-name-main/"
            members = zf.namelist()
            if not members:
                return False
            prefix = members[0].split("/")[0] + "/"

            for member in members:
                # Strip the top-level prefix
                relative = member[len(prefix):]
                if not relative:
                    continue

                # Skip unwanted dirs/files
                parts = Path(relative).parts
                if any(part in _SKIP_DIRS for part in parts):
                    continue
                if any(relative.endswith(ext) for ext in _SKIP_EXTS):
                    continue

                target = project_root / relative
                if member.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(member))

    except Exception as exc:
        print(f"[github_scout] Extraction failed: {exc}")
        return False

    print(f"[github_scout] Extracted into {project_root}")
    return True


def analyze_repo_files(project_root: Path) -> dict[str, Any]:
    """
    Walk all source files in project_root and detect DB/auth/storage patterns.

    Returns a dict:
      {
        "detections": [
          {
            "service_type": "database",
            "original_provider": "mongodb",
            "files_affected": ["backend/db.js", ...],
            "replacement": "supabase",
          },
          ...
        ],
        "framework": "react-vite" | "nextjs" | "fastapi" | "express" | ...,
        "has_env_example": bool,
        "file_count": int,
      }
    """
    detections: dict[str, dict] = {}  # keyed by original_provider

    file_count = 0
    framework  = _detect_framework_from_disk(project_root)

    for fpath in _iter_source_files(project_root):
        file_count += 1
        try:
            content = fpath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        relative = fpath.relative_to(project_root).as_posix()

        for pattern, svc_type, orig_provider, replacement in _DETECTION_RULES:
            if re.search(pattern, content, re.MULTILINE):
                if orig_provider not in detections:
                    detections[orig_provider] = {
                        "service_type": svc_type,
                        "original_provider": orig_provider,
                        "files_affected": [],
                        "replacement": replacement,
                    }
                if relative not in detections[orig_provider]["files_affected"]:
                    detections[orig_provider]["files_affected"].append(relative)

    # Services are normalized onto the user's preferred managed stack.
    for det in detections.values():
        if det["service_type"] == "auth":
            det["replacement"] = _PREFERRED_AUTH
        elif det["service_type"] == "database":
            det["replacement"] = _PREFERRED_DB
        elif det["service_type"] == "storage":
            det["replacement"] = _PREFERRED_STORAGE
        elif det["service_type"] == "email":
            det["replacement"] = _PREFERRED_EMAIL
        elif det["service_type"] == "payments":
            det["replacement"] = _PREFERRED_PAYMENTS

    has_env_example = (project_root / ".env.example").exists() or \
                      (project_root / ".env.local.example").exists()

    result = {
        "detections": list(detections.values()),
        "framework": framework,
        "has_env_example": has_env_example,
        "file_count": file_count,
        "repo_root": str(project_root),
        "dependencies": _detect_dependency_signals(project_root),
        "env_vars": _detect_env_vars(project_root),
        "key_files": _detect_key_files(project_root),
    }

    total = len(detections)
    svc_summary = ", ".join(
        f"{d['original_provider']}->{d['replacement']}" for d in result["detections"]
    )
    print(f"[github_scout] Analysis done. {file_count} files, {total} service(s) detected: {svc_summary or 'none'}")
    return result


def generate_adaptation_plan(analysis: dict[str, Any], repo: dict) -> dict[str, Any]:
    """
    Build the adaptation plan dict that will be stored in PipelineState.
    This is a plain dict (will be coerced into RepoAdaptationPlan by the schema).
    """
    detections = analysis.get("detections", [])

    # Collect orphan config files to delete (provider-specific configs)
    files_to_delete: list[str] = []
    for det in detections:
        provider = det["original_provider"]
        if provider == "prisma":
            files_to_delete.append("prisma/schema.prisma")
            files_to_delete.append("prisma/migrations")
        elif provider in ("firebase-firestore", "firebase-auth", "firebase-admin", "firebase-storage"):
            files_to_delete.extend(["firebase.json", "firebaseConfig.js", ".firebaserc"])
        elif provider == "mongodb":
            files_to_delete.append("mongo-init.js")

    # Deduplicate
    files_to_delete = list(dict.fromkeys(files_to_delete))

    auth_services = [d for d in detections if d["service_type"] == "auth"]
    db_services = [d for d in detections if d["service_type"] == "database"]
    storage_services = [d for d in detections if d["service_type"] == "storage"]
    email_services = [d for d in detections if d["service_type"] == "email"]
    payment_services = [d for d in detections if d["service_type"] == "payments"]
    other_services = [
        d for d in detections
        if d["service_type"] not in ("auth", "database", "storage", "email", "payments")
    ]

    summary_parts = [f"Downloaded '{repo.get('full_name', 'unknown')}' ({repo.get('stargazers_count', 0)} stars)."]
    if db_services:
        db_names = ", ".join(d["original_provider"] for d in db_services)
        replacements = ", ".join(sorted({d["replacement"] for d in db_services}))
        summary_parts.append(f"DB ({db_names}) -> {replacements}.")
    if auth_services:
        auth_names = ", ".join(d["original_provider"] for d in auth_services)
        summary_parts.append(f"Auth ({auth_names}) -> Clerk.")
    if storage_services:
        storage_names = ", ".join(d["original_provider"] for d in storage_services)
        replacements = ", ".join(sorted({d["replacement"] for d in storage_services}))
        summary_parts.append(f"Storage ({storage_names}) -> {replacements}.")
    if email_services:
        email_names = ", ".join(d["original_provider"] for d in email_services)
        summary_parts.append(f"Email ({email_names}) -> Resend.")
    if payment_services:
        payment_names = ", ".join(d["original_provider"] for d in payment_services)
        summary_parts.append(f"Payments ({payment_names}) -> Stripe.")
    if other_services:
        other_names = ", ".join(f"{d['original_provider']}->{d['replacement']}" for d in other_services)
        summary_parts.append(f"Other services: {other_names}.")
    if not detections:
        summary_parts.append("No services detected; coder will still use Clerk, Supabase/Neon, Supabase Storage/Cloudinary, Resend, and Stripe when those features are needed.")
    dependencies = analysis.get("dependencies", [])
    env_vars = analysis.get("env_vars", [])
    key_files = analysis.get("key_files", [])
    external_connections = [
        f"{d['service_type']}: {d['original_provider']} -> {d['replacement']}"
        for d in detections
    ]
    if env_vars:
        external_connections.append("env vars: " + ", ".join(env_vars[:12]))
    affected_count = sum(len(d["files_affected"]) for d in detections)
    if affected_count:
        summary_parts.append(f"{affected_count} file(s) need adaptation.")
    summary = " ".join(summary_parts)

    return {
        "repo_url": repo.get("html_url", ""),
        "repo_name": repo.get("full_name", ""),
        "detected_services": detections,
        "files_to_delete": files_to_delete,
        "adaptation_summary": summary,
        "framework": analysis.get("framework", ""),
        "dependencies": dependencies,
        "env_vars": env_vars,
        "key_files": key_files,
        "external_connections": external_connections,
    }



def _detect_dependency_signals(root: Path) -> list[str]:
    signals: list[str] = []

    for package_json in _iter_named_files(root, "package.json", limit=6):
        rel = package_json.relative_to(root).as_posix()
        try:
            data = json.loads(package_json.read_text(encoding="utf-8"))
        except Exception:
            continue
        deps = data.get("dependencies") or {}
        dev_deps = data.get("devDependencies") or {}
        scripts = data.get("scripts") or {}
        if deps:
            signals.append(f"{rel} dependencies: {', '.join(list(deps.keys())[:24])}")
        if dev_deps:
            signals.append(f"{rel} devDependencies: {', '.join(list(dev_deps.keys())[:18])}")
        if scripts:
            signals.append(f"{rel} scripts: {', '.join(list(scripts.keys())[:12])}")

    for req_file in _iter_named_files(root, "requirements.txt", limit=4):
        rel = req_file.relative_to(root).as_posix()
        try:
            packages = [
                line.strip()
                for line in req_file.read_text(encoding="utf-8", errors="ignore").splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
        except Exception:
            continue
        if packages:
            signals.append(f"{rel} packages: {', '.join(packages[:24])}")

    for pyproject in _iter_named_files(root, "pyproject.toml", limit=4):
        rel = pyproject.relative_to(root).as_posix()
        try:
            content = pyproject.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        names = re.findall(r'^\s*"?([A-Za-z0-9_.-]+)"?\s*[=><~]', content, re.MULTILINE)
        if names:
            signals.append(f"{rel} dependency hints: {', '.join(names[:24])}")

    return _dedupe_strings(signals, limit=40)


def _detect_env_vars(root: Path) -> list[str]:
    env_vars: set[str] = set()
    patterns = (
        r"process\.env\.([A-Z][A-Z0-9_]{2,})",
        r"import\.meta\.env\.([A-Z][A-Z0-9_]{2,})",
        r"os\.getenv\(['\"]([A-Z][A-Z0-9_]{2,})['\"]",
        r"os\.environ(?:\.get)?\(['\"]([A-Z][A-Z0-9_]{2,})['\"]",
        r"os\.environ\[['\"]([A-Z][A-Z0-9_]{2,})['\"]\]",
        r"Deno\.env\.get\(['\"]([A-Z][A-Z0-9_]{2,})['\"]",
    )

    for fpath in _iter_source_files(root):
        try:
            content = fpath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pattern in patterns:
            env_vars.update(re.findall(pattern, content))

    for env_file_name in (".env.example", ".env.local.example", ".env.sample"):
        for env_file in _iter_named_files(root, env_file_name, limit=4):
            try:
                content = env_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            env_vars.update(re.findall(r"^\s*([A-Z][A-Z0-9_]{2,})\s*=", content, re.MULTILINE))

    return sorted(env_vars)[:60]


def _detect_key_files(root: Path) -> list[str]:
    preferred_names = {
        "package.json", "requirements.txt", "pyproject.toml", "vite.config.js",
        "vite.config.ts", "next.config.js", "next.config.mjs", "main.py",
        "app.py", "server.py", "index.js", "server.js", "src/App.jsx",
        "src/App.tsx", "src/main.jsx", "src/main.tsx", "README.md",
    }
    key_files: list[str] = []
    for fpath in _iter_source_files(root):
        rel = fpath.relative_to(root).as_posix()
        basename = fpath.name
        if rel in preferred_names or basename in preferred_names:
            key_files.append(rel)
        if len(key_files) >= 50:
            break
    return _dedupe_strings(key_files, limit=50)


def _iter_named_files(root: Path, filename: str, *, limit: int = 10):
    count = 0
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in _SKIP_DIRS]
        if filename in filenames:
            yield Path(current_root) / filename
            count += 1
            if count >= limit:
                return


def _dedupe_strings(values: list[str], *, limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = re.sub(r"\s+", " ", str(value)).strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def _build_github_search_queries(
    prompt: str,
    tech_stack: str = "",
    *,
    max_queries: int = 4,
    llm_phrases: list[str] | None = None,
) -> list[str]:
    """Build exact-first GitHub queries, then broaden with LLM/synonym phrases."""
    lang = _detect_language(tech_stack + " " + prompt)
    phrases = _build_search_phrases(prompt, tech_stack, max_phrases=max_queries, llm_phrases=llm_phrases)

    queries: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        query = _github_query_from_phrase(phrase, lang)
        normalized = re.sub(r"\s+", " ", query.lower()).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            queries.append(query)
        if len(queries) >= max_queries:
            break
    return queries


def _build_search_phrases(
    prompt: str,
    tech_stack: str = "",
    *,
    max_phrases: int = 4,
    llm_phrases: list[str] | None = None,
) -> list[str]:
    """Return exact user keywords first, then LLM and deterministic expansions."""
    exact_keywords = _extract_search_keywords(prompt, tech_stack)
    exact_phrase = " ".join(exact_keywords[:6])

    phrases: list[str] = []
    if exact_phrase:
        phrases.append(exact_phrase)

    if llm_phrases is None:
        llm_phrases = _generate_llm_search_phrases(prompt, tech_stack, limit=max(0, max_phrases - len(phrases)))

    phrases.extend(llm_phrases)
    phrases.extend(_heuristic_search_phrases(prompt, tech_stack))
    return _dedupe_search_phrases(phrases, limit=max_phrases)


def _github_query_from_phrase(phrase: str, lang: str = "") -> str:
    sanitized = _sanitize_search_phrase(phrase)
    if not sanitized:
        return ""
    query = f"{sanitized} in:readme,description"
    if lang:
        query += f" language:{lang}"
    return query


def _generate_llm_search_phrases(prompt: str, tech_stack: str, *, limit: int = 3) -> list[str]:
    """Ask an available chat provider for concise GitHub search phrases."""
    if limit <= 0 or not _GITHUB_SEARCH_LLM_ENABLED:
        return []

    system = (
        "You generate GitHub repository search phrases. Return only a JSON array "
        "of short phrases, no markdown. Keep phrases under 6 words. Include common "
        "synonyms and product-category wording, not full sentences."
    )
    user = (
        f"User request: {prompt}\n"
        f"Tech stack hint: {tech_stack or '(none)'}\n"
        f"Return {limit} alternative GitHub search phrases."
    )

    content = _call_search_query_llm(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    phrases = _parse_llm_phrase_list(content)
    if phrases:
        print(f"[github_scout] LLM query expansion: {phrases[:limit]}")
    return phrases[:limit]


def _call_search_query_llm(messages: list[dict[str, str]]) -> str | None:
    """Use the first configured OpenAI-compatible provider for query expansion."""
    providers = _configured_search_query_llm_providers()
    if not providers:
        print("[github_scout] LLM query expansion skipped: no provider API key configured.")
        return None

    payload_base = {
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 180,
    }
    for provider in providers:
        payload = {**payload_base, "model": provider["model"]}
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            provider["url"],
            data=body,
            headers={
                "Authorization": f"Bearer {provider['api_key']}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=_GITHUB_SEARCH_LLM_TIMEOUT_SECONDS) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = data.get("choices", [{}])[0].get("message", {}).get("content")
            if content:
                return str(content)
        except Exception as exc:
            print(f"[github_scout] LLM query expansion failed via {provider['name']}: {exc}")
    return None


def _configured_search_query_llm_providers() -> list[dict[str, str]]:
    providers: list[dict[str, str]] = []
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if openrouter_key:
        providers.append({
            "name": "openrouter",
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "api_key": openrouter_key,
            "model": os.getenv("GITHUB_SEARCH_LLM_MODEL")
            or os.getenv("OPENROUTER_PLANNER_FALLBACK_MODEL", "minimax/minimax-m2.5:free"),
        })

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key:
        providers.append({
            "name": "groq",
            "url": "https://api.groq.com/openai/v1/chat/completions",
            "api_key": groq_key,
            "model": os.getenv("GITHUB_SEARCH_GROQ_MODEL")
            or os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant"),
        })

    fireworks_key = os.getenv("FIREWORKS_API_KEY", "").strip()
    if fireworks_key:
        providers.append({
            "name": "fireworks",
            "url": "https://api.fireworks.ai/inference/v1/chat/completions",
            "api_key": fireworks_key,
            "model": os.getenv("GITHUB_SEARCH_FIREWORKS_MODEL")
            or os.getenv("FIREWORKS_DEBUG_MODEL", "accounts/fireworks/models/gpt-oss-120b"),
        })
    return providers


def _parse_llm_phrase_list(content: str | None) -> list[str]:
    if not content:
        return []
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    values: list[Any]
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            parsed = parsed.get("queries") or parsed.get("phrases") or parsed.get("search_terms") or []
        values = parsed if isinstance(parsed, list) else []
    except Exception:
        values = [line.strip(" -0123456789.\t") for line in text.splitlines() if line.strip()]

    phrases: list[str] = []
    for value in values:
        if isinstance(value, str):
            sanitized = _sanitize_search_phrase(value)
            if sanitized:
                phrases.append(sanitized)
    return _dedupe_search_phrases(phrases, limit=6)


def _heuristic_search_phrases(prompt: str, tech_stack: str = "") -> list[str]:
    """Small deterministic backstop for common app categories when LLM is unavailable."""
    text = f"{prompt} {tech_stack}".lower()
    phrases: list[str] = []
    if "car" in text or "vehicle" in text or "automotive" in text:
        if "market" in text or "sell" in text or "dealer" in text or "commerce" in text:
            phrases.extend([
                "car marketplace",
                "vehicle marketplace",
                "used car dealer platform",
                "automotive ecommerce",
            ])
    if "chat" in text or "message" in text or "whatsapp" in text:
        phrases.extend(["messaging app", "realtime chat app", "socket.io chat"])
    if "attendance" in text or "attandance" in text:
        phrases.extend(["attendance management system", "employee attendance system"])
        if "face" in text or "facial" in text or "recognition" in text or "biometric" in text:
            phrases.extend([
                "facial recognition attendance",
                "face attendance system",
                "biometric attendance system",
            ])
    if "music" in text or "spotify" in text:
        phrases.extend(["music streaming app", "spotify clone", "audio player app"])
    if "shop" in text or "ecommerce" in text or "amazon" in text:
        phrases.extend(["ecommerce store", "shopping cart app", "online marketplace"])
    if "ai" in text or "machine learning" in text:
        phrases.extend([f"ai {phrase}" for phrase in phrases[:3]])
    return phrases


def _sanitize_search_phrase(phrase: str) -> str:
    tokens = re.findall(r"[a-z0-9.#+@-]+", phrase.lower())
    cleaned = [token for token in tokens if len(token) >= 2][:8]
    return " ".join(cleaned)


def _dedupe_search_phrases(phrases: list[str], *, limit: int) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for phrase in phrases:
        sanitized = _sanitize_search_phrase(phrase)
        normalized = re.sub(r"\s+", " ", sanitized).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
        if len(result) >= limit:
            break
    return result


def _extract_search_keywords(prompt: str, tech_stack: str) -> list[str]:
    """Pull meaningful keywords from the prompt + tech stack for the GitHub query."""
    combined = (prompt + " " + tech_stack).lower()
    # Remove stopwords and short tokens
    stopwords = {
        "a", "an", "the", "with", "using", "and", "or", "for", "to", "in",
        "of", "on", "at", "is", "be", "build", "create", "make", "app",
        "application", "project", "site", "website", "web", "full", "stack",
        "simple", "basic", "complete", "modern",
    }
    tokens = re.findall(r"[a-z0-9.#+@-]+", combined)
    seen: set[str] = set()
    keywords: list[str] = []
    for t in tokens:
        if (len(t) >= 3 or t in {"ai", "ml", "ui", "ux"}) and t not in stopwords and t not in seen:
            seen.add(t)
            keywords.append(t)
    return keywords


def _detect_language(text: str) -> str:
    """Heuristically pick a GitHub language filter."""
    t = text.lower()
    if "react" in t or "next" in t or "vue" in t or "svelte" in t or "angular" in t:
        return "JavaScript"
    if "fastapi" in t or "flask" in t or "django" in t or "python" in t:
        return "Python"
    if "express" in t or "node" in t:
        return "JavaScript"
    if "rust" in t:
        return "Rust"
    if "go" in t or "golang" in t:
        return "Go"
    return ""


def _detect_framework_from_disk(root: Path) -> str:
    """Read config files to guess the framework."""
    pkg_json = root / "package.json"
    if pkg_json.exists():
        try:
            data = json.loads(pkg_json.read_text(encoding="utf-8"))
            deps = {**(data.get("dependencies") or {}), **(data.get("devDependencies") or {})}
            if "next" in deps:
                return "nextjs"
            if "vite" in deps:
                return "react-vite"
            if "react" in deps:
                return "react"
            if "express" in deps:
                return "express"
            if "vue" in deps:
                return "vue"
        except Exception:
            pass
    if (root / "requirements.txt").exists() or (root / "pyproject.toml").exists():
        for f in ["main.py", "app.py", "server.py"]:
            fpath = root / f
            if fpath.exists():
                content = fpath.read_text(encoding="utf-8", errors="ignore").lower()
                if "fastapi" in content:
                    return "fastapi"
                if "flask" in content:
                    return "flask"
                if "django" in content:
                    return "django"
        return "python"
    return "unknown"


def _iter_source_files(root: Path):
    """Yield Path objects for all readable source files, skipping unwanted dirs/exts."""
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune unwanted directories in-place
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS and not d.startswith(".")
        ]
        for fname in filenames:
            fpath = Path(dirpath) / fname
            if any(fname.endswith(ext) for ext in _SKIP_EXTS):
                continue
            yield fpath
