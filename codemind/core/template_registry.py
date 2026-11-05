from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, FrozenSet

from codemind.core.schemas import Plan

@dataclass(frozen=True)
class VerifierConfig:
    forbidden_imports: tuple[str, ...] = field(default_factory=tuple)
    forbidden_patterns: tuple[str, ...] = field(default_factory=tuple)
    required_structure: tuple[str, ...] = field(default_factory=tuple)
    static_analysis_enabled: bool = True
    build_verification_enabled: bool = True
    health_check_enabled: bool = True

@dataclass(frozen=True)
class TemplateConfig:
    template_id: str
    display_name: str
    category: str  # e.g., 'webapp', 'desktop', 'api', 'dashboard'
    aliases: frozenset[str]
    runtime_type: str  # 'node' or 'python'
    package_manager: str  # 'npm' or 'pip'
    has_frontend: bool
    has_backend: bool
    database_type: Optional[str]  # 'sqlite', 'supabase', or None
    entry_files: tuple[str, ...]
    required_files: tuple[str, ...]
    forbidden_files: tuple[str, ...]
    install_command: str
    build_command: Optional[str]
    run_command: str
    health_check_port: Optional[int]
    architecture_rules: tuple[str, ...]
    coder_rules: tuple[str, ...]
    verifier_rules: VerifierConfig
    env_var_rules: tuple[str, ...]
    fallback_keywords: tuple[str, ...]
    coder_model_tier: str  # 'frontend', 'backend', or 'reasoning'
    
    @property
    def template_path(self) -> str:
        """Returns the relative path to the template directory based on category and id."""
        return f"{self.category}/{self.template_id}"

TEMPLATE_REGISTRY: Dict[str, TemplateConfig] = {
    "react-vite": TemplateConfig(
        template_id="react-vite",
        display_name="React Vite SPA",
        category="webapp",
        aliases=frozenset(["react", "vite", "fullstack"]),
        runtime_type="node",
        package_manager="npm",
        has_frontend=True,
        has_backend=False,
        database_type=None,
        entry_files=("src/main.jsx", "src/App.jsx"),
        required_files=("package.json", "vite.config.js", "index.html", "src/main.jsx"),
        forbidden_files=("next.config.js", "next.config.mjs"),
        install_command="npm install",
        build_command="npm run build",
        run_command="npm run dev",
        health_check_port=5173, # Or 3000 depending on Vite config/script
        architecture_rules=(
            "Single Page Application using React and Vite.",
            "Use React Router for client-side routing.",
            "Components should be modular and reusable.",
        ),
        coder_rules=(
            "Use functional components and React hooks.",
            "Files containing JSX must use .jsx extension.",
            "index.html must remain at the project root.",
            "Use absolute imports or relative imports carefully, prefer relative for local components.",
        ),
        verifier_rules=VerifierConfig(
            forbidden_imports=("next", "vue", "svelte", "express", "flask", "fastapi", "django"),
            forbidden_patterns=("from flask", "import flask", "from django"),
            required_structure=("src/", "package.json"),
        ),
        env_var_rules=("VITE_SUPABASE_URL", "VITE_SUPABASE_ANON_KEY", "VITE_CLERK_PUBLISHABLE_KEY"),
        fallback_keywords=("react", "vite", "vue"),
        coder_model_tier="frontend",
    ),
    "nextjs-app": TemplateConfig(
        template_id="nextjs-app",
        display_name="Next.js App Router",
        category="webapp",
        aliases=frozenset(["nextjs", "next"]),
        runtime_type="node",
        package_manager="npm",
        has_frontend=True,
        has_backend=True,
        database_type=None, # Often Supabase via env
        entry_files=("app/layout.tsx", "app/page.tsx"),
        required_files=("package.json", "next.config.mjs", "app/layout.tsx", "app/page.tsx"),
        forbidden_files=("vite.config.js", "src/main.jsx"),
        install_command="npm install",
        build_command="npm run build",
        run_command="npm run dev",
        health_check_port=3000,
        architecture_rules=(
            "Use Next.js App Router (app/ directory).",
            "Server Components by default, Client Components only when needed.",
            "API routes in app/api/.",
        ),
        coder_rules=(
            "Use TypeScript.",
            "Server components by default.",
            "'use client' only when necessary.",
        ),
        verifier_rules=VerifierConfig(
            forbidden_imports=("vue", "svelte", "flask", "fastapi", "django", "createApp"),
            forbidden_patterns=("from vue", "import vue", "createApp"),
            required_structure=("app/", "package.json", "next.config.mjs"),
        ),
        env_var_rules=("NEXT_PUBLIC_SUPABASE_URL", "NEXT_PUBLIC_SUPABASE_ANON_KEY", "NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY"),
        fallback_keywords=("next",),
        coder_model_tier="frontend",
    ),
    "fastapi": TemplateConfig(
        template_id="fastapi",
        display_name="FastAPI REST API",
        category="api",
        aliases=frozenset([]),
        runtime_type="python",
        package_manager="pip",
        has_frontend=False,
        has_backend=True,
        database_type=None,
        entry_files=("main.py",),
        required_files=("main.py", "requirements.txt"),
        forbidden_files=("package.json", "next.config.mjs", "vite.config.js"),
        install_command="pip install -r requirements.txt",
        build_command=None,
        run_command="uvicorn main:app --host 0.0.0.0 --port 5000",
        health_check_port=5000,
        architecture_rules=(
            "Layered architecture with routers, services, models, schemas.",
            "Use APIRouter for route grouping.",
            "Pydantic models for request/response validation.",
        ),
        coder_rules=(
            "Use type hints everywhere.",
            "Use async def for all route handlers.",
            "Use Pydantic BaseModel for all schemas.",
        ),
        verifier_rules=VerifierConfig(
            forbidden_imports=("flask", "django", "streamlit", "tkinter"),
            forbidden_patterns=("from flask", "import flask", "from django"),
            required_structure=("main.py",),
        ),
        env_var_rules=("DATABASE_URL",),
        fallback_keywords=("fastapi",),
        coder_model_tier="backend",
    ),
    "fastapi-sqlite": TemplateConfig(
        template_id="fastapi-sqlite",
        display_name="FastAPI with SQLite",
        category="api",
        aliases=frozenset(["fastapi_sqlite", "fastapi-sql"]),
        runtime_type="python",
        package_manager="pip",
        has_frontend=False,
        has_backend=True,
        database_type="sqlite",
        entry_files=("main.py", "database.py"),
        required_files=("main.py", "database.py", "requirements.txt"),
        forbidden_files=("package.json",),
        install_command="pip install -r requirements.txt",
        build_command=None,
        run_command="uvicorn main:app --host 0.0.0.0 --port 5000",
        health_check_port=5000,
        architecture_rules=(
            "Layered architecture with routers, services, models, schemas.",
            "Use APIRouter for route grouping.",
            "Pydantic models for request/response validation.",
            "Local SQLite database configuration via SQLAlchemy.",
        ),
        coder_rules=(
            "Use type hints everywhere.",
            "Use async def for all route handlers.",
            "Use Pydantic BaseModel for all schemas.",
            "Configure SQLAlchemy for SQLite with check_same_thread=False.",
        ),
        verifier_rules=VerifierConfig(
            forbidden_imports=("flask", "django", "streamlit", "tkinter"),
            forbidden_patterns=("from flask", "import flask", "from django"),
            required_structure=("main.py", "database.py"),
        ),
        env_var_rules=("DATABASE_URL",),
        fallback_keywords=("fastapi", "sqlite"),
        coder_model_tier="backend",
    ),
    "python-tkinter-sqlite": TemplateConfig(
        template_id="python-tkinter-sqlite",
        display_name="Python Tkinter Desktop App",
        category="desktop",
        aliases=frozenset(["tkinter"]),
        runtime_type="python",
        package_manager="pip",
        has_frontend=True,
        has_backend=True,
        database_type="sqlite",
        entry_files=("main.py", "database.py"),
        required_files=("main.py", "requirements.txt"),
        forbidden_files=("package.json", "index.html"),
        install_command="pip install -r requirements.txt",
        build_command=None,
        run_command="python main.py",
        health_check_port=None,
        architecture_rules=(
            "MVC pattern with frames for UI components.",
            "Local SQLite database configuration.",
            "Offline only, no browser access.",
        ),
        coder_rules=(
            "Use ttk widgets over basic tk widgets for better native look.",
            "Use either pack or grid geometry manager consistently.",
            "Keep long-running tasks off the main GUI thread.",
        ),
        verifier_rules=VerifierConfig(
            forbidden_imports=("electron", "react", "fastapi", "flask", "django"),
            forbidden_patterns=("import electron", "require('electron')", "from flask", "from fastapi"),
            required_structure=("main.py",),
            health_check_enabled=False,
        ),
        env_var_rules=(),
        fallback_keywords=("tkinter", "desktop"),
        coder_model_tier="backend",
    ),
    "streamlit-app": TemplateConfig(
        template_id="streamlit-app",
        display_name="Streamlit Dashboard",
        category="dashboard",
        aliases=frozenset(["streamlit"]),
        runtime_type="python",
        package_manager="pip",
        has_frontend=True,
        has_backend=True,
        database_type=None,
        entry_files=("app.py",),
        required_files=("app.py", "requirements.txt"),
        forbidden_files=("package.json",),
        install_command="pip install -r requirements.txt",
        build_command=None,
        run_command="streamlit run app.py --server.headless true --server.port 5000",
        health_check_port=5000,
        architecture_rules=(
            "Single-page application pattern with Streamlit.",
            "Organize with st.sidebar for navigation if needed.",
            "Use st.session_state for state management.",
        ),
        coder_rules=(
            "Use st.cache_data decorator for data caching.",
            "Organize layout with columns and containers.",
        ),
        verifier_rules=VerifierConfig(
            forbidden_imports=("flask", "fastapi", "django", "tkinter"),
            forbidden_patterns=("from flask", "from fastapi", "from django"),
            required_structure=("app.py",),
        ),
        env_var_rules=(),
        fallback_keywords=("streamlit",),
        coder_model_tier="backend",
    ),
    "electron-react-app": TemplateConfig(
        template_id="electron-react-app",
        display_name="Electron React Desktop App",
        category="desktop",
        aliases=frozenset(["electron"]),
        runtime_type="node",
        package_manager="npm",
        has_frontend=True,
        has_backend=False,
        database_type=None,
        entry_files=("electron/main.js", "src/main.jsx"),
        required_files=("package.json", "electron/main.js", "src/main.jsx", "index.html"),
        forbidden_files=("next.config.mjs", "app.py"),
        install_command="npm install",
        build_command="npm run build",
        run_command="npm run dev",
        health_check_port=None, # Electron apps are desktop apps
        architecture_rules=(
            "Main/Renderer process separation.",
            "IPC communication between processes.",
            "React for the renderer process UI.",
        ),
        coder_rules=(
            "Use contextBridge for security in preload scripts.",
            "Keep Node.js APIs out of the renderer process.",
            "Use functional components in React.",
        ),
        verifier_rules=VerifierConfig(
            forbidden_imports=("tkinter", "pyqt"),
            forbidden_patterns=("from tkinter", "import tkinter"),
            required_structure=("electron/", "src/", "package.json"),
            health_check_enabled=False,
        ),
        env_var_rules=(),
        fallback_keywords=("electron",),
        coder_model_tier="frontend",
    ),
}

def resolve_template(project_type: str) -> Optional[TemplateConfig]:
    """Resolves a template config from an exact ID or known alias."""
    if not project_type:
        return None
    ptype = project_type.lower().strip()
    
    if ptype in TEMPLATE_REGISTRY:
        return TEMPLATE_REGISTRY[ptype]
        
    for template_id, config in TEMPLATE_REGISTRY.items():
        if ptype in config.aliases:
            return config
            
    return None

def resolve_template_from_plan(plan: Plan) -> Optional[TemplateConfig]:
    """Fallback resolution based on plan content keywords."""
    if not plan:
        return None
        
    techstack_lower = (getattr(plan, "techstack", "") or "").lower()
    plan_text = " ".join(filter(None, [
        (getattr(plan, "name", "") or "").lower(),
        (getattr(plan, "description", "") or "").lower(),
        " ".join(str(f) for f in (getattr(plan, "features", []) or [])).lower(),
        techstack_lower,
    ]))
    
    # Check electron first as it often includes react/web keywords
    if "electron" in plan_text:
        return TEMPLATE_REGISTRY["electron-react-app"]
    if "streamlit" in plan_text:
        return TEMPLATE_REGISTRY["streamlit-app"]
    if "tkinter" in plan_text or ("desktop" in plan_text and "electron" not in plan_text):
        return TEMPLATE_REGISTRY["python-tkinter-sqlite"]
    if "fastapi" in plan_text and "sqlite" in plan_text:
        return TEMPLATE_REGISTRY["fastapi-sqlite"]
    if "next" in techstack_lower:
        return TEMPLATE_REGISTRY["nextjs-app"]
    if any(fw in techstack_lower for fw in ("react", "vite", "vue")):
        return TEMPLATE_REGISTRY["react-vite"]
    if "fastapi" in techstack_lower:
        return TEMPLATE_REGISTRY["fastapi"]
        
    return None

def get_all_template_ids() -> List[str]:
    """Returns a list of all canonical template IDs."""
    return list(TEMPLATE_REGISTRY.keys())

def is_frontend_template(template_id: str) -> bool:
    """Checks if the template ID belongs to a frontend-heavy project."""
    template = resolve_template(template_id)
    if template:
        return template.has_frontend
    # Fallback for unknown/legacy strings
    return template_id in {"react", "react-vite", "nextjs", "nextjs-app", "vue", "svelte", "fullstack", "electron-react-app"}

def is_backend_template(template_id: str) -> bool:
    """Checks if the template ID belongs to a backend-heavy project."""
    template = resolve_template(template_id)
    if template:
        # We classify purely backend or python-heavy as backend for coder routing
        return not template.has_frontend or template.runtime_type == "python"
    # Fallback
    return template_id in {"fastapi", "fastapi-sqlite", "streamlit-app", "python-tkinter-sqlite"}

def get_coder_model_tier(template_id: str, file_ext: str) -> str:
    """Determines the appropriate coder model tier based on template and file extension."""
    template = resolve_template(template_id)
    if template:
        return template.coder_model_tier
    return "reasoning"
