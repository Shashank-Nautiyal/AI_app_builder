import os
import shutil
import urllib.request
import json
from pathlib import Path

DEPENDENCY_MAP = {
    # Frontend (React + Vite + Tailwind)
    "frontend": ["react", "react-dom", "vite", "@vitejs/plugin-react", "tailwindcss", "postcss", "autoprefixer"],
    "styling": ["tailwindcss", "postcss", "autoprefixer"],
    "icons": ["lucide-react"],
    "charts": ["recharts"],
    "routing": ["react-router-dom"],
    
    # Fullstack (Next.js)
    "fullstack": ["next", "react", "react-dom", "tailwindcss", "postcss", "autoprefixer"],
    
    # Backend (FastAPI)
    "backend": ["fastapi", "uvicorn"],
    "database": ["sqlalchemy", "databases"],
    "auth": ["@clerk/clerk-react"],
    "streamlit": ["streamlit"],
    "desktop": [],
    "gui": [],
    "sqlite": [],
    "electron": ["electron", "concurrently", "wait-on"],
}

TRUSTED_LIBS = {
    "react", "react-dom", "next", "vite", "axios", "framer-motion", "clsx", "tailwind-merge", 
    "lucide-react", "recharts", "react-router-dom", "@vitejs/plugin-react",
    "express", "cors", "dotenv", "@supabase/supabase-js", "@clerk/clerk-react",
    "@clerk/clerk-sdk-node", "@clerk/nextjs", "resend", "stripe", "face-api.js",
    "@neondatabase/serverless", "cloudinary",
    "tailwindcss", "postcss", "autoprefixer",
    "requests", "numpy", "pandas", "python-dotenv", "jinja2", "fastapi", "uvicorn",
    "sqlalchemy", "databases", "pydantic", "streamlit", "flask", "supabase",
    "electron", "concurrently", "wait-on"
}

def _looks_like_missing_env(value: str | None) -> bool:
    if not value:
        return True
    lowered = value.strip().lower()
    return any(token in lowered for token in ("your-", "your_", "placeholder", "example", "changeme", "change-me"))

def is_popular(dep_name: str, package_manager: str = "npm") -> bool:
    """Validate dependency by checking popularity / existence on PyPI or NPM."""
    try:
        if package_manager == "npm":
            url = f"https://registry.npmjs.org/{dep_name}"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    # Validating if it has been published
                    if "time" in data and "created" in data["time"]:
                        return True
        elif package_manager == "pip":
            url = f"https://pypi.org/pypi/{dep_name}/json"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    return True
    except Exception as e:
        print(f"[ProjectRunner] Validation failed for {dep_name}: {e}")
    return False

def _dependency_name(dep: str, package_manager: str = "npm") -> str:
    dep = dep.strip()
    if package_manager == "npm":
        if dep.startswith("@") and "/" in dep:
            scope, rest = dep.split("/", 1)
            package = rest.split("@", 1)[0]
            return f"{scope}/{package}".strip()
        return dep.split("@", 1)[0].strip()
    return dep.split("==", 1)[0].split(">=", 1)[0].split("<=", 1)[0].split("~=", 1)[0].split(">", 1)[0].split("<", 1)[0].strip()

def validate_dependency(dep: str, package_manager: str = "npm") -> bool:
    dep_name = _dependency_name(dep, package_manager)
    if not dep_name:
        return False
    
    if dep_name in TRUSTED_LIBS:
        return True
    
    # Block suspicious names
    suspicious_keywords = ["malware", "test", "hacked", "exploit", "virus", "trojan"]
    if any(k in dep_name.lower() for k in suspicious_keywords):
        return False
        
    return is_popular(dep_name, package_manager)

def copy_template(template_name: str, target_path: str | Path):
    target = Path(target_path).resolve()
    from template_registry import resolve_template
    template = resolve_template(template_name)
    if not template:
        print(f"[ProjectRunner] Warning: Template {template_name} does not exist in registry.")
        return False
        
    template_dir = Path(__file__).parent / "templates" / template.template_path
    
    if not template_dir.exists():
        print(f"[ProjectRunner] Warning: Template dir {template_dir} does not exist. Generating raw.")
        return False
        
    if target.exists():
        shutil.copytree(template_dir, target, dirs_exist_ok=True)
    else:
        shutil.copytree(template_dir, target)
    return True

class ProjectRunner:
    def __init__(self, project_dir: str | Path):
        self.project_dir = Path(project_dir).resolve()
        
    def detect_type(self) -> str:
        package_json = self.project_dir / "package.json"
        if package_json.exists():
            try:
                package_data = json.loads(package_json.read_text(encoding="utf-8"))
                deps = {
                    *(package_data.get("dependencies") or {}).keys(),
                    *(package_data.get("devDependencies") or {}).keys(),
                }
                if "electron" in deps or (self.project_dir / "electron" / "main.cjs").exists():
                    return "electron-react-app"
            except Exception:
                pass
        if (self.project_dir / "next.config.js").exists() or (self.project_dir / "next.config.mjs").exists():
            return "nextjs-app"
        if (self.project_dir / "vite.config.js").exists() or (self.project_dir / "vite.config.ts").exists():
            return "react-vite"
        app_py = self.project_dir / "app.py"
        if app_py.exists():
            try:
                if "streamlit" in app_py.read_text(encoding="utf-8", errors="ignore").lower():
                    return "streamlit-app"
            except Exception:
                pass
        if (self.project_dir / "main.py").exists() and (self.project_dir / "database.py").exists():
            try:
                main_text = (self.project_dir / "main.py").read_text(encoding="utf-8", errors="ignore").lower()
                if "tkinter" in main_text:
                    return "python-tkinter-sqlite"
            except Exception:
                pass
        if (self.project_dir / "requirements.txt").exists() or (self.project_dir / "main.py").exists():
            return "fastapi"
        return "unknown"
        
    def resolve_dependencies(self, feature: str, package_manager: str = "npm") -> list[str]:
        if feature in DEPENDENCY_MAP:
            return DEPENDENCY_MAP[feature]
        return []

    def get_install_command(self) -> str | None:
        ptype = self.detect_type()
        from template_registry import resolve_template
        template = resolve_template(ptype)
        if template:
            return template.install_command
        return None

    def provision_backend(self, app_name: str):
        ptype = self.detect_type()
        from template_registry import resolve_template
        template = resolve_template(ptype)
        if not template:
            return
            
        # Clean app_name to be a safe schema name
        safe_name = "".join(c for c in app_name if c.isalnum() or c == "_").lower()
        if not safe_name:
            safe_name = "default_app"
            
        if template.database_type == "supabase" or "NEXT_PUBLIC_SUPABASE_URL" in template.env_var_rules or "VITE_SUPABASE_URL" in template.env_var_rules:
            # Option A: Supabase auto-provisioning
            sb_url = os.environ.get("SUPABASE_URL")
            sb_key = os.environ.get("SUPABASE_ANON_KEY")
            
            if _looks_like_missing_env(sb_url) or _looks_like_missing_env(sb_key):
                print(
                    f"[ProjectRunner] Supabase keys unavailable for '{safe_name}'. "
                    "Skipping runtime .env.local provisioning."
                )
                env_file = self.project_dir / ".env.example"
                if not env_file.exists():
                    env_file.write_text(
                        "NEXT_PUBLIC_SUPABASE_URL=https://your-supabase-project.supabase.co\n"
                        "NEXT_PUBLIC_SUPABASE_ANON_KEY=your-supabase-anon-key\n"
                        "VITE_SUPABASE_URL=https://your-supabase-project.supabase.co\n"
                        "VITE_SUPABASE_ANON_KEY=your-supabase-anon-key\n",
                        encoding="utf-8",
                    )
            else:
                print(f"[ProjectRunner] Provisioning schema '{safe_name}' in shared Supabase project...")
                # In a real environment with an admin DB connection string:
                # engine.execute(f"CREATE SCHEMA IF NOT EXISTS {safe_name};")
                
                env_content = ""
                if "NEXT_PUBLIC_SUPABASE_URL" in template.env_var_rules:
                    env_content += f"NEXT_PUBLIC_SUPABASE_URL={sb_url}\nNEXT_PUBLIC_SUPABASE_ANON_KEY={sb_key}\n"
                if "VITE_SUPABASE_URL" in template.env_var_rules:
                    env_content += f"VITE_SUPABASE_URL={sb_url}\nVITE_SUPABASE_ANON_KEY={sb_key}\n"
                
                env_file = self.project_dir / ".env.local"
                if not env_file.exists():
                    env_file.write_text(env_content, encoding="utf-8")
        elif template.database_type == "sqlite":
            # Option B: FastAPI/Tkinter + SQLite auto-provisioning
            print(f"[ProjectRunner] Provisioning local SQLite for '{safe_name}'...")
            env_content = f"DATABASE_URL=sqlite:///./app.db\n"
            env_file = self.project_dir / ".env"
            if not env_file.exists():
                env_file.write_text(env_content, encoding="utf-8")

    def get_run_command(self) -> str | None:
        import sys
        is_win = sys.platform == "win32"
        ptype = self.detect_type()
        
        from template_registry import resolve_template
        template = resolve_template(ptype)
        if not template:
            if is_win:
                return "start /b python app.py > server.log 2>&1"
            return "nohup python app.py > server.log 2>&1 & echo $!"
            
        cmd = template.run_command
        if is_win:
            return f"start /b {cmd} > server.log 2>&1"
        return f"nohup {cmd} > server.log 2>&1 & echo $!"

