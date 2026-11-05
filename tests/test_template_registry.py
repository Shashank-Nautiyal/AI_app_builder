import pytest
from template_registry import (
    resolve_template,
    resolve_template_from_plan,
    is_frontend_template,
    is_backend_template,
    get_coder_model_tier,
)
from schemas import Plan, File

def test_resolve_template_known():
    template = resolve_template("react-vite")
    assert template is not None
    assert template.template_id == "react-vite"
    assert template.category == "webapp"
    assert template.install_command == "npm install"
    assert "npm run dev" in template.run_command

def test_resolve_template_unknown():
    template = resolve_template("nonexistent-template")
    assert template is None

def test_resolve_template_from_plan_direct():
    plan = Plan(
        project_type="nextjs-app",
        name="Test Project",
        description="Test project description",
        features=["A test feature"],
        techstack="nextjs",
        files=[File(path="test.js", purpose="test")]
    )
    template = resolve_template_from_plan(plan)
    assert template is not None
    assert template.template_id == "nextjs-app"

def test_resolve_template_from_plan_sniffing():
    plan = Plan(
        project_type="unknown",
        name="Test Project",
        description="A desktop electron app",
        features=["Some electron feature"],
        techstack="react",
        files=[File(path="main.js", purpose="test")]
    )
    template = resolve_template_from_plan(plan)
    assert template is not None
    assert template.template_id == "electron-react-app"

def test_is_frontend_template():
    assert is_frontend_template("react-vite") is True
    assert is_frontend_template("nextjs-app") is True
    assert is_frontend_template("fastapi") is False

def test_is_backend_template():
    assert is_backend_template("fastapi") is True
    assert is_backend_template("python-tkinter-sqlite") is True
    assert is_backend_template("react-vite") is False

def test_get_coder_model_tier():
    assert get_coder_model_tier("react-vite", ".js") == "frontend"
    assert get_coder_model_tier("fastapi", ".py") == "backend"
    assert get_coder_model_tier("unknown", ".txt") == "reasoning"
