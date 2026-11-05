import pytest
from codemind.services.project_context import build_project_context
from codemind.core.schemas import Plan, File

def test_build_project_context_with_known_template():
    plan = Plan(
        project_type="react-vite",
        name="My App",
        description="A cool app that does things",
        features=["feature 1"],
        techstack="React, Vite",
        files=[File(path="main.js", purpose="entry")]
    )
    
    ctx = build_project_context(plan, {})
    assert ctx is not None
    assert ctx.template is not None
    assert ctx.template.template_id == "react-vite"
    assert ctx.project_name == "My App"
    assert ctx.techstack == "React, Vite"
    assert ctx.source == "planner"

def test_build_project_context_unknown_template():
    plan = Plan(
        project_type="unknown-weird-stack",
        name="My App",
        description="A cool app that does things",
        features=["feature 1"],
        techstack="Unknown",
        files=[File(path="main.js", purpose="entry")]
    )
    
    ctx = build_project_context(plan, {})
    assert ctx is None
