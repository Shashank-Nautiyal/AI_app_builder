from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from codemind.core.schemas import Plan
from codemind.core.template_registry import TemplateConfig, resolve_template, resolve_template_from_plan

@dataclass(frozen=True)
class ProjectContext:
    """Immutable metadata carrier that flows through the agent pipeline."""
    template: TemplateConfig          # Resolved from registry
    project_name: str                 # From plan.name
    project_description: str          # From plan.description
    techstack: str                    # From plan.techstack
    features: tuple[str, ...]         # From plan.features (frozen)
    source: str                       # "planner" | "github_scout" | "fallback"

def build_project_context(plan: Plan | None, state: dict[str, Any]) -> ProjectContext | None:
    """Builds the ProjectContext from the current pipeline state."""
    if not plan:
        return None

    # Determine source
    repo_plan = state.get("repo_adaptation_plan")
    if repo_plan:
        source = "github_scout"
    else:
        source = "planner"

    # Attempt resolution
    project_type = getattr(plan, "project_type", "")
    template = resolve_template(project_type)
    
    if not template:
        # Fallback to keyword sniffing
        template = resolve_template_from_plan(plan)
        if template:
            source = "fallback"

    if not template:
        print(f"[ProjectContext] WARNING: could not resolve a template for project_type={project_type!r}")
        return None

    return ProjectContext(
        template=template,
        project_name=getattr(plan, "name", "") or "unknown",
        project_description=getattr(plan, "description", "") or "",
        techstack=getattr(plan, "techstack", "") or "",
        features=tuple(getattr(plan, "features", []) or []),
        source=source,
    )
