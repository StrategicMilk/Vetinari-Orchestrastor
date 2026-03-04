"""
Vetinari UI Planner Agent

LLM-powered UI/UX planning agent that generates real design specifications,
component maps, design tokens, wireframes, and accessibility guidelines based
on actual project requirements.
"""

import logging
from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    AgentType,
    VerificationResult,
)

logger = logging.getLogger(__name__)


class UIPlannerAgent(BaseAgent):
    """UI Planner agent — LLM-powered front-end design, UX flows, and UI scaffolding."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.UI_PLANNER, config)
        self._framework = self._config.get("framework", "react")

    def get_system_prompt(self) -> str:
        return f"""You are Vetinari's UI Planner — a senior front-end architect and UX designer.
Your role is to convert project requirements into complete, production-ready UI specifications.

Framework preference: {self._framework}

You MUST analyse the actual requirements provided and produce a TAILORED response.
Do NOT produce generic boilerplate — every design decision must reference the specific project.

Required output (JSON):
{{
  "ui_spec": {{
    "title": "...",
    "app_type": "...",
    "pages": [{{"name": "...", "route": "...", "description": "...", "components": []}}],
    "navigation": [],
    "state_management": "...",
    "api_integration": []
  }},
  "component_map": {{
    "layout": [],
    "shared": [],
    "feature_specific": {{}}
  }},
  "design_tokens": {{
    "colors": {{}},
    "typography": {{}},
    "spacing": {{}},
    "shadows": {{}},
    "breakpoints": {{}}
  }},
  "components": [
    {{
      "name": "...",
      "description": "...",
      "props": [],
      "state": [],
      "events": [],
      "html_scaffold": "...",
      "css_classes": []
    }}
  ],
  "css_scaffold": "/* Complete CSS with design system */",
  "wireframes": [{{"page": "...", "layout": "...", "sections": [], "interactions": []}}],
  "ux_flows": [{{"name": "...", "trigger": "...", "steps": [], "success_state": "...", "error_state": "..."}}],
  "accessibility": {{
    "wcag_level": "AA",
    "aria_patterns": [],
    "keyboard_nav": [],
    "guidelines": []
  }},
  "responsive_strategy": "...",
  "performance_considerations": [],
  "summary": "..."
}}

Base design decisions on modern best practices (2025/2026). Consider:
- Responsive design (mobile-first)
- Dark/light mode support
- WCAG AA accessibility
- Performance (lazy loading, code splitting)
- Component reusability
"""

    def get_capabilities(self) -> List[str]:
        return [
            "ui_design",
            "wireframe_creation",
            "component_mapping",
            "design_token_definition",
            "ux_flow_design",
            "accessibility_planning",
        ]

    def execute(self, task: AgentTask) -> AgentResult:
        """Execute the UI planning task using LLM inference."""
        if not self.validate_task(task):
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Invalid task for {self._agent_type.value}"],
            )

        task = self.prepare_task(task)

        try:
            plan_or_requirements = task.context.get("plan", task.description)
            artifacts = task.context.get("artifacts", [])

            # Search for current UI best practices
            best_practices = ""
            try:
                search_results = self._search(
                    f"{self._framework} UI design system best practices 2025 accessibility"
                )
                if search_results:
                    best_practices = "\n".join(
                        [r.get("snippet", "") for r in search_results[:3]]
                    )
            except Exception:
                pass

            context_parts = [f"Project requirements:\n{plan_or_requirements}"]
            if artifacts:
                context_parts.append(
                    f"Available artifacts:\n{', '.join(str(a) for a in artifacts[:5])}"
                )
            if best_practices:
                context_parts.append(
                    f"Current best practices reference:\n{best_practices[:800]}"
                )

            prompt = (
                "\n\n".join(context_parts)
                + f"\n\nDesign a complete {self._framework} UI specification for the above. "
                "Return a single JSON object matching the required output format exactly."
            )

            ui_plan = self._infer_json(
                prompt=prompt,
                fallback=self._fallback_ui_design(plan_or_requirements),
            )

            # Ensure all required keys exist
            ui_plan.setdefault(
                "ui_spec",
                {"title": "Application", "pages": [], "navigation": []},
            )
            ui_plan.setdefault("component_map", {"layout": [], "shared": []})
            ui_plan.setdefault("design_tokens", {"colors": {}, "typography": {}, "spacing": {}})
            ui_plan.setdefault("components", [])
            ui_plan.setdefault("css_scaffold", "")
            ui_plan.setdefault("wireframes", [])
            ui_plan.setdefault("ux_flows", [])
            ui_plan.setdefault("accessibility", {"wcag_level": "AA", "guidelines": []})
            ui_plan.setdefault("summary", "UI specification generated")

            result = AgentResult(
                success=True,
                output=ui_plan,
                metadata={
                    "framework": self._framework,
                    "components_count": len(ui_plan.get("components", [])),
                    "pages_count": len(ui_plan.get("ui_spec", {}).get("pages", [])),
                },
            )
            task = self.complete_task(task, result)
            return result

        except Exception as e:
            self._log("error", f"UI planning failed: {e}")
            return AgentResult(success=False, output=None, errors=[str(e)])

    def verify(self, output: Any) -> VerificationResult:
        issues = []
        score = 1.0

        if not isinstance(output, dict):
            issues.append({"type": "invalid_type", "message": "Output must be a dict"})
            return VerificationResult(passed=False, issues=issues, score=0.0)

        if not output.get("ui_spec"):
            issues.append({"type": "missing_spec", "message": "UI specification missing"})
            score -= 0.25
        if not output.get("component_map"):
            issues.append({"type": "missing_components", "message": "Component map missing"})
            score -= 0.2
        if not output.get("design_tokens"):
            issues.append({"type": "missing_tokens", "message": "Design tokens missing"})
            score -= 0.15
        if not output.get("components"):
            issues.append({"type": "missing_scaffolding", "message": "UI scaffolding missing"})
            score -= 0.2
        if not output.get("wireframes"):
            issues.append({"type": "missing_wireframes", "message": "Wireframes missing"})
            score -= 0.1

        passed = score >= 0.5
        return VerificationResult(passed=passed, issues=issues, score=max(0.0, score))

    def _fallback_ui_design(self, requirements: str) -> Dict[str, Any]:
        """Structured fallback when LLM inference fails."""
        app_name = requirements[:40].strip() if requirements else "Application"
        return {
            "ui_spec": {
                "title": app_name,
                "app_type": "web",
                "pages": [
                    {"name": "Dashboard", "route": "/", "description": "Main view"},
                    {"name": "Settings", "route": "/settings", "description": "Configuration"},
                ],
                "navigation": ["Dashboard", "Settings"],
                "state_management": "React Context / Zustand",
                "api_integration": [],
            },
            "component_map": {
                "layout": ["AppShell", "Header", "Sidebar", "Footer"],
                "shared": ["Button", "Input", "Card", "Modal", "Toast", "Spinner"],
                "feature_specific": {},
            },
            "design_tokens": {
                "colors": {
                    "primary": "#3b82f6",
                    "secondary": "#6366f1",
                    "success": "#10b981",
                    "danger": "#ef4444",
                    "warning": "#f59e0b",
                    "background": "#0f172a",
                    "surface": "#1e293b",
                    "text": "#f1f5f9",
                    "text_muted": "#94a3b8",
                },
                "typography": {
                    "font_family": "'Inter', system-ui, sans-serif",
                    "sizes": {"xs": "12px", "sm": "14px", "base": "16px", "lg": "20px", "xl": "24px", "2xl": "32px"},
                    "weights": {"normal": 400, "medium": 500, "semibold": 600, "bold": 700},
                },
                "spacing": {"1": "4px", "2": "8px", "3": "12px", "4": "16px", "6": "24px", "8": "32px", "12": "48px"},
                "shadows": {"sm": "0 1px 3px rgba(0,0,0,0.3)", "md": "0 4px 12px rgba(0,0,0,0.3)"},
                "breakpoints": {"sm": "640px", "md": "768px", "lg": "1024px", "xl": "1280px"},
            },
            "components": [
                {
                    "name": "Button",
                    "description": "Primary action button",
                    "props": ["label", "onClick", "variant", "disabled", "loading"],
                    "html_scaffold": '<button class="btn btn-primary" type="button">Label</button>',
                    "css_classes": ["btn", "btn-primary", "btn-secondary", "btn-danger"],
                },
                {
                    "name": "Card",
                    "description": "Content container",
                    "props": ["title", "children", "actions"],
                    "html_scaffold": '<div class="card"><div class="card-header"><h3></h3></div><div class="card-body"></div></div>',
                    "css_classes": ["card", "card-header", "card-body", "card-footer"],
                },
            ],
            "css_scaffold": ":root{--primary:#3b82f6;--bg:#0f172a;--surface:#1e293b;--text:#f1f5f9}*{box-sizing:border-box}body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text)}",
            "wireframes": [
                {"page": "Dashboard", "layout": "Header + Sidebar + Main grid", "sections": ["KPI cards", "Activity feed", "Quick actions"], "interactions": []},
            ],
            "ux_flows": [
                {"name": "Primary action", "trigger": "User clicks CTA", "steps": ["Validate input", "Call API", "Show result"], "success_state": "Success toast", "error_state": "Error message"},
            ],
            "accessibility": {
                "wcag_level": "AA",
                "aria_patterns": ["role='button'", "aria-label", "aria-live regions for status updates"],
                "keyboard_nav": ["Tab order", "Enter/Space for buttons", "Escape to close modals"],
                "guidelines": ["4.5:1 contrast ratio for normal text", "All images need alt text", "Focus indicators visible"],
            },
            "responsive_strategy": "Mobile-first with CSS Grid and Flexbox",
            "performance_considerations": ["Code splitting per route", "Lazy load below-fold content", "Optimise images"],
            "summary": f"UI specification generated for: {app_name}",
        }


_ui_planner_agent: Optional[UIPlannerAgent] = None


def get_ui_planner_agent(config: Optional[Dict[str, Any]] = None) -> UIPlannerAgent:
    global _ui_planner_agent
    if _ui_planner_agent is None:
        _ui_planner_agent = UIPlannerAgent(config)
    return _ui_planner_agent
