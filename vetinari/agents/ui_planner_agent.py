"""
Vetinari UI Planner Agent

The UI Planner agent is responsible for front-end design, UX flows,
and UI scaffolding.
"""

from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    AgentType,
    VerificationResult
)


class UIPlannerAgent(BaseAgent):
    """UI Planner agent - front-end design, UX flows, and UI scaffolding."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.UI_PLANNER, config)
        self._framework = self._config.get("framework", "react")
        
    def get_system_prompt(self) -> str:
        return """You are Vetinari's UI Planner. Convert backend plans into user-friendly UI scaffolds, 
wireframes, and a CSS system. Output a UI spec, component map, and sample HTML/CSS/JS scaffolding 
aligned with the design tokens.

You must:
1. Design user-friendly interfaces aligned with backend logic
2. Create component maps and wireframes
3. Define design tokens (colors, spacing, typography)
4. Generate HTML/CSS/JS scaffolding
5. Document UX flows and interactions
6. Provide accessibility guidelines

Output format must include ui_spec, component_map, design_tokens, components, and wireframes."""
    
    def get_capabilities(self) -> List[str]:
        return [
            "ui_design",
            "wireframe_creation",
            "component_mapping",
            "design_token_definition",
            "ux_flow_design",
            "accessibility_planning"
        ]
    
    def execute(self, task: AgentTask) -> AgentResult:
        """Execute the UI planning task.
        
        Args:
            task: The task containing the plan or requirements
            
        Returns:
            AgentResult containing the UI specification
        """
        if not self.validate_task(task):
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Invalid task for {self._agent_type.value}"]
            )
        
        task = self.prepare_task(task)
        
        try:
            plan_or_requirements = task.context.get("plan", task.description)
            
            # Generate UI plan (simulated - in production would use actual design generation)
            ui_plan = self._design_ui(plan_or_requirements)
            
            return AgentResult(
                success=True,
                output=ui_plan,
                metadata={
                    "framework": self._framework,
                    "components_count": len(ui_plan.get("components", [])),
                    "pages_count": len(ui_plan.get("ui_spec", {}).get("pages", []))
                }
            )
            
        except Exception as e:
            self._log("error", f"UI planning failed: {str(e)}")
            return AgentResult(
                success=False,
                output=None,
                errors=[str(e)]
            )
    
    def verify(self, output: Any) -> VerificationResult:
        """Verify the UI specification meets quality standards.
        
        Args:
            output: The UI specification to verify
            
        Returns:
            VerificationResult with pass/fail status
        """
        issues = []
        score = 1.0
        
        if not isinstance(output, dict):
            issues.append({"type": "invalid_type", "message": "Output must be a dict"})
            score -= 0.5
            return VerificationResult(passed=False, issues=issues, score=score)
        
        # Check for UI spec
        if not output.get("ui_spec"):
            issues.append({"type": "missing_spec", "message": "UI specification missing"})
            score -= 0.25
        
        # Check for component map
        if not output.get("component_map"):
            issues.append({"type": "missing_components", "message": "Component map missing"})
            score -= 0.2
        
        # Check for design tokens
        if not output.get("design_tokens"):
            issues.append({"type": "missing_tokens", "message": "Design tokens missing"})
            score -= 0.15
        
        # Check for scaffolding
        if not output.get("components"):
            issues.append({"type": "missing_scaffolding", "message": "UI scaffolding missing"})
            score -= 0.2
        
        # Check for wireframes
        if not output.get("wireframes"):
            issues.append({"type": "missing_wireframes", "message": "Wireframes missing"})
            score -= 0.1
        
        passed = score >= 0.5
        return VerificationResult(passed=passed, issues=issues, score=max(0, score))
    
    def _design_ui(self, plan_or_requirements: str) -> Dict[str, Any]:
        """Design UI from plan or requirements.
        
        Args:
            plan_or_requirements: The plan or requirements to design from
            
        Returns:
            Dictionary containing UI specification and scaffolding
        """
        # This is a simplified implementation
        # In production, this would use actual design generation
        
        design_tokens = {
            "colors": {
                "primary": "#007bff",
                "secondary": "#6c757d",
                "success": "#28a745",
                "danger": "#dc3545",
                "background": "#f8f9fa",
                "text": "#212529"
            },
            "typography": {
                "font_family": "Segoe UI, Tahoma, Geneva, Verdana, sans-serif",
                "sizes": {"small": "12px", "base": "16px", "large": "20px", "xl": "24px"},
                "weights": {"normal": 400, "semibold": 600, "bold": 700}
            },
            "spacing": {
                "xs": "4px",
                "sm": "8px",
                "md": "16px",
                "lg": "24px",
                "xl": "32px"
            }
        }
        
        components = [
            {
                "name": "Button",
                "description": "Reusable button component",
                "props": ["label", "onClick", "variant", "disabled"],
                "html": '<button class="btn btn-primary">Label</button>'
            },
            {
                "name": "Input",
                "description": "Text input field",
                "props": ["placeholder", "value", "onChange", "type"],
                "html": '<input type="text" placeholder="Enter text" class="form-control">'
            },
            {
                "name": "Card",
                "description": "Content container",
                "props": ["title", "children"],
                "html": '<div class="card"><div class="card-body">Content</div></div>'
            }
        ]
        
        css_scaffold = '''
/* Design System CSS */
:root {
  --primary: #007bff;
  --secondary: #6c757d;
  --success: #28a745;
  --danger: #dc3545;
  --background: #f8f9fa;
  --text: #212529;
  --spacing-sm: 8px;
  --spacing-md: 16px;
  --spacing-lg: 24px;
}

* {
  box-sizing: border-box;
}

body {
  font-family: Segoe UI, Tahoma, Geneva, Verdana, sans-serif;
  color: var(--text);
  background-color: var(--background);
}

.btn {
  padding: var(--spacing-sm) var(--spacing-md);
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
  transition: all 0.3s ease;
}

.btn-primary {
  background-color: var(--primary);
  color: white;
}

.btn-primary:hover {
  opacity: 0.9;
}

.card {
  border: 1px solid #ddd;
  border-radius: 4px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.card-body {
  padding: var(--spacing-md);
}

.form-control {
  width: 100%;
  padding: var(--spacing-sm);
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 16px;
}

.form-control:focus {
  outline: none;
  border-color: var(--primary);
  box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.25);
}
'''
        
        return {
            "ui_spec": {
                "title": "UI Specification",
                "pages": [
                    {"name": "Dashboard", "description": "Main dashboard"},
                    {"name": "Settings", "description": "User settings page"},
                    {"name": "Profile", "description": "User profile page"}
                ],
                "navigation": ["Dashboard", "Settings", "Profile", "Logout"]
            },
            "component_map": {
                "layout": ["Header", "Sidebar", "Main", "Footer"],
                "common": ["Button", "Input", "Card", "Modal", "Dropdown"],
                "dashboard": ["StatCard", "Chart", "Table", "Timeline"]
            },
            "design_tokens": design_tokens,
            "components": components,
            "css_scaffold": css_scaffold,
            "wireframes": [
                {"page": "Dashboard", "layout": "Grid with cards"},
                {"page": "Settings", "layout": "Form with sections"},
                {"page": "Profile", "layout": "Profile header with details"}
            ],
            "ux_flows": [
                {"flow": "Login", "steps": ["Enter credentials", "Validate", "Navigate to dashboard"]},
                {"flow": "Update profile", "steps": ["Navigate to profile", "Edit fields", "Save changes"]}
            ],
            "accessibility": {
                "wcag_level": "AA",
                "guidelines": [
                    "All images must have alt text",
                    "Color contrast ratio must be at least 4.5:1",
                    "Interactive elements must be keyboard accessible",
                    "Form labels must be associated with inputs"
                ]
            },
            "summary": "Generated UI specification with component library and design tokens"
        }


# Singleton instance
_ui_planner_agent: Optional[UIPlannerAgent] = None


def get_ui_planner_agent(config: Optional[Dict[str, Any]] = None) -> UIPlannerAgent:
    """Get the singleton UI Planner agent instance."""
    global _ui_planner_agent
    if _ui_planner_agent is None:
        _ui_planner_agent = UIPlannerAgent(config)
    return _ui_planner_agent
