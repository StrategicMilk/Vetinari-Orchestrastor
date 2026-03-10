"""Legacy redirect — use PlannerAgent directly."""
from vetinari.agents.planner_agent import PlannerAgent as CostPlannerAgent, get_planner_agent as get_cost_planner_agent  # noqa: F401

# Legacy flat pricing dict (cost per 1K tokens).  Kept for backward compat.
MODEL_PRICING = {
    # Local / free models
    "qwen2.5-coder-7b": 0.0,
    "qwen2.5-coder-14b": 0.0,
    "qwen2.5-72b": 0.0,
    "deepseek-coder-v2": 0.0,
    "codellama-34b": 0.0,
    "llama-3.1-8b": 0.0,
    "llama-3.1-70b": 0.0,
    "mistral-7b": 0.0,
    "phi-3-medium": 0.0,
    "gemma-2-27b": 0.0,
    # Commercial APIs
    "claude-opus-4": 0.075,
    "claude-sonnet-4": 0.015,
    "claude-haiku-3": 0.00125,
    "gpt-4o": 0.015,
    "gpt-4o-mini": 0.00075,
    "gemini-2.0-flash": 0.0,
    "gemini-1.5-pro": 0.00625,
    "command-r-plus": 0.015,
}

__all__ = ["CostPlannerAgent", "get_cost_planner_agent", "MODEL_PRICING"]
