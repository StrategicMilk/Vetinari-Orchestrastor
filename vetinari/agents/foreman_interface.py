"""Foreman agent interface contract — planning and goal decomposition.

Imported by vetinari.agents.interfaces — do not use directly.
"""

from __future__ import annotations

from vetinari.agents.interface_types import AgentInterface, Capability, CapabilityType
from vetinari.types import AgentType

# ===== FOREMAN INTERFACE =====
FOREMAN_INTERFACE = AgentInterface(
    agent_name="Foreman",
    agent_type=AgentType.FOREMAN.value,
    version="1.0.0",
    capabilities=[
        Capability(
            name="goal_decomposition",
            type=CapabilityType.ANALYSIS,
            description="Decompose a high-level goal into an ordered set of tasks",
            input_schema={
                "type": "object",
                "properties": {
                    "goal": {"type": "string"},
                    "context": {"type": "object"},
                },
                "required": ["goal"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "tasks": {"type": "array"},
                    "plan_id": {"type": "string"},
                },
            },
        ),
        Capability(
            name="task_sequencing",
            type=CapabilityType.ANALYSIS,
            description="Sequence tasks with dependency resolution",
            input_schema={
                "type": "object",
                "properties": {
                    "tasks": {"type": "array"},
                },
                "required": ["tasks"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "ordered_tasks": {"type": "array"},
                    "dependency_graph": {"type": "object"},
                },
            },
        ),
        Capability(
            name="user_clarification",
            type=CapabilityType.ANALYSIS,
            description="Ask the user clarifying questions before planning",
            input_schema={
                "type": "object",
                "properties": {
                    "ambiguity": {"type": "string"},
                },
                "required": ["ambiguity"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array"},
                },
            },
        ),
        Capability(
            name="plan_consolidation",
            type=CapabilityType.SYNTHESIS,
            description="Consolidate multiple plans or partial plans into one",
            input_schema={
                "type": "object",
                "properties": {
                    "plans": {"type": "array"},
                },
                "required": ["plans"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "consolidated_plan": {"type": "object"},
                },
            },
        ),
        Capability(
            name="context_management",
            type=CapabilityType.ANALYSIS,
            description="Manage and summarise accumulated context",
            input_schema={
                "type": "object",
                "properties": {
                    "context": {"type": "object"},
                    "max_tokens": {"type": "integer"},
                },
                "required": ["context"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "pruned_context": {"type": "object"},
                    "summary": {"type": "string"},
                },
            },
        ),
        Capability(
            name="dependency_resolution",
            type=CapabilityType.ANALYSIS,
            description="Detect and resolve circular or missing task dependencies",
            input_schema={
                "type": "object",
                "properties": {
                    "tasks": {"type": "array"},
                },
                "required": ["tasks"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "resolved_tasks": {"type": "array"},
                    "warnings": {"type": "array"},
                },
            },
        ),
    ],
    required_context=["goal", "available_agents"],
    error_codes={
        "AMBIGUOUS_GOAL": "Goal requires clarification before planning",
        "CIRCULAR_DEPENDENCY": "Task dependency graph contains a cycle",
        "NO_VIABLE_PLAN": "Could not produce a valid plan for the given goal",
    },
)
