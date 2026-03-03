import uuid
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class DodDorLevel(Enum):
    LIGHT = "Light"
    STANDARD = "Standard"
    HARD = "Hard"


@dataclass
class MicroTaskTemplate:
    template_id: str
    name: str
    description: str
    agent_type: str
    keywords: List[str]
    prompt_template: str
    dod_level: str
    dor_level: str
    estimated_effort: float
    max_depth: int


@dataclass
class DecompositionEvent:
    event_id: str
    plan_id: str
    task_id: str
    parent_task_id: str
    depth: int
    seeds_used: List[str]
    subtasks_created: int
    timestamp: str
    model_response: str


class DecompositionEngine:
    DEFAULT_MAX_DEPTH = 14
    MIN_MAX_DEPTH = 12
    MAX_MAX_DEPTH = 16

    SEED_MIX = {
        "oracle": 0.50,
        "researcher": 0.25,
        "explorer": 0.25
    }

    SEED_RATE = {
        "base": 2,
        "max": 4,
        "subtask_min": 20,
        "subtask_max": 40
    }

    def __init__(self):
        self._load_templates()
        self.decomposition_history: List[DecompositionEvent] = []

    def _load_templates(self):
        self.templates: List[MicroTaskTemplate] = [
            MicroTaskTemplate(
                template_id="tmpl_001",
                name="Analyze Requirements",
                description="Analyze and break down requirements into specific tasks",
                agent_type="oracle",
                keywords=["analyze", "requirements", "breakdown", "specification"],
                prompt_template="Analyze the following requirements and break them down into specific, actionable tasks: {input}",
                dod_level="Standard",
                dor_level="Standard",
                estimated_effort=1.5,
                max_depth=14
            ),
            MicroTaskTemplate(
                template_id="tmpl_002",
                name="Research Technologies",
                description="Research and evaluate technologies, libraries, or frameworks",
                agent_type="researcher",
                keywords=["research", "evaluate", "technology", "library", "framework", "compare"],
                prompt_template="Research and evaluate the following technologies, providing pros and cons: {input}",
                dod_level="Standard",
                dor_level="Hard",
                estimated_effort=2.0,
                max_depth=14
            ),
            MicroTaskTemplate(
                template_id="tmpl_003",
                name="Explore Solutions",
                description="Explore different solution approaches and patterns",
                agent_type="explorer",
                keywords=["explore", "solution", "approach", "pattern", "alternative"],
                prompt_template="Explore different solution approaches for: {input}",
                dod_level="Light",
                dor_level="Standard",
                estimated_effort=1.0,
                max_depth=14
            ),
            MicroTaskTemplate(
                template_id="tmpl_004",
                name="Implement Core Logic",
                description="Implement the core business logic or functionality",
                agent_type="builder",
                keywords=["implement", "core", "logic", "functionality", "build", "create"],
                prompt_template="Implement the core functionality for: {input}",
                dod_level="Standard",
                dor_level="Hard",
                estimated_effort=3.0,
                max_depth=14
            ),
            MicroTaskTemplate(
                template_id="tmpl_005",
                name="Design API",
                description="Design REST API endpoints and data contracts",
                agent_type="oracle",
                keywords=["api", "endpoint", "rest", "design", "contract", "interface"],
                prompt_template="Design API endpoints for: {input}",
                dod_level="Standard",
                dor_level="Standard",
                estimated_effort=1.5,
                max_depth=14
            ),
            MicroTaskTemplate(
                template_id="tmpl_006",
                name="Create Database Schema",
                description="Design and implement database schema and migrations",
                agent_type="builder",
                keywords=["database", "schema", "migration", "table", "model", "sql"],
                prompt_template="Create database schema for: {input}",
                dod_level="Standard",
                dor_level="Hard",
                estimated_effort=2.0,
                max_depth=14
            ),
            MicroTaskTemplate(
                template_id="tmpl_007",
                name="Build UI Components",
                description="Build user interface components",
                agent_type="builder",
                keywords=["ui", "interface", "component", "frontend", "visual", "render"],
                prompt_template="Build UI components for: {input}",
                dod_level="Standard",
                dor_level="Standard",
                estimated_effort=2.5,
                max_depth=14
            ),
            MicroTaskTemplate(
                template_id="tmpl_008",
                name="Write Tests",
                description="Write unit tests and integration tests",
                agent_type="evaluator",
                keywords=["test", "unit", "integration", "coverage", "spec", "verify"],
                prompt_template="Write tests for: {input}",
                dod_level="Hard",
                dor_level="Hard",
                estimated_effort=2.0,
                max_depth=14
            ),
            MicroTaskTemplate(
                template_id="tmpl_009",
                name="Refactor Code",
                description="Refactor and improve existing code",
                agent_type="builder",
                keywords=["refactor", "improve", "cleanup", "optimize", "restructure"],
                prompt_template="Refactor the following code: {input}",
                dod_level="Light",
                dor_level="Standard",
                estimated_effort=1.5,
                max_depth=14
            ),
            MicroTaskTemplate(
                template_id="tmpl_010",
                name="Document Code",
                description="Create documentation and code comments",
                agent_type="librarian",
                keywords=["document", "docs", "readme", "comment", "explain", "guide"],
                prompt_template="Create documentation for: {input}",
                dod_level="Light",
                dor_level="Light",
                estimated_effort=1.0,
                max_depth=14
            ),
            MicroTaskTemplate(
                template_id="tmpl_011",
                name="Setup Infrastructure",
                description="Setup CI/CD, containers, or deployment infrastructure",
                agent_type="explorer",
                keywords=["ci", "cd", "deploy", "docker", "kubernetes", "infrastructure", "setup"],
                prompt_template="Setup infrastructure for: {input}",
                dod_level="Standard",
                dor_level="Standard",
                estimated_effort=2.5,
                max_depth=14
            ),
            MicroTaskTemplate(
                template_id="tmpl_012",
                name="Security Review",
                description="Review code for security vulnerabilities",
                agent_type="evaluator",
                keywords=["security", "vulnerability", "audit", "review", "safe", "threat"],
                prompt_template="Perform security review for: {input}",
                dod_level="Hard",
                dor_level="Hard",
                estimated_effort=2.0,
                max_depth=14
            ),
            MicroTaskTemplate(
                template_id="tmpl_013",
                name="Performance Optimization",
                description="Optimize code for performance",
                agent_type="builder",
                keywords=["performance", "optimize", "speed", "efficient", "bottleneck"],
                prompt_template="Optimize performance for: {input}",
                dod_level="Standard",
                dor_level="Hard",
                estimated_effort=2.0,
                max_depth=14
            ),
            MicroTaskTemplate(
                template_id="tmpl_014",
                name="Error Handling",
                description="Implement error handling and logging",
                agent_type="builder",
                keywords=["error", "exception", "handling", "logging", "fallback", "retry"],
                prompt_template="Implement error handling for: {input}",
                dod_level="Standard",
                dor_level="Standard",
                estimated_effort=1.0,
                max_depth=14
            ),
            MicroTaskTemplate(
                template_id="tmpl_015",
                name="Data Processing",
                description="Process, transform, or analyze data",
                agent_type="researcher",
                keywords=["data", "process", "transform", "analyze", "etl", "pipeline"],
                prompt_template="Process data for: {input}",
                dod_level="Standard",
                dor_level="Standard",
                estimated_effort=2.0,
                max_depth=14
            ),
            MicroTaskTemplate(
                template_id="tmpl_016",
                name="Integration Work",
                description="Integrate with external services or APIs",
                agent_type="builder",
                keywords=["integrate", "external", "api", "service", "webhook", "connect"],
                prompt_template="Integrate with external services for: {input}",
                dod_level="Standard",
                dor_level="Hard",
                estimated_effort=2.5,
                max_depth=14
            ),
            MicroTaskTemplate(
                template_id="tmpl_017",
                name="Code Review",
                description="Review code for quality and best practices",
                agent_type="evaluator",
                keywords=["review", "quality", "best practice", "standard", "lint"],
                prompt_template="Review code for: {input}",
                dod_level="Standard",
                dor_level="Hard",
                estimated_effort=1.5,
                max_depth=14
            ),
            MicroTaskTemplate(
                template_id="tmpl_018",
                name="Configuration Setup",
                description="Setup configuration management and environment handling",
                agent_type="builder",
                keywords=["config", "environment", "setting", "env", "variable"],
                prompt_template="Setup configuration for: {input}",
                dod_level="Light",
                dor_level="Standard",
                estimated_effort=1.0,
                max_depth=14
            ),
            MicroTaskTemplate(
                template_id="tmpl_019",
                name="Validate Implementation",
                description="Validate implementation against requirements",
                agent_type="evaluator",
                keywords=["validate", "verify", "requirement", "acceptance", "criteria"],
                prompt_template="Validate implementation against: {input}",
                dod_level="Hard",
                dor_level="Hard",
                estimated_effort=1.5,
                max_depth=14
            ),
            MicroTaskTemplate(
                template_id="tmpl_020",
                name="Synthesize Results",
                description="Synthesize results from multiple subtasks",
                agent_type="synthesizer",
                keywords=["synthesize", "combine", "merge", "assemble", "final", "deliver"],
                prompt_template="Synthesize results from: {input}",
                dod_level="Standard",
                dor_level="Hard",
                estimated_effort=2.0,
                max_depth=14
            )
        ]

    def get_templates(self, keywords: List[str] = None, agent_type: str = None, dod_level: str = None) -> List[MicroTaskTemplate]:
        results = self.templates
        
        if keywords:
            results = [t for t in results if any(kw.lower() in t.keywords or any(k in kw for k in t.keywords) for kw in keywords)]
        
        if agent_type:
            results = [t for t in results if t.agent_type == agent_type]
        
        if dod_level:
            results = [t for t in results if t.dod_level == dod_level]
        
        return results

    def get_dod_criteria(self, level: str) -> List[str]:
        dod_criteria = {
            "Light": [
                "Basic functionality implemented",
                "Code compiles without errors",
                "Basic error handling present"
            ],
            "Standard": [
                "All functionality implemented as specified",
                "Code follows project conventions",
                "Basic error handling and logging",
                "Code is readable and maintainable",
                "Basic tests pass"
            ],
            "Hard": [
                "All functionality fully tested (unit + integration)",
                "Complete error handling and edge cases",
                "Performance meets requirements",
                "Security review passed",
                "Documentation complete",
                "Code review approved",
                "All acceptance criteria met"
            ]
        }
        return dod_criteria.get(level, dod_criteria["Standard"])

    def get_dor_criteria(self, level: str) -> List[str]:
        dor_criteria = {
            "Light": [
                "Task completed",
                "Output is functional"
            ],
            "Standard": [
                "Task completed with good quality",
                "Output meets basic requirements",
                "Minor issues documented"
            ],
            "Hard": [
                "Task completed with exceptional quality",
                "All edge cases handled",
                "Performance optimized",
                "Fully documented",
                "Ready for production",
                "Reviewed and approved"
            ]
        }
        return dor_criteria.get(level, dor_criteria["Standard"])

    def select_seeds(self, seed_rate: int = 2) -> List[str]:
        total_seeds = min(seed_rate, self.SEED_RATE["max"])
        seeds = []
        
        num_oracle = int(total_seeds * self.SEED_MIX["oracle"])
        num_researcher = int(total_seeds * self.SEED_MIX["researcher"])
        num_explorer = total_seeds - num_oracle - num_researcher
        
        seeds.extend(["oracle"] * num_oracle)
        seeds.extend(["researcher"] * num_researcher)
        seeds.extend(["explorer"] * num_explorer)
        
        random.shuffle(seeds)
        return seeds[:total_seeds]

    def decompose_task(self, task_prompt: str, parent_task_id: str, depth: int, max_depth: int, plan_id: str) -> List[Dict]:
        if depth >= max_depth:
            return []

        templates = self.get_templates(keywords=task_prompt.lower().split())
        
        if not templates:
            templates = random.sample(self.templates, min(3, len(self.templates)))

        seeds = self.select_seeds()
        
        subtasks = []
        for i, template in enumerate(templates[:self.SEED_RATE["subtask_max"]]):
            subtask_id = f"{parent_task_id}_st{i+1}"
            
            effective_dod = template.dod_level
            effective_dor = template.dor_level
            
            if depth > max_depth - 3:
                effective_dod = "Light"
                effective_dor = "Light"
            
            subtask = {
                "task_id": subtask_id,
                "description": template.name,
                "prompt": template.prompt_template.format(input=task_prompt),
                "agent_type": template.agent_type,
                "parent_id": parent_task_id,
                "depth": depth + 1,
                "max_depth": max_depth,
                "dependencies": [],
                "dod_level": effective_dod,
                "dor_level": effective_dor,
                "estimated_effort": template.estimated_effort,
                "priority": 5 - (depth * 0.3),
                "subtasks": []
            }
            subtasks.append(subtask)

        event = DecompositionEvent(
            event_id=f"devent_{uuid.uuid4().hex[:8]}",
            plan_id=plan_id,
            task_id=parent_task_id,
            parent_task_id=parent_task_id,
            depth=depth,
            seeds_used=seeds,
            subtasks_created=len(subtasks),
            timestamp=datetime.now().isoformat(),
            model_response=""
        )
        self.decomposition_history.append(event)

        return subtasks

    def get_decomposition_history(self, plan_id: str = None) -> List[DecompositionEvent]:
        if plan_id:
            return [e for e in self.decomposition_history if e.plan_id == plan_id]
        return self.decomposition_history

    def clear_history(self, plan_id: str = None):
        if plan_id:
            self.decomposition_history = [e for e in self.decomposition_history if e.plan_id != plan_id]
        else:
            self.decomposition_history = []


decomposition_engine = DecompositionEngine()
