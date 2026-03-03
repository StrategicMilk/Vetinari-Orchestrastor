import os
import logging
import uuid
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime

from .plan_types import (
    Plan, PlanCandidate, Subtask, PlanStatus, SubtaskStatus,
    TaskDomain, PlanRiskLevel, DefinitionOfDone, DefinitionOfReady,
    TaskRationale, PlanGenerationRequest, PlanApprovalRequest
)
from .memory import get_memory_store, MemoryStore

logger = logging.getLogger(__name__)

PLAN_MODE_DEFAULT = os.environ.get("PLAN_MODE_DEFAULT", "true").lower() in ("1", "true", "yes")
PLAN_MODE_ENABLE = os.environ.get("PLAN_MODE_ENABLE", "true").lower() in ("1", "true", "yes")
DRY_RUN_ENABLED = os.environ.get("DRY_RUN_ENABLED", "false").lower() in ("1", "true", "yes")
DRY_RUN_RISK_THRESHOLD = float(os.environ.get("DRY_RUN_RISK_THRESHOLD", "0.25"))
DEPTH_CAP = int(os.environ.get("PLAN_DEPTH_CAP", "16"))
MAX_CANDIDATES = int(os.environ.get("PLAN_MAX_CANDIDATES", "3"))


class PlanModeEngine:
    """Plan Mode Engine - generates, evaluates, and approves plans.
    
    This engine implements the plan-first orchestration pattern:
    1. Generate plan candidates from goals
    2. Evaluate and rank candidates
    3. Allow dry-run mode with auto-approval for low-risk plans
    4. Support manual approval for high-risk plans
    5. Execute approved plans with subtask tracking
    """
    
    def __init__(self, memory_store: Optional[MemoryStore] = None):
        self.memory = memory_store or get_memory_store()
        self.plan_depth_cap = DEPTH_CAP
        self.max_candidates = MAX_CANDIDATES
        self.dry_run_risk_threshold = DRY_RUN_RISK_THRESHOLD
        
        self._domain_templates = self._load_domain_templates()
        self._agent_templates = self._load_agent_templates()
    
    def _load_domain_templates(self) -> Dict[TaskDomain, List[Dict]]:
        """Load domain-specific subtask templates."""
        return {
            TaskDomain.CODING: [
                {
                    "description": "Define API surface and data models",
                    "domain": TaskDomain.CODING,
                    "definition_of_done": DefinitionOfDone(
                        criteria=["API spec written", "Data models defined", "Interfaces documented"]
                    ),
                    "definition_of_ready": DefinitionOfReady(
                        prerequisites=["Requirements understood"]
                    )
                },
                {
                    "description": "Implement core functionality",
                    "domain": TaskDomain.CODING,
                    "definition_of_done": DefinitionOfDone(
                        criteria=["Core logic implemented", "Code compiles", "Basic tests pass"]
                    ),
                    "definition_of_ready": DefinitionOfReady(
                        prerequisites=["API surface defined"]
                    )
                },
                {
                    "description": "Write unit tests",
                    "domain": TaskDomain.CODING,
                    "definition_of_done": DefinitionOfDone(
                        criteria=["Unit tests written", "Coverage > 80%", "All tests pass"]
                    ),
                    "definition_of_ready": DefinitionOfReady(
                        prerequisites=["Core functionality implemented"]
                    )
                },
                {
                    "description": "Integrate with existing components",
                    "domain": TaskDomain.CODING,
                    "definition_of_done": DefinitionOfDone(
                        criteria=["Integration points wired", "Integration tests pass"]
                    ),
                    "definition_of_ready": DefinitionOfReady(
                        prerequisites=["Unit tests passing"]
                    )
                },
                {
                    "description": "Refactor for clarity and maintainability",
                    "domain": TaskDomain.CODING,
                    "definition_of_done": DefinitionOfDone(
                        criteria=["Code reviewed", "Linting passes", "No critical tech debt"]
                    ),
                    "definition_of_ready": DefinitionOfReady(
                        prerequisites=["Integration complete"]
                    )
                }
            ],
            TaskDomain.DATA_PROCESSING: [
                {
                    "description": "Define data schema and sources",
                    "domain": TaskDomain.DATA_PROCESSING,
                    "definition_of_done": DefinitionOfDone(
                        criteria=["Schema documented", "Source systems identified"]
                    ),
                    "definition_of_ready": DefinitionOfReady(prerequisites=[])
                },
                {
                    "description": "Build data ingestion pipeline",
                    "domain": TaskDomain.DATA_PROCESSING,
                    "definition_of_done": DefinitionOfDone(
                        criteria=["Ingestion working", "Data validated at source"]
                    ),
                    "definition_of_ready": DefinitionOfReady(
                        prerequisites=["Schema defined"]
                    )
                },
                {
                    "description": "Implement transformation logic",
                    "domain": TaskDomain.DATA_PROCESSING,
                    "definition_of_done": DefinitionOfDone(
                        criteria=["Transformations applied", "Output validated"]
                    ),
                    "definition_of_ready": DefinitionOfReady(
                        prerequisites=["Ingestion working"]
                    )
                },
                {
                    "description": "Implement data quality checks",
                    "domain": TaskDomain.DATA_PROCESSING,
                    "definition_of_done": DefinitionOfDone(
                        criteria=["Quality checks implemented", "Alerts configured"]
                    ),
                    "definition_of_ready": DefinitionOfReady(
                        prerequisites=["Transformations complete"]
                    )
                },
                {
                    "description": "Create deployment and scheduling",
                    "domain": TaskDomain.DATA_PROCESSING,
                    "definition_of_done": DefinitionOfDone(
                        criteria=["Pipeline scheduled", "Monitoring active"]
                    ),
                    "definition_of_ready": DefinitionOfReady(
                        prerequisites=["Quality checks passing"]
                    )
                }
            ],
            TaskDomain.INFRA: [
                {
                    "description": "Define metrics and observability requirements",
                    "domain": TaskDomain.INFRA,
                    "definition_of_done": DefinitionOfDone(
                        criteria=["Metrics catalog created", "SLOs defined"]
                    ),
                    "definition_of_ready": DefinitionOfReady(prerequisites=[])
                },
                {
                    "description": "Implement health checks and readiness probes",
                    "domain": TaskDomain.INFRA,
                    "definition_of_done": DefinitionOfDone(
                        criteria=["Health endpoints implemented", "Probes configured"]
                    ),
                    "definition_of_ready": DefinitionOfReady(
                        prerequisites=["Metrics defined"]
                    )
                },
                {
                    "description": "Create monitoring dashboards",
                    "domain": TaskDomain.INFRA,
                    "definition_of_done": DefinitionOfDone(
                        criteria=["Dashboards created", "Key metrics visible"]
                    ),
                    "definition_of_ready": DefinitionOfReady(
                        prerequisites=["Health checks working"]
                    )
                },
                {
                    "description": "Configure alerting rules",
                    "domain": TaskDomain.INFRA,
                    "definition_of_done": DefinitionOfDone(
                        criteria=["Alert rules configured", "On-call defined"]
                    ),
                    "definition_of_ready": DefinitionOfReady(
                        prerequisites=["Dashboards complete"]
                    )
                },
                {
                    "description": "Document runbooks",
                    "domain": TaskDomain.INFRA,
                    "definition_of_done": DefinitionOfDone(
                        criteria=["Runbooks written", "Team trained"]
                    ),
                    "definition_of_ready": DefinitionOfReady(
                        prerequisites=["Alert rules configured"]
                    )
                }
            ],
            TaskDomain.DOCS: [
                {
                    "description": "Outline documentation structure",
                    "domain": TaskDomain.DOCS,
                    "definition_of_done": DefinitionOfDone(
                        criteria=["TOC created", "Audience defined"]
                    ),
                    "definition_of_ready": DefinitionOfReady(prerequisites=[])
                },
                {
                    "description": "Draft main sections",
                    "domain": TaskDomain.DOCS,
                    "definition_of_done": DefinitionOfDone(
                        criteria=["Sections drafted", "Technical accuracy verified"]
                    ),
                    "definition_of_ready": DefinitionOfReady(
                        prerequisites=["Structure outlined"]
                    )
                },
                {
                    "description": "Add usage examples and code snippets",
                    "domain": TaskDomain.DOCS,
                    "definition_of_done": DefinitionOfDone(
                        criteria=["Examples added", "Tested for accuracy"]
                    ),
                    "definition_of_ready": DefinitionOfReady(
                        prerequisites=["Main sections drafted"]
                    )
                },
                {
                    "description": "Review and get feedback",
                    "domain": TaskDomain.DOCS,
                    "definition_of_done": DefinitionOfDone(
                        criteria=["Peer review complete", "Feedback addressed"]
                    ),
                    "definition_of_ready": DefinitionOfReady(
                        prerequisites=["Examples added"]
                    )
                },
                {
                    "description": "Finalize and publish",
                    "domain": TaskDomain.DOCS,
                    "definition_of_done": DefinitionOfDone(
                        criteria=["Published", "Indexed", "Searchable"]
                    ),
                    "definition_of_ready": DefinitionOfReady(
                        prerequisites=["Review complete"]
                    )
                }
            ],
            TaskDomain.AI_EXPERIMENTS: [
                {
                    "description": "Define experiment metrics and success criteria",
                    "domain": TaskDomain.AI_EXPERIMENTS,
                    "definition_of_done": DefinitionOfDone(
                        criteria=["Metrics defined", "Baseline established"]
                    ),
                    "definition_of_ready": DefinitionOfReady(prerequisites=[])
                },
                {
                    "description": "Design experiment configuration",
                    "domain": TaskDomain.AI_EXPERIMENTS,
                    "definition_of_done": DefinitionOfDone(
                        criteria=["Config documented", "Controls defined"]
                    ),
                    "definition_of_ready": DefinitionOfReady(
                        prerequisites=["Metrics defined"]
                    )
                },
                {
                    "description": "Run experiments",
                    "domain": TaskDomain.AI_EXPERIMENTS,
                    "definition_of_done": DefinitionOfDone(
                        criteria=["Experiments executed", "Data collected"]
                    ),
                    "definition_of_ready": DefinitionOfReady(
                        prerequisites=["Config ready"]
                    )
                },
                {
                    "description": "Analyze results",
                    "domain": TaskDomain.AI_EXPERIMENTS,
                    "definition_of_done": DefinitionOfDone(
                        criteria=["Results analyzed", "Statistical significance checked"]
                    ),
                    "definition_of_ready": DefinitionOfReady(
                        prerequisites=["Data collected"]
                    )
                },
                {
                    "description": "Document insights and recommendations",
                    "domain": TaskDomain.AI_EXPERIMENTS,
                    "definition_of_done": DefinitionOfDone(
                        criteria=["Insights documented", "Recommendations clear"]
                    ),
                    "definition_of_ready": DefinitionOfReady(
                        prerequisites=["Analysis complete"]
                    )
                }
            ],
            TaskDomain.RESEARCH: [
                {
                    "description": "Gather sources and literature",
                    "domain": TaskDomain.RESEARCH,
                    "definition_of_done": DefinitionOfDone(
                        criteria=["Sources collected", "Relevant papers identified"]
                    ),
                    "definition_of_ready": DefinitionOfReady(prerequisites=[])
                },
                {
                    "description": "Summarize findings",
                    "domain": TaskDomain.RESEARCH,
                    "definition_of_done": DefinitionOfDone(
                        criteria=["Summary written", "Key insights extracted"]
                    ),
                    "definition_of_ready": DefinitionOfReady(
                        prerequisites=["Sources gathered"]
                    )
                },
                {
                    "description": "Compare approaches",
                    "domain": TaskDomain.RESEARCH,
                    "definition_of_done": DefinitionOfDone(
                        criteria=["Comparison matrix created", "Tradeoffs identified"]
                    ),
                    "definition_of_ready": DefinitionOfReady(
                        prerequisites=["Summaries complete"]
                    )
                },
                {
                    "description": "Propose recommendations",
                    "domain": TaskDomain.RESEARCH,
                    "definition_of_done": DefinitionOfDone(
                        criteria=["Recommendations clear", "Action items defined"]
                    ),
                    "definition_of_ready": DefinitionOfReady(
                        prerequisites=["Comparison done"]
                    )
                },
                {
                    "description": "Validate against goals",
                    "domain": TaskDomain.RESEARCH,
                    "definition_of_done": DefinitionOfDone(
                        criteria=["Validation complete", "Final report ready"]
                    ),
                    "definition_of_ready": DefinitionOfReady(
                        prerequisites=["Recommendations proposed"]
                    )
                }
            ]
        }
    
    def _load_agent_templates(self) -> Dict[str, List[Dict]]:
        """Load agent-specific subtask templates."""
        return {
            "planner": [
                {"description": "Clarify objective and success criteria", "agent": "planner"},
                {"description": "Identify constraints and requirements", "agent": "planner"},
                {"description": "Draft initial plan skeleton", "agent": "planner"},
                {"description": "Enumerate dependencies and blockers", "agent": "planner"},
                {"description": "Propose alternative plan variants", "agent": "planner"},
                {"description": "Assess risks per plan variant", "agent": "planner"},
                {"description": "Provide final plan with justification", "agent": "planner"}
            ],
            "decomposer": [
                {"description": "Break down high-level features into modules", "agent": "decomposer"},
                {"description": "Decompose modules into interfaces and contracts", "agent": "decomposer"},
                {"description": "Split complex tasks by data flow", "agent": "decomposer"},
                {"description": "Identify edge cases and error paths", "agent": "decomposer"},
                {"description": "Group related subtasks for efficiency", "agent": "decomposer"}
            ],
            "breaker": [
                {"description": "Break API surface into individual endpoints", "agent": "breaker"},
                {"description": "Split data processing into ETL steps", "agent": "breaker"},
                {"description": "Decompose deployment into config steps", "agent": "breaker"},
                {"description": "Further break complex subtasks", "agent": "breaker"},
                {"description": "Ensure atomicity of each subtask", "agent": "breaker"}
            ],
            "assigner": [
                {"description": "Map tasks to local vs cloud models", "agent": "assigner"},
                {"description": "Select optimal model per task type", "agent": "assigner"},
                {"description": "Balance load across models", "agent": "assigner"},
                {"description": "Record assignment rationale", "agent": "assigner"},
                {"description": "Validate model capabilities match requirements", "agent": "assigner"}
            ],
            "executor": [
                {"description": "Install dependencies", "agent": "executor"},
                {"description": "Run core execution step", "agent": "executor"},
                {"description": "Collect and validate results", "agent": "executor"},
                {"description": "Handle errors and retries", "agent": "executor"},
                {"description": "Report execution status", "agent": "executor"}
            ],
            "explainer": [
                {"description": "Cite capability match for model selection", "agent": "explainer"},
                {"description": "Cite context fit rationale", "agent": "explainer"},
                {"description": "Document policy compliance notes", "agent": "explainer"},
                {"description": "Explain trade-offs considered", "agent": "explainer"},
                {"description": "Summarize decision justification", "agent": "explainer"}
            ],
            "memory": [
                {"description": "Log plan outcome to memory store", "agent": "memory"},
                {"description": "Record model performance metrics", "agent": "memory"},
                {"description": "Archive plan rationale for future reference", "agent": "memory"},
                {"description": "Update success rates based on outcome", "agent": "memory"},
                {"description": "Prune old plans per retention policy", "agent": "memory"}
            ]
        }
    
    def generate_plan(self, request: PlanGenerationRequest) -> Plan:
        """Generate a plan from a goal.
        
        This creates multiple plan candidates, evaluates them, and returns
        a Plan object ready for approval or execution.
        """
        logger.info(f"Generating plan for goal: {request.goal[:100]}...")
        
        plan = Plan(
            goal=request.goal,
            constraints=request.constraints,
            dry_run=request.dry_run,
            plan_candidates=[]
        )
        
        domain = request.domain_hint or self._infer_domain(request.goal)
        
        candidates = self._generate_candidates(
            goal=request.goal,
            constraints=request.constraints,
            domain=domain,
            max_candidates=request.max_candidates,
            depth_cap=request.plan_depth_cap
        )
        
        plan.plan_candidates = candidates
        
        if candidates:
            best_candidate = min(candidates, key=lambda c: c.risk_score)
            plan.chosen_plan_id = best_candidate.plan_id
            plan.plan_justification = best_candidate.justification
            plan.risk_score = best_candidate.risk_score
            plan.risk_level = best_candidate.risk_level
            plan.subtasks = self._create_subtasks_from_candidate(best_candidate, plan.plan_id)
            plan.dependencies = best_candidate.dependencies
        
        plan.calculate_risk_score()
        
        if request.dry_run:
            plan.status = PlanStatus.DRAFT
            if plan.risk_score <= self.dry_run_risk_threshold:
                plan.auto_approved = True
                plan.status = PlanStatus.APPROVED
                plan.approved_by = "system_auto"
                plan.approved_at = datetime.now().isoformat()
        else:
            plan.status = PlanStatus.DRAFT
        
        self._persist_plan(plan)
        
        logger.info(f"Plan generated: {plan.plan_id}, risk_score={plan.risk_score:.2f}, "
                   f"subtasks={len(plan.subtasks)}, auto_approved={plan.auto_approved}")
        
        return plan
    
    def _infer_domain(self, goal: str) -> TaskDomain:
        """Infer the domain from the goal text."""
        goal_lower = goal.lower()
        
        if any(kw in goal_lower for kw in ["code", "implement", "build", "feature", "api", "function"]):
            return TaskDomain.CODING
        elif any(kw in goal_lower for kw in ["etl", "data", "pipeline", "process", "transform"]):
            return TaskDomain.DATA_PROCESSING
        elif any(kw in goal_lower for kw in ["infra", "deploy", "monitor", "logging", "ci/cd"]):
            return TaskDomain.INFRA
        elif any(kw in goal_lower for kw in ["document", "docs", "write", "guide"]):
            return TaskDomain.DOCS
        elif any(kw in goal_lower for kw in ["experiment", "model", "test", "benchmark", "evaluate"]):
            return TaskDomain.AI_EXPERIMENTS
        elif any(kw in goal_lower for kw in ["research", "analyze", "study", "investigate"]):
            return TaskDomain.RESEARCH
        else:
            return TaskDomain.GENERAL
    
    def _generate_candidates(self, goal: str, constraints: str, domain: TaskDomain,
                            max_candidates: int, depth_cap: int) -> List[PlanCandidate]:
        """Generate multiple plan candidates."""
        candidates = []
        
        templates = self._domain_templates.get(domain, self._domain_templates[TaskDomain.GENERAL])
        
        for i in range(min(max_candidates, 3)):
            candidate = PlanCandidate(
                plan_id=f"plan_{uuid.uuid4().hex[:8]}",
                plan_version=1,
                summary=f"Plan variant {i+1} for: {goal[:50]}...",
                description=f"Implementation plan for: {goal}",
                justification=f"Generated based on {domain.value} domain patterns",
                risk_score=0.15 + (i * 0.1),
                estimated_duration_seconds=3600.0 * (1 + i * 0.5),
                estimated_cost=10.0 * (1 + i * 0.3),
                subtask_count=len(templates) + i * 2,
                max_depth=min(depth_cap, 3 + i),
                domains=[domain]
            )
            
            if candidate.risk_score >= 0.75:
                candidate.risk_level = PlanRiskLevel.CRITICAL
            elif candidate.risk_score >= 0.5:
                candidate.risk_level = PlanRiskLevel.HIGH
            elif candidate.risk_score >= 0.25:
                candidate.risk_level = PlanRiskLevel.MEDIUM
            else:
                candidate.risk_level = PlanRiskLevel.LOW
            
            candidate.dependencies = self._generate_dependencies(len(templates) + i * 2)
            
            candidates.append(candidate)
        
        return candidates
    
    def _generate_dependencies(self, subtask_count: int) -> Dict[str, List[str]]:
        """Generate task dependencies."""
        deps = {}
        for i in range(subtask_count):
            task_id = f"subtask_{i:03d}"
            if i > 0 and i % 3 == 0:
                deps[task_id] = [f"subtask_{i-1:03d}"]
            else:
                deps[task_id] = []
        return deps
    
    def _create_subtasks_from_candidate(self, candidate: PlanCandidate, plan_id: str) -> List[Subtask]:
        """Create subtasks from a plan candidate."""
        subtasks = []
        
        domain = candidate.domains[0] if candidate.domains else TaskDomain.GENERAL
        templates = self._domain_templates.get(domain, [])
        
        for i, template in enumerate(templates):
            subtask = Subtask(
                subtask_id=f"subtask_{i:03d}",
                plan_id=plan_id,
                description=template.get("description", f"Task {i+1}"),
                domain=template.get("domain", domain),
                depth=0,
                status=SubtaskStatus.PENDING,
                definition_of_done=template.get("definition_of_done", DefinitionOfDone()),
                definition_of_ready=template.get("definition_of_ready", DefinitionOfReady()),
                time_estimate_seconds=candidate.estimated_duration_seconds / len(templates),
                cost_estimate=candidate.estimated_cost / len(templates)
            )
            subtasks.append(subtask)
        
        return subtasks
    
    def _persist_plan(self, plan: Plan) -> bool:
        """Persist plan to memory store."""
        plan_data = plan.to_dict()
        plan_data["plan_json"] = json.dumps(plan.to_dict())
        
        success = self.memory.write_plan_history(plan_data)
        
        for subtask in plan.subtasks:
            self.memory.write_subtask_memory(subtask.to_dict())
        
        return success
    
    def approve_plan(self, request: PlanApprovalRequest) -> Plan:
        """Approve or reject a plan."""
        plan_data_list = self.memory.query_plan_history(plan_id=request.plan_id)
        
        if not plan_data_list:
            raise ValueError(f"Plan not found: {request.plan_id}")
        
        plan = Plan.from_dict(plan_data_list[0])
        
        if request.approved:
            plan.status = PlanStatus.APPROVED
            plan.approved_by = request.approver
            plan.approved_at = datetime.now().isoformat()
            plan.auto_approved = False
        else:
            plan.status = PlanStatus.REJECTED
            plan.plan_justification = request.reason
        
        plan.updated_at = datetime.now().isoformat()
        
        self._persist_plan(plan)
        
        logger.info(f"Plan {plan.plan_id} {request.approved if 'approved' else 'rejected'} "
                   f"by {request.approver}")
        
        return plan
    
    def get_plan(self, plan_id: str) -> Optional[Plan]:
        """Retrieve a plan by ID."""
        plan_data_list = self.memory.query_plan_history(plan_id=plan_id)
        
        if not plan_data_list:
            return None
        
        return Plan.from_dict(plan_data_list[0])
    
    def get_plan_history(self, goal_contains: Optional[str] = None, 
                        limit: int = 10) -> List[Dict]:
        """Get plan history."""
        return self.memory.query_plan_history(goal_contains=goal_contains, limit=limit)
    
    def get_subtasks(self, plan_id: str) -> List[Subtask]:
        """Get all subtasks for a plan."""
        subtask_data = self.memory.query_subtasks(plan_id=plan_id)
        return [Subtask.from_dict(s) for s in subtask_data]
    
    def update_subtask_status(self, plan_id: str, subtask_id: str, 
                             status: SubtaskStatus, outcome: str = None) -> bool:
        """Update a subtask's status."""
        subtask_data_list = self.memory.query_subtasks(subtask_id=subtask_id)
        
        if not subtask_data_list:
            return False
        
        subtask_data = subtask_data_list[0]
        subtask_data["status"] = status.value
        if outcome:
            subtask_data["outcome"] = outcome
        
        subtask_data["updated_at"] = datetime.now().isoformat()
        
        return self.memory.write_subtask_memory(subtask_data)
    
    def calculate_plan_risk(self, plan: Plan) -> float:
        """Calculate risk score for a plan."""
        return plan.calculate_risk_score()
    
    def is_low_risk(self, risk_score: float) -> bool:
        """Check if a risk score is below the threshold for auto-approval."""
        return risk_score <= self.dry_run_risk_threshold


_plan_engine: Optional[PlanModeEngine] = None


def get_plan_engine() -> PlanModeEngine:
    """Get or create the global plan engine instance."""
    global _plan_engine
    if _plan_engine is None:
        _plan_engine = PlanModeEngine()
    return _plan_engine


def init_plan_engine(memory_store: MemoryStore = None) -> PlanModeEngine:
    """Initialize a new plan engine instance."""
    global _plan_engine
    _plan_engine = PlanModeEngine(memory_store=memory_store)
    return _plan_engine
