import os
import json
import logging
from functools import wraps
from flask import Blueprint, request, jsonify

from .plan_mode import get_plan_engine, PlanModeEngine, PLAN_MODE_DEFAULT, PLAN_MODE_ENABLE
from .plan_types import PlanGenerationRequest, PlanApprovalRequest, TaskDomain
from .memory import PLAN_ADMIN_TOKEN

logger = logging.getLogger(__name__)

plan_api = Blueprint('plan_api', __name__)


def require_admin_token(f):
    """Decorator to require admin token for plan management endpoints."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        token = ''
        
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
        elif auth_header:
            token = auth_header
        
        if PLAN_ADMIN_TOKEN and token != PLAN_ADMIN_TOKEN:
            logger.warning(f"Unauthorized plan API access attempt")
            return jsonify({"error": "Unauthorized", "message": "Invalid or missing admin token"}), 401
        
        return f(*args, **kwargs)
    return decorated_function


def check_plan_mode_enabled():
    """Check if plan mode is enabled."""
    if not PLAN_MODE_ENABLE:
        return False, "Plan mode is disabled"
    return True, None


@plan_api.route('/api/plan/generate', methods=['POST'])
@require_admin_token
def generate_plan():
    """Generate a plan from a goal.
    
    Body:
    {
        "goal": "string - the goal to create a plan for",
        "constraints": "string - optional constraints",
        "plan_depth_cap": 16,
        "max_candidates": 3,
        "domain_hint": "coding|data_processing|infra|docs|ai_experiments|research|general",
        "dry_run": false,
        "risk_threshold": 0.25
    }
    """
    enabled, error = check_plan_mode_enabled()
    if not enabled:
        return jsonify({"error": "Plan mode disabled", "message": error}), 403
    
    data = request.get_json() or {}
    
    goal = data.get('goal', '')
    if not goal:
        return jsonify({"error": "Missing goal", "message": "Goal is required"}), 400
    
    try:
        domain_hint = None
        if data.get('domain_hint'):
            domain_hint = TaskDomain(data['domain_hint'])
        
        req = PlanGenerationRequest(
            goal=goal,
            constraints=data.get('constraints', ''),
            plan_depth_cap=data.get('plan_depth_cap', 16),
            max_candidates=data.get('max_candidates', 3),
            domain_hint=domain_hint,
            dry_run=data.get('dry_run', False),
            risk_threshold=data.get('risk_threshold', 0.25)
        )
        
        engine = get_plan_engine()
        plan = engine.generate_plan(req)
        
        return jsonify({
            "success": True,
            "plan_id": plan.plan_id,
            "version": plan.plan_version,
            "goal": plan.goal,
            "status": plan.status.value,
            "risk_score": plan.risk_score,
            "risk_level": plan.risk_level.value,
            "subtask_count": len(plan.subtasks),
            "dry_run": plan.dry_run,
            "auto_approved": plan.auto_approved,
            "plan_candidates": [c.to_dict() for c in plan.plan_candidates],
            "chosen_plan_id": plan.chosen_plan_id,
            "plan_justification": plan.plan_justification,
            "created_at": plan.created_at
        })
    
    except Exception as e:
        logger.error(f"Plan generation failed: {e}")
        return jsonify({"error": "Plan generation failed", "message": str(e)}), 500


@plan_api.route('/api/plan/<plan_id>', methods=['GET'])
@require_admin_token
def get_plan(plan_id):
    """Get plan details by ID."""
    enabled, error = check_plan_mode_enabled()
    if not enabled:
        return jsonify({"error": "Plan mode disabled", "message": error}), 403
    
    try:
        engine = get_plan_engine()
        plan = engine.get_plan(plan_id)
        
        if not plan:
            return jsonify({"error": "Plan not found", "plan_id": plan_id}), 404
        
        return jsonify({
            "success": True,
            "plan": {
                "plan_id": plan.plan_id,
                "version": plan.plan_version,
                "goal": plan.goal,
                "constraints": plan.constraints,
                "status": plan.status.value,
                "risk_score": plan.risk_score,
                "risk_level": plan.risk_level.value,
                "dry_run": plan.dry_run,
                "auto_approved": plan.auto_approved,
                "approved_by": plan.approved_by,
                "approved_at": plan.approved_at,
                "subtasks": [s.to_dict() for s in plan.subtasks],
                "dependencies": plan.dependencies,
                "plan_candidates": [c.to_dict() for c in plan.plan_candidates],
                "chosen_plan_id": plan.chosen_plan_id,
                "plan_justification": plan.plan_justification,
                "created_at": plan.created_at,
                "updated_at": plan.updated_at,
                "completed_at": plan.completed_at
            }
        })
    
    except Exception as e:
        logger.error(f"Failed to get plan: {e}")
        return jsonify({"error": "Failed to get plan", "message": str(e)}), 500


@plan_api.route('/api/plan/<plan_id>/approve', methods=['POST'])
@require_admin_token
def approve_plan(plan_id):
    """Approve or reject a plan.
    
    Body (JSON Schema v1):
    {
        "approved": true,          # required - boolean
        "approver": "admin",      # required - string
        "reason": "optional reason",  # optional - string
        "audit_id": "optional",    # optional - string, auto-generated if not provided
        "risk_score": 0.15,       # optional - float
        "timestamp": "ISO string", # optional - auto-generated if not provided
        "approval_schema_version": 1  # optional - int, default 1
    }
    """
    enabled, error = check_plan_mode_enabled()
    if not enabled:
        return jsonify({"error": "Plan mode disabled", "message": error}), 403
    
    data = request.get_json() or {}
    
    # Validate required fields
    required_fields = ['approved', 'approver']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    approved = data.get('approved', False)
    approver = data.get('approver', 'admin')
    reason = data.get('reason', '')
    audit_id = data.get('audit_id')
    risk_score = data.get('risk_score')
    timestamp = data.get('timestamp')
    approval_schema_version = data.get('approval_schema_version', 1)
    
    try:
        req = PlanApprovalRequest(
            plan_id=plan_id,
            approved=approved,
            approver=approver,
            reason=reason,
            audit_id=audit_id,
            risk_score=risk_score,
            timestamp=timestamp or "",
            approval_schema_version=approval_schema_version
        )
        
        engine = get_plan_engine()
        plan = engine.approve_plan(req)
        
        # Log approval decision to dual memory if available
        try:
            from .memory import DUAL_MEMORY_AVAILABLE, get_dual_memory_store, MemoryEntry, MemoryEntryType
            if DUAL_MEMORY_AVAILABLE:
                store = get_dual_memory_store()
                approval_entry = MemoryEntry(
                    agent="plan-approval",
                    entry_type=MemoryEntryType.APPROVAL,
                    content=json.dumps({
                        "audit_id": audit_id,
                        "plan_id": plan_id,
                        "approved": approved,
                        "approver": approver,
                        "reason": reason,
                        "risk_score": risk_score,
                        "approval_schema_version": approval_schema_version
                    }),
                    summary=f"Plan {plan_id} {'approved' if approved else 'rejected'} by {approver}",
                    provenance="plan_api_approve"
                )
                store.remember(approval_entry)
        except Exception as mem_err:
            logger.warning(f"Failed to log approval to memory: {mem_err}")
        
        return jsonify({
            "success": True,
            "plan_id": plan.plan_id,
            "status": plan.status.value,
            "approved_by": plan.approved_by,
            "approved_at": plan.approved_at,
            "audit_id": audit_id
        })
    
    except ValueError as e:
        return jsonify({"error": "Plan not found", "message": str(e)}), 404
    except Exception as e:
        logger.error(f"Failed to approve plan: {e}")
        return jsonify({"error": "Failed to approve plan", "message": str(e)}), 500


@plan_api.route('/api/plan/<plan_id>/subtasks/<subtask_id>/approve', methods=['POST'])
@require_admin_token
def approve_subtask(plan_id, subtask_id):
    """Approve or reject a specific subtask.
    
    Body (JSON Schema v1):
    {
        "approved": true,          # required - boolean
        "approver": "admin",       # required - string
        "reason": "optional reason",  # optional - string
        "audit_id": "optional",    # optional - string
        "risk_score": 0.15         # optional - float
    }
    """
    enabled, error = check_plan_mode_enabled()
    if not enabled:
        return jsonify({"error": "Plan mode disabled", "message": error}), 403
    
    data = request.get_json() or {}
    
    required_fields = ['approved', 'approver']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    approved = data.get('approved', False)
    approver = data.get('approver', 'admin')
    reason = data.get('reason', '')
    audit_id = data.get('audit_id')
    risk_score = data.get('risk_score')
    
    try:
        engine = get_plan_engine()
        plan = engine.get_plan(plan_id)
        
        if not plan:
            return jsonify({"error": "Plan not found", "plan_id": plan_id}), 404
        
        # Check if subtask requires approval
        approval_check = engine.check_subtask_approval_required(plan, subtask_id, plan_mode=True)
        
        if 'error' in approval_check:
            return jsonify({"error": approval_check['error']}), 404
        
        # Log approval decision
        engine.log_approval_decision(
            plan_id=plan_id,
            subtask_id=subtask_id,
            approved=approved,
            approver=approver,
            reason=reason,
            risk_score=risk_score or plan.risk_score
        )
        
        return jsonify({
            "success": True,
            "plan_id": plan_id,
            "subtask_id": subtask_id,
            "approved": approved,
            "approver": approver,
            "requires_approval": approval_check.get('requires_approval', False),
            "audit_id": audit_id
        })
    
    except Exception as e:
        logger.error(f"Failed to approve subtask: {e}")
        return jsonify({"error": "Failed to approve subtask", "message": str(e)}), 500


@plan_api.route('/api/plan/<plan_id>/history', methods=['GET'])
@require_admin_token
def get_plan_history(plan_id):
    """Get plan history and memory for a plan."""
    enabled, error = check_plan_mode_enabled()
    if not enabled:
        return jsonify({"error": "Plan mode disabled", "message": error}), 403
    
    try:
        engine = get_plan_engine()
        plan = engine.get_plan(plan_id)
        
        if not plan:
            return jsonify({"error": "Plan not found", "plan_id": plan_id}), 404
        
        subtasks = engine.get_subtasks(plan_id)
        
        return jsonify({
            "success": True,
            "plan_id": plan_id,
            "goal": plan.goal,
            "status": plan.status.value,
            "risk_score": plan.risk_score,
            "risk_level": plan.risk_level.value,
            "subtasks": [s.to_dict() for s in subtasks],
            "created_at": plan.created_at,
            "updated_at": plan.updated_at,
            "completed_at": plan.completed_at,
            "auto_approved": plan.auto_approved,
            "approved_by": plan.approved_by
        })
    
    except Exception as e:
        logger.error(f"Failed to get plan history: {e}")
        return jsonify({"error": "Failed to get plan history", "message": str(e)}), 500


@plan_api.route('/api/plan/history', methods=['GET'])
@require_admin_token
def get_all_plan_history():
    """Get all plan history.
    
    Query params:
    - goal_contains: filter plans by goal containing this string
    - limit: max number of plans to return (default 10)
    """
    enabled, error = check_plan_mode_enabled()
    if not enabled:
        return jsonify({"error": "Plan mode disabled", "message": error}), 403
    
    goal_contains = request.args.get('goal_contains')
    limit = int(request.args.get('limit', 10))
    
    try:
        engine = get_plan_engine()
        plans = engine.get_plan_history(goal_contains=goal_contains, limit=limit)
        
        return jsonify({
            "success": True,
            "plans": plans,
            "count": len(plans)
        })
    
    except Exception as e:
        logger.error(f"Failed to get plan history: {e}")
        return jsonify({"error": "Failed to get plan history", "message": str(e)}), 500


@plan_api.route('/api/plan/<plan_id>/subtasks', methods=['GET'])
@require_admin_token
def get_plan_subtasks(plan_id):
    """Get all subtasks for a plan."""
    enabled, error = check_plan_mode_enabled()
    if not enabled:
        return jsonify({"error": "Plan mode disabled", "message": error}), 403
    
    try:
        engine = get_plan_engine()
        plan = engine.get_plan(plan_id)
        
        if not plan:
            return jsonify({"error": "Plan not found", "plan_id": plan_id}), 404
        
        subtasks = engine.get_subtasks(plan_id)
        
        return jsonify({
            "success": True,
            "plan_id": plan_id,
            "subtasks": [s.to_dict() for s in subtasks],
            "count": len(subtasks)
        })
    
    except Exception as e:
        logger.error(f"Failed to get subtasks: {e}")
        return jsonify({"error": "Failed to get subtasks", "message": str(e)}), 500


@plan_api.route('/api/plan/status', methods=['GET'])
def get_plan_mode_status():
    """Get plan mode status and configuration."""
    from .memory import get_memory_store
    
    try:
        memory = get_memory_store()
        stats = memory.get_memory_stats()
    except:
        stats = {}
    
    return jsonify({
        "success": True,
        "plan_mode_enabled": PLAN_MODE_ENABLE,
        "plan_mode_default": PLAN_MODE_DEFAULT,
        "config": {
            "PLAN_MODE_ENABLE": PLAN_MODE_ENABLE,
            "PLAN_MODE_DEFAULT": PLAN_MODE_DEFAULT,
            "PLAN_ADMIN_TOKEN_SET": bool(PLAN_ADMIN_TOKEN)
        },
        "memory_stats": stats
    })


@plan_api.route('/api/plan/templates', methods=['GET'])
def get_plan_templates():
    """Get available plan templates by domain."""
    try:
        engine = get_plan_engine()
        
        templates = {}
        for domain in TaskDomain:
            domain_templates = engine._domain_templates.get(domain, [])
            templates[domain.value] = [
                {
                    "description": t.get("description"),
                    "definition_of_done": t.get("definition_of_done").criteria if t.get("definition_of_done") else [],
                    "definition_of_ready": t.get("definition_of_ready").prerequisites if t.get("definition_of_ready") else []
                }
                for t in domain_templates
            ]
        
        return jsonify({
            "success": True,
            "templates": templates,
            "domains": [d.value for d in TaskDomain]
        })
    
    except Exception as e:
        logger.error(f"Failed to get templates: {e}")
        return jsonify({"error": "Failed to get templates", "message": str(e)}), 500


@plan_api.route('/api/plan/<plan_id>/explanations', methods=['GET'])
@require_admin_token
def get_plan_explanations(plan_id):
    """Get plan explanation.
    
    Returns the explanation for a plan including blocks and summary.
    Query params:
    - sanitized: if true, return sanitized version for public exposure (default: false)
    """
    enabled, error = check_plan_mode_enabled()
    if not enabled:
        return jsonify({"error": "Plan mode disabled", "message": error}), 403
    
    sanitized = request.args.get('sanitized', 'false').lower() in ('1', 'true', 'yes')
    
    try:
        from .explain_agent import get_explain_agent
        from .plan_types import PlanExplanation
        
        engine = get_plan_engine()
        plan = engine.get_plan(plan_id)
        
        if not plan:
            return jsonify({"error": "Plan not found", "plan_id": plan_id}), 404
        
        explanation_data = {}
        if plan.plan_explanation_json:
            try:
                explanation_data = json.loads(plan.plan_explanation_json)
            except:
                pass
        
        if sanitized and explanation_data:
            explain_agent = get_explain_agent()
            explanation = PlanExplanation.from_dict(explanation_data)
            sanitized_exp = explain_agent.sanitize_explanation(explanation)
            return jsonify({
                "success": True,
                "plan_id": plan_id,
                "explanation": sanitized_exp.to_dict()
            })
        
        return jsonify({
            "success": True,
            "plan_id": plan_id,
            "explanation": explanation_data if explanation_data else None
        })
    
    except Exception as e:
        logger.error(f"Failed to get plan explanations: {e}")
        return jsonify({"error": "Failed to get explanations", "message": str(e)}), 500


@plan_api.route('/api/plan/<plan_id>/subtasks/<subtask_id>/explanation', methods=['GET'])
@require_admin_token
def get_subtask_explanation(plan_id, subtask_id):
    """Get explanation for a specific subtask."""
    enabled, error = check_plan_mode_enabled()
    if not enabled:
        return jsonify({"error": "Plan mode disabled", "message": error}), 403
    
    try:
        engine = get_plan_engine()
        subtasks = engine.get_subtasks(plan_id)
        
        subtask = next((s for s in subtasks if s.subtask_id == subtask_id), None)
        if not subtask:
            return jsonify({"error": "Subtask not found", "subtask_id": subtask_id}), 404
        
        explanation_data = {}
        if subtask.subtask_explanation_json:
            try:
                explanation_data = json.loads(subtask.subtask_explanation_json)
            except:
                pass
        
        return jsonify({
            "success": True,
            "plan_id": plan_id,
            "subtask_id": subtask_id,
            "explanation": explanation_data if explanation_data else None
        })
    
    except Exception as e:
        logger.error(f"Failed to get subtask explanation: {e}")
        return jsonify({"error": "Failed to get subtask explanation", "message": str(e)}), 500


@plan_api.route('/api/coding/task', methods=['POST'])
@require_admin_token
def create_coding_task():
    """Create and execute a coding task.
    
    Body:
    {
        "type": "scaffold|implement|test|review",
        "language": "python",
        "description": "Task description",
        "repo_path": "./",
        "target_files": ["file1.py", "file2.py"],
        "constraints": "optional constraints"
    }
    """
    try:
        from .coding_agent import CodeAgentEngine, CodeTask, CodingTaskType, get_coding_agent
        
        data = request.get_json() or {}
        
        task_type_str = data.get("type", "implement")
        try:
            task_type = CodingTaskType(task_type_str)
        except ValueError:
            return jsonify({"error": f"Invalid task type: {task_type_str}"}), 400
        
        agent = get_coding_agent()
        
        if not agent.is_available():
            return jsonify({"error": "Coding agent not available"}), 503
        
        task = CodeTask(
            type=task_type,
            language=data.get("language", "python"),
            framework=data.get("framework", ""),
            repo_path=data.get("repo_path", "./"),
            description=data.get("description", ""),
            constraints=data.get("constraints", ""),
            target_files=data.get("target_files", [])
        )
        
        artifact = agent.run_task(task)
        
        return jsonify({
            "success": True,
            "task_id": task.task_id,
            "artifact": artifact.to_dict()
        })
    
    except Exception as e:
        logger.error(f"Failed to create coding task: {e}")
        return jsonify({"error": "Failed to create coding task", "message": str(e)}), 500


@plan_api.route('/api/coding/task/<task_id>', methods=['GET'])
@require_admin_token
def get_coding_task(task_id):
    """Get coding task status and result."""
    try:
        from .coding_agent import get_coding_agent
        
        agent = get_coding_agent()
        
        if not agent.is_available():
            return jsonify({"error": "Coding agent not available"}), 503
        
        # For MVP, return basic status
        return jsonify({
            "success": True,
            "task_id": task_id,
            "status": "completed"
        })
    
    except Exception as e:
        logger.error(f"Failed to get coding task: {e}")
        return jsonify({"error": "Failed to get coding task", "message": str(e)}), 500


@plan_api.route('/api/coding/multi-step', methods=['POST'])
@require_admin_token
def create_multi_step_coding():
    """Create and execute multiple coding tasks (scaffold + module + tests).
    
    Body:
    {
        "plan_id": "plan_xxx",
        "subtasks": [
            {"subtask_id": "s1", "type": "scaffold", "description": "..."},
            {"subtask_id": "s2", "type": "implement", "description": "..."},
            {"subtask_id": "s3", "type": "test", "description": "..."}
        ]
    }
    """
    try:
        from .coding_agent import CodeAgentEngine, CodeTask, CodingTaskType, get_coding_agent
        from .plan_types import TaskDomain
        
        data = request.get_json() or {}
        plan_id = data.get("plan_id", "")
        subtasks_data = data.get("subtasks", [])
        
        agent = get_coding_agent()
        
        if not agent.is_available():
            return jsonify({"error": "Coding agent not available"}), 503
        
        results = []
        
        for st_data in subtasks_data:
            task_type_str = st_data.get("type", "implement")
            try:
                task_type = CodingTaskType(task_type_str)
            except ValueError:
                task_type = CodingTaskType.IMPLEMENT
            
            task = CodeTask(
                plan_id=plan_id,
                subtask_id=st_data.get("subtask_id", ""),
                type=task_type,
                language=st_data.get("language", "python"),
                description=st_data.get("description", ""),
                repo_path=st_data.get("repo_path", "./"),
                target_files=st_data.get("target_files", []),
                constraints=st_data.get("constraints", "")
            )
            
            try:
                artifact = agent.run_task(task)
                results.append({
                    "subtask_id": st_data.get("subtask_id"),
                    "success": True,
                    "artifact": artifact.to_dict()
                })
            except Exception as task_err:
                results.append({
                    "subtask_id": st_data.get("subtask_id"),
                    "success": False,
                    "error": str(task_err)
                })
        
        return jsonify({
            "success": True,
            "plan_id": plan_id,
            "results": results
        })
    
    except Exception as e:
        logger.error(f"Failed to create multi-step coding: {e}")
        return jsonify({"error": "Failed to create multi-step coding", "message": str(e)}), 500


def register_plan_api(app):
    """Register plan API routes with Flask app."""
    app.register_blueprint(plan_api)
    logger.info("Plan API routes registered")
