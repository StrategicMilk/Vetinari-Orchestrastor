import os
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
    
    Body:
    {
        "approved": true,
        "approver": "admin",
        "reason": "optional reason"
    }
    """
    enabled, error = check_plan_mode_enabled()
    if not enabled:
        return jsonify({"error": "Plan mode disabled", "message": error}), 403
    
    data = request.get_json() or {}
    approved = data.get('approved', False)
    approver = data.get('approver', 'admin')
    reason = data.get('reason', '')
    
    try:
        req = PlanApprovalRequest(
            plan_id=plan_id,
            approved=approved,
            approver=approver,
            reason=reason
        )
        
        engine = get_plan_engine()
        plan = engine.approve_plan(req)
        
        return jsonify({
            "success": True,
            "plan_id": plan.plan_id,
            "status": plan.status.value,
            "approved_by": plan.approved_by,
            "approved_at": plan.approved_at
        })
    
    except ValueError as e:
        return jsonify({"error": "Plan not found", "message": str(e)}), 404
    except Exception as e:
        logger.error(f"Failed to approve plan: {e}")
        return jsonify({"error": "Failed to approve plan", "message": str(e)}), 500


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


def register_plan_api(app):
    """Register plan API routes with Flask app."""
    app.register_blueprint(plan_api)
    logger.info("Plan API routes registered")
