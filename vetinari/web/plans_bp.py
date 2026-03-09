"""Flask Blueprint for plan, subtask, decomposition, and template routes.

Extracted from vetinari/web_ui.py. All plan management, subtask tree,
decomposition lab, assignment pass, and template versioning endpoints live here.
"""

import json
import logging
from pathlib import Path

from flask import Blueprint, jsonify, request

from vetinari.web.shared import PROJECT_ROOT, get_orchestrator, current_config, validate_json_request

logger = logging.getLogger(__name__)

plans_bp = Blueprint('plans', __name__)


@plans_bp.errorhandler(Exception)
def _handle_exception(e):
    logger.error("Unhandled exception in plans_bp: %s", e, exc_info=True)
    return jsonify({"error": str(e)}), 500


# ============ PLAN ENDPOINTS ============

@plans_bp.route('/api/plans', methods=['POST'])
def api_plan_create():
    from vetinari.planning.planning import plan_manager
    data, err = validate_json_request()
    if err:
        return err

    plan = plan_manager.create_plan(
        title=data.get('title', ''),
        prompt=data.get('prompt', ''),
        created_by=data.get('created_by', 'user'),
        waves_data=data.get('waves')
    )

    return jsonify(plan.to_dict()), 201


@plans_bp.route('/api/plans', methods=['GET'])
def api_plans_list():
    from vetinari.planning.planning import plan_manager
    status = request.args.get('status')
    try:
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid 'limit' or 'offset' parameter"}), 400

    plans = plan_manager.list_plans(status=status, limit=limit, offset=offset)

    return jsonify({
        "plans": [p.to_dict() for p in plans],
        "total": len(plans),
        "limit": limit,
        "offset": offset
    })


@plans_bp.route('/api/plans/<plan_id>', methods=['GET'])
def api_plan_get(plan_id):
    from vetinari.planning.planning import plan_manager
    plan = plan_manager.get_plan(plan_id)

    if not plan:
        return jsonify({"error": "Plan not found"}), 404

    return jsonify(plan.to_dict())


@plans_bp.route('/api/plans/<plan_id>', methods=['PUT'])
def api_plan_update(plan_id):
    from vetinari.planning.planning import plan_manager
    data, err = validate_json_request()
    if err:
        return err

    plan = plan_manager.update_plan(plan_id, data)

    if not plan:
        return jsonify({"error": "Plan not found"}), 404

    return jsonify(plan.to_dict())


@plans_bp.route('/api/plans/<plan_id>', methods=['DELETE'])
def api_plan_delete(plan_id):
    from vetinari.planning.planning import plan_manager

    if plan_manager.delete_plan(plan_id):
        return "", 204
    return jsonify({"error": "Plan not found"}), 404


@plans_bp.route('/api/plans/<plan_id>/start', methods=['POST'])
def api_plan_start(plan_id):
    from vetinari.planning.planning import plan_manager
    plan = plan_manager.start_plan(plan_id)

    if not plan:
        return jsonify({"error": "Plan not found"}), 404

    return jsonify({
        "plan_id": plan.plan_id,
        "status": plan.status,
        "started_at": plan.updated_at
    })


@plans_bp.route('/api/plans/<plan_id>/pause', methods=['POST'])
def api_plan_pause(plan_id):
    from vetinari.planning.planning import plan_manager
    plan = plan_manager.pause_plan(plan_id)

    if not plan:
        return jsonify({"error": "Plan not found"}), 404

    return jsonify({
        "plan_id": plan.plan_id,
        "status": plan.status,
        "paused_at": plan.updated_at
    })


@plans_bp.route('/api/plans/<plan_id>/resume', methods=['POST'])
def api_plan_resume(plan_id):
    from vetinari.planning.planning import plan_manager
    plan = plan_manager.resume_plan(plan_id)

    if not plan:
        return jsonify({"error": "Plan not found"}), 404

    return jsonify({
        "plan_id": plan.plan_id,
        "status": plan.status,
        "resumed_at": plan.updated_at
    })


@plans_bp.route('/api/plans/<plan_id>/cancel', methods=['POST'])
def api_plan_cancel(plan_id):
    from vetinari.planning.planning import plan_manager
    plan = plan_manager.cancel_plan(plan_id)

    if not plan:
        return jsonify({"error": "Plan not found"}), 404

    return jsonify({
        "plan_id": plan.plan_id,
        "status": plan.status,
        "cancelled_at": plan.updated_at
    })


@plans_bp.route('/api/plans/<plan_id>/status', methods=['GET'])
def api_plan_status(plan_id):
    from vetinari.planning.planning import plan_manager
    plan = plan_manager.get_plan(plan_id)

    if not plan:
        return jsonify({"error": "Plan not found"}), 404

    return jsonify({
        "plan_id": plan.plan_id,
        "status": plan.status,
        "current_wave": plan.current_wave.wave_id if plan.current_wave else None,
        "completed_tasks": plan.completed_tasks,
        "running_tasks": sum(1 for w in plan.waves for t in w.tasks if t.status == "running"),
        "pending_tasks": sum(1 for w in plan.waves for t in w.tasks if t.status == "pending"),
        "failed_tasks": sum(1 for w in plan.waves for t in w.tasks if t.status == "failed"),
        "progress_percent": plan.progress_percent
    })


# ============ DECOMPOSITION LAB ENDPOINTS ============

@plans_bp.route('/api/decomposition/templates', methods=['GET'])
def api_decomposition_templates():
    from vetinari.planning.decomposition import decomposition_engine
    keywords = request.args.get('keywords', '').split(',') if request.args.get('keywords') else None
    agent_type = request.args.get('agent_type')
    dod_level = request.args.get('dod_level')

    templates = decomposition_engine.get_templates(
        keywords=keywords,
        agent_type=agent_type,
        dod_level=dod_level
    )

    return jsonify({
        "templates": [t.__dict__ for t in templates],
        "total": len(templates)
    })


@plans_bp.route('/api/decomposition/dod-dor', methods=['GET'])
def api_decomposition_dod_dor():
    from vetinari.planning.decomposition import decomposition_engine
    level = request.args.get('level', 'Standard')

    return jsonify({
        "dod_criteria": decomposition_engine.get_dod_criteria(level),
        "dor_criteria": decomposition_engine.get_dor_criteria(level),
        "levels": ["Light", "Standard", "Hard"]
    })


@plans_bp.route('/api/decomposition/decompose', methods=['POST'])
def api_decomposition_decompose():
    from vetinari.planning.decomposition import decomposition_engine
    data, err = validate_json_request()
    if err:
        return err

    task_prompt = data.get('task_prompt', '')
    parent_task_id = data.get('parent_task_id', 'root')
    try:
        depth = int(data.get('depth', 0))
        max_depth = int(data.get('max_depth', 14))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid 'depth' or 'max_depth' parameter"}), 400
    plan_id = data.get('plan_id', 'default')

    if max_depth < 12:
        max_depth = 12
    elif max_depth > 16:
        max_depth = 16

    subtasks = decomposition_engine.decompose_task(
        task_prompt=task_prompt,
        parent_task_id=parent_task_id,
        depth=depth,
        max_depth=max_depth,
        plan_id=plan_id
    )

    return jsonify({
        "subtasks": subtasks,
        "count": len(subtasks),
        "depth": depth,
        "max_depth": max_depth
    })


@plans_bp.route('/api/decomposition/decompose-agent', methods=['POST'])
def api_decomposition_decompose_agent():
    from vetinari.agents.decomposition_agent import decomposition_agent
    from vetinari.planning.planning import plan_manager
    data, err = validate_json_request()
    if err:
        return err

    plan_id = data.get('plan_id')
    prompt = data.get('prompt', '')

    if not plan_id:
        return jsonify({"error": "plan_id required"}), 400

    plan = plan_manager.get_plan(plan_id)
    if not plan:
        plan = plan_manager.create_plan(
            title=f"Plan {plan_id}",
            prompt=prompt,
            created_by="system"
        )

    result = decomposition_agent.decompose_from_prompt(plan, prompt)

    return jsonify(result)


@plans_bp.route('/api/decomposition/knobs', methods=['GET'])
def api_decomposition_knobs():
    from vetinari.agents.decomposition_agent import RECURSION_KNOBS, SEED_RATE, SEED_MIX, DEFAULT_MAX_DEPTH, MIN_MAX_DEPTH, MAX_MAX_DEPTH

    return jsonify({
        "recursion_knobs": RECURSION_KNOBS,
        "seed_mix": SEED_MIX,
        "seed_rate": SEED_RATE,
        "default_max_depth": DEFAULT_MAX_DEPTH,
        "min_max_depth": MIN_MAX_DEPTH,
        "max_max_depth": MAX_MAX_DEPTH
    })


@plans_bp.route('/api/decomposition/history', methods=['GET'])
def api_decomposition_history():
    from vetinari.planning.decomposition import decomposition_engine
    plan_id = request.args.get('plan_id')

    history = decomposition_engine.get_decomposition_history(plan_id)

    return jsonify({
        "history": [
            {
                "event_id": e.event_id,
                "plan_id": e.plan_id,
                "task_id": e.task_id,
                "depth": e.depth,
                "seeds_used": e.seeds_used,
                "subtasks_created": e.subtasks_created,
                "timestamp": e.timestamp
            }
            for e in history
        ],
        "total": len(history)
    })


@plans_bp.route('/api/decomposition/seed-config', methods=['GET'])
def api_decomposition_seed_config():
    from vetinari.planning.decomposition import decomposition_engine

    return jsonify({
        "seed_mix": decomposition_engine.SEED_MIX,
        "seed_rate": decomposition_engine.SEED_RATE,
        "default_max_depth": decomposition_engine.DEFAULT_MAX_DEPTH,
        "min_max_depth": decomposition_engine.MIN_MAX_DEPTH,
        "max_max_depth": decomposition_engine.MAX_MAX_DEPTH
    })


# ============ SUBTASK TREE ENDPOINTS ============

@plans_bp.route('/api/subtasks/<plan_id>', methods=['GET'])
def api_get_subtasks(plan_id):
    from vetinari.planning.subtask_tree import subtask_tree
    parent_id = request.args.get('parent_id')

    if parent_id:
        subtasks = subtask_tree.get_subtasks_by_parent(plan_id, parent_id)
    else:
        subtasks = subtask_tree.get_root_subtasks(plan_id)

    return jsonify({
        "plan_id": plan_id,
        "subtasks": [s.to_dict() for s in subtasks],
        "total": len(subtasks)
    })


@plans_bp.route('/api/subtasks/<plan_id>', methods=['POST'])
def api_create_subtask(plan_id):
    from vetinari.planning.subtask_tree import subtask_tree
    data, err = validate_json_request()
    if err:
        return err

    parent_id = data.get('parent_id', 'root')
    depth = data.get('depth', 0)
    description = data.get('description', '')
    prompt = data.get('prompt', '')
    agent_type = data.get('agent_type', 'builder')
    max_depth = data.get('max_depth', 14)
    max_depth_override = data.get('max_depth_override', 0)
    dod_level = data.get('dod_level', 'Standard')
    dor_level = data.get('dor_level', 'Standard')
    estimated_effort = data.get('estimated_effort', 1.0)
    inputs = data.get('inputs', [])
    outputs = data.get('outputs', [])
    decomposition_seed = data.get('decomposition_seed', '')

    subtask = subtask_tree.create_subtask(
        plan_id=plan_id,
        parent_id=parent_id,
        depth=depth,
        description=description,
        prompt=prompt,
        agent_type=agent_type,
        max_depth=max_depth,
        max_depth_override=max_depth_override,
        dod_level=dod_level,
        dor_level=dor_level,
        estimated_effort=estimated_effort,
        inputs=inputs,
        outputs=outputs,
        decomposition_seed=decomposition_seed
    )

    return jsonify(subtask.to_dict())


@plans_bp.route('/api/subtasks/<plan_id>/<subtask_id>', methods=['PUT'])
def api_update_subtask(plan_id, subtask_id):
    from vetinari.planning.subtask_tree import subtask_tree
    data, err = validate_json_request()
    if err:
        return err

    subtask = subtask_tree.update_subtask(plan_id, subtask_id, data)

    if not subtask:
        return jsonify({"error": "Subtask not found"}), 404

    return jsonify(subtask.to_dict())


@plans_bp.route('/api/subtasks/<plan_id>/tree', methods=['GET'])
def api_get_subtask_tree(plan_id):
    from vetinari.planning.subtask_tree import subtask_tree

    all_subtasks = subtask_tree.get_all_subtasks(plan_id)
    tree_depth = subtask_tree.get_tree_depth(plan_id)

    return jsonify({
        "plan_id": plan_id,
        "subtasks": [s.to_dict() for s in all_subtasks],
        "total": len(all_subtasks),
        "depth": tree_depth
    })


# ============ ASSIGNMENT PASS ENDPOINTS ============

@plans_bp.route('/api/assignments/execute-pass', methods=['POST'])
def api_assignment_execute_pass():
    from vetinari.planning.assignment_pass import execute_assignment_pass
    data, err = validate_json_request()
    if err:
        return err

    plan_id = data.get('plan_id')
    auto_assign = data.get('auto_assign', True)

    if not plan_id:
        return jsonify({"error": "plan_id required"}), 400

    result = execute_assignment_pass(plan_id, auto_assign)

    return jsonify(result)


@plans_bp.route('/api/assignments/<plan_id>', methods=['GET'])
def api_get_assignments(plan_id):
    from vetinari.planning.subtask_tree import subtask_tree

    all_subtasks = subtask_tree.get_all_subtasks(plan_id)

    assignments = []
    for st in all_subtasks:
        assignments.append({
            'subtask_id': st.subtask_id,
            'description': st.description,
            'agent_type': st.agent_type,
            'assigned_agent': st.assigned_agent,
            'status': st.status,
            'depth': st.depth
        })

    return jsonify({
        "plan_id": plan_id,
        "assignments": assignments,
        "total": len(assignments)
    })


@plans_bp.route('/api/assignments/<plan_id>/<subtask_id>', methods=['PUT'])
def api_override_assignment(plan_id, subtask_id):
    from vetinari.planning.subtask_tree import subtask_tree
    data, err = validate_json_request()
    if err:
        return err

    assigned_agent = data.get('assigned_agent')
    if not assigned_agent:
        return jsonify({"error": "assigned_agent required"}), 400

    subtask = subtask_tree.update_subtask(plan_id, subtask_id, {
        'assigned_agent': assigned_agent,
        'status': 'assigned'
    })

    if not subtask:
        return jsonify({"error": "Subtask not found"}), 404

    return jsonify(subtask.to_dict())


# ============ TEMPLATE VERSIONING ENDPOINTS ============

@plans_bp.route('/api/templates/versions', methods=['GET'])
def api_template_versions():
    from vetinari.template_loader import template_loader
    versions = template_loader.list_versions()
    default = template_loader.default_version()
    return jsonify({"versions": versions, "default": default})


@plans_bp.route('/api/templates', methods=['GET'])
def api_templates():
    from vetinari.template_loader import template_loader
    version = request.args.get('version')
    agent_type = request.args.get('agent_type')

    templates = template_loader.load_templates(version=version, agent_type=agent_type)

    return jsonify({
        "templates": templates,
        "total": len(templates),
        "version": version or template_loader.default_version()
    })


@plans_bp.route('/api/plans/<plan_id>/migrate_templates', methods=['POST'])
def api_migrate_templates(plan_id):
    from vetinari.planning.planning import plan_manager
    from vetinari.template_loader import template_loader

    data, err = validate_json_request()
    if err:
        return err
    target_version = data.get("target_version")
    dry_run = data.get("dry_run", True)

    if not target_version:
        return jsonify({"error": "target_version required"}), 400

    available_versions = template_loader.list_versions()
    if target_version not in available_versions:
        return jsonify({"error": f"Invalid target version. Available: {available_versions}"}), 400

    plan = plan_manager.get_plan(plan_id)
    if not plan:
        return jsonify({"error": "Plan not found"}), 404

    from_version = plan.template_version

    if dry_run:
        target_templates = template_loader.load_templates(version=target_version)
        current_templates = template_loader.load_templates(version=from_version) if from_version != target_version else []

        differences = []
        current_ids = {t['template_id'] for t in current_templates}
        target_ids = {t['template_id'] for t in target_templates}

        added = list(target_ids - current_ids)
        removed = list(current_ids - target_ids)

        if added:
            differences.append({"type": "added", "template_ids": added})
        if removed:
            differences.append({"type": "removed", "template_ids": removed})

        recommendation = "re-decompose" if differences else "map-in-place"

        return jsonify({
            "plan_id": plan_id,
            "from_version": from_version,
            "to_version": target_version,
            "dry_run": dry_run,
            "differences": differences,
            "recommendation": recommendation
        })
    else:
        plan_manager.update_plan(plan_id, {"template_version": target_version})

        return jsonify({
            "plan_id": plan_id,
            "from_version": from_version,
            "to_version": target_version,
            "dry_run": dry_run,
            "status": "migrated",
            "message": f"Plan migrated from {from_version} to {target_version}"
        })
