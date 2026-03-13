"""Task decomposition API routes."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from vetinari.web import is_admin_user

bp = Blueprint("decomposition", __name__)


@bp.route("/api/decomposition/templates", methods=["GET"])
def api_decomposition_templates():
    try:
        from vetinari.decomposition import decomposition_engine

        keywords = request.args.get("keywords", "").split(",") if request.args.get("keywords") else None
        agent_type = request.args.get("agent_type")
        dod_level = request.args.get("dod_level")

        templates = decomposition_engine.get_templates(keywords=keywords, agent_type=agent_type, dod_level=dod_level)

        return jsonify({"templates": [t.__dict__ for t in templates], "total": len(templates)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/decomposition/dod-dor", methods=["GET"])
def api_decomposition_dod_dor():
    try:
        from vetinari.decomposition import decomposition_engine

        level = request.args.get("level", "Standard")

        return jsonify(
            {
                "dod_criteria": decomposition_engine.get_dod_criteria(level),
                "dor_criteria": decomposition_engine.get_dor_criteria(level),
                "levels": ["Light", "Standard", "Hard"],
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/decomposition/decompose", methods=["POST"])
def api_decomposition_decompose():
    if not is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.decomposition import decomposition_engine

        data = request.json or {}

        task_prompt = data.get("task_prompt", "")
        parent_task_id = data.get("parent_task_id", "root")
        depth = int(data.get("depth", 0))
        max_depth = int(data.get("max_depth", 14))
        plan_id = data.get("plan_id", "default")

        if max_depth < 12:
            max_depth = 12
        elif max_depth > 16:
            max_depth = 16

        subtasks = decomposition_engine.decompose_task(
            task_prompt=task_prompt, parent_task_id=parent_task_id, depth=depth, max_depth=max_depth, plan_id=plan_id
        )

        return jsonify({"subtasks": subtasks, "count": len(subtasks), "depth": depth, "max_depth": max_depth})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/decomposition/decompose-agent", methods=["POST"])
def api_decomposition_decompose_agent():
    if not is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.decomposition_agent import decomposition_agent
        from vetinari.planning import plan_manager

        data = request.json or {}

        plan_id = data.get("plan_id")
        prompt = data.get("prompt", "")

        if not plan_id:
            return jsonify({"error": "plan_id required"}), 400

        plan = plan_manager.get_plan(plan_id)
        if not plan:
            plan = plan_manager.create_plan(title=f"Plan {plan_id}", prompt=prompt, created_by="system")

        result = decomposition_agent.decompose_from_prompt(plan, prompt)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/decomposition/knobs", methods=["GET"])
def api_decomposition_knobs():
    try:
        from vetinari.decomposition_agent import (
            DEFAULT_MAX_DEPTH,
            MAX_MAX_DEPTH,
            MIN_MAX_DEPTH,
            RECURSION_KNOBS,
            SEED_MIX,
            SEED_RATE,
        )

        return jsonify(
            {
                "recursion_knobs": RECURSION_KNOBS,
                "seed_mix": SEED_MIX,
                "seed_rate": SEED_RATE,
                "default_max_depth": DEFAULT_MAX_DEPTH,
                "min_max_depth": MIN_MAX_DEPTH,
                "max_max_depth": MAX_MAX_DEPTH,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/decomposition/history", methods=["GET"])
def api_decomposition_history():
    try:
        from vetinari.decomposition import decomposition_engine

        plan_id = request.args.get("plan_id")

        history = decomposition_engine.get_decomposition_history(plan_id)

        return jsonify(
            {
                "history": [
                    {
                        "event_id": e.event_id,
                        "plan_id": e.plan_id,
                        "task_id": e.task_id,
                        "depth": e.depth,
                        "seeds_used": e.seeds_used,
                        "subtasks_created": e.subtasks_created,
                        "timestamp": e.timestamp,
                    }
                    for e in history
                ],
                "total": len(history),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/decomposition/seed-config", methods=["GET"])
def api_decomposition_seed_config():
    try:
        from vetinari.decomposition import decomposition_engine

        return jsonify(
            {
                "seed_mix": decomposition_engine.SEED_MIX,
                "seed_rate": decomposition_engine.SEED_RATE,
                "default_max_depth": decomposition_engine.DEFAULT_MAX_DEPTH,
                "min_max_depth": decomposition_engine.MIN_MAX_DEPTH,
                "max_max_depth": decomposition_engine.MAX_MAX_DEPTH,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
