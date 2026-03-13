"""Ponder (model deliberation) API routes."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from vetinari.web import is_admin_user

bp = Blueprint("ponder", __name__)


@bp.route("/api/ponder/choose-model", methods=["POST"])
def api_ponder_choose_model():
    if not is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.ponder import rank_models

        data = request.json or {}
        task_description = data.get("task_description", "")
        top_n = data.get("top_n", 3)
        template_version = data.get("template_version", "v1")

        if not task_description:
            return jsonify({"error": "task_description required"}), 400

        result = rank_models(task_description, top_n, template_version)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/ponder/templates", methods=["GET"])
def api_ponder_templates():
    try:
        from vetinari.ponder import PonderEngine

        version = request.args.get("version", "v1")
        engine = PonderEngine(template_version=version)
        templates = engine.get_template_prompts()

        return jsonify({"templates": templates, "total": len(templates), "version": version})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/ponder/models", methods=["GET"])
def api_ponder_models():
    try:
        from vetinari.ponder import get_available_models

        models = get_available_models()
        return jsonify({"models": models, "total": len(models)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/ponder/plan/<plan_id>", methods=["POST"])
def api_ponder_run_plan(plan_id):
    if not is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.ponder import ponder_project_for_plan

        result = ponder_project_for_plan(plan_id)

        if not result.get("success", False):
            return jsonify(result), 400

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/ponder/plan/<plan_id>", methods=["GET"])
def api_ponder_get_plan(plan_id):
    try:
        from vetinari.ponder import get_ponder_results_for_plan

        result = get_ponder_results_for_plan(plan_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/ponder/health", methods=["GET"])
def api_ponder_health():
    try:
        from vetinari.ponder import get_ponder_health

        health = get_ponder_health()
        return jsonify(health)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
