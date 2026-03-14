"""Rules configuration API routes."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from vetinari.web import is_admin_user

bp = Blueprint("rules", __name__)


@bp.route("/api/rules", methods=["GET"])
def api_rules_get():
    """Get all rules configuration.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.rules_manager import get_rules_manager

        rm = get_rules_manager()
        return jsonify(rm.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/rules/global", methods=["GET", "POST"])
def api_rules_global():
    """Get or set global rules.

    Returns:
        Tuple of results.
    """
    if request.method == "POST" and not is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.rules_manager import get_rules_manager

        rm = get_rules_manager()
        if request.method == "POST":
            data = request.json or {}
            rules = data.get("rules", [])
            if isinstance(rules, str):
                rules = [r.strip() for r in rules.splitlines() if r.strip()]
            rm.set_global_rules(rules)
            return jsonify({"status": "saved", "rules": rm.get_global_rules()})
        return jsonify({"rules": rm.get_global_rules()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/rules/global-prompt", methods=["GET", "POST"])
def api_rules_global_prompt():
    """Get or set the global system prompt override.

    Returns:
        Tuple of results.
    """
    if request.method == "POST" and not is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.rules_manager import get_rules_manager

        rm = get_rules_manager()
        if request.method == "POST":
            data = request.json or {}
            rm.set_global_system_prompt(data.get("prompt", ""))
            return jsonify({"status": "saved"})
        return jsonify({"prompt": rm.get_global_system_prompt()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/rules/project/<project_id>", methods=["GET", "POST"])
def api_rules_project(project_id):
    """Get or set rules for a specific project.

    Returns:
        Tuple of results.
    """
    if request.method == "POST" and not is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.rules_manager import get_rules_manager

        rm = get_rules_manager()
        if request.method == "POST":
            data = request.json or {}
            rules = data.get("rules", [])
            if isinstance(rules, str):
                rules = [r.strip() for r in rules.splitlines() if r.strip()]
            rm.set_project_rules(project_id, rules)
            return jsonify({"status": "saved", "project_id": project_id, "rules": rm.get_project_rules(project_id)})
        return jsonify({"project_id": project_id, "rules": rm.get_project_rules(project_id)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/rules/model/<path:model_id>", methods=["GET", "POST"])
def api_rules_model(model_id):
    """Get or set rules for a specific model.

    Returns:
        Tuple of results.
    """
    if request.method == "POST" and not is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.rules_manager import get_rules_manager

        rm = get_rules_manager()
        if request.method == "POST":
            data = request.json or {}
            rules = data.get("rules", [])
            if isinstance(rules, str):
                rules = [r.strip() for r in rules.splitlines() if r.strip()]
            rm.set_model_rules(model_id, rules)
            return jsonify({"status": "saved", "model_id": model_id, "rules": rm.get_model_rules(model_id)})
        return jsonify({"model_id": model_id, "rules": rm.get_model_rules(model_id)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
