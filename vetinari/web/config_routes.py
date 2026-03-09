"""Configuration routes: credentials, rules, model-relay/catalog, ponder endpoints.

System management routes (agents, memory, decisions, sandbox, search, ADR,
image generation, training) live in system_mgmt_routes.py.
"""

import logging
import os

from flask import Blueprint, jsonify, request


from vetinari.web.shared import PROJECT_ROOT, current_config, get_orchestrator, _is_admin_user, validate_json_request

logger = logging.getLogger(__name__)

config_bp = Blueprint('config', __name__)


# ============ ADMIN CREDENTIAL ENDPOINTS ============

@config_bp.route('/api/admin/credentials', methods=['GET'])
def api_admin_list_credentials():
    if not _is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.credentials import credential_manager
        credentials = credential_manager.list()
        health = credential_manager.health()
        return jsonify({"status": "ok", "credentials": credentials, "health": health})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@config_bp.route('/api/admin/credentials/<source_type>', methods=['POST'])
def api_admin_set_credential(source_type):
    if not _is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.credentials import credential_manager
        data, err = validate_json_request()
        if err:
            return err
        token = data.get('token', '')
        credential_type = data.get('credential_type', 'bearer')
        rotation_days = data.get('rotation_days', 30)
        note = data.get('note', '')
        if not token:
            return jsonify({"error": "Token is required"}), 400
        credential_manager.set_credential(
            source_type=source_type,
            token=token,
            credential_type=credential_type,
            rotation_days=rotation_days,
            note=note
        )
        return jsonify({"status": "ok", "message": f"Credential set for {source_type}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@config_bp.route('/api/admin/credentials/<source_type>/rotate', methods=['POST'])
def api_admin_rotate_credential(source_type):
    if not _is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.credentials import credential_manager
        data, err = validate_json_request()
        if err:
            return err
        new_token = data.get('token', '')
        if not new_token:
            return jsonify({"error": "New token is required"}), 400
        success = credential_manager.rotate(source_type, new_token)
        if success:
            return jsonify({"status": "ok", "message": f"Credential rotated for {source_type}"})
        else:
            return jsonify({"error": "Credential not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@config_bp.route('/api/admin/credentials/<source_type>', methods=['DELETE'])
def api_admin_delete_credential(source_type):
    if not _is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.credentials import credential_manager
        credential_manager.vault.remove_credential(source_type)
        return jsonify({"status": "ok", "message": f"Credential removed for {source_type}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@config_bp.route('/api/admin/credentials/health', methods=['GET'])
def api_admin_credentials_health():
    if not _is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.credentials import credential_manager
        health = credential_manager.health()
        return jsonify({"status": "ok", "health": health})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@config_bp.route('/api/admin/permissions', methods=['GET'])
def api_admin_permissions():
    return jsonify({"admin": _is_admin_user()}), 200


# ============ MODEL RELAY ENDPOINTS ============

@config_bp.route('/api/model-catalog', methods=['GET'])
def api_models_list():
    """Model relay catalog (static/configured models). Use /api/models for live LM Studio discovery."""
    try:
        from vetinari.model_relay import model_relay
        models = model_relay.get_all_models()
        return jsonify({"models": [m.to_dict() for m in models]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@config_bp.route('/api/models/<model_id>', methods=['GET'])
def api_model_get(model_id):
    try:
        from vetinari.model_relay import model_relay
        model = model_relay.get_model(model_id)
        if not model:
            return jsonify({"error": "Model not found"}), 404
        return jsonify(model.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@config_bp.route('/api/models/select', methods=['POST'])
def api_model_select():
    try:
        from vetinari.model_relay import model_relay
        data, err = validate_json_request()
        if err:
            return err
        selection = model_relay.pick_model_for_task(
            task_type=data.get('task_type'),
            context=data.get('context')
        )
        return jsonify(selection.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@config_bp.route('/api/models/policy', methods=['GET'])
def api_model_policy_get():
    try:
        from vetinari.model_relay import model_relay
        policy = model_relay.get_policy()
        return jsonify(policy.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@config_bp.route('/api/models/policy', methods=['PUT'])
def api_model_policy_update():
    try:
        from vetinari.model_relay import model_relay, RoutingPolicy
        data, err = validate_json_request()
        if err:
            return err
        policy = RoutingPolicy.from_dict(data)
        model_relay.set_policy(policy)
        return jsonify({"status": "updated", "policy": policy.to_dict()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@config_bp.route('/api/models/reload', methods=['POST'])
def api_models_reload():
    try:
        from vetinari.model_relay import model_relay
        model_relay.reload_catalog()
        return jsonify({"status": "reloaded", "models_loaded": len(model_relay.get_all_models())})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============ PONDER ENDPOINTS ============

@config_bp.route('/api/ponder/choose-model', methods=['POST'])
def api_ponder_choose_model():
    try:
        from vetinari.models.ponder import rank_models, get_available_models
        data, err = validate_json_request()
        if err:
            return err
        task_description = data.get("task_description", "")
        top_n = data.get("top_n", 3)
        template_version = data.get("template_version", "v1")
        if not task_description:
            return jsonify({"error": "task_description required"}), 400
        result = rank_models(task_description, top_n, template_version)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@config_bp.route('/api/ponder/templates', methods=['GET'])
def api_ponder_templates():
    try:
        from vetinari.models.ponder import PonderEngine
        version = request.args.get("version", "v1")
        engine = PonderEngine(template_version=version)
        templates = engine.get_template_prompts()
        return jsonify({"templates": templates, "total": len(templates), "version": version})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@config_bp.route('/api/ponder/models', methods=['GET'])
def api_ponder_models():
    try:
        from vetinari.models.ponder import get_available_models
        models = get_available_models()
        return jsonify({"models": models, "total": len(models)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@config_bp.route('/api/ponder/plan/<plan_id>', methods=['POST'])
def api_ponder_run_plan(plan_id):
    try:
        from vetinari.models.ponder import ponder_project_for_plan
        result = ponder_project_for_plan(plan_id)
        if not result.get("success", False):
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@config_bp.route('/api/ponder/plan/<plan_id>', methods=['GET'])
def api_ponder_get_plan(plan_id):
    try:
        from vetinari.models.ponder import get_ponder_results_for_plan
        result = get_ponder_results_for_plan(plan_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@config_bp.route('/api/ponder/health', methods=['GET'])
def api_ponder_health():
    try:
        from vetinari.models.ponder import get_ponder_health
        health = get_ponder_health()
        return jsonify(health)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============ RULES CONFIGURATION ENDPOINTS ============

@config_bp.route('/api/rules', methods=['GET'])
def api_rules_get():
    """Get all rules configuration."""
    try:
        from vetinari.rules_manager import get_rules_manager
        rm = get_rules_manager()
        return jsonify(rm.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@config_bp.route('/api/rules/global', methods=['GET', 'POST'])
def api_rules_global():
    """Get or set global rules."""
    try:
        from vetinari.rules_manager import get_rules_manager
        rm = get_rules_manager()
        if request.method == 'POST':
            data, err = validate_json_request()
            if err:
                return err
            rules = data.get('rules', [])
            if isinstance(rules, str):
                rules = [r.strip() for r in rules.splitlines() if r.strip()]
            rm.set_global_rules(rules)
            return jsonify({"status": "saved", "rules": rm.get_global_rules()})
        return jsonify({"rules": rm.get_global_rules()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@config_bp.route('/api/rules/global-prompt', methods=['GET', 'POST'])
def api_rules_global_prompt():
    """Get or set the global system prompt override."""
    try:
        from vetinari.rules_manager import get_rules_manager
        rm = get_rules_manager()
        if request.method == 'POST':
            data, err = validate_json_request()
            if err:
                return err
            rm.set_global_system_prompt(data.get('prompt', ''))
            return jsonify({"status": "saved"})
        return jsonify({"prompt": rm.get_global_system_prompt()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@config_bp.route('/api/rules/project/<project_id>', methods=['GET', 'POST'])
def api_rules_project(project_id):
    """Get or set rules for a specific project."""
    try:
        from vetinari.rules_manager import get_rules_manager
        rm = get_rules_manager()
        if request.method == 'POST':
            data, err = validate_json_request()
            if err:
                return err
            rules = data.get('rules', [])
            if isinstance(rules, str):
                rules = [r.strip() for r in rules.splitlines() if r.strip()]
            rm.set_project_rules(project_id, rules)
            return jsonify({"status": "saved", "project_id": project_id, "rules": rm.get_project_rules(project_id)})
        return jsonify({"project_id": project_id, "rules": rm.get_project_rules(project_id)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@config_bp.route('/api/rules/model/<path:model_id>', methods=['GET', 'POST'])
def api_rules_model(model_id):
    """Get or set rules for a specific model."""
    try:
        from vetinari.rules_manager import get_rules_manager
        rm = get_rules_manager()
        if request.method == 'POST':
            data, err = validate_json_request()
            if err:
                return err
            rules = data.get('rules', [])
            if isinstance(rules, str):
                rules = [r.strip() for r in rules.splitlines() if r.strip()]
            rm.set_model_rules(model_id, rules)
            return jsonify({"status": "saved", "model_id": model_id, "rules": rm.get_model_rules(model_id)})
        return jsonify({"model_id": model_id, "rules": rm.get_model_rules(model_id)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
