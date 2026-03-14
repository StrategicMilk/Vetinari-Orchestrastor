"""Flask Blueprint for admin, agent, memory, sandbox, ADR, model-relay,.

ponder, rules, training, and image generation routes.

All routes extracted from vetinari/web_ui.py.  Shared state is imported
from vetinari.web.shared so there are no circular dependencies.
"""

from __future__ import annotations

import logging
import os
import uuid

from flask import Blueprint, jsonify, request

from vetinari.web import require_admin
from vetinari.web.shared import _is_admin_user, current_config

logger = logging.getLogger(__name__)

admin_bp = Blueprint("admin", __name__)


# ============ ADMIN CREDENTIAL ENDPOINTS ============


@admin_bp.route("/api/admin/credentials", methods=["GET"])
def api_admin_list_credentials():
    """Api admin list credentials.

    Returns:
        Tuple of results.
    """
    if not _is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.credentials import credential_manager

        credentials = credential_manager.list()
        health = credential_manager.health()

        return jsonify({"status": "ok", "credentials": credentials, "health": health})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/admin/credentials/<source_type>", methods=["POST"])
def api_admin_set_credential(source_type):
    """Api admin set credential.

    Returns:
        Tuple of results.
    """
    if not _is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.credentials import credential_manager

        data = request.json or {}
        token = data.get("token", "")
        credential_type = data.get("credential_type", "bearer")
        rotation_days = data.get("rotation_days", 30)
        note = data.get("note", "")

        if not token:
            return jsonify({"error": "Token is required"}), 400

        credential_manager.set_credential(
            source_type=source_type,
            token=token,
            credential_type=credential_type,
            rotation_days=rotation_days,
            note=note,
        )

        return jsonify({"status": "ok", "message": f"Credential set for {source_type}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/admin/credentials/<source_type>/rotate", methods=["POST"])
def api_admin_rotate_credential(source_type):
    """Api admin rotate credential.

    Returns:
        Tuple of results.
    """
    if not _is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.credentials import credential_manager

        data = request.json or {}
        new_token = data.get("token", "")

        if not new_token:
            return jsonify({"error": "New token is required"}), 400

        success = credential_manager.rotate(source_type, new_token)

        if success:
            return jsonify({"status": "ok", "message": f"Credential rotated for {source_type}"})
        else:
            return jsonify({"error": "Credential not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/admin/credentials/<source_type>", methods=["DELETE"])
def api_admin_delete_credential(source_type):
    """Api admin delete credential.

    Returns:
        Tuple of results.
    """
    if not _is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.credentials import credential_manager

        credential_manager.vault.remove_credential(source_type)

        return jsonify({"status": "ok", "message": f"Credential removed for {source_type}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/admin/credentials/health", methods=["GET"])
def api_admin_credentials_health():
    """Api admin credentials health.

    Returns:
        Tuple of results.
    """
    if not _is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.credentials import credential_manager

        health = credential_manager.health()

        return jsonify({"status": "ok", "health": health})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/admin/permissions", methods=["GET"])
def api_admin_permissions():
    return jsonify({"admin": _is_admin_user()}), 200


# ============ AGENT ORCHESTRATION ENDPOINTS ============


@admin_bp.route("/api/agents/status", methods=["GET"])
@require_admin
def api_agents_status():
    """Api agents status.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.multi_agent_orchestrator import MultiAgentOrchestrator

        orch = MultiAgentOrchestrator.get_instance()

        if orch is None:
            return jsonify({"agents": []})

        # Use existing to_dict() which handles enum serialization correctly
        return jsonify({"agents": orch.get_agent_status()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/agents/initialize", methods=["POST"])
@require_admin
def api_agents_initialize():
    """Api agents initialize.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.multi_agent_orchestrator import MultiAgentOrchestrator

        orch = MultiAgentOrchestrator.get_instance()

        if orch is None:
            orch = MultiAgentOrchestrator()

        orch.initialize_agents()

        agent_names = [a.name for a in orch.agents.values()]
        return jsonify({"status": "initialized", "agents": agent_names})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/agents/active", methods=["GET"])
@require_admin
def api_agents_active():
    """Api agents active.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.multi_agent_orchestrator import MultiAgentOrchestrator

        orch = MultiAgentOrchestrator.get_instance()

        if orch is None:
            return jsonify({"agents": []})

        agents = []
        colors = ["#6366f1", "#8b5cf6", "#ec4899", "#14b8a6", "#f59e0b", "#ef4444", "#3b82f6", "#10b981"]
        icons = {
            "explorer": "fa-compass",
            "librarian": "fa-book",
            "oracle": "fa-globe",
            "ui_planner": "fa-palette",  # underscore, not hyphen
            "builder": "fa-hammer",
            "researcher": "fa-search",
            "evaluator": "fa-check-circle",
            "synthesizer": "fa-brain",
            "planner": "fa-sitemap",
            "security_auditor": "fa-shield-alt",
            "data_engineer": "fa-database",
        }

        for i, agent in enumerate(orch.agents.values()):
            # Safely get the string value of the enum
            agent_type_val = agent.agent_type.value if hasattr(agent.agent_type, "value") else str(agent.agent_type)
            agents.append(
                {
                    "name": agent.name,
                    "role": agent_type_val,
                    "color": colors[i % len(colors)],
                    "icon": icons.get(agent_type_val, "fa-robot"),
                    "tasks_completed": agent.tasks_completed,
                    "current_task": agent.current_task.to_dict()
                    if agent.current_task and hasattr(agent.current_task, "to_dict")
                    else None,
                    "state": agent.state.value if hasattr(agent.state, "value") else str(agent.state),
                }
            )

        return jsonify({"agents": agents})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/agents/tasks", methods=["GET"])
@require_admin
def api_agents_tasks():
    """Api agents tasks.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.multi_agent_orchestrator import MultiAgentOrchestrator

        orch = MultiAgentOrchestrator.get_instance()

        if orch is None:
            return jsonify({"tasks": []})

        tasks = []
        for task in orch.task_queue:
            tasks.append(
                {
                    "id": task.get("id"),
                    "description": task.get("description"),
                    "status": task.get("status", "pending"),
                    "agent": task.get("assigned_agent", "unassigned"),
                }
            )

        return jsonify({"tasks": tasks})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============ SHARED MEMORY ENDPOINTS ============


@admin_bp.route("/api/memory", methods=["GET"])
@require_admin
def api_memory():
    """Api memory.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.shared_memory import SharedMemory

        memory = SharedMemory.get_instance()

        memories = memory.get_all()

        return jsonify({"memories": memories})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============ DECISION ENDPOINTS ============


@admin_bp.route("/api/decisions/pending", methods=["GET"])
@require_admin
def api_decisions_pending():
    """Api decisions pending.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.shared_memory import SharedMemory

        memory = SharedMemory.get_instance()

        decisions = memory.get_memories_by_type("decision")

        pending = []
        for d in decisions:
            if not d.get("resolved"):
                pending.append(
                    {
                        "id": d.get("id"),
                        "prompt": d.get("content", ""),
                        "options": d.get("options", []),
                        "context": d.get("context", {}),
                    }
                )

        return jsonify({"decisions": pending})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/decisions", methods=["POST"])
@require_admin
def api_decisions_submit():
    """Api decisions submit.

    Returns:
        The jsonify result.
    """
    try:
        data = request.json
        decision_id = data.get("decision_id")
        choice = data.get("choice")

        from vetinari.shared_memory import SharedMemory

        memory = SharedMemory.get_instance()

        memory.resolve_decision(decision_id, choice)

        return jsonify({"status": "resolved", "choice": choice})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============ MODEL RELAY ENDPOINTS ============


@admin_bp.route("/api/model-catalog", methods=["GET"])
@require_admin
def api_models_list():
    """Model relay catalog (static/configured models). Use /api/models for live LM Studio discovery.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.model_relay import model_relay

        models = model_relay.get_all_models()

        return jsonify({"models": [m.to_dict() for m in models]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/models/<model_id>", methods=["GET"])
@require_admin
def api_model_get(model_id):
    """Api model get.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.model_relay import model_relay

        model = model_relay.get_model(model_id)

        if not model:
            return jsonify({"error": "Model not found"}), 404

        return jsonify(model.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/models/select", methods=["POST"])
@require_admin
def api_model_select():
    """Api model select.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.model_relay import model_relay

        data = request.json

        selection = model_relay.pick_model_for_task(task_type=data.get("task_type"), context=data.get("context"))

        return jsonify(selection.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/models/policy", methods=["GET"])
@require_admin
def api_model_policy_get():
    """Api model policy get.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.model_relay import model_relay

        policy = model_relay.get_policy()

        return jsonify(policy.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/models/policy", methods=["PUT"])
@require_admin
def api_model_policy_update():
    """Api model policy update.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.model_relay import RoutingPolicy, model_relay

        data = request.json

        policy = RoutingPolicy.from_dict(data)
        model_relay.set_policy(policy)

        return jsonify({"status": "updated", "policy": policy.to_dict()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/models/reload", methods=["POST"])
@require_admin
def api_models_reload():
    """Api models reload.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.model_relay import model_relay

        model_relay.reload_catalog()

        return jsonify({"status": "reloaded", "models_loaded": len(model_relay.get_all_models())})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============ SANDBOX ENDPOINTS ============


@admin_bp.route("/api/sandbox/execute", methods=["POST"])
@require_admin
def api_sandbox_execute():
    """Api sandbox execute.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.sandbox import sandbox_manager

        data = request.json

        result = sandbox_manager.execute(
            code=data.get("code", ""),
            sandbox_type=data.get("sandbox_type", "in_process"),
            timeout=data.get("timeout", 30),
            context=data.get("context"),
        )

        return jsonify(result.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/sandbox/status", methods=["GET"])
@require_admin
def api_sandbox_status():
    """Api sandbox status.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.sandbox import sandbox_manager

        status = sandbox_manager.get_status()

        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/sandbox/audit", methods=["GET"])
@require_admin
def api_sandbox_audit():
    """Api sandbox audit.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.sandbox import sandbox_manager

        limit = int(request.args.get("limit", 100))

        audit = sandbox_manager.get_audit_log(limit)

        return jsonify({"audit_entries": audit, "total": len(audit)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============ SEARCH ENDPOINTS ============


@admin_bp.route("/api/code-search", methods=["GET"])
@require_admin
def api_code_search():
    """Search within project code using CocoIndex/ripgrep backends.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.code_search import code_search_registry

        query = request.args.get("q", "")
        limit = int(request.args.get("limit", 10))
        request.args.get("language")
        backend = request.args.get("backend", "cocoindex")

        if not query:
            return jsonify({"error": "Query required"}), 400

        adapter = code_search_registry.get_adapter(backend)
        results = adapter.search(query, limit=limit)

        return jsonify(
            {"query": query, "backend": backend, "results": [r.to_dict() for r in results], "total": len(results)}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/search/index", methods=["POST"])
@require_admin
def api_search_index():
    """Api search index.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.code_search import code_search_registry

        data = request.json

        project_path = data.get("project_path")
        backend = data.get("backend", "cocoindex")
        force = data.get("force", False)

        if not project_path:
            return jsonify({"error": "project_path required"}), 400

        adapter = code_search_registry.get_adapter(backend)
        success = adapter.index_project(project_path, force=force)

        return jsonify({"status": "indexing" if success else "error", "project_path": project_path, "backend": backend})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/search/status", methods=["GET"])
@require_admin
def api_search_status():
    """Api search status.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.code_search import code_search_registry

        backends = {}
        for name in code_search_registry.list_backends():
            try:
                backends[name] = code_search_registry.get_backend_info(name)
            except Exception:
                backends[name] = {"name": name, "status": "error"}

        return jsonify({"backends": backends, "default_backend": code_search_registry.DEFAULT_BACKEND})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============ ADR ENDPOINTS ============


@admin_bp.route("/api/adr", methods=["GET"])
@require_admin
def api_adr_list():
    """Api adr list.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.adr import adr_system

        status = request.args.get("status")
        category = request.args.get("category")
        limit = int(request.args.get("limit", 50))

        adrs = adr_system.list_adrs(status=status, category=category, limit=limit)

        return jsonify({"adrs": [a.to_dict() for a in adrs], "total": len(adrs)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/adr/<adr_id>", methods=["GET"])
@require_admin
def api_adr_get(adr_id):
    """Api adr get.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.adr import adr_system

        adr = adr_system.get_adr(adr_id)

        if not adr:
            return jsonify({"error": "ADR not found"}), 404

        return jsonify(adr.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/adr", methods=["POST"])
@require_admin
def api_adr_create():
    """Api adr create.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.adr import adr_system

        data = request.json

        title = data.get("title")
        category = data.get("category", "architecture")
        context = data.get("context", "")
        decision = data.get("decision", "")
        consequences = data.get("consequences", "")

        if not title:
            return jsonify({"error": "title required"}), 400

        adr = adr_system.create_adr(
            title=title,
            category=category,
            context=context,
            decision=decision,
            consequences=consequences,
            created_by=data.get("created_by", "user"),
        )

        return jsonify(adr.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/adr/<adr_id>", methods=["PUT"])
@require_admin
def api_adr_update(adr_id):
    """Api adr update.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.adr import adr_system

        data = request.json

        adr = adr_system.update_adr(adr_id, data)

        if not adr:
            return jsonify({"error": "ADR not found"}), 404

        return jsonify(adr.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/adr/<adr_id>/deprecate", methods=["POST"])
@require_admin
def api_adr_deprecate(adr_id):
    """Api adr deprecate.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.adr import adr_system

        data = request.json or {}

        replacement_id = data.get("replacement_id")

        adr = adr_system.deprecate_adr(adr_id, replacement_id)

        if not adr:
            return jsonify({"error": "ADR not found"}), 404

        return jsonify(adr.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/adr/propose", methods=["POST"])
@require_admin
def api_adr_propose():
    """Api adr propose.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.adr import adr_system

        data = request.json

        context = data.get("context", "")
        num_options = int(data.get("num_options", 3))

        proposal = adr_system.generate_proposal(context, num_options)

        return jsonify(
            {
                "question": proposal.question,
                "options": proposal.options,
                "recommended": proposal.recommended,
                "rationale": proposal.rationale,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/adr/propose/accept", methods=["POST"])
@require_admin
def api_adr_propose_accept():
    """Api adr propose accept.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.adr import adr_system

        data = request.json

        question = data.get("question", "")
        options = data.get("options", [])
        recommended = data.get("recommended", 0)
        title = data.get("title", "Proposed Decision")
        category = data.get("category", "architecture")

        from vetinari.adr import ADRProposal

        proposal = ADRProposal(question=question, options=options, recommended=recommended)

        adr = adr_system.accept_proposal(proposal, title, category)

        return jsonify(adr.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/adr/statistics", methods=["GET"])
@require_admin
def api_adr_statistics():
    """Api adr statistics.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.adr import adr_system

        stats = adr_system.get_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/adr/is-high-stakes", methods=["GET"])
@require_admin
def api_adr_is_high_stakes():
    """Api adr is high stakes.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.adr import adr_system

        category = request.args.get("category", "architecture")
        is_high_stakes = adr_system.is_high_stakes(category)
        return jsonify({"is_high_stakes": is_high_stakes, "category": category})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============ PONDER ENDPOINTS ============


@admin_bp.route("/api/ponder/choose-model", methods=["POST"])
@require_admin
def api_ponder_choose_model():
    """Api ponder choose model.

    Returns:
        The jsonify result.
    """
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


@admin_bp.route("/api/ponder/templates", methods=["GET"])
@require_admin
def api_ponder_templates():
    """Api ponder templates.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.ponder import PonderEngine

        version = request.args.get("version", "v1")
        engine = PonderEngine(template_version=version)
        templates = engine.get_template_prompts()

        return jsonify({"templates": templates, "total": len(templates), "version": version})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/ponder/models", methods=["GET"])
@require_admin
def api_ponder_models():
    """Api ponder models.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.ponder import get_available_models

        models = get_available_models()
        return jsonify({"models": models, "total": len(models)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/ponder/plan/<plan_id>", methods=["POST"])
@require_admin
def api_ponder_run_plan(plan_id):
    """Api ponder run plan.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.ponder import ponder_project_for_plan

        result = ponder_project_for_plan(plan_id)

        if not result.get("success", False):
            return jsonify(result), 400

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/ponder/plan/<plan_id>", methods=["GET"])
@require_admin
def api_ponder_get_plan(plan_id):
    """Api ponder get plan.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.ponder import get_ponder_results_for_plan

        result = get_ponder_results_for_plan(plan_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/ponder/health", methods=["GET"])
@require_admin
def api_ponder_health():
    """Api ponder health.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.ponder import get_ponder_health

        health = get_ponder_health()
        return jsonify(health)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============ RULES CONFIGURATION ENDPOINTS ============


@admin_bp.route("/api/rules", methods=["GET"])
@require_admin
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


@admin_bp.route("/api/rules/global", methods=["GET", "POST"])
@require_admin
def api_rules_global():
    """Get or set global rules.

    Returns:
        The jsonify result.
    """
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


@admin_bp.route("/api/rules/global-prompt", methods=["GET", "POST"])
@require_admin
def api_rules_global_prompt():
    """Get or set the global system prompt override.

    Returns:
        The jsonify result.
    """
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


@admin_bp.route("/api/rules/project/<project_id>", methods=["GET", "POST"])
@require_admin
def api_rules_project(project_id):
    """Get or set rules for a specific project.

    Returns:
        The jsonify result.
    """
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


@admin_bp.route("/api/rules/model/<path:model_id>", methods=["GET", "POST"])
@require_admin
def api_rules_model(model_id):
    """Get or set rules for a specific model.

    Returns:
        The jsonify result.
    """
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


# ============ IMAGE GENERATION ENDPOINTS ============


@admin_bp.route("/api/generate-image", methods=["POST"])
@require_admin
def api_generate_image():
    """Generate an image asset via the ImageGeneratorAgent.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.agents.contracts import AgentTask
        from vetinari.agents.image_generator_agent import get_image_generator_agent

        data = request.json or {}

        description = data.get("description", "")
        if not description:
            return jsonify({"error": "description required"}), 400

        agent = get_image_generator_agent(
            {
                "sd_host": current_config.get("sd_host", os.environ.get("SD_WEBUI_HOST", "http://localhost:7860")),  # noqa: VET041
                "sd_enabled": data.get("sd_enabled", True),
                "width": data.get("width", 512),
                "height": data.get("height", 512),
                "steps": data.get("steps", 20),
            }
        )

        task = AgentTask(
            task_id=f"img_{uuid.uuid4().hex[:8]}",
            description=description,
            prompt=description,
            context=data.get("context", {}),
        )

        result = agent.execute(task)
        return jsonify(
            {
                "success": result.success,
                "output": result.output,
                "errors": result.errors or [],
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/sd-status", methods=["GET"])
@require_admin
def api_sd_status():
    """Check Stable Diffusion WebUI connection status.

    Returns:
        Tuple of results.
    """
    try:
        import requests as _req

        host = current_config.get("sd_host", os.environ.get("SD_WEBUI_HOST", "http://localhost:7860"))  # noqa: VET041
        resp = _req.get(f"{host}/sdapi/v1/options", timeout=5)
        if resp.status_code == 200:
            return jsonify({"status": "connected", "host": host})
        return jsonify({"status": "error", "code": resp.status_code}), 200
    except Exception as e:
        return jsonify({"status": "disconnected", "error": str(e)}), 200


# ============ TRAINING ENDPOINTS ============


@admin_bp.route("/api/training/stats", methods=["GET"])
@require_admin
def api_training_stats():
    """Get training data statistics.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.learning.training_data import get_training_collector

        collector = get_training_collector()
        stats = collector.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e), "total_records": 0}), 200


@admin_bp.route("/api/training/export", methods=["POST"])
@require_admin
def api_training_export():
    """Export training data for a given format.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.learning.training_data import get_training_collector

        data = request.json or {}
        export_format = data.get("format", "sft")  # sft | dpo | prompts
        collector = get_training_collector()

        if export_format == "dpo":
            dataset = collector.export_dpo_dataset()
        elif export_format == "prompts":
            dataset = collector.export_prompt_variants()
        else:
            dataset = collector.export_sft_dataset()

        return jsonify(
            {
                "format": export_format,
                "count": len(dataset),
                "data": dataset[:100],  # Return first 100 for preview
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/training/start", methods=["POST"])
@require_admin
def api_training_start():
    """Start a training run (async).

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.training.pipeline import TrainingPipeline

        data = request.json or {}

        tier = data.get("tier", "general")  # general|coding|research|review|individual
        model_id = data.get("model_id", "")
        min_quality = float(data.get("min_quality", 0.7))

        def _run():
            try:
                pipeline = TrainingPipeline()
                pipeline.run(
                    base_model=model_id or "qwen2.5-coder-7b",
                    training_type=tier,
                    min_quality_score=min_quality,
                )
                logger.info("Training run completed: tier=%s, model=%s", tier, model_id)
            except Exception as te:
                logger.error("Training run failed: %s", te)

        import threading as _t

        _t.Thread(target=_run, daemon=True).start()

        return jsonify(
            {"status": "started", "tier": tier, "model_id": model_id, "message": "Training run started in background"}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
