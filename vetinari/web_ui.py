import os
import json
import logging
import yaml
import threading
import time
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, render_template_string, jsonify, request, send_from_directory
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    _APSCHEDULER_AVAILABLE = True
except ImportError:
    _APSCHEDULER_AVAILABLE = False
    BackgroundScheduler = None

# Import shared state from the web package
from vetinari.web.shared import (  # noqa: F401
    PROJECT_ROOT,
    current_config,
    ENABLE_EXTERNAL_DISCOVERY,
    get_orchestrator,
    _register_project_task,
    _cancel_project_task,
    _get_sse_queue,
    _push_sse_event,
    _cleanup_project_state,
    _get_models_cached,
    refresh_model_cache,
    trigger_light_search,
    _is_admin_user,
    _project_external_model_enabled,
)

logger = logging.getLogger(__name__)

app = Flask(__name__,
    template_folder=str(PROJECT_ROOT / 'ui' / 'templates'),
    static_folder=str(PROJECT_ROOT / 'ui' / 'static'))

# Model search scheduler (only if APScheduler is installed)
scheduler = None
if _APSCHEDULER_AVAILABLE:
    try:
        scheduler = BackgroundScheduler()
        scheduler.start()
    except Exception as _sched_err:
        logging.warning(f"[Vetinari] Could not start background scheduler: {_sched_err}")

if scheduler is not None:
    try:
        scheduler.add_job(
            func=refresh_model_cache,
            trigger="interval",
            days=30,
            id="monthly_model_refresh",
            name="Monthly model cache refresh"
        )
    except Exception as _job_err:
        logging.warning(f"[Vetinari] Could not add scheduler job: {_job_err}")

# Read HTML template
template_path = PROJECT_ROOT / 'ui' / 'templates' / 'index.html'
if template_path.exists():
    with open(template_path, 'r', encoding='utf-8') as f:
        HTML_TEMPLATE = f.read()
else:
    HTML_TEMPLATE = "<html><body><h1>Template not found</h1></body></html>"

# Serve static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(str(PROJECT_ROOT / 'ui' / 'static'), filename)

# Main page
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

# API: Get system status
@app.route('/api/status')
def api_status():
    return jsonify({
        "status": "running",
        "host": current_config["host"],
        "config_path": current_config["config_path"],
        "api_token": "***" if current_config.get("api_token") else "",
        "default_models": current_config.get("default_models", []),
        "fallback_models": current_config.get("fallback_models", []),
        "uncensored_fallback_models": current_config.get("uncensored_fallback_models", []),
        "memory_budget_gb": current_config.get("memory_budget_gb", 32),
        "active_model_id": current_config.get("active_model_id")
    })

@app.route('/api/token-stats')
def api_token_stats():
    """Return token usage statistics from the analytics system."""
    stats = {
        "total_tokens_used": 0,
        "total_cost_usd": 0.0,
        "by_model": {},
        "by_provider": {},
        "session_requests": 0,
    }
    try:
        from vetinari.telemetry import get_telemetry_collector
        tel = get_telemetry_collector()
        if hasattr(tel, "get_summary"):
            summary = tel.get_summary()
            if summary:
                stats.update(summary)
    except Exception:
        pass
    try:
        from vetinari.analytics.cost import get_cost_tracker
        cost_summary = get_cost_tracker().get_summary()
        if cost_summary:
            stats["total_cost_usd"] = cost_summary.get("total_cost_usd", 0.0)
            stats["by_model"] = cost_summary.get("by_model", {})
    except Exception:
        pass
    return jsonify(stats)


# API: Global search across projects and outputs
@app.route('/api/search', methods=['GET'])
def api_global_search():
    """Search across project names, descriptions, and outputs."""
    query = request.args.get('q', '').strip().lower()
    if not query:
        return jsonify({"results": [], "query": ""})

    results = []
    projects_dir = PROJECT_ROOT / 'projects'
    if projects_dir.exists():
        for proj_dir in projects_dir.iterdir():
            if not proj_dir.is_dir():
                continue
            config_path = proj_dir / 'project.yaml'
            if not config_path.exists():
                continue
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
                name = config.get('project_name', proj_dir.name)
                desc = config.get('description', '')
                if query in name.lower() or query in desc.lower():
                    results.append({
                        "type": "project",
                        "id": proj_dir.name,
                        "name": name,
                        "description": desc[:100],
                        "status": config.get('status', 'unknown'),
                    })
                # Search task outputs
                for task_dir in (proj_dir / 'outputs').glob('*/output.txt') if (proj_dir / 'outputs').exists() else []:
                    try:
                        content = task_dir.read_text(encoding='utf-8', errors='ignore')
                        if query in content.lower():
                            results.append({
                                "type": "output",
                                "project_id": proj_dir.name,
                                "task_id": task_dir.parent.name,
                                "preview": content[:150],
                            })
                    except Exception:
                        pass
            except Exception:
                pass

    return jsonify({"results": results[:20], "query": query, "total": len(results)})

# API: Get available models (uses TTL cache — does NOT block on every request)
@app.route('/api/models')
def api_models():
    try:
        models = _get_models_cached()
        return jsonify({"models": models, "cached": True, "count": len(models)})
    except Exception as e:
        return jsonify({"error": str(e), "models": []}), 500

# API: Force-refresh model discovery, bypassing the cache
@app.route('/api/models/refresh', methods=['POST', 'GET'])
def api_models_refresh():
    try:
        models = _get_models_cached(force=True)
        return jsonify({"models": models, "cached": False, "count": len(models)})
    except Exception as e:
        return jsonify({"error": str(e), "models": []}), 500

# API: Score models for a task
@app.route('/api/score-models', methods=['POST'])
def api_score_models():
    try:
        data = request.json
        task_description = data.get('task_description', '')

        orb = get_orchestrator()
        orb.model_pool.discover_models()

        if not orb.model_pool.models:
            return jsonify({"error": "No models available"}), 400

        task_lower = task_description.lower()

        # Determine required capabilities
        required_capabilities = []
        if any(word in task_lower for word in ['code', 'implement', 'build', 'create', 'python', 'javascript', 'script', 'api', 'web', 'function', 'class']):
            required_capabilities.append('code_gen')
        if any(word in task_lower for word in ['document', 'readme', 'explain', 'comment', 'docs', 'description']):
            required_capabilities.append('docs')
        if any(word in task_lower for word in ['chat', 'conversation', 'message', 'respond', 'reply']):
            required_capabilities.append('chat')

        # Score each model
        scored_models = []
        for m in orb.model_pool.models:
            model_name = m.get('name', '')
            capabilities = m.get('capabilities', [])

            score = 0
            capability_matches = []
            for req in required_capabilities:
                if req in capabilities:
                    score += 10
                    capability_matches.append(req)

            memory_gb = m.get('memory_gb', 99)
            if memory_gb <= 2:
                score += 2
            elif memory_gb <= 8:
                score += 1

            scored_models.append({
                "name": model_name,
                "score": score,
                "capabilities": capabilities,
                "memory_gb": memory_gb,
                "matches": capability_matches
            })

        # Sort by score descending
        scored_models.sort(key=lambda x: x['score'], reverse=True)

        return jsonify({"models": scored_models})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API: Get tasks
@app.route('/api/tasks')
def api_tasks():
    try:
        config_path = Path(PROJECT_ROOT) / current_config["config_path"]
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            tasks = config.get("tasks", [])
            return jsonify({"tasks": tasks})
        return jsonify({"tasks": []})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API: Run a task
@app.route('/api/run-task', methods=['POST'])
def api_run_task():
    data = request.json
    task_id = data.get('task_id')

    if not task_id:
        return jsonify({"error": "task_id is required"}), 400

    try:
        orb = get_orchestrator()

        def run_task():
            orb.run_task(task_id)

        thread = threading.Thread(target=run_task)
        thread.start()

        return jsonify({"status": "started", "task_id": task_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API: Run all tasks
@app.route('/api/run-all', methods=['POST'])
def api_run_all():
    try:
        orb = get_orchestrator()

        def run_all():
            orb.run_all()

        thread = threading.Thread(target=run_all)
        thread.start()

        return jsonify({"status": "started"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API: Run custom prompt
@app.route('/api/run-prompt', methods=['POST'])
def api_run_prompt():
    data = request.json
    prompt = data.get('prompt', '')
    model = data.get('model', '')
    system_prompt = data.get('system_prompt', 'You are a helpful coding assistant.')

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    try:
        orb = get_orchestrator()

        # Use the model if specified, otherwise use first available
        if not model:
            if orb.model_pool.models:
                model = orb.model_pool.models[0].get('name', '')
            else:
                return jsonify({"error": "No models available"}), 400

        # Run the prompt directly using the adapter
        result = orb.adapter.chat(model, system_prompt, prompt)

        # Save output
        task_id = "custom_" + uuid.uuid4().hex[:12]
        output_path = PROJECT_ROOT / 'outputs' / task_id
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / 'output.txt').write_text(result.get('output', ''), encoding='utf-8')

        return jsonify({
            "status": "completed",
            "task_id": task_id,
            "response": result.get('output', ''),
            "model": model,
            "latency_ms": result.get('latency_ms', 0)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API: Create a plan using the planning engine
@app.route('/api/plan', methods=['POST'])
def api_create_plan():
    data = request.json
    goal = data.get('goal', '')
    system_prompt = data.get('system_prompt', 'You are a helpful coding assistant.')

    if not goal:
        return jsonify({"error": "goal is required"}), 400

    try:
        orb = get_orchestrator()

        # Discover models first
        orb.model_pool.discover_models()

        # Get available models
        available_models = orb.model_pool.models
        if not available_models:
            return jsonify({"error": "No models available"}), 400

        # Get default and fallback models from config
        default_models = current_config.get("default_models", [])
        fallback_models = current_config.get("fallback_models", [])

        # Initialize planning engine
        from vetinari.planning_engine import PlanningEngine
        planner = PlanningEngine(default_models=default_models, fallback_models=fallback_models)

        # Create the plan
        plan = planner.plan(goal, system_prompt, available_models)

        return jsonify({
            "status": "ok",
            "plan": plan.to_dict()
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route('/api/system-prompts')
def api_system_prompts():
    try:
        prompts_dir = PROJECT_ROOT / 'system_prompts'
        prompts_dir.mkdir(parents=True, exist_ok=True)

        prompts = []
        for f in prompts_dir.glob('*.txt'):
            prompts.append({
                "name": f.stem,
                "content": f.read_text(encoding='utf-8').strip()
            })

        return jsonify({"prompts": prompts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/system-prompts', methods=['POST'])
def api_save_system_prompt():
    try:
        data = request.json
        name = data.get('name', '')
        content = data.get('content', '')

        if not name:
            return jsonify({"error": "name is required"}), 400

        prompts_dir = PROJECT_ROOT / 'system_prompts'
        prompts_dir.mkdir(parents=True, exist_ok=True)

        prompt_file = prompts_dir / f"{name}.txt"
        prompt_file.write_text(content, encoding='utf-8')

        return jsonify({"status": "saved", "name": name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/system-prompts/<name>', methods=['DELETE'])
def api_delete_system_prompt(name):
    try:
        prompt_file = PROJECT_ROOT / 'system_prompts' / f"{name}.txt"
        if prompt_file.exists():
            prompt_file.unlink()
        return jsonify({"status": "deleted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API: Get output for a specific task (from projects)
@app.route('/api/output/<task_id>')
def api_output(task_id):
    # Try to find the task in any project
    projects_dir = PROJECT_ROOT / 'projects'
    if projects_dir.exists():
        for p in projects_dir.iterdir():
            if p.is_dir():
                output_path = p / 'outputs' / task_id / 'output.txt'
                if output_path.exists():
                    content = output_path.read_text(encoding='utf-8')
                    return jsonify({"output": content, "task_id": task_id, "project_id": p.name})

    # Fallback to old location
    output_path = PROJECT_ROOT / 'outputs' / task_id / 'output.txt'
    if output_path.exists():
        content = output_path.read_text(encoding='utf-8')
        return jsonify({"output": content, "task_id": task_id})
    return jsonify({"output": "", "task_id": task_id})

# API: Get all tasks across all projects for output dropdown
@app.route('/api/all-tasks')
def api_all_tasks():
    try:
        projects_dir = PROJECT_ROOT / 'projects'
        all_tasks = []

        if projects_dir.exists():
            for p in sorted(projects_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                if p.is_dir():
                    config_file = p / 'project.yaml'
                    if config_file.exists():
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config = yaml.safe_load(f) or {}
                            tasks = config.get("tasks", [])
                            for t in tasks:
                                task_id = t.get("id", "")
                                all_tasks.append({
                                    "project_id": p.name,
                                    "project_name": config.get("project_name", p.name),
                                    "task_id": task_id,
                                    "description": t.get("description", ""),
                                    "assigned_model": t.get("assigned_model_id", "")
                                })

        return jsonify({"tasks": all_tasks})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/discover')
def api_discover():
    try:
        models = _get_models_cached(force=True)
        return jsonify({
            "discovered": len(models),
            "models": models,
            "status": "ok"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API: Update configuration
@app.route('/api/config', methods=['POST'])
def api_config():
    global orchestrator
    data = request.json

    host = None
    api_token = None

    if 'host' in data:
        host = data['host']
        current_config["host"] = host
    if 'config_path' in data:
        current_config["config_path"] = data['config_path']
    if 'api_token' in data:
        api_token = data['api_token']
        current_config["api_token"] = api_token

    # Reset orchestrator and invalidate model cache
    orchestrator = None
    global _models_cache_ts
    _models_cache_ts = 0.0   # Force fresh discovery on next request

    return jsonify({"status": "updated", "config": current_config})

# API: Check for upgrades
@app.route('/api/upgrade-check')
def api_upgrade_check():
    try:
        orb = get_orchestrator()
        upgrades = orb.upgrader.check_for_upgrades()
        return jsonify({"upgrades": upgrades})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API: Get workflow tree (returns projects with tasks for workflow view)
@app.route('/api/workflow')
def api_workflow():
    try:
        include_archived = request.args.get('include_archived', 'false').lower() == 'true'
        search_query = request.args.get('search', '').lower()
        status_filter = request.args.get('status', '').lower()

        projects_dir = PROJECT_ROOT / 'projects'
        projects = []

        if projects_dir.exists():
            for p in sorted(projects_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                if p.is_dir():
                    config_file = p / 'project.yaml'
                    conv_file = p / 'conversation.json'
                    outputs_dir = p / 'outputs'

                    project_data = {
                        "id": p.name,
                        "name": p.name,
                        "path": str(p),
                        "tasks": [],
                        "status": "unknown",
                        "archived": False
                    }

                    if config_file.exists():
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config = yaml.safe_load(f) or {}
                            project_data["name"] = config.get("project_name", p.name)
                            project_data["description"] = config.get("description", "")
                            project_data["goal"] = config.get("high_level_goal", "")
                            project_data["model"] = config.get("model", "")
                            project_data["active_model_id"] = config.get("active_model_id", "")
                            project_data["status"] = config.get("status", "unknown")
                            project_data["warnings"] = config.get("warnings", [])
                            project_data["archived"] = config.get("archived", False)

                            # Skip archived unless requested
                            if project_data["archived"] and not include_archived:
                                continue

                            # Apply search filter
                            if search_query:
                                searchable = f"{project_data['name']} {project_data['description']} {project_data['goal']}".lower()
                                if search_query not in searchable:
                                    continue

                            # Apply status filter
                            if status_filter and project_data["status"] != status_filter:
                                continue

                            planned_tasks = config.get("tasks", [])

                            completed_tasks = set()
                            if outputs_dir.exists():
                                for task_dir in outputs_dir.iterdir():
                                    if task_dir.is_dir():
                                        output_file = task_dir / 'output.txt'
                                        if output_file.exists():
                                            completed_tasks.add(task_dir.name)

                            for t in planned_tasks:
                                task_id = t.get("id", "")
                                project_data["tasks"].append({
                                    "id": task_id,
                                    "description": t.get("description", ""),
                                    "assigned_model": t.get("assigned_model_id", ""),
                                    "status": "completed" if task_id in completed_tasks else ("running" if project_data["status"] == "running" else "pending"),
                                    "model_override": t.get("model_override", "")
                                })

                    if conv_file.exists():
                        with open(conv_file, 'r', encoding='utf-8') as f:
                            conv = json.load(f)
                            project_data["message_count"] = len(conv)

                    projects.append(project_data)

        return jsonify({"projects": projects})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API: Get model configuration (defaults and fallbacks)
@app.route('/api/model-config')
def api_model_config():
    return jsonify({
        "default_models": current_config.get("default_models", []),
        "fallback_models": current_config.get("fallback_models", []),
        "uncensored_fallback_models": current_config.get("uncensored_fallback_models", []),
        "memory_budget_gb": current_config.get("memory_budget_gb", 32)
    })

# API: Update model configuration
@app.route('/api/model-config', methods=['POST'])
def api_update_model_config():
    data = request.json

    if 'default_models' in data:
        current_config["default_models"] = data['default_models']
    if 'fallback_models' in data:
        current_config["fallback_models"] = data['fallback_models']
    if 'uncensored_fallback_models' in data:
        current_config["uncensored_fallback_models"] = data['uncensored_fallback_models']
    if 'memory_budget_gb' in data:
        current_config["memory_budget_gb"] = int(data['memory_budget_gb'])

    return jsonify({
        "status": "updated",
        "default_models": current_config.get("default_models", []),
        "fallback_models": current_config.get("fallback_models", []),
        "uncensored_fallback_models": current_config.get("uncensored_fallback_models", []),
        "memory_budget_gb": current_config.get("memory_budget_gb", 32)
    })

# API: Swap to a different model (per-project)
@app.route('/api/swap-model', methods=['POST'])
def api_swap_model():
    data = request.json
    project_id = data.get('project_id')
    new_model = data.get('model_id', '')

    if not new_model:
        return jsonify({"error": "model_id is required"}), 400

    # If project_id provided, update that project's active model
    if project_id:
        project_dir = PROJECT_ROOT / 'projects' / project_id
        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404

        config_file = project_dir / 'project.yaml'
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}

        config['active_model_id'] = new_model

        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)

        return jsonify({
            "status": "swapped",
            "project_id": project_id,
            "active_model_id": new_model
        })

    # Otherwise, update global active model
    current_config["active_model_id"] = new_model

    return jsonify({
        "status": "swapped",
        "active_model_id": new_model
    })

# ---------------------------------------------------------------------------
# Register Flask Blueprints
# ---------------------------------------------------------------------------
from vetinari.web.projects_bp import projects_bp
from vetinari.web.plans_bp import plans_bp
from vetinari.web.admin_bp import admin_bp
from vetinari.web.preferences import preferences_bp

app.register_blueprint(projects_bp)
app.register_blueprint(plans_bp)
app.register_blueprint(admin_bp)
app.register_blueprint(preferences_bp)

# Phase 8: Learning & Analytics API blueprints
try:
    from vetinari.web.learning_api import learning_bp
    app.register_blueprint(learning_bp)
    logger.info("Learning API blueprint registered")
except Exception as e:
    logger.warning(f"Learning API blueprint not available: {e}")

try:
    from vetinari.web.analytics_api import analytics_bp
    app.register_blueprint(analytics_bp)
    logger.info("Analytics API blueprint registered")
except Exception as e:
    logger.warning(f"Analytics API blueprint not available: {e}")

# Register Plan Mode API endpoints
try:
    from vetinari.plan_api import register_plan_api
    register_plan_api(app)
    logger.info("Plan Mode API registered successfully")
except ImportError as e:
    logger.warning(f"Plan Mode API not available: {e}")
except Exception as e:
    logger.warning(f"Failed to register Plan Mode API: {e}")


if __name__ == '__main__':
    _debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
    _port = int(os.environ.get("VETINARI_WEB_PORT", 5000))
    _host = os.environ.get("VETINARI_WEB_HOST", "0.0.0.0")
    app.run(host=_host, port=_port, debug=_debug)
