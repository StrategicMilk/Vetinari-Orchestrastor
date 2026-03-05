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

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load .env file if present (before reading env vars)
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    try:
        for _line in _env_file.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                _k, _v = _k.strip(), _v.strip()
                if _k and _v and _k not in os.environ:
                    os.environ[_k] = _v
    except Exception:
        pass

app = Flask(__name__,
    template_folder=str(PROJECT_ROOT / 'ui' / 'templates'),
    static_folder=str(PROJECT_ROOT / 'ui' / 'static'))

# CORS support — restrict origins via VETINARI_CORS_ORIGINS env var
_CORS_ORIGINS = os.environ.get("VETINARI_CORS_ORIGINS", "*")

@app.after_request
def _add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = _CORS_ORIGINS
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

# Global state
orchestrator = None

# ---------------------------------------------------------------------------
# Task cancellation registry
# ---------------------------------------------------------------------------
_cancel_flags: dict = {}  # project_id -> threading.Event


def _register_project_task(project_id: str) -> "threading.Event":
    import threading as _threading
    flag = _threading.Event()
    _cancel_flags[project_id] = flag
    return flag


def _cancel_project_task(project_id: str) -> bool:
    flag = _cancel_flags.get(project_id)
    if flag:
        flag.set()
        return True
    return False

# ---------------------------------------------------------------------------
# SSE stream registry (project_id -> queue of messages)
# ---------------------------------------------------------------------------
import queue as _queue
_sse_streams: dict = {}  # project_id -> queue.Queue


def _get_sse_queue(project_id: str):
    if project_id not in _sse_streams:
        _sse_streams[project_id] = _queue.Queue(maxsize=200)
    return _sse_streams[project_id]


def _push_sse_event(project_id: str, event_type: str, data: dict) -> None:
    """Push an SSE event to all listeners for a project."""
    import json as _json
    q = _sse_streams.get(project_id)
    if q:
        try:
            q.put_nowait({"event": event_type, "data": _json.dumps(data)})
        except Exception:
            pass  # Queue full — drop event

# ---------------------------------------------------------------------------
# Model discovery cache  (avoids blocking the UI on every request)
# ---------------------------------------------------------------------------
_models_cache: list = []          # Last successful list of model dicts
_models_cache_ts: float = 0.0     # Unix timestamp of last successful discovery
_MODELS_CACHE_TTL: float = 60.0   # Seconds before cache is considered stale

def _get_models_cached(force: bool = False) -> list:
    """Return models from cache if fresh, otherwise run discovery and cache result."""
    global _models_cache, _models_cache_ts
    now = time.time()
    if not force and _models_cache and (now - _models_cache_ts) < _MODELS_CACHE_TTL:
        return _models_cache
    try:
        orb = get_orchestrator()
        orb.model_pool.discover_models()
        fresh = [
            {
                "id": m.get("id", ""),
                "name": m.get("name", m.get("id", "")),
                "capabilities": m.get("capabilities", []),
                "context_len": m.get("context_len", 0),
                "memory_gb": m.get("memory_gb", 0),
                "version": m.get("version", ""),
            }
            for m in orb.model_pool.models
        ]
        if fresh:                          # Only update cache on non-empty result
            _models_cache = fresh
            _models_cache_ts = now
        return fresh if fresh else _models_cache   # Return stale if discovery empty
    except Exception as e:
        import logging as _log
        _log.warning(f"[web_ui] Model discovery failed: {e}")
        return _models_cache               # Return stale cache on error

current_config = {
    "host": os.environ.get("LM_STUDIO_HOST", "http://localhost:1234"),
    "config_path": "manifest/vetinari.yaml",
    "api_token": os.environ.get("LM_STUDIO_API_TOKEN") or os.environ.get("VETINARI_API_TOKEN", ""),
    "default_models": ["qwen3-coder-next", "qwen3-30b-a3b-gemini-pro-high-reasoning-2507-hi8"],
    "fallback_models": ["llama-3.2-1b-instruct", "qwen2.5-0.5b-instruct", "devstral-small-2505-deepseek-v3.2-speciale-distill"],
    "uncensored_fallback_models": ["qwen3-vl-32b-gemini-heretic-uncensored-thinking", "glm-4.7-flash-uncensored-heretic-neo-code-imatrix-max"],
    "memory_budget_gb": 48,
    "active_model_id": None
}

# Global feature flag for external model discovery (default enabled)
ENABLE_EXTERNAL_DISCOVERY = str(os.environ.get("ENABLE_EXTERNAL_DISCOVERY", "true")).lower() in ("1", "true", "yes")

# Model search scheduler (only if APScheduler is installed)
scheduler = None
if _APSCHEDULER_AVAILABLE:
    try:
        scheduler = BackgroundScheduler()
        scheduler.start()
    except Exception as _sched_err:
        logging.warning(f"[Vetinari] Could not start background scheduler: {_sched_err}")

def refresh_model_cache():
    try:
        from vetinari.model_search import ModelSearchEngine
        search_engine = ModelSearchEngine()
        search_engine.refresh_all_caches()
        print(f"[Vetinari] Model cache refreshed at {datetime.now()}")
    except Exception as e:
        print(f"[Vetinari] Error refreshing model cache: {e}")

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

def trigger_light_search(project_id: str, task_description: str):
    try:
        from vetinari.model_search import ModelSearchEngine
        search_engine = ModelSearchEngine()
        
        lm_models = []
        try:
            from vetinari.model_pool import ModelPool
            model_pool = ModelPool(current_config, current_config.get("host", "http://localhost:1234"))
            model_pool.discover_models()
            lm_models = model_pool.list_models()
        except Exception:
            pass
        
        candidates = search_engine.search_for_task(task_description, lm_models)
        return candidates
    except Exception as e:
        print(f"[Vetinari] Light search error: {e}")
        return []

def get_orchestrator():
    global orchestrator
    if orchestrator is None:
        # Import here to avoid circular imports
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from vetinari.orchestrator import Orchestrator
        api_token = current_config.get("api_token") or None
        orchestrator = Orchestrator(current_config["config_path"], current_config["host"], api_token=api_token)
    return orchestrator

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


# API: Server-Sent Events stream for real-time task updates
@app.route('/api/project/<project_id>/stream')
def api_project_stream(project_id):
    """SSE endpoint — subscribe to real-time events for a project."""
    from flask import Response, stream_with_context
    import json as _json

    def generate():
        q = _get_sse_queue(project_id)
        # Send an initial connected event
        yield f"data: {_json.dumps({'type': 'connected', 'project_id': project_id})}\n\n"
        while True:
            try:
                msg = q.get(timeout=25)
                if msg is None:  # Sentinel: stream closed
                    yield f"data: {_json.dumps({'type': 'done'})}\n\n"
                    break
                yield f"event: {msg['event']}\ndata: {msg['data']}\n\n"
            except Exception:
                # Heartbeat to keep connection alive
                yield f"data: {_json.dumps({'type': 'heartbeat'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# API: Cancel a running project task
@app.route('/api/project/<project_id>/cancel', methods=['POST'])
def api_cancel_project(project_id):
    """Cancel a running project execution."""
    cancelled = _cancel_project_task(project_id)
    _push_sse_event(project_id, "cancelled", {"project_id": project_id, "status": "cancelled"})

    # Update project status in config
    try:
        project_dir = PROJECT_ROOT / 'projects' / project_id
        config_path = project_dir / 'project.yaml'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                project_config = yaml.safe_load(f) or {}
            project_config['status'] = 'cancelled'
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(project_config, f)
    except Exception:
        pass

    return jsonify({"status": "cancelled" if cancelled else "not_found", "project_id": project_id})


# API: Token usage statistics
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

# API: Create new project from prompt - uses planning engine to create and execute tasks
@app.route('/api/new-project', methods=['POST'])
def api_new_project():
    data = request.json
    goal = data.get('goal', '')
    model = data.get('model', '')
    system_prompt = data.get('system_prompt', 'You are a helpful coding assistant.')
    auto_run = data.get('auto_run', False)  # Default False: show plan for approval before running
    project_name = data.get('project_name', '')
    project_rules = data.get('project_rules', '')
    required_features = data.get('required_features', [])
    things_to_avoid = data.get('things_to_avoid', [])
    expected_outputs = data.get('expected_outputs', [])
    tech_stack = data.get('tech_stack', '')
    platforms = data.get('platforms', [])
    priority = data.get('priority', 'quality')

    if not goal:
        return jsonify({"error": "goal is required"}), 400
    
    try:
        orb = get_orchestrator()
        
        # Discover models first
        orb.model_pool.discover_models()
        
        # Get available models with their capabilities
        available_models = orb.model_pool.models
        if not available_models:
            return jsonify({"error": "No models available"}), 400
        
        # Get config settings
        default_models = current_config.get("default_models", [])
        fallback_models = current_config.get("fallback_models", [])
        uncensored_fallback_models = current_config.get("uncensored_fallback_models", [])
        memory_budget_gb = current_config.get("memory_budget_gb", 32)
        
        # Initialize planning engine with memory budget and fallbacks
        from vetinari.planning_engine import PlanningEngine
        planner = PlanningEngine(
            default_models=default_models, 
            fallback_models=fallback_models,
            uncensored_fallback_models=uncensored_fallback_models,
            memory_budget_gb=memory_budget_gb
        )
        
        # Create the plan
        plan = planner.plan(goal, system_prompt, available_models, planning_model=model or None)
        
        # Convert plan tasks to dict format
        tasks = [t.to_dict() for t in plan.tasks]
        
        # Save tasks to config
        project_dir = PROJECT_ROOT / 'projects' / f'project_{uuid.uuid4().hex[:12]}'
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Create project config
        planning_model = plan.notes.split(": ")[-1] if plan.notes else (model or available_models[0].get('name', ''))
        project_config = {
            "project_name": project_name or goal[:50],
            "description": goal,
            "high_level_goal": goal,
            "goal": goal,
            "tasks": tasks,
            "model": planning_model,
            "active_model_id": model or planning_model,
            "plan_notes": plan.notes,
            "warnings": plan.warnings,
            "system_prompt": system_prompt,
            "project_rules": project_rules,
            "required_features": required_features,
            "things_to_avoid": things_to_avoid,
            "expected_outputs": expected_outputs,
            "tech_stack": tech_stack,
            "platforms": platforms,
            "priority": priority,
            "status": "planned" if not auto_run else "running",
            "archived": False
        }

        # Save project rules to RulesManager for injection into agent prompts
        if project_rules:
            try:
                from vetinari.rules_manager import get_rules_manager
                rm = get_rules_manager()
                rules_list = [r.strip() for r in project_rules.splitlines() if r.strip()]
                rm.set_project_rules(project_dir.name, rules_list)
            except Exception as _re:
                logging.warning(f"Could not save project rules: {_re}")
        
        config_path = project_dir / 'project.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(project_config, f)
        
        # Build a comprehensive system prompt for the AI agent
        agent_system_prompt = system_prompt or """You are Vetinari, an autonomous AI orchestration agent. Your role is to:
1. Understand the user's goal
2. Plan the best workflow to accomplish it
3. Break down the work into clear, executable tasks
4. Assign the best available model to each task based on its capabilities
5. Execute tasks and compile results

When the user provides a goal:
- Analyze what they want to accomplish
- Identify the best approach
- Create a detailed plan with numbered tasks
- For each task, specify: description, inputs, outputs, and dependencies
- Respond directly to the user with your analysis and plan

Be concise but thorough. Focus on creating actionable, clear tasks."""
        
        # Call the model with the user's actual goal
        planning_model_id = (model or plan.notes.split(": ")[-1]) if plan.notes else (model or (available_models[0].get('name', '') if available_models else ''))
        
        model_response = ""
        try:
            # Send the user's actual goal to the model and get its response
            result = orb.adapter.chat(planning_model_id, agent_system_prompt, goal)
            model_response = result.get('output', '')
            print(f"[Vetinari] Model response received: {len(model_response)} chars")
        except Exception as e:
            print(f"Error getting model response: {e}")
            model_response = ""
        
        # Build the conversation with model's actual response
        if model_response and len(model_response) > 10:
            task_plan = model_response
        else:
            # Fallback to structured plan if model response is empty/short
            task_plan = f"I've analyzed your goal and created {len(tasks)} tasks to complete it.\n\n"
            task_plan += "Warnings:\n" + "\n".join([f"- {w}" for w in plan.warnings]) + "\n\n" if plan.warnings else ""
            task_plan += "Plan:\n" + "\n".join([f"- {t['id']}: {t['description']} (using: {t['assigned_model_id']})" for t in tasks])
        
        conversation = [
            {"role": "user", "content": goal},
            {"role": "assistant", "content": task_plan}
        ]
        
        # Save conversation to JSON file
        conv_file = project_dir / 'conversation.json'
        with open(conv_file, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, indent=2)
        
        # If auto_run is True, run tasks in background thread
        if auto_run:
            _proj_id = project_dir.name
            _cancel_event = _register_project_task(_proj_id)
            _get_sse_queue(_proj_id)  # Pre-create queue

            def run_tasks_background():
                try:
                    # Update config to show running
                    project_config["status"] = "running"
                    with open(config_path, 'w', encoding='utf-8') as f:
                        yaml.dump(project_config, f)

                    _push_sse_event(_proj_id, "status", {"status": "running", "total_tasks": len(tasks)})

                    # Build agent context once — all tasks route through the full agent framework,
                    # enabling quality scoring, feedback loops, and Thompson sampling.
                    from vetinari.agent_dispatcher import dispatch_task, build_agent_context
                    agent_context = build_agent_context(orb)

                    results = []
                    task_outputs_text = []

                    for idx, task in enumerate(tasks):
                        # Check cancel flag
                        if _cancel_event.is_set():
                            _push_sse_event(_proj_id, "cancelled", {"message": "Cancelled by user"})
                            break

                        task_id = task['id']
                        task_model = task['assigned_model_id'] or model or (available_models[0].get('name', '') if available_models else '')

                        _push_sse_event(_proj_id, "task_start", {
                            "task_id": task_id,
                            "task_index": idx,
                            "total": len(tasks),
                            "description": task['description'],
                            "model": task_model,
                        })

                        # Route task through the appropriate specialized agent
                        # (previously called orb.adapter.chat() directly, bypassing the framework)
                        task['output_dir'] = str(project_dir / 'outputs' / task_id / 'generated')
                        task_output, task_success, task_meta = dispatch_task(
                            task_dict=task,
                            context=agent_context,
                            model_id=task_model or None,
                        )
                        tokens_used = task_meta.get('tokens_used', 0)

                        # Save task output text
                        task_output_dir = project_dir / 'outputs' / task_id
                        task_output_dir.mkdir(parents=True, exist_ok=True)
                        (task_output_dir / 'output.txt').write_text(task_output, encoding='utf-8')

                        # If the agent wrote files to output_dir, also parse any remaining
                        # code blocks from the text output for backward compatibility
                        generated_dir = task_output_dir / 'generated'
                        if not generated_dir.exists() or not any(generated_dir.iterdir()):
                            code_blocks = orb.executor._parse_code_blocks(task_output)
                            if code_blocks:
                                generated_dir.mkdir(parents=True, exist_ok=True)
                                for filename, code in code_blocks.items():
                                    filepath = generated_dir / filename
                                    filepath.write_text(code, encoding='utf-8')
                                    print(f"[Vetinari] Written: {filepath}")

                        task_status = "completed" if task_success else "failed"
                        results.append({
                            "task_id": task_id,
                            "model_used": task_model,
                            "agent_type": task_meta.get("agent_type", "builder"),
                            "agent_name": task_meta.get("agent_name", "Builder"),
                            "status": task_status,
                            "output": task_output,
                        })

                        task_outputs_text.append(
                            f"=== Task {task_id} [{task_meta.get('agent_name', 'Builder')}] "
                            f"(using {task_model}): {task['description']} ===\n\n{task_output}"
                        )

                        _push_sse_event(_proj_id, "task_complete", {
                            "task_id": task_id,
                            "task_index": idx,
                            "total": len(tasks),
                            "status": task_status,
                            "agent_type": task_meta.get("agent_type", "builder"),
                            "tokens_used": tokens_used,
                            "output_length": len(task_output),
                        })
                    
                    # Add results to conversation
                    results_text = "Tasks completed! Here are the results:\n\n" + "="*50 + "\n\n".join(task_outputs_text)
                    conversation.append({"role": "assistant", "content": results_text})
                    
                    with open(conv_file, 'w', encoding='utf-8') as f:
                        json.dump(conversation, f, indent=2)
                    
                    # Update config to show completed
                    project_config["status"] = "completed"
                    with open(config_path, 'w', encoding='utf-8') as f:
                        yaml.dump(project_config, f)

                    _push_sse_event(_proj_id, "status", {
                        "status": "completed",
                        "total_tasks": len(tasks),
                        "completed_tasks": len(results),
                    })
                    
                    # Assemble final deliverable
                    try:
                        final_dir = project_dir / 'final_delivery'
                        final_dir.mkdir(parents=True, exist_ok=True)
                        final_report_path = final_dir / 'final_report.md'
                        
                        task_entries = []
                        for task in tasks:
                            tid = task['id']
                            output_path = project_dir / 'outputs' / tid / 'output.txt'
                            output = output_path.read_text(encoding='utf-8') if output_path.exists() else ''
                            gen_dir = project_dir / 'outputs' / tid / 'generated'
                            generated = []
                            if gen_dir.exists():
                                for x in gen_dir.iterdir():
                                    if x.is_file():
                                        generated.append(x.name)
                            task_entries.append({
                                'id': tid,
                                'description': task.get('description', ''),
                                'assigned_model': task.get('assigned_model_id', ''),
                                'output': output,
                                'generated': generated
                            })
                        
                        lines = []
                        lines.append(f"# Final Deliverable for {project_dir.name}")
                        lines.append("")
                        lines.append("## Project Summary")
                        lines.append(f"**Goal:** {project_config.get('high_level_goal', 'N/A')}")
                        lines.append(f"**Status:** completed")
                        lines.append("")
                        lines.append("## Task Summary")
                        for te in task_entries:
                            status = "✓" if te['output'] else "○"
                            lines.append(f"- [{status}] [{te['id']}] {te['description']} (Model: {te['assigned_model']})")
                        lines.append("")
                        lines.append("## Detailed Outputs by Task")
                        for te in task_entries:
                            if te['output']:
                                lines.append(f"### Task {te['id']}: {te['description']}")
                                lines.append(f"**Model:** {te['assigned_model']}")
                                lines.append("")
                                lines.append("**Output:**")
                                lines.append("```text")
                                content = te['output']
                                if len(content) > 4000:
                                    content = content[:4000] + "\n... [truncated] ..."
                                lines.append(content)
                                lines.append("```")
                                if te['generated']:
                                    lines.append("")
                                    lines.append("**Generated Files:**")
                                    for gf in te['generated']:
                                        lines.append(f"- {gf}")
                                lines.append("")
                        
                        final_content = "\n".join(lines)
                        final_report_path.write_text(final_content, encoding='utf-8')
                        project_config["final_delivery_path"] = str(final_report_path)
                        with open(config_path, 'w', encoding='utf-8') as f:
                            yaml.dump(project_config, f)
                        
                        print(f"[Vetinari] Final deliverable assembled: {final_report_path}")
                    except Exception as assemble_err:
                        logging.error(f"Error assembling final deliverable: {assemble_err}")
                        
                except Exception as e:
                    logging.error(f"Error running tasks: {e}")
                    project_config["status"] = "error"
                    project_config["error"] = str(e)
                    with open(config_path, 'w', encoding='utf-8') as f:
                        yaml.dump(project_config, f)
            
            thread = threading.Thread(target=run_tasks_background)
            thread.start()
        
        return jsonify({
            "status": "planned" if not auto_run else "started",
            "project_id": project_dir.name,
            "project_path": str(project_dir),
            "tasks": tasks,
            "results": [],
            "model": project_config["model"],
            "active_model_id": project_config.get("active_model_id", ""),
            "warnings": plan.warnings,
            "conversation": conversation,
            "needs_context": plan.needs_context,
            "follow_up_question": plan.follow_up_question
        })
        
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

# API: List all projects
@app.route('/api/projects')
def api_projects():
    try:
        include_archived = request.args.get('include_archived', 'false').lower() == 'true'
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
                        import yaml
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
                            
                            # Skip archived projects unless explicitly requested
                            if project_data["archived"] and not include_archived:
                                continue
                            
                            # Get planned tasks
                            planned_tasks = config.get("tasks", [])
                            
                            # Check which tasks have outputs
                            completed_tasks = set()
                            if outputs_dir.exists():
                                for task_dir in outputs_dir.iterdir():
                                    if task_dir.is_dir():
                                        output_file = task_dir / 'output.txt'
                                        if output_file.exists():
                                            completed_tasks.add(task_dir.name)
                            
                            # Build task status
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
                        import json
                        with open(conv_file, 'r', encoding='utf-8') as f:
                            conv = json.load(f)
                            project_data["message_count"] = len(conv)
                    
                    projects.append(project_data)
        
        return jsonify({"projects": projects})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API: Get single project details
@app.route('/api/project/<project_id>')
def api_project(project_id):
    try:
        project_dir = PROJECT_ROOT / 'projects' / project_id
        
        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404
        
        # Load config
        config_file = project_dir / 'project.yaml'
        config = {}
        if config_file.exists():
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        
        # Load conversation
        conv_file = project_dir / 'conversation.json'
        conversation = []
        if conv_file.exists():
            import json
            with open(conv_file, 'r', encoding='utf-8') as f:
                conversation = json.load(f)
        
        # Get task outputs - but also include planned tasks from config
        tasks = []
        planned_tasks = config.get("tasks", [])
        
        # Check which tasks have outputs
        completed_task_ids = set()
        task_outputs = {}
        outputs_dir = project_dir / 'outputs'
        if outputs_dir.exists():
            for task_dir in sorted(outputs_dir.iterdir()):
                if task_dir.is_dir():
                    task_id = task_dir.name
                    output_file = task_dir / 'output.txt'
                    output = ""
                    if output_file.exists():
                        output = output_file.read_text(encoding='utf-8')
                        completed_task_ids.add(task_id)
                        task_outputs[task_id] = output
                    
                    # Get generated files
                    generated_dir = task_dir / 'generated'
                    files = []
                    if generated_dir.exists():
                        for f in generated_dir.iterdir():
                            if f.is_file():
                                files.append({"name": f.name, "path": str(f)})
                    
                    task_outputs[task_id + "_files"] = files
        
        # Build task list from planned tasks with status
        project_status = config.get("status", "unknown")
        for t in planned_tasks:
            task_id = t.get("id", "")
            tasks.append({
                "id": task_id,
                "description": t.get("description", ""),
                "assigned_model": t.get("assigned_model_id", ""),
                "output": task_outputs.get(task_id, ""),
                "files": task_outputs.get(task_id + "_files", []),
                "status": "completed" if task_id in completed_task_ids else ("running" if project_status == "running" else "pending")
            })
        
        # Also include any output tasks that weren't in the planned tasks
        for task_id in completed_task_ids:
            if not any(t["id"] == task_id for t in tasks):
                tasks.append({
                    "id": task_id,
                    "description": "Additional task",
                    "assigned_model": "",
                    "output": task_outputs.get(task_id, ""),
                    "files": task_outputs.get(task_id + "_files", []),
                    "status": "completed"
                })
        
        return jsonify({
            "id": project_id,
            "config": config,
            "conversation": conversation,
            "tasks": tasks
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API: Send message to existing project
@app.route('/api/project/<project_id>/message', methods=['POST'])
def api_project_message(project_id):
    try:
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({"error": "message is required"}), 400
        
        project_dir = PROJECT_ROOT / 'projects' / project_id
        
        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404
        
        orb = get_orchestrator()
        
        # Load existing conversation
        conv_file = project_dir / 'conversation.json'
        conversation = []
        if conv_file.exists():
            import json
            with open(conv_file, 'r', encoding='utf-8') as f:
                conversation = json.load(f)
        
        # Get model from project config
        config_file = project_dir / 'project.yaml'
        model = "qwen2.5-0.5b-instruct"
        if config_file.exists():
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                model = config.get("model", model)
        
        # Add user message to conversation
        conversation.append({"role": "user", "content": message})
        
        # Build context from conversation
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
        
        # Get AI response
        system_prompt = "You are a helpful coding assistant. Provide detailed, practical code solutions."
        result = orb.adapter.chat(model, system_prompt, context)
        response = result.get('output', '')
        
        # Add assistant response
        conversation.append({"role": "assistant", "content": response})
        
        # Save conversation
        import json
        with open(conv_file, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, indent=2)
        
        return jsonify({
            "status": "completed",
            "response": response,
            "conversation": conversation
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

# API: Add task to project
@app.route('/api/project/<project_id>/task', methods=['POST'])
def api_add_task(project_id):
    try:
        data = request.json
        project_dir = PROJECT_ROOT / 'projects' / project_id
        
        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404
        
        config_file = project_dir / 'project.yaml'
        if not config_file.exists():
            return jsonify({"error": "Project config not found"}), 404
        
        import yaml
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        tasks = config.get("tasks", [])
        
        # Generate new task ID
        new_id = data.get('id', f"t{len(tasks) + 1}")
        
        # Check if ID already exists
        if any(t['id'] == new_id for t in tasks):
            return jsonify({"error": f"Task ID '{new_id}' already exists"}), 400
        
        new_task = {
            "id": new_id,
            "description": data.get('description', ''),
            "inputs": data.get('inputs', []),
            "outputs": data.get('outputs', []),
            "dependencies": data.get('dependencies', []),
            "model_override": data.get('model_override', '')
        }
        
        tasks.append(new_task)
        config['tasks'] = tasks
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        
        return jsonify({"status": "added", "task": new_task})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API: Update task
@app.route('/api/project/<project_id>/task/<task_id>', methods=['PUT'])
def api_update_task(project_id, task_id):
    try:
        data = request.json
        project_dir = PROJECT_ROOT / 'projects' / project_id
        
        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404
        
        config_file = project_dir / 'project.yaml'
        if not config_file.exists():
            return jsonify({"error": "Project config not found"}), 404
        
        import yaml
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        tasks = config.get("tasks", [])
        task_found = False
        
        for i, task in enumerate(tasks):
            if task['id'] == task_id:
                # Update fields
                if 'description' in data:
                    tasks[i]['description'] = data['description']
                if 'inputs' in data:
                    tasks[i]['inputs'] = data['inputs']
                if 'outputs' in data:
                    tasks[i]['outputs'] = data['outputs']
                if 'dependencies' in data:
                    tasks[i]['dependencies'] = data['dependencies']
                if 'model_override' in data:
                    tasks[i]['model_override'] = data['model_override']
                task_found = True
                break
        
        if not task_found:
            return jsonify({"error": "Task not found"}), 404
        
        config['tasks'] = tasks
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        
        return jsonify({"status": "updated", "task": tasks[i]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API: Delete task
@app.route('/api/project/<project_id>/task/<task_id>', methods=['DELETE'])
def api_delete_task(project_id, task_id):
    try:
        project_dir = PROJECT_ROOT / 'projects' / project_id
        
        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404
        
        config_file = project_dir / 'project.yaml'
        if not config_file.exists():
            return jsonify({"error": "Project config not found"}), 404
        
        import yaml
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        tasks = config.get("tasks", [])
        tasks = [t for t in tasks if t['id'] != task_id]
        
        # Also remove this task from any dependencies
        for task in tasks:
            if task_id in task.get('dependencies', []):
                task['dependencies'] = [d for d in task['dependencies'] if d != task_id]
        
        config['tasks'] = tasks
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        
        return jsonify({"status": "deleted", "task_id": task_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API: Get project outputs for review
@app.route('/api/project/<project_id>/review')
def api_project_review(project_id):
    try:
        project_dir = PROJECT_ROOT / 'projects' / project_id
        
        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404
        
        # Load config
        config_file = project_dir / 'project.yaml'
        config = {}
        if config_file.exists():
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        
        # Get task outputs
        outputs_dir = project_dir / 'outputs'
        review_data = {
            "project_id": project_id,
            "goal": config.get("high_level_goal", ""),
            "tasks": [],
            "model": config.get("model", ""),
            "has_all_outputs": True
        }
        
        tasks = config.get("tasks", [])
        for task in tasks:
            task_id = task["id"]
            task_output_dir = outputs_dir / task_id
            
            output_content = ""
            status = "pending"
            files = []
            
            if task_output_dir.exists():
                output_file = task_output_dir / "output.txt"
                if output_file.exists():
                    output_content = output_file.read_text(encoding='utf-8')
                    status = "completed"
                
                generated_dir = task_output_dir / "generated"
                if generated_dir.exists():
                    for f in generated_dir.iterdir():
                        if f.is_file():
                            files.append({
                                "name": f.name,
                                "path": str(f)
                            })
            
            if status == "pending":
                review_data["has_all_outputs"] = False
            
            review_data["tasks"].append({
                "id": task_id,
                "description": task.get("description", ""),
                "model_override": task.get("model_override", ""),
                "status": status,
                "output": output_content[:2000] if output_content else "",  # Truncate for UI
                "output_length": len(output_content) if output_content else 0,
                "files": files
            })
        
        return jsonify(review_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API: Approve outputs and trigger merge
@app.route('/api/project/<project_id>/approve', methods=['POST'])
def api_approve_outputs(project_id):
    try:
        project_dir = PROJECT_ROOT / 'projects' / project_id
        
        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404
        
        # Mark outputs as approved
        approval_file = project_dir / "outputs_approved.txt"
        from datetime import datetime
        approval_file.write_text(f"Approved at: {datetime.now().isoformat()}")
        
        return jsonify({"status": "approved", "project_id": project_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/project/<project_id>/run', methods=['POST'])
def api_run_project(project_id):
    """Start (or re-start) task execution for a saved project.

    This is the plan-approval gate: the frontend shows the plan, the user
    reviews it, then POSTs here to begin execution. Tasks route through
    the full agent framework (not orb.adapter.chat directly).
    """
    try:
        project_dir = PROJECT_ROOT / 'projects' / project_id
        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404

        config_path = project_dir / 'project.yaml'
        if not config_path.exists():
            return jsonify({"error": "Project config not found"}), 404

        with open(config_path, 'r', encoding='utf-8') as f:
            project_config = yaml.safe_load(f)

        current_status = project_config.get('status', '')
        if current_status in ('running',):
            return jsonify({"error": "Project is already running"}), 409
        if current_status == 'completed':
            return jsonify({"error": "Project already completed. Clone it to re-run."}), 409

        tasks = project_config.get('tasks', [])
        if not tasks:
            return jsonify({"error": "No tasks found in project"}), 400

        model = project_config.get('active_model_id', '') or project_config.get('model', '')
        system_prompt = project_config.get('system_prompt', 'You are a helpful coding assistant.')
        conv_file = project_dir / 'conversation.json'

        # Register cancel event and SSE queue
        _cancel_event = _register_project_task(project_id)
        _get_sse_queue(project_id)

        def _run_approved_tasks():
            try:
                project_config['status'] = 'running'
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(project_config, f)

                _push_sse_event(project_id, 'status', {'status': 'running', 'total_tasks': len(tasks)})

                orb = get_orchestrator()
                from vetinari.agent_dispatcher import dispatch_task, build_agent_context
                agent_context = build_agent_context(orb)

                results = []
                task_outputs_text = []

                for idx, task in enumerate(tasks):
                    if _cancel_event.is_set():
                        _push_sse_event(project_id, 'cancelled', {'message': 'Cancelled by user'})
                        break

                    task_id = task.get('id', f'task_{idx}')
                    task_model = task.get('assigned_model_id', '') or model

                    _push_sse_event(project_id, 'task_start', {
                        'task_id': task_id,
                        'task_index': idx,
                        'total': len(tasks),
                        'description': task.get('description', ''),
                        'model': task_model,
                    })

                    task['output_dir'] = str(project_dir / 'outputs' / task_id / 'generated')
                    task_output, task_success, task_meta = dispatch_task(
                        task_dict=task,
                        context=agent_context,
                        model_id=task_model or None,
                    )

                    task_output_dir = project_dir / 'outputs' / task_id
                    task_output_dir.mkdir(parents=True, exist_ok=True)
                    (task_output_dir / 'output.txt').write_text(task_output, encoding='utf-8')

                    task_status = 'completed' if task_success else 'failed'
                    results.append({
                        'task_id': task_id,
                        'model_used': task_model,
                        'agent_type': task_meta.get('agent_type', 'builder'),
                        'status': task_status,
                        'output': task_output,
                    })
                    task_outputs_text.append(
                        f"=== Task {task_id} [{task_meta.get('agent_name','Builder')}]: "
                        f"{task.get('description','')} ===\n\n{task_output}"
                    )

                    _push_sse_event(project_id, 'task_complete', {
                        'task_id': task_id,
                        'task_index': idx,
                        'total': len(tasks),
                        'status': task_status,
                        'agent_type': task_meta.get('agent_type', 'builder'),
                        'tokens_used': task_meta.get('tokens_used', 0),
                        'output_length': len(task_output),
                    })

                # Save conversation
                conversation = []
                if conv_file.exists():
                    try:
                        with open(conv_file, 'r', encoding='utf-8') as f:
                            conversation = json.load(f)
                    except Exception:
                        conversation = []
                results_text = 'Tasks completed!\n\n' + '='*50 + '\n\n'.join(task_outputs_text)
                conversation.append({'role': 'assistant', 'content': results_text})
                with open(conv_file, 'w', encoding='utf-8') as f:
                    json.dump(conversation, f, indent=2)

                project_config['status'] = 'completed'
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(project_config, f)

                _push_sse_event(project_id, 'status', {
                    'status': 'completed',
                    'total_tasks': len(tasks),
                    'completed_tasks': len(results),
                })
            except Exception as e:
                logging.error(f'[run_project] Error: {e}')
                _push_sse_event(project_id, 'error', {'message': str(e)})

        import threading as _threading
        t = _threading.Thread(target=_run_approved_tasks, daemon=True)
        t.start()

        return jsonify({'status': 'running', 'project_id': project_id, 'total_tasks': len(tasks)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/project/<project_id>/download-report', methods=['GET'])
def api_download_report(project_id):
    """Download or display the final report for a completed project."""
    try:
        project_dir = PROJECT_ROOT / 'projects' / project_id
        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404

        # Look for the final report in standard locations
        candidates = [
            project_dir / 'final_delivery' / 'final_report.md',
            project_dir / 'final_delivery' / 'report.md',
            project_dir / 'final_report.md',
        ]
        report_path = next((p for p in candidates if p.exists()), None)

        if report_path:
            from flask import send_file
            return send_file(str(report_path), mimetype='text/markdown', as_attachment=False)

        # Fallback: generate a quick summary from task outputs
        outputs_dir = project_dir / 'outputs'
        summary_lines = [f"# Project Report: {project_id}\n"]
        if outputs_dir.exists():
            for task_dir in sorted(outputs_dir.iterdir()):
                out_file = task_dir / 'output.txt'
                if out_file.exists():
                    summary_lines.append(f"## {task_dir.name}\n")
                    content = out_file.read_text(encoding='utf-8', errors='replace')
                    summary_lines.append(content[:2000])
                    summary_lines.append("\n\n---\n")
        summary = "\n".join(summary_lines) if len(summary_lines) > 1 else "No outputs found."
        from flask import Response
        return Response(summary, mimetype='text/markdown')
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# API: Merge project to final project space
@app.route('/api/project/<project_id>/merge', methods=['POST'])
def api_merge_project(project_id):
    try:
        import shutil
        from datetime import datetime
        
        project_dir = PROJECT_ROOT / 'projects' / project_id
        
        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404
        
        # Create final_project directory
        final_root = PROJECT_ROOT / 'final_projects'
        final_root.mkdir(parents=True, exist_ok=True)
        
        final_dir = final_root / project_id
        if final_dir.exists():
            shutil.rmtree(final_dir)
        final_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy outputs
        outputs_dir = project_dir / 'outputs'
        final_outputs = final_dir / 'outputs'
        if outputs_dir.exists():
            shutil.copytree(outputs_dir, final_outputs)
        
        # Copy and merge generated files
        final_artifacts = final_dir / 'artifacts'
        final_artifacts.mkdir(parents=True, exist_ok=True)
        
        if outputs_dir.exists():
            for task_subdir in outputs_dir.iterdir():
                if task_subdir.is_dir():
                    generated_dir = task_subdir / 'generated'
                    if generated_dir.exists():
                        for f in generated_dir.iterdir():
                            if f.is_file():
                                # Copy with task prefix to avoid conflicts
                                dest = final_artifacts / f"{task_subdir.name}_{f.name}"
                                shutil.copy2(f, dest)
        
        # Create manifest.yaml
        config_file = project_dir / 'project.yaml'
        manifest = {
            "project_id": project_id,
            "created_at": datetime.now().isoformat(),
            "source": str(project_dir)
        }
        
        if config_file.exists():
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                manifest["goal"] = config.get("high_level_goal", "")
                manifest["tasks"] = config.get("tasks", [])
                manifest["model"] = config.get("model", "")
        
        manifest_file = final_dir / 'manifest.yaml'
        with open(manifest_file, 'w', encoding='utf-8') as f:
            yaml.dump(manifest, f)
        
        # Create README.md
        readme_content = f"""# Project: {project_id}

{manifest.get('goal', 'No description')}

## Tasks
"""
        for task in manifest.get('tasks', []):
            readme_content += f"- **{task.get('id')}**: {task.get('description', '')}\n"
        
        readme_content += """
## Files

Generated artifacts are in the `artifacts/` directory.

## Usage

Run the generated code from the artifacts folder.
"""
        
        readme_file = final_dir / 'README.md'
        readme_file.write_text(readme_content)
        
        # Create build_report.json
        build_report = {
            "timestamp": datetime.now().isoformat(),
            "project_id": project_id,
            "tasks": manifest.get('tasks', []),
            "model": manifest.get('model', '')
        }
        
        report_file = final_dir / 'build_report.json'
        import json
        report_file.write_text(json.dumps(build_report, indent=2))
        
        return jsonify({
            "status": "merged",
            "project_id": project_id,
            "final_path": str(final_dir)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API: System prompts presets
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

# API: Get task output by project and task ID
@app.route('/api/project/<project_id>/task/<task_id>/output')
def api_task_output(project_id, task_id):
    try:
        project_dir = PROJECT_ROOT / 'projects' / project_id
        
        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404
        
        output_path = project_dir / 'outputs' / task_id / 'output.txt'
        output = ""
        if output_path.exists():
            output = output_path.read_text(encoding='utf-8')
        
        # Get generated files
        generated_dir = project_dir / 'outputs' / task_id / 'generated'
        files = []
        if generated_dir.exists():
            for f in generated_dir.iterdir():
                if f.is_file():
                    files.append({"name": f.name, "path": str(f)})
        
        return jsonify({
            "project_id": project_id,
            "task_id": task_id,
            "output": output,
            "files": files
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API: Trigger model discovery and refresh the cache
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

# API: Rename a project
@app.route('/api/project/<project_id>/rename', methods=['POST'])
def api_rename_project(project_id):
    data = request.json
    new_name = data.get('name', '')
    new_description = data.get('description', '')
    
    project_dir = PROJECT_ROOT / 'projects' / project_id
    if not project_dir.exists():
        return jsonify({"error": "Project not found"}), 404
    
    config_file = project_dir / 'project.yaml'
    if not config_file.exists():
        return jsonify({"error": "Project config not found"}), 404
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    
    if new_name:
        config['project_name'] = new_name
    if new_description is not None:
        config['description'] = new_description
    
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
    
    return jsonify({
        "status": "renamed",
        "project_id": project_id,
        "project_name": config.get("project_name"),
        "description": config.get("description")
    })

# API: Archive/unarchive a project
@app.route('/api/project/<project_id>/archive', methods=['POST'])
def api_archive_project(project_id):
    data = request.json
    archive = data.get('archive', True)
    
    project_dir = PROJECT_ROOT / 'projects' / project_id
    if not project_dir.exists():
        return jsonify({"error": "Project not found"}), 404
    
    config_file = project_dir / 'project.yaml'
    if not config_file.exists():
        return jsonify({"error": "Project config not found"}), 404
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    
    config['archived'] = archive
    config['status'] = 'archived' if archive else 'completed'
    
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
    
    return jsonify({
        "status": "archived" if archive else "unarchived",
        "project_id": project_id,
        "archived": archive
    })

# API: Delete a project
@app.route('/api/project/<project_id>', methods=['DELETE'])
def api_delete_project(project_id):
    try:
        import shutil
        project_dir = PROJECT_ROOT / 'projects' / project_id
        
        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404
        
        # Delete the project directory
        shutil.rmtree(project_dir)
        
        return jsonify({"status": "deleted", "project_id": project_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API: Assemble final deliverable from task outputs
@app.route('/api/project/<project_id>/assemble', methods=['POST'])
def api_project_assemble(project_id):
    try:
        project_dir = PROJECT_ROOT / 'projects' / project_id
        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404
        
        final_dir = project_dir / 'final_delivery'
        final_dir.mkdir(parents=True, exist_ok=True)
        final_report_path = final_dir / 'final_report.md'
        
        # Load config and tasks
        config_file = project_dir / 'project.yaml'
        if not config_file.exists():
            return jsonify({"error": "Project config not found"}), 404
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        planned_tasks = config.get('tasks', [])
        
        task_entries = []
        for t in planned_tasks:
            tid = t.get('id', '')
            output_path = project_dir / 'outputs' / tid / 'output.txt'
            output = output_path.read_text(encoding='utf-8') if output_path.exists() else ''
            gen_dir = project_dir / 'outputs' / tid / 'generated'
            generated = []
            if gen_dir.exists():
                for x in gen_dir.iterdir():
                    if x.is_file():
                        generated.append(x.name)
            task_entries.append({
                'id': tid,
                'description': t.get('description', ''),
                'assigned_model': t.get('assigned_model_id', ''),
                'output': output,
                'generated': generated
            })
        
        lines = []
        lines.append(f"# Final Deliverable for {project_id}")
        lines.append("")
        lines.append("## Project Summary")
        lines.append(f"**Goal:** {config.get('high_level_goal', 'N/A')}")
        lines.append(f"**Status:** {config.get('status', 'unknown')}")
        lines.append("")
        lines.append("## Task Summary")
        for te in task_entries:
            status = "✓" if te['output'] else "○"
            lines.append(f"- [{status}] [{te['id']}] {te['description']} (Model: {te['assigned_model']})")
        lines.append("")
        lines.append("## Detailed Outputs by Task")
        for te in task_entries:
            if te['output']:
                lines.append(f"### Task {te['id']}: {te['description']}")
                lines.append(f"**Model:** {te['assigned_model']}")
                lines.append("")
                lines.append("**Output:**")
                lines.append("```text")
                content = te['output']
                if len(content) > 4000:
                    content = content[:4000] + "\n... [truncated] ..."
                lines.append(content)
                lines.append("```")
                if te['generated']:
                    lines.append("")
                    lines.append("**Generated Files:**")
                    for gf in te['generated']:
                        lines.append(f"- {gf}")
                lines.append("")
        
        final_content = "\n".join(lines)
        final_report_path.write_text(final_content, encoding='utf-8')
        
        # Update config with final delivery path
        config['final_delivery_path'] = str(final_report_path)
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        
        return jsonify({
            "project_id": project_id, 
            "final_report_path": str(final_report_path), 
            "status": "assembled",
            "task_count": len(task_entries)
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

# API: Get build artifacts
@app.route('/api/artifacts')
def api_artifacts():
    build_dir = PROJECT_ROOT / 'build' / 'artifacts'
    artifacts = []
    if build_dir.exists():
        for f in build_dir.iterdir():
            if f.is_file():
                artifacts.append({
                    "name": f.name,
                    "size": f.stat().st_size,
                    "path": str(f)
                })
    return jsonify({"artifacts": artifacts})

# API: Safe file read for agent (OpenCode-like)
@app.route('/api/project/<project_id>/files/read', methods=['POST'])
def api_read_file(project_id):
    try:
        data = request.json
        file_path = data.get('path', '')
        
        if not file_path:
            return jsonify({"error": "path is required"}), 400
        
        project_dir = PROJECT_ROOT / 'projects' / project_id
        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404
        
        # Whitelist: only allow reads within project workspace
        workspace_dir = project_dir / 'workspace'
        allowed_base = workspace_dir.resolve()
        
        # Resolve the target path
        target_path = (workspace_dir / file_path).resolve()
        
        # Security check: ensure path is within workspace
        try:
            target_path.relative_to(allowed_base)
        except ValueError:
            return jsonify({"error": "Access denied: path outside workspace"}), 403
        
        if not target_path.exists():
            return jsonify({"error": "File not found"}), 404
        
        if not target_path.is_file():
            return jsonify({"error": "Not a file"}), 400
        
        # Read the file
        content = target_path.read_text(encoding='utf-8')
        
        # Log the IO operation
        print(f"[Vetinari IO] Read: {target_path} (project: {project_id})")
        
        return jsonify({
            "status": "ok",
            "path": str(target_path.relative_to(project_dir)),
            "content": content,
            "size": len(content)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API: Safe file write for agent (OpenCode-like)
@app.route('/api/project/<project_id>/files/write', methods=['POST'])
def api_write_file(project_id):
    try:
        data = request.json
        file_path = data.get('path', '')
        content = data.get('content', '')
        
        if not file_path:
            return jsonify({"error": "path is required"}), 400
        
        project_dir = PROJECT_ROOT / 'projects' / project_id
        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404
        
        # Whitelist: only allow writes within project workspace
        workspace_dir = project_dir / 'workspace'
        workspace_dir.mkdir(parents=True, exist_ok=True)
        allowed_base = workspace_dir.resolve()
        
        # Resolve the target path
        target_path = (workspace_dir / file_path).resolve()
        
        # Security check: ensure path is within workspace
        try:
            target_path.relative_to(allowed_base)
        except ValueError:
            return jsonify({"error": "Access denied: path outside workspace"}), 403
        
        # Create parent directories if needed
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the file
        target_path.write_text(content, encoding='utf-8')
        
        # Log the IO operation
        print(f"[Vetinari IO] Write: {target_path} (project: {project_id}, size: {len(content)})")
        
        return jsonify({
            "status": "ok",
            "path": str(target_path.relative_to(project_dir)),
            "size": len(content)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API: List workspace files
@app.route('/api/project/<project_id>/files/list')
def api_list_files(project_id):
    try:
        project_dir = PROJECT_ROOT / 'projects' / project_id
        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404
        
        workspace_dir = project_dir / 'workspace'
        if not workspace_dir.exists():
            return jsonify({"files": []})
        
        files = []
        for f in workspace_dir.rglob('*'):
            if f.is_file():
                files.append({
                    "path": str(f.relative_to(workspace_dir)),
                    "size": f.stat().st_size,
                    "modified": f.stat().st_mtime
                })
        
        return jsonify({"files": files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def _is_admin_user() -> bool:
    """
    Check if the current request comes from an admin user.
    For local deployments, localhost requests are always admin.
    For network deployments, check VETINARI_ADMIN_TOKEN env var.
    """
    admin_token = os.environ.get("VETINARI_ADMIN_TOKEN", "")
    if admin_token:
        # Token-based auth
        auth_header = request.headers.get("Authorization", "")
        req_token = request.headers.get("X-Admin-Token", "")
        provided = req_token or auth_header.replace("Bearer ", "")
        return provided == admin_token
    # No token configured -- allow local requests
    remote = request.remote_addr or ""
    return remote in ("127.0.0.1", "::1", "localhost")


def _project_external_model_enabled(project_dir) -> bool:
    """
    Check if external model discovery is enabled for a specific project.
    Reads from the project's config file, defaults to global setting.
    """
    try:
        config_file = Path(project_dir) / "project.yaml"
        if config_file.exists():
            with open(config_file) as f:
                cfg = yaml.safe_load(f) or {}
            return cfg.get("enable_external_model_discovery", ENABLE_EXTERNAL_DISCOVERY)
    except Exception:
        pass
    return ENABLE_EXTERNAL_DISCOVERY


@app.route('/api/project/<project_id>/model-search', methods=['POST'])
def api_model_search(project_id):
    try:
        project_dir = PROJECT_ROOT / 'projects' / project_id
        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404
        if not ENABLE_EXTERNAL_DISCOVERY:
            return jsonify({"error": "External discovery globally disabled"}), 403
        if not _project_external_model_enabled(project_dir):
            return jsonify({"error": "External model discovery disabled for this project"}), 403
        from vetinari.live_model_search import LiveModelSearchAdapter
        
        data = request.json or {}
        task_description = data.get('task_description', '')
        
        project_dir = PROJECT_ROOT / 'projects' / project_id
        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404
        
        search_adapter = LiveModelSearchAdapter()
        
        lm_models = []
        try:
            from vetinari.model_pool import ModelPool
            model_pool = ModelPool(current_config, current_config.get("host", "http://localhost:1234"))
            model_pool.discover_models()
            lm_models = model_pool.list_models()
        except Exception as e:
            print(f"Could not get LM Studio models: {e}")
        
        candidates = search_adapter.search(task_description, lm_models)
        
        return jsonify({
            "status": "ok",
            "candidates": [c.to_dict() for c in candidates],
            "count": len(candidates)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/project/<project_id>/task/<task_id>/override', methods=['POST'])
def api_task_override(project_id, task_id):
    try:
        data = request.json or {}
        model_id = data.get('model_id', '')
        
        project_dir = PROJECT_ROOT / 'projects' / project_id
        config_file = project_dir / 'project.yaml'
        
        if not config_file.exists():
            return jsonify({"error": "Project not found"}), 404
        
        with open(config_file) as f:
            config = yaml.safe_load(f)
        
        tasks = config.get('tasks', [])
        for task in tasks:
            if task.get('id') == task_id:
                task['model_override'] = model_id
                break
        
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return jsonify({
            "status": "ok",
            "task_id": task_id,
            "model_override": model_id
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/project/<project_id>/refresh-models', methods=['POST'])
def api_refresh_models(project_id):
    try:
        from vetinari.live_model_search import LiveModelSearchAdapter
        
        return jsonify({
            "status": "ok",
            "message": "Model cache refreshed (live search enabled)"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/credentials', methods=['GET'])
def api_admin_list_credentials():
    if not _is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.credentials import credential_manager
        
        credentials = credential_manager.list()
        health = credential_manager.health()
        
        return jsonify({
            "status": "ok",
            "credentials": credentials,
            "health": health
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/credentials/<source_type>', methods=['POST'])
def api_admin_set_credential(source_type):
    if not _is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.credentials import credential_manager
        
        data = request.json or {}
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
        
        return jsonify({
            "status": "ok",
            "message": f"Credential set for {source_type}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/credentials/<source_type>/rotate', methods=['POST'])
def api_admin_rotate_credential(source_type):
    if not _is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.credentials import credential_manager
        
        data = request.json or {}
        new_token = data.get('token', '')
        
        if not new_token:
            return jsonify({"error": "New token is required"}), 400
        
        success = credential_manager.rotate(source_type, new_token)
        
        if success:
            return jsonify({
                "status": "ok",
                "message": f"Credential rotated for {source_type}"
            })
        else:
            return jsonify({"error": "Credential not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/credentials/<source_type>', methods=['DELETE'])
def api_admin_delete_credential(source_type):
    if not _is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.credentials import credential_manager
        
        credential_manager.vault.remove_credential(source_type)
        
        return jsonify({
            "status": "ok",
            "message": f"Credential removed for {source_type}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/credentials/health', methods=['GET'])
def api_admin_credentials_health():
    if not _is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.credentials import credential_manager
        
        health = credential_manager.health()
        
        return jsonify({
            "status": "ok",
            "health": health
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/permissions', methods=['GET'])
def api_admin_permissions():
    return jsonify({"admin": _is_admin_user()}), 200

# Agent orchestration endpoints
@app.route('/api/agents/status', methods=['GET'])
def api_agents_status():
    try:
        from vetinari.multi_agent_orchestrator import MultiAgentOrchestrator
        orch = MultiAgentOrchestrator.get_instance()
        
        if orch is None:
            return jsonify({"agents": []})
        
        # Use existing to_dict() which handles enum serialization correctly
        return jsonify({"agents": orch.get_agent_status()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/agents/initialize', methods=['POST'])
def api_agents_initialize():
    try:
        from vetinari.multi_agent_orchestrator import MultiAgentOrchestrator
        orch = MultiAgentOrchestrator.get_instance()
        
        if orch is None:
            orch = MultiAgentOrchestrator()
        
        orch.initialize_agents()
        
        agent_names = [a.name for a in orch.agents.values()]
        return jsonify({
            "status": "initialized",
            "agents": agent_names
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/agents/active', methods=['GET'])
def api_agents_active():
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
            "ui_planner": "fa-palette",   # underscore, not hyphen
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
            agents.append({
                "name": agent.name,
                "role": agent_type_val,
                "color": colors[i % len(colors)],
                "icon": icons.get(agent_type_val, "fa-robot"),
                "tasks_completed": agent.tasks_completed,
                "current_task": agent.current_task.to_dict() if agent.current_task and hasattr(agent.current_task, "to_dict") else None,
                "state": agent.state.value if hasattr(agent.state, "value") else str(agent.state),
            })
        
        return jsonify({"agents": agents})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/agents/tasks', methods=['GET'])
def api_agents_tasks():
    try:
        from vetinari.multi_agent_orchestrator import MultiAgentOrchestrator
        orch = MultiAgentOrchestrator.get_instance()
        
        if orch is None:
            return jsonify({"tasks": []})
        
        tasks = []
        for task in orch.task_queue:
            tasks.append({
                "id": task.get("id"),
                "description": task.get("description"),
                "status": task.get("status", "pending"),
                "agent": task.get("assigned_agent", "unassigned")
            })
        
        return jsonify({"tasks": tasks})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Shared memory endpoints
@app.route('/api/memory', methods=['GET'])
def api_memory():
    try:
        from vetinari.shared_memory import SharedMemory
        memory = SharedMemory.get_instance()
        
        memories = memory.get_all()
        
        return jsonify({"memories": memories})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Decision endpoints
@app.route('/api/decisions/pending', methods=['GET'])
def api_decisions_pending():
    try:
        from vetinari.shared_memory import SharedMemory
        memory = SharedMemory.get_instance()
        
        decisions = memory.get_memories_by_type("decision")
        
        pending = []
        for d in decisions:
            if not d.get("resolved"):
                pending.append({
                    "id": d.get("id"),
                    "prompt": d.get("content", ""),
                    "options": d.get("options", []),
                    "context": d.get("context", {})
                })
        
        return jsonify({"decisions": pending})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/decisions', methods=['POST'])
def api_decisions_submit():
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

# ============ PLAN ENDPOINTS ============

@app.route('/api/plans', methods=['POST'])
def api_plan_create():
    try:
        from vetinari.planning import plan_manager
        data = request.json
        
        plan = plan_manager.create_plan(
            title=data.get('title', ''),
            prompt=data.get('prompt', ''),
            created_by=data.get('created_by', 'user'),
            waves_data=data.get('waves')
        )
        
        return jsonify(plan.to_dict()), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/plans', methods=['GET'])
def api_plans_list():
    try:
        from vetinari.planning import plan_manager
        status = request.args.get('status')
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        
        plans = plan_manager.list_plans(status=status, limit=limit, offset=offset)
        
        return jsonify({
            "plans": [p.to_dict() for p in plans],
            "total": len(plans),
            "limit": limit,
            "offset": offset
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/plans/<plan_id>', methods=['GET'])
def api_plan_get(plan_id):
    try:
        from vetinari.planning import plan_manager
        plan = plan_manager.get_plan(plan_id)
        
        if not plan:
            return jsonify({"error": "Plan not found"}), 404
        
        return jsonify(plan.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/plans/<plan_id>', methods=['PUT'])
def api_plan_update(plan_id):
    try:
        from vetinari.planning import plan_manager
        data = request.json
        
        plan = plan_manager.update_plan(plan_id, data)
        
        if not plan:
            return jsonify({"error": "Plan not found"}), 404
        
        return jsonify(plan.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/plans/<plan_id>', methods=['DELETE'])
def api_plan_delete(plan_id):
    try:
        from vetinari.planning import plan_manager
        
        if plan_manager.delete_plan(plan_id):
            return "", 204
        return jsonify({"error": "Plan not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/plans/<plan_id>/start', methods=['POST'])
def api_plan_start(plan_id):
    try:
        from vetinari.planning import plan_manager
        plan = plan_manager.start_plan(plan_id)
        
        if not plan:
            return jsonify({"error": "Plan not found"}), 404
        
        return jsonify({
            "plan_id": plan.plan_id,
            "status": plan.status,
            "started_at": plan.updated_at
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/plans/<plan_id>/pause', methods=['POST'])
def api_plan_pause(plan_id):
    try:
        from vetinari.planning import plan_manager
        plan = plan_manager.pause_plan(plan_id)
        
        if not plan:
            return jsonify({"error": "Plan not found"}), 404
        
        return jsonify({
            "plan_id": plan.plan_id,
            "status": plan.status,
            "paused_at": plan.updated_at
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/plans/<plan_id>/resume', methods=['POST'])
def api_plan_resume(plan_id):
    try:
        from vetinari.planning import plan_manager
        plan = plan_manager.resume_plan(plan_id)
        
        if not plan:
            return jsonify({"error": "Plan not found"}), 404
        
        return jsonify({
            "plan_id": plan.plan_id,
            "status": plan.status,
            "resumed_at": plan.updated_at
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/plans/<plan_id>/cancel', methods=['POST'])
def api_plan_cancel(plan_id):
    try:
        from vetinari.planning import plan_manager
        plan = plan_manager.cancel_plan(plan_id)
        
        if not plan:
            return jsonify({"error": "Plan not found"}), 404
        
        return jsonify({
            "plan_id": plan.plan_id,
            "status": plan.status,
            "cancelled_at": plan.updated_at
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/plans/<plan_id>/status', methods=['GET'])
def api_plan_status(plan_id):
    try:
        from vetinari.planning import plan_manager
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
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============ MODEL RELAY ENDPOINTS ============

@app.route('/api/model-catalog', methods=['GET'])
def api_models_list():
    """Model relay catalog (static/configured models). Use /api/models for live LM Studio discovery."""
    try:
        from vetinari.model_relay import model_relay
        models = model_relay.get_all_models()
        
        return jsonify({
            "models": [m.to_dict() for m in models]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/models/<model_id>', methods=['GET'])
def api_model_get(model_id):
    try:
        from vetinari.model_relay import model_relay
        model = model_relay.get_model(model_id)
        
        if not model:
            return jsonify({"error": "Model not found"}), 404
        
        return jsonify(model.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/models/select', methods=['POST'])
def api_model_select():
    try:
        from vetinari.model_relay import model_relay
        data = request.json
        
        selection = model_relay.pick_model_for_task(
            task_type=data.get('task_type'),
            context=data.get('context')
        )
        
        return jsonify(selection.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/models/policy', methods=['GET'])
def api_model_policy_get():
    try:
        from vetinari.model_relay import model_relay
        policy = model_relay.get_policy()
        
        return jsonify(policy.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/models/policy', methods=['PUT'])
def api_model_policy_update():
    try:
        from vetinari.model_relay import model_relay, RoutingPolicy
        data = request.json
        
        policy = RoutingPolicy.from_dict(data)
        model_relay.set_policy(policy)
        
        return jsonify({"status": "updated", "policy": policy.to_dict()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/models/reload', methods=['POST'])
def api_models_reload():
    try:
        from vetinari.model_relay import model_relay
        model_relay.reload_catalog()
        
        return jsonify({
            "status": "reloaded",
            "models_loaded": len(model_relay.get_all_models())
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============ SANDBOX ENDPOINTS ============

@app.route('/api/sandbox/execute', methods=['POST'])
def api_sandbox_execute():
    try:
        from vetinari.sandbox import sandbox_manager
        data = request.json
        
        result = sandbox_manager.execute(
            code=data.get('code', ''),
            sandbox_type=data.get('sandbox_type', 'in_process'),
            timeout=data.get('timeout', 30),
            context=data.get('context')
        )
        
        return jsonify(result.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/sandbox/status', methods=['GET'])
def api_sandbox_status():
    try:
        from vetinari.sandbox import sandbox_manager
        status = sandbox_manager.get_status()
        
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/sandbox/audit', methods=['GET'])
def api_sandbox_audit():
    try:
        from vetinari.sandbox import sandbox_manager
        limit = int(request.args.get('limit', 100))
        
        audit = sandbox_manager.get_audit_log(limit)
        
        return jsonify({"audit_entries": audit, "total": len(audit)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============ SEARCH ENDPOINTS ============

@app.route('/api/code-search', methods=['GET'])
def api_code_search():
    """Search within project code using CocoIndex/ripgrep backends."""
    try:
        from vetinari.code_search import code_search_registry
        query = request.args.get('q', '')
        limit = int(request.args.get('limit', 10))
        language = request.args.get('language')
        backend = request.args.get('backend', 'cocoindex')

        if not query:
            return jsonify({"error": "Query required"}), 400

        adapter = code_search_registry.get_adapter(backend)
        results = adapter.search(query, limit=limit)

        return jsonify({
            "query": query,
            "backend": backend,
            "results": [r.to_dict() for r in results],
            "total": len(results)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/search/index', methods=['POST'])
def api_search_index():
    try:
        from vetinari.code_search import code_search_registry
        data = request.json
        
        project_path = data.get('project_path')
        backend = data.get('backend', 'cocoindex')
        force = data.get('force', False)
        
        if not project_path:
            return jsonify({"error": "project_path required"}), 400
        
        adapter = code_search_registry.get_adapter(backend)
        success = adapter.index_project(project_path, force=force)
        
        return jsonify({
            "status": "indexing" if success else "error",
            "project_path": project_path,
            "backend": backend
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/search/status', methods=['GET'])
def api_search_status():
    try:
        from vetinari.code_search import code_search_registry
        
        backends = {}
        for name in code_search_registry.list_backends():
            try:
                backends[name] = code_search_registry.get_backend_info(name)
            except Exception:
                backends[name] = {"name": name, "status": "error"}
        
        return jsonify({
            "backends": backends,
            "default_backend": code_search_registry.DEFAULT_BACKEND
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============ DECOMPOSITION LAB ENDPOINTS ============

@app.route('/api/decomposition/templates', methods=['GET'])
def api_decomposition_templates():
    try:
        from vetinari.decomposition import decomposition_engine
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
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/decomposition/dod-dor', methods=['GET'])
def api_decomposition_dod_dor():
    try:
        from vetinari.decomposition import decomposition_engine
        level = request.args.get('level', 'Standard')
        
        return jsonify({
            "dod_criteria": decomposition_engine.get_dod_criteria(level),
            "dor_criteria": decomposition_engine.get_dor_criteria(level),
            "levels": ["Light", "Standard", "Hard"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/decomposition/decompose', methods=['POST'])
def api_decomposition_decompose():
    try:
        from vetinari.decomposition import decomposition_engine
        data = request.json
        
        task_prompt = data.get('task_prompt', '')
        parent_task_id = data.get('parent_task_id', 'root')
        depth = int(data.get('depth', 0))
        max_depth = int(data.get('max_depth', 14))
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
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/decomposition/decompose-agent', methods=['POST'])
def api_decomposition_decompose_agent():
    try:
        from vetinari.decomposition_agent import decomposition_agent
        from vetinari.planning import plan_manager
        data = request.json
        
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
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/decomposition/knobs', methods=['GET'])
def api_decomposition_knobs():
    try:
        from vetinari.decomposition_agent import RECURSION_KNOBS, SEED_RATE, SEED_MIX, DEFAULT_MAX_DEPTH, MIN_MAX_DEPTH, MAX_MAX_DEPTH
        
        return jsonify({
            "recursion_knobs": RECURSION_KNOBS,
            "seed_mix": SEED_MIX,
            "seed_rate": SEED_RATE,
            "default_max_depth": DEFAULT_MAX_DEPTH,
            "min_max_depth": MIN_MAX_DEPTH,
            "max_max_depth": MAX_MAX_DEPTH
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/decomposition/history', methods=['GET'])
def api_decomposition_history():
    try:
        from vetinari.decomposition import decomposition_engine
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
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/decomposition/seed-config', methods=['GET'])
def api_decomposition_seed_config():
    try:
        from vetinari.decomposition import decomposition_engine
        
        return jsonify({
            "seed_mix": decomposition_engine.SEED_MIX,
            "seed_rate": decomposition_engine.SEED_RATE,
            "default_max_depth": decomposition_engine.DEFAULT_MAX_DEPTH,
            "min_max_depth": decomposition_engine.MIN_MAX_DEPTH,
            "max_max_depth": decomposition_engine.MAX_MAX_DEPTH
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============ ADR ENDPOINTS ============

@app.route('/api/adr', methods=['GET'])
def api_adr_list():
    try:
        from vetinari.adr import adr_system
        status = request.args.get('status')
        category = request.args.get('category')
        limit = int(request.args.get('limit', 50))
        
        adrs = adr_system.list_adrs(status=status, category=category, limit=limit)
        
        return jsonify({
            "adrs": [a.to_dict() for a in adrs],
            "total": len(adrs)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/adr/<adr_id>', methods=['GET'])
def api_adr_get(adr_id):
    try:
        from vetinari.adr import adr_system
        adr = adr_system.get_adr(adr_id)
        
        if not adr:
            return jsonify({"error": "ADR not found"}), 404
        
        return jsonify(adr.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/adr', methods=['POST'])
def api_adr_create():
    try:
        from vetinari.adr import adr_system
        data = request.json
        
        title = data.get('title')
        category = data.get('category', 'architecture')
        context = data.get('context', '')
        decision = data.get('decision', '')
        consequences = data.get('consequences', '')
        
        if not title:
            return jsonify({"error": "title required"}), 400
        
        adr = adr_system.create_adr(
            title=title,
            category=category,
            context=context,
            decision=decision,
            consequences=consequences,
            created_by=data.get('created_by', 'user')
        )
        
        return jsonify(adr.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/adr/<adr_id>', methods=['PUT'])
def api_adr_update(adr_id):
    try:
        from vetinari.adr import adr_system
        data = request.json
        
        adr = adr_system.update_adr(adr_id, data)
        
        if not adr:
            return jsonify({"error": "ADR not found"}), 404
        
        return jsonify(adr.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/adr/<adr_id>/deprecate', methods=['POST'])
def api_adr_deprecate(adr_id):
    try:
        from vetinari.adr import adr_system
        data = request.json or {}
        
        replacement_id = data.get('replacement_id')
        
        adr = adr_system.deprecate_adr(adr_id, replacement_id)
        
        if not adr:
            return jsonify({"error": "ADR not found"}), 404
        
        return jsonify(adr.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/adr/propose', methods=['POST'])
def api_adr_propose():
    try:
        from vetinari.adr import adr_system, ADRProposal
        data = request.json
        
        context = data.get('context', '')
        num_options = int(data.get('num_options', 3))
        
        proposal = adr_system.generate_proposal(context, num_options)
        
        return jsonify({
            "question": proposal.question,
            "options": proposal.options,
            "recommended": proposal.recommended,
            "rationale": proposal.rationale
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/adr/propose/accept', methods=['POST'])
def api_adr_propose_accept():
    try:
        from vetinari.adr import adr_system
        data = request.json
        
        question = data.get('question', '')
        options = data.get('options', [])
        recommended = data.get('recommended', 0)
        title = data.get('title', 'Proposed Decision')
        category = data.get('category', 'architecture')
        
        from vetinari.adr import ADRProposal
        proposal = ADRProposal(
            question=question,
            options=options,
            recommended=recommended
        )
        
        adr = adr_system.accept_proposal(proposal, title, category)
        
        return jsonify(adr.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/adr/statistics', methods=['GET'])
def api_adr_statistics():
    try:
        from vetinari.adr import adr_system
        stats = adr_system.get_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/adr/is-high-stakes', methods=['GET'])
def api_adr_is_high_stakes():
    try:
        from vetinari.adr import adr_system
        category = request.args.get('category', 'architecture')
        is_high_stakes = adr_system.is_high_stakes(category)
        return jsonify({"is_high_stakes": is_high_stakes, "category": category})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============ SUBTASK TREE ENDPOINTS ============

@app.route('/api/subtasks/<plan_id>', methods=['GET'])
def api_get_subtasks(plan_id):
    try:
        from vetinari.subtask_tree import subtask_tree
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
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/subtasks/<plan_id>', methods=['POST'])
def api_create_subtask(plan_id):
    try:
        from vetinari.subtask_tree import subtask_tree
        data = request.json
        
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
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/subtasks/<plan_id>/<subtask_id>', methods=['PUT'])
def api_update_subtask(plan_id, subtask_id):
    try:
        from vetinari.subtask_tree import subtask_tree
        data = request.json
        
        subtask = subtask_tree.update_subtask(plan_id, subtask_id, data)
        
        if not subtask:
            return jsonify({"error": "Subtask not found"}), 404
        
        return jsonify(subtask.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/subtasks/<plan_id>/tree', methods=['GET'])
def api_get_subtask_tree(plan_id):
    try:
        from vetinari.subtask_tree import subtask_tree
        
        all_subtasks = subtask_tree.get_all_subtasks(plan_id)
        tree_depth = subtask_tree.get_tree_depth(plan_id)
        
        return jsonify({
            "plan_id": plan_id,
            "subtasks": [s.to_dict() for s in all_subtasks],
            "total": len(all_subtasks),
            "depth": tree_depth
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============ ASSIGNMENT PASS ENDPOINTS ============

@app.route('/api/assignments/execute-pass', methods=['POST'])
def api_assignment_execute_pass():
    try:
        from vetinari.assignment_pass import execute_assignment_pass
        data = request.json or {}
        
        plan_id = data.get('plan_id')
        auto_assign = data.get('auto_assign', True)
        
        if not plan_id:
            return jsonify({"error": "plan_id required"}), 400
        
        result = execute_assignment_pass(plan_id, auto_assign)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/assignments/<plan_id>', methods=['GET'])
def api_get_assignments(plan_id):
    try:
        from vetinari.subtask_tree import subtask_tree
        
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
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/assignments/<plan_id>/<subtask_id>', methods=['PUT'])
def api_override_assignment(plan_id, subtask_id):
    try:
        from vetinari.subtask_tree import subtask_tree
        data = request.json
        
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
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============ TEMPLATE VERSIONING ENDPOINTS ============

@app.route('/api/templates/versions', methods=['GET'])
def api_template_versions():
    try:
        from vetinari.template_loader import template_loader
        versions = template_loader.list_versions()
        default = template_loader.default_version()
        return jsonify({"versions": versions, "default": default})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/templates', methods=['GET'])
def api_templates():
    try:
        from vetinari.template_loader import template_loader
        version = request.args.get('version')
        agent_type = request.args.get('agent_type')
        
        templates = template_loader.load_templates(version=version, agent_type=agent_type)
        
        return jsonify({
            "templates": templates,
            "total": len(templates),
            "version": version or template_loader.default_version()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/plans/<plan_id>/migrate_templates', methods=['POST'])
def api_migrate_templates(plan_id):
    try:
        from vetinari.planning import plan_manager
        from vetinari.template_loader import template_loader
        
        data = request.json or {}
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
            plan.template_version = target_version
            plan_manager._save_plan(plan)
            
            return jsonify({
                "plan_id": plan_id,
                "from_version": from_version,
                "to_version": target_version,
                "dry_run": dry_run,
                "status": "migrated",
                "message": f"Plan migrated from {from_version} to {target_version}"
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/ponder/choose-model', methods=['POST'])
def api_ponder_choose_model():
    try:
        from vetinari.ponder import rank_models, get_available_models
        
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


@app.route('/api/ponder/templates', methods=['GET'])
def api_ponder_templates():
    try:
        from vetinari.ponder import PonderEngine
        
        version = request.args.get("version", "v1")
        engine = PonderEngine(template_version=version)
        templates = engine.get_template_prompts()
        
        return jsonify({
            "templates": templates,
            "total": len(templates),
            "version": version
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/ponder/models', methods=['GET'])
def api_ponder_models():
    try:
        from vetinari.ponder import get_available_models
        
        models = get_available_models()
        return jsonify({
            "models": models,
            "total": len(models)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/ponder/plan/<plan_id>', methods=['POST'])
def api_ponder_run_plan(plan_id):
    try:
        from vetinari.ponder import ponder_project_for_plan
        
        result = ponder_project_for_plan(plan_id)
        
        if not result.get("success", False):
            return jsonify(result), 400
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/ponder/plan/<plan_id>', methods=['GET'])
def api_ponder_get_plan(plan_id):
    try:
        from vetinari.ponder import get_ponder_results_for_plan
        
        result = get_ponder_results_for_plan(plan_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/ponder/health', methods=['GET'])
def api_ponder_health():
    try:
        from vetinari.ponder import get_ponder_health
        
        health = get_ponder_health()
        return jsonify(health)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============ RULES CONFIGURATION ENDPOINTS ============

@app.route('/api/rules', methods=['GET'])
def api_rules_get():
    """Get all rules configuration."""
    try:
        from vetinari.rules_manager import get_rules_manager
        rm = get_rules_manager()
        return jsonify(rm.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/rules/global', methods=['GET', 'POST'])
def api_rules_global():
    """Get or set global rules."""
    try:
        from vetinari.rules_manager import get_rules_manager
        rm = get_rules_manager()
        if request.method == 'POST':
            data = request.json or {}
            rules = data.get('rules', [])
            if isinstance(rules, str):
                rules = [r.strip() for r in rules.splitlines() if r.strip()]
            rm.set_global_rules(rules)
            return jsonify({"status": "saved", "rules": rm.get_global_rules()})
        return jsonify({"rules": rm.get_global_rules()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/rules/global-prompt', methods=['GET', 'POST'])
def api_rules_global_prompt():
    """Get or set the global system prompt override."""
    try:
        from vetinari.rules_manager import get_rules_manager
        rm = get_rules_manager()
        if request.method == 'POST':
            data = request.json or {}
            rm.set_global_system_prompt(data.get('prompt', ''))
            return jsonify({"status": "saved"})
        return jsonify({"prompt": rm.get_global_system_prompt()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/rules/project/<project_id>', methods=['GET', 'POST'])
def api_rules_project(project_id):
    """Get or set rules for a specific project."""
    try:
        from vetinari.rules_manager import get_rules_manager
        rm = get_rules_manager()
        if request.method == 'POST':
            data = request.json or {}
            rules = data.get('rules', [])
            if isinstance(rules, str):
                rules = [r.strip() for r in rules.splitlines() if r.strip()]
            rm.set_project_rules(project_id, rules)
            return jsonify({"status": "saved", "project_id": project_id, "rules": rm.get_project_rules(project_id)})
        return jsonify({"project_id": project_id, "rules": rm.get_project_rules(project_id)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/rules/model/<path:model_id>', methods=['GET', 'POST'])
def api_rules_model(model_id):
    """Get or set rules for a specific model."""
    try:
        from vetinari.rules_manager import get_rules_manager
        rm = get_rules_manager()
        if request.method == 'POST':
            data = request.json or {}
            rules = data.get('rules', [])
            if isinstance(rules, str):
                rules = [r.strip() for r in rules.splitlines() if r.strip()]
            rm.set_model_rules(model_id, rules)
            return jsonify({"status": "saved", "model_id": model_id, "rules": rm.get_model_rules(model_id)})
        return jsonify({"model_id": model_id, "rules": rm.get_model_rules(model_id)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============ GOAL VERIFICATION ENDPOINTS ============

@app.route('/api/project/<project_id>/verify-goal', methods=['POST'])
def api_verify_goal(project_id):
    """Verify the final deliverable against the original project goal."""
    try:
        from vetinari.goal_verifier import get_goal_verifier
        data = request.json or {}

        goal = data.get('goal', '')
        final_output = data.get('final_output', '')
        required_features = data.get('required_features', [])
        things_to_avoid = data.get('things_to_avoid', [])
        task_outputs = data.get('task_outputs', [])
        expected_outputs = data.get('expected_outputs', [])

        if not goal:
            # Try to load from project file
            proj_dir = PROJECT_ROOT / 'projects' / project_id
            config_path = proj_dir / 'project.yaml'
            if config_path.exists():
                import yaml as _yaml
                with open(config_path) as f:
                    proj_config = _yaml.safe_load(f) or {}
                goal = proj_config.get('goal', proj_config.get('description', ''))
                required_features = required_features or proj_config.get('required_features', [])
                things_to_avoid = things_to_avoid or proj_config.get('things_to_avoid', [])

        if not goal:
            return jsonify({"error": "goal is required"}), 400

        verifier = get_goal_verifier()
        report = verifier.verify(
            project_id=project_id,
            goal=goal,
            final_output=final_output,
            required_features=required_features,
            things_to_avoid=things_to_avoid,
            task_outputs=task_outputs,
            expected_outputs=expected_outputs,
        )

        return jsonify({
            "report": report.to_dict(),
            "corrective_tasks": report.get_corrective_tasks(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============ IMAGE GENERATION ENDPOINTS ============

@app.route('/api/generate-image', methods=['POST'])
def api_generate_image():
    """Generate an image asset via the ImageGeneratorAgent."""
    try:
        from vetinari.agents.image_generator_agent import get_image_generator_agent
        from vetinari.agents.contracts import AgentTask
        data = request.json or {}

        description = data.get('description', '')
        if not description:
            return jsonify({"error": "description required"}), 400

        agent = get_image_generator_agent({
            "sd_host": current_config.get("sd_host", os.environ.get("SD_WEBUI_HOST", "http://localhost:7860")),
            "sd_enabled": data.get("sd_enabled", True),
            "width": data.get("width", 512),
            "height": data.get("height", 512),
            "steps": data.get("steps", 20),
        })

        task = AgentTask(
            task_id=f"img_{uuid.uuid4().hex[:8]}",
            description=description,
            prompt=description,
            context=data.get("context", {}),
        )

        result = agent.execute(task)
        return jsonify({
            "success": result.success,
            "output": result.output,
            "errors": result.errors or [],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/sd-status', methods=['GET'])
def api_sd_status():
    """Check Stable Diffusion WebUI connection status."""
    try:
        import requests as _req
        host = current_config.get("sd_host", os.environ.get("SD_WEBUI_HOST", "http://localhost:7860"))
        resp = _req.get(f"{host}/sdapi/v1/options", timeout=5)
        if resp.status_code == 200:
            return jsonify({"status": "connected", "host": host})
        return jsonify({"status": "error", "code": resp.status_code}), 200
    except Exception as e:
        return jsonify({"status": "disconnected", "error": str(e)}), 200


# ============ TRAINING ENDPOINTS ============

@app.route('/api/training/stats', methods=['GET'])
def api_training_stats():
    """Get training data statistics."""
    try:
        from vetinari.learning.training_data import get_training_collector
        collector = get_training_collector()
        stats = collector.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e), "total_records": 0}), 200


@app.route('/api/training/export', methods=['POST'])
def api_training_export():
    """Export training data for a given format."""
    try:
        from vetinari.learning.training_data import get_training_collector
        data = request.json or {}
        export_format = data.get('format', 'sft')  # sft | dpo | prompts
        collector = get_training_collector()

        if export_format == 'dpo':
            dataset = collector.export_dpo_dataset()
        elif export_format == 'prompts':
            dataset = collector.export_prompt_variants()
        else:
            dataset = collector.export_sft_dataset()

        return jsonify({
            "format": export_format,
            "count": len(dataset),
            "data": dataset[:100]  # Return first 100 for preview
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/training/start', methods=['POST'])
def api_training_start():
    """Start a training run (async)."""
    try:
        from vetinari.training.pipeline import TrainingPipeline
        data = request.json or {}

        tier = data.get('tier', 'general')        # general|coding|research|review|individual
        model_id = data.get('model_id', '')
        min_quality = float(data.get('min_quality', 0.7))

        def _run():
            try:
                pipeline = TrainingPipeline()
                pipeline.run(
                    base_model=model_id or 'qwen2.5-coder-7b',
                    training_type=tier,
                    min_quality_score=min_quality,
                )
                logger.info(f"Training run completed: tier={tier}, model={model_id}")
            except Exception as te:
                logger.error(f"Training run failed: {te}")

        import threading as _t
        _t.Thread(target=_run, daemon=True).start()

        return jsonify({
            "status": "started",
            "tier": tier,
            "model_id": model_id,
            "message": "Training run started in background"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Register Plan Mode API endpoints
try:
    from vetinari.plan_api import register_plan_api
    register_plan_api(app)
    print("[Vetinari] Plan Mode API registered successfully")
except ImportError as e:
    print(f"[Vetinari] Warning: Plan Mode API not available: {e}")
except Exception as e:
    print(f"[Vetinari] Warning: Failed to register Plan Mode API: {e}")


if __name__ == '__main__':
    _debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
    _port = int(os.environ.get("VETINARI_WEB_PORT", 5000))
    _host = os.environ.get("VETINARI_WEB_HOST", "0.0.0.0")
    app.run(host=_host, port=_port, debug=_debug)
