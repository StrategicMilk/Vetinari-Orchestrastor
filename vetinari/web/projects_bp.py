"""
Flask Blueprint for all project-related API routes.

Extracted from vetinari/web_ui.py. All route handlers are copied verbatim;
only the decorator prefix changes from @app.route to @projects_bp.route.
"""

import os
import json
import logging
import yaml
import uuid
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta

from flask import Blueprint, jsonify, request, Response

from vetinari.web.shared import (
    PROJECT_ROOT, current_config, ENABLE_EXTERNAL_DISCOVERY,
    get_orchestrator,
    _register_project_task, _cancel_project_task,
    _get_sse_queue, _push_sse_event, _cleanup_project_state,
    _get_models_cached, trigger_light_search,
    _is_admin_user, _project_external_model_enabled,
)
from vetinari.web import require_admin

logger = logging.getLogger(__name__)

projects_bp = Blueprint('projects', __name__)


# API: Server-Sent Events stream for real-time task updates
@projects_bp.route('/api/project/<project_id>/stream')
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
            "Access-Control-Allow-Origin": request.headers.get("Origin", "http://localhost:5000"),
        },
    )


# API: Cancel a running project task
@projects_bp.route('/api/project/<project_id>/cancel', methods=['POST'])
@require_admin
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

    # Clean up in-memory state after cancellation
    _cleanup_project_state(project_id)

    return jsonify({"status": "cancelled" if cancelled else "not_found", "project_id": project_id})


@projects_bp.route('/api/new-project', methods=['POST'])
@require_admin
def api_new_project():
    data = request.json
    goal = data.get('goal', '')
    model = data.get('model', '')
    system_prompt = data.get('system_prompt', 'You are a helpful coding assistant.')
    auto_run = data.get('auto_run', False)
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
            logger.debug(f"Model response received: {len(model_response)} chars")
        except Exception as e:
            logger.error(f"Error getting model response: {e}")
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

                    results = []
                    task_outputs_text = []

                    for idx, task in enumerate(tasks):
                        # Check cancel flag
                        if _cancel_event.is_set():
                            _push_sse_event(_proj_id, "cancelled", {"message": "Cancelled by user"})
                            break

                        task_id = task['id']
                        task_model = task['assigned_model_id'] or model or available_models[0].get('name', '')

                        _push_sse_event(_proj_id, "task_start", {
                            "task_id": task_id,
                            "task_index": idx,
                            "total": len(tasks),
                            "description": task['description'],
                            "model": task_model,
                        })

                        task_prompt = f"""Task: {task['description']}

Inputs: {', '.join(task['inputs'])}
Outputs: {', '.join(task['outputs'])}

Implement this task. Output the code as code blocks with filenames."""

                        task_result = orb.adapter.chat(task_model, system_prompt, task_prompt)
                        task_output = task_result.get('output', '')
                        tokens_used = task_result.get('tokens_used', 0)

                        # Save task output
                        task_output_dir = project_dir / 'outputs' / task_id
                        task_output_dir.mkdir(parents=True, exist_ok=True)
                        (task_output_dir / 'output.txt').write_text(task_output, encoding='utf-8')

                        code_blocks = orb.executor._parse_code_blocks(task_output)
                        if code_blocks:
                            generated_dir = task_output_dir / 'generated'
                            generated_dir.mkdir(parents=True, exist_ok=True)
                            for filename, code in code_blocks.items():
                                filepath = generated_dir / filename
                                filepath.write_text(code, encoding='utf-8')
                                logger.debug(f"Written: {filepath}")

                        results.append({
                            "task_id": task_id,
                            "model_used": task_model,
                            "status": "completed",
                            "output": task_output,
                        })

                        task_outputs_text.append(
                            f"=== Task {task_id} (using {task_model}): {task['description']} ===\n\n{task_output}"
                        )

                        _push_sse_event(_proj_id, "task_complete", {
                            "task_id": task_id,
                            "task_index": idx,
                            "total": len(tasks),
                            "status": "completed",
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

                        logger.info(f"Final deliverable assembled: {final_report_path}")
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
@projects_bp.route('/api/projects')
@require_admin
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
@projects_bp.route('/api/project/<project_id>')
@require_admin
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
@projects_bp.route('/api/project/<project_id>/message', methods=['POST'])
@require_admin
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
@projects_bp.route('/api/project/<project_id>/task', methods=['POST'])
@require_admin
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
@projects_bp.route('/api/project/<project_id>/task/<task_id>', methods=['PUT'])
@require_admin
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
@projects_bp.route('/api/project/<project_id>/task/<task_id>', methods=['DELETE'])
@require_admin
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
@projects_bp.route('/api/project/<project_id>/review')
@require_admin
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
@projects_bp.route('/api/project/<project_id>/approve', methods=['POST'])
@require_admin
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


# API: Merge project to final project space
@projects_bp.route('/api/project/<project_id>/merge', methods=['POST'])
@require_admin
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


# API: Get task output by project and task ID
@projects_bp.route('/api/project/<project_id>/task/<task_id>/output')
@require_admin
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


# API: Rename a project
@projects_bp.route('/api/project/<project_id>/rename', methods=['POST'])
@require_admin
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
@projects_bp.route('/api/project/<project_id>/archive', methods=['POST'])
@require_admin
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
@projects_bp.route('/api/project/<project_id>', methods=['DELETE'])
@require_admin
def api_delete_project(project_id):
    try:
        import shutil
        project_dir = PROJECT_ROOT / 'projects' / project_id

        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404

        # Clean up in-memory state to prevent memory leaks
        _cleanup_project_state(project_id)
        # Delete the project directory
        shutil.rmtree(project_dir)

        return jsonify({"status": "deleted", "project_id": project_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# API: Assemble final deliverable from task outputs
@projects_bp.route('/api/project/<project_id>/assemble', methods=['POST'])
@require_admin
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
@projects_bp.route('/api/artifacts')
@require_admin
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
@projects_bp.route('/api/project/<project_id>/files/read', methods=['POST'])
@require_admin
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
        logger.debug(f"IO Read: {target_path} (project: {project_id})")

        return jsonify({
            "status": "ok",
            "path": str(target_path.relative_to(project_dir)),
            "content": content,
            "size": len(content)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# API: Safe file write for agent (OpenCode-like)
@projects_bp.route('/api/project/<project_id>/files/write', methods=['POST'])
@require_admin
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
        logger.debug(f"IO Write: {target_path} (project: {project_id}, size: {len(content)})")

        return jsonify({
            "status": "ok",
            "path": str(target_path.relative_to(project_dir)),
            "size": len(content)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# API: List workspace files
@projects_bp.route('/api/project/<project_id>/files/list')
@require_admin
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


@projects_bp.route('/api/project/<project_id>/model-search', methods=['POST'])
@require_admin
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
            logger.warning(f"Could not get LM Studio models: {e}")

        candidates = search_adapter.search(task_description, lm_models)

        return jsonify({
            "status": "ok",
            "candidates": [c.to_dict() for c in candidates],
            "count": len(candidates)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@projects_bp.route('/api/project/<project_id>/task/<task_id>/override', methods=['POST'])
@require_admin
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


@projects_bp.route('/api/project/<project_id>/refresh-models', methods=['POST'])
@require_admin
def api_refresh_models(project_id):
    try:
        from vetinari.live_model_search import LiveModelSearchAdapter

        return jsonify({
            "status": "ok",
            "message": "Model cache refreshed (live search enabled)"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@projects_bp.route('/api/project/<project_id>/verify-goal', methods=['POST'])
@require_admin
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
