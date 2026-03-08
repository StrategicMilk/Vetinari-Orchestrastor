"""Task execution routes: stream, cancel, new-project, message, review, approve,
merge, task-output, assemble, model-search, task-override, refresh-models, verify-goal.

Project CRUD routes (list, get, add/update/delete tasks, rename, archive, delete)
live in crud_routes.py.
"""

import json
import logging
import threading
import traceback
import uuid
import yaml
from datetime import datetime
from pathlib import Path

from flask import Blueprint, Response, jsonify, request

from vetinari.web.shared import (
    PROJECT_ROOT, current_config, ENABLE_EXTERNAL_DISCOVERY,
    get_orchestrator,
    _register_project_task, _cancel_project_task,
    _get_sse_queue, _push_sse_event, _cleanup_project_state,
    _get_models_cached, trigger_light_search,
    _is_admin_user, _project_external_model_enabled,
    validate_json_request,
)

logger = logging.getLogger(__name__)

task_exec_bp = Blueprint('task_exec', __name__)


# ---------------------------------------------------------------------------
# SSE stream
# ---------------------------------------------------------------------------

@task_exec_bp.route('/api/project/<project_id>/stream')
def api_project_stream(project_id):
    """SSE endpoint — subscribe to real-time events for a project."""
    from flask import stream_with_context
    import json as _json

    def generate():
        q = _get_sse_queue(project_id)
        yield f"data: {_json.dumps({'type': 'connected', 'project_id': project_id})}\n\n"
        while True:
            try:
                msg = q.get(timeout=25)
                if msg is None:
                    yield f"data: {_json.dumps({'type': 'done'})}\n\n"
                    break
                yield f"event: {msg['event']}\ndata: {msg['data']}\n\n"
            except Exception:
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


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------

@task_exec_bp.route('/api/project/<project_id>/cancel', methods=['POST'])
def api_cancel_project(project_id):
    """Cancel a running project execution."""
    cancelled = _cancel_project_task(project_id)
    _push_sse_event(project_id, "cancelled", {"project_id": project_id, "status": "cancelled"})

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

    _cleanup_project_state(project_id)

    return jsonify({"status": "cancelled" if cancelled else "not_found", "project_id": project_id})


# ---------------------------------------------------------------------------
# New project (planning + optional auto-run)
# ---------------------------------------------------------------------------

@task_exec_bp.route('/api/new-project', methods=['POST'])
def api_new_project():
    data, err = validate_json_request()
    if err:
        return err
    goal = data.get('goal', '')
    model = data.get('model', '')
    system_prompt = data.get('system_prompt', 'You are a helpful coding assistant.')
    auto_run = data.get('auto_run', True)
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

        orb.model_pool.discover_models()

        available_models = orb.model_pool.models
        if not available_models:
            return jsonify({"error": "No models available"}), 400

        default_models = current_config.get("default_models", [])
        fallback_models = current_config.get("fallback_models", [])
        uncensored_fallback_models = current_config.get("uncensored_fallback_models", [])
        memory_budget_gb = current_config.get("memory_budget_gb", 32)

        from vetinari.planning_engine import PlanningEngine
        planner = PlanningEngine(
            default_models=default_models,
            fallback_models=fallback_models,
            uncensored_fallback_models=uncensored_fallback_models,
            memory_budget_gb=memory_budget_gb
        )

        plan = planner.plan(goal, system_prompt, available_models, planning_model=model or None)

        tasks = [t.to_dict() for t in plan.tasks]

        project_dir = PROJECT_ROOT / 'projects' / f'project_{uuid.uuid4().hex[:12]}'
        project_dir.mkdir(parents=True, exist_ok=True)

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

        if project_rules:
            try:
                from vetinari.rules_manager import get_rules_manager
                rm = get_rules_manager()
                rules_list = [r.strip() for r in project_rules.splitlines() if r.strip()]
                rm.set_project_rules(project_dir.name, rules_list)
            except Exception as _re:
                logger.warning(f"Could not save project rules: {_re}")

        config_path = project_dir / 'project.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(project_config, f)

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

        planning_model_id = (model or plan.notes.split(": ")[-1]) if plan.notes else (model or (available_models[0].get('name', '') if available_models else ''))

        model_response = ""
        try:
            result = orb.adapter.chat(planning_model_id, agent_system_prompt, goal)
            model_response = result.get('output', '')
            logger.debug(f"Model response received: {len(model_response)} chars")
        except Exception as e:
            logger.error(f"Error getting model response: {e}")
            model_response = ""

        if model_response and len(model_response) > 10:
            task_plan = model_response
        else:
            task_plan = f"I've analyzed your goal and created {len(tasks)} tasks to complete it.\n\n"
            task_plan += "Warnings:\n" + "\n".join([f"- {w}" for w in plan.warnings]) + "\n\n" if plan.warnings else ""
            task_plan += "Plan:\n" + "\n".join([f"- {t['id']}: {t['description']} (using: {t['assigned_model_id']})" for t in tasks])

        conversation = [
            {"role": "user", "content": goal},
            {"role": "assistant", "content": task_plan}
        ]

        conv_file = project_dir / 'conversation.json'
        with open(conv_file, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, indent=2)

        if auto_run:
            _proj_id = project_dir.name
            _cancel_event = _register_project_task(_proj_id)
            _get_sse_queue(_proj_id)

            def run_tasks_background():
                try:
                    project_config["status"] = "running"
                    with open(config_path, 'w', encoding='utf-8') as f:
                        yaml.dump(project_config, f)

                    _push_sse_event(_proj_id, "status", {"status": "running", "total_tasks": len(tasks)})

                    results = []
                    task_outputs_text = []

                    for idx, task in enumerate(tasks):
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

                    results_text = "Tasks completed! Here are the results:\n\n" + "="*50 + "\n\n".join(task_outputs_text)
                    conversation.append({"role": "assistant", "content": results_text})

                    with open(conv_file, 'w', encoding='utf-8') as f:
                        json.dump(conversation, f, indent=2)

                    project_config["status"] = "completed"
                    with open(config_path, 'w', encoding='utf-8') as f:
                        yaml.dump(project_config, f)

                    _push_sse_event(_proj_id, "status", {
                        "status": "completed",
                        "total_tasks": len(tasks),
                        "completed_tasks": len(results),
                    })

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
                        logger.error(f"Error assembling final deliverable: {assemble_err}")

                except Exception as e:
                    logger.error(f"Error running tasks: {e}")
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
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

@task_exec_bp.route('/api/project/<project_id>/message', methods=['POST'])
def api_project_message(project_id):
    try:
        data, err = validate_json_request()
        if err:
            return err
        message = data.get('message', '')

        if not message:
            return jsonify({"error": "message is required"}), 400

        project_dir = PROJECT_ROOT / 'projects' / project_id

        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404

        orb = get_orchestrator()

        conv_file = project_dir / 'conversation.json'
        conversation = []
        if conv_file.exists():
            with open(conv_file, 'r', encoding='utf-8') as f:
                conversation = json.load(f)

        config_file = project_dir / 'project.yaml'
        model = "qwen2.5-0.5b-instruct"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                proj_cfg = yaml.safe_load(f) or {}
            model = proj_cfg.get("active_model_id") or proj_cfg.get("model") or model

        conversation.append({"role": "user", "content": message})

        system_prompt = "You are a helpful AI assistant working on a project."
        try:
            result = orb.adapter.chat(model, system_prompt, message)
            response = result.get('output', '')
        except Exception as e:
            logger.error(f"Error getting model response: {e}")
            response = f"Error: {str(e)}"

        conversation.append({"role": "assistant", "content": response})

        with open(conv_file, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, indent=2)

        return jsonify({
            "status": "ok",
            "response": response,
            "conversation_length": len(conversation)
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ---------------------------------------------------------------------------
# Review / approve / merge
# ---------------------------------------------------------------------------

@task_exec_bp.route('/api/project/<project_id>/review')
def api_project_review(project_id):
    try:
        project_dir = PROJECT_ROOT / 'projects' / project_id

        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404

        config_file = project_dir / 'project.yaml'
        config = {}
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

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
                "output": output_content[:2000] if output_content else "",
                "output_length": len(output_content) if output_content else 0,
                "files": files
            })

        return jsonify(review_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@task_exec_bp.route('/api/project/<project_id>/approve', methods=['POST'])
def api_approve_outputs(project_id):
    try:
        project_dir = PROJECT_ROOT / 'projects' / project_id

        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404

        approval_file = project_dir / "outputs_approved.txt"
        approval_file.write_text(f"Approved at: {datetime.now().isoformat()}")

        return jsonify({"status": "approved", "project_id": project_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@task_exec_bp.route('/api/project/<project_id>/merge', methods=['POST'])
def api_merge_project(project_id):
    try:
        import shutil

        project_dir = PROJECT_ROOT / 'projects' / project_id

        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404

        final_root = PROJECT_ROOT / 'final_projects'
        final_root.mkdir(parents=True, exist_ok=True)

        final_dir = final_root / project_id
        if final_dir.exists():
            shutil.rmtree(final_dir)
        final_dir.mkdir(parents=True, exist_ok=True)

        outputs_dir = project_dir / 'outputs'
        final_outputs = final_dir / 'outputs'
        if outputs_dir.exists():
            shutil.copytree(outputs_dir, final_outputs)

        final_artifacts = final_dir / 'artifacts'
        final_artifacts.mkdir(parents=True, exist_ok=True)

        if outputs_dir.exists():
            for task_subdir in outputs_dir.iterdir():
                if task_subdir.is_dir():
                    generated_dir = task_subdir / 'generated'
                    if generated_dir.exists():
                        for f in generated_dir.iterdir():
                            if f.is_file():
                                dest = final_artifacts / f"{task_subdir.name}_{f.name}"
                                shutil.copy2(f, dest)

        config_file = project_dir / 'project.yaml'
        manifest = {
            "project_id": project_id,
            "created_at": datetime.now().isoformat(),
            "source": str(project_dir)
        }

        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                manifest["goal"] = config.get("high_level_goal", "")
                manifest["tasks"] = config.get("tasks", [])
                manifest["model"] = config.get("model", "")

        manifest_file = final_dir / 'manifest.yaml'
        with open(manifest_file, 'w', encoding='utf-8') as f:
            yaml.dump(manifest, f)

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

        (final_dir / 'README.md').write_text(readme_content)

        build_report = {
            "timestamp": datetime.now().isoformat(),
            "project_id": project_id,
            "tasks": manifest.get('tasks', []),
            "model": manifest.get('model', '')
        }

        (final_dir / 'build_report.json').write_text(json.dumps(build_report, indent=2))

        return jsonify({
            "status": "merged",
            "project_id": project_id,
            "final_path": str(final_dir)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Task output
# ---------------------------------------------------------------------------

@task_exec_bp.route('/api/project/<project_id>/task/<task_id>/output')
def api_task_output(project_id, task_id):
    try:
        project_dir = PROJECT_ROOT / 'projects' / project_id

        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404

        output_path = project_dir / 'outputs' / task_id / 'output.txt'
        output = ""
        if output_path.exists():
            output = output_path.read_text(encoding='utf-8')

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


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------

@task_exec_bp.route('/api/project/<project_id>/assemble', methods=['POST'])
def api_project_assemble(project_id):
    try:
        project_dir = PROJECT_ROOT / 'projects' / project_id
        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404

        final_dir = project_dir / 'final_delivery'
        final_dir.mkdir(parents=True, exist_ok=True)
        final_report_path = final_dir / 'final_report.md'

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
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ---------------------------------------------------------------------------
# Model search / override / refresh / verify-goal
# ---------------------------------------------------------------------------

@task_exec_bp.route('/api/project/<project_id>/model-search', methods=['POST'])
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

        data, err = validate_json_request()
        if err:
            return err
        task_description = data.get('task_description', '')

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


@task_exec_bp.route('/api/project/<project_id>/task/<task_id>/override', methods=['POST'])
def api_task_override(project_id, task_id):
    try:
        data, err = validate_json_request()
        if err:
            return err
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


@task_exec_bp.route('/api/project/<project_id>/refresh-models', methods=['POST'])
def api_refresh_models(project_id):
    try:
        from vetinari.live_model_search import LiveModelSearchAdapter  # noqa: F401

        return jsonify({
            "status": "ok",
            "message": "Model cache refreshed (live search enabled)"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@task_exec_bp.route('/api/project/<project_id>/verify-goal', methods=['POST'])
def api_verify_goal(project_id):
    """Verify the final deliverable against the original project goal."""
    try:
        from vetinari.goal_verifier import get_goal_verifier
        data, err = validate_json_request()
        if err:
            return err

        goal = data.get('goal', '')
        final_output = data.get('final_output', '')
        required_features = data.get('required_features', [])
        things_to_avoid = data.get('things_to_avoid', [])
        task_outputs = data.get('task_outputs', [])
        expected_outputs = data.get('expected_outputs', [])

        if not goal:
            proj_dir = PROJECT_ROOT / 'projects' / project_id
            config_path = proj_dir / 'project.yaml'
            if config_path.exists():
                with open(config_path) as f:
                    proj_config = yaml.safe_load(f) or {}
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
