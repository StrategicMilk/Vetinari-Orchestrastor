"""Project CRUD routes (list, get, add/update/delete tasks, rename, archive, delete).

Task execution routes (new-project, stream, cancel, message, review, approve, merge,
assemble, model-search, verify-goal, etc.) live in task_execution_routes.py.
"""

import logging
import yaml
import json
import shutil
from pathlib import Path

from flask import Blueprint, jsonify, request

from vetinari.web.shared import (
    PROJECT_ROOT,
    _cleanup_project_state,
    validate_json_request,
)

logger = logging.getLogger(__name__)

crud_bp = Blueprint('crud', __name__)


# ---------------------------------------------------------------------------
# Projects list / detail
# ---------------------------------------------------------------------------

@crud_bp.route('/api/projects')
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

                            if project_data["archived"] and not include_archived:
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


@crud_bp.route('/api/project/<project_id>')
def api_project(project_id):
    try:
        project_dir = PROJECT_ROOT / 'projects' / project_id

        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404

        config_file = project_dir / 'project.yaml'
        config = {}
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

        conv_file = project_dir / 'conversation.json'
        conversation = []
        if conv_file.exists():
            with open(conv_file, 'r', encoding='utf-8') as f:
                conversation = json.load(f)

        tasks = []
        planned_tasks = config.get("tasks", [])

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

                    generated_dir = task_dir / 'generated'
                    files = []
                    if generated_dir.exists():
                        for f in generated_dir.iterdir():
                            if f.is_file():
                                files.append({"name": f.name, "path": str(f)})

                    task_outputs[task_id + "_files"] = files

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


# ---------------------------------------------------------------------------
# Task CRUD
# ---------------------------------------------------------------------------

@crud_bp.route('/api/project/<project_id>/task', methods=['POST'])
def api_add_task(project_id):
    try:
        data, err = validate_json_request()
        if err:
            return err
        project_dir = PROJECT_ROOT / 'projects' / project_id

        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404

        config_file = project_dir / 'project.yaml'
        if not config_file.exists():
            return jsonify({"error": "Project config not found"}), 404

        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        tasks = config.get("tasks", [])

        new_id = data.get('id', f"t{len(tasks) + 1}")

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


@crud_bp.route('/api/project/<project_id>/task/<task_id>', methods=['PUT'])
def api_update_task(project_id, task_id):
    try:
        data, err = validate_json_request()
        if err:
            return err
        project_dir = PROJECT_ROOT / 'projects' / project_id

        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404

        config_file = project_dir / 'project.yaml'
        if not config_file.exists():
            return jsonify({"error": "Project config not found"}), 404

        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        tasks = config.get("tasks", [])
        task_found = False

        for i, task in enumerate(tasks):
            if task['id'] == task_id:
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


@crud_bp.route('/api/project/<project_id>/task/<task_id>', methods=['DELETE'])
def api_delete_task(project_id, task_id):
    try:
        project_dir = PROJECT_ROOT / 'projects' / project_id

        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404

        config_file = project_dir / 'project.yaml'
        if not config_file.exists():
            return jsonify({"error": "Project config not found"}), 404

        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        tasks = config.get("tasks", [])
        tasks = [t for t in tasks if t['id'] != task_id]

        for task in tasks:
            if task_id in task.get('dependencies', []):
                task['dependencies'] = [d for d in task['dependencies'] if d != task_id]

        config['tasks'] = tasks

        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)

        return jsonify({"status": "deleted", "task_id": task_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Project metadata mutations
# ---------------------------------------------------------------------------

@crud_bp.route('/api/project/<project_id>/rename', methods=['POST'])
def api_rename_project(project_id):
    data, err = validate_json_request()
    if err:
        return err
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


@crud_bp.route('/api/project/<project_id>/archive', methods=['POST'])
def api_archive_project(project_id):
    data, err = validate_json_request()
    if err:
        return err
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


@crud_bp.route('/api/project/<project_id>', methods=['DELETE'])
def api_delete_project(project_id):
    try:
        project_dir = PROJECT_ROOT / 'projects' / project_id

        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404

        _cleanup_project_state(project_id)
        shutil.rmtree(project_dir)

        return jsonify({"status": "deleted", "project_id": project_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

@crud_bp.route('/api/artifacts')
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


@crud_bp.route('/api/project/<project_id>/files/read', methods=['POST'])
def api_read_file(project_id):
    try:
        data, err = validate_json_request()
        if err:
            return err
        file_path = data.get('path', '')

        if not file_path:
            return jsonify({"error": "path is required"}), 400

        project_dir = PROJECT_ROOT / 'projects' / project_id
        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404

        workspace_dir = project_dir / 'workspace'
        allowed_base = workspace_dir.resolve()

        target_path = (workspace_dir / file_path).resolve()

        try:
            target_path.relative_to(allowed_base)
        except ValueError:
            return jsonify({"error": "Access denied: path outside workspace"}), 403

        if not target_path.exists():
            return jsonify({"error": "File not found"}), 404

        if not target_path.is_file():
            return jsonify({"error": "Not a file"}), 400

        content = target_path.read_text(encoding='utf-8')
        logger.debug(f"IO Read: {target_path} (project: {project_id})")

        return jsonify({
            "status": "ok",
            "path": str(target_path.relative_to(project_dir)),
            "content": content,
            "size": len(content)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@crud_bp.route('/api/project/<project_id>/files/write', methods=['POST'])
def api_write_file(project_id):
    try:
        data, err = validate_json_request()
        if err:
            return err
        file_path = data.get('path', '')
        content = data.get('content', '')

        if not file_path:
            return jsonify({"error": "path is required"}), 400

        project_dir = PROJECT_ROOT / 'projects' / project_id
        if not project_dir.exists():
            return jsonify({"error": "Project not found"}), 404

        workspace_dir = project_dir / 'workspace'
        workspace_dir.mkdir(parents=True, exist_ok=True)
        allowed_base = workspace_dir.resolve()

        target_path = (workspace_dir / file_path).resolve()

        try:
            target_path.relative_to(allowed_base)
        except ValueError:
            return jsonify({"error": "Access denied: path outside workspace"}), 403

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content, encoding='utf-8')
        logger.debug(f"IO Write: {target_path} (project: {project_id}, size: {len(content)})")

        return jsonify({
            "status": "ok",
            "path": str(target_path.relative_to(project_dir)),
            "size": len(content)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@crud_bp.route('/api/project/<project_id>/files/list')
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
