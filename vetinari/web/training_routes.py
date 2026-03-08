"""Training, image generation, and Stable Diffusion API routes."""

import logging
import os
import uuid

from flask import Blueprint, jsonify, request

bp = Blueprint('training', __name__)
logger = logging.getLogger(__name__)


@bp.route('/api/generate-image', methods=['POST'])
def api_generate_image():
    """Generate an image asset via the ImageGeneratorAgent."""
    try:
        from vetinari.agents.image_generator_agent import get_image_generator_agent
        from vetinari.agents.contracts import AgentTask
        from vetinari.web_ui import current_config
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


@bp.route('/api/sd-status', methods=['GET'])
def api_sd_status():
    """Check Stable Diffusion WebUI connection status."""
    try:
        import requests as _req
        from vetinari.web_ui import current_config
        host = current_config.get("sd_host", os.environ.get("SD_WEBUI_HOST", "http://localhost:7860"))
        resp = _req.get(f"{host}/sdapi/v1/options", timeout=5)
        if resp.status_code == 200:
            return jsonify({"status": "connected", "host": host})
        return jsonify({"status": "error", "code": resp.status_code}), 200
    except Exception as e:
        return jsonify({"status": "disconnected", "error": str(e)}), 200


@bp.route('/api/training/stats', methods=['GET'])
def api_training_stats():
    """Get training data statistics."""
    try:
        from vetinari.learning.training_data import get_training_collector
        collector = get_training_collector()
        stats = collector.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e), "total_records": 0}), 200


@bp.route('/api/training/export', methods=['POST'])
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


@bp.route('/api/training/start', methods=['POST'])
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
                logger.info("Training run completed: tier=%s, model=%s", tier, model_id)
            except Exception as te:
                logger.error("Training run failed: %s", te)

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
