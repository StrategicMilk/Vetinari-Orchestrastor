from __future__ import annotations

import json
import os
import zipfile
from datetime import datetime
from pathlib import Path


class Builder:
    def __init__(self, config: dict):
        self.config = config

    def build_final_artifact(self, results: list | None = None):
        outputs_dir = Path(self.config.get("outputs_dir", "outputs"))
        artifacts_dir = Path(self.config.get("build", {}).get("artifacts", "build/artifacts"))
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        zip_path = artifacts_dir / "vetinari_cli_skeleton.zip"

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(outputs_dir):
                for f in files:
                    p = Path(root) / f
                    z.write(p, arcname=str(p.relative_to(outputs_dir)))

        # Create build report if results provided
        if results:
            report = {
                "timestamp": datetime.now().isoformat(),
                "total_tasks": len(results),
                "completed": sum(1 for r in results if r.get("status") == "completed"),
                "failed": sum(1 for r in results if r.get("status") != "completed"),
                "tasks": [
                    {
                        "task_id": r.get("task_id"),
                        "status": r.get("status"),
                        "model": r.get("model_id"),
                        "latency_ms": r.get("latency_ms"),
                        "error": r.get("error"),
                    }
                    for r in results
                ],
            }
            report_path = artifacts_dir / "build_report.json"
            report_path.write_text(json.dumps(report, indent=2))

        return str(zip_path)
