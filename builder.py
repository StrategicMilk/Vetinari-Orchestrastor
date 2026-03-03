import os
import zipfile
from pathlib import Path

class Builder:
    def __init__(self, config: dict):
        self.config = config

    def build_final_artifact(self):
        outputs_dir = Path(self.config.get("outputs_dir", "outputs"))
        artifacts_dir = Path(self.config.get("build", {}).get("artifacts", "build/artifacts"))
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        zip_path = artifacts_dir / "vetinari_cli_skeleton.zip"

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(outputs_dir):
                for f in files:
                    p = Path(root) / f
                    z.write(p, arcname=str(p.relative_to(outputs_dir)))
        return str(zip_path)