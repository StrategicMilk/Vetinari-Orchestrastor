import json
from pathlib import Path
from typing import List, Dict, Optional

BASE = Path(__file__).resolve().parent.parent / "templates"

class TemplateLoader:
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or BASE

    def list_versions(self) -> List[str]:
        manifest = self.base_path / "versions.json"
        if not manifest.exists():
            return ["v1"]
        try:
            with open(manifest, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("versions", ["v1"])
        except Exception:
            return ["v1"]

    def load_templates_for_agent(self, version: str, agent_type: str) -> List[Dict]:
        filename_map = {
            "explorer": "explorer.json",
            "librarian": "librarian.json",
            "oracle": "oracle.json",
            "ui_planner": "ui_planner.json",
            "builder": "builder.json",
            "researcher": "researcher.json",
            "evaluator": "evaluator.json",
            "synthesizer": "synthesizer.json",
        }
        filename = filename_map.get(agent_type)
        if not filename:
            return []
        path = self.base_path / version / filename
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data
        except Exception:
            return []

    def load_templates(self, version: Optional[str] = None, agent_type: Optional[str] = None) -> List[Dict]:
        ver = version or self.default_version()
        if agent_type:
            return self.load_templates_for_agent(ver, agent_type)
        templates = []
        for atype in ["explorer","librarian","oracle","ui_planner","builder","researcher","evaluator","synthesizer"]:
            templates.extend(self.load_templates_for_agent(ver, atype) or [])
        return templates

    def default_version(self) -> str:
        versions = self.list_versions()
        return versions[0] if versions else "v1"


template_loader = TemplateLoader()
