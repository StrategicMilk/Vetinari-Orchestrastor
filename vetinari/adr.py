from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ADRStatus(Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"
    SUPERSEDED = "superseded"


class ADRCategory(Enum):
    ARCHITECTURE = "architecture"
    SECURITY = "security"
    DATA_FLOW = "data_flow"
    API_DESIGN = "api_design"
    AGENT_DESIGN = "agent_design"
    DECOMPOSITION = "decomposition"
    PERFORMANCE = "performance"
    INTEGRATION = "integration"


HIGH_STAKES_CATEGORIES = {ADRCategory.ARCHITECTURE, ADRCategory.SECURITY, ADRCategory.DATA_FLOW}


@dataclass
class ADR:
    adr_id: str
    title: str
    category: str
    context: str
    decision: str
    status: str = ADRStatus.PROPOSED.value
    consequences: str = ""
    related_adrs: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    created_by: str = "system"
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "adr_id": self.adr_id,
            "title": self.title,
            "category": self.category,
            "context": self.context,
            "decision": self.decision,
            "status": self.status,
            "consequences": self.consequences,
            "related_adrs": self.related_adrs,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "created_by": self.created_by,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ADR:
        return cls(
            adr_id=data.get("adr_id", ""),
            title=data.get("title", ""),
            category=data.get("category", "architecture"),
            context=data.get("context", ""),
            decision=data.get("decision", ""),
            status=data.get("status", ADRStatus.PROPOSED.value),
            consequences=data.get("consequences", ""),
            related_adrs=data.get("related_adrs", []),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            created_by=data.get("created_by", "system"),
            notes=data.get("notes", ""),
        )


@dataclass
class ADRProposal:
    question: str
    options: list[dict[str, Any]]
    recommended: int = 0
    rationale: str = ""


class ADRSystem:
    _instance = None

    @classmethod
    def get_instance(cls, storage_path: str | None = None) -> ADRSystem:
        if cls._instance is None:
            cls._instance = cls(storage_path)
        return cls._instance

    def __init__(self, storage_path: str | None = None):
        if storage_path is None:
            storage_path = Path.home() / ".lmstudio" / "projects" / "Vetinari" / "adr"

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.adrs: dict[str, ADR] = {}
        self._load_adrs()

    def _load_adrs(self):
        for file in self.storage_path.glob("*.json"):
            try:
                with open(file) as f:
                    data = json.load(f)
                    adr = ADR.from_dict(data)
                    self.adrs[adr.adr_id] = adr
            except Exception as e:
                logger.error("Error loading ADR %s: %s", file, e)

    def _save_adr(self, adr: ADR):
        file_path = self.storage_path / f"{adr.adr_id}.json"
        with open(file_path, "w") as f:
            json.dump(adr.to_dict(), f, indent=2)

    def create_adr(
        self, title: str, category: str, context: str, decision: str, consequences: str = "", created_by: str = "user"
    ) -> ADR:
        adr_id = f"ADR-{len(self.adrs) + 1:04d}"
        now = datetime.now().isoformat()

        adr = ADR(
            adr_id=adr_id,
            title=title,
            category=category,
            context=context,
            decision=decision,
            consequences=consequences,
            created_at=now,
            updated_at=now,
            created_by=created_by,
        )

        self.adrs[adr_id] = adr
        self._save_adr(adr)
        return adr

    def get_adr(self, adr_id: str) -> ADR | None:
        return self.adrs.get(adr_id)

    def list_adrs(self, status: str | None = None, category: str | None = None, limit: int = 50) -> list[ADR]:
        results = list(self.adrs.values())

        if status:
            results = [a for a in results if a.status == status]
        if category:
            results = [a for a in results if a.category == category]

        results.sort(key=lambda a: a.created_at, reverse=True)
        return results[:limit]

    def update_adr(self, adr_id: str, updates: dict) -> ADR | None:
        adr = self.adrs.get(adr_id)
        if not adr:
            return None

        for key, value in updates.items():
            if hasattr(adr, key):
                setattr(adr, key, value)

        adr.updated_at = datetime.now().isoformat()
        self._save_adr(adr)
        return adr

    def deprecate_adr(self, adr_id: str, replacement_id: str | None = None) -> ADR | None:
        adr = self.adrs.get(adr_id)
        if not adr:
            return None

        adr.status = ADRStatus.DEPRECATED.value
        if replacement_id:
            adr.related_adrs.append(replacement_id)
            replacement_adr = self.adrs.get(replacement_id)
            if replacement_adr:
                replacement_adr.related_adrs.append(adr_id)

        adr.updated_at = datetime.now().isoformat()
        self._save_adr(adr)
        return adr

    def is_high_stakes(self, category: str) -> bool:
        try:
            cat = ADRCategory(category)
            return cat in HIGH_STAKES_CATEGORIES
        except ValueError:
            return False

    def generate_proposal(self, context: str, num_options: int = 3) -> ADRProposal:
        options = []

        example_options = [
            {
                "id": "option_1",
                "description": "Use centralized architecture with a single orchestrator",
                "pros": ["Simple to understand", "Easy to coordinate"],
                "cons": ["Single point of failure", "Harder to scale"],
            },
            {
                "id": "option_2",
                "description": "Use distributed agent mesh with peer-to-peer communication",
                "pros": ["More resilient", "Better scaling"],
                "cons": ["More complex coordination", "Harder to debug"],
            },
            {
                "id": "option_3",
                "description": "Use hierarchical decomposition with manager agents",
                "pros": ["Balanced complexity", "Good for large tasks"],
                "cons": ["Requires careful hierarchy design", "Latency in deep trees"],
            },
        ]

        options = example_options[:num_options]

        return ADRProposal(
            question=context,
            options=options,
            recommended=0,
            rationale="Option 1 provides the best balance of simplicity and functionality for initial implementation.",
        )

    def accept_proposal(self, proposal: ADRProposal, title: str, category: str) -> ADR:
        decision = "; ".join([f"{o['id']}: {o['description']}" for o in proposal.options])

        consequences = "\n".join([f"Pros: {', '.join(o.get('pros', []))}" for o in proposal.options])

        return self.create_adr(
            title=title,
            category=category,
            context=proposal.question,
            decision=decision,
            consequences=consequences,
            created_by="system",
        )

    def get_statistics(self) -> dict[str, Any]:
        stats = {"total": len(self.adrs), "by_status": {}, "by_category": {}, "high_stakes_count": 0}

        for adr in self.adrs.values():
            stats["by_status"][adr.status] = stats["by_status"].get(adr.status, 0) + 1
            stats["by_category"][adr.category] = stats["by_category"].get(adr.category, 0) + 1

            if self.is_high_stakes(adr.category):
                stats["high_stakes_count"] += 1

        return stats


adr_system = ADRSystem.get_instance()
