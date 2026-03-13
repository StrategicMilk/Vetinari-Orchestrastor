"""Legacy redirect — use QualityAgent directly."""

from __future__ import annotations

from vetinari.agents.consolidated.quality_agent import QualityAgent as SecurityAuditorAgent
from vetinari.agents.consolidated.quality_agent import get_quality_agent as get_security_auditor_agent

__all__ = ["SecurityAuditorAgent", "get_security_auditor_agent"]
