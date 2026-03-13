"""Legacy redirect — use ConsolidatedOracleAgent directly."""

from __future__ import annotations

from vetinari.agents.consolidated.oracle_agent import ConsolidatedOracleAgent as OracleAgent
from vetinari.agents.consolidated.oracle_agent import get_consolidated_oracle_agent as get_oracle_agent

__all__ = ["OracleAgent", "get_oracle_agent"]
