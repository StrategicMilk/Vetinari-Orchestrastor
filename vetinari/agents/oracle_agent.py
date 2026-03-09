"""Legacy redirect — use ConsolidatedOracleAgent directly."""
from vetinari.agents.consolidated.oracle_agent import ConsolidatedOracleAgent as OracleAgent, get_consolidated_oracle_agent as get_oracle_agent  # noqa: F401

__all__ = ["OracleAgent", "get_oracle_agent"]
