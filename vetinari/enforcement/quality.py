"""Quality gate enforcement for Vetinari agents.

Validates that an agent's output quality score meets the minimum threshold
configured in its ``AgentSpec.quality_gate_score``.
"""

from __future__ import annotations

import logging

from vetinari.agents.contracts import get_agent_spec
from vetinari.exceptions import QualityGateFailed
from vetinari.types import AgentType

logger = logging.getLogger(__name__)


class QualityGateEnforcer:
    """Validates agent output quality scores against the threshold in AgentSpec.

    Reads ``AgentSpec.quality_gate_score`` for the given agent type and
    raises ``QualityGateFailed`` if the provided score falls below the threshold.

    Example:
        >>> enforcer = QualityGateEnforcer()
        >>> enforcer.validate(AgentType.WORKER, quality_score=0.9)  # passes
        >>> enforcer.validate(AgentType.WORKER, quality_score=0.5)  # raises
    """

    def validate(self, agent_type: AgentType, quality_score: float) -> None:
        """Validate that quality_score meets the agent's minimum threshold.

        Args:
            agent_type: The agent type whose specification provides the threshold.
            quality_score: The quality score produced by (or assigned to) the
                agent's output.  Expected range is 0.0-1.0.

        Raises:
            QualityGateFailed: If quality_score < spec.quality_gate_score.
        """
        spec = get_agent_spec(agent_type)
        if spec is None:
            logger.warning(
                "No AgentSpec found for %s — skipping quality gate validation",
                agent_type,
            )
            return

        threshold = spec.quality_gate_score
        if quality_score < threshold:
            raise QualityGateFailed(
                f"Agent {agent_type.value!r} quality score {quality_score:.3f} is below "
                f"the required threshold of {threshold:.3f}. "
                "Improve output quality or lower quality_gate_score in AgentSpec.",
                agent_type=agent_type.value,
                quality_score=quality_score,
                threshold=threshold,
            )

        logger.debug(
            "Quality gate passed for %s: %.3f >= %.3f",
            agent_type.value,
            quality_score,
            threshold,
        )
