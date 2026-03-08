"""
Agent Debate & Consensus Protocol
===================================

Structured multi-agent debate where agents argue positions, critique
each other, and converge toward consensus through iterative rounds.

Supports multiple protocols:
- DebateProtocol: Adversarial for/against argumentation with voting
- ConsensusProtocol: Collaborative position refinement for knowledge tasks

Usage:
    from vetinari.orchestration.debate import DebateProtocol, ConsensusProtocol

    protocol = DebateProtocol()
    result = protocol.debate("Should we use microservices?", agents, rounds=3)
"""

import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class VotingStrategy(Enum):
    """How to tally votes in a debate."""
    MAJORITY = "majority"
    WEIGHTED = "weighted"           # Weight by quality score
    UNANIMOUS = "unanimous"


class DebateRole(Enum):
    """Role assigned to an agent in a debate."""
    PROPONENT = "proponent"
    OPPONENT = "opponent"
    JUDGE = "judge"
    NEUTRAL = "neutral"


@dataclass
class Argument:
    """A single argument made by an agent during a round."""
    agent_id: str
    agent_type: str
    role: str
    position: str           # "for", "against", "neutral"
    content: str            # The argument text
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.5
    round_number: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Vote:
    """A vote cast by an agent or judge."""
    voter_id: str
    voter_type: str
    position: str           # "for", "against", "abstain"
    weight: float = 1.0     # Quality-based weight
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DebateResult:
    """Outcome of a structured debate."""
    debate_id: str = ""
    topic: str = ""
    winner: str = ""                    # "for", "against", "consensus", "inconclusive"
    final_position: str = ""            # Synthesized conclusion
    rounds_completed: int = 0
    total_arguments: int = 0
    votes_for: int = 0
    votes_against: int = 0
    votes_abstain: int = 0
    weighted_score_for: float = 0.0
    weighted_score_against: float = 0.0
    arguments: List[Dict[str, Any]] = field(default_factory=list)
    votes: List[Dict[str, Any]] = field(default_factory=list)
    consensus_reached: bool = False
    convergence_history: List[float] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentPosition:
    """An agent's current position in a consensus protocol."""
    agent_id: str
    agent_type: str
    position: str
    confidence: float = 0.5
    iteration: int = 0
    revised: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DebateProtocol:
    """
    Structured adversarial debate between agents.

    Agents are assigned roles (proponent/opponent), argue in rounds,
    and a final vote determines the outcome.
    """

    def __init__(
        self,
        voting_strategy: VotingStrategy = VotingStrategy.WEIGHTED,
        min_confidence_threshold: float = 0.3,
        agent_caller: Optional[Callable] = None,
    ):
        """
        Args:
            voting_strategy: How to tally the final vote.
            min_confidence_threshold: Minimum confidence to count an argument.
            agent_caller: Callable(agent, prompt) -> str for invoking agents.
                          If None, uses a stub that returns empty strings.
        """
        self._voting = voting_strategy
        self._min_confidence = min_confidence_threshold
        self._call_agent = agent_caller or self._default_agent_caller
        logger.info("DebateProtocol initialized (voting=%s)", voting_strategy.value)

    def debate(
        self,
        topic: str,
        agents: List[Any],
        rounds: int = 3,
        context: Optional[Dict[str, Any]] = None,
    ) -> DebateResult:
        """
        Run a structured debate on a topic.

        Args:
            topic: The proposition to debate.
            agents: List of agent objects (must have .agent_type or a string id).
            rounds: Number of debate rounds.
            context: Optional context shared with all agents.

        Returns:
            DebateResult with winner, arguments, votes, and convergence data.
        """
        debate_id = f"debate_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        all_arguments: List[Argument] = []
        convergence: List[float] = []

        logger.info("Starting debate %s: '%s' with %d agents, %d rounds",
                     debate_id, topic[:80], len(agents), rounds)

        # Assign roles: first half proponents, second half opponents
        mid = max(1, len(agents) // 2)
        role_map: Dict[str, DebateRole] = {}
        for i, agent in enumerate(agents):
            aid = self._agent_id(agent)
            role_map[aid] = DebateRole.PROPONENT if i < mid else DebateRole.OPPONENT

        # Run debate rounds
        for round_num in range(1, rounds + 1):
            round_args: List[Argument] = []

            for agent in agents:
                aid = self._agent_id(agent)
                role = role_map[aid]
                position = "for" if role == DebateRole.PROPONENT else "against"

                # Build prompt with previous arguments for critique
                prior_args = [a for a in all_arguments if a.agent_id != aid]
                prompt = self._build_debate_prompt(
                    topic, position, round_num, rounds, prior_args, context
                )

                try:
                    response = self._call_agent(agent, prompt)
                    confidence = self._extract_confidence(response)

                    arg = Argument(
                        agent_id=aid,
                        agent_type=self._agent_type(agent),
                        role=role.value,
                        position=position,
                        content=response,
                        confidence=confidence,
                        round_number=round_num,
                    )
                    round_args.append(arg)
                except Exception as exc:
                    logger.warning("Agent %s failed in round %d: %s", aid, round_num, exc)

            all_arguments.extend(round_args)

            # Track convergence: how similar are for/against confidence levels
            for_conf = [a.confidence for a in round_args if a.position == "for"]
            against_conf = [a.confidence for a in round_args if a.position == "against"]
            avg_for = sum(for_conf) / len(for_conf) if for_conf else 0.5
            avg_against = sum(against_conf) / len(against_conf) if against_conf else 0.5
            spread = abs(avg_for - avg_against)
            convergence.append(spread)

            logger.debug("Round %d: %d arguments, spread=%.2f", round_num, len(round_args), spread)

        # Voting phase
        votes = self._collect_votes(topic, agents, all_arguments, role_map)

        # Tally results
        result = self._tally_votes(
            debate_id, topic, all_arguments, votes, convergence, rounds
        )
        result.duration_seconds = time.time() - start_time

        logger.info(
            "Debate %s concluded: winner=%s, for=%d, against=%d (%.1fs)",
            debate_id, result.winner, result.votes_for, result.votes_against,
            result.duration_seconds,
        )
        return result

    def _build_debate_prompt(
        self, topic: str, position: str, round_num: int, total_rounds: int,
        prior_args: List[Argument], context: Optional[Dict[str, Any]],
    ) -> str:
        """Build a debate prompt for an agent."""
        ctx_str = ""
        if context:
            ctx_str = f"\nContext: {context}\n"

        prior_str = ""
        if prior_args:
            summaries = []
            for a in prior_args[-6:]:  # Last 6 arguments for context window
                summaries.append(f"  [{a.role}] {a.content[:200]}")
            prior_str = "\nPrevious arguments:\n" + "\n".join(summaries) + "\n"

        return (
            f"DEBATE Round {round_num}/{total_rounds}\n"
            f"Topic: {topic}\n"
            f"Your position: {position.upper()}\n"
            f"{ctx_str}{prior_str}\n"
            f"Provide a clear, evidence-based argument {position} the topic. "
            f"Address counterarguments from previous rounds. "
            f"End with 'Confidence: X.X' (0.0 to 1.0)."
        )

    def _extract_confidence(self, response: str) -> float:
        """Extract confidence score from agent response."""
        import re
        match = re.search(r'[Cc]onfidence:\s*([\d.]+)', response)
        if match:
            try:
                val = float(match.group(1))
                return max(0.0, min(1.0, val))
            except ValueError:
                pass
        return 0.5

    def _collect_votes(
        self, topic: str, agents: List[Any],
        arguments: List[Argument], role_map: Dict[str, DebateRole],
    ) -> List[Vote]:
        """Collect votes from all agents after the debate."""
        votes: List[Vote] = []
        summary = self._summarize_arguments(arguments)

        for agent in agents:
            aid = self._agent_id(agent)
            prompt = (
                f"VOTE on debate topic: {topic}\n\n"
                f"Summary of arguments:\n{summary}\n\n"
                f"Cast your vote: 'for', 'against', or 'abstain'. "
                f"Provide brief reasoning."
            )
            try:
                response = self._call_agent(agent, prompt)
                position = self._extract_vote_position(response)
                quality = self._estimate_quality(agent, arguments, aid)
                votes.append(Vote(
                    voter_id=aid,
                    voter_type=self._agent_type(agent),
                    position=position,
                    weight=quality,
                    reasoning=response[:500],
                ))
            except Exception as exc:
                logger.warning("Agent %s failed to vote: %s", aid, exc)

        return votes

    def _tally_votes(
        self, debate_id: str, topic: str, arguments: List[Argument],
        votes: List[Vote], convergence: List[float], rounds: int,
    ) -> DebateResult:
        """Tally votes and determine the debate outcome."""
        v_for = [v for v in votes if v.position == "for"]
        v_against = [v for v in votes if v.position == "against"]
        v_abstain = [v for v in votes if v.position == "abstain"]

        if self._voting == VotingStrategy.WEIGHTED:
            w_for = sum(v.weight for v in v_for)
            w_against = sum(v.weight for v in v_against)
            winner = "for" if w_for > w_against else "against" if w_against > w_for else "inconclusive"
        elif self._voting == VotingStrategy.UNANIMOUS:
            if not v_against and len(v_for) > 0:
                winner = "for"
            elif not v_for and len(v_against) > 0:
                winner = "against"
            else:
                winner = "inconclusive"
        else:  # MAJORITY
            winner = "for" if len(v_for) > len(v_against) else "against" if len(v_against) > len(v_for) else "inconclusive"

        return DebateResult(
            debate_id=debate_id,
            topic=topic,
            winner=winner,
            final_position=f"The debate concluded '{winner}' the proposition.",
            rounds_completed=rounds,
            total_arguments=len(arguments),
            votes_for=len(v_for),
            votes_against=len(v_against),
            votes_abstain=len(v_abstain),
            weighted_score_for=sum(v.weight for v in v_for),
            weighted_score_against=sum(v.weight for v in v_against),
            arguments=[a.to_dict() for a in arguments],
            votes=[v.to_dict() for v in votes],
            consensus_reached=(winner != "inconclusive" and len(convergence) > 1 and convergence[-1] < 0.2),
            convergence_history=convergence,
        )

    def _summarize_arguments(self, arguments: List[Argument]) -> str:
        """Create a brief summary of all arguments."""
        for_args = [a for a in arguments if a.position == "for"]
        against_args = [a for a in arguments if a.position == "against"]
        lines = ["FOR:"]
        for a in for_args[-3:]:
            lines.append(f"  - {a.content[:150]}")
        lines.append("AGAINST:")
        for a in against_args[-3:]:
            lines.append(f"  - {a.content[:150]}")
        return "\n".join(lines)

    def _extract_vote_position(self, response: str) -> str:
        """Extract vote position from response text."""
        lower = response.lower()
        if "abstain" in lower:
            return "abstain"
        # Count keyword occurrences to determine intent
        for_count = lower.count(" for ") + lower.count("vote: for") + lower.count("vote for")
        against_count = lower.count("against") + lower.count("vote: against")
        if for_count > against_count:
            return "for"
        elif against_count > for_count:
            return "against"
        return "abstain"

    def _estimate_quality(self, agent: Any, arguments: List[Argument], agent_id: str) -> float:
        """Estimate agent quality weight based on argument consistency."""
        agent_args = [a for a in arguments if a.agent_id == agent_id]
        if not agent_args:
            return 0.5
        avg_conf = sum(a.confidence for a in agent_args) / len(agent_args)
        return max(0.1, min(1.0, avg_conf))

    def _agent_id(self, agent: Any) -> str:
        """Extract an agent identifier."""
        if hasattr(agent, "agent_type"):
            return agent.agent_type.value if hasattr(agent.agent_type, "value") else str(agent.agent_type)
        return str(id(agent))

    def _agent_type(self, agent: Any) -> str:
        """Extract agent type string."""
        if hasattr(agent, "agent_type"):
            return agent.agent_type.value if hasattr(agent.agent_type, "value") else str(agent.agent_type)
        return type(agent).__name__

    @staticmethod
    def _default_agent_caller(agent: Any, prompt: str) -> str:
        """Default stub caller when no real agent caller is provided."""
        if hasattr(agent, "_infer"):
            return agent._infer(prompt)
        return ""


class ConsensusProtocol:
    """
    Collaborative consensus-building protocol for knowledge tasks.

    Unlike debate, all agents start with individual positions and
    iteratively refine them based on peer critique until convergence.
    """

    def __init__(
        self,
        convergence_threshold: float = 0.85,
        max_iterations: int = 5,
        agent_caller: Optional[Callable] = None,
    ):
        self._convergence_threshold = convergence_threshold
        self._max_iterations = max_iterations
        self._call_agent = agent_caller or DebateProtocol._default_agent_caller
        logger.info("ConsensusProtocol initialized (threshold=%.2f)", convergence_threshold)

    def seek_consensus(
        self,
        topic: str,
        agents: List[Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> DebateResult:
        """
        Run an iterative consensus-building process.

        Each agent states their position, reviews others' positions,
        and revises until positions converge or max iterations reached.

        Args:
            topic: The question or topic to reach consensus on.
            agents: List of participating agents.
            context: Optional shared context.

        Returns:
            DebateResult with consensus_reached flag and final position.
        """
        consensus_id = f"consensus_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        all_arguments: List[Argument] = []
        convergence: List[float] = []

        logger.info("Starting consensus %s: '%s' with %d agents",
                     consensus_id, topic[:80], len(agents))

        # Initial positions
        positions: Dict[str, AgentPosition] = {}
        for agent in agents:
            aid = self._agent_id(agent)
            prompt = (
                f"CONSENSUS BUILDING\nTopic: {topic}\n"
                f"{'Context: ' + str(context) if context else ''}\n\n"
                f"State your position clearly. End with 'Confidence: X.X' (0.0 to 1.0)."
            )
            try:
                response = self._call_agent(agent, prompt)
                confidence = DebateProtocol._default_extract_confidence(response)
                positions[aid] = AgentPosition(
                    agent_id=aid,
                    agent_type=self._agent_type(agent),
                    position=response,
                    confidence=confidence,
                    iteration=0,
                )
                all_arguments.append(Argument(
                    agent_id=aid,
                    agent_type=self._agent_type(agent),
                    role="neutral",
                    position="neutral",
                    content=response,
                    confidence=confidence,
                    round_number=0,
                ))
            except Exception as exc:
                logger.warning("Agent %s failed initial position: %s", aid, exc)

        # Iterative refinement
        for iteration in range(1, self._max_iterations + 1):
            # Share all positions and ask for revision
            position_summary = self._format_positions(positions)

            revised_count = 0
            for agent in agents:
                aid = self._agent_id(agent)
                prompt = (
                    f"CONSENSUS ROUND {iteration}\nTopic: {topic}\n\n"
                    f"Current positions from all participants:\n{position_summary}\n\n"
                    f"Review other positions and revise yours if their arguments are compelling. "
                    f"Explain what you agree/disagree with. End with 'Confidence: X.X'."
                )
                try:
                    response = self._call_agent(agent, prompt)
                    confidence = DebateProtocol._default_extract_confidence(response)
                    old_pos = positions.get(aid)
                    revised = old_pos is not None and abs(confidence - old_pos.confidence) > 0.1
                    positions[aid] = AgentPosition(
                        agent_id=aid,
                        agent_type=self._agent_type(agent),
                        position=response,
                        confidence=confidence,
                        iteration=iteration,
                        revised=revised,
                    )
                    if revised:
                        revised_count += 1
                    all_arguments.append(Argument(
                        agent_id=aid,
                        agent_type=self._agent_type(agent),
                        role="neutral",
                        position="neutral",
                        content=response,
                        confidence=confidence,
                        round_number=iteration,
                    ))
                except Exception as exc:
                    logger.warning("Agent %s failed in iteration %d: %s", aid, iteration, exc)

            # Measure convergence: average confidence agreement
            confidences = [p.confidence for p in positions.values()]
            if confidences:
                avg = sum(confidences) / len(confidences)
                agreement = 1.0 - (sum(abs(c - avg) for c in confidences) / len(confidences))
            else:
                agreement = 0.0
            convergence.append(agreement)

            logger.debug("Consensus iteration %d: agreement=%.2f, revised=%d",
                         iteration, agreement, revised_count)

            if agreement >= self._convergence_threshold:
                logger.info("Consensus reached at iteration %d (agreement=%.2f)",
                            iteration, agreement)
                break

        # Build final result
        consensus_reached = len(convergence) > 0 and convergence[-1] >= self._convergence_threshold
        final_position = self._synthesize_positions(positions) if consensus_reached else "No consensus reached"

        result = DebateResult(
            debate_id=consensus_id,
            topic=topic,
            winner="consensus" if consensus_reached else "inconclusive",
            final_position=final_position,
            rounds_completed=len(convergence),
            total_arguments=len(all_arguments),
            arguments=[a.to_dict() for a in all_arguments],
            consensus_reached=consensus_reached,
            convergence_history=convergence,
            duration_seconds=time.time() - start_time,
        )
        return result

    def _format_positions(self, positions: Dict[str, AgentPosition]) -> str:
        """Format all current positions for sharing."""
        lines = []
        for aid, pos in positions.items():
            lines.append(f"Agent {aid} (confidence {pos.confidence:.2f}):\n  {pos.position[:300]}")
        return "\n\n".join(lines)

    def _synthesize_positions(self, positions: Dict[str, AgentPosition]) -> str:
        """Create a synthesis of converged positions."""
        sorted_positions = sorted(positions.values(), key=lambda p: p.confidence, reverse=True)
        if sorted_positions:
            return sorted_positions[0].position
        return ""

    def _agent_id(self, agent: Any) -> str:
        if hasattr(agent, "agent_type"):
            return agent.agent_type.value if hasattr(agent.agent_type, "value") else str(agent.agent_type)
        return str(id(agent))

    def _agent_type(self, agent: Any) -> str:
        if hasattr(agent, "agent_type"):
            return agent.agent_type.value if hasattr(agent.agent_type, "value") else str(agent.agent_type)
        return type(agent).__name__


# Helper so ConsensusProtocol can reuse confidence extraction
DebateProtocol._default_extract_confidence = staticmethod(
    lambda response: DebateProtocol(agent_caller=lambda a, p: "")._extract_confidence(response)
)


def select_protocol(task_type: str) -> str:
    """
    Select the appropriate debate protocol based on task type.

    Args:
        task_type: Type of task (e.g., "design_decision", "knowledge", "risk_assessment").

    Returns:
        "debate" or "consensus" indicating which protocol to use.
    """
    adversarial_types = {"design_decision", "architecture", "risk_assessment", "tradeoff", "strategy"}
    consensus_types = {"knowledge", "analysis", "documentation", "research", "summarization"}

    if task_type.lower() in adversarial_types:
        return "debate"
    if task_type.lower() in consensus_types:
        return "consensus"
    # Default: consensus for knowledge-heavy, debate for decision tasks
    return "consensus"
