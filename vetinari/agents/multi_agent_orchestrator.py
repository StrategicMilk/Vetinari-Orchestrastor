"""
Multi-Agent Orchestrator
========================
Wraps the TwoLayerOrchestrator to provide a stable multi-agent orchestration
interface used by the web UI agent endpoints.
"""

import logging
import threading
from typing import Any, Dict, List, Optional

from vetinari.types import AgentType

logger = logging.getLogger(__name__)

# Agent state constants
AGENT_STATE_IDLE = "idle"
AGENT_STATE_RUNNING = "running"
AGENT_STATE_ERROR = "error"


class AgentStatus:
    """Lightweight status snapshot for a single agent."""

    def __init__(self, name: str, agent_type: str):
        self.name = name
        self.agent_type = agent_type
        self.state = AGENT_STATE_IDLE
        self.tasks_completed = 0
        self.current_task: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "agent_type": self.agent_type,
            "state": self.state,
            "tasks_completed": self.tasks_completed,
            "current_task": self.current_task,
        }


class MultiAgentOrchestrator:
    """
    Multi-agent orchestrator wrapping TwoLayerOrchestrator.
    Provides agent status tracking and task dispatch.
    """

    _instance: Optional["MultiAgentOrchestrator"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.agents: Dict[str, AgentStatus] = {}
        self.task_queue: List[Dict[str, Any]] = []
        self._initialized = False

    @classmethod
    def get_instance(cls) -> Optional["MultiAgentOrchestrator"]:
        return cls._instance

    @classmethod
    def get_or_create(cls) -> "MultiAgentOrchestrator":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def initialize_agents(self) -> None:
        """Initialize status trackers for all known agent types."""
        self.agents = {
            a.value: AgentStatus(name=a.value.replace("_", " ").title(), agent_type=a.value)
            for a in AgentType
        }
        self._initialized = True
        logger.info(f"MultiAgentOrchestrator initialized with {len(self.agents)} agents")

    def get_agent_status(self) -> List[Dict[str, Any]]:
        """Return status for all agents."""
        if not self._initialized:
            self.initialize_agents()
        return [a.to_dict() for a in self.agents.values()]

    def dispatch_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a task to the appropriate agent via TwoLayerOrchestrator."""
        try:
            from vetinari.two_layer_orchestration import TwoLayerOrchestrator
            orch = TwoLayerOrchestrator()
            results = orch.generate_and_execute(
                goal=task.get("description", ""),
                context=task.get("context", {})
            )
            return {"status": "dispatched", "results": results}
        except Exception as e:
            logger.error(f"Task dispatch failed: {e}")
            return {"status": "error", "error": str(e)}
