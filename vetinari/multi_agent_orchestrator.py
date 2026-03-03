import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING = "waiting"

class AgentType(Enum):
    EXPLORER = "explorer"
    LIBRARIAN = "librarian"
    ORACLE = "oracle"
    UI_PLANNER = "ui_planner"
    BUILDER = "builder"
    RESEARCHER = "researcher"
    EVALUATOR = "evaluator"
    SYNTHESIZER = "synthesizer"

@dataclass
class AgentConfig:
    agent_type: AgentType
    name: str
    description: str
    model: str = ""
    thinking_variant: str = "medium"
    enabled: bool = True
    system_prompt: str = ""

@dataclass
class AgentTask:
    task_id: str
    agent_type: AgentType
    description: str
    prompt: str
    status: AgentStatus = AgentStatus.IDLE
    result: Any = None
    error: str = ""
    started_at: str = ""
    completed_at: str = ""
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        result = {
            "id": self.task_id,
            "description": self.description,
            "prompt": self.prompt,
            "status": self.status.value if isinstance(self.status, AgentStatus) else self.status,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "dependencies": self.dependencies,
            "agent": self.agent_type.value
        }
        return result

@dataclass
class Agent:
    agent_id: str
    agent_type: AgentType
    name: str
    description: str
    model: str
    thinking_variant: str
    status: AgentStatus
    state: str = "idle"  # For API compatibility
    tasks_completed: int = 0  # For API compatibility
    current_task: Optional[AgentTask] = None
    history: List[AgentTask] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "name": self.name,
            "description": self.description,
            "model": self.model,
            "thinking_variant": self.thinking_variant,
            "status": self.status.value,
            "state": self.state,
            "tasks_completed": self.tasks_completed,
            "current_task": self.current_task.to_dict() if self.current_task else None,
            "history": [h.to_dict() for h in self.history],
            "metrics": self.metrics
        }

class MultiAgentOrchestrator:
    DEFAULT_AGENTS = [
        AgentConfig(
            AgentType.EXPLORER,
            "Explorer",
            "Finds code fast — optimized for codebase search",
            "qwen2.5-coder-7b",
            "high"
        ),
        AgentConfig(
            AgentType.LIBRARIAN,
            "Librarian",
            "Digs deep into docs — researches libraries, finds examples",
            "qwen2.5-72b",
            "medium"
        ),
        AgentConfig(
            AgentType.ORACLE,
            "Oracle",
            "Thinks strategically — architecture decisions, debugging",
            "qwen3-30b-a3b",
            "xhigh"
        ),
        AgentConfig(
            AgentType.UI_PLANNER,
            "UI Planner",
            "Designs beautifully — CSS, animations, interfaces",
            "qwen2.5-vl-32b",
            "high"
        ),
        AgentConfig(
            AgentType.RESEARCHER,
            "Researcher",
            "Comprehensive exploration and research",
            "qwen2.5-72b",
            "medium"
        ),
        AgentConfig(
            AgentType.EVALUATOR,
            "Evaluator",
            "Evaluates results and quality",
            "qwen2.5-coder-7b",
            "medium"
        ),
        AgentConfig(
            AgentType.SYNTHESIZER,
            "Synthesizer",
            "Synthesizes results from multiple agents",
            "qwen2.5-72b",
            "high"
        )
    ]
    
    _instance = None
    
    @classmethod
    def get_instance(cls, lm_studio_host: str = "http://localhost:1234"):
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls(lm_studio_host)
        return cls._instance
    
    def __init__(self, lm_studio_host: str = "http://localhost:1234"):
        self.lm_studio_host = lm_studio_host
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, AgentTask] = {}
        self.task_counter = 0
        self.task_queue: List[Dict] = []  # For API compatibility
        self._initialize_agents()

    def _initialize_agents(self):
        for config in self.DEFAULT_AGENTS:
            if config.enabled:
                agent_id = f"agent_{config.agent_type.value}"
                agent = Agent(
                    agent_id=agent_id,
                    agent_type=config.agent_type,
                    name=config.name,
                    description=config.description,
                    model=config.model,
                    thinking_variant=config.thinking_variant,
                    status=AgentStatus.IDLE,
                    state="idle",
                    tasks_completed=0
                )
                self.agents[agent_id] = agent
    
    def initialize_agents(self):
        """Re-initialize agents (for API compatibility)."""
        self._initialize_agents()
        return [a.name for a in self.agents.values()]

    def create_task(
        self,
        agent_type: AgentType,
        description: str,
        prompt: str,
        dependencies: List[str] = None
    ) -> AgentTask:
        self.task_counter += 1
        task_id = f"task_{self.task_counter}"

        task = AgentTask(
            task_id=task_id,
            agent_type=agent_type,
            description=description,
            prompt=prompt,
            dependencies=dependencies or [],
            status=AgentStatus.IDLE
        )

        self.tasks[task_id] = task
        return task

    def run_parallel_tasks(self, tasks: List[AgentTask]) -> Dict[str, Any]:
        results = {}

        for task in tasks:
            task.started_at = datetime.now().isoformat()
            task.status = AgentStatus.RUNNING

            agent_id = f"agent_{task.agent_type.value}"
            if agent_id in self.agents:
                self.agents[agent_id].status = AgentStatus.RUNNING
                self.agents[agent_id].current_task = task

            try:
                result = self._execute_task(task)
                task.result = result
                task.status = AgentStatus.COMPLETED
                task.completed_at = datetime.now().isoformat()

                if agent_id in self.agents:
                    self.agents[agent_id].status = AgentStatus.IDLE
                    self.agents[agent_id].current_task = None
                    self.agents[agent_id].history.append(task)

                results[task.task_id] = result

            except Exception as e:
                task.status = AgentStatus.FAILED
                task.error = str(e)
                task.completed_at = datetime.now().isoformat()

                if agent_id in self.agents:
                    self.agents[agent_id].status = AgentStatus.FAILED

                results[task.task_id] = {"error": str(e)}

        return results

    def _execute_task(self, task: AgentTask) -> Any:
        agent_id = f"agent_{task.agent_type.value}"
        agent = self.agents.get(agent_id)

        if not agent:
            raise ValueError(f"Agent {agent_id} not found")

        logger.info(f"Executing task {task.task_id} with agent {agent.name}")

        time.sleep(0.1)

        return {
            "status": "completed",
            "agent": agent.name,
            "task": task.description,
            "timestamp": datetime.now().isoformat()
        }

    def get_agent_status(self) -> List[Dict]:
        return [agent.to_dict() for agent in self.agents.values()]

    def get_task_status(self) -> List[Dict]:
        return [task.to_dict() for task in self.tasks.values()]

    def get_orchestrator_status(self) -> Dict:
        return {
            "agents": self.get_agent_status(),
            "tasks": self.get_task_status(),
            "total_tasks": len(self.tasks),
            "completed_tasks": len([t for t in self.tasks.values() if t.status == AgentStatus.COMPLETED]),
            "failed_tasks": len([t for t in self.tasks.values() if t.status == AgentStatus.FAILED]),
            "running_tasks": len([t for t in self.tasks.values() if t.status == AgentStatus.RUNNING])
        }


# Global singleton instance
orchestrator = MultiAgentOrchestrator.get_instance()
