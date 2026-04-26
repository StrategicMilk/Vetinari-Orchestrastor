"""Training subsystem: agent fine-tuning, curriculum design, and data generation."""

from __future__ import annotations

from vetinari.training.adapter_registry import (  # noqa: VET123 — list functions have no external callers but removing causes VET120
    list_adapters_by_task_type,
    list_deployed_adapters,
)
from vetinari.training.agent_trainer import AgentTrainer
from vetinari.training.continual_learning import (
    LoRAAdapterManager,
    ReplayBuffer,
    STABLERegularizer,
)
from vetinari.training.curriculum import (
    CurriculumPhase,
    TrainingActivity,
    TrainingActivityType,
    TrainingCurriculum,
)
from vetinari.training.data_seeder import SeedDataset, TrainingDataSeeder
from vetinari.training.external_data import DatasetInfo, DatasetSpec, ExternalDataManager
from vetinari.training.idle_scheduler import IdleDetector, IdleTrainingJob, TrainingScheduler
from vetinari.training.pipeline import ContextDistillationDatasetBuilder, DistillationDatasetInfo
from vetinari.training.synthetic_data import (  # noqa: VET123 — generate_reasoning_chains has no external callers but removing causes VET120
    MagpieGenerator,
    StrategyDistiller,
    SyntheticDataGenerator,
    generate_reasoning_chains,
)
from vetinari.training.synthetic_generators import (  # noqa: VET123 — store_distilled_strategies has no external callers but removing causes VET120
    store_distilled_strategies,
)
from vetinari.training.validation import PostTrainingValidator, PreTrainingValidator

__all__ = [
    "AgentTrainer",
    "ContextDistillationDatasetBuilder",
    "CurriculumPhase",
    "DatasetInfo",
    "DatasetSpec",
    "DistillationDatasetInfo",
    "ExternalDataManager",
    "IdleDetector",
    "IdleTrainingJob",
    "LoRAAdapterManager",
    "MagpieGenerator",
    "PostTrainingValidator",
    "PreTrainingValidator",
    "ReplayBuffer",
    "STABLERegularizer",
    "SeedDataset",
    "StrategyDistiller",
    "SyntheticDataGenerator",
    "TrainingActivity",
    "TrainingActivityType",
    "TrainingCurriculum",
    "TrainingDataSeeder",
    "TrainingScheduler",
    "generate_reasoning_chains",
    "list_adapters_by_task_type",
    "list_deployed_adapters",
    "store_distilled_strategies",
]
