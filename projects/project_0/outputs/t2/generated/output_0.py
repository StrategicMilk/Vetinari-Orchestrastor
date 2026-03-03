# filename: workflow_engine.py
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Status(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

@dataclass
class TaskResult:
    task_id: str
    status: Status
    output: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class BaseTask(ABC):
    """Abstract base class for all workflow tasks."""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
    
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> TaskResult:
        pass

# --- Concrete Task Implementations ---

class DataValidationTask(BaseTask):
    """Validates input data before processing."""
    
    def execute(self, context: Dict[str, Any]) -> TaskResult:
        try:
            data = context.get("data", [])
            if not isinstance(data, list) or len(data) == 0:
                raise ValueError("Data must be a non-empty list.")
            
            # Simulate validation logic
            validated_items = [item for item in data if isinstance(item, (int, float))]
            
            logger.info(f"Validation passed. {len(validated_items)} items valid.")
            return TaskResult(
                task_id=self.task_id,
                status=Status.COMPLETED,
                output={"validated_count": len(validated_items)}
            )
        except Exception as e:
            logger.error(f"Task {self.task_id} failed: {e}")
            return TaskResult(
                task_id=self.task_id,
                status=Status.FAILED,
                error_message=str(e)
            )

class DataProcessingTask(BaseTask):
    """Processes validated data."""
    
    def execute(self, context: Dict[str, Any]) -> TaskResult:
        try:
            # Simulate processing
            raw_data = context.get("data", [])
            processed_data = [x * 2 for x in raw_data if isinstance(x, (int, float))]
            
            logger.info(f"Processing complete. Generated {len(processed_data)} results.")
            return TaskResult(
                task_id=self.task_id,
                status=Status.COMPLETED,
                output={"processed_data": processed_data}
            )
        except Exception as e:
            logger.error(f"Task {self.task_id} failed: {e}")
            return TaskResult(
                task_id=self.task_id,
                status=Status.FAILED,
                error_message=str(e)
            )

# --- Workflow Orchestrator ---

class WorkflowEngine:
    def __init__(self, tasks: List[BaseTask]):
        self.tasks = tasks
    
    def run(self, initial_context: Dict[str, Any]) -> List[TaskResult]:
        """Execute the workflow sequentially."""
        context = initial_context.copy()
        results: List[TaskResult] = []
        
        for task in self.tasks:
            logger.info(f"Starting Task: {task.task_id}")
            
            # If previous task failed and we want to stop on error, uncomment below
            # if results and results[-1].status == Status.FAILED:
            #     break
            
            result = task.execute(context)
            results.append(result)
            
            # Update context with output for the next task
            if result.status == Status.COMPLETED and result.output:
                context.update(result.output)
                
        return results

# --- Main Execution Block ---

if __name__ == "__main__":
    # 1. Define Tasks
    validation_task = DataValidationTask(task_id="validate_input")
    processing_task = DataProcessingTask(task_id="process_data")
    
    # 2. Initialize Engine
    engine = WorkflowEngine(tasks=[validation_task, processing_task])
    
    # 3. Run Workflow
    sample_data = {"data": [10, "invalid", 20, 30.5]}
    final_results = engine.run(sample_data)
    
    # 4. Output Summary
    print("\n--- Workflow Execution Summary ---")
    for res in final_results:
        print(f"Task {res.task_id}: {res.status.value}")
        if res.error_message:
            print(f"  Error: {res.error_message}")