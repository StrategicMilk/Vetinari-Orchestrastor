# core/workflow_engine.py

import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class WorkflowStep:
    name: str
    action: Callable[[Dict[str, Any]], Dict[str, Any]]
    on_failure: Optional[Callable[[Exception], None]] = None
    status: StepStatus = StepStatus.PENDING

@dataclass
class WorkflowResult:
    workflow_id: str
    success: bool
    results: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

class WorkflowEngine:
    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.steps: List[WorkflowStep] = []
        self.global_context: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def add_step(self, name: str, action: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """Registers a new step to the workflow pipeline."""
        step = WorkflowStep(name=name, action=action)
        self.steps.append(step)
        logger.info(f"Added step '{name}' to workflow {self.workflow_id}")
        return self

    def _execute_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Executes a single step with error handling and state updates."""
        step.status = StepStatus.RUNNING
        logger.info(f"Executing step: {step.name}")

        try:
            # Execute the action with current context
            result = step.action(self.global_context)
            
            # Update global context with new data (merge results)
            if isinstance(result, dict):
                self.global_context.update(result)

            step.status = StepStatus.COMPLETED
            logger.info(f"Step completed: {step.name}")
            return {"step": step.name, "status": "success", "data": result}

        except Exception as e:
            step.status = StepStatus.FAILED
            error_msg = f"Error in step '{step.name}': {str(e)}"
            
            # Call failure handler if defined
            if step.on_failure:
                try:
                    step.on_failure(e)
                except Exception as retry_error:
                    logger.error(f"Failure handler also failed for {step.name}: {retry_error}")

            logger.error(error_msg)
            raise  # Re-raise to stop workflow

    def run(self) -> WorkflowResult:
        """Runs the entire pipeline sequentially."""
        logger.info(f"Starting Workflow ID: {self.workflow_id}")
        start_time = time.time()
        
        results = []
        errors = []

        with self._lock:
            for step in self.steps:
                try:
                    result = self._execute_step(step)
                    results.append(result)
                except Exception as e:
                    errors.append(str(e))
                    # Decide whether to stop or continue on failure
                    # For this implementation, we stop the workflow on first error
                    logger.warning("Workflow halted due to step failure.")
                    break

        duration = time.time() - start_time
        success = len(errors) == 0
        
        final_result = WorkflowResult(
            workflow_id=self.workflow_id,
            success=success,
            results=results,
            errors=errors
        )

        logger.info(f"Workflow {self.workflow_id} finished in {duration:.2f}s. Success: {success}")
        return final_result

# --- Example Usage & Concrete Actions ---

def step_fetch_data(context: Dict[str, Any]) -> Dict[str, Any]:
    """Simulates fetching data from a source."""
    time.sleep(0.1)  # Simulate I/O
    return {"raw_data": "User Profile Data", "user_id": 12345}

def step_process_data(context: Dict[str, Any]) -> Dict[str, Any]:
    """Processes the raw data."""
    raw = context.get("raw_data")
    if not raw:
        raise ValueError("No data to process")
    
    # Simulate transformation
    processed = f"Processed: {raw}"
    return {"clean_data": processed}

def step_save_to_db(context: Dict[str, Any]) -> Dict[str, Any]:
    """Saves the processed data."""
    data = context.get("clean_data")
    if "Processed" not in str(data):
        raise ValueError("Data integrity check failed")
    
    # Simulate DB save
    return {"db_status": "saved", "record_id": "rec_abc123"}

if __name__ == "__main__":
    # 1. Initialize Engine
    engine = WorkflowEngine(workflow_id="wf_user_onboarding")

    # 2. Define Pipeline
    (engine
        .add_step("Fetch User Data", step_fetch_data)
        .add_step("Process & Validate", step_process_data)
        .add_step("Persist Record", step_save_to_db)
    )

    # 3. Execute
    result = engine.run()

    if result.success:
        print("\nWorkflow Successful!")
        for res in result.results:
            print(f"- {res['step']}: OK")
    else:
        print("\nWorkflow Failed!")
        for err in result.errors:
            print(f"- Error: {err}")