"""
CodeBridge - External coding service integration.

This module provides a bridge to external coding services (like CodeNomad)
for offloading heavier coding tasks.
"""

import os
import logging
import json
import uuid
import requests
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
from vetinari.types import CodingTaskType as BridgeTaskType, CodingTaskStatus as BridgeTaskStatus  # canonical enums

logger = logging.getLogger(__name__)


@dataclass
class BridgeTaskSpec:
    """Specification for a bridge task."""
    task_id: str = field(default_factory=lambda: f"bridge_{uuid.uuid4().hex[:8]}")
    task_type: BridgeTaskType = BridgeTaskType.IMPLEMENT
    language: str = "python"
    framework: str = ""
    repo_path: str = ""
    description: str = ""
    constraints: str = ""
    target_files: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BridgeTaskResult:
    """Result from a bridge task."""
    task_id: str
    status: BridgeTaskStatus = BridgeTaskStatus.PENDING
    success: bool = False
    output_files: List[str] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    logs: str = ""
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None


class CodeBridge:
    """Bridge to external coding services.
    
    This provides:
    - Task submission to external services
    - Status polling
    - Artifact retrieval
    - Fallback to in-process coder
    """
    
    def __init__(self, endpoint: str = None, api_key: str = None):
        self.endpoint = endpoint or os.environ.get("CODE_BRIDGE_ENDPOINT", "http://localhost:4096")
        self.api_key = api_key or os.environ.get("CODE_BRIDGE_API_KEY", "")
        self.enabled = os.environ.get("CODE_BRIDGE_ENABLED", "false").lower() in ("1", "true", "yes")
        self.timeout = int(os.environ.get("CODE_BRIDGE_TIMEOUT", "30"))
        
        logger.info("CodeBridge initialized (enabled=%s, endpoint=%s)", self.enabled, self.endpoint)
    
    def is_available(self) -> bool:
        """Check if the bridge is available."""
        if not self.enabled:
            return False
        
        try:
            response = requests.get(f"{self.endpoint}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning("Bridge health check failed: %s", e)
            return False
    
    def submit_task(self, spec: BridgeTaskSpec) -> BridgeTaskResult:
        """Submit a task to the external bridge."""
        
        if not self.enabled:
            logger.warning("CodeBridge is not enabled")
            return BridgeTaskResult(
                task_id=spec.task_id,
                status=BridgeTaskStatus.FAILED,
                success=False,
                error="CodeBridge is not enabled"
            )
        
        logger.info("Submitting task to bridge: %s (%s)", spec.task_id, spec.task_type.value)
        
        try:
            payload = {
                "task_id": spec.task_id,
                "task_type": spec.task_type.value,
                "language": spec.language,
                "framework": spec.framework,
                "repo_path": spec.repo_path,
                "description": spec.description,
                "constraints": spec.constraints,
                "target_files": spec.target_files,
                "context": spec.context
            }
            
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = requests.post(
                f"{self.endpoint}/tasks",
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return BridgeTaskResult(
                    task_id=spec.task_id,
                    status=BridgeTaskStatus(result.get("status", "pending")),
                    success=result.get("success", False),
                    output_files=result.get("output_files", []),
                    artifacts=result.get("artifacts", []),
                    logs=result.get("logs", ""),
                    error=result.get("error")
                )
            else:
                logger.error("Bridge task submission failed: %s", response.status_code)
                return BridgeTaskResult(
                    task_id=spec.task_id,
                    status=BridgeTaskStatus.FAILED,
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text}"
                )
                
        except requests.exceptions.Timeout:
            logger.error("Bridge task timed out after %ss", self.timeout)
            return BridgeTaskResult(
                task_id=spec.task_id,
                status=BridgeTaskStatus.FAILED,
                success=False,
                error="Task timed out"
            )
        except Exception as e:
            logger.error("Bridge task submission failed: %s", e)
            return BridgeTaskResult(
                task_id=spec.task_id,
                status=BridgeTaskStatus.FAILED,
                success=False,
                error=str(e)
            )
    
    def get_task_status(self, task_id: str) -> BridgeTaskResult:
        """Poll task status from the bridge."""
        
        if not self.enabled:
            return BridgeTaskResult(
                task_id=task_id,
                status=BridgeTaskStatus.FAILED,
                error="CodeBridge not enabled"
            )
        
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = requests.get(
                f"{self.endpoint}/tasks/{task_id}",
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return BridgeTaskResult(
                    task_id=task_id,
                    status=BridgeTaskStatus(result.get("status", "pending")),
                    success=result.get("success", False),
                    output_files=result.get("output_files", []),
                    artifacts=result.get("artifacts", []),
                    logs=result.get("logs", ""),
                    error=result.get("error")
                )
            else:
                return BridgeTaskResult(
                    task_id=task_id,
                    status=BridgeTaskStatus.FAILED,
                    error=f"HTTP {response.status_code}"
                )
                
        except Exception as e:
            logger.error("Failed to get task status: %s", e)
            return BridgeTaskResult(
                task_id=task_id,
                status=BridgeTaskStatus.FAILED,
                error=str(e)
            )
    
    def get_artifacts(self, task_id: str) -> List[Dict[str, Any]]:
        """Fetch artifacts from a completed task."""
        
        result = self.get_task_status(task_id)
        
        if result.status == BridgeTaskStatus.COMPLETED and result.success:
            return result.artifacts
        
        return []
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        
        if not self.enabled:
            return False
        
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = requests.delete(
                f"{self.endpoint}/tasks/{task_id}",
                headers=headers,
                timeout=self.timeout
            )
            
            return response.status_code in (200, 204)
            
        except Exception as e:
            logger.error("Failed to cancel task: %s", e)
            return False
    
    def list_tasks(self, status: Optional[BridgeTaskStatus] = None) -> List[BridgeTaskResult]:
        """List tasks from the bridge."""
        
        if not self.enabled:
            return []
        
        try:
            params = {}
            if status:
                params["status"] = status.value
            
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = requests.get(
                f"{self.endpoint}/tasks",
                params=params,
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                tasks = response.json()
                return [
                    BridgeTaskResult(
                        task_id=t.get("task_id"),
                        status=BridgeTaskStatus(t.get("status", "pending")),
                        success=t.get("success", False),
                        output_files=t.get("output_files", []),
                        artifacts=t.get("artifacts", []),
                        logs=t.get("logs", ""),
                        error=t.get("error")
                    )
                    for t in tasks
                ]
            
        except Exception as e:
            logger.error("Failed to list tasks: %s", e)
        
        return []


_code_bridge: Optional[CodeBridge] = None


def get_code_bridge() -> CodeBridge:
    """Get or create the global code bridge instance."""
    global _code_bridge
    if _code_bridge is None:
        _code_bridge = CodeBridge()
    return _code_bridge


def init_code_bridge(endpoint: str = None, api_key: str = None) -> CodeBridge:
    """Initialize a new code bridge instance."""
    global _code_bridge
    _code_bridge = CodeBridge(endpoint=endpoint, api_key=api_key)
    return _code_bridge
