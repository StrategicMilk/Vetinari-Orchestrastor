# Vetinari Phase 1 Sandbox Specification

## Overview

This document defines the two-layer sandbox architecture for Phase 1:
- **Layer 1**: In-Process Safe Sandbox for quick agent code execution
- **Layer 2**: External Plugin Sandbox for third-party plugins

---

## 1. Layer 1: In-Process Safe Sandbox

### 1.1 Purpose

Quick, lightweight execution of agent-generated code snippets within the same process. Designed for:
- Simple transformations
- Data processing
- Quick calculations

### 1.2 Design Principles

1. **Minimal Surface**: Restrict available functions to safe subset
2. **Timeout Enforcement**: Hard timeouts prevent infinite loops
3. **Memory Limits**: Track and limit memory usage
4. **No Side Effects**: Block file I/O, network, and system calls

### 1.3 Allowed Built-ins

```python
ALLOWED_BUILTINS = {
    # Type constructors
    'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple',
    'type', 'range', 'enumerate', 'zip', 'map', 'filter',
    
    # String operations
    'strip', 'split', 'join', 'replace', 'upper', 'lower', 'title',
    'startswith', 'endswith', 'find', 'count', 'format',
    
    # Collection operations
    'len', 'sum', 'min', 'max', 'sorted', 'reversed', 'any', 'all',
    'append', 'extend', 'pop', 'push', 'shift', 'get', 'keys', 'values',
    'items', 'copy', 'clear',
    
    # Math operations
    'abs', 'round', 'pow', 'sqrt', 'sin', 'cos', 'tan', 'log', 'exp',
    
    # Utilities
    'print', 'isinstance', 'hasattr', 'getattr', 'setattr',
    'isinstance', 'issubclass', 'callable', 'id', 'hash',
}
```

### 1.4 Blocked Operations

```python
BLOCKED_BUILTINS = {
    # File I/O
    'open', 'file', 'read', 'write',
    
    # Dynamic code execution
    'eval', 'exec', 'compile', '__import__',
    
    # System
    'exit', 'quit', 'quit', 'input', 'raw_input',
    
    # Reflection
    'vars', 'dir', 'globals', 'locals', 'memoryview',
    
    # Attributes
    '__builtins__', '__globals__', '__locals__',
}
```

### 1.5 Implementation

```python
import signal
import threading
import tracemalloc
from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class SandboxResult:
    success: bool
    result: Any = None
    error: str = ""
    execution_time_ms: int = 0
    memory_used_mb: float = 0.0
    execution_id: str = ""

class InProcessSandbox:
    """In-process sandbox for safe code execution"""
    
    def __init__(
        self, 
        timeout: int = 30,
        max_memory_mb: int = 512,
        allowed_builtins: set = None
    ):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.allowed_builtins = allowed_builtins or ALLOWED_BUILTINS
        self.execution_counter = 0
        
    def execute(self, code: str, context: Dict[str, Any] = None) -> SandboxResult:
        """Execute code in restricted environment"""
        import uuid
        import time
        
        execution_id = f"exec_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        # Start memory tracking
        tracemalloc.start()
        
        # Prepare restricted globals
        restricted_globals = {
            '__builtins__': {k: v for k, v in __builtins__.items() 
                           if k in self.allowed_builtins}
        }
        
        # Add safe context
        if context:
            restricted_globals.update(context)
            
        # Setup timeout handler
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Execution timeout after {self.timeout}s")
            
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.timeout)
        
        try:
            # Execute code
            result = eval(code, restricted_globals, {})
            signal.alarm(0)  # Cancel alarm
            
            # Get memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            return SandboxResult(
                success=True,
                result=result,
                execution_time_ms=int((time.time() - start_time) * 1000),
                memory_used_mb=peak / (1024 * 1024),
                execution_id=execution_id
            )
            
        except TimeoutError as e:
            signal.alarm(0)
            tracemalloc.stop()
            return SandboxResult(
                success=False,
                error=f"Timeout: {str(e)}",
                execution_time_ms=self.timeout * 1000,
                execution_id=execution_id
            )
            
        except Exception as e:
            signal.alarm(0)
            tracemalloc.stop()
            return SandboxResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                execution_time_ms=int((time.time() - start_time) * 1000),
                execution_id=execution_id
            )
```

### 1.6 Configuration

```yaml
# sandbox.yaml
sandbox:
  in_process:
    timeout_seconds: 30
    max_memory_mb: 512
    allowed_builtins:
      - str
      - int
      - float
      - bool
      - list
      - dict
      - set
      - tuple
      - range
      - enumerate
      - zip
      - map
      - filter
      - len
      - sum
      - min
      - max
      - sorted
      - any
      - all
      - abs
      - round
      - print
    blocked_builtins:
      - open
      - eval
      - exec
      - compile
      - __import__
      - input
      - exit
```

---

## 2. Layer 2: External Plugin Sandbox

### 2.1 Purpose

Isolated environment for running third-party plugins:
- **Process Isolation**: Plugins run in separate process
- **Permission System**: Explicit allow-listing
- **Audit Logging**: All operations logged
- **Resource Limits**: Memory and CPU quotas

### 2.2 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Vetinari Core                          │
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │  Plugin     │    │  Sandbox    │    │   Audit     │   │
│  │  Loader     │───▶│  Manager    │───▶│   Logger    │   │
│  └─────────────┘    └─────────────┘    └─────────────┘   │
│                            │                               │
└────────────────────────────┼───────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │    External Plugin Process   │
              │  ┌────────────────────────┐  │
              │  │    Plugin Sandbox      │  │
              │  │  - Restricted imports │  │
              │  │  - Timeout enforcement │  │
              │  │  - Memory limits       │  │
              │  └────────────────────────┘  │
              └──────────────────────────────┘
```

### 2.3 Plugin Interface

```python
@dataclass
class PluginManifest:
    """Plugin metadata"""
    name: str
    version: str
    author: str
    description: str
    permissions: List[str]
    hooks: List[str]
    dependencies: List[str]

@dataclass  
class PluginInstance:
    """Loaded plugin instance"""
    manifest: PluginManifest
    module: Any
    permissions: Set[str]
    loaded_at: str
    
class ExternalPluginSandbox:
    """Sandbox for external plugins"""
    
    ALLOWED_HOOKS = {
        'read_file': {'description': 'Read file contents'},
        'write_file': {'description': 'Write file contents'},
        'search_code': {'description': 'Search code'},
        'http_request': {'description': 'HTTP requests', 'domains': []},
    }
    
    def __init__(
        self,
        plugin_dir: str = "./plugins",
        timeout: int = 300,
        max_memory_mb: int = 2048
    ):
        self.plugin_dir = Path(plugin_dir)
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.loaded_plugins: Dict[str, PluginInstance] = {}
        self.audit_log: List[AuditEntry] = []
        
    def discover_plugins(self) -> List[PluginManifest]:
        """Discover available plugins"""
        manifests = []
        for plugin_path in self.plugin_dir.iterdir():
            if plugin_path.is_dir():
                manifest_file = plugin_path / "manifest.yaml"
                if manifest_file.exists():
                    with open(manifest_file) as f:
                        data = yaml.safe_load(f)
                        manifests.append(PluginManifest(**data))
        return manifests
        
    def load_plugin(self, name: str) -> PluginInstance:
        """Load a plugin into sandbox"""
        # Verify permissions
        # Load with restricted imports
        # Return instance
        
    def execute_hook(
        self, 
        plugin_name: str, 
        hook_name: str, 
        params: dict
    ) -> Any:
        """Execute a plugin hook"""
        # Check permissions
        # Log operation
        # Execute with timeout
        # Return result
        
    def get_audit_log(self) -> List[AuditEntry]:
        """Get audit log entries"""
        return self.audit_log
```

### 2.4 Plugin Manifest Format

```yaml
# plugins/my-plugin/manifest.yaml
name: my-plugin
version: 1.0.0
author: "Author Name"
description: "My awesome plugin"

permissions:
  - read_file
  - write_file
  - search_code

hooks:
  - on_task_start
  - on_task_complete
  - on_plan_complete

dependencies:
  - requests

# hooks/my-plugin.py
def on_task_start(task):
    print(f"Task {task.task_id} starting")
    return task
    
def on_task_complete(task, result):
    print(f"Task {task.task_id} completed")
    return result
```

### 2.5 Permission System

```python
class PermissionChecker:
    """Check and enforce permissions"""
    
    PERMISSION_MATRIX = {
        'read_file': {
            'description': 'Read file contents',
            'risk': 'medium',
            'requires': ['file_path'],
            'blocked_paths': ['/etc', '/root', '~/.ssh']
        },
        'write_file': {
            'description': 'Write file contents', 
            'risk': 'high',
            'requires': ['file_path'],
            'blocked_paths': ['/etc', '/root', '/bin']
        },
        'http_request': {
            'description': 'Make HTTP requests',
            'risk': 'medium',
            'requires': ['url'],
            'allowed_domains': []  # Empty = no restrictions
        },
        'shell_exec': {
            'description': 'Execute shell commands',
            'risk': 'critical',
            'requires': ['command'],
            'allowed_commands': []  # Empty = none allowed
        }
    }
    
    def check_permission(self, permission: str, params: dict) -> bool:
        """Check if permission is allowed"""
        # Check permission exists
        # Check params
        # Check blocked paths/domains
        # Return True/False
```

### 2.6 Audit Logging

```python
@dataclass
class AuditEntry:
    timestamp: str
    plugin_name: str
    operation: str
    params: dict
    result: Any
    success: bool
    duration_ms: int
    
class AuditLogger:
    """Log all sandbox operations"""
    
    def __init__(self, log_dir: str = "./logs/sandbox"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def log(self, entry: AuditEntry):
        """Log an audit entry"""
        log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(asdict(entry)) + '\n')
```

### 2.7 Configuration

```yaml
# sandbox.yaml
sandbox:
  external:
    plugin_dir: "./plugins"
    isolation: "process"  # or "container" for Phase 2
    timeout_seconds: 300
    max_memory_mb: 2048
    
    allowed_hooks:
      - read_file
      - write_file
      - search_code
      
    blocked_hooks:
      - shell_exec
      - network_raw
      - file_delete
      - process_spawn
      
    allowed_domains:
      - "api.github.com"
      - "api.openai.com"
      
    # Plugin security
    require_signature: false  # Phase 2: true
    allow_network: true
    allow_file_write: true
    
    # Audit
    audit_enabled: true
    audit_log_dir: "./logs/sandbox"
    audit_retention_days: 30
```

---

## 3. Sandbox Policy (Phase 1)

### 3.1 Policy Document

```yaml
# vetinari/sandbox_policy.yaml
version: "1.0"
policy:
  in_process:
    enabled: true
    default_timeout: 30
    max_memory_mb: 512
    
  external:
    enabled: true
    isolation: "process"
    default_timeout: 300
    max_memory_mb: 2048
    
  # Global rules
  rules:
    # Code execution
    allow_code_execution: true
    require_approval_for_external: true
    
    # Network
    allow_network: true
    blocked_domains: []
    
    # File system
    allow_file_read: true
    allow_file_write: true
    blocked_paths:
      - "/etc"
      - "/root"
      - "~/.ssh"
      - "*.pem"
      - "*.key"
      
  # Approval workflow
  approval:
    auto_approve_builtin: true
    require_approval_for:
      - shell_exec
      - network_raw
      - file_delete
```

### 3.2 Allowed Plugin Hooks (Phase 1)

| Hook | Description | Risk Level |
|------|-------------|-------------|
| read_file | Read file contents | Medium |
| write_file | Write file contents | High |
| search_code | Search code | Low |
| http_request | Make HTTP requests | Medium |

### 3.3 Blocked in Phase 1

| Hook | Description | Reason |
|------|-------------|--------|
| shell_exec | Execute shell commands | Too risky |
| network_raw | Raw socket access | Too risky |
| file_delete | Delete files | Too risky |
| process_spawn | Spawn new processes | Too risky |

---

## 4. API Integration

### 4.1 Execute Endpoint

```python
@app.route('/api/sandbox/execute', methods=['POST'])
def execute_sandbox():
    data = request.json
    
    sandbox_type = data.get('sandbox_type', 'in_process')
    code = data.get('code')
    timeout = data.get('timeout', 30)
    
    if sandbox_type == 'in_process':
        sandbox = InProcessSandbox(timeout=timeout)
        result = sandbox.execute(code)
    else:
        return ErrorResponse("External sandbox not implemented in Phase 1")
        
    return jsonify(asdict(result))
```

---

## 5. Testing Strategy

### 5.1 Unit Tests

- Test allowed/blocked builtins
- Test timeout enforcement
- Test memory limits
- Test error handling

### 5.2 Integration Tests

- Test plugin loading
- Test permission checking
- Test audit logging
- Test timeout propagation

### 5.3 Security Tests

- Attempt to escape sandbox
- Test blocked operations
- Test memory exhaustion
- Test CPU exhaustion

---

## 6. Success Criteria

- [ ] In-process sandbox executes code safely
- [ ] Timeouts enforced for all executions
- [ ] Memory limits tracked
- [ ] Plugin loader discovers plugins
- [ ] Permission system enforced
- [ ] Audit logging captures all operations
- [ ] Policy configurable via YAML

---

*Document Version: 1.0*
*Phase: 1*
*Last Updated: 2026-03-02*
