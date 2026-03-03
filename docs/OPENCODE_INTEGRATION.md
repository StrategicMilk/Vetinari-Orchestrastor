# Vetinari OpenCode Integration Guide

## Overview

This document outlines the integration of OpenCode's advanced LLM orchestration patterns into Vetinari, making it the most comprehensive, user-friendly, and effective LLM orchestration tool available.

### Key Integration Points

The integration adds five major components inspired by OpenCode's architecture:

1. **Execution Context System** - Agent-based execution modes with permission enforcement
2. **Standardized Tool Interface** - Unified tool definitions with metadata and safety checks
3. **Provider Agnosticism Layer** - Multi-provider support with intelligent provider selection
4. **Enhanced CLI** - Rich user feedback and explicit mode indicators
5. **Verification Pipeline** - Comprehensive post-execution validation and security scanning

---

## 1. Execution Context System

### Overview

Inspired by OpenCode's agent model (build/plan), Vetinari now supports multiple execution modes with distinct permission levels:

- **PLANNING Mode** (Read-only): For analysis, planning, and code exploration
- **EXECUTION Mode** (Full access): For implementing tasks and making changes
- **SANDBOX Mode** (Restricted): For testing untrusted code with minimal permissions

### Files

- `vetinari/execution_context.py` - Main execution context module

### Key Classes

#### `ExecutionMode`
```python
class ExecutionMode(Enum):
    PLANNING = "planning"      # Read-only exploration
    EXECUTION = "execution"    # Full access
    SANDBOX = "sandbox"        # Restricted testing
```

#### `ToolPermission`
```python
class ToolPermission(Enum):
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    BASH_EXECUTE = "bash_execute"
    PYTHON_EXECUTE = "python_execute"
    MODEL_INFERENCE = "model_inference"
    MODEL_DISCOVERY = "model_discovery"
    NETWORK_REQUEST = "network_request"
    DATABASE_WRITE = "database_write"
    MEMORY_WRITE = "memory_write"
    GIT_COMMIT = "git_commit"
    GIT_PUSH = "git_push"
```

#### `ContextManager`
Manages the execution context stack and enforces permissions.

```python
from vetinari.execution_context import get_context_manager, ExecutionMode

manager = get_context_manager()

# Switch to planning mode
with manager.temporary_mode(ExecutionMode.PLANNING):
    # Operations here run in read-only planning mode
    pass

# Check permissions
if manager.check_permission(ToolPermission.FILE_WRITE):
    # Safe to write files
    pass
```

### Usage Example

```python
from vetinari.execution_context import get_context_manager, ExecutionMode

context_manager = get_context_manager()

# Switch to execution mode
context_manager.switch_mode(ExecutionMode.EXECUTION, task_id="t1")

# Perform operations...

# Get context status
status = context_manager.get_status()
print(status)
# Output:
# {
#     'mode': 'execution',
#     'task_id': 't1',
#     'started_at': '2026-03-03T...',
#     'operations_count': 5,
#     'permissions': ['file_read', 'file_write', ...]
# }

# Return to previous context
context_manager.pop_context()
```

### Permission Policies

Each mode has a default permission policy that can be customized:

**Planning Mode Permissions:**
- ✅ FILE_READ
- ✅ MODEL_INFERENCE
- ✅ MODEL_DISCOVERY
- ✅ NETWORK_REQUEST (read-only)
- ⚠️ BASH_EXECUTE (requires confirmation)
- ⚠️ PYTHON_EXECUTE (requires confirmation)
- ❌ FILE_WRITE, FILE_DELETE
- ❌ GIT_PUSH, GIT_COMMIT

**Execution Mode Permissions:**
- ✅ All read permissions
- ✅ FILE_WRITE, FILE_DELETE
- ✅ BASH_EXECUTE, PYTHON_EXECUTE
- ✅ DATABASE_WRITE, MEMORY_WRITE
- ✅ GIT_COMMIT
- ⚠️ GIT_PUSH (requires confirmation)

**Sandbox Mode Permissions:**
- ✅ FILE_READ
- ✅ PYTHON_EXECUTE
- ✅ MODEL_INFERENCE
- ⚠️ BASH_EXECUTE (requires confirmation)
- ❌ FILE_WRITE, FILE_DELETE
- ❌ GIT operations, DATABASE operations

---

## 2. Standardized Tool Interface

### Overview

Tools are the fundamental units of work in Vetinari. The new `Tool` base class provides:

- Standardized metadata and parameter definitions
- Permission requirements and mode restrictions
- Input/output validation
- Automatic safety checks
- Audit trail recording

### Files

- `vetinari/tool_interface.py` - Tool interface module

### Key Classes

#### `ToolMetadata`
```python
@dataclass
class ToolMetadata:
    name: str
    description: str
    category: ToolCategory
    version: str
    author: str
    parameters: List[ToolParameter]
    required_permissions: List[ToolPermission]
    allowed_modes: List[ExecutionMode]
    tags: List[str]
```

#### `ToolParameter`
```python
@dataclass
class ToolParameter:
    name: str
    type: Type
    description: str
    required: bool
    default: Optional[Any]
    allowed_values: Optional[List[Any]]
    
    def validate(self, value: Any) -> bool:
        # Validates value against type and constraints
        ...
```

#### `Tool` (Abstract Base Class)
```python
class Tool(ABC):
    def __init__(self, metadata: ToolMetadata):
        self.metadata = metadata
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Implement tool-specific logic"""
        pass
    
    def run(self, **kwargs) -> ToolResult:
        """
        Main entry point with:
        - Input validation
        - Permission checking
        - Pre/post-execution hooks
        - Audit trail recording
        """
        pass
```

### Creating a Custom Tool

```python
from vetinari.tool_interface import Tool, ToolMetadata, ToolParameter, ToolCategory, ToolResult
from vetinari.execution_context import ExecutionMode, ToolPermission

class FileReadTool(Tool):
    def __init__(self):
        metadata = ToolMetadata(
            name="read_file",
            description="Read contents of a file",
            category=ToolCategory.FILE_OPERATIONS,
            version="1.0.0",
            parameters=[
                ToolParameter(
                    name="path",
                    type=str,
                    description="File path to read",
                    required=True,
                ),
                ToolParameter(
                    name="encoding",
                    type=str,
                    description="File encoding",
                    required=False,
                    default="utf-8",
                ),
            ],
            required_permissions=[ToolPermission.FILE_READ],
            allowed_modes=[ExecutionMode.PLANNING, ExecutionMode.EXECUTION],
        )
        super().__init__(metadata)
    
    def execute(self, path: str, encoding: str = "utf-8", **kwargs) -> ToolResult:
        try:
            with open(path, 'r', encoding=encoding) as f:
                content = f.read()
            return ToolResult(
                success=True,
                output=content,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
            )

# Register and use
registry = get_tool_registry()
tool = FileReadTool()
registry.register(tool)

# Run with automatic permission checking
result = tool.run(path="example.txt")
```

### Tool Registry

```python
from vetinari.tool_interface import get_tool_registry

registry = get_tool_registry()

# Register tools
registry.register(my_tool)

# List all tools
all_tools = registry.list_tools()

# Get tools for current mode
current_mode_tools = registry.list_tools_for_mode(ExecutionMode.EXECUTION)

# Get tools by category
file_tools = registry.get_tools_by_category(ToolCategory.FILE_OPERATIONS)

# Get tools requiring specific permission
write_tools = registry.get_tools_requiring_permission(ToolPermission.FILE_WRITE)
```

---

## 3. Provider Agnosticism Layer

### Overview

Vetinari now supports multiple LLM providers with:

- Unified adapter interface for all providers (OpenAI, Claude, Cohere, Gemini, etc.)
- Automatic provider discovery and health monitoring
- Intelligent model selection based on task requirements
- Provider fallback on errors
- Cost and performance metrics tracking

### Files

- `vetinari/adapter_manager.py` - Enhanced adapter management
- `vetinari/adapters/` - Provider-specific adapters

### Key Classes

#### `AdapterManager`
```python
from vetinari.adapter_manager import get_adapter_manager

manager = get_adapter_manager()

# Register providers
from vetinari.adapters.base import ProviderConfig, ProviderType

config = ProviderConfig(
    provider_type=ProviderType.OPENAI,
    name="openai_main",
    endpoint="https://api.openai.com/v1",
    api_key="sk-...",
)
manager.register_provider(config, "openai")

# List providers
providers = manager.list_providers()

# Health check
health = manager.health_check()

# Discover models
models = manager.discover_models()

# Select best provider for task
provider_name, model_info = manager.select_provider_for_task(
    task_requirements={
        "required_capabilities": ["code_gen"],
        "input_tokens": 2000,
        "max_latency_ms": 10000,
    },
    preferred_provider="openai"
)

# Run inference with automatic fallback
from vetinari.adapters.base import InferenceRequest

request = InferenceRequest(
    model_id="gpt-4",
    prompt="Write hello world",
    system_prompt="You are a helpful assistant",
)

response = manager.infer(request, provider_name=provider_name, fallback_on_error=True)
```

### Supported Providers

- ✅ OpenAI (GPT-4, GPT-3.5, etc.)
- ✅ Anthropic Claude
- ✅ Google Gemini
- ✅ Cohere
- ✅ LM Studio (local models)
- ✅ Hugging Face (extensible)
- ✅ Replicate (extensible)

### Adding a New Provider

```python
from vetinari.adapters.base import ProviderAdapter, ProviderConfig, ProviderType, ModelInfo, InferenceRequest, InferenceResponse

class CustomProviderAdapter(ProviderAdapter):
    def discover_models(self) -> List[ModelInfo]:
        # Query your provider API
        pass
    
    def health_check(self) -> Dict[str, Any]:
        # Check provider health
        pass
    
    def infer(self, request: InferenceRequest) -> InferenceResponse:
        # Execute inference
        pass
    
    def get_capabilities(self) -> Dict[str, List[str]]:
        # Return model capabilities
        pass

# Register the new adapter
from vetinari.adapters.registry import AdapterRegistry

AdapterRegistry.register_adapter(
    ProviderType.CUSTOM,
    CustomProviderAdapter
)
```

### Provider Metrics

```python
manager = get_adapter_manager()

# Get metrics for all providers
all_metrics = manager.get_metrics()

# Get metrics for specific provider
provider_metrics = manager.get_metrics("openai")
# Returns:
# {
#     "name": "openai",
#     "provider_type": "openai",
#     "health_status": "healthy",
#     "successful_inferences": 42,
#     "failed_inferences": 2,
#     "success_rate": 0.955,
#     "avg_latency_ms": 1250.5,
#     "total_tokens_used": 125000,
#     "estimated_cost": 3.75,
# }
```

---

## 4. Enhanced CLI

### Overview

The CLI now provides rich user feedback with:

- Explicit execution mode banners
- Provider and context status display
- Health check and metrics
- Confirmation prompts for risky operations
- Structured logging and audit trails

### Usage

```bash
# Run in planning mode (read-only exploration)
vetinari --mode planning --task t1

# Run in execution mode (full access)
vetinari --mode execution --task t1

# Sandbox mode for testing
vetinari --mode sandbox --task t1

# Check provider status
vetinari --providers

# Run health check on all providers
vetinari --health-check

# Show execution context
vetinari --context

# Verbose logging
vetinari --verbose

# Check for model upgrades
vetinari --upgrade
```

### CLI Output Example

```
============================================================
⚙️  EXECUTION MODE (Full Access)
============================================================

▶️  Running task: t1

🔌 Provider Status (3 provider(s)):
   ✅ openai
      Health: healthy
      Success Rate: 95.5%
      Avg Latency: 1250ms
   ✅ claude
      Health: healthy
      Success Rate: 100.0%
      Avg Latency: 850ms
   ⚠️  cohere
      Health: degraded
      Success Rate: 75.0%
      Avg Latency: 2100ms

📊 Execution Context:
   Mode: execution
   Task: t1
   Operations: 5
   Permissions: file_read, file_write, bash_execute, ...

✅ Workflow completed
```

---

## 5. Verification Pipeline

### Overview

Comprehensive post-execution verification ensures:

- Code syntax validity
- Security (no secrets, no dangerous patterns)
- Safe imports
- JSON structure validation
- Custom verification rules

### Files

- `vetinari/verification.py` - Verification system

### Key Classes

#### `VerificationLevel`
```python
class VerificationLevel(Enum):
    NONE = "none"           # No verification
    BASIC = "basic"         # Basic checks only
    STANDARD = "standard"   # Standard checks (default)
    STRICT = "strict"       # Strict checks
    PARANOID = "paranoid"   # Maximum checks
```

#### `VerificationPipeline`
```python
from vetinari.verification import get_verifier_pipeline, VerificationLevel

pipeline = get_verifier_pipeline()

# Add custom verifier
from vetinari.verification import Verifier, VerificationResult, VerificationStatus

class CustomVerifier(Verifier):
    def __init__(self):
        super().__init__("my_check")
    
    def verify(self, content: str) -> VerificationResult:
        # Custom verification logic
        pass

pipeline.add_verifier(CustomVerifier())

# Run verification
results = pipeline.verify(code_output)

# Get summary
summary = pipeline.get_summary(results)
# Returns:
# {
#     "overall_status": "PASSED",
#     "total_checks": 4,
#     "total_issues": 0,
#     "error_count": 0,
#     "warning_count": 0,
#     "checks": {...}
# }
```

### Built-in Verifiers

- **CodeSyntaxVerifier** - Checks Python syntax validity
- **SecurityVerifier** - Detects secrets and dangerous patterns
- **ImportVerifier** - Validates safe imports
- **JSONStructureVerifier** - Validates JSON structure

### Usage Example

```python
from vetinari.verification import get_verifier_pipeline

pipeline = get_verifier_pipeline()

# Verify code output
results = pipeline.verify(model_output)

# Check for failures
summary = pipeline.get_summary(results)
if summary["overall_status"] == "PASSED":
    print("✅ Verification passed")
else:
    print("❌ Verification failed")
    for check_name, check_result in results.items():
        if check_result.issues:
            for issue in check_result.issues:
                print(f"  - {issue.severity}: {issue.message}")
```

---

## Integration Checklist

- [x] ExecutionContext system with modes and permissions
- [x] Tool interface with metadata and safety checks
- [x] AdapterManager with provider selection
- [x] Enhanced CLI with mode indicators and status
- [x] Verification pipeline for post-execution validation
- [x] Security scanning (secrets, dangerous patterns)
- [x] Audit trail and operation recording
- [ ] Update orchestrator to use new systems
- [ ] Migrate existing skills to new Tool interface
- [ ] Create example tools and verifiers
- [ ] Add comprehensive tests
- [ ] Update README and documentation

---

## Migration Guide

### For Existing Skills

Convert existing skills to the new Tool interface:

```python
# OLD:
class MySkill:
    def execute(self, params):
        # Implementation
        pass

# NEW:
from vetinari.tool_interface import Tool, ToolMetadata, ToolParameter, ToolCategory
from vetinari.execution_context import ExecutionMode, ToolPermission

class MyTool(Tool):
    def __init__(self):
        metadata = ToolMetadata(
            name="my_tool",
            description="My tool description",
            category=ToolCategory.CODE_EXECUTION,
            parameters=[...],
            required_permissions=[...],
            allowed_modes=[ExecutionMode.EXECUTION],
        )
        super().__init__(metadata)
    
    def execute(self, **kwargs):
        # Implementation
        return ToolResult(success=True, output=result)
```

### For Existing Adapters

Use the unified interface in AdapterManager:

```python
# OLD:
adapter = LMStudioAdapter(...)
response = adapter.chat(...)

# NEW:
from vetinari.adapter_manager import get_adapter_manager
from vetinari.adapters.base import ProviderConfig, ProviderType, InferenceRequest

manager = get_adapter_manager()
config = ProviderConfig(
    provider_type=ProviderType.LM_STUDIO,
    endpoint="http://localhost:1234"
)
manager.register_provider(config, "lmstudio")

request = InferenceRequest(model_id="...", prompt="...")
response = manager.infer(request, provider_name="lmstudio")
```

---

## Performance Considerations

1. **Context Switching**: Lightweight operation, minimal overhead
2. **Permission Checking**: Fast lookup with compiled patterns
3. **Provider Selection**: Heuristic-based scoring (O(n*m) for n providers, m models)
4. **Verification**: Configurable strictness levels to balance speed/safety
5. **Audit Trail**: Optional batching to reduce memory overhead

---

## Security Best Practices

1. **Always use Planning mode for exploration** of untrusted code
2. **Enable PARANOID verification** for user-generated code
3. **Regularly check provider health** and metrics
4. **Use Secret Scanner** before storing outputs in memory
5. **Maintain audit trails** for compliance and debugging
6. **Restrict permissions by default**, grant as needed
7. **Use provider fallback** for resilience

---

## Future Enhancements

- [ ] Interactive confirmation prompts in CLI
- [ ] Custom permission policies via YAML configuration
- [ ] Provider cost optimization algorithms
- [ ] Machine learning-based provider selection
- [ ] Real-time monitoring dashboard
- [ ] Advanced audit trail analysis
- [ ] Custom verification rule DSL
- [ ] Distributed execution across multiple machines

---

## Support

For questions or issues, refer to:
- [OpenCode Docs](https://opencode.ai/docs)
- [Vetinari README](./README.md)
- Issue tracker on GitHub
