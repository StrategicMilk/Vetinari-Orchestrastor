# Implementation Roadmap for OpenCode Integration

## Phase 1: Foundation (COMPLETED ✅)

### Completed Components

1. ✅ **ExecutionContext System** (`vetinari/execution_context.py`)
   - ExecutionMode enum (PLANNING, EXECUTION, SANDBOX)
   - ToolPermission enum with 12 permission types
   - ContextManager with context stacking and enforcement
   - Pre/post-execution hooks and audit trails

2. ✅ **Tool Interface** (`vetinari/tool_interface.py`)
   - Abstract Tool base class
   - ToolMetadata and ToolParameter dataclasses
   - ToolCategory enum
   - ToolRegistry for tool management
   - VerificationResult and ToolResult dataclasses

3. ✅ **AdapterManager** (`vetinari/adapter_manager.py`)
   - Multi-provider adapter management
   - ProviderMetrics tracking
   - Intelligent model selection
   - Fallback provider support
   - Health monitoring and discovery

4. ✅ **Enhanced CLI** (`cli.py`)
   - Execution mode selection (--mode)
   - Provider status display (--providers)
   - Health checks (--health-check)
   - Context status (--context)
   - Rich visual feedback with emoji indicators

5. ✅ **Verification Pipeline** (`vetinari/verification.py`)
   - CodeSyntaxVerifier
   - SecurityVerifier with secret scanning
   - ImportVerifier for safe imports
   - JSONStructureVerifier
   - Customizable verification levels
   - Comprehensive issue reporting

6. ✅ **Documentation** (`docs/OPENCODE_INTEGRATION.md`)
   - Complete integration guide
   - Usage examples for all systems
   - Security best practices
   - Migration guide for existing code

---

## Phase 2: Integration (NEXT)

### 2.1 Update Orchestrator
**File:** `vetinari/orchestrator.py`

**Tasks:**
- [ ] Import ExecutionContext and AdapterManager
- [ ] Initialize context manager at startup
- [ ] Initialize adapter manager with configured providers
- [ ] Switch execution mode based on task requirements
- [ ] Record operations in audit trail
- [ ] Run verification pipeline on task outputs
- [ ] Handle permission errors gracefully

**Example Changes:**
```python
from vetinari.execution_context import get_context_manager, ExecutionMode
from vetinari.adapter_manager import get_adapter_manager
from vetinari.verification import get_verifier_pipeline

class Orchestrator:
    def __init__(self, manifest_path: str):
        # Existing initialization...
        self.context_manager = get_context_manager()
        self.adapter_manager = get_adapter_manager()
        self.verifier = get_verifier_pipeline()
        
    def run_task(self, task_id: str):
        # Switch to execution mode for this task
        self.context_manager.switch_mode(ExecutionMode.EXECUTION, task_id)
        
        # Run task...
        result = self.executor.execute_task(task_id)
        
        # Verify output
        verification = self.verifier.verify(result.output)
        
        # Pop context
        self.context_manager.pop_context()
```

### 2.2 Migrate Skills to Tools
**Files:** `skills/` directory → Implement Tool interface

**Tasks:**
- [ ] Identify all existing skills
- [ ] Create Tool wrapper for each skill
- [ ] Define ToolMetadata with parameters and permissions
- [ ] Implement Tool.execute() method
- [ ] Register tools in ToolRegistry
- [ ] Add unit tests for each tool

**Skill Candidates for Migration:**
- [ ] `skills/builder` → BuilderTool
- [ ] `skills/evaluator` → EvaluatorTool
- [ ] `skills/explorer` → ExplorerTool
- [ ] `skills/librarian` → LibrarianTool
- [ ] `skills/oracle` → OracleTool
- [ ] `skills/researcher` → ResearcherTool
- [ ] `skills/synthesizer` → SynthesizerTool
- [ ] `skills/ui-planner` → UIPlannerTool

### 2.3 Update Model Selection
**File:** `vetinari/orchestrator.py` → model selection logic

**Tasks:**
- [ ] Integrate AdapterManager for provider selection
- [ ] Use task requirements to select best provider/model
- [ ] Handle provider fallback automatically
- [ ] Track provider metrics during execution
- [ ] Log provider decisions in audit trail

### 2.4 Add Permission Enforcement to Executor
**File:** `vetinari/executor.py`

**Tasks:**
- [ ] Check permissions before tool execution
- [ ] Prompt for confirmation on dangerous operations
- [ ] Deny operations based on execution mode
- [ ] Log permission checks in audit trail
- [ ] Handle permission errors gracefully

---

## Phase 3: Testing & Validation (NEXT AFTER PHASE 2)

### 3.1 Unit Tests
**Location:** `tests/`

**Test Files to Create:**
- [ ] `test_execution_context.py`
  - Test mode switching
  - Test permission checking
  - Test context stacking
  - Test audit trail recording

- [ ] `test_tool_interface.py`
  - Test tool registration
  - Test input validation
  - Test permission enforcement
  - Test execution hooks

- [ ] `test_adapter_manager.py`
  - Test provider registration
  - Test model discovery
  - Test provider selection
  - Test fallback behavior

- [ ] `test_verification.py`
  - Test syntax verification
  - Test security scanning
  - Test custom verifiers
  - Test verification summary

### 3.2 Integration Tests
**Location:** `tests/`

**Test Scenarios:**
- [ ] End-to-end workflow in planning mode
- [ ] End-to-end workflow in execution mode
- [ ] Mode switching during execution
- [ ] Provider fallback on error
- [ ] Multi-provider inference
- [ ] Audit trail completeness
- [ ] Permission enforcement across all tools

### 3.3 Example Workflows
**Location:** `examples/` (create new directory)

**Example Scripts:**
- [ ] `01_planning_mode_example.py` - Demonstrate planning mode
- [ ] `02_execution_mode_example.py` - Demonstrate execution mode
- [ ] `03_multi_provider_example.py` - Multiple providers
- [ ] `04_custom_tool_example.py` - Creating custom tools
- [ ] `05_verification_example.py` - Verification pipeline

---

## Phase 4: Documentation & Polish (AFTER PHASE 3)

### 4.1 User Documentation
- [ ] Update main README.md
- [ ] Create quick start guide
- [ ] Create execution mode guide
- [ ] Create tool development guide
- [ ] Create provider setup guide
- [ ] Create troubleshooting guide

### 4.2 API Documentation
- [ ] Generate docstring documentation
- [ ] Create architecture diagrams
- [ ] Create sequence diagrams for workflows
- [ ] Create permission matrix documentation
- [ ] Create provider comparison table

### 4.3 Video & Visual Content
- [ ] Create demo video for planning vs execution mode
- [ ] Create tool development walkthrough
- [ ] Create multi-provider setup tutorial

### 4.4 Performance Optimization
- [ ] Profile context switching overhead
- [ ] Optimize permission checking
- [ ] Optimize verification pipeline
- [ ] Cache provider discovery results
- [ ] Implement lazy loading where appropriate

---

## Phase 5: Advanced Features (FUTURE)

### 5.1 Interactive CLI
- [ ] Add interactive mode with REPL
- [ ] Add task scheduling and monitoring
- [ ] Add real-time execution visualization
- [ ] Add interactive confirmation prompts

### 5.2 Distributed Execution
- [ ] Add remote executor support
- [ ] Add task queue and worker system
- [ ] Add result aggregation
- [ ] Add distributed tracing

### 5.3 Advanced Verification
- [ ] ML-based code quality scoring
- [ ] Automated test generation
- [ ] Coverage analysis
- [ ] Performance profiling

### 5.4 Provider Intelligence
- [ ] Cost optimization algorithms
- [ ] Performance prediction
- [ ] Automatic load balancing
- [ ] Provider recommendation engine

### 5.5 Observability
- [ ] Real-time monitoring dashboard
- [ ] Advanced analytics
- [ ] Cost reporting
- [ ] Usage patterns analysis

---

## Development Guidelines

### Code Standards
- Follow PEP 8 style guide
- Use type hints everywhere
- Include docstrings for all public methods
- Add logging at appropriate levels
- Write tests for new code

### Git Workflow
1. Create feature branch: `git checkout -b feature/component-name`
2. Make changes in small, focused commits
3. Run tests before committing
4. Push and create pull request
5. Address review feedback
6. Merge when approved

### Testing Requirements
- New code must have unit tests
- Integration tests for multi-component changes
- All tests must pass before merge
- Aim for >80% code coverage

### Documentation Requirements
- Update docstrings
- Update relevant guide documents
- Add example code if applicable
- Update changelog

---

## Success Metrics

By completion of Phase 2, Vetinari should:

✅ Support multiple execution modes with clear permission boundaries
✅ Manage tools through a unified, safe interface
✅ Support multiple LLM providers seamlessly
✅ Provide rich, informative CLI feedback
✅ Verify outputs for security and correctness
✅ Maintain complete audit trails
✅ Deliver products with minimal user effort

---

## Quick Reference

### Key Files Created
- `vetinari/execution_context.py` - Execution modes and permissions
- `vetinari/tool_interface.py` - Tool interface and registry
- `vetinari/adapter_manager.py` - Provider management
- `vetinari/verification.py` - Output verification
- `cli.py` - Enhanced CLI
- `docs/OPENCODE_INTEGRATION.md` - Integration guide

### Key Directories
- `vetinari/adapters/` - Provider adapters
- `vetinari/agents/` - Agent implementations
- `skills/` - Available skills/tools
- `tests/` - Test suite
- `docs/` - Documentation
- `examples/` - Example usage

### Quick Start Commands
```bash
# View help
python -m vetinari.cli --help

# Run in planning mode
python -m vetinari.cli --mode planning --task t1

# Check providers
python -m vetinari.cli --providers

# Check context
python -m vetinari.cli --context
```

---

## Questions & Support

Refer to:
1. `docs/OPENCODE_INTEGRATION.md` - Comprehensive guide
2. OpenCode documentation - Original patterns
3. Code comments and docstrings
4. Tests for usage examples
