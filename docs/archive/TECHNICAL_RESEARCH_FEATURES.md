# Technical Research: Advanced Features for Vetinari LM Studio Integration

## Executive Summary

This research covers 11 advanced features for optimizing local LLM orchestration in Vetinari using LM Studio (localhost:1234, OpenAI-compatible API). Each feature includes implementation patterns, LM Studio API specifics, estimated effort, and dependencies. The research is based on LM Studio 0.4.6 (March 2026) with both llama.cpp and MLX backend support.

---

## 1. SLM/LLM Hybrid Routing

### What It Is & Why It Matters

Route simple inference tasks to small models (3-7B parameters) and complex reasoning tasks to large models (13B+). This reduces latency for straightforward tasks while preserving accuracy for complex work. Typical savings: 40-60% latency reduction for suitable tasks.

### LM Studio API Capabilities (2025-2026)

**Multi-Model Management:**
- LM Studio v0.4.0+ supports **programmatic multi-model loading** via `POST /api/v1/models/load`
- Models can be loaded/unloaded dynamically with instance identifiers
- **Auto-Evict** feature unloads idle models before loading new ones, enabling seamless switching
- **Idle TTL** (Time-To-Live): Configure per-model timeout; idle timer resets on each request

**Key Endpoints:**
```
GET /api/v0/models          → Returns loaded models with state: "loaded" | "not-loaded"
POST /api/v1/models/load    → Load a model (returns error if already loaded)
POST /api/v1/models/unload  → Unload a model
GET /api/v1/status          → System load, VRAM usage
```

**Model Switching Latency:**
- First load: ~2-8 seconds (depends on model size, GPU VRAM)
- Switch (with Auto-Evict): ~3-5 seconds
- No switch (model cached): <50ms overhead

### Implementation Architecture

```python
# vetinari/hybrid_router.py
from typing import Optional, List
from pydantic import BaseModel
import requests
import logging

logger = logging.getLogger(__name__)

class ModelProfile(BaseModel):
    """Model profile with routing characteristics."""
    model_id: str
    size_b: int  # parameters in billions
    latency_ms: float
    context_len: int
    is_loaded: bool = False
    last_used: float = 0.0
    evict_ttl_seconds: int = 300

class HybridRouter:
    """Routes tasks to appropriate model size."""

    def __init__(self, lm_studio_url: str = "http://localhost:1234"):
        self.lm_studio_url = lm_studio_url
        self.session = requests.Session()
        self.models: Dict[str, ModelProfile] = {}
        self.complexity_threshold = 0.6  # Tune based on your workload

    def discover_models(self) -> List[ModelProfile]:
        """Discover and profile available models."""
        try:
            response = self.session.get(f"{self.lm_studio_url}/v1/models")
            data = response.json()
            models_list = data.get("data", data.get("models", []))

            for m in models_list:
                profile = ModelProfile(
                    model_id=m["id"],
                    size_b=self._estimate_size(m["id"]),  # Extract from name or metadata
                    latency_ms=m.get("latency_estimate_ms", 1000),
                    context_len=m.get("context_len", 2048),
                    is_loaded=m.get("state") == "loaded"
                )
                self.models[m["id"]] = profile

            logger.info(f"Discovered {len(self.models)} models")
            return list(self.models.values())
        except Exception as e:
            logger.error(f"Model discovery failed: {e}")
            return []

    def estimate_task_complexity(self, prompt: str, task_type: str) -> float:
        """
        Estimate task complexity on scale 0.0-1.0.

        Heuristics:
        - Length: longer prompts generally require more context
        - Keywords: reasoning, analysis, code review → higher complexity
        - Task type: simple classification < reasoning < code generation
        """
        complexity = 0.0

        # Length heuristic: 0.0-0.4
        tokens_estimate = len(prompt.split()) / 150  # rough token estimate
        if tokens_estimate < 10:
            complexity += 0.1
        elif tokens_estimate < 50:
            complexity += 0.2
        elif tokens_estimate < 200:
            complexity += 0.3
        else:
            complexity += 0.4

        # Keyword heuristic: 0.0-0.6
        high_complexity_keywords = [
            "analyze", "debug", "refactor", "architecture",
            "security", "performance", "design pattern"
        ]
        if any(kw in prompt.lower() for kw in high_complexity_keywords):
            complexity += 0.4
        elif any(kw in prompt.lower() for kw in ["write", "generate", "code"]):
            complexity += 0.25

        # Task type heuristic: 0.0-0.3
        task_complexity_map = {
            "classification": 0.1,
            "summarization": 0.15,
            "qa": 0.2,
            "coding": 0.35,
            "code_review": 0.45,
            "reasoning": 0.5,
            "planning": 0.55,
            "security_audit": 0.65,
        }
        complexity += task_complexity_map.get(task_type, 0.25)

        return min(1.0, complexity)

    def select_model(self, prompt: str, task_type: str) -> Optional[str]:
        """Select best model for task."""
        complexity = self.estimate_task_complexity(prompt, task_type)

        if complexity < self.complexity_threshold:
            # Use small model (3-7B)
            candidates = [m for m in self.models.values()
                         if m.size_b <= 7]
        else:
            # Use large model (13B+)
            candidates = [m for m in self.models.values()
                         if m.size_b >= 13]

        if not candidates:
            # Fallback: use largest available
            candidates = sorted(self.models.values(), key=lambda m: m.size_b, reverse=True)

        # Prefer already-loaded models
        loaded = [m for m in candidates if m.is_loaded]
        selected = loaded[0] if loaded else candidates[0]

        logger.info(f"Selected {selected.model_id} (complexity={complexity:.2f}, size={selected.size_b}B)")
        return selected.model_id

    def load_model_if_needed(self, model_id: str) -> bool:
        """Load model if not already loaded; respects Auto-Evict."""
        if self.models[model_id].is_loaded:
            return True

        try:
            payload = {"modelIdentifier": model_id}
            resp = self.session.post(
                f"{self.lm_studio_url}/api/v1/models/load",
                json=payload,
                timeout=30
            )
            resp.raise_for_status()
            self.models[model_id].is_loaded = True
            logger.info(f"Loaded model {model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to load {model_id}: {e}")
            return False
```

### Dependencies

```toml
[dev-dependencies]
requests = "^2.31"
pydantic = "^2.0"
```

### Estimated Effort

- **Implementation:** 8-12 hours
  - Model profiling and discovery: 2-3 hours
  - Complexity heuristics refinement: 3-4 hours
  - Load/unload orchestration: 2-3 hours
  - Testing with real models: 2-3 hours

- **Integration:** 4-6 hours
  - Agent selection logic in orchestrator
  - Monitoring and metrics
  - Tuning threshold parameters

---

## 2. Speculative Decoding

### What It Is & Why It Matters

Uses a small draft model to predict multiple future tokens, which a larger target model verifies in parallel. Performance gains: **1.5-2.8x throughput** without quality loss. Most effective for long-context outputs (code generation, documentation).

### LM Studio Support (2025-2026)

**Status:** LM Studio does **not** natively expose speculative decoding APIs. However, it can be implemented at the orchestrator layer using two model instances.

### Implementation Architecture

```python
# vetinari/speculative_decoder.py
import asyncio
from typing import AsyncGenerator
from vetinari.adapters.lmstudio_adapter import LMStudioProviderAdapter

class SpeculativeDraftModel:
    """Small draft model for token prediction."""
    def __init__(self, adapter: LMStudioProviderAdapter, model_id: str = "phi-2"):
        self.adapter = adapter
        self.model_id = model_id

    async def predict_tokens(self, prefix: str, k: int = 4) -> List[str]:
        """Predict next k tokens using draft model."""
        request = InferenceRequest(
            model_id=self.model_id,
            prompt=prefix,
            max_tokens=k,
            temperature=0.5,
        )
        response = self.adapter.infer(request)
        # Parse response into token list
        tokens = response.output.split()[:k]
        return tokens

class SpeculativeDecoder:
    """Orchestrates draft + target model verification."""

    def __init__(self,
                 draft_model_id: str = "phi-2",
                 target_model_id: str = "mistral-7b",
                 draft_adapter: LMStudioProviderAdapter = None,
                 target_adapter: LMStudioProviderAdapter = None):
        self.draft_model = SpeculativeDraftModel(draft_adapter, draft_model_id)
        self.target_adapter = target_adapter
        self.target_model_id = target_model_id
        self.gamma = 4  # Number of speculative tokens (tunable)

    async def infer_speculative(self, prompt: str, max_tokens: int = 256) -> AsyncGenerator[str, None]:
        """
        Speculative decoding: draft predicts, target verifies.

        Algorithm:
        1. Draft model generates γ tokens
        2. Target model scores all γ+1 positions (parallel)
        3. Accept tokens with probability min(1, P_target / P_draft)
        4. Regenerate rejected tokens
        5. Repeat until max_tokens or EOS
        """
        prompt_len = len(prompt.split())
        generated_tokens = 0
        current_prefix = prompt

        while generated_tokens < max_tokens:
            # Step 1: Draft γ tokens
            draft_tokens = await self.draft_model.predict_tokens(
                current_prefix,
                k=self.gamma
            )

            if not draft_tokens:
                break

            # Step 2: Target model scores all positions
            # For simplicity, we'll verify the first drafted token
            # (Full implementation would score all γ+1 positions in parallel)
            candidate = draft_tokens[0]
            test_prompt = f"{current_prefix} {candidate}"

            target_request = InferenceRequest(
                model_id=self.target_model_id,
                prompt=test_prompt,
                max_tokens=1,
                temperature=0.0,
            )
            target_response = self.target_adapter.infer(target_request)

            # For MVP: accept if target agrees with draft
            if candidate in target_response.output or len(draft_tokens) == 1:
                yield f" {candidate}"
                current_prefix = test_prompt
                generated_tokens += 1
            else:
                # Rejection: use target's choice
                target_token = target_response.output.split()[0]
                yield f" {target_token}"
                current_prefix = f"{current_prefix} {target_token}"
                generated_tokens += 1
```

### Estimated Effort

- **Research & tuning:** 12-16 hours
  - Literature review on verification strategies
  - Tuning γ (number of speculative tokens)
  - Measuring throughput gains

- **Implementation:** 16-20 hours
  - Token-level API changes
  - Parallel verification logic
  - Error recovery and edge cases

**Note:** This is best implemented as an **opt-in mode** due to complexity. Start with simple single-token draft for MVP.

---

## 3. Continuous Batching

### What It Is & Why It Matters

Dynamically batches multiple concurrent inference requests into a single GPU computation, maximizing throughput. Key for high-concurrency scenarios (multi-agent orchestration). Throughput gain: **3-5x** with 4-8 concurrent requests.

### LM Studio Support (2025-2026)

**Status:** **Fully supported** via llama.cpp and MLX backends.

**Configuration:**
- `GET /api/v0/config` returns `max_concurrent_predictions` setting
- Default: 4-8 concurrent requests (configurable in LM Studio UI)
- Unified KV cache feature (default enabled) optimizes memory across requests

### Implementation Architecture

```python
# vetinari/continuous_batch_manager.py
import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class BatchRequest:
    """A single request in a batch."""
    request_id: str
    model_id: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    future: asyncio.Future = None

@dataclass
class BatchConfig:
    """Continuous batching configuration."""
    max_batch_size: int = 8
    max_wait_ms: int = 100  # Max wait for batch to fill
    timeout_seconds: int = 60

class ContinuousBatchManager:
    """Manages dynamic batching of requests."""

    def __init__(self, adapter, config: BatchConfig = None):
        self.adapter = adapter
        self.config = config or BatchConfig()
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.active_batches: Dict[str, List[BatchRequest]] = {}
        self._worker_task = None

    async def start(self):
        """Start the batch processing worker."""
        self._worker_task = asyncio.create_task(self._batch_worker())

    async def submit(self, request: BatchRequest) -> str:
        """Submit a request for batching. Returns request_id."""
        future = asyncio.Future()
        request.future = future
        await self.request_queue.put(request)

        # Wait for result (respects timeout)
        try:
            result = await asyncio.wait_for(future, timeout=self.config.timeout_seconds)
            return result
        except asyncio.TimeoutError:
            logger.error(f"Request {request.request_id} timed out")
            raise

    async def _batch_worker(self):
        """Worker that continuously collects and executes batches."""
        while True:
            try:
                # Collect requests for a batch
                batch = []

                try:
                    # Wait for first request (blocking)
                    req = await asyncio.wait_for(
                        self.request_queue.get(),
                        timeout=None
                    )
                    batch.append(req)

                    # Collect additional requests up to max_batch_size
                    while len(batch) < self.config.max_batch_size:
                        try:
                            req = await asyncio.wait_for(
                                self.request_queue.get(),
                                timeout=self.config.max_wait_ms / 1000.0
                            )
                            batch.append(req)
                        except asyncio.TimeoutError:
                            # Batch timeout: process what we have
                            break

                    # Process batch
                    await self._process_batch(batch)

                except asyncio.TimeoutError:
                    continue

            except Exception as e:
                logger.error(f"Batch worker error: {e}")

    async def _process_batch(self, batch: List[BatchRequest]):
        """Execute a batch of requests in parallel (respecting LM Studio limits)."""
        logger.info(f"Processing batch of {len(batch)} requests")

        # LM Studio handles batching internally; we just send concurrent requests
        tasks = [
            asyncio.create_task(self._execute_request(req))
            for req in batch
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Resolve futures with results
        for req, result in zip(batch, results):
            if isinstance(result, Exception):
                req.future.set_exception(result)
            else:
                req.future.set_result(result)

    async def _execute_request(self, req: BatchRequest) -> str:
        """Execute a single request via the adapter."""
        from vetinari.adapters.lmstudio_adapter import InferenceRequest

        inference_req = InferenceRequest(
            model_id=req.model_id,
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )

        response = self.adapter.infer(inference_req)
        if response.status != "ok":
            raise RuntimeError(f"Inference failed: {response.error}")

        return response.output

    async def get_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        return {
            "queue_size": self.request_queue.qsize(),
            "active_batches": len(self.active_batches),
        }
```

### Usage in Orchestrator

```python
# In TwoLayerOrchestrator or agent execution
async def execute_parallel_agents(agents: List[BaseAgent], tasks: List[Task]):
    """Execute multiple agents with continuous batching."""
    batch_mgr = ContinuousBatchManager(self.adapter)
    await batch_mgr.start()

    futures = []
    for agent, task in zip(agents, tasks):
        req = BatchRequest(
            request_id=f"{agent.name}_{task.id}",
            model_id=task.assigned_model_id,
            prompt=task.prompt,
        )
        future = await batch_mgr.submit(req)
        futures.append((agent, future))

    results = await asyncio.gather(*[f for _, f in futures])
    return results
```

### Estimated Effort

- **Implementation:** 8-12 hours
  - Async queue management: 2-3 hours
  - Batch scheduling logic: 2-3 hours
  - Error handling and timeouts: 2-3 hours
  - Integration testing: 2-3 hours

---

## 4. Context Window Management

### What It Is & Why It Matters

Implements **ConversationSummaryBufferMemory** pattern: keep recent messages verbatim, summarize older ones. Prevents context overflow while preserving critical history. Enables longer conversations without token waste.

### Architecture

```python
# vetinari/memory/context_window_manager.py
import tiktoken
from typing import List, Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ContextWindowManager:
    """Manages context window with automatic summarization."""

    def __init__(self,
                 max_tokens: int = 4096,
                 summary_ratio: float = 0.7,
                 model: str = "gpt-4"):
        self.max_tokens = max_tokens
        self.summary_threshold = int(max_tokens * summary_ratio)
        self.model = model

        # Token counter (use tiktoken for OpenAI models, estimate for others)
        try:
            self.encoder = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback: rough estimate (1 token ≈ 4 chars)
            self.encoder = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoder:
            return len(self.encoder.encode(text))
        else:
            return len(text) // 4

    def format_message(self, role: str, content: str) -> str:
        """Format a message for token counting."""
        return f"{role}: {content}\n"

    def build_context(self, messages: List[Dict[str, str]],
                     system_prompt: Optional[str] = None) -> str:
        """
        Build context from messages, applying summarization if needed.

        Strategy:
        1. Always keep most recent N messages
        2. Summarize older messages if total exceeds threshold
        3. Maintain system prompt + recent buffer + summary
        """
        if not messages:
            return system_prompt or ""

        # Format all messages
        formatted = []
        total_tokens = 0

        if system_prompt:
            formatted.append(f"SYSTEM: {system_prompt}")
            total_tokens = self.count_tokens(system_prompt)

        # Count backwards from most recent
        message_strings = []
        for msg in reversed(messages):
            msg_str = self.format_message(msg["role"], msg["content"])
            msg_tokens = self.count_tokens(msg_str)
            message_strings.append((msg_str, msg_tokens))
            total_tokens += msg_tokens

        # Reverse back to chronological
        message_strings.reverse()

        # Check if we need summarization
        if total_tokens <= self.summary_threshold:
            # No summarization needed
            formatted.extend([msg for msg, _ in message_strings])
            return "\n".join(formatted)

        # Summarization needed: keep recent, summarize old
        logger.info(f"Context token count {total_tokens} exceeds threshold {self.summary_threshold}. Summarizing...")

        # Always keep last 3 messages
        keep_recent = min(3, len(message_strings))
        recent_messages = message_strings[-keep_recent:]
        old_messages = message_strings[:-keep_recent]

        # Build summary of old messages
        old_text = "\n".join([msg for msg, _ in old_messages])
        summary = self._generate_summary(old_text)

        formatted.append(f"[SUMMARY OF EARLIER CONVERSATION]\n{summary}\n")
        formatted.extend([msg for msg, _ in recent_messages])

        result = "\n".join(formatted)
        final_tokens = self.count_tokens(result)
        logger.info(f"After summarization: {final_tokens} tokens (saved {total_tokens - final_tokens})")

        return result

    def _generate_summary(self, text: str, max_summary_tokens: int = 200) -> str:
        """Generate summary of conversation segment."""
        # This would call an agent to summarize
        # For MVP: use simple extraction
        lines = text.split("\n")
        summary_lines = lines[::max(1, len(lines) // 5)]  # Sample every Nth line
        return "\n".join(summary_lines[:10])

class AgentMemoryBuffer:
    """Per-agent memory with automatic summarization on handoff."""

    def __init__(self, agent_name: str, max_tokens: int = 2048):
        self.agent_name = agent_name
        self.messages: List[Dict[str, str]] = []
        self.context_mgr = ContextWindowManager(max_tokens=max_tokens)

    def add_message(self, role: str, content: str):
        """Add message to agent's memory."""
        self.messages.append({"role": role, "content": content, "timestamp": datetime.now()})

    def get_context(self, system_prompt: Optional[str] = None) -> str:
        """Get formatted context for agent inference."""
        return self.context_mgr.build_context(self.messages, system_prompt)

    def transfer_to_next_agent(self, next_agent_name: str) -> Dict[str, Any]:
        """Prepare memory for handoff to next agent."""
        context = self.get_context()
        summary_tokens = self.context_mgr.count_tokens(context)

        return {
            "from_agent": self.agent_name,
            "to_agent": next_agent_name,
            "context_summary": context,
            "message_count": len(self.messages),
            "context_tokens": summary_tokens,
            "timestamp": datetime.now().isoformat(),
        }
```

### Integration with DualMemoryStore

```python
# In vetinari/memory/dual_memory.py - extend search to include context optimization
def search_with_context_limit(self, query: str,
                              context_limit_tokens: int = 2048) -> str:
    """Search and return results within token limit."""
    entries = self.search(query, limit=50)  # Search broadly

    context_mgr = ContextWindowManager(max_tokens=context_limit_tokens)
    formatted_entries = []

    for entry in entries:
        entry_text = f"[{entry.agent}] {entry.summary}\n{entry.content}"
        formatted_entries.append({
            "role": "context",
            "content": entry_text
        })

    # Build context respecting token limit
    return context_mgr.build_context(formatted_entries)
```

### Dependencies

```toml
[dependencies]
tiktoken = "^0.5"  # For OpenAI token counting
```

### Estimated Effort

- **Implementation:** 6-10 hours
  - Token counter setup: 1-2 hours
  - Message formatting and buffering: 2-3 hours
  - Summarization orchestration: 2-3 hours
  - Integration with memory system: 1-2 hours

---

## 5. Typed Output Schemas (JSON Schema Enforcement)

### What It Is & Why It Matters

Define **Pydantic models** for each agent's expected output, validate via JSON schema. Guarantees predictable, parseable agent outputs. Prevents parsing failures downstream.

### Architecture

```python
# vetinari/schemas/agent_outputs.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum
import json

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class PlannerOutput(BaseModel):
    """Expected output from PlannerAgent."""
    plan_id: str = Field(..., description="Unique plan identifier")
    goal: str = Field(..., description="Original goal")
    tasks: List[Dict[str, Any]] = Field(..., description="Ordered list of tasks")
    estimated_tokens: int = Field(..., ge=1, le=100000, description="Token estimate")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Risk assessment")

    @validator('tasks')
    def validate_tasks(cls, v):
        if not v:
            raise ValueError("Tasks list cannot be empty")
        return v

class BuilderOutput(BaseModel):
    """Expected output from BuilderAgent."""
    task_id: str
    status: TaskStatus
    code: Optional[str] = None
    error: Optional[str] = None
    files_modified: List[str] = Field(default_factory=list)
    quality_score: float = Field(ge=0.0, le=1.0)

class EvaluatorOutput(BaseModel):
    """Expected output from EvaluatorAgent."""
    evaluation_id: str
    success: bool
    quality_metrics: Dict[str, float] = Field(default_factory=dict)
    issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)

class AgentOutputRegistry:
    """Registry of schema definitions for all agents."""

    SCHEMAS = {
        "planner": PlannerOutput,
        "builder": BuilderOutput,
        "evaluator": EvaluatorOutput,
        # ... more agents
    }

    @classmethod
    def get_schema(cls, agent_type: str) -> BaseModel:
        """Get schema class for agent."""
        return cls.SCHEMAS.get(agent_type)

    @classmethod
    def get_json_schema(cls, agent_type: str) -> Dict[str, Any]:
        """Get JSON schema for agent output."""
        schema_class = cls.get_schema(agent_type)
        return schema_class.model_json_schema()

    @classmethod
    def validate_output(cls, agent_type: str, output_json: str) -> tuple[bool, Any, Optional[str]]:
        """
        Validate agent output against schema.

        Returns: (success, parsed_output, error_message)
        """
        try:
            schema_class = cls.get_schema(agent_type)
            if not schema_class:
                return False, None, f"No schema for agent type '{agent_type}'"

            parsed = json.loads(output_json)
            validated = schema_class(**parsed)
            return True, validated, None

        except json.JSONDecodeError as e:
            return False, None, f"JSON parse error: {e}"
        except Exception as e:  # ValidationError from pydantic
            return False, None, f"Validation error: {e}"

class SchemaEnforcingAdapter:
    """Wraps a provider adapter to enforce output schemas."""

    def __init__(self, adapter, agent_type: str):
        self.adapter = adapter
        self.agent_type = agent_type
        self.schema = AgentOutputRegistry.get_schema(agent_type)
        self.json_schema = AgentOutputRegistry.get_json_schema(agent_type)

    def infer_with_schema(self, request) -> tuple[bool, Optional[BaseModel], str]:
        """
        Inference with schema enforcement and retry logic.

        Strategy:
        1. Inject schema into system prompt
        2. Request JSON output
        3. Validate and parse
        4. Retry with clarification if validation fails
        """

        # Inject schema into prompt
        schema_instruction = f"""
You MUST respond with valid JSON matching this schema:

{json.dumps(self.json_schema, indent=2)}

IMPORTANT:
- Return ONLY valid JSON, no markdown or extra text
- All required fields must be present
- Numbers must be within specified ranges
"""

        enhanced_system_prompt = (request.system_prompt or "") + "\n" + schema_instruction
        request.system_prompt = enhanced_system_prompt
        request.temperature = 0.0  # Deterministic for parsing

        # First attempt
        response = self.adapter.infer(request)
        success, parsed, error = AgentOutputRegistry.validate_output(
            self.agent_type,
            response.output
        )

        if success:
            return True, parsed, response.output

        # Retry with clarification
        if error:
            retry_prompt = f"{request.prompt}\n\nPrevious response failed validation: {error}\nPlease retry with valid JSON."
            request.prompt = retry_prompt
            response = self.adapter.infer(request)
            success, parsed, error = AgentOutputRegistry.validate_output(
                self.agent_type,
                response.output
            )

            if success:
                return True, parsed, response.output

        return False, None, f"Schema validation failed after retry: {error}"
```

### Usage in Agents

```python
# In vetinari/agents/base_agent.py
class BaseAgent:
    def _infer_json(self, prompt: str, system_prompt: str = "") -> Dict[str, Any]:
        """Infer with JSON schema enforcement."""
        schema_adapter = SchemaEnforcingAdapter(self.adapter, self.agent_type)
        success, parsed, raw_output = schema_adapter.infer_with_schema(
            InferenceRequest(
                model_id=self.model_id,
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=2048,
            )
        )

        if not success:
            logger.error(f"Schema validation failed: {raw_output}")
            raise ValueError(f"Agent output validation failed: {raw_output}")

        return parsed.dict()
```

### Estimated Effort

- **Implementation:** 6-8 hours
  - Define Pydantic models for each agent: 3-4 hours
  - Schema enforcement wrapper: 1-2 hours
  - Integration and testing: 2-3 hours

---

## 6. Circuit Breakers

### What It Is & Why It Matters

Temporarily disable failing agents with exponential backoff. Prevents cascading failures. Typical recovery: 5-10 minutes with exponential backoff.

### Implementation

```python
# vetinari/resilience/circuit_breaker.py
from enum import Enum
from datetime import datetime, timedelta
import time
import logging

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    """Circuit breaker for agent or model availability."""

    def __init__(self,
                 name: str,
                 failure_threshold: int = 5,
                 recovery_timeout_seconds: int = 60,
                 max_timeout_seconds: int = 3600,
                 backoff_multiplier: float = 2.0):
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.max_timeout_seconds = max_timeout_seconds
        self.backoff_multiplier = backoff_multiplier

        self.last_failure_time: Optional[datetime] = None
        self.last_state_change: Optional[datetime] = None
        self.consecutive_successes = 0
        self.half_open_attempts = 0

    def record_success(self):
        """Record a successful call."""
        self.failure_count = 0
        self.last_failure_time = None

        if self.state == CircuitState.HALF_OPEN:
            self.consecutive_successes += 1
            if self.consecutive_successes >= 3:  # 3 successes to close
                self._transition_to(CircuitState.CLOSED)

    def record_failure(self):
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self._transition_to(CircuitState.OPEN)

        if self.state == CircuitState.HALF_OPEN:
            self.consecutive_successes = 0
            self._transition_to(CircuitState.OPEN)

    def can_attempt(self) -> bool:
        """Check if a call can be attempted."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Calculate backoff timeout
            timeout = self._calculate_timeout()
            elapsed = (datetime.now() - self.last_state_change).total_seconds()

            if elapsed >= timeout:
                # Transition to half-open to test recovery
                self._transition_to(CircuitState.HALF_OPEN)
                return True

            logger.warning(f"Circuit {self.name} is OPEN, will retry in {timeout - elapsed:.1f}s")
            return False

        if self.state == CircuitState.HALF_OPEN:
            self.half_open_attempts += 1
            return self.half_open_attempts <= 3  # Limit attempts in half-open

        return False

    def _calculate_timeout(self) -> float:
        """Calculate exponential backoff timeout."""
        # Count state transitions (increments of backoff)
        multiplier = min(
            2 ** self.failure_count,  # Exponential
            self.max_timeout_seconds / self.recovery_timeout_seconds
        )
        timeout = self.recovery_timeout_seconds * multiplier
        return min(timeout, self.max_timeout_seconds)

    def _transition_to(self, new_state: CircuitState):
        """Transition to new state."""
        old_state = self.state
        self.state = new_state
        self.last_state_change = datetime.now()
        self.consecutive_successes = 0
        self.half_open_attempts = 0

        logger.info(f"Circuit {self.name}: {old_state.value} → {new_state.value}")

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "recovery_timeout_seconds": self._calculate_timeout(),
        }

class CircuitBreakerRegistry:
    """Registry of circuit breakers per agent/model."""

    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}

    def get_or_create(self, name: str) -> CircuitBreaker:
        """Get or create circuit breaker by name."""
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name)
        return self.breakers[name]

    def check_availability(self, name: str) -> tuple[bool, str]:
        """Check if resource is available."""
        breaker = self.get_or_create(name)
        if breaker.can_attempt():
            return True, "Available"
        else:
            return False, f"Circuit {breaker.state.value}, next attempt in {breaker._calculate_timeout():.0f}s"

class ResilientAgentProxy:
    """Wraps agent with circuit breaker and retry logic."""

    def __init__(self, agent, circuit_breaker: CircuitBreaker):
        self.agent = agent
        self.circuit_breaker = circuit_breaker

    def infer(self, prompt: str, max_retries: int = 3) -> str:
        """Infer with circuit breaker protection."""
        for attempt in range(max_retries):
            if not self.circuit_breaker.can_attempt():
                available, reason = self.circuit_breaker.can_attempt(), "Circuit open"
                raise RuntimeError(f"Agent {self.agent.name} unavailable: {reason}")

            try:
                result = self.agent._infer(prompt)
                self.circuit_breaker.record_success()
                return result

            except Exception as e:
                self.circuit_breaker.record_failure()
                logger.error(f"Agent {self.agent.name} failed (attempt {attempt+1}): {e}")

                if attempt < max_retries - 1:
                    # Exponential backoff between retries
                    wait_time = (2 ** attempt)
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)

        raise RuntimeError(f"Agent {self.agent.name} failed after {max_retries} retries")
```

### Integration in Orchestrator

```python
# In vetinari/orchestration/two_layer.py
class TwoLayerOrchestrator:
    def __init__(self):
        self.circuit_registry = CircuitBreakerRegistry()

    def execute_agent(self, agent: BaseAgent, task: Task) -> str:
        """Execute agent with circuit breaker."""
        breaker = self.circuit_registry.get_or_create(agent.name)
        proxy = ResilientAgentProxy(agent, breaker)

        try:
            return proxy.infer(task.prompt)
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            # Fallback: route to alternative agent or skip task
            raise
```

### Estimated Effort

- **Implementation:** 4-6 hours
  - Core circuit breaker logic: 1-2 hours
  - Exponential backoff calculation: 1 hour
  - Integration with agents: 1-2 hours
  - Testing and tuning: 1-2 hours

---

## 7. Distributed Tracing (OpenTelemetry)

### What It Is & Why It Matters

Instrument agent calls as spans in OpenTelemetry. Visualize request flows through agents. Enables latency analysis and bottleneck identification.

### Implementation

```python
# vetinari/observability/tracing.py
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class VetinariTracingSetup:
    """Initialize OpenTelemetry tracing for Vetinari."""

    @staticmethod
    def setup_jaeger(service_name: str = "vetinari",
                     jaeger_host: str = "localhost",
                     jaeger_port: int = 6831) -> TracerProvider:
        """Setup Jaeger exporter for OpenTelemetry."""
        jaeger_exporter = JaegerExporter(
            agent_host_name=jaeger_host,
            agent_port=jaeger_port,
        )

        trace_provider = TracerProvider(
            resource=Resource.create({SERVICE_NAME: service_name})
        )
        trace_provider.add_span_processor(SimpleSpanProcessor(jaeger_exporter))
        trace.set_tracer_provider(trace_provider)

        logger.info(f"Tracing initialized: Jaeger at {jaeger_host}:{jaeger_port}")
        return trace_provider

class AgentSpanContext:
    """Context manager for instrumenting agent calls."""

    def __init__(self, agent_name: str, task_id: str, tracer=None):
        self.agent_name = agent_name
        self.task_id = task_id
        self.tracer = tracer or trace.get_tracer(__name__)
        self.span = None

    def __enter__(self):
        """Start a span for agent execution."""
        self.span = self.tracer.start_span(
            name=f"agent.{self.agent_name}.execute",
            attributes={
                "agent.name": self.agent_name,
                "task.id": self.task_id,
                "service.name": "vetinari",
            }
        )
        self.span.__enter__()
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the span."""
        if exc_type:
            self.span.set_attribute("error", True)
            self.span.set_attribute("error.type", exc_type.__name__)
        self.span.__exit__(exc_type, exc_val, exc_tb)

    def record_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Record an event within the span."""
        if self.span:
            self.span.add_event(name, attributes or {})

class InstrumentedBaseAgent:
    """Extended BaseAgent with automatic tracing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracer = trace.get_tracer(__name__)

    def _infer_with_tracing(self, prompt: str, task_id: str = "") -> str:
        """Infer with automatic tracing."""
        with AgentSpanContext(self.name, task_id, self.tracer) as span:
            span.set_attribute("prompt.length", len(prompt))
            span.set_attribute("model.id", self.model_id)

            try:
                # Record preprocessing
                span.record_event("preprocessing_start")
                processed_prompt = self._preprocess(prompt)
                span.record_event("preprocessing_end")

                # Record inference
                span.record_event("inference_start")
                result = self._infer(processed_prompt)
                span.record_event("inference_end")
                span.set_attribute("result.length", len(result))

                return result

            except Exception as e:
                span.record_event("error", {"error.message": str(e)})
                raise

class OrchestratorTracing:
    """Traces the full orchestration pipeline."""

    def __init__(self):
        self.tracer = trace.get_tracer(__name__)

    def trace_pipeline(self, goal: str):
        """Trace complete pipeline execution."""
        with self.tracer.start_as_current_span("orchestrator.pipeline") as span:
            span.set_attribute("goal", goal[:100])  # Truncate long goals

            # Each stage becomes a child span
            with self.tracer.start_as_current_span("stage.planning") as planning_span:
                planning_span.set_attribute("stage_number", 1)
                # ... planner execution

            with self.tracer.start_as_current_span("stage.execution") as exec_span:
                exec_span.set_attribute("stage_number", 2)
                # ... agent execution

            with self.tracer.start_as_current_span("stage.evaluation") as eval_span:
                eval_span.set_attribute("stage_number", 3)
                # ... evaluation
```

### Dependencies

```toml
[dependencies]
opentelemetry-api = "^1.20"
opentelemetry-sdk = "^1.20"
opentelemetry-exporter-jaeger = "^1.20"
```

### Visualization

To visualize traces:
1. Start Jaeger locally: `docker run -d --name jaeger -p 16686:16686 jaegertracing/all-in-one`
2. View traces at `http://localhost:16686`

### Estimated Effort

- **Implementation:** 6-8 hours
  - OpenTelemetry setup: 1-2 hours
  - Span instrumentation: 2-3 hours
  - Integration with orchestrator: 2-3 hours

---

## 8. Agent Performance Dashboard

### What It Is & Why It Matters

Real-time metrics dashboard: tokens/task, latency, error rate, cost per model. Early detection of performance degradation.

### Architecture

```python
# vetinari/analytics/dashboard_backend.py
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Any
import sqlite3
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class AgentMetric:
    """A single metric observation."""
    agent_name: str
    task_id: str
    model_id: str
    task_type: str
    latency_ms: float
    tokens_generated: int
    tokens_prompt: int
    cost_usd: float
    status: str  # "success", "failed", "timeout"
    timestamp: datetime
    error_message: Optional[str] = None

class MetricsCollector:
    """Collects agent metrics during execution."""

    def __init__(self, db_path: str = "./metrics.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize SQLite schema for metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                task_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                task_type TEXT,
                latency_ms REAL NOT NULL,
                tokens_generated INTEGER,
                tokens_prompt INTEGER,
                cost_usd REAL,
                status TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                error_message TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_costs (
                model_id TEXT PRIMARY KEY,
                cost_per_1k_input REAL,
                cost_per_1k_output REAL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_timestamp ON agent_metrics(agent_name, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_timestamp ON agent_metrics(model_id, timestamp)")

        conn.commit()
        conn.close()

    def record_metric(self, metric: AgentMetric) -> bool:
        """Record a metric observation."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO agent_metrics
                (agent_name, task_id, model_id, task_type, latency_ms,
                 tokens_generated, tokens_prompt, cost_usd, status, timestamp, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.agent_name, metric.task_id, metric.model_id, metric.task_type,
                metric.latency_ms, metric.tokens_generated, metric.tokens_prompt,
                metric.cost_usd, metric.status, metric.timestamp.isoformat(),
                metric.error_message
            ))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to record metric: {e}")
            return False

    def query_metrics(self,
                     agent_name: Optional[str] = None,
                     model_id: Optional[str] = None,
                     hours: int = 24) -> List[Dict[str, Any]]:
        """Query recent metrics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cutoff = datetime.now() - timedelta(hours=hours)

            query = "SELECT * FROM agent_metrics WHERE timestamp > ?"
            params = [cutoff.isoformat()]

            if agent_name:
                query += " AND agent_name = ?"
                params.append(agent_name)

            if model_id:
                query += " AND model_id = ?"
                params.append(model_id)

            query += " ORDER BY timestamp DESC"

            cursor.execute(query, params)
            rows = cursor.fetchall()

            # Convert to dict
            columns = [desc[0] for desc in cursor.description]
            results = [dict(zip(columns, row)) for row in rows]

            conn.close()
            return results
        except Exception as e:
            logger.error(f"Failed to query metrics: {e}")
            return []

class PerformanceAnalyzer:
    """Analyzes metrics to compute aggregates."""

    def __init__(self, collector: MetricsCollector):
        self.collector = collector

    def get_agent_summary(self, agent_name: str, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for an agent."""
        metrics = self.collector.query_metrics(agent_name=agent_name, hours=hours)

        if not metrics:
            return {"agent": agent_name, "error": "No metrics found"}

        latencies = [m["latency_ms"] for m in metrics]
        costs = [m["cost_usd"] for m in metrics if m["cost_usd"]]
        successes = [m for m in metrics if m["status"] == "success"]
        failures = [m for m in metrics if m["status"] == "failed"]

        return {
            "agent": agent_name,
            "total_tasks": len(metrics),
            "success_rate": len(successes) / len(metrics) if metrics else 0.0,
            "failure_rate": len(failures) / len(metrics) if metrics else 0.0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
            "p95_latency_ms": sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0.0,
            "p99_latency_ms": sorted(latencies)[int(0.99 * len(latencies))] if latencies else 0.0,
            "total_cost_usd": sum(costs) if costs else 0.0,
            "avg_cost_per_task": sum(costs) / len(costs) if costs else 0.0,
        }

    def get_model_comparison(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Compare model performance."""
        metrics = self.collector.query_metrics(hours=hours)

        models = {}
        for m in metrics:
            model_id = m["model_id"]
            if model_id not in models:
                models[model_id] = []
            models[model_id].append(m)

        summaries = []
        for model_id, model_metrics in models.items():
            latencies = [m["latency_ms"] for m in model_metrics]
            costs = [m["cost_usd"] for m in model_metrics if m["cost_usd"]]
            successes = len([m for m in model_metrics if m["status"] == "success"])

            summaries.append({
                "model": model_id,
                "tasks": len(model_metrics),
                "success_rate": successes / len(model_metrics) if model_metrics else 0.0,
                "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
                "total_cost_usd": sum(costs) if costs else 0.0,
            })

        return sorted(summaries, key=lambda x: x["avg_latency_ms"])
```

### Flask Dashboard Endpoints

```python
# vetinari/web_ui.py - add these routes
from flask import Flask, jsonify
from vetinari.analytics.dashboard_backend import MetricsCollector, PerformanceAnalyzer

app = Flask(__name__)
collector = MetricsCollector()
analyzer = PerformanceAnalyzer(collector)

@app.route("/api/metrics/agent/<agent_name>", methods=["GET"])
def get_agent_metrics(agent_name):
    """Get agent performance summary."""
    summary = analyzer.get_agent_summary(agent_name)
    return jsonify(summary)

@app.route("/api/metrics/models", methods=["GET"])
def get_model_comparison():
    """Compare all models."""
    comparison = analyzer.get_model_comparison()
    return jsonify(comparison)

@app.route("/api/metrics/recent", methods=["GET"])
def get_recent_metrics():
    """Get recent raw metrics."""
    metrics = collector.query_metrics(hours=1)
    return jsonify({"metrics": metrics, "count": len(metrics)})
```

### Frontend Dashboard (React/Vue)

```javascript
// frontend/src/components/MetricsDashboard.vue
<template>
  <div class="metrics-dashboard">
    <div class="agent-summary">
      <h2>Agent Performance</h2>
      <div v-for="agent in agents" :key="agent.agent" class="agent-card">
        <h3>{{ agent.agent }}</h3>
        <p>Success Rate: {{ (agent.success_rate * 100).toFixed(2) }}%</p>
        <p>Avg Latency: {{ agent.avg_latency_ms.toFixed(0) }}ms</p>
        <p>Cost: ${{ agent.total_cost_usd.toFixed(4) }}</p>
      </div>
    </div>

    <div class="model-comparison">
      <h2>Model Comparison</h2>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Tasks</th>
            <th>Success Rate</th>
            <th>Avg Latency (ms)</th>
            <th>Total Cost</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="model in models" :key="model.model">
            <td>{{ model.model }}</td>
            <td>{{ model.tasks }}</td>
            <td>{{ (model.success_rate * 100).toFixed(1) }}%</td>
            <td>{{ model.avg_latency_ms.toFixed(0) }}</td>
            <td>${{ model.total_cost_usd.toFixed(4) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
```

### Estimated Effort

- **Implementation:** 8-12 hours
  - Metrics collection: 2-3 hours
  - SQLite schema and queries: 2-3 hours
  - Analytics computation: 2-3 hours
  - Flask endpoints: 1-2 hours
  - Frontend dashboard: 2-4 hours

---

## 9. Dynamic Complexity Routing

### What It Is & Why It Matters

Estimate task complexity before full pipeline execution. Skip stages for simple tasks (e.g., QA → direct inference instead of full planning). Reduces latency for 40-50% of tasks.

### Implementation

```python
# vetinari/routing/complexity_estimator.py
from enum import Enum
from typing import List, Tuple
import re
import logging

logger = logging.getLogger(__name__)

class TaskComplexity(Enum):
    TRIVIAL = 1      # < 50 tokens total
    SIMPLE = 2       # 50-200 tokens, single domain
    MODERATE = 3     # 200-500 tokens, multi-domain
    COMPLEX = 4      # 500-2000 tokens, requires reasoning
    VERY_COMPLEX = 5 # > 2000 tokens, security/architecture

class ComplexityEstimator:
    """Estimates task complexity using heuristics."""

    # Keyword weights
    TRIVIAL_KEYWORDS = ["hello", "what is", "define", "list", "explain briefly"]
    SIMPLE_KEYWORDS = ["write", "generate", "create", "basic", "simple"]
    MODERATE_KEYWORDS = ["analyze", "refactor", "improve", "design", "implement"]
    COMPLEX_KEYWORDS = ["architecture", "security", "performance", "optimization", "design pattern"]
    VERY_COMPLEX_KEYWORDS = ["security audit", "penetration", "vulnerability", "threat model"]

    def estimate(self, prompt: str, task_type: str = "general") -> Tuple[TaskComplexity, float]:
        """
        Estimate complexity and return (complexity, confidence).
        Confidence: 0.0-1.0
        """

        # Token-based heuristic
        token_count = len(prompt.split())
        if token_count < 50:
            base_complexity = TaskComplexity.TRIVIAL
        elif token_count < 200:
            base_complexity = TaskComplexity.SIMPLE
        elif token_count < 500:
            base_complexity = TaskComplexity.MODERATE
        elif token_count < 2000:
            base_complexity = TaskComplexity.COMPLEX
        else:
            base_complexity = TaskComplexity.VERY_COMPLEX

        # Keyword analysis
        prompt_lower = prompt.lower()
        keyword_complexity = self._keyword_analysis(prompt_lower)

        # Task type heuristic
        type_complexity = self._task_type_complexity(task_type)

        # Aggregate (majority voting with confidence)
        complexities = [base_complexity, keyword_complexity, type_complexity]
        final = max(complexities, key=lambda x: x.value)  # Take highest

        # Confidence: higher if all signals agree
        agreement = len([c for c in complexities if c == final]) / 3.0

        logger.info(f"Complexity estimated: {final.name} (confidence={agreement:.2f})")
        return final, agreement

    def _keyword_analysis(self, text: str) -> TaskComplexity:
        """Analyze keywords to determine complexity."""
        if any(kw in text for kw in self.VERY_COMPLEX_KEYWORDS):
            return TaskComplexity.VERY_COMPLEX
        elif any(kw in text for kw in self.COMPLEX_KEYWORDS):
            return TaskComplexity.COMPLEX
        elif any(kw in text for kw in self.MODERATE_KEYWORDS):
            return TaskComplexity.MODERATE
        elif any(kw in text for kw in self.SIMPLE_KEYWORDS):
            return TaskComplexity.SIMPLE
        elif any(kw in text for kw in self.TRIVIAL_KEYWORDS):
            return TaskComplexity.TRIVIAL
        else:
            return TaskComplexity.MODERATE  # Default to moderate

    def _task_type_complexity(self, task_type: str) -> TaskComplexity:
        """Map task type to complexity."""
        mapping = {
            "qa": TaskComplexity.SIMPLE,
            "summarization": TaskComplexity.SIMPLE,
            "classification": TaskComplexity.SIMPLE,
            "extraction": TaskComplexity.MODERATE,
            "coding": TaskComplexity.COMPLEX,
            "code_review": TaskComplexity.COMPLEX,
            "reasoning": TaskComplexity.COMPLEX,
            "planning": TaskComplexity.VERY_COMPLEX,
            "security_audit": TaskComplexity.VERY_COMPLEX,
            "architecture": TaskComplexity.VERY_COMPLEX,
        }
        return mapping.get(task_type, TaskComplexity.MODERATE)

class AdaptivePipeline:
    """Execute different pipeline stages based on complexity."""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.estimator = ComplexityEstimator()

    def execute(self, goal: str, task_type: str = "general") -> str:
        """Execute goal with adaptive pipeline."""

        # Estimate complexity
        complexity, confidence = self.estimator.estimate(goal, task_type)

        if confidence < 0.5:
            logger.warning(f"Low confidence estimate ({confidence:.2f}), using full pipeline")
            complexity = TaskComplexity.COMPLEX  # Fallback to safe choice

        logger.info(f"Task complexity: {complexity.name}, executing adaptive pipeline")

        # Route based on complexity
        if complexity == TaskComplexity.TRIVIAL:
            # Direct inference, no planning
            return self._simple_inference(goal)

        elif complexity == TaskComplexity.SIMPLE:
            # Skip planning, direct building
            return self._simple_build(goal)

        elif complexity in [TaskComplexity.MODERATE, TaskComplexity.COMPLEX]:
            # Full pipeline
            return self.orchestrator.execute_full_pipeline(goal)

        else:  # VERY_COMPLEX
            # Full pipeline + extra review stages
            return self.orchestrator.execute_full_pipeline_with_review(goal)

    def _simple_inference(self, goal: str) -> str:
        """Direct inference for trivial tasks."""
        logger.info("Executing trivial task via direct inference")
        # Use smallest available model
        response = self.orchestrator.adapter.infer(
            model_id="smallest",
            prompt=goal,
            max_tokens=256
        )
        return response.output

    def _simple_build(self, goal: str) -> str:
        """Build without planning for simple tasks."""
        logger.info("Executing simple task via direct building")
        builder = self.orchestrator.builder_agent
        return builder._infer(goal)
```

### Integration in Orchestrator

```python
# In TwoLayerOrchestrator
def execute(self, goal: str, task_type: str = None) -> str:
    """Execute goal with optional adaptive routing."""

    if os.getenv("ADAPTIVE_ROUTING", "true").lower() == "true":
        pipeline = AdaptivePipeline(self)
        return pipeline.execute(goal, task_type or "general")
    else:
        # Standard full pipeline
        return self.execute_full_pipeline(goal)
```

### Estimated Effort

- **Implementation:** 4-6 hours
  - Heuristic design: 1-2 hours
  - Pipeline branches: 1-2 hours
  - Integration and testing: 1-2 hours

---

## 10. Advanced Analytics & Cost Optimization

### What It Is & Why It Matters

Track token costs per agent, model, task type. Predict costs before execution. Alert when approaching budget limits. Typical savings: 15-25% through better routing.

### Implementation

```python
# vetinari/analytics/cost_tracker.py
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import sqlite3
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelPricing:
    """Pricing for a model."""
    model_id: str
    cost_per_1k_input_tokens: float
    cost_per_1k_output_tokens: float

class CostTracker:
    """Tracks token costs and budgets."""

    def __init__(self, db_path: str = "./costs.db"):
        self.db_path = db_path
        self.pricing_cache: Dict[str, ModelPricing] = {}
        self._init_db()

    def _init_db(self):
        """Initialize cost tracking database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS token_costs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                task_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost_usd REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_pricing (
                model_id TEXT PRIMARY KEY,
                cost_per_1k_input REAL,
                cost_per_1k_output REAL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS budget_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                budget_name TEXT UNIQUE,
                monthly_limit_usd REAL,
                current_spend_usd REAL,
                alert_threshold_pct REAL DEFAULT 0.80,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_alert_at DATETIME
            )
        """)

        conn.commit()
        conn.close()

    def register_model_pricing(self, pricing: ModelPricing):
        """Register pricing for a model."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO model_pricing
            (model_id, cost_per_1k_input, cost_per_1k_output, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """, (pricing.model_id, pricing.cost_per_1k_input_tokens, pricing.cost_per_1k_output_tokens))

        conn.commit()
        conn.close()
        self.pricing_cache[pricing.model_id] = pricing

    def record_token_cost(self, agent_name: str, task_id: str, model_id: str,
                         input_tokens: int, output_tokens: int) -> float:
        """Record token usage and return cost."""
        pricing = self.pricing_cache.get(model_id)
        if not pricing:
            logger.warning(f"No pricing for {model_id}, using default")
            pricing = ModelPricing(model_id, 0.0015, 0.002)  # Default OpenAI-like

        cost = (input_tokens * pricing.cost_per_1k_input_tokens / 1000) + \
               (output_tokens * pricing.cost_per_1k_output_tokens / 1000)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO token_costs
            (agent_name, task_id, model_id, input_tokens, output_tokens, cost_usd)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (agent_name, task_id, model_id, input_tokens, output_tokens, cost))

        conn.commit()
        conn.close()

        return cost

    def get_monthly_cost(self) -> float:
        """Get current month's spending."""
        today = datetime.now()
        month_start = today.replace(day=1)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT SUM(cost_usd) FROM token_costs
            WHERE timestamp >= ?
        """, (month_start.isoformat(),))

        result = cursor.fetchone()[0]
        conn.close()

        return result or 0.0

    def get_cost_by_agent(self, days: int = 30) -> Dict[str, float]:
        """Get costs aggregated by agent."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT agent_name, SUM(cost_usd) as total_cost
            FROM token_costs
            WHERE timestamp >= ?
            GROUP BY agent_name
            ORDER BY total_cost DESC
        """, (cutoff,))

        results = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()

        return results

    def get_cost_by_model(self, days: int = 30) -> Dict[str, float]:
        """Get costs aggregated by model."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT model_id, SUM(cost_usd) as total_cost
            FROM token_costs
            WHERE timestamp >= ?
            GROUP BY model_id
            ORDER BY total_cost DESC
        """, (cutoff,))

        results = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()

        return results

    def predict_monthly_cost(self) -> float:
        """Predict end-of-month cost based on current trajectory."""
        today = datetime.now()
        days_elapsed = today.day
        current_spend = self.get_monthly_cost()

        if days_elapsed == 0:
            return 0.0

        daily_rate = current_spend / days_elapsed
        days_remaining = 31 - days_elapsed

        predicted_total = current_spend + (daily_rate * days_remaining)
        return predicted_total

class BudgetAlertSystem:
    """Monitors costs and alerts on budget overages."""

    def __init__(self, cost_tracker: CostTracker):
        self.cost_tracker = cost_tracker

    def set_budget(self, budget_name: str, monthly_limit_usd: float,
                   alert_threshold_pct: float = 0.80):
        """Set a budget with alert threshold."""
        conn = sqlite3.connect(self.cost_tracker.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO budget_alerts
            (budget_name, monthly_limit_usd, alert_threshold_pct)
            VALUES (?, ?, ?)
        """, (budget_name, monthly_limit_usd, alert_threshold_pct))

        conn.commit()
        conn.close()

    def check_budget_alerts(self) -> List[Dict[str, any]]:
        """Check all budgets for alerts."""
        alerts = []
        current_spend = self.cost_tracker.get_monthly_cost()

        conn = sqlite3.connect(self.cost_tracker.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM budget_alerts")
        budgets = cursor.fetchall()
        conn.close()

        for budget in budgets:
            budget_name, monthly_limit, alert_threshold = budget[1], budget[2], budget[3]

            if current_spend >= (monthly_limit * alert_threshold):
                pct = (current_spend / monthly_limit) * 100
                alerts.append({
                    "budget": budget_name,
                    "limit_usd": monthly_limit,
                    "current_spend_usd": current_spend,
                    "percent_of_budget": pct,
                    "status": "warning" if pct < 100 else "critical"
                })

        return alerts
```

### Cost Optimization Strategies

```python
# vetinari/analytics/cost_optimizer.py
class CostOptimizer:
    """Suggests cost optimizations."""

    def __init__(self, cost_tracker: CostTracker):
        self.cost_tracker = cost_tracker

    def get_optimization_recommendations(self) -> List[str]:
        """Suggest cost optimizations."""
        recommendations = []

        # Analyze costs by agent
        agent_costs = self.cost_tracker.get_cost_by_agent()
        top_agent = max(agent_costs, key=agent_costs.get)

        if agent_costs[top_agent] > sum(agent_costs.values()) * 0.5:
            recommendations.append(
                f"Agent '{top_agent}' uses {agent_costs[top_agent]:.2f} USD ({(agent_costs[top_agent]/sum(agent_costs.values())*100):.1f}% of total). "
                f"Consider optimizing its prompts or using smaller models."
            )

        # Analyze by model
        model_costs = self.cost_tracker.get_cost_by_model()
        for model_id, cost in sorted(model_costs.items(), key=lambda x: x[1], reverse=True)[:3]:
            recommendations.append(
                f"Model '{model_id}' costs ${cost:.2f}. "
                f"Consider using a smaller/cheaper alternative if suitable."
            )

        # Predict monthly cost
        predicted = self.cost_tracker.predict_monthly_cost()
        current = self.cost_tracker.get_monthly_cost()

        if predicted > 1000:
            recommendations.append(
                f"Predicted monthly cost: ${predicted:.2f} (currently ${current:.2f}). "
                f"Consider implementing caching or request deduplication."
            )

        return recommendations
```

### Estimated Effort

- **Implementation:** 8-10 hours
  - Cost tracking database: 2 hours
  - Pricing model integration: 1-2 hours
  - Budget alerts: 1-2 hours
  - Cost optimizer: 2-3 hours
  - Dashboard integration: 1-2 hours

---

## 11. Memory System Consolidation

### Current State

Vetinari has **multiple memory systems**:
1. **OcMemoryStore** — Persistent episodic/semantic memory (oc-style)
2. **MnemosyneMemoryStore** — Alternative episodic memory
3. **DualMemoryStore** — Coordinates both backends
4. **MemoryStore** — Plan execution tracking (SQLite, distinct from agent memory)
5. **SharedMemory** — Legacy deprecated system
6. **AgentMemoryBuffer** — Per-agent conversation memory (context window)

### Consolidation Strategy

```python
# vetinari/memory/unified_memory.py
"""
Unified Memory Architecture for Vetinari

This module provides a single coherent memory system replacing:
- DualMemoryStore (agent episodic/semantic)
- MemoryStore (plan tracking)
- SharedMemory (deprecated)
- Per-agent buffers

A single IMemoryStore interface for all use cases.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class MemoryCategory(Enum):
    """Categories of memory in unified system."""
    # Agent episodic (experiences, conversations)
    AGENT_INTERACTION = "agent_interaction"
    AGENT_DISCOVERY = "agent_discovery"
    AGENT_DECISION = "agent_decision"

    # Planning & execution tracking
    PLAN_HISTORY = "plan_history"
    SUBTASK_OUTCOME = "subtask_outcome"

    # Model performance
    MODEL_PERFORMANCE = "model_performance"

    # Context & intermediate results
    CONTEXT_SUMMARY = "context_summary"
    INFERENCE_RESULT = "inference_result"

@dataclass
class UnifiedMemoryEntry:
    """Single unified memory entry."""
    id: str
    category: MemoryCategory
    agent: str
    content: str
    summary: str
    metadata: Dict[str, Any]
    timestamp: int
    provenance: str = "agent"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category.value,
            "agent": self.agent,
            "content": self.content,
            "summary": self.summary,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "provenance": self.provenance,
        }

class IUnifiedMemoryStore(ABC):
    """Unified memory interface replacing all backends."""

    @abstractmethod
    def remember(self, entry: UnifiedMemoryEntry) -> str:
        """Store a memory entry."""
        pass

    @abstractmethod
    def search(self, query: str,
               category: Optional[MemoryCategory] = None,
               agent: Optional[str] = None,
               limit: int = 10) -> List[UnifiedMemoryEntry]:
        """Search memories."""
        pass

    @abstractmethod
    def get_plan_history(self, plan_id: str) -> List[UnifiedMemoryEntry]:
        """Get all memories for a plan."""
        pass

    @abstractmethod
    def get_agent_context(self, agent: str, max_tokens: int = 2048) -> str:
        """Get formatted context for agent (summary + recent interactions)."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get memory stats."""
        pass

class UnifiedMemoryStore(IUnifiedMemoryStore):
    """
    Single unified memory store.
    Internally delegates to OcMemoryStore + MemoryStore for dual persistence.
    Presents unified interface to callers.
    """

    def __init__(self, oc_backend, memstore_backend, dual_memory_backend):
        self.oc = oc_backend
        self.memstore = memstore_backend
        self.dual = dual_memory_backend

    def remember(self, entry: UnifiedMemoryEntry) -> str:
        """Write to both backends for resilience."""
        # Convert to appropriate format for each backend
        entry_dict = entry.to_dict()

        # Write to OcMemoryStore (episodic/semantic)
        if entry.category in [MemoryCategory.AGENT_INTERACTION, MemoryCategory.AGENT_DISCOVERY]:
            oc_entry = {
                "id": entry.id,
                "agent": entry.agent,
                "entry_type": entry.category.value,
                "content": entry.content,
                "summary": entry.summary,
                "metadata": entry.metadata,
            }
            self.oc.remember(oc_entry)

        # Write to MemoryStore (plan tracking)
        if entry.category in [MemoryCategory.PLAN_HISTORY, MemoryCategory.SUBTASK_OUTCOME]:
            plan_id = entry.metadata.get("plan_id")
            if plan_id:
                plan_data = {
                    "plan_id": plan_id,
                    "goal": entry.metadata.get("goal"),
                    "content": entry.content,
                    "status": entry.metadata.get("status", "in_progress"),
                }
                self.memstore.write_plan_history(plan_data)

        return entry.id

    def search(self, query: str,
               category: Optional[MemoryCategory] = None,
               agent: Optional[str] = None,
               limit: int = 10) -> List[UnifiedMemoryEntry]:
        """Search across both backends."""
        # Query OcMemoryStore
        oc_results = self.oc.search(query, agent=agent, limit=limit)

        # Convert to unified format
        unified_results = []
        for oc_entry in oc_results:
            entry = UnifiedMemoryEntry(
                id=oc_entry.get("id"),
                category=MemoryCategory(oc_entry.get("entry_type", "agent_interaction")),
                agent=oc_entry.get("agent"),
                content=oc_entry.get("content"),
                summary=oc_entry.get("summary"),
                metadata=oc_entry.get("metadata", {}),
                timestamp=oc_entry.get("timestamp", 0),
            )
            if not category or entry.category == category:
                unified_results.append(entry)

        return unified_results[:limit]

    def get_plan_history(self, plan_id: str) -> List[UnifiedMemoryEntry]:
        """Get all memories for a plan."""
        plan_data = self.memstore.query_plan_history(plan_id=plan_id)

        results = []
        if plan_data:
            for plan in plan_data:
                entry = UnifiedMemoryEntry(
                    id=plan.get("plan_id"),
                    category=MemoryCategory.PLAN_HISTORY,
                    agent="planner",
                    content=plan.get("goal"),
                    summary=plan.get("plan_justification", ""),
                    metadata={
                        "plan_id": plan.get("plan_id"),
                        "status": plan.get("status"),
                        "risk_score": plan.get("risk_score"),
                    },
                    timestamp=0,
                )
                results.append(entry)

        return results

    def get_agent_context(self, agent: str, max_tokens: int = 2048) -> str:
        """Get formatted context for agent."""
        # Search for agent's recent interactions
        interactions = self.search("", agent=agent, limit=20)

        # Format for context
        context_lines = [f"Agent: {agent}"]
        token_count = 0

        for interaction in interactions:
            line = f"[{interaction.category.value}] {interaction.summary}"
            tokens = len(line.split())

            if token_count + tokens > max_tokens:
                break

            context_lines.append(line)
            token_count += tokens

        return "\n".join(context_lines)

    def get_stats(self) -> Dict[str, Any]:
        """Get memory stats."""
        oc_stats = self.oc.get_stats()
        memstore_stats = self.memstore.get_memory_stats()

        return {
            "total_entries": oc_stats.get("total_entries", 0) + memstore_stats.get("total_plans", 0),
            "episodic_entries": oc_stats.get("total_entries", 0),
            "plan_records": memstore_stats.get("total_plans", 0),
            "subtask_records": memstore_stats.get("total_subtasks", 0),
            "model_performance_records": memstore_stats.get("total_model_records", 0),
        }

# Migration guide for existing code
def migrate_to_unified_memory():
    """
    Migration path from multiple systems to unified.

    Old code:
        dual_store = DualMemoryStore()
        memory_store = MemoryStore()
        shared_mem = SharedMemory()

    New code:
        unified_store = UnifiedMemoryStore(oc, memstore, dual)

    All three use the same interface.
    """
    logger.info("Unified memory system is recommended for new code")
```

### Migration Plan

| Old System | New Category | Status |
|-----------|-------------|--------|
| DualMemoryStore | agent episodic/semantic | Wrapped in Unified |
| MemoryStore (plan tracking) | plan_history, subtask_outcome | Wrapped in Unified |
| SharedMemory | Deprecated → OC | Remove in v3.5 |
| Per-agent buffers | context_summary | Unified handling |
| EpisodeMemory | agent_interaction | OC backend |

### Estimated Effort

- **Research & design:** 4-6 hours
  - Analyze all existing systems: 2 hours
  - Design unified schema: 2-3 hours
  - Document migration path: 1 hour

- **Implementation:** 12-16 hours
  - Unified interface: 2-3 hours
  - Backend wrapping: 3-4 hours
  - Migration utilities: 2-3 hours
  - Testing: 3-4 hours
  - Documentation: 2-3 hours

- **Deprecation cycle:** 2-4 weeks
  - Phase 1: Introduce unified system alongside legacy
  - Phase 2: Migrate agents to unified API
  - Phase 3: Deprecate legacy systems
  - Phase 4: Remove legacy code (v3.5+)

---

## Summary Table: Feature Implementation Roadmap

| Feature | Effort (hrs) | Complexity | Priority | Dependencies |
|---------|------|----------|----------|------|
| **SLM/LLM Hybrid Routing** | 12-18 | Medium | High | Profiled models |
| **Speculative Decoding** | 28-36 | High | Medium | Draft + target models |
| **Continuous Batching** | 8-12 | Low | High | Async orchestration |
| **Context Window Management** | 6-10 | Low | High | tiktoken, memory system |
| **Typed Output Schemas** | 6-8 | Low | High | Pydantic |
| **Circuit Breakers** | 4-6 | Low | Medium | Resilience patterns |
| **Distributed Tracing** | 6-8 | Low | Medium | OpenTelemetry |
| **Performance Dashboard** | 8-12 | Low | Medium | SQLite, Flask, React |
| **Dynamic Complexity Routing** | 4-6 | Low | Medium | Heuristics library |
| **Cost Optimization** | 8-10 | Low | Medium | Analytics DB |
| **Memory Consolidation** | 16-22 | Medium | High | All memory backends |
| **TOTAL** | **106-148 hours** | - | - | - |

---

## Getting Started: Recommended Implementation Order

### Phase 1: Foundation (Weeks 1-2, ~30 hours)
1. **Continuous Batching** (8-12h) — High-impact, low complexity
2. **Typed Output Schemas** (6-8h) — Foundational for reliability
3. **Context Window Management** (6-10h) — Unlocks longer conversations

### Phase 2: Intelligence (Weeks 3-4, ~30 hours)
4. **SLM/LLM Hybrid Routing** (12-18h) — Significant latency savings
5. **Dynamic Complexity Routing** (4-6h) — Quick wins for simple tasks

### Phase 3: Resilience (Weeks 5, ~20 hours)
6. **Circuit Breakers** (4-6h) — Prevents cascading failures
7. **Distributed Tracing** (6-8h) — Visibility into system behavior
8. **Performance Dashboard** (8-12h) — Monitoring and optimization

### Phase 4: Optimization (Weeks 6-8, ~30 hours)
9. **Cost Optimization** (8-10h) — Budget management
10. **Memory Consolidation** (16-22h) — System simplification
11. **Speculative Decoding** (28-36h) — Advanced optimization

---

## LM Studio Configuration Recommendations

```json
{
  "core": {
    "max_concurrent_predictions": 8,
    "gpu_device_id": 0
  },
  "models": {
    "small_model": {
      "id": "phi-2",
      "context_length": 2048,
      "idle_ttl_seconds": 300
    },
    "large_model": {
      "id": "mistral-7b",
      "context_length": 4096,
      "idle_ttl_seconds": 600
    }
  },
  "kv_cache": {
    "unified": true,
    "preallocation_ratio": 0.8
  },
  "backends": {
    "preferred": "llama.cpp",
    "fallback": "mlx"
  }
}
```

---

## References

### LM Studio API Documentation
- [Manage Models in Memory](https://lmstudio.ai/docs/typescript/manage-models/loading)
- [Idle TTL and Auto-Evict](https://lmstudio.ai/docs/developer/core/ttl-and-auto-evict)
- [Parallel Requests](https://lmstudio.ai/docs/app/advanced/parallel-requests)
- [LM Studio Blog 0.4.0](https://lmstudio.ai/blog/0.4.0)
- [Unified MLX Engine](https://lmstudio.ai/blog/unified-mlx-engine)

### Advanced Topics
- [Speculative Decoding Survey](https://blog.codingconfessions.com/p/a-selective-survey-of-speculative-decoding)
- [Circuit Breaker Pattern](https://medium.com/@usama19026/building-resilient-applications-circuit-breaker-pattern-with-exponential-backoff-fc14ba0a0beb)
- [OpenTelemetry for LLMs](https://opentelemetry.io/blog/2024/llm-observability/)
- [Pydantic LLM Integration](https://pydantic.dev/articles/llm-intro)
- [Token Counting](https://ai.google.dev/gemini-api/docs/tokens)
- [LLM Routing Strategies](https://medium.com/google-cloud/a-developers-guide-to-model-routing-1f21ecc34d60)

---

**Document Version:** 1.0
**Date:** March 2026
**LM Studio Version:** 0.4.6
**Vetinari Version:** 0.3.0+
