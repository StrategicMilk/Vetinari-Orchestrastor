# Vetinari Feature Research: AI Agent Orchestration Landscape (March 2026)

Deep research into popular and requested features across the AI agent orchestration ecosystem, with specific integration recommendations for Vetinari.

---

## Table of Contents

1. [Popular Framework Features](#1-popular-framework-features)
2. [Local-First AI Features](#2-local-first-ai-features)
3. [Agent Orchestration Trends (2025-2026)](#3-agent-orchestration-trends)
4. [Monitoring & Observability](#4-monitoring--observability)
5. [Collaboration Features](#5-collaboration-features)
6. [Developer Experience](#6-developer-experience)
7. [Security Features](#7-security-features)
8. [Deployment Patterns](#8-deployment-patterns)
9. [Agent Memory State of the Art](#9-agent-memory-state-of-the-art)
10. [Plugin/Extension Systems](#10-pluginextension-systems)

---

## 1. Popular Framework Features

### What Makes Frameworks Successful

The top frameworks have converged on distinct philosophies:

- **LangGraph** (graph-based workflows): Stateful directed graphs with conditional branching, durable execution (survives failures and resumes), human-in-the-loop checkpoints, and comprehensive memory (short-term working + long-term persistent). Reached v1.0 in late 2025 and became the default runtime for all LangChain agents.

- **CrewAI** (role-based teams): Intuitive role assignment with clear responsibilities. Excels at task-oriented collaboration with built-in business workflow patterns. Fastest time-to-value for team-based automation.

- **AutoGen** (conversational agents): Multi-agent conversation architecture with asynchronous task execution. v0.4 adopted event-driven architecture with three layers: Core, AgentChat, and Extensions.

- **OpenHands/OpenDevin** (coding agents): Docker-sandboxed runtime with bash shell, web browser, and IPython server. Agents can write code, browse the web, and run commands in isolated environments.

- **SWE-Agent** (bug fixing): Reads issue context, modifies multiple files, submits complete PRs with reasoning and test cases. Clear separation between LLM planning and execution engine.

### Key Success Factors Across All Frameworks

| Factor | Description |
|---|---|
| Graph-based orchestration | All major frameworks converging on graph/workflow execution models |
| Durable execution | Agents persist through failures and resume automatically |
| Human-in-the-loop | Inspect and modify state at any checkpoint |
| MCP protocol support | Standardized tool integration ("the USB port for agents") |
| Streaming & real-time | Token-by-token streaming of agent reasoning |
| Multi-model support | Route different tasks to different models |

### Vetinari Integration Opportunities

| Feature | Impact | Effort | Notes |
|---|---|---|---|
| **Graph-based plan visualization** | HIGH | MEDIUM | Vetinari already has wave-based DAG execution; expose the task graph in the web UI with interactive visualization (nodes = tasks, edges = dependencies). Users expect to see and manipulate the execution graph. |
| **Durable execution / checkpointing** | HIGH | HIGH | Persist orchestrator state to disk so that interrupted plans can resume from the last completed wave. LangGraph's durability is a top differentiator. |
| **Human-in-the-loop gates** | HIGH | MEDIUM | Add approval checkpoints between waves or before high-risk tasks. The approval_workflow.md already exists; wire it into TwoLayerOrchestrator. |
| **Streaming agent output** | MEDIUM | MEDIUM | Stream agent reasoning tokens to the Flask web UI via Server-Sent Events (SSE) or WebSocket. Users want to watch agents think in real time. |
| **Multi-framework interop** | LOW | HIGH | The "Agentic Mesh" trend (LangGraph brain orchestrating CrewAI teams) is emerging but premature for a local-first system. |

---

## 2. Local-First AI Features

### Most Requested Features for Local-First Systems

1. **Complete data privacy**: Zero data leaves the machine. This is the #1 reason users choose local-first.
2. **Cost efficiency**: Eliminate per-token API costs (up to 100% reduction).
3. **Offline operation**: Full functionality without internet connectivity.
4. **Local LLM support**: Native integration with Ollama, LM Studio, vLLM -- not just API wrappers.
5. **Real-time data processing**: Agents subscribing to real-time events for dynamic environments.
6. **Parallel task execution**: Background task execution while starting new tasks (2026 trend).
7. **Context persistence**: Persistent memory that survives across sessions.
8. **Low-code/no-code interfaces**: Visual builders and templates for rapid deployment.
9. **Enterprise observability**: Built-in telemetry, hooks, and filters for monitoring AI behavior.

### Vetinari Integration Opportunities

| Feature | Impact | Effort | Notes |
|---|---|---|---|
| **Offline mode indicator** | MEDIUM | LOW | Show connection status in web UI. Vetinari already runs locally via LM Studio; make this a visible selling point. |
| **Cost savings dashboard** | HIGH | LOW | Track estimated API costs avoided by running locally. Compare against OpenAI/Anthropic pricing. Users love seeing savings quantified. Vetinari's analytics module already has cost tracking -- extend it. |
| **Visual plan builder** | HIGH | HIGH | Drag-and-drop task graph editor in the web UI. Low-code trend is strong. Let users visually compose agent workflows. |
| **Real-time event subscriptions** | MEDIUM | HIGH | File watcher / webhook integration so agents react to external events (file changes, git pushes, CI results). |
| **Background execution queue** | HIGH | MEDIUM | Queue multiple plans for sequential or parallel execution. Users want to fire-and-forget multiple tasks. |
| **Model download manager** | MEDIUM | MEDIUM | In-UI model browsing and downloading from Hugging Face / Ollama registry. Reduce friction of local model management. |

---

## 3. Agent Orchestration Trends (2025-2026)

### MCP (Model Context Protocol)

- Created by Anthropic, donated to Linux Foundation's AAIF in December 2025.
- 97+ million monthly SDK downloads by February 2026.
- Adopted by every major AI provider (Anthropic, OpenAI, Google, Microsoft, Amazon).
- 50+ technology partners (Atlassian, MongoDB, PayPal, Salesforce, SAP).
- Purpose: Universal adapter connecting agents to tools, APIs, and data sources.
- Analogy: "USB port for agents" -- before MCP, every framework had its own tool integration.

### A2A (Agent-to-Agent Protocol)

- Created by Google (April 2025), donated to Linux Foundation (June 2025).
- Purpose: Standardized secure communication and delegation between autonomous agents.
- Enables agent discovery, capability negotiation, and task delegation.
- Key distinction: MCP = agent-to-tool; A2A = agent-to-agent.

### ACP (Agent Communication Protocol)

- Emerging third standard for structured inter-agent messaging.
- Complementary to both MCP and A2A.

### The "Agentic Mesh" Future

The industry is moving toward modular ecosystems where different frameworks interoperate. A LangGraph "brain" might orchestrate a CrewAI "marketing team" while calling specialized tools. Standardization via MCP and A2A makes this possible.

### Vetinari Integration Opportunities

| Feature | Impact | Effort | Notes |
|---|---|---|---|
| **MCP server implementation** | HIGH | MEDIUM | Expose Vetinari's agents as MCP tool servers. Other agents/IDEs could invoke Vetinari's Planner, Researcher, etc. as tools. This is the single highest-leverage protocol adoption. |
| **MCP client support** | HIGH | MEDIUM | Let Vetinari agents call external MCP servers (databases, APIs, file systems). Instantly gains access to the entire MCP ecosystem (GitHub, Slack, databases, etc.). |
| **A2A agent discovery** | MEDIUM | HIGH | Implement A2A Agent Cards so external agents can discover and delegate to Vetinari agents. Future-proofs for multi-system orchestration. |
| **A2A task delegation** | MEDIUM | HIGH | Accept inbound A2A task requests. Vetinari becomes a "specialist service" that other orchestrators can call. |
| **Protocol gateway** | LOW | HIGH | Unified gateway that translates between MCP, A2A, and internal Vetinari protocols. Premature until both protocols mature further. |

---

## 4. Monitoring & Observability

### Essential Features (Industry Consensus)

From Langfuse, Helicone, Braintrust, and others, the essential observability features are:

1. **Distributed tracing**: Nested spans showing the full execution tree (plan -> wave -> agent -> LLM call -> tool call). Every decision point captured.
2. **Cost tracking**: Per-request, per-agent, per-plan cost attribution. Budget alerts and spending limits.
3. **Latency monitoring**: P50/P95/P99 latency for each agent, model, and tool. Bottleneck identification.
4. **Quality scoring**: Automated evaluation of agent outputs (LLM-as-judge, rule-based, human feedback collection).
5. **Prompt versioning**: Track prompt changes across runs. A/B test different prompts.
6. **Token usage analytics**: Input/output token counts per request. Context window utilization.
7. **Error classification**: Categorize failures (model errors, tool failures, timeout, safety violations).
8. **Real-time dashboards**: Live view of running plans with drill-down into individual agents.
9. **Alerting**: Configurable alerts for cost spikes, latency degradation, error rate increases.
10. **Export/integration**: OpenTelemetry export, webhook notifications, Slack/Discord alerts.

### Observability Platform Strengths

| Platform | Strength |
|---|---|
| **Langfuse** | Open-source, comprehensive tracing, prompt management, evaluation, user feedback |
| **Helicone** | AI gateway with routing/failover/caching, cost attribution with virtual API keys, budget alerts |
| **Braintrust** | Evaluation-first (QA framework for LLMs), 80x faster queries, production trace analysis |
| **Arize Phoenix** | Open-source, strong on embeddings analysis and drift detection |

### Vetinari Integration Opportunities

| Feature | Impact | Effort | Notes |
|---|---|---|---|
| **OpenTelemetry trace export** | HIGH | MEDIUM | Instrument TwoLayerOrchestrator with OTel spans. Export to any OTel-compatible backend (Jaeger, Grafana Tempo, Langfuse). Makes Vetinari observable by any enterprise monitoring stack. |
| **Built-in trace viewer** | HIGH | MEDIUM | Add a trace/timeline view to the web UI showing plan -> wave -> agent -> LLM call hierarchy with timing. Vetinari already has analytics; this is the visual complement. |
| **LLM-as-judge evaluation** | MEDIUM | MEDIUM | After each agent completes, optionally run a lightweight evaluation pass scoring quality. Feed scores into the Thompson Sampling model router. |
| **Prompt versioning UI** | MEDIUM | LOW | Track and display which system prompts were used for each agent invocation. Enable A/B testing of prompts. |
| **Budget controls** | HIGH | LOW | Set per-plan and per-agent token/cost budgets. Halt execution when limits are reached. Vetinari's cost tracking already exists; add enforcement. |
| **Langfuse integration** | MEDIUM | LOW | Langfuse is open-source and self-hostable (fits local-first philosophy). Add a Langfuse exporter as an optional integration. |

---

## 5. Collaboration Features

### Current State of the Art

The industry is entering the era of "multiplayer AI" where agents participate directly in team conversations, maintain context across interactions, and coordinate with both humans and other AI systems.

Key patterns:

1. **Shared workspaces**: Multiple users observe and interact with the same plan execution.
2. **Role-based access**: Different team members have different permissions (viewer, operator, admin).
3. **Real-time monitoring**: Live dashboards showing agent activity across the team.
4. **Commenting/annotation**: Users can annotate agent outputs, flag issues, provide feedback.
5. **Agent-human handoff**: Agents escalate to humans when confidence is low; humans delegate back.
6. **Shared memory**: Team-wide knowledge base that all agents and users can contribute to.
7. **Audit trail**: Complete history of who (human or agent) did what and when.

### Vetinari Integration Opportunities

| Feature | Impact | Effort | Notes |
|---|---|---|---|
| **Multi-user web UI** | HIGH | HIGH | Add authentication and session management to Flask. Multiple users can view/control plans simultaneously. WebSocket for real-time sync. |
| **Plan sharing** | MEDIUM | LOW | Export/import plans as JSON. Share plan templates between users or teams. |
| **Commenting system** | MEDIUM | MEDIUM | Allow users to annotate agent outputs in the web UI. Comments persist with the plan. |
| **Agent-human handoff** | HIGH | MEDIUM | When an agent's confidence is below threshold, pause execution and request human input via the web UI. Already partially supported by approval workflow. |
| **Shared memory workspace** | MEDIUM | MEDIUM | Extend DualMemoryStore to support named workspaces. Teams share a common knowledge base. |
| **Activity feed** | MEDIUM | LOW | Real-time feed in the web UI showing all agent actions, completions, and decisions. Like a Slack channel for agent activity. |

---

## 6. Developer Experience

### Key DX Trends (2025-2026)

1. **Agentic IDEs**: Cursor, Windsurf, and others have agents that index codebases, watch terminals, and offer edits based on deep project understanding.
2. **CLI-first workflows**: Developers prefer terminal-based agent interfaces (Claude Code, Gemini CLI) for iterative debugging and multi-step tasks.
3. **In-editor chat**: Explanations, debugging, and documentation support without leaving the editor.
4. **Debug mode**: Human-in-the-loop debugging where agents instrument code with logging (Cursor 2.0).
5. **Multi-tool layering**: Tools don't compete, they layer -- editor assistants for speed, agents for multi-file changes, security tools for scanning, review platforms for PRs.
6. **Project context awareness**: Agents that understand the full project structure, not just individual files.

### Vetinari Integration Opportunities

| Feature | Impact | Effort | Notes |
|---|---|---|---|
| **CLI interface** | HIGH | MEDIUM | Add a `vetinari` CLI (`vetinari plan "build auth system"`, `vetinari status`, `vetinari logs`). CLI-first developers are a large segment. Use Click or Typer. |
| **VS Code extension** | HIGH | HIGH | Show plan status, agent activity, and memory in a VS Code sidebar. Let users trigger plans from the editor. |
| **Debug/explain mode** | HIGH | MEDIUM | Verbose mode where agents explain every decision step-by-step. Invaluable for understanding and trusting the system. |
| **Project context indexing** | HIGH | MEDIUM | Auto-index the project structure (files, functions, dependencies) on startup. Feed this to Researcher and Planner agents for better task decomposition. Already partially exists in Researcher's code_discovery mode. |
| **Configuration file** | MEDIUM | LOW | `vetinari.yaml` project config file for model preferences, agent settings, safety rules. Reduce UI-only configuration. |
| **REPL/interactive mode** | MEDIUM | LOW | Interactive shell where users can chat with agents, inspect memory, and run ad-hoc queries. |

---

## 7. Security Features

### Becoming Standard (2025-2026)

OWASP released an "AI Agent Security Top 10" for 2026, establishing industry baselines:

1. **Sandboxed execution**: Docker/MicroVM isolation for agent-generated code. No host system access.
2. **Permission-gated tool access**: Every tool wrapped in a PermissionManager requiring explicit approval. Least-privilege by default.
3. **Immutable audit trails**: Log every action (who initiated, what was done, why). Regulatory requirements demand not just "what" but "why."
4. **Input validation**: Sanitize all agent inputs to prevent prompt injection, SQL injection, and path traversal.
5. **Output filtering**: Content safety filters on all agent outputs before they reach users.
6. **Rate limiting**: Per-agent and per-tool rate limits to prevent runaway execution.
7. **Secret management**: Never expose API keys, tokens, or credentials in agent context. Use vault-based secret injection.
8. **Network isolation**: Agents cannot make arbitrary network requests. Allowlist-based network policies.
9. **Execution timeouts**: Hard time limits on all agent operations. Kill processes that exceed limits.
10. **Compliance reporting**: Generate audit reports for SOC 2, GDPR, HIPAA compliance.

### Vetinari Integration Opportunities

| Feature | Impact | Effort | Notes |
|---|---|---|---|
| **Permission system overhaul** | HIGH | MEDIUM | Vetinari has `safety/` module. Extend it with explicit permission declarations per agent per tool. Builder can write files; Researcher cannot. Make this configurable. |
| **Structured audit log** | HIGH | MEDIUM | Every agent action logged in a structured format (JSON lines) with timestamp, agent, action, inputs, outputs, decision rationale. Essential for trust and debugging. |
| **Execution sandbox** | HIGH | HIGH | Run Builder agent's code execution in Docker containers. Vetinari already runs locally, but sandboxing prevents accidental damage. |
| **Secret vault integration** | MEDIUM | LOW | Integrate with a local secret store (e.g., keyring, HashiCorp Vault). Never pass secrets through agent context windows. Vetinari already has CredentialVault -- ensure it's used consistently. |
| **Rate limiting per agent** | MEDIUM | LOW | Configurable rate limits on LLM calls per agent per minute. Prevent runaway loops from consuming all model capacity. |
| **Network policy enforcement** | MEDIUM | MEDIUM | Allowlist of domains/endpoints each agent can access. Researcher can access docs; Builder cannot make network requests. |
| **Compliance report generator** | LOW | MEDIUM | Auto-generate audit reports from structured logs. Niche but valuable for enterprise adoption. |

---

## 8. Deployment Patterns

### Popular Patterns (2025-2026)

1. **Docker Compose (local)**: Most common for local-first systems. Declarative stack definition with model servers, agent runtime, memory stores, and web UI as separate services.

2. **Kubernetes (production)**: One pod per agent role, autoscaled by demand. GPU scheduling with NVIDIA operator. Message buses (Kafka, RabbitMQ) for inter-agent communication. 65% of enterprise AI uses hybrid architectures by 2025.

3. **Edge deployment**: Agents on edge appliances / IoT gateways for local data processing. Only sync summaries to cloud. Dell predicts significant edge AI growth in 2026.

4. **Hybrid cloud**: Centralized control plane orchestrates distributed agents. Cloud controllers dispatch workloads to edge/on-prem agents. State sync across environments.

5. **Docker Model Runner**: Docker's native model serving (February 2026) with vLLM backend and Metal GPU acceleration on Apple Silicon. Integrated with LangChain, AutoGen, ControlFlow.

### Vetinari Integration Opportunities

| Feature | Impact | Effort | Notes |
|---|---|---|---|
| **Docker Compose deployment** | HIGH | LOW | Create `docker-compose.yml` with LM Studio, Vetinari server, and optional Langfuse as services. One-command deployment. |
| **Docker Model Runner support** | MEDIUM | MEDIUM | Add adapter for Docker Model Runner as an alternative to LM Studio. Expands deployment options without requiring LM Studio. |
| **Health check endpoints** | HIGH | LOW | Add `/api/health` and `/api/ready` to Flask. Standard for any containerized service. Required for Docker/K8s. |
| **Configuration via environment variables** | MEDIUM | LOW | Support `VETINARI_MODEL_URL`, `VETINARI_PORT`, etc. Standard for 12-factor apps and container deployment. |
| **Helm chart** | LOW | MEDIUM | Kubernetes Helm chart for production deployment. Premature unless enterprise demand exists. |
| **State export/import** | MEDIUM | LOW | Export full system state (plans, memory, config) as a portable archive. Enables migration between deployments. |

---

## 9. Agent Memory State of the Art

### Memory Taxonomy (2026 Consensus)

1. **Episodic memory**: Remembers specific events, conversations, and cases. "What happened last time we deployed to production?"
2. **Semantic memory**: General knowledge, rules, facts, domain expertise. "Our API uses JWT authentication with RS256."
3. **Procedural memory**: Learned skills and routines. "How to run the test suite for this project."

### Leading Solutions

| Solution | Architecture | Strength | Weakness |
|---|---|---|---|
| **Mem0** | Hybrid vector + graph | Fastest path to production, 92% latency improvement, managed service | Vendor lock-in risk |
| **Zep** | Temporal knowledge graph | Best for entity/relationship tracking over time, tracks how facts change | More complex architecture |
| **LangMem** | Flat key-value + vector | Simple, integrates with LangGraph | Framework-dependent, high latency (p50: 18s) |
| **Letta** (f.k.a. MemGPT) | Tiered memory (core/archival/recall) | Self-editing memory, OS-like architecture | Complex setup |

### Key Architectural Patterns

- **Dual-layer architecture**: Hot path (recent messages + summarized graph state) + cold path (retrieval from specialized services). This is the 2026 production pattern.
- **Graph memory**: Knowledge graphs with entity extraction and relationship modeling outperform flat vector stores for reasoning tasks (+26% accuracy in Mem0 research).
- **Temporal tracking**: Zep's temporal knowledge graph tracks how facts change over time -- critical for long-running projects.
- **Cross-agent memory sharing**: Agents share a common memory space but with scoped access (agent-specific vs. shared).

### Vetinari Integration Opportunities

| Feature | Impact | Effort | Notes |
|---|---|---|---|
| **Graph-based memory** | HIGH | HIGH | Extend DualMemoryStore with a lightweight knowledge graph (e.g., using NetworkX or a local graph DB). Extract entities and relationships from agent outputs. Enables reasoning like "which files did Builder modify that Researcher flagged as risky?" |
| **Temporal memory tracking** | HIGH | MEDIUM | Track how facts change over time. When Researcher discovers new info that contradicts earlier findings, flag the conflict. Essential for long-running plans. |
| **Episodic memory store** | HIGH | MEDIUM | Store complete execution episodes (plan + inputs + outputs + decisions). Enable "what happened last time we tried X?" queries. Use SQLite for local storage. |
| **Memory compaction** | MEDIUM | MEDIUM | Periodically summarize old memories to keep the knowledge base manageable. LLM-based summarization of stale entries. |
| **Cross-plan memory** | MEDIUM | LOW | Allow agents to recall learnings from previous plans. "Last time we built a REST API, we used Flask with SQLAlchemy." Currently memory is plan-scoped; make it project-scoped. |
| **Memory search API** | MEDIUM | LOW | Expose memory search in the web UI. Let users browse and edit what agents remember. Transparency builds trust. |

---

## 10. Plugin/Extension Systems

### Popular Approaches

1. **Semantic Kernel (Microsoft)**: Modular plugin system where existing code becomes AI-callable through OpenAPI-based plugins. Built-in connectors for AI services. Most mature enterprise plugin architecture.

2. **AutoGen v0.4 (three-layer)**: Core Layer (foundations) -> AgentChat Layer (task API) -> Extensions Layer (third-party integrations). Clean separation of concerns.

3. **Haystack (deepset)**: Components convert into callable tools automatically. Mix components from different providers. Extension without framework rewrites.

4. **MCP as universal plugin protocol**: MCP servers are effectively plugins. Any MCP server can extend any MCP-compatible agent. This is rapidly becoming the dominant extension pattern.

5. **LangChain Tools**: Function-based tools with schema definitions. Largest ecosystem of pre-built tools.

### Vetinari Integration Opportunities

| Feature | Impact | Effort | Notes |
|---|---|---|---|
| **Agent plugin system** | HIGH | HIGH | Define a `VetinariPlugin` base class with hooks: `on_plan_created`, `on_wave_started`, `on_agent_invoked`, `on_task_completed`, `on_plan_finished`. Users register plugins via config. Enables custom logging, notifications, integrations. |
| **Custom agent registration** | HIGH | MEDIUM | Allow users to register custom agents beyond the 6 built-in ones. Define agents via YAML (name, system prompt, capabilities, file jurisdiction). Loaded from `plugins/agents/`. |
| **Tool registry** | HIGH | MEDIUM | Formalize the tool system. Each tool has a schema (name, description, parameters, returns). Agents discover available tools at runtime. Foundation for MCP support. |
| **MCP server plugin** | HIGH | MEDIUM | Each MCP server becomes a plugin. `vetinari install mcp-server-github` adds GitHub tools to all agents. Leverages the 50+ existing MCP servers. |
| **Webhook/event system** | MEDIUM | LOW | Fire webhooks on plan events (started, completed, failed). Enables integration with CI/CD, Slack, Discord, email without writing plugins. |
| **Plugin marketplace concept** | LOW | HIGH | Browse and install community plugins. Premature but worth designing the architecture to support it later. |

---

## Priority Recommendations for Vetinari

### Tier 1: Highest Impact, Reasonable Effort (Do First)

| # | Feature | Impact | Effort | Rationale |
|---|---|---|---|---|
| 1 | **MCP client support** | HIGH | MEDIUM | Instantly connects Vetinari to the entire MCP ecosystem. Let agents use GitHub, Slack, databases, file systems via standard protocol. |
| 2 | **OpenTelemetry tracing** | HIGH | MEDIUM | Makes Vetinari observable by any monitoring stack. Foundation for the trace viewer, Langfuse integration, and debugging. |
| 3 | **CLI interface** | HIGH | MEDIUM | `vetinari plan`, `vetinari status`, `vetinari logs`. CLI-first developers are underserved. Complements the web UI. |
| 4 | **Durable execution / checkpointing** | HIGH | HIGH | Resume interrupted plans. This is LangGraph's #1 differentiator. Critical for long-running tasks over local models (which are slower). |
| 5 | **Budget controls** | HIGH | LOW | Enforce per-plan token/cost limits. Already have cost tracking; add enforcement. Prevents runaway execution. |
| 6 | **Docker Compose deployment** | HIGH | LOW | One-command deployment. Removes setup friction entirely. |
| 7 | **Structured audit log** | HIGH | MEDIUM | JSON-lines log of every agent action. Foundation for compliance, debugging, and trust. |
| 8 | **Debug/explain mode** | HIGH | MEDIUM | Verbose mode showing agent decision rationale. Essential for users to understand and trust the system. |

### Tier 2: High Impact, Higher Effort (Do Next)

| # | Feature | Impact | Effort | Rationale |
|---|---|---|---|---|
| 9 | **Graph-based memory** | HIGH | HIGH | Knowledge graph in DualMemoryStore. Enables cross-agent reasoning and entity tracking. |
| 10 | **MCP server implementation** | HIGH | MEDIUM | Expose Vetinari agents as MCP tools. Other systems can use Vetinari as a specialist. |
| 11 | **Agent plugin system** | HIGH | HIGH | Lifecycle hooks for custom extensions. Foundation for the plugin ecosystem. |
| 12 | **Visual plan builder** | HIGH | HIGH | Drag-and-drop task graph editor. Captures the low-code market. |
| 13 | **Built-in trace viewer** | HIGH | MEDIUM | Timeline visualization of plan execution in the web UI. |
| 14 | **VS Code extension** | HIGH | HIGH | Plan status, agent activity, and memory in VS Code sidebar. |
| 15 | **Human-in-the-loop gates** | HIGH | MEDIUM | Approval checkpoints between waves or before high-risk tasks. |

### Tier 3: Medium Impact, Variable Effort (Do Later)

| # | Feature | Impact | Effort | Rationale |
|---|---|---|---|---|
| 16 | **Episodic memory store** | HIGH | MEDIUM | Store execution episodes for cross-plan learning. |
| 17 | **Temporal memory tracking** | HIGH | MEDIUM | Track fact changes over time. Flag contradictions. |
| 18 | **Cost savings dashboard** | MEDIUM | LOW | Show money saved by running locally vs. API. Great marketing. |
| 19 | **Tool registry** | HIGH | MEDIUM | Formalize tool schemas. Foundation for MCP and plugin system. |
| 20 | **Multi-user web UI** | HIGH | HIGH | Authentication, sessions, real-time sync. Enterprise requirement. |
| 21 | **A2A protocol support** | MEDIUM | HIGH | Agent discovery and delegation. Future-proofs for multi-system world. |
| 22 | **Execution sandbox** | HIGH | HIGH | Docker isolation for Builder's code execution. |
| 23 | **Permission system overhaul** | HIGH | MEDIUM | Per-agent per-tool permission declarations. |
| 24 | **LLM-as-judge evaluation** | MEDIUM | MEDIUM | Auto-score agent outputs. Feed into Thompson Sampling router. |
| 25 | **Custom agent registration** | HIGH | MEDIUM | User-defined agents via YAML. |

---

## Competitive Positioning

### Vetinari's Unique Strengths (to emphasize)

1. **Six-agent cognitive architecture**: No other framework has this level of specialized role separation with clear cognitive division of labour. Most use generic "agent" abstractions.
2. **Local-first with LM Studio**: True privacy and zero API costs. Growing demand as data sovereignty concerns increase.
3. **Thompson Sampling model routing**: Sophisticated bandit-based model selection is rare. Most frameworks use static routing or simple fallback chains.
4. **Wave-based execution**: Structured parallel execution with dependency management. More principled than CrewAI's simple sequential/parallel modes.
5. **Built-in safety guardrails**: Proactive safety (content filtering, permission policies) rather than bolt-on.
6. **Quality gates**: Mandatory Quality agent review of all Builder output. No other framework enforces this by default.

### Gaps to Close

1. **No MCP support**: This is now table stakes. Every major framework supports MCP.
2. **No CLI interface**: CLI-first developers have no entry point.
3. **No durable execution**: Plans that crash are lost. LangGraph solved this in 2025.
4. **No OpenTelemetry**: Can't integrate with enterprise monitoring stacks.
5. **No plugin system**: Users can't extend the system without modifying source code.
6. **Memory is basic**: No knowledge graph, no episodic memory, no temporal tracking. The state of the art has moved significantly.

---

## Sources

### Framework Comparisons
- [LangGraph vs CrewAI vs AutoGen: Top 10 AI Agent Frameworks (o-mega.ai)](https://o-mega.ai/articles/langgraph-vs-crewai-vs-autogen-top-10-agent-frameworks-2026)
- [Top 6 AI Agent Frameworks in 2026 (Turing)](https://www.turing.com/resources/ai-agent-frameworks)
- [CrewAI vs LangGraph vs AutoGen (DataCamp)](https://www.datacamp.com/tutorial/crewai-vs-langgraph-vs-autogen)
- [AI Agent Frameworks Compared 2026 (Arsum)](https://arsum.com/blog/posts/ai-agent-frameworks/)
- [Top 5 Agentic AI Frameworks 2026 (FutureAGI)](https://futureagi.substack.com/p/top-5-agentic-ai-frameworks-to-watch)

### Local-First AI
- [Top 10 Open Source AI Agents You Can Run Locally (Fast.io)](https://fast.io/resources/top-10-open-source-ai-agents/)
- [5 Key Trends Shaping Agentic Development 2026 (The New Stack)](https://thenewstack.io/5-key-trends-shaping-agentic-development-in-2026/)
- [AI Agent Trends 2026 (Salesmate)](https://www.salesmate.io/blog/future-of-ai-agents/)

### Protocols (MCP, A2A)
- [MCP vs A2A: Complete Guide 2026 (DEV Community)](https://dev.to/pockit_tools/mcp-vs-a2a-the-complete-guide-to-ai-agent-protocols-in-2026-30li)
- [A2A Protocol (a2aprotocol.ai)](https://a2aprotocol.ai/)
- [What Is Agent2Agent Protocol (IBM)](https://www.ibm.com/think/topics/agent2agent-protocol)
- [AI Agent Protocols 2026 (ruh.ai)](https://www.ruh.ai/blogs/ai-agent-protocols-2026-complete-guide)
- [MCP vs A2A (Auth0)](https://auth0.com/blog/mcp-vs-a2a/)

### Observability
- [AI Observability Tools Buyer's Guide 2026 (Braintrust)](https://www.braintrust.dev/articles/best-ai-observability-tools-2026)
- [8 AI Observability Platforms Compared (Softcery)](https://softcery.com/lab/top-8-observability-platforms-for-ai-agents-in-2025)
- [Complete Guide to LLM Observability (Helicone)](https://www.helicone.ai/blog/the-complete-guide-to-LLM-observability-platforms)
- [Top 5 AI Agent Observability Platforms 2026 (o-mega.ai)](https://o-mega.ai/articles/top-5-ai-agent-observability-platforms-the-ultimate-2026-guide)

### Security
- [AI Agent Security Best Practices (IBM)](https://www.ibm.com/think/tutorials/ai-agent-security)
- [How to Sandbox AI Agents 2026 (Northflank)](https://northflank.com/blog/how-to-sandbox-ai-agents)
- [OWASP AI Agent Security Top 10 2026](https://medium.com/@oracle_43885/owasps-ai-agent-security-top-10-agent-security-risks-2026-fc5c435e86eb)
- [AI Agent Security Cheat Sheet (OWASP)](https://cheatsheetseries.owasp.org/cheatsheets/AI_Agent_Security_Cheat_Sheet.html)
- [Security for Production AI Agents 2026 (iain.so)](https://iain.so/security-for-production-ai-agents-in-2026)

### Deployment
- [Docker AI for Agent Builders (KDnuggets)](https://www.kdnuggets.com/docker-ai-for-agent-builders-models-tools-and-cloud-offload)
- [Agentic AI and Docker Architecture (dasroot.net)](https://dasroot.net/posts/2026/03/agentic-ai-docker-architecture-performance-security/)
- [Hybrid AI Agent Architectures 2025 (markaicode.com)](https://markaicode.com/tech/hybrid-ai-agent-architectures-2025/)

### Memory Systems
- [Graph Memory for AI Agents (Mem0)](https://mem0.ai/blog/graph-memory-solutions-ai-agents)
- [Mem0 vs Zep vs LangMem vs MemoClaw (DEV Community)](https://dev.to/anajuliabit/mem0-vs-zep-vs-langmem-vs-memoclaw-ai-agent-memory-comparison-2026-1l1k)
- [A-MEM: Agentic Memory for LLM Agents (arXiv)](https://arxiv.org/abs/2502.12110)
- [Beyond Short-term Memory: 3 Types of Long-term Memory (MLM)](https://machinelearningmastery.com/beyond-short-term-memory-the-3-types-of-long-term-memory-ai-agents-need/)

### Developer Experience
- [Best AI Coding Agents 2026 (Faros AI)](https://www.faros.ai/blog/best-ai-coding-agents-2026)
- [Best Agentic IDEs 2026 (Builder.io)](https://www.builder.io/blog/agentic-ide)
- [From Single User to Team Collaboration (The New Stack)](https://thenewstack.io/the-next-era-of-ai-from-single-user-to-team-collaboration/)

### Plugin Systems
- [Agentic AI Frameworks: Key Components (Exabeam)](https://www.exabeam.com/explainers/agentic-ai/agentic-ai-frameworks-key-components-top-8-options/)
- [Multi-Agent Frameworks for Enterprise (adopt.ai)](https://www.adopt.ai/blog/multi-agent-frameworks)

### Thompson Sampling
- [How Thompson Sampling Works (SourcePilot)](https://sourcepilot.co/blog/2025/11/22/how-thompson-sampling-works)
