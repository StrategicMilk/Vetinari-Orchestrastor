# Third-Party Licenses And Attribution Ledger

> Release status: initial public ledger. This is a root attribution
> artifact, but it is not final legal signoff. Rows marked `release-blocking`
> or `unresolved` must be closed before a release artifact can claim complete
> license, NOTICE, provenance, or privacy proof.

Checked for the AM Workbench public export on 2026-04-22.

## Project License

| Surface | Evidence | Disposition |
| --- | --- | --- |
| Vetinari package metadata | `pyproject.toml` declares `license = "MIT"`. | Root `LICENSE` matches MIT. |
| Root NOTICE state | `NOTICE` exists in the public export. | Maintained as root notice metadata. |
| Root third-party ledger state | `THIRD-PARTY-LICENSES.md` exists in the public export. | Maintained as the release-blocking ledger. |

## Python Dependency Ledger

The active virtualenv does not include `pip-licenses` or `pip-audit`, so this
table is a direct-dependency ledger from `pyproject.toml` plus installed package
metadata from `importlib.metadata`. It is not a complete transitive license
closure.

| Group | Requirement | Installed evidence | License evidence | Release disposition |
| --- | --- | --- | --- | --- |
| runtime | `requests>=2.28.0` | 2.33.1 | Apache-2.0 metadata | compatible; attribution required |
| runtime | `httpx>=0.27.0` | 0.28.1 | BSD-3-Clause metadata | compatible; attribution required |
| runtime | `urllib3>=1.26.0` | 2.6.3 | metadata license not populated | unresolved before release |
| runtime | `pyyaml>=6.0` | 6.0.3 | MIT metadata | compatible; attribution required |
| runtime | `apscheduler>=3.10.0` | 3.11.2 | MIT metadata | compatible; attribution required |
| runtime | `psutil>=5.9.0` | 7.2.2 | BSD-3-Clause metadata | compatible; attribution required |
| runtime | `pydantic>=2.0` | 2.12.5 | metadata license not populated | unresolved before release |
| runtime | `pydantic-settings>=2.0` | 2.13.1 | MIT classifier | compatible; attribution required |
| runtime | `huggingface-hub>=0.20` | 0.36.2 | Apache metadata | compatible; attribution required |
| runtime | `rich>=13.0` | 14.3.3 | MIT metadata | compatible; attribution required |
| runtime | `structlog>=24.0` | 25.5.0 | Apache/MIT classifiers | compatible; attribution required |
| runtime | `litestar>=2.12` | 2.21.1 | MIT metadata | compatible; attribution required |
| runtime | `msgspec>=0.18.0` | 0.20.0 | metadata license not populated | unresolved before release |
| runtime | `uvicorn>=0.30` | 0.43.0 | metadata license not populated | unresolved before release |
| runtime | `asgiref>=3.7` | 3.11.1 | BSD-3-Clause metadata | compatible; attribution required |
| runtime | `stamina>=24.0.0` | 25.2.0 | MIT classifier | compatible; attribution required |
| runtime | `defusedxml>=0.7.0` | 0.7.1 | Python Software Foundation License metadata | compatible; attribution required |

## Optional Dependency Risk Ledger

| Surface | Evidence | Release disposition |
| --- | --- | --- |
| Cloud providers | `pyproject.toml` optional `cloud` group plus `config/cloud_providers.yaml` names Anthropic, OpenAI, Google Gemini, Cohere, Hugging Face Inference, and Replicate endpoints. | compatible package licenses must still be verified transitively; user prompt/content transfer requires privacy disclosure and consent. |
| Local/model extras | `llama-cpp-python`, `vllm`, `diffusers`, `torch`, `transformers`, `onnxruntime`, `sentence-transformers`, `lancedb`, and related ML extras are optional. | release-blocking until transitive licenses, native binary terms, and model artifact terms are resolved. |
| Guardrails extras | `nemoguardrails` and `llm_guard` are declared but not installed in this virtualenv. | unresolved before release. |
| Training extras | `datasets`, `peft`, `trl`, `bitsandbytes`, `unsloth`, `faiss-cpu`, and related packages are optional and partly installed. | release-blocking until license, native binary, auto-install, and dataset redistribution terms are resolved. |
| Notifications extras | `desktop-notifier` is installed with unresolved metadata; `pystray` is installed with LGPLv3 metadata. | `pystray` is conditional and needs linking/distribution review before release. |
| Developer tooling | pytest, build, ruff, mypy, pyright, vulture, and peers are dev-only when excluded from release artifacts. | verify dev-only boundary and transitive obligations before publishing source bundles. |

## Browser Asset And Frontend Ledger

| Surface | Evidence | Release disposition |
| --- | --- | --- |
| Svelte browser shell | `ui/svelte/index.html` references Google Fonts and Font Awesome/cdnjs. | release-blocking if shipped without vendoring or documenting attribution, CSP, and disclosure. |
| Svelte source dependencies | `ui/svelte/package-lock.json` records Chart.js 4.5.1 MIT, highlight.js 11.11.1 BSD-3-Clause, marked 9.1.6 MIT, Svelte 5.55.1 MIT, Vite 6.4.1 MIT, and `@sveltejs/vite-plugin-svelte` 5.1.1 MIT. | dev-only while frontend is package-excluded; attribution required if shipped. |
| Generated Svelte bundles | `ui/static/svelte/js/main.js`, `vendor-chart-*`, `vendor-hljs-*`, and `vendor-marked-*` exist in the workspace. | generated dormant bundles; release-blocking if included without source, license, and rebuild provenance. |
| Frontend local dependency tree | `ui/svelte/node_modules/**` contains generated package state and native helper binaries such as esbuild variants. | remove-before-release. |

## Model, Dataset, And Tool Provenance Ledger

Public status lives in `docs/security/license-notes.md`.

| Surface | Evidence | Release disposition |
| --- | --- | --- |
| Local model artifacts | Workspace contains `ui/svelte/models/model-00001-of-00016.safetensors` around 3.99 GB. | release-blocking unless removed, excluded, or assigned exact source/license/revision/hash/redistribution terms. |
| GGUF/model download recommendations | `vetinari/cli_packaging_data.py`, `vetinari/cli_packaging_models.py`, and setup wizard paths reference Hugging Face repos, GGUF filenames, mutable `revision="main"`, and post-download hash display. | release-blocking until model card, base model license, immutable revision, expected digest, and attribution are recorded. |
| External datasets | `vetinari/training/external_data.py` and `data_seeder.py` list Hugging Face datasets including The Stack, Code Contests, MBPP, APPS, Competition Math, SmolTalk, Alpaca, OpenOrca, HH-RLHF, and UltraFeedback. | release-blocking until dataset license, source, privacy class, consent, retention, and redistribution status are recorded per dataset. |
| Tool execution surfaces | Docker/Jaeger/SearXNG helpers, MCP `npx -y`, moving `uvx` package invocations, and training auto-install flows execute or fetch third-party artifacts. | release-blocking unless pinned, allowlisted, isolated, and disclosed as dev-only or supported. |

## Privacy And Data-Flow Ledger

Public status lives in `docs/security/license-notes.md`.

| Surface | Classification | Release disposition |
| --- | --- | --- |
| Prompts, completions, project files, attachments, exports, traces, logs, memory, training data, feedback, generated authority, tool outputs, cloud provider calls, webhooks, and model/catalog identities | privacy-bearing by default | release-blocking unless redaction, access scope, retention/delete, export, and disclosure are proven for each retained flow. |
