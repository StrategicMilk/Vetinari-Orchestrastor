# Known Limitations

AM Workbench is local-first and still assumes a trusted single-operator setup.

## Current Limitations

- The dashboard and API are not a hardened multi-tenant internet service.
  Expose them beyond localhost only behind an external auth/proxy layer.
- Native `vllm` support on Windows is a Windows-to-WSL operator pattern, not a
  pure Windows-native backend.
- GGUF fallback and native Hugging Face-format model discovery require local
  model files or reachable backend endpoints. Model files are not shipped in the
  repository.
- The Svelte app is the canonical UI source, but the Litestar server does not
  yet expose a single final public UI shell contract.
- Training and fine-tuning features are optional and require heavier ML extras.
  They are not part of the minimal install path.
- Optional cloud adapters can call external providers when configured. Treat
  prompt/content transfer as privacy-bearing and operator-controlled.
- Benchmark helpers are smoke and regression checks, not formal p50/p95/p99
  performance claims.
- Release license/provenance closure is incomplete until unresolved optional
  dependencies, model sources, datasets, generated bundles, and privacy flows
  are reviewed.

## Resolved Or Bounded

- Private maintainer assets are excluded from the public export boundary.
- Python release artifacts exclude UI workspaces, model files, generated local
  state, and private working audit corpora.
- The public branch keeps the Python package name `vetinari` for compatibility
  while presenting the product as AM Workbench.
