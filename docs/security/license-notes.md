# License And Provenance Notes

This page is the public summary for license, asset, model, and data-flow status.
It does not replace a legal review.

## Project License

AM Workbench, including the Vetinari engine, is released under the MIT License.
See `LICENSE`, `NOTICE`, and `THIRD-PARTY-LICENSES.md`.

## Dependency Status

Runtime Python dependencies are listed in `pyproject.toml` and summarized in
`THIRD-PARTY-LICENSES.md`. Some transitive licenses and optional dependency
licenses remain unresolved before any release can claim complete license
closure.

## Frontend Assets

`ui/svelte` is source. `ui/static/svelte` is generated build output, including
bundles such as `ui/static/svelte/js/main.js`.

Generated bundles must not be treated as final public web assets until source,
license, attribution, CSP, and rebuild provenance are reviewed.

## Model Artifacts

Model weights, GGUF files, safetensors files, ONNX files, PyTorch checkpoints,
and similar assets are not repository content. Operators are responsible for
selecting, downloading, licensing, and placing model files in local model
directories.

## Datasets And Training

Training extras may interact with external datasets and model hubs. Dataset
license, privacy, consent, redistribution, and retention terms must be reviewed
before claiming a training workflow is release-ready.

## Privacy-Bearing Data

Prompts, completions, project files, uploaded attachments, logs, traces,
memory, feedback, training records, provider calls, webhook payloads, and model
catalog identities can all be privacy-bearing. Operators should configure
retention, redaction, and provider usage accordingly.
