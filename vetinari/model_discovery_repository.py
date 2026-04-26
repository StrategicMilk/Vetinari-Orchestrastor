"""Hugging Face repository resolution for model discovery.

The repository mixin resolves mutable repository requests into immutable commit
and artifact metadata before downloads or UI catalog responses use them.
"""

from __future__ import annotations

import logging
from pathlib import PurePosixPath
from typing import Any

from vetinari.constants import MODEL_DISCOVERY_TIMEOUT
from vetinari.model_discovery_artifacts import (
    _MAX_REPO_FILES,
    _NATIVE_DOWNLOAD_BACKENDS,
    _extract_lfs_metadata,
    _infer_file_quantization,
    _infer_model_family,
    _matches_artifact_filters,
    _normalize_backend,
    _normalize_model_format,
    _safe_hf_filename,
    _select_native_snapshot_files,
    _validate_repo_id,
)
from vetinari.model_discovery_types import RepoModelFile, RepoModelSnapshot

logger = logging.getLogger(__name__)


class _ModelDiscoveryRepository:
    """Repository metadata behavior mixed into ModelDiscovery downloads."""

    def _resolve_repo_file(self, repo_id: str, filename: str, revision: str | None = None) -> RepoModelFile:
        """Resolve a repo file to immutable Hugging Face metadata."""
        repo_id = _validate_repo_id(repo_id)
        relative = _safe_hf_filename(filename)

        try:
            from huggingface_hub import HfApi  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is not installed") from exc

        api = HfApi()
        info = api.model_info(
            repo_id=repo_id,
            revision=revision,
            timeout=MODEL_DISCOVERY_TIMEOUT,
            files_metadata=True,
            token=False,
        )
        resolved_revision = getattr(info, "sha", None)
        if not resolved_revision:
            raise RuntimeError("Hugging Face did not return an immutable revision for the model repo")

        match = None
        for sibling in getattr(info, "siblings", []) or []:
            if getattr(sibling, "rfilename", None) == str(relative):
                match = sibling
                break
        if match is None:
            raise FileNotFoundError(f"{filename!r} was not found in Hugging Face repo {repo_id!r}")

        size, sha256 = _extract_lfs_metadata(match)
        return RepoModelFile(
            repo_id=repo_id,
            filename=str(relative),
            revision=resolved_revision,
            requested_revision=revision,
            size=size,
            sha256=sha256,
        )

    def _resolve_repo_snapshot(
        self,
        repo_id: str,
        *,
        backend: str,
        model_format: str,
        revision: str | None = None,
        objective: str | None = None,
        family: str | None = None,
        quantization: str | None = None,
        file_type: str | None = None,
        min_size_gb: float | None = None,
        max_size_gb: float | None = None,
    ) -> RepoModelSnapshot:
        """Resolve a native Hugging Face snapshot to immutable metadata."""
        repo_id = _validate_repo_id(repo_id)

        try:
            from huggingface_hub import HfApi  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is not installed") from exc

        info = HfApi().model_info(
            repo_id=repo_id,
            revision=revision,
            timeout=MODEL_DISCOVERY_TIMEOUT,
            files_metadata=True,
            token=False,
        )
        resolved_revision = getattr(info, "sha", None)
        if not resolved_revision:
            raise RuntimeError("Hugging Face did not return an immutable revision for the model repo")

        files = _select_native_snapshot_files(
            list(getattr(info, "siblings", []) or []),
            repo_id=repo_id,
            model_format=model_format,
            objective=objective,
            family=family,
            quantization=quantization,
            file_type=file_type,
            min_size_gb=min_size_gb,
            max_size_gb=max_size_gb,
        )
        return RepoModelSnapshot(
            repo_id=repo_id,
            revision=resolved_revision,
            requested_revision=revision,
            backend=backend,
            model_format=model_format,
            files=files,
        )

    def get_repo_files(
        self,
        repo_id: str,
        vram_gb: int = 32,
        use_case: str = "general",
        *,
        backend: str = "llama_cpp",
        model_format: str | None = None,
        revision: str | None = None,
        objective: str | None = None,
        family: str | None = None,
        quantization: str | None = None,
        file_type: str | None = None,
        min_size_gb: float | None = None,
        max_size_gb: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return bounded model artifact descriptors for a Hugging Face repository.

        The response resolves the repository to the current immutable commit SHA
        and marks that SHA on every returned file.  Callers must pass that
        revision back into download operations if they need exact provenance.

        Args:
            repo_id: Hugging Face repository id to inspect.
            vram_gb: Available VRAM used for GGUF scoring and filtering.
            use_case: Model-use objective used for GGUF scoring.
            backend: Target runtime backend.
            model_format: Optional model artifact format override.
            revision: Optional branch, tag, or commit SHA to inspect.
            objective: Optional native snapshot objective filter.
            family: Optional model-family filter.
            quantization: Optional quantization filter.
            file_type: Optional artifact suffix/type filter.
            min_size_gb: Optional minimum artifact size.
            max_size_gb: Optional maximum artifact size.

        Returns:
            Bounded artifact dictionaries with immutable revision provenance.

        Raises:
            ValueError: If repository, backend, format, revision, or filters are invalid.
            FileNotFoundError: If no supported artifacts match the request.
            RuntimeError: If Hugging Face metadata cannot be resolved.
        """
        repo_id = _validate_repo_id(repo_id)
        backend = _normalize_backend(backend)
        model_format = _normalize_model_format(backend, model_format)

        if backend in _NATIVE_DOWNLOAD_BACKENDS:
            snapshot = self._resolve_repo_snapshot(
                repo_id,
                backend=backend,
                model_format=model_format,
                revision=revision,
                objective=objective,
                family=family,
                quantization=quantization,
                file_type=file_type,
                min_size_gb=min_size_gb,
                max_size_gb=max_size_gb,
            )
            return [
                {
                    **file.to_dict(),
                    "repo_id": snapshot.repo_id,
                    "revision": snapshot.revision,
                    "backend": snapshot.backend,
                    "format": snapshot.model_format,
                    "artifact_type": "snapshot_file",
                    "use_case": use_case,
                    "source_type": "huggingface",
                    "family": _infer_model_family(f"{snapshot.repo_id}/{file.filename}"),
                    "quantization": _infer_file_quantization(file.filename, model_format=snapshot.model_format),
                    "file_type": PurePosixPath(file.filename).suffix.lower().lstrip("."),
                }
                for file in snapshot.files
            ]

        try:
            from huggingface_hub import HfApi  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is not installed") from exc

        info = HfApi().model_info(
            repo_id=repo_id,
            revision=revision,
            timeout=MODEL_DISCOVERY_TIMEOUT,
            files_metadata=True,
            token=False,
        )
        resolved_revision = getattr(info, "sha", None)
        if not resolved_revision:
            raise RuntimeError("Hugging Face did not return an immutable revision for the model repo")

        files: list[dict[str, Any]] = []
        max_bytes = int(vram_gb * 0.9 * 1024**3) if vram_gb > 0 else None
        for sibling in getattr(info, "siblings", []) or []:
            name = getattr(sibling, "rfilename", "")
            if not isinstance(name, str) or not name.lower().endswith(".gguf"):
                continue
            size, sha256 = _extract_lfs_metadata(sibling)
            if max_bytes is not None and size is not None and size > max_bytes:
                continue
            if not _matches_artifact_filters(
                name=f"{repo_id}/{name}",
                size=size,
                model_format="gguf",
                objective=objective,
                family=family,
                quantization=quantization,
                file_type=file_type,
                min_size_gb=min_size_gb,
                max_size_gb=max_size_gb,
            ):
                continue
            files.append({
                "filename": name,
                "repo_id": repo_id,
                "revision": resolved_revision,
                "size": size,
                "sha256": sha256,
                "backend": "llama_cpp",
                "format": "gguf",
                "artifact_type": "file",
                "use_case": use_case,
                "source_type": "huggingface",
                "family": _infer_model_family(f"{repo_id}/{name}"),
                "quantization": _infer_file_quantization(name, model_format="gguf"),
                "file_type": "gguf",
            })
            if len(files) >= _MAX_REPO_FILES:
                break
        return files
