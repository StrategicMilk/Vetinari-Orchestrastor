"""Managed model download lifecycle for model discovery.

The download mixin tracks foreground and background downloads, validates cache
hits, and persists bounded status dictionaries for the public facade.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any

from vetinari.constants import DEFAULT_NATIVE_MODELS_DIR, MODEL_DISCOVERY_TIMEOUT, OPERATOR_MODELS_CACHE_DIR
from vetinari.model_discovery_artifacts import (
    _NATIVE_DOWNLOAD_BACKENDS,
    _ensure_free_space,
    _materialized_snapshot_files,
    _normalize_backend,
    _normalize_model_format,
    _resolve_destination,
    _resolve_snapshot_destination,
    _sha256_file,
    _snapshot_marker_path,
    _validate_existing_download,
    _validate_existing_snapshot,
    _validate_model_header,
    _write_download_marker,
    _write_snapshot_marker,
)
from vetinari.model_discovery_cache import _load_download_state, _public_download_state, _write_download_state
from vetinari.model_discovery_repository import _ModelDiscoveryRepository
from vetinari.model_discovery_types import RepoModelFile, RepoModelSnapshot

logger = logging.getLogger(__name__)

_DOWNLOAD_LOCK = threading.Lock()
# Download jobs are written by ModelDiscovery.start_download worker threads,
# read by get_download_status(), and protected by _DOWNLOAD_LOCK.
_DOWNLOAD_JOBS: dict[str, dict[str, Any]] = {}
# Cancellation events are written by start_download(), read by worker threads,
# and protected by _DOWNLOAD_LOCK for the lifetime of a tracked download.
_DOWNLOAD_CANCEL_EVENTS: dict[str, Event] = {}
_DOWNLOAD_STATE_FILENAME = "download_jobs.json"  # Per-cache persisted background-download state.
_ACTIVE_DOWNLOAD_STATES = {"started", "running", "canceling"}  # States converted to interrupted on restart.


class _ModelDiscoveryDownloads(_ModelDiscoveryRepository):
    """Download behavior mixed into the public ModelDiscovery facade."""

    def _persist_job_locked(self, download_id: str) -> None:
        state = _DOWNLOAD_JOBS.get(download_id)
        if not state:
            return
        state_path = Path(str(state.get("_state_path") or self.download_state_path))
        persisted = _load_download_state(state_path)
        persisted[download_id] = _public_download_state(state)
        _write_download_state(state_path, persisted)

    def _persist_external_state(self, state: dict[str, Any]) -> None:
        download_id = str(state.get("download_id") or "")
        if not download_id:
            return
        state_path = Path(str(state.get("_state_path") or self.download_state_path))
        persisted = _load_download_state(state_path)
        persisted[download_id] = _public_download_state(state)
        _write_download_state(state_path, persisted)

    def _load_persisted_download_status(self, download_id: str) -> dict[str, Any] | None:
        persisted = _load_download_state(self.download_state_path)
        state = persisted.get(download_id)
        if state is None:
            return None
        if state.get("status") in _ACTIVE_DOWNLOAD_STATES:
            state = dict(state)
            state["status"] = "interrupted"
            state["error"] = state.get("error") or "download process exited before completion"
            state["completed_at"] = state.get("completed_at") or datetime.now(timezone.utc).isoformat()
            persisted[download_id] = state
            _write_download_state(self.download_state_path, persisted)
        return dict(state)

    def _complete_download(
        self,
        repo_file: RepoModelFile,
        destination: Path,
        cancel_event: Event | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """Download, verify, and atomically publish one model artifact."""
        try:
            from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is not installed") from exc

        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("download canceled before start")

        destination.parent.mkdir(parents=True, exist_ok=True)
        _ensure_free_space(destination.parent, repo_file.size)

        if destination.exists():
            digest = _validate_existing_download(destination, repo_file)
            return {
                "status": "completed",
                "repo_id": repo_file.repo_id,
                "filename": repo_file.filename,
                "revision": repo_file.revision,
                "backend": "llama_cpp",
                "format": "gguf",
                "artifact_type": "file",
                "path": str(destination),
                "sha256": digest,
                "bytes_downloaded": destination.stat().st_size,
                "download_id": job_id,
            }

        with tempfile.TemporaryDirectory(prefix="vetinari_model_download_") as temp_root:
            local_path = Path(
                hf_hub_download(
                    repo_id=repo_file.repo_id,
                    filename=repo_file.filename,
                    local_dir=temp_root,
                    revision=repo_file.revision,
                    resume_download=False,
                    local_dir_use_symlinks=False,
                    token=False,
                    etag_timeout=min(10, MODEL_DISCOVERY_TIMEOUT),
                )
            )

            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("download canceled before completion")
            if not local_path.exists():
                raise FileNotFoundError(f"download backend did not materialize {repo_file.filename!r}")

            _validate_model_header(local_path)
            digest = _sha256_file(local_path)
            if repo_file.sha256 and digest.lower() != repo_file.sha256.lower():
                raise ValueError(
                    f"downloaded model digest mismatch for {repo_file.filename}: "
                    f"expected {repo_file.sha256.lower()}, got {digest.lower()}"
                )

            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(local_path, destination)
            _write_download_marker(destination, repo_file, digest)

        return {
            "status": "completed",
            "repo_id": repo_file.repo_id,
            "filename": repo_file.filename,
            "revision": repo_file.revision,
            "backend": "llama_cpp",
            "format": "gguf",
            "artifact_type": "file",
            "path": str(destination),
            "sha256": digest,
            "bytes_downloaded": destination.stat().st_size,
            "download_id": job_id,
        }

    def _complete_snapshot_download(
        self,
        snapshot: RepoModelSnapshot,
        destination: Path,
        cancel_event: Event | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """Download, verify, and publish one native HF snapshot directory."""
        try:
            from huggingface_hub import snapshot_download  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is not installed") from exc

        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("download canceled before start")

        destination.parent.mkdir(parents=True, exist_ok=True)
        _ensure_free_space(destination.parent, snapshot.total_size)

        if destination.exists():
            files = _validate_existing_snapshot(destination, snapshot)
            return {
                "status": "completed",
                "repo_id": snapshot.repo_id,
                "revision": snapshot.revision,
                "backend": snapshot.backend,
                "format": snapshot.model_format,
                "artifact_type": "snapshot",
                "path": str(destination),
                "manifest_path": str(_snapshot_marker_path(destination)),
                "files": files,
                "bytes_total": snapshot.total_size,
                "bytes_downloaded": sum(int(file.get("size") or 0) for file in files),
                "download_id": job_id,
            }

        with tempfile.TemporaryDirectory(prefix="vetinari_native_model_download_") as temp_root:
            temp_destination = Path(temp_root) / "snapshot"
            snapshot_download(
                repo_id=snapshot.repo_id,
                revision=snapshot.revision,
                local_dir=str(temp_destination),
                local_dir_use_symlinks=False,
                allow_patterns=[file.filename for file in snapshot.files],
                token=False,
                etag_timeout=min(10, MODEL_DISCOVERY_TIMEOUT),
            )

            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("download canceled before completion")
            files = _materialized_snapshot_files(temp_destination, snapshot)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temp_destination), str(destination))
            _write_snapshot_marker(destination, snapshot, files)

        return {
            "status": "completed",
            "repo_id": snapshot.repo_id,
            "revision": snapshot.revision,
            "backend": snapshot.backend,
            "format": snapshot.model_format,
            "artifact_type": "snapshot",
            "path": str(destination),
            "manifest_path": str(_snapshot_marker_path(destination)),
            "files": files,
            "bytes_total": snapshot.total_size,
            "bytes_downloaded": sum(int(file.get("size") or 0) for file in files),
            "download_id": job_id,
        }

    def download_model(
        self,
        repo_id: str,
        filename: str | None = None,
        models_dir: str | Path | None = None,
        revision: str | None = None,
        *,
        backend: str = "llama_cpp",
        model_format: str | None = None,
    ) -> dict[str, Any]:
        """Synchronously download a model with integrity and provenance checks.

        Args:
            repo_id: Hugging Face repository id to download from.
            filename: GGUF filename to download; unused for native snapshot backends.
            models_dir: Optional destination root.
            revision: Optional branch, tag, or commit SHA to pin.
            backend: Target runtime backend.
            model_format: Optional model artifact format override.

        Returns:
            Download metadata including provenance, destination path, and digest.

        Raises:
            ValueError: If required identifiers or format values are invalid.
            FileNotFoundError: If the requested artifact cannot be found.
            RuntimeError: If the download fails integrity or provenance checks.
            OSError: If the destination cannot be created or written.
        """
        backend = _normalize_backend(backend)
        model_format = _normalize_model_format(backend, model_format)
        if backend in _NATIVE_DOWNLOAD_BACKENDS:
            root = Path(models_dir or DEFAULT_NATIVE_MODELS_DIR)
            root.mkdir(parents=True, exist_ok=True)
            snapshot = self._resolve_repo_snapshot(
                repo_id,
                backend=backend,
                model_format=model_format,
                revision=revision,
            )
            destination = _resolve_snapshot_destination(root, snapshot)
            return self._complete_snapshot_download(snapshot, destination)

        if not filename:
            raise ValueError("filename is required for GGUF downloads")
        root = Path(models_dir or OPERATOR_MODELS_CACHE_DIR)
        root.mkdir(parents=True, exist_ok=True)
        repo_file = self._resolve_repo_file(repo_id, filename, revision=revision)
        destination = _resolve_destination(root, repo_file.filename)
        return self._complete_download(repo_file, destination)

    def start_download(
        self,
        repo_id: str,
        filename: str | None = None,
        models_dir: str | Path | None = None,
        revision: str | None = None,
        *,
        backend: str = "llama_cpp",
        model_format: str | None = None,
    ) -> dict[str, Any]:
        """Start a tracked background model download or return a completed hit.

        Args:
            repo_id: Hugging Face repository id to download from.
            filename: GGUF filename to download; unused for native snapshot backends.
            models_dir: Optional destination root.
            revision: Optional branch, tag, or commit SHA to pin.
            backend: Target runtime backend.
            model_format: Optional model artifact format override.

        Returns:
            Public download-state dictionary for the started or cached download.

        Raises:
            ValueError: If required identifiers or format values are invalid.
            FileNotFoundError: If the requested artifact cannot be found.
            RuntimeError: If the download state cannot be initialized.
            OSError: If the destination cannot be created or written.
        """
        backend = _normalize_backend(backend)
        model_format = _normalize_model_format(backend, model_format)
        if backend in _NATIVE_DOWNLOAD_BACKENDS:
            return self._start_snapshot_download(
                repo_id,
                models_dir=models_dir,
                revision=revision,
                backend=backend,
                model_format=model_format,
            )

        if not filename:
            raise ValueError("filename is required for GGUF downloads")
        root = Path(models_dir or OPERATOR_MODELS_CACHE_DIR)
        root.mkdir(parents=True, exist_ok=True)
        repo_file = self._resolve_repo_file(repo_id, filename, revision=revision)
        destination = _resolve_destination(root, repo_file.filename)

        if destination.exists():
            return self._complete_download(repo_file, destination)

        destination.parent.mkdir(parents=True, exist_ok=True)
        _ensure_free_space(destination.parent, repo_file.size)
        job_id = uuid.uuid4().hex
        cancel_event = Event()
        state: dict[str, Any] = {
            "download_id": job_id,
            "status": "started",
            "repo_id": repo_file.repo_id,
            "filename": repo_file.filename,
            "revision": repo_file.revision,
            "backend": "llama_cpp",
            "format": "gguf",
            "artifact_type": "file",
            "path": str(destination),
            "bytes_total": repo_file.size,
            "bytes_downloaded": 0,
            "error": None,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "_state_path": str(self.download_state_path),
        }

        with _DOWNLOAD_LOCK:
            _DOWNLOAD_JOBS[job_id] = dict(state)
            _DOWNLOAD_CANCEL_EVENTS[job_id] = cancel_event
            self._persist_job_locked(job_id)

        def _worker() -> None:
            with _DOWNLOAD_LOCK:
                _DOWNLOAD_JOBS[job_id]["status"] = "running"
                self._persist_job_locked(job_id)
            try:
                result = self._complete_download(repo_file, destination, cancel_event=cancel_event, job_id=job_id)
                with _DOWNLOAD_LOCK:
                    _DOWNLOAD_JOBS[job_id].update(result)
                    _DOWNLOAD_JOBS[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
                    self._persist_job_locked(job_id)
            except Exception as exc:
                status = "canceled" if cancel_event.is_set() else "failed"
                with _DOWNLOAD_LOCK:
                    _DOWNLOAD_JOBS[job_id].update({
                        "status": status,
                        "error": str(exc),
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    })
                    self._persist_job_locked(job_id)
            finally:
                with _DOWNLOAD_LOCK:
                    _DOWNLOAD_CANCEL_EVENTS.pop(job_id, None)

        threading.Thread(target=_worker, name=f"model-download-{job_id[:8]}", daemon=True).start()
        return _public_download_state(state)

    def _start_snapshot_download(
        self,
        repo_id: str,
        *,
        models_dir: str | Path | None = None,
        revision: str | None = None,
        backend: str,
        model_format: str,
    ) -> dict[str, Any]:
        root = Path(models_dir or DEFAULT_NATIVE_MODELS_DIR)
        root.mkdir(parents=True, exist_ok=True)
        snapshot = self._resolve_repo_snapshot(repo_id, backend=backend, model_format=model_format, revision=revision)
        destination = _resolve_snapshot_destination(root, snapshot)

        if destination.exists():
            return self._complete_snapshot_download(snapshot, destination)

        destination.parent.mkdir(parents=True, exist_ok=True)
        _ensure_free_space(destination.parent, snapshot.total_size)
        job_id = uuid.uuid4().hex
        cancel_event = Event()
        state: dict[str, Any] = {
            "download_id": job_id,
            "status": "started",
            "repo_id": snapshot.repo_id,
            "revision": snapshot.revision,
            "backend": snapshot.backend,
            "format": snapshot.model_format,
            "artifact_type": "snapshot",
            "path": str(destination),
            "manifest_path": str(_snapshot_marker_path(destination)),
            "bytes_total": snapshot.total_size,
            "bytes_downloaded": 0,
            "file_count": len(snapshot.files),
            "error": None,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "_state_path": str(self.download_state_path),
        }

        with _DOWNLOAD_LOCK:
            _DOWNLOAD_JOBS[job_id] = dict(state)
            _DOWNLOAD_CANCEL_EVENTS[job_id] = cancel_event
            self._persist_job_locked(job_id)

        def _worker() -> None:
            with _DOWNLOAD_LOCK:
                _DOWNLOAD_JOBS[job_id]["status"] = "running"
                self._persist_job_locked(job_id)
            try:
                result = self._complete_snapshot_download(
                    snapshot,
                    destination,
                    cancel_event=cancel_event,
                    job_id=job_id,
                )
                with _DOWNLOAD_LOCK:
                    _DOWNLOAD_JOBS[job_id].update(result)
                    _DOWNLOAD_JOBS[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
                    self._persist_job_locked(job_id)
            except Exception as exc:
                status = "canceled" if cancel_event.is_set() else "failed"
                with _DOWNLOAD_LOCK:
                    _DOWNLOAD_JOBS[job_id].update({
                        "status": status,
                        "error": str(exc),
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    })
                    self._persist_job_locked(job_id)
            finally:
                with _DOWNLOAD_LOCK:
                    _DOWNLOAD_CANCEL_EVENTS.pop(job_id, None)

        threading.Thread(target=_worker, name=f"native-model-download-{job_id[:8]}", daemon=True).start()
        return _public_download_state(state)

    def get_download_status(self, download_id: str) -> dict[str, Any] | None:
        """Return a bounded status object for a tracked download.

        Returns:
            Public download-state dictionary, or None when the id is unknown.
        """
        with _DOWNLOAD_LOCK:
            status = _DOWNLOAD_JOBS.get(download_id)
            if status:
                return _public_download_state(status)
        return self._load_persisted_download_status(download_id)

    def cancel_download(self, download_id: str) -> bool:
        """Request cancellation of a running tracked download.

        Returns:
            True when a running download accepted cancellation.
        """
        with _DOWNLOAD_LOCK:
            event = _DOWNLOAD_CANCEL_EVENTS.get(download_id)
            if event is None:
                return False
            event.set()
            if download_id in _DOWNLOAD_JOBS:
                _DOWNLOAD_JOBS[download_id]["status"] = "canceling"
                self._persist_job_locked(download_id)
            return True
