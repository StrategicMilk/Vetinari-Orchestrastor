"""Upgrader module."""

from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)


class Upgrader:
    """Check for and install model upgrades from a remote benchmarks endpoint.

    ``check_for_upgrades`` fetches candidate models from the URL configured in
    ``config["benchmarks_source"][0]``.  When no endpoint is configured or the
    fetch fails, an empty list is returned (no hardcoded data).

    ``install_upgrade`` is a stub — actual model installation requires
    integration with LM Studio's model management API, which is not yet
    available.
    """

    def __init__(self, config: dict):
        self.config = config
        self._memory_budget_gb: int = config.get(
            "memory_budget_gb",
            config.get("discovery_filters", {}).get("max_model_memory_gb", 96),
        )

    def check_for_upgrades(self) -> list[dict[str, Any]]:
        """Fetch upgrade candidates from the configured benchmarks source.

        Returns a (possibly empty) list of candidate dicts, each containing at
        least ``name``, ``version``, and ``memory_gb`` keys.  Candidates whose
        ``memory_gb`` exceeds the configured budget are filtered out.

        Returns:
            The result string.
        """
        candidates: list[dict[str, Any]] = []

        # Resolve the benchmarks URL from config
        source = self.config.get("benchmarks_source")
        if isinstance(source, list) and source:
            url = source[0]
        elif isinstance(source, str) and source:
            url = source
        else:
            logger.debug("No benchmarks_source configured — skipping upgrade check")
            return candidates

        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, dict):
                data = data.get("models", data.get("data", []))
            if not isinstance(data, list):
                logger.warning("Unexpected benchmarks response format from %s", url)
                return candidates

            for m in data:
                if not isinstance(m, dict):
                    continue
                mem = m.get("memory_gb", 0)
                if mem <= self._memory_budget_gb:
                    candidates.append(m)
                else:
                    logger.debug(
                        "Skipping upgrade candidate %s — exceeds memory budget (%sGB > %sGB)",
                        m.get("name", "?"),
                        mem,
                        self._memory_budget_gb,
                    )
        except requests.exceptions.RequestException as e:
            logger.warning("Upgrade check failed (network): %s", e)
        except (ValueError, KeyError) as e:
            logger.warning("Upgrade check failed (parse): %s", e)

        logger.info("Upgrade check complete: %d candidate(s)", len(candidates))
        return candidates

    def install_upgrade(self, candidate: dict) -> bool:
        """Install a model upgrade via LM Studio's local API.

        Attempts to download/install the model through LM Studio's
        ``/api/v0/models/download`` endpoint (available in LM Studio ≥0.3).
        Falls back to logging the Hugging Face download URL if the API
        is unreachable or the endpoint is not supported.

        Returns ``True`` if the download was accepted or the fallback URL
        was logged; ``False`` on unrecoverable errors.

        Returns:
            True if successful, False otherwise.
        """
        name = candidate.get("name", "unknown")
        version = candidate.get("version", "?")
        hf_repo = candidate.get("huggingface_repo", "")
        filename = candidate.get("filename", "")

        # Build the model identifier LM Studio expects
        model_id = hf_repo or name
        if not model_id:
            logger.error("Cannot install upgrade: no model identifier provided")
            return False

        # ── Attempt 1: LM Studio local model management API ──
        lmstudio_host = self.config.get("lmstudio_host", "http://localhost:1234")  # noqa: VET041
        api_url = f"{lmstudio_host}/api/v0/models/download"

        payload = {"model": model_id}
        if filename:
            payload["filename"] = filename

        try:
            resp = requests.post(api_url, json=payload, timeout=15)
            if resp.status_code in (200, 202):
                logger.info(
                    "LM Studio accepted download for %s v%s (model_id=%s)",
                    name,
                    version,
                    model_id,
                )
                return True
            elif resp.status_code == 404:
                # Endpoint not supported in this LM Studio version
                logger.info("LM Studio /api/v0/models/download not available (404) — falling back to URL logging")
            else:
                logger.warning(
                    "LM Studio download API returned %d: %s",
                    resp.status_code,
                    resp.text[:200],
                )
        except requests.exceptions.ConnectionError:
            logger.info(
                "LM Studio not reachable at %s — falling back to URL logging",
                lmstudio_host,
            )
        except requests.exceptions.RequestException as exc:
            logger.warning("LM Studio API request failed: %s", exc)

        # ── Attempt 2: Log the direct Hugging Face download URL ──
        if hf_repo:
            hf_url = f"https://huggingface.co/{hf_repo}"
            if filename:
                hf_url = f"https://huggingface.co/{hf_repo}/resolve/main/{filename}"
            logger.info(
                "Manual download available for %s v%s: %s",
                name,
                version,
                hf_url,
            )
            return True

        logger.warning(
            "Install requested for %s v%s but no HF repo or LM Studio API available",
            name,
            version,
        )
        return False
