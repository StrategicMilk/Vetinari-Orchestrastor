"""
Contract Registry — vetinari.drift.contract_registry  (Phase 7)

Tracks a content-hash fingerprint for every versioned contract (dataclass,
schema dict, or plain dict).  When a fingerprint changes between runs the
registry raises ``ContractDriftError``, providing a precise diff for
engineers to review.

Usage
-----
    from vetinari.drift.contract_registry import get_contract_registry

    reg = get_contract_registry()
    reg.register("Plan", {"plan_id": "...", "goal": "...", "version": "v0.1.0"})
    reg.snapshot()          # persist current hashes to disk

    # Later — detect drift
    changed = reg.check_drift()
    for name, info in changed.items():
        print(f"{name}: was {info['previous']}, now {info['current']}")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_SNAPSHOT_PATH = ".vetinari/drift_snapshots/contracts.json"


class ContractDriftError(Exception):
    """Raised when a registered contract fingerprint changes unexpectedly."""


def _fingerprint(obj: Any) -> str:
    """Return a stable SHA-256 hex digest of an object."""
    if is_dataclass(obj) and not isinstance(obj, type):
        data = asdict(obj)
    elif hasattr(obj, "to_dict"):
        data = obj.to_dict()
    elif isinstance(obj, dict):
        data = obj
    else:
        data = {"__repr__": repr(obj)}
    canonical = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


class ContractRegistry:
    """
    Thread-safe contract fingerprint registry.  Singleton — use
    ``get_contract_registry()``.
    """

    _instance: Optional["ContractRegistry"] = None
    _class_lock = threading.Lock()

    def __new__(cls) -> "ContractRegistry":
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._setup()
        return cls._instance

    def _setup(self) -> None:
        self._lock = threading.RLock()
        self._current:  Dict[str, str] = {}   # name → hash
        self._previous: Dict[str, str] = {}   # name → hash (last snapshot)
        self._snapshot_path = Path(_DEFAULT_SNAPSHOT_PATH)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, contract: Any) -> str:
        """
        Register and fingerprint a contract object.

        Args:
            name:     Unique contract name (e.g. ``"Plan"``, ``"AgentTask"``).
            contract: Dataclass instance, dict, or any object with ``to_dict()``.

        Returns:
            The SHA-256 hex digest of the contract.
        """
        h = _fingerprint(contract)
        with self._lock:
            self._current[name] = h
        logger.debug("Registered contract '%s' → %s", name, h[:12])
        return h

    def register_many(self, contracts: Dict[str, Any]) -> Dict[str, str]:
        """Register multiple contracts at once.  Returns name → hash map."""
        return {name: self.register(name, obj) for name, obj in contracts.items()}

    # ------------------------------------------------------------------
    # Snapshot (persistence)
    # ------------------------------------------------------------------

    def snapshot(self, path: Optional[str] = None) -> None:
        """Persist current fingerprints to disk."""
        p = Path(path) if path else self._snapshot_path
        p.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {
                "timestamp": time.time(),
                "hashes": dict(self._current),
            }
        p.write_text(json.dumps(payload, indent=2))
        logger.info("Contract snapshot written to %s (%d entries)",
                    p, len(payload["hashes"]))

    def load_snapshot(self, path: Optional[str] = None) -> bool:
        """Load a previously saved snapshot into ``_previous``."""
        p = Path(path) if path else self._snapshot_path
        if not p.exists():
            logger.debug("No snapshot at %s; starting fresh.", p)
            return False
        try:
            payload = json.loads(p.read_text())
            with self._lock:
                self._previous = payload.get("hashes", {})
            logger.info("Loaded snapshot from %s (%d entries)",
                        p, len(self._previous))
            return True
        except Exception as exc:
            logger.error("Failed to load snapshot: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def check_drift(
        self,
        raise_on_drift: bool = False,
    ) -> Dict[str, Dict[str, str]]:
        """
        Compare current fingerprints against the loaded snapshot.

        Returns a dict of drifted contracts:
            ``{ name: {"previous": "...", "current": "..."} }``

        Args:
            raise_on_drift: If True, raises ``ContractDriftError`` when drift
                            is found.
        """
        with self._lock:
            current  = dict(self._current)
            previous = dict(self._previous)

        drifted: Dict[str, Dict[str, str]] = {}

        # Changed or new contracts
        for name, h in current.items():
            prev_h = previous.get(name)
            if prev_h is not None and prev_h != h:
                drifted[name] = {"previous": prev_h, "current": h}
                logger.warning("Contract drift: '%s' hash changed", name)

        # Removed contracts
        for name in previous:
            if name not in current:
                drifted[name] = {"previous": previous[name], "current": "REMOVED"}
                logger.warning("Contract drift: '%s' was removed", name)

        if drifted and raise_on_drift:
            names = ", ".join(drifted)
            raise ContractDriftError(
                f"Contract drift detected in: {names}"
            )
        return drifted

    def is_stable(self) -> bool:
        """Return True when no drift is detected."""
        return len(self.check_drift()) == 0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_contracts(self) -> List[str]:
        with self._lock:
            return sorted(self._current.keys())

    def get_hash(self, name: str) -> Optional[str]:
        with self._lock:
            return self._current.get(name)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "registered": len(self._current),
                "snapshotted": len(self._previous),
                "drifted": len(self.check_drift()),
            }

    def clear(self) -> None:
        with self._lock:
            self._current.clear()
            self._previous.clear()


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------

def get_contract_registry() -> ContractRegistry:
    """Return the global ContractRegistry singleton."""
    return ContractRegistry()


def reset_contract_registry() -> None:
    """Destroy the singleton (for tests)."""
    with ContractRegistry._class_lock:
        ContractRegistry._instance = None
