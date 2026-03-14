"""Credentials module."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from cryptography.fernet import Fernet

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logging.warning("cryptography library not available - credentials will not be encrypted")  # noqa: VET050 — module-level import warning


@dataclass
class Credential:
    """Encrypted credential with metadata and rotation tracking."""
    source_type: str
    credential_type: str
    token: str
    scopes: list[str] = field(default_factory=list)
    rotation_days: int = 30
    last_rotated: str = ""
    next_rotation_due: str = ""
    access_controls: list[str] = field(default_factory=list)
    token_source: str = "manual"  # noqa: S105
    note: str = ""
    enabled: bool = True

    def to_dict(self) -> dict:
        """Convert to dict.

        Returns:
            Dictionary of results.
        """
        d = asdict(self)
        d.pop("token", None)
        return d

    def needs_rotation(self) -> bool:
        """Needs rotation.

        Returns:
            True if successful, False otherwise.
        """
        if not self.next_rotation_due:
            return True
        try:
            due = datetime.fromisoformat(self.next_rotation_due)
            return datetime.now() >= due
        except (ValueError, TypeError):
            return True


class CredentialVault:
    """Credential vault."""
    def __init__(self, vault_path: str | None = None):
        if vault_path is None:
            vault_path = Path.home() / ".lmstudio" / "projects" / "Vetinari" / "vault"

        self.vault_path = Path(vault_path)
        self.vault_path.mkdir(parents=True, exist_ok=True)

        self.credentials_file = self.vault_path / "credentials.enc"
        self.meta_file = self.vault_path / "credentials_meta.json"

        self._key = self._get_or_create_key()
        self._fernet = None
        if CRYPTO_AVAILABLE and self._key:
            try:
                self._fernet = Fernet(self._key)
            except Exception as e:
                logger.warning("Failed to initialize Fernet: %s", e)

        self._credentials: dict[str, Credential] = {}
        self._load()

    def _get_or_create_key(self) -> bytes:
        if not CRYPTO_AVAILABLE:
            return None

        key_file = self.vault_path / ".key"
        if key_file.exists():
            try:
                return key_file.read_bytes()
            except Exception:
                logger.debug("Failed to read encryption key file", exc_info=True)

        try:
            key = Fernet.generate_key()
            key_file.write_bytes(key)
            try:
                os.chmod(str(key_file), 0o600)
            except Exception:
                logger.debug("chmod on key file failed (expected on Windows)", exc_info=True)
            return key
        except Exception as e:
            logger.warning("Failed to generate encryption key: %s", e)
            return None

    def _load(self):
        if not self.credentials_file.exists():
            return

        try:
            with open(self.credentials_file, "rb") as f:
                encrypted = f.read()

            if not encrypted:
                return

            # Try to decrypt if fernet is available
            if self._fernet:
                try:
                    decrypted = self._fernet.decrypt(encrypted)
                    data = json.loads(decrypted)
                    for source_type, cred_data in data.items():
                        self._credentials[source_type] = Credential(**cred_data)
                    return
                except Exception as e:
                    logger.warning("Failed to decrypt credentials: %s", e)

            # Fallback: try to load as plain JSON (for migration from unencrypted store).
            # Log a warning so operators know the vault is unencrypted (P1.H9).
            try:
                data = json.loads(encrypted.decode("utf-8"))
                logger.warning(
                    "Loaded credentials from unencrypted store at %s — "
                    "re-save credentials to encrypt them with the current key.",
                    self.credentials_file,
                )
                for source_type, cred_data in data.items():
                    self._credentials[source_type] = Credential(**cred_data)
            except Exception:
                logger.debug("Failed to load credentials as plain JSON fallback", exc_info=True)
        except Exception as e:
            logger.warning("Could not load credentials: %s", e)

    def _save(self):
        data = {k: asdict(v) for k, v in self._credentials.items()}
        json_str = json.dumps(data)

        # Encrypt if fernet is available (P1.H9: never silently fall back to plaintext)
        if self._fernet:
            # Let the exception propagate — a failed save is safer than a plaintext save.
            encrypted = self._fernet.encrypt(json_str.encode())
            with open(self.credentials_file, "wb") as f:
                f.write(encrypted)
        else:
            # P1.H9: Fail closed — refuse to store credentials without encryption.
            # This prevents accidental plaintext credential storage.
            raise RuntimeError(
                "Cannot save credentials: encryption is unavailable. "
                "Install the 'cryptography' package (`pip install cryptography`) "
                "to enable the credential vault."
            )

        self._save_meta()

    def _save_meta(self):
        meta = {}
        for source_type, cred in self._credentials.items():
            meta[source_type] = cred.to_dict()

        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def get_credential(self, source_type: str) -> Credential | None:
        """Get credential.

        Returns:
            The Credential | None result.
        """
        cred = self._credentials.get(source_type)
        if cred and cred.enabled:
            return cred
        return None

    def get_token(self, source_type: str) -> str | None:
        """Get token.

        Returns:
            The result string.
        """
        cred = self.get_credential(source_type)
        return cred.token if cred else None

    def set_credential(self, source_type: str, credential: Credential):
        """Set credential.

        Args:
            source_type: The source type.
            credential: The credential.
        """
        credential.last_rotated = datetime.now().isoformat()

        due = datetime.now() + timedelta(days=credential.rotation_days)
        credential.next_rotation_due = due.isoformat()

        self._credentials[source_type] = credential
        self._save()
        logger.info("Credential set for %s", source_type)

    def remove_credential(self, source_type: str):
        """Remove credential."""
        if source_type in self._credentials:
            del self._credentials[source_type]
            self._save()

    def rotate_credential(self, source_type: str, new_token: str) -> bool:
        """Rotate credential.

        Returns:
            True if successful, False otherwise.

        Args:
            source_type: The source type.
            new_token: The new token.
        """
        if source_type not in self._credentials:
            return False

        cred = self._credentials[source_type]
        cred.token = new_token
        cred.last_rotated = datetime.now().isoformat()

        due = datetime.now() + timedelta(days=cred.rotation_days)
        cred.next_rotation_due = due.isoformat()

        self._save()
        logger.info("Credential rotated for %s", source_type)
        return True

    def list_credentials(self) -> dict[str, dict]:
        return {k: v.to_dict() for k, v in self._credentials.items()}

    def get_health(self) -> dict[str, Any]:
        """Get health.

        Returns:
            The result string.
        """
        health = {}
        for source_type, cred in self._credentials.items():
            health[source_type] = {
                "enabled": cred.enabled,
                "last_rotated": cred.last_rotated,
                "next_rotation_due": cred.next_rotation_due,
                "needs_rotation": cred.needs_rotation(),
                "credential_type": cred.credential_type,
                "note": cred.note,
            }
        return health

    def is_admin(self, user_id: str) -> bool:
        """Check if admin.

        Returns:
            True if successful, False otherwise.
        """
        admins_file = self.vault_path / "admins.json"
        if admins_file.exists():
            with open(admins_file, encoding="utf-8") as f:
                admins = json.load(f)
                return user_id in admins.get("admins", [])
        return False

    def add_admin(self, user_id: str):
        """Add admin."""
        admins_file = self.vault_path / "admins.json"
        admins = {"admins": []}
        if admins_file.exists():
            with open(admins_file, encoding="utf-8") as f:
                admins = json.load(f)

        if user_id not in admins.get("admins", []):
            admins["admins"].append(user_id)
            with open(admins_file, "w", encoding="utf-8") as f:
                json.dump(admins, f, indent=2)


class CredentialManager:
    """Credential manager."""
    def __init__(self):
        self.vault = CredentialVault()

    def get_token(self, source_type: str) -> str | None:
        return self.vault.get_token(source_type)

    def has_credential(self, source_type: str) -> bool:
        return self.vault.get_token(source_type) is not None

    def set_credential(
        self,
        source_type: str,
        token: str,
        credential_type: str = "bearer",
        scopes: list[str] | None = None,
        rotation_days: int = 30,
        note: str = "",
    ):
        """Set credential.

        Args:
            source_type: The source type.
            token: The token.
            credential_type: The credential type.
            scopes: The scopes.
            rotation_days: The rotation days.
            note: The note.
        """
        cred = Credential(
            source_type=source_type,
            credential_type=credential_type,
            token=token,
            scopes=scopes or [],
            rotation_days=rotation_days,
            note=note,
        )
        self.vault.set_credential(source_type, cred)

    def rotate(self, source_type: str, new_token: str) -> bool:
        return self.vault.rotate_credential(source_type, new_token)

    def list(self) -> dict[str, dict]:
        return self.vault.list_credentials()

    def health(self) -> dict[str, Any]:
        return self.vault.get_health()


credential_manager = CredentialManager()
