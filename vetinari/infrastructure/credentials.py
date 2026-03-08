import json
import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logging.warning("cryptography library not available - credentials will not be encrypted")

import base64
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class Credential:
    source_type: str
    credential_type: str
    token: str
    scopes: List[str] = field(default_factory=list)
    rotation_days: int = 30
    last_rotated: str = ""
    next_rotation_due: str = ""
    access_controls: List[str] = field(default_factory=list)
    token_source: str = "manual"
    note: str = ""
    enabled: bool = True
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d.pop('token', None)
        return d
    
    def needs_rotation(self) -> bool:
        if not self.next_rotation_due:
            return True
        try:
            due = datetime.fromisoformat(self.next_rotation_due)
            return datetime.now() >= due
        except Exception:
            return True


class CredentialVault:
    def __init__(self, vault_path: str = None):
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
                logger.warning(f"Failed to initialize Fernet: {e}")
        
        self._credentials: Dict[str, Credential] = {}
        self._load()
    
    @staticmethod
    def _enforce_key_permissions(key_path: Path) -> None:
        """Enforce restrictive file permissions on key files (non-Windows only)."""
        import platform
        if platform.system() != "Windows":
            try:
                os.chmod(str(key_path), 0o600)
            except OSError as e:
                logger.warning(f"Could not set permissions on {key_path}: {e}")

    def _get_or_create_key(self) -> bytes:
        if not CRYPTO_AVAILABLE:
            return None

        key_file = self.vault_path / ".key"
        if key_file.exists():
            try:
                key = key_file.read_bytes()
                # Ensure permissions are correct on every load
                self._enforce_key_permissions(key_file)
                return key
            except Exception:
                pass

        try:
            key = Fernet.generate_key()
            key_file.write_bytes(key)
            self._enforce_key_permissions(key_file)
            return key
        except Exception as e:
            logger.warning(f"Failed to generate encryption key: {e}")
            return None
    
    def _load(self):
        if not self.credentials_file.exists():
            return
        
        try:
            with open(self.credentials_file, 'rb') as f:
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
                    logger.warning(f"Failed to decrypt credentials: {e}")
            
            # Warn if credentials appear to be stored as plaintext JSON
            try:
                data = json.loads(encrypted.decode('utf-8'))
                # If we got here, credentials are in plaintext -- log warning
                # but load them for migration purposes, then re-encrypt on next save
                logger.warning(
                    "Credentials file contains unencrypted data. "
                    "They will be encrypted on the next save operation. "
                    "Install the 'cryptography' package if not already installed."
                )
                for source_type, cred_data in data.items():
                    self._credentials[source_type] = Credential(**cred_data)
            except Exception:
                logger.error("Could not decrypt or parse credentials file.")
        except Exception as e:
            logger.warning(f"Could not load credentials: {e}")
    
    def _save(self):
        data = {k: asdict(v) for k, v in self._credentials.items()}
        json_str = json.dumps(data)

        # Encrypt -- never fall back to plaintext storage
        if self._fernet:
            try:
                encrypted = self._fernet.encrypt(json_str.encode())
                with open(self.credentials_file, 'wb') as f:
                    f.write(encrypted)
            except Exception as e:
                logger.error(f"Encryption failed, refusing to store credentials in plaintext: {e}")
                raise RuntimeError(
                    "Cannot save credentials: encryption failed and plaintext "
                    "fallback is disabled for security. Install the 'cryptography' "
                    "package and ensure the encryption key is valid."
                ) from e
        else:
            logger.error(
                "Cannot save credentials: 'cryptography' package is not installed. "
                "Plaintext credential storage is not permitted."
            )
            raise RuntimeError(
                "Cannot save credentials without encryption. "
                "Install the 'cryptography' package: pip install cryptography"
            )

        self._save_meta()
    
    def _save_meta(self):
        meta = {}
        for source_type, cred in self._credentials.items():
            meta[source_type] = cred.to_dict()
        
        with open(self.meta_file, 'w') as f:
            json.dump(meta, f, indent=2)
    
    def get_credential(self, source_type: str) -> Optional[Credential]:
        cred = self._credentials.get(source_type)
        if cred and cred.enabled:
            return cred
        return None
    
    def get_token(self, source_type: str) -> Optional[str]:
        cred = self.get_credential(source_type)
        return cred.token if cred else None
    
    def set_credential(self, source_type: str, credential: Credential):
        credential.last_rotated = datetime.now().isoformat()
        
        due = datetime.now() + timedelta(days=credential.rotation_days)
        credential.next_rotation_due = due.isoformat()
        
        self._credentials[source_type] = credential
        self._save()
        logger.info(f"Credential set for {source_type}")
    
    def remove_credential(self, source_type: str):
        if source_type in self._credentials:
            del self._credentials[source_type]
            self._save()
    
    def rotate_credential(self, source_type: str, new_token: str) -> bool:
        if source_type not in self._credentials:
            return False
        
        cred = self._credentials[source_type]
        cred.token = new_token
        cred.last_rotated = datetime.now().isoformat()
        
        due = datetime.now() + timedelta(days=cred.rotation_days)
        cred.next_rotation_due = due.isoformat()
        
        self._save()
        logger.info(f"Credential rotated for {source_type}")
        return True
    
    def list_credentials(self) -> Dict[str, Dict]:
        return {k: v.to_dict() for k, v in self._credentials.items()}
    
    def get_health(self) -> Dict[str, Any]:
        health = {}
        for source_type, cred in self._credentials.items():
            health[source_type] = {
                "enabled": cred.enabled,
                "last_rotated": cred.last_rotated,
                "next_rotation_due": cred.next_rotation_due,
                "needs_rotation": cred.needs_rotation(),
                "credential_type": cred.credential_type,
                "note": cred.note
            }
        return health
    
    def rotate_encryption_key(self) -> bool:
        """Generate a new encryption key and re-encrypt all stored credentials.

        This supports key rotation: a new Fernet key is generated, all
        existing credential data is decrypted with the old key and
        re-encrypted with the new key, and the old key file is replaced.

        Returns:
            True if rotation succeeded, False otherwise.
        """
        if not CRYPTO_AVAILABLE:
            logger.error("Cannot rotate encryption key: 'cryptography' package not installed.")
            return False

        if not self._fernet:
            logger.error("Cannot rotate encryption key: no existing encryption context.")
            return False

        # Ensure all credentials are loaded into memory (already decrypted)
        if not self._credentials and self.credentials_file.exists():
            self._load()

        # Generate new key
        try:
            new_key = Fernet.generate_key()
            new_fernet = Fernet(new_key)
        except Exception as e:
            logger.error(f"Failed to generate new encryption key: {e}")
            return False

        # Re-encrypt all credential data with the new key
        data = {k: asdict(v) for k, v in self._credentials.items()}
        json_str = json.dumps(data)

        try:
            encrypted = new_fernet.encrypt(json_str.encode())
            with open(self.credentials_file, 'wb') as f:
                f.write(encrypted)
        except Exception as e:
            logger.error(f"Failed to re-encrypt credentials with new key: {e}")
            return False

        # Write the new key, replacing the old one
        key_file = self.vault_path / ".key"
        try:
            key_file.write_bytes(new_key)
            self._enforce_key_permissions(key_file)
        except Exception as e:
            logger.error(f"Failed to write new key file: {e}")
            return False

        # Update internal state
        self._key = new_key
        self._fernet = new_fernet

        logger.info("Encryption key rotated successfully. All credentials re-encrypted.")
        return True

    def is_admin(self, user_id: str) -> bool:
        admins_file = self.vault_path / "admins.json"
        if admins_file.exists():
            with open(admins_file) as f:
                admins = json.load(f)
                return user_id in admins.get("admins", [])
        return True
    
    def add_admin(self, user_id: str):
        admins_file = self.vault_path / "admins.json"
        admins = {"admins": []}
        if admins_file.exists():
            with open(admins_file) as f:
                admins = json.load(f)
        
        if user_id not in admins.get("admins", []):
            admins["admins"].append(user_id)
            with open(admins_file, 'w') as f:
                json.dump(admins, f, indent=2)


class CredentialManager:
    def __init__(self):
        self.vault = CredentialVault()
    
    def get_token(self, source_type: str) -> Optional[str]:
        return self.vault.get_token(source_type)
    
    def has_credential(self, source_type: str) -> bool:
        return self.vault.get_token(source_type) is not None
    
    def set_credential(self, source_type: str, token: str, credential_type: str = "bearer",
                      scopes: List[str] = None, rotation_days: int = 30, note: str = ""):
        cred = Credential(
            source_type=source_type,
            credential_type=credential_type,
            token=token,
            scopes=scopes or [],
            rotation_days=rotation_days,
            note=note
        )
        self.vault.set_credential(source_type, cred)
    
    def rotate(self, source_type: str, new_token: str) -> bool:
        return self.vault.rotate_credential(source_type, new_token)
    
    def list(self) -> Dict[str, Dict]:
        return self.vault.list_credentials()
    
    def health(self) -> Dict[str, Any]:
        return self.vault.get_health()

    def rotate_encryption_key(self) -> bool:
        """Rotate the vault encryption key and re-encrypt all credentials."""
        return self.vault.rotate_encryption_key()


credential_manager = CredentialManager()
