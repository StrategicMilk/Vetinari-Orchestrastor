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
    
    def _get_or_create_key(self) -> bytes:
        if not CRYPTO_AVAILABLE:
            return None
            
        key_file = self.vault_path / ".key"
        if key_file.exists():
            try:
                return key_file.read_bytes()
            except Exception:
                pass
        
        try:
            key = Fernet.generate_key()
            key_file.write_bytes(key)
            try:
                os.chmod(str(key_file), 0o600)
            except Exception:
                pass  # May fail on Windows
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
            
            # Fallback: try to load as plain JSON (for migration)
            try:
                data = json.loads(encrypted.decode('utf-8'))
                for source_type, cred_data in data.items():
                    self._credentials[source_type] = Credential(**cred_data)
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Could not load credentials: {e}")
    
    def _save(self):
        data = {k: asdict(v) for k, v in self._credentials.items()}
        json_str = json.dumps(data)
        
        # Encrypt if fernet is available
        if self._fernet:
            try:
                encrypted = self._fernet.encrypt(json_str.encode())
                with open(self.credentials_file, 'wb') as f:
                    f.write(encrypted)
            except Exception as e:
                logger.warning(f"Encryption failed, falling back to plain JSON: {e}")
                # Fallback to plain JSON
                with open(self.credentials_file, 'w') as f:
                    f.write(json_str)
        else:
            # Save as plain JSON if encryption not available
            with open(self.credentials_file, 'w') as f:
                f.write(json_str)
        
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


credential_manager = CredentialManager()
