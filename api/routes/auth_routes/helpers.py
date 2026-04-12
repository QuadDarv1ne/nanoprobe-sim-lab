"""
Auth helpers

Утилиты, константы, хэширование паролей, валидация.
"""

import hashlib
import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from passlib.context import CryptContext

from api.security.jwt_config import get_default_passwords, get_jwt_secret

logger = logging.getLogger(__name__)

# Audit logger для security событий
audit_logger = logging.getLogger("audit.security")

# JWT конфигурация
JWT_SECRET = get_jwt_secret()
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_MINUTES = int(os.getenv("JWT_EXPIRATION_MINUTES", "60"))
JWT_REFRESH_EXPIRATION_DAYS = int(os.getenv("JWT_REFRESH_EXPIRATION_DAYS", "7"))

# Argon2 + bcrypt fallback
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _get_or_create_hash(username: str, password: str, cache: Dict[str, str]) -> str:
    """
    Получить или создать хэш пароля с кэшированием.

    Args:
        username: Имя пользователя
        password: Пароль в открытом виде
        cache: Словарь кэша хэшей

    Returns:
        Хэш пароля
    """
    cache_key = f"{username}:{hashlib.sha256(password.encode()).hexdigest()[:16]}"
    if cache_key in cache:
        return cache[cache_key]
    new_hash = pwd_context.hash(password)
    cache[cache_key] = new_hash
    return new_hash


class AuditEventType(str, Enum):
    """Типы audit событий."""

    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    TOKEN_REFRESH = "token_refresh"
    TWO_FA_SETUP = "2fa_setup"
    TWO_FA_VERIFY = "2fa_verify"
    PASSWORD_CHANGE = "password_change"


def log_audit_event(
    event_type: AuditEventType,
    username: str,
    request=None,
    details: Optional[Dict] = None,
    **extra,
):
    """Записать audit событие."""
    event_data = {
        "event_type": event_type.value,
        "username": username,
        "details": details or {},
        **extra,
    }
    audit_logger.info(f"Audit: {json.dumps(event_data, ensure_ascii=False)}")


def validate_password_strength(password: str) -> tuple[bool, str]:
    """Проверить силу пароля."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if len(password) > 128:
        return False, "Password must not exceed 128 characters"
    if not any(c.isupper() for c in password):
        return False, "Password must contain an uppercase letter"
    if not any(c.isdigit() for c in password):
        return False, "Password must contain a digit"
    if not any(c.islower() for c in password):
        return False, "Password must contain a lowercase letter"
    if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
        return False, "Password must contain a special character"
    return True, ""


def hash_password(password: str) -> str:
    """Хэшировать пароль с Argon2."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Проверить пароль."""
    return pwd_context.verify(plain_password, hashed_password)


def _initialize_users_db():
    """Инициализирует базу пользователей."""
    from api.security.jwt_config import get_default_passwords

    default_passwords = get_default_passwords()
    hash_cache_file = Path("data/.password_hashes.json")

    cached_hashes = {}
    if hash_cache_file.exists():
        try:
            cached_hashes = json.loads(hash_cache_file.read_text())
        except Exception as e:
            logger.warning(f"Failed to load password hash cache: {e}")
            cached_hashes = {}

    admin_hash = _get_or_create_hash("admin", default_passwords["admin"], cached_hashes)
    user_hash = _get_or_create_hash("user", default_passwords["user"], cached_hashes)

    try:
        hash_cache_file.parent.mkdir(parents=True, exist_ok=True)
        hash_cache_file.write_text(json.dumps(cached_hashes))
        hash_cache_file.chmod(0o600)
    except Exception as e:
        logger.warning(f"Could not save password hash cache: {e}")

    try:
        from utils.database import get_database

        db = get_database()
        db.upsert_user("admin", admin_hash, "admin")
        db.upsert_user("user", user_hash, "user")
    except Exception as e:
        logger.warning(f"Could not initialize users in SQLite: {e}")

    return {"admin": admin_hash, "user": user_hash}


def _get_users_db() -> Dict[str, dict]:
    """Получить базу пользователей."""
    default_passwords = get_default_passwords()
    hash_cache_file = Path("data/.password_hashes.json")

    cached_hashes = {}
    if hash_cache_file.exists():
        try:
            cached_hashes = json.loads(hash_cache_file.read_text())
        except Exception as e:
            logger.debug(f"Failed to load password hash cache: {e}")
            cached_hashes = {}

    admin_hash = _get_or_create_hash("admin", default_passwords["admin"], cached_hashes)
    user_hash = _get_or_create_hash("user", default_passwords["user"], cached_hashes)

    return {
        "admin": {
            "id": 1,
            "username": "admin",
            "password_hash": admin_hash,
            "role": "admin",
            "created_at": "2026-03-11T00:00:00",
            "last_login": None,
        },
        "user": {
            "id": 2,
            "username": "user",
            "password_hash": user_hash,
            "role": "user",
            "created_at": "2026-03-11T00:00:00",
            "last_login": None,
        },
    }
