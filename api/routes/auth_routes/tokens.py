"""
Auth tokens

JWT токены, refresh token rotation, Redis storage.
"""

import logging
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import jwt

from api.routes.auth_routes.helpers import (
    JWT_ALGORITHM,
    JWT_EXPIRATION_MINUTES,
    JWT_REFRESH_EXPIRATION_DAYS,
    JWT_SECRET,
)

logger = logging.getLogger(__name__)

# In-memory хранилище (fallback если Redis недоступен)
_in_memory_tokens: Dict[str, float] = {}
_in_memory_tokens_lock = __import__("threading").Lock()
_IN_MEMORY_TOKENS_MAX_AGE = JWT_REFRESH_EXPIRATION_DAYS * 86400
_IN_MEMORY_TOKENS_MAX_SIZE = 10000


def _get_redis_client():
    """Получение Redis клиента."""
    try:
        from api.security.jwt_config import get_redis_client

        return get_redis_client()
    except Exception as e:
        logger.warning(f"Redis not available, using in-memory storage: {e}")
        return None


def _cleanup_expired_tokens():
    """Очистка протухших токенов."""
    now = time.time()
    expired = [jti for jti, ts in _in_memory_tokens.items() if now - ts > _IN_MEMORY_TOKENS_MAX_AGE]
    for jti in expired:
        _in_memory_tokens.pop(jti, None)

    if len(_in_memory_tokens) > _IN_MEMORY_TOKENS_MAX_SIZE:
        sorted_tokens = sorted(_in_memory_tokens.items(), key=lambda x: x[1])
        for jti, _ in sorted_tokens[: len(sorted_tokens) // 2]:
            _in_memory_tokens.pop(jti, None)


def _store_refresh_token(jti: str, username: str):
    """Сохранение refresh токена в Redis."""
    redis_client = _get_redis_client()
    if redis_client:
        try:
            key = f"refresh_token:{jti}"
            redis_client.setex(key, JWT_REFRESH_EXPIRATION_DAYS * 86400, username)
            return
        except Exception as e:
            logger.error(f"Failed to store token in Redis: {e}")

    with _in_memory_tokens_lock:
        _in_memory_tokens[jti] = time.time()
        if len(_in_memory_tokens) % 100 == 0:
            _cleanup_expired_tokens()


def _is_token_valid(jti: str) -> bool:
    """Проверка валидности refresh токена."""
    redis_client = _get_redis_client()
    if redis_client:
        try:
            key = f"refresh_token:{jti}"
            return redis_client.exists(key)
        except Exception as e:
            logger.error(f"Failed to check token in Redis: {e}")
            with _in_memory_tokens_lock:
                ts = _in_memory_tokens.get(jti)
                if ts and time.time() - ts <= _IN_MEMORY_TOKENS_MAX_AGE:
                    return True
                _in_memory_tokens.pop(jti, None)
                return False

    with _in_memory_tokens_lock:
        ts = _in_memory_tokens.get(jti)
        if ts is None:
            return False
        if time.time() - ts > _IN_MEMORY_TOKENS_MAX_AGE:
            _in_memory_tokens.pop(jti, None)
            return False
        return True


def _revoke_all_user_tokens(username: str):
    """Отмена всех refresh токенов пользователя."""
    redis_client = _get_redis_client()
    if redis_client:
        try:
            pattern = "refresh_token:*"
            for key in redis_client.scan_iter(match=pattern):
                if redis_client.get(key) == username:
                    redis_client.delete(key)
        except Exception as e:
            logger.error(f"Failed to revoke all user tokens in Redis: {e}")

    with _in_memory_tokens_lock:
        _in_memory_tokens.clear()


def _revoke_refresh_token(jti: str):
    """Отмена refresh токена."""
    redis_client = _get_redis_client()
    if redis_client:
        try:
            key = f"refresh_token:{jti}"
            redis_client.delete(key)
            return
        except Exception as e:
            logger.error(f"Failed to revoke token in Redis: {e}")

    with _in_memory_tokens_lock:
        _in_memory_tokens.pop(jti, None)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Создание access токена."""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRATION_MINUTES)

    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


def create_refresh_token(data: dict) -> str:
    """Создание refresh токена с уникальным jti."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=JWT_REFRESH_EXPIRATION_DAYS)
    jti = secrets.token_urlsafe(16)
    to_encode.update({"exp": expire, "type": "refresh", "jti": jti})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)

    _store_refresh_token(jti, data.get("sub", "unknown"))
    return encoded_jwt


def decode_token(token: str, allow_expired: bool = False) -> dict:
    """Декодирование JWT токена."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        if allow_expired:
            return jwt.decode(
                token,
                JWT_SECRET,
                algorithms=[JWT_ALGORITHM],
                options={"verify_exp": False},
            )
        return {"error": "token_expired"}
    except jwt.InvalidTokenError:
        return {"error": "invalid_token"}


def revoke_refresh_token(jti: str):
    """Отмена refresh токена (rotation)."""
    _revoke_refresh_token(jti)


def is_token_valid(jti: str) -> bool:
    """Проверка валидности refresh токена."""
    return _is_token_valid(jti)


def revoke_all_user_tokens(username: str):
    """Отмена всех refresh токенов пользователя."""
    _revoke_all_user_tokens(username)
