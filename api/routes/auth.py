"""
Auth API Router - Unified

Объединяет все auth endpoint'ы из отдельных модулей:
- helpers: Утилиты, хэширование, валидация паролей
- tokens: JWT токены, refresh token rotation
- endpoints: Login, refresh, 2FA, logout

Обратная совместимость: все старые URL'ы работают.
"""

from fastapi import APIRouter

from api.routes.auth_routes.endpoints import router as endpoints_router
from api.routes.auth_routes.helpers import (
    JWT_ALGORITHM,
    JWT_EXPIRATION_MINUTES,
    JWT_REFRESH_EXPIRATION_DAYS,
    JWT_SECRET,
    AuditEventType,
    _get_users_db,
    hash_password,
    log_audit_event,
    ph,
    validate_password_strength,
    verify_password,
)
from api.routes.auth_routes.tokens import (
    _cleanup_expired_tokens,
    _in_memory_tokens,
    _in_memory_tokens_lock,
    _is_token_valid,
    _revoke_refresh_token,
    _store_refresh_token,
    create_access_token,
    create_refresh_token,
    decode_token,
    is_token_valid,
    revoke_all_user_tokens,
    revoke_refresh_token,
)

router = APIRouter()
router.include_router(endpoints_router)

__all__ = [
    "router",
    "JWT_ALGORITHM",
    "JWT_EXPIRATION_MINUTES",
    "JWT_REFRESH_EXPIRATION_DAYS",
    "JWT_SECRET",
    "AuditEventType",
    "hash_password",
    "log_audit_event",
    "ph",
    "validate_password_strength",
    "verify_password",
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "is_token_valid",
    "revoke_all_user_tokens",
    "revoke_refresh_token",
    "_store_refresh_token",
    "_is_token_valid",
    "_revoke_refresh_token",
    "_cleanup_expired_tokens",
    "_in_memory_tokens",
    "_in_memory_tokens_lock",
]
