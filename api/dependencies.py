"""
Зависимости для API роутов
Общие зависимости для всех роутов
"""

from fastapi import HTTPException, status, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from functools import wraps
import logging
from utils.database import DatabaseManager
from utils.caching.redis_cache import RedisCache
from utils.batch_processor import BatchProcessor
from api.error_handlers import AuthorizationError, RateLimitError, DatabaseError
import os
import jwt

logger = logging.getLogger(__name__)

security = HTTPBearer()

# JWT Secret из централизованного источника
def _get_jwt_secret() -> str:
    """Получает JWT секрет из централизованного источника"""
    try:
        from api.security.jwt_config import get_jwt_secret
        return get_jwt_secret()
    except Exception as e:
        logger.error(f"Failed to get JWT secret: {e}")
        raise RuntimeError("JWT secret configuration error")

JWT_ALGORITHM = "HS256"


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """
    Получение текущего пользователя из JWT токена.
    Проверяет in-memory USERS_DB, затем SQLite для динамически созданных пользователей.
    """
    from api.routes.auth import _get_users_db

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Неверные учетные данные",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        token = credentials.credentials
        jwt_secret = _get_jwt_secret()
        payload = jwt.decode(token, jwt_secret, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")

        if username is None:
            raise credentials_exception

        user = _get_users_db().get(username)
        if user:
            return user

        # Fallback: check SQLite for dynamically created users
        try:
            from api.state import get_db_manager
            db = get_db_manager()
            db_user = db.get_user(username)
            if db_user:
                return {
                    "id": db_user["id"],
                    "username": db_user["username"],
                    "password_hash": db_user["password_hash"],
                    "role": db_user["role"],
                    "created_at": db_user.get("created_at", ""),
                    "last_login": db_user.get("last_login"),
                }
        except Exception as e:
            logger.warning(f"SQLite user lookup failed: {e}")

        raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception


def require_admin(current_user: dict) -> dict:
    """
    Проверка роли администратора

    Args:
        current_user: Данные текущего пользователя

    Returns:
        dict: Данные пользователя если это админ

    Raises:
        AuthorizationError: Если у пользователя нет роли администратора
    """
    if current_user.get("role") != "admin":
        raise AuthorizationError("Требуется роль администратора")
    return current_user


def get_db() -> DatabaseManager:
    """Зависимость для получения менеджера БД"""
    from api.state import get_db_manager
    try:
        return get_db_manager()
    except RuntimeError as e:
        logger.error(f"Database manager not initialized: {e}")
        raise DatabaseError(str(e))


def get_redis_cache() -> Optional[RedisCache]:
    """Зависимость для получения Redis кэша"""
    from api.state import get_redis
    return get_redis()


def get_redis_cache_required() -> RedisCache:
    """Зависимость для получения Redis кэша (обязательный)"""
    from api.state import get_redis_required
    try:
        return get_redis_required()
    except RuntimeError as e:
        logger.error(f"Redis cache not available: {e}")
        raise DatabaseError(str(e))


def get_batch_processor() -> BatchProcessor:
    """
    Зависимость для получения процессора пакетной обработки

    Returns:
        BatchProcessor: Экземпляр процессора
    """
    return BatchProcessor()


def get_client_ip(request: Request) -> str:
    """
    Получение IP адреса клиента

    Args:
        request: FastAPI request объект

    Returns:
        str: IP адрес клиента
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(max_requests: int = 10, window_seconds: int = 60):
    """
    Декоратор для ограничения частоты запросов

    Args:
        max_requests: Максимальное количество запросов
        window_seconds: Окно времени в секундах

    Использование:
        @router.post("/login")
        @rate_limit(max_requests=5, window_seconds=60)
        async def login(...):
            ...
    """
    from utils.security.rate_limiter import RateLimiter

    def decorator(func):
        """Декоратор для ограничения частоты запросов"""
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            from api.dependencies import get_client_ip
            client_ip = get_client_ip(request)
            rate_limiter = RateLimiter()

            if not rate_limiter.is_allowed(client_ip, max_requests, window_seconds):
                retry_after = rate_limiter.get_retry_after(
                    client_ip, max_requests, window_seconds
                )
                raise RateLimitError(retry_after)

            return await func(request, *args, **kwargs)
        return wrapper
    return decorator
