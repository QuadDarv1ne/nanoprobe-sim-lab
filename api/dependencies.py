# -*- coding: utf-8 -*-
"""
Зависимости для API роутов
Общие зависимости для всех роутов
"""

from fastapi import HTTPException, status, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Callable
from functools import wraps
from utils.database import DatabaseManager
from utils.redis_cache import RedisCache
from utils.batch_processor import BatchProcessor
import os
import jwt

security = HTTPBearer()
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """
    Получение текущего пользователя из JWT токена

    Args:
        credentials: HTTP Bearer credentials

    Returns:
        dict: Данные пользователя

    Raises:
        HTTPException: Если токен недействителен
    """
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Недействительный токен",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return {"user_id": user_id, "payload": payload}
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Недействительный токен",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_admin(current_user: dict) -> dict:
    """
    Проверка роли администратора

    Args:
        current_user: Данные текущего пользователя

    Returns:
        dict: Данные пользователя если это админ

    Raises:
        HTTPException: Если у пользователя нет роли администратора
    """
    payload = current_user.get("payload", {})
    role = payload.get("role", "user")
    if role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Требуется роль администратора"
        )
    return current_user


def get_db() -> DatabaseManager:
    """
    Зависимость для получения менеджера БД
    
    Returns:
        DatabaseManager: Экземпляр менеджера базы данных
        
    Raises:
        HTTPException: Если БД недоступна
    """
    from api.main import db_manager
    if db_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="База данных недоступна"
        )
    return db_manager


def get_redis_cache() -> Optional[RedisCache]:
    """
    Зависимость для получения Redis кэша
    
    Returns:
        Optional[RedisCache]: Экземпляр Redis кэша или None
    """
    from api.main import redis_cache
    return redis_cache


def get_redis_cache_required() -> RedisCache:
    """
    Зависимость для получения Redis кэша (обязательный)
    
    Returns:
        RedisCache: Экземпляр Redis кэша
        
    Raises:
        HTTPException: Если Redis недоступен
    """
    from api.main import redis_cache
    if redis_cache is None or not redis_cache.is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis кэш недоступен"
        )
    return redis_cache


def get_batch_processor() -> BatchProcessor:
    """
    Зависимость для получения процессора пакетной обработки
    
    Returns:
        BatchProcessor: Экземпляр процессора
    """
    return BatchProcessor()


def require_admin(current_user: dict) -> dict:
    """
    Проверка роли администратора
    
    Args:
        current_user: Данные текущего пользователя
        
    Returns:
        dict: Данные пользователя если это админ
        
    Raises:
        HTTPException: Если у пользователя нет роли администратора
    """
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Требуется роль администратора"
        )
    return current_user


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
    from utils.rate_limiter import RateLimiter
    
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            from api.dependencies import get_client_ip
            client_ip = get_client_ip(request)
            rate_limiter = RateLimiter()
            
            if not rate_limiter.is_allowed(client_ip, max_requests, window_seconds):
                retry_after = rate_limiter.get_retry_after(
                    client_ip, max_requests, window_seconds
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Слишком много запросов",
                    headers={"Retry-After": str(retry_after)}
                )
            
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator
