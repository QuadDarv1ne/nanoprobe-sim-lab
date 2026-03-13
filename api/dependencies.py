# -*- coding: utf-8 -*-
"""
Зависимости для API роутов
Общие зависимости для всех роутов
"""

from fastapi import HTTPException, status
from typing import Optional
from utils.database import DatabaseManager
from utils.redis_cache import RedisCache
from utils.batch_processor import BatchProcessor
import os


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
