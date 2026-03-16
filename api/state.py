"""
Глобальные состояния приложения
Хранит ссылки на общие ресурсы (БД, Redis, etc.)
"""

from typing import Optional
from utils.database.database import DatabaseManager
from utils.caching.redis_cache import RedisCache

# Глобальные состояния
db_manager: Optional[DatabaseManager] = None
redis_cache: Optional[RedisCache] = None


def get_db_manager() -> DatabaseManager:
    """Получить менеджер БД"""
    if db_manager is None:
        raise RuntimeError("Database manager not initialized")
    return db_manager


def get_redis() -> Optional[RedisCache]:
    """Получить Redis кэш"""
    return redis_cache


def get_redis_required() -> RedisCache:
    """Получить Redis кэш (обязательно)"""
    if redis_cache is None or not redis_cache.is_available():
        raise RuntimeError("Redis cache not available")
    return redis_cache


def set_db_manager(manager: DatabaseManager) -> None:
    """Установить менеджер БД"""
    global db_manager
    db_manager = manager


def set_redis(cache: RedisCache) -> None:
    """Установить Redis кэш"""
    global redis_cache
    redis_cache = cache
