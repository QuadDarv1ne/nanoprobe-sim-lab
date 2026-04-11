"""
Глобальные состояния приложения
Хранит ссылки на общие ресурсы (БД, Redis, etc.)
"""

import os
import platform
from typing import Any, Dict, Optional

from utils.caching.redis_cache import RedisCache
from utils.database import DatabaseManager

# Глобальные состояния
db_manager: Optional[DatabaseManager] = None
redis_cache: Optional[RedisCache] = None
app_state: Dict[str, Any] = {}


def get_system_disk_usage():
    r"""
    Cross-platform disk usage.
    On Windows uses SYSTEMDRIVE env var (default C:\), on Unix uses '/'.
    Returns psutil.sdiskusage namedtuple.
    """
    import psutil

    if platform.system() == "Windows":
        path = os.environ.get("SYSTEMDRIVE", "C:\\")
    else:
        path = "/"
    return psutil.disk_usage(path)


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


def get_app_state(key: str, default: Any = None) -> Any:
    """Получить значение из состояния приложения"""
    return app_state.get(key, default)


def set_app_state(key: str, value: Any) -> None:
    """Установить значение в состоянии приложения"""
    app_state[key] = value


def clear_app_state() -> None:
    """Очистить состояние приложения"""
    app_state.clear()


def init_app_state(db: DatabaseManager, redis: RedisCache) -> None:
    """Инициализировать состояние приложения"""
    set_db_manager(db)
    set_redis(redis)
