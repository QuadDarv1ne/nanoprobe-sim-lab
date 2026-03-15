"""
Redis кэш для Nanoprobe Sim Lab
Кэширование результатов запросов для ускорения API
"""

import json
import hashlib
import asyncio
from typing import Any, Optional, Callable
from functools import wraps
from datetime import timedelta
import redis
from pathlib import Path


class RedisCache:
    """Менеджер Redis кэша"""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        """TODO: Add description"""
        self.host = host
        self.port = port
        self.db = db
        self._client: Optional[redis.Redis] = None
        self._enabled = True

    @property
    def client(self) -> Optional[redis.Redis]:
        """TODO: Add description"""
        if self._client is None:
            try:
                self._client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30,
                )
                self._client.ping()
            except (redis.ConnectionError, redis.TimeoutError, redis.RedisError):
                self._enabled = False
                self._client = None
        return self._client

    def get(self, key: str) -> Optional[Any]:
        """TODO: Add description"""
        if not self._enabled or not self.client:
            return None
        try:
            value = self.client.get(key)
            return json.loads(value) if value else None
        except (redis.RedisError, json.JSONDecodeError):
            return None

    def set(self, key: str, value: Any, expire: int = 300) -> bool:
        """TODO: Add description"""
        if not self._enabled or not self.client:
            return False
        try:
            serialized = json.dumps(value, ensure_ascii=False, default=str)
            return bool(self.client.setex(key, expire, serialized))
        except (redis.RedisError, TypeError):
            return False

    def delete(self, key: str) -> bool:
        """TODO: Add description"""
        if not self._enabled or not self.client:
            return False
        try:
            return bool(self.client.delete(key))
        except redis.RedisError:
            return False

    def clear_pattern(self, pattern: str) -> int:
        """
        Удаление ключей по паттерну (безопасно для production).
        Использует SCAN вместо KEYS для production безопасности.
        """
        if not self._enabled or not self.client:
            return 0
        try:
            deleted = 0
            cursor = 0
            while True:
                cursor, keys = self.client.scan(cursor, match=pattern, count=100)
                if keys:
                    deleted += self.client.delete(*keys)
                if cursor == 0:
                    break
            return deleted
        except redis.RedisError:
            return 0

    def delete_many(self, *keys: str) -> int:
        """Удаление нескольких ключей"""
        if not self._enabled or not self.client or not keys:
            return 0
        try:
            return self.client.delete(*keys)
        except redis.RedisError:
            return 0

    def exists(self, key: str) -> bool:
        """Проверка существования ключа"""
        if not self._enabled or not self.client:
            return False
        try:
            return bool(self.client.exists(key))
        except redis.RedisError:
            return False

    def generate_key(self, prefix: str, *args) -> str:
        """TODO: Add description"""
        key_data = ":".join(str(arg) for arg in args)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()[:16]
        return f"{prefix}:{key_hash}"

    def is_available(self) -> bool:
        """TODO: Add description"""
        return self._enabled and self.client is not None

    def get_stats(self) -> dict:
        """TODO: Add description"""
        if not self._enabled or not self.client:
            return {"available": False}
        try:
            info = self.client.info("stats")
            keyspace = self.client.info("keyspace")
            return {
                "available": True,
                "connected": True,
                "total_connections_received": info.get("total_connections_received", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "keys_count": sum(
                    db.get("keys", 0) for db in keyspace.values()
                ) if keyspace else 0,
            }
        except redis.RedisError:
            return {"available": False, "error": "Connection error"}

    def close(self):
        """Закрытие соединения с Redis"""
        if self._client:
            try:
                self._client.close()
            except redis.RedisError:
                pass
            finally:
                self._client = None

    def invalidate_by_prefix(self, prefix: str) -> int:
        """
        Инвалидация кэша по префиксу

        Args:
            prefix: Префикс ключей для удаления

        Returns:
            Количество удалённых ключей
        """
        return self.clear_pattern(f"{prefix}:*")

    def cache_json(self, key: str, data: Any, expire: int = 300) -> bool:
        """Кэширование JSON данных"""
        return self.set(f"json:{key}", data, expire)

    def get_json(self, key: str) -> Optional[dict]:
        """Получение JSON данных из кэша"""
        return self.get(f"json:{key}")


cache = RedisCache()


def cached(prefix: str = "api", expire: int = 300):
    """
    Декоратор для кэширования результатов async функций.

    Args:
        prefix: Префикс для ключа кэша
        expire: Время жизни кэша в секундах
    """
    def decorator(func: Callable) -> Callable:
        """TODO: Add description"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            from api.main import redis_cache

            # Генерация ключа
            cache_key = f"{prefix}:{func.__name__}:"
            cache_key += hashlib.md5(
                f"{args}:{kwargs}".encode()
            ).hexdigest()[:16]

            # Попытка получить из кэша
            if redis_cache and redis_cache.is_available():
                cached_value = redis_cache.get(cache_key)
                if cached_value is not None:
                    return cached_value

            # Вызов функции
            result = await func(*args, **kwargs)

            # Сохранение в кэш
            if redis_cache and redis_cache.is_available():
                redis_cache.set(cache_key, result, expire)

            return result
        return wrapper
    return decorator


def cached_sync(prefix: str = "api", expire: int = 300):
    """
    Декоратор для кэширования результатов sync функций.

    Args:
        prefix: Префикс для ключа кэша
        expire: Время жизни кэша в секундах
    """
    def decorator(func: Callable) -> Callable:
        """TODO: Add description"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            """TODO: Add description"""
            from utils.caching.redis_cache import cache

            # Генерация ключа
            cache_key = f"{prefix}:{func.__name__}:"
            cache_key += hashlib.md5(
                f"{sorted(kwargs.items())}".encode()
            ).hexdigest()[:16]

            # Попытка получить из кэша
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            # Вызов функции
            result = func(*args, **kwargs)

            # Сохранение в кэш
            cache.set(cache_key, result, expire)

            return result
        return wrapper
    return decorator
