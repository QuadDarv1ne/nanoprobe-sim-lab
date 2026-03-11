# -*- coding: utf-8 -*-
"""
Redis кэш для Nanoprobe Sim Lab
Кэширование результатов запросов для ускорения API
"""

import json
import hashlib
from typing import Any, Optional
from datetime import timedelta
import redis
from pathlib import Path


class RedisCache:
    """Менеджер Redis кэша"""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        """
        Инициализация Redis кэша.

        Args:
            host: Хост Redis
            port: Порт Redis
            db: Номер БД Redis
        """
        self.host = host
        self.port = port
        self.db = db
        self._client: Optional[redis.Redis] = None
        self._enabled = True

    @property
    def client(self) -> Optional[redis.Redis]:
        """Получение Redis клиента"""
        if self._client is None:
            try:
                self._client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    decode_responses=True,
                    socket_connect_timeout=5,
                )
                self._client.ping()
            except (redis.ConnectionError, redis.TimeoutError):
                self._enabled = False
                self._client = None
        return self._client

    def get(self, key: str) -> Optional[Any]:
        """
        Получение значения из кэша.

        Args:
            key: Ключ

        Returns:
            Значение или None
        """
        if not self._enabled or not self.client:
            return None

        try:
            value = self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except (redis.RedisError, json.JSONDecodeError):
            return None

    def set(
        self,
        key: str,
        value: Any,
        expire: int = 300
    ) -> bool:
        """
        Сохранение значения в кэш.

        Args:
            key: Ключ
            value: Значение
            expire: Время жизни в секундах (по умолчанию 5 минут)

        Returns:
            True если успешно
        """
        if not self._enabled or not self.client:
            return False

        try:
            serialized = json.dumps(value, ensure_ascii=False, default=str)
            return self.client.setex(key, expire, serialized)
        except (redis.RedisError, TypeError):
            return False

    def delete(self, key: str) -> bool:
        """
        Удаление ключа из кэша.

        Args:
            key: Ключ

        Returns:
            True если ключ удалён
        """
        if not self._enabled or not self.client:
            return False

        try:
            return bool(self.client.delete(key))
        except redis.RedisError:
            return False

    def clear_pattern(self, pattern: str) -> int:
        """
        Очистка ключей по паттерну.

        Args:
            pattern: Паттерн (например, "scans:*")

        Returns:
            Количество удалённых ключей
        """
        if not self._enabled or not self.client:
            return 0

        try:
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except redis.RedisError:
            return 0

    def generate_key(self, prefix: str, *args) -> str:
        """
        Генерация ключа кэша.

        Args:
            prefix: Префикс ключа
            *args: Аргументы для хэширования

        Returns:
            Сгенерированный ключ
        """
        key_data = ":".join(str(arg) for arg in args)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()[:16]
        return f"{prefix}:{key_hash}"

    def is_available(self) -> bool:
        """
        Проверка доступности Redis.

        Returns:
            True если Redis доступен
        """
        return self._enabled and self.client is not None

    def get_stats(self) -> dict:
        """
        Получение статистики Redis.

        Returns:
            Статистика
        """
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


# Глобальный экземляр кэша
cache = RedisCache()


def cached(
    prefix: str = "api",
    expire: int = 300,
    key_func=None
):
    """
    Декоратор для кэширования результатов функций.

    Args:
        prefix: Префикс для ключа кэша
        expire: Время жизни кэша в секундах
        key_func: Функция для генерации ключа (опционально)

    Returns:
        Декоратор
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Генерация ключа
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = cache.generate_key(
                    prefix,
                    func.__name__,
                    *args,
                    **kwargs
                )

            # Попытка получить из кэша
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            # Вызов функции
            result = await func(*args, **kwargs) if func.__name__.startswith("async") else func(*args, **kwargs)

            # Сохранение в кэш
            cache.set(cache_key, result, expire)

            return result
        return wrapper
    return decorator
