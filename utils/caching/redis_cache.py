"""
Redis кэш для Nanoprobe Sim Lab
Кэширование результатов запросов для ускорения API
"""

import hashlib
import json
import logging
import os
import time
from functools import wraps
from typing import Any, Callable, Optional

import redis

logger = logging.getLogger(__name__)


class RedisCache:
    """Менеджер Redis кэша с автоматическим reconnect"""

    # Константы для reconnect логики
    RECONNECT_INTERVAL = 30  # секунд между попытками подключения
    RECONNECT_MAX_ATTEMPTS = 10  # максимум попыток
    RECONNECT_BACKOFF_FACTOR = 1.5  # экспоненциальный backoff

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: Optional[int] = None,
        password: Optional[str] = None,
    ):
        # Приоритет: аргументы → env → defaults
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port or int(os.getenv("REDIS_PORT", "6379"))
        self.db = db if db is not None else int(os.getenv("REDIS_DB", "0"))
        self.password = password or os.getenv("REDIS_PASSWORD") or None
        self._client: Optional[redis.Redis] = None
        self._enabled = True

        # Reconnect state
        self._last_connect_attempt: float = 0
        self._connect_attempts: int = 0
        self._connection_lost: bool = False

    def _should_attempt_reconnect(self) -> bool:
        """Проверка стоит ли пытаться переподключиться"""
        if self._connect_attempts >= self.RECONNECT_MAX_ATTEMPTS:
            return False

        now = time.time()
        elapsed = now - self._last_connect_attempt

        # Экспоненциальный backoff
        backoff = self.RECONNECT_INTERVAL * (
            self.RECONNECT_BACKOFF_FACTOR ** min(self._connect_attempts, 5)
        )

        return elapsed >= backoff

    @property
    def client(self) -> Optional[redis.Redis]:
        """Получение Redis клиента с автоматическим reconnect"""
        # Если клиент уже создан, проверяем соединение
        if self._client is not None:
            try:
                self._client.ping()
                # Соединение живо, сбрасываем флаг потери
                if self._connection_lost:
                    logger.info("Redis connection restored")
                    self._connection_lost = False
                return self._client
            except (redis.ConnectionError, redis.TimeoutError, OSError):
                # Соединение потеряно
                self._connection_lost = True
                self._client = None
                self._connect_attempts = 0  # Сбрасываем для новых попыток
                logger.warning("Redis connection lost, will attempt reconnect")

        # Пробуем переподключиться
        if self._should_attempt_reconnect():
            self._last_connect_attempt = time.time()
            self._connect_attempts += 1

            try:
                self._client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                    retry_on_timeout=True,
                    health_check_interval=10,
                )
                self._client.ping()
                self._enabled = True
                self._connect_attempts = 0  # Сброс после успешного подключения
                logger.info(f"Redis connected successfully (attempt {self._connect_attempts})")
                return self._client
            except (redis.ConnectionError, redis.TimeoutError, redis.RedisError, OSError) as e:
                self._client = None
                if self._connect_attempts <= 3:
                    logger.debug(f"Redis connect attempt {self._connect_attempts} failed: {e}")
                else:
                    logger.warning(f"Redis connect attempt {self._connect_attempts} failed: {e}")

        # Не пытаемся подключиться сейчас
        if not self._should_attempt_reconnect():
            self._enabled = False

        return None

    def get(self, key: str) -> Optional[Any]:
        if not self._enabled or not self.client:
            return None
        try:
            value = self.client.get(key)
            return json.loads(value) if value else None
        except (redis.RedisError, json.JSONDecodeError):
            return None

    def set(self, key: str, value: Any, expire: int = 300) -> bool:
        if not self._enabled or not self.client:
            return False
        try:
            serialized = json.dumps(value, ensure_ascii=False, default=str)
            return bool(self.client.setex(key, expire, serialized))
        except (redis.RedisError, TypeError):
            return False

    def delete(self, key: str) -> bool:
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
        key_data = ":".join(str(arg) for arg in args)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()[:16]
        return f"{prefix}:{key_hash}"

    def is_available(self) -> bool:
        return self._enabled and self.client is not None

    def get_stats(self) -> dict:
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
                "keys_count": sum(db.get("keys", 0) for db in keyspace.values()) if keyspace else 0,
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
        @wraps(func)
        async def wrapper(*args, **kwargs):
            from api.state import get_redis

            # Generate cache key from query params only (exclude Depends-injected objects)
            # Filter out non-primitive types that would make the key unstable
            stable_kwargs = {
                k: v
                for k, v in kwargs.items()
                if isinstance(v, (str, int, float, bool, type(None), list, dict, tuple))
            }
            cache_key = f"{prefix}:{func.__name__}:"
            cache_key += hashlib.md5(
                json.dumps(stable_kwargs, sort_keys=True, default=str).encode()
            ).hexdigest()[:16]

            # Попытка получить из кэша
            redis_instance = get_redis()
            if redis_instance and redis_instance.is_available():
                cached_value = redis_instance.get(cache_key)
                if cached_value is not None:
                    return cached_value

            # Вызов функции
            result = await func(*args, **kwargs)

            # Сохранение в кэш
            if redis_instance and redis_instance.is_available():
                redis_instance.set(cache_key, result, expire)

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
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Используем app state если доступен, иначе модульный cache
            try:
                from api.state import get_redis

                redis_instance = get_redis()
            except Exception:
                redis_instance = cache

            # Генерация ключа
            cache_key = f"{prefix}:{func.__name__}:"
            cache_key += hashlib.md5(f"{sorted(kwargs.items())}".encode()).hexdigest()[:16]

            # Попытка получить из кэша
            if redis_instance and redis_instance.is_available():
                cached_value = redis_instance.get(cache_key)
                if cached_value is not None:
                    return cached_value

            # Вызов функции
            result = func(*args, **kwargs)

            # Сохранение в кэш
            if redis_instance and redis_instance.is_available():
                redis_instance.set(cache_key, result, expire)

            return result

        return wrapper

    return decorator
