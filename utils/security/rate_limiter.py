"""
Rate Limiter для Nanoprobe Sim Lab API
Защита от brute force и DDoS атак
"""

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import wraps
from typing import Dict, List, Optional, Tuple

from fastapi import Request

from api.error_handlers import RateLimitError


@dataclass
class RateLimitInfo:
    """Информация о rate limit"""

    requests: List[float] = field(default_factory=list)
    blocked_until: float = 0.0
    violation_count: int = 0


class RateLimiter:
    """Rate limiter с скользящим окном, прогрессивной блокировкой и автоматической очисткой"""

    _instance: Optional["RateLimiter"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "RateLimiter":
        """Singleton паттерн"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self.requests: Dict[str, RateLimitInfo] = defaultdict(RateLimitInfo)
        self._cleanup_lock = threading.Lock()

        # Конфигурация прогрессивной блокировки
        self.progressive_blocking = True
        self.block_multipliers = [2, 5, 15, 60]  # множители блокировки в минутах
        self._initialized = True

    def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> bool:
        now = time.time()
        window_start = now - window_seconds

        with self._cleanup_lock:
            info = self.requests[key]

            # Проверка блокировки
            if info.blocked_until > now:
                return False

            # Очистка старых запросов
            info.requests = [ts for ts in info.requests if ts > window_start]

            if len(info.requests) >= max_requests:
                # Прогрессивная блокировка при нарушениях
                if self.progressive_blocking:
                    info.violation_count += 1
                    multiplier_idx = min(info.violation_count - 1, len(self.block_multipliers) - 1)
                    block_minutes = self.block_multipliers[multiplier_idx]
                    info.blocked_until = now + (block_minutes * 60)
                return False

            info.requests.append(now)
            return True

    def get_retry_after(self, key: str, max_requests: int, window_seconds: int) -> int:
        info = self.requests.get(key)
        if not info:
            return 0

        now = time.time()

        # Если заблокирован
        if info.blocked_until > now:
            return int(info.blocked_until - now)

        if not info.requests:
            return 0

        oldest = min(info.requests)
        retry_after = int(oldest + window_seconds - time.time())
        return max(0, retry_after)

    def cleanup_old_entries(self, max_age_seconds: int = 3600):
        """Очистка старых записей для экономии памяти"""
        now = time.time()
        cutoff = now - max_age_seconds

        with self._cleanup_lock:
            empty_keys = []
            for key, info in self.requests.items():
                # Сброс блокировки если истекла
                if info.blocked_until < now:
                    info.blocked_until = 0.0
                    info.violation_count = 0

                info.requests = [ts for ts in info.requests if ts > cutoff]
                if not info.requests and info.blocked_until == 0:
                    empty_keys.append(key)

            for key in empty_keys:
                del self.requests[key]

    def get_request_count(self, key: str, window_seconds: int) -> int:
        """Получение количества запросов за окно"""
        now = time.time()
        window_start = now - window_seconds
        info = self.requests.get(key)
        if not info:
            return 0
        return len([ts for ts in info.requests if ts > window_start])

    def get_status(self, key: str, max_requests: int, window_seconds: int) -> Dict:
        """Получение статуса rate limiting"""
        now = time.time()
        info = self.requests.get(key)

        if not info:
            return {
                "requests_made": 0,
                "requests_remaining": max_requests,
                "blocked": False,
                "retry_after": 0,
            }

        window_start = now - window_seconds
        requests_in_window = len([ts for ts in info.requests if ts > window_start])
        blocked = info.blocked_until > now

        return {
            "requests_made": requests_in_window,
            "requests_remaining": max(0, max_requests - requests_in_window),
            "blocked": blocked,
            "retry_after": int(info.blocked_until - now) if blocked else 0,
            "violation_count": info.violation_count,
        }

    def reset(self, key: str):
        """Сброс rate limit для ключа"""
        with self._cleanup_lock:
            if key in self.requests:
                del self.requests[key]

    def is_blocked(self, key: str) -> Tuple[bool, str]:
        """Проверка, заблокирован ли ключ"""
        now = time.time()
        info = self.requests.get(key)
        if info and info.blocked_until > now:
            remaining = info.blocked_until - now
            return True, f"Заблокировано на {remaining / 60:.1f} минут"
        return False, ""


limiter = RateLimiter()

# Автоматическая очистка rate limiter каждые 5 минут
_rate_limit_cleanup_task = None


async def _rate_limit_cleanup_loop():
    """Фоновый цикл очистки старых записей rate limiter"""
    import asyncio

    while True:
        try:
            await asyncio.sleep(300)  # 5 минут
            limiter.cleanup_old_entries(max_age_seconds=3600)
        except Exception as e:
            import logging

            logging.getLogger(__name__).error(f"Rate limiter cleanup error: {e}")


def start_rate_limit_cleanup():
    """Запуск автоматической очистки rate limiter"""
    import asyncio

    global _rate_limit_cleanup_task
    if _rate_limit_cleanup_task is None or _rate_limit_cleanup_task.done():
        _rate_limit_cleanup_task = asyncio.create_task(_rate_limit_cleanup_loop())
        import logging

        logging.getLogger(__name__).info("Rate limiter auto-cleanup started (every 5 minutes)")


def rate_limit(max_requests: int = 10, window_seconds: int = 60):
    """
    Декоратор для rate limiting с прогрессивной блокировкой.

    Args:
        max_requests: Максимум запросов в окно
        window_seconds: Размер окна в секундах
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            client_ip = request.client.host
            key = f"{func.__name__}:{client_ip}"

            if not limiter.is_allowed(key, max_requests, window_seconds):
                retry_after = limiter.get_retry_after(key, max_requests, window_seconds)
                raise RateLimitError(retry_after)

            return await func(request, *args, **kwargs)

        return wrapper

    return decorator
