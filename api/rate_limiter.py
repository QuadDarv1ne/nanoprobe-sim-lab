"""
Comprehensive Rate Limiting для Nanoprobe Sim Lab API
Защита от DDoS, bruteforce атак и злоупотреблений

Endpoints:
- Auth: 5 запросов/мин (login, register)
- Write operations: 30 запросов/мин (POST, PUT, DELETE)
- Read operations: 100 запросов/мин (GET)
- Download: 20 запросов/мин
- SSTV/External: 10 запросов/мин

Требования:
- slowapi
- Redis (опционально для production)
"""

import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

# ==================== Limiter Configuration ====================


def _create_limiter() -> Limiter:
    """
    Создаёт limiter на основе текущих env-переменных.
    Вызывается при setup_rate_limiter, когда .env уже загружен.
    """
    redis_disabled = os.getenv("REDIS_DISABLED", "0") == "1"
    default_limits = ["100/minute", "20/second"]

    if not redis_disabled:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        try:
            return Limiter(
                key_func=get_remote_address,
                default_limits=default_limits,
                storage_uri=redis_url,
                strategy="fixed-window",
            )
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(
                f"Redis limiter failed, falling back to memory: {e}"
            )

    return Limiter(
        key_func=get_remote_address,
        default_limits=default_limits,
        storage_uri="memory://",
        strategy="fixed-window",
    )


# Placeholder — заменяется в setup_rate_limiter после загрузки .env
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100/minute", "20/second"],
    storage_uri="memory://",
    strategy="fixed-window",
)


# ==================== Setup ====================


def setup_rate_limiter(app):
    """
    Настройка rate limiter для FastAPI приложения.
    Вызывается после загрузки .env, пересоздаёт limiter с актуальными настройками.
    """
    global limiter
    limiter = _create_limiter()

    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)


# ==================== Rate Limit Decorators ====================


def auth_limit(max_requests: int = 5, window: int = 60):
    return limiter.limit(f"{max_requests}/{window}seconds")


def api_limit(max_requests: int = 100, window: int = 60):
    return limiter.limit(f"{max_requests}/{window}seconds")


def write_limit(max_requests: int = 30, window: int = 60):
    return limiter.limit(f"{max_requests}/{window}seconds")


def download_limit(max_requests: int = 20, window: int = 60):
    return limiter.limit(f"{max_requests}/{window}seconds")


def external_limit(max_requests: int = 10, window: int = 60):
    return limiter.limit(f"{max_requests}/{window}seconds")


def sstv_limit(max_requests: int = 10, window: int = 60):
    return limiter.limit(f"{max_requests}/{window}seconds")


# ==================== Custom Rate Limit Response ====================


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """
    Кастомный обработчик превышения лимита

    Returns:
        JSONResponse с информацией о лимите и Retry-After
    """
    retry_after = get_retry_after(exc)

    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": "Слишком много запросов. Пожалуйста, подождите.",
            "detail": str(exc.detail),
            "retry_after": retry_after,
            "retry_after_formatted": format_retry_after(retry_after),
        },
        headers={
            "Retry-After": str(retry_after),
            "X-RateLimit-Limit": get_limit_from_exception(exc),
            "X-RateLimit-Remaining": "0",
        },
    )


def get_retry_after(exc: RateLimitExceeded) -> int:
    """
    Получение времени до сброса лимита

    Returns:
        Количество секунд до сброса
    """
    try:
        detail = str(exc.detail)
        if "retry after" in detail.lower():
            parts = detail.split("retry after")
            if len(parts) > 1:
                time_str = parts[1].strip().split()[0]
                return max(1, min(3600, int(time_str)))  # 1s - 1h
    except (ValueError, IndexError, AttributeError):
        pass

    return 60  # Default 1 минута


def format_retry_after(seconds: int) -> str:
    """Форматирование времени до сброса в человекочитаемый вид"""
    if seconds < 60:
        return f"{seconds} сек"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes} мин"
    else:
        hours = seconds // 3600
        return f"{hours} час"


def get_limit_from_exception(exc: RateLimitExceeded) -> str:
    """Извлечение лимита из исключения"""
    try:
        detail = str(exc.detail)
        if "per" in detail:
            parts = detail.split("per")
            if len(parts) > 1:
                return parts[1].strip()
    except Exception as e:
        logger.debug(f"Failed to parse rate limit from exception: {e}")
    return "100/minute"


# ==================== Rate Limit Stats ====================

_rate_limit_stats: Dict[str, Dict] = {}


def inc_rate_limit_hit(endpoint: str):
    """Инкремент счётчика попаданий rate limit"""
    if endpoint not in _rate_limit_stats:
        _rate_limit_stats[endpoint] = {"hits": 0, "blocked": 0}
    _rate_limit_stats[endpoint]["hits"] += 1


def inc_rate_limit_blocked(endpoint: str):
    """Инкремент счётчика блокировок rate limit"""
    if endpoint not in _rate_limit_stats:
        _rate_limit_stats[endpoint] = {"hits": 0, "blocked": 0}
    _rate_limit_stats[endpoint]["blocked"] += 1


def get_rate_limit_stats() -> Dict:
    """
    Получение статистики rate limit

    Returns:
        Статистика по endpoint'ам
    """
    return _rate_limit_stats.copy()


def reset_rate_limit_stats():
    """Сброс статистики rate limit"""
    global _rate_limit_stats
    _rate_limit_stats = {}


# ==================== IP Whitelist/Blacklist ====================

import asyncio

_ip_whitelist: set = set()
_ip_blacklist: set = set()
_ip_lock = asyncio.Lock()  # Async-safe access to IP lists


async def whitelist_ip(ip: str):
    """Добавить IP в whitelist (без ограничений)"""
    async with _ip_lock:
        _ip_whitelist.add(ip)


async def blacklist_ip(ip: str):
    """Добавить IP в blacklist (полная блокировка)"""
    async with _ip_lock:
        _ip_blacklist.add(ip)


async def remove_from_whitelist(ip: str):
    """Удалить IP из whitelist"""
    async with _ip_lock:
        _ip_whitelist.discard(ip)


async def remove_from_blacklist(ip: str):
    """Удалить IP из blacklist"""
    async with _ip_lock:
        _ip_blacklist.discard(ip)


async def is_ip_whitelisted(ip: str) -> bool:
    """Проверка IP в whitelist"""
    async with _ip_lock:
        return ip in _ip_whitelist


async def is_ip_blacklisted(ip: str) -> bool:
    """Проверка IP в blacklist"""
    async with _ip_lock:
        return ip in _ip_blacklist


async def get_ip_lists() -> Dict:
    """Получение списков IP"""
    async with _ip_lock:
        return {"whitelist": list(_ip_whitelist), "blacklist": list(_ip_blacklist)}
