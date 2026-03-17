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

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse
from typing import Dict, Optional
import os


# ==================== Limiter Configuration ====================

# Лимитер с использованием in-memory (Redis опционален)
redis_disabled = os.getenv("REDIS_DISABLED", "0") == "1"
if redis_disabled:
    # In-memory limiter без Redis
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[
            "100/minute",  # Default limit для всех endpoints
            "20/second",   # Burst protection
        ],
        storage_uri="memory://",
        strategy="fixed-window"
    )
else:
    # Redis limiter для production
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[
            "100/minute",
            "20/second",
        ],
        storage_uri=redis_url,
        strategy="fixed-window"
    )


# ==================== Setup ====================

def setup_rate_limiter(app):
    """
    Настройка rate limiter для FastAPI приложения

    Добавляет:
    - Middleware для автоматического rate limiting
    - Exception handler
    - State для доступа в роутах

    Args:
        app: FastAPI приложение
    """
    # Отключаем middleware если Redis отключён (избегаем ошибок ConnectionError)
    redis_disabled = os.getenv("REDIS_DISABLED", "0") == "1"
    if redis_disabled:
        # В режиме без Redis просто добавляем exception handler
        app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
        return
    
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)


# ==================== Rate Limit Decorators ====================

def auth_limit(max_requests: int = 5, window: int = 60):
    """
    Строгий лимит для auth endpoints (login, register, 2FA)
    
    Args:
        max_requests: Максимум запросов
        window: Окно времени в секундах

    Returns:
        Декоратор rate limit
    """
    return limiter.limit(f"{max_requests}/{window}seconds")


def api_limit(max_requests: int = 100, window: int = 60):
    """
    Стандартный лимит для обычных API endpoints (GET)

    Args:
        max_requests: Максимум запросов
        window: Окно времени в секундах

    Returns:
        Декоратор rate limit
    """
    return limiter.limit(f"{max_requests}/{window}seconds")


def write_limit(max_requests: int = 30, window: int = 60):
    """
    Лимит для write операций (POST, PUT, DELETE)

    Args:
        max_requests: Максимум запросов
        window: Окно времени в секундах

    Returns:
        Декоратор rate limit
    """
    return limiter.limit(f"{max_requests}/{window}seconds")


def download_limit(max_requests: int = 20, window: int = 60):
    """
    Лимит для download endpoints (файлы, изображения)

    Args:
        max_requests: Максимум запросов
        window: Окно времени в секундах

    Returns:
        Декоратор rate limit
    """
    return limiter.limit(f"{max_requests}/{window}seconds")


def external_limit(max_requests: int = 10, window: int = 60):
    """
    Лимит для external API endpoints (NASA, external services)

    Args:
        max_requests: Максимум запросов
        window: Окно времени в секундах

    Returns:
        Декоратор rate limit
    """
    return limiter.limit(f"{max_requests}/{window}seconds")


def sstv_limit(max_requests: int = 10, window: int = 60):
    """
    Лимит для SSTV operations (decode, upload)

    Args:
        max_requests: Максимум запросов
        window: Окно времени в секундах

    Returns:
        Декоратор rate limit
    """
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
            "retry_after_formatted": format_retry_after(retry_after)
        },
        headers={
            "Retry-After": str(retry_after),
            "X-RateLimit-Limit": get_limit_from_exception(exc),
            "X-RateLimit-Remaining": "0"
        }
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
    except Exception:
        pass
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

_ip_whitelist: set = set()
_ip_blacklist: set = set()


def whitelist_ip(ip: str):
    """Добавить IP в whitelist (без ограничений)"""
    _ip_whitelist.add(ip)


def blacklist_ip(ip: str):
    """Добавить IP в blacklist (полная блокировка)"""
    _ip_blacklist.add(ip)


def remove_from_whitelist(ip: str):
    """Удалить IP из whitelist"""
    _ip_whitelist.discard(ip)


def remove_from_blacklist(ip: str):
    """Удалить IP из blacklist"""
    _ip_blacklist.discard(ip)


def is_ip_whitelisted(ip: str) -> bool:
    """Проверка IP в whitelist"""
    return ip in _ip_whitelist


def is_ip_blacklisted(ip: str) -> bool:
    """Проверка IP в blacklist"""
    return ip in _ip_blacklist


def get_ip_lists() -> Dict:
    """Получение списков IP"""
    return {
        "whitelist": list(_ip_whitelist),
        "blacklist": list(_ip_blacklist)
    }
