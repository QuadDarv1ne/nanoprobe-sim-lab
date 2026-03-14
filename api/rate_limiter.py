"""
Rate Limiting для Nanoprobe Sim Lab API
Защита от DDoS и bruteforce атак
"""

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from fastapi.responses import JSONResponse
from typing import Dict, Optional


# Лимитер с использованием Redis (если доступен) или in-memory
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100/minute"],  # Default limit для всех endpoints
    storage_uri="memory://"  # Будет заменено на Redis если доступен
)


def setup_rate_limiter(app):
    """
    Настройка rate limiter для FastAPI приложения

    Args:
        app: FastAPI приложение
    """
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ==================== Rate Limit Decorators ====================

def auth_limit(max_requests: int = 10, window: int = 60):
    """
    Лимит для auth endpoints (login, register)

    Args:
        max_requests: Максимум запросов
        window: Окно времени в секундах

    Returns:
        Декоратор rate limit
    """
    return limiter.limit(f"{max_requests}/{window}seconds")


def api_limit(max_requests: int = 100, window: int = 60):
    """
    Лимит для обычных API endpoints

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
    Лимит для download endpoints

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
        JSONResponse с информацией о лимите
    """
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "message": "Слишком много запросов. Пожалуйста, подождите.",
            "detail": str(exc.detail),
            "retry_after": get_retry_after(exc)
        }
    )


def get_retry_after(exc: RateLimitExceeded) -> Optional[int]:
    """
    Получение времени до сброса лимита

    Returns:
        Количество секунд до сброса
    """
    # Парсинг времени из ошибки
    try:
        detail = str(exc.detail)
        if "retry after" in detail.lower():
            # Извлечение времени из сообщения
            parts = detail.split("retry after")
            if len(parts) > 1:
                time_str = parts[1].strip().split()[0]
                return int(time_str)
    except Exception:
        pass

    return 60  # Default 1 минута


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
