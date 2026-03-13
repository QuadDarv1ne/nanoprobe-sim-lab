# -*- coding: utf-8 -*-
"""
External Services API routes с Circuit Breaker
Интеграция с внешними сервисами через circuit breaker
"""

from fastapi import APIRouter, HTTPException, status, Query
from typing import Optional, Dict, Any
import logging
import requests
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/external", tags=["External Services"])


# Circuit breaker для внешних API
from utils.circuit_breaker import circuit_breaker, get_circuit_breaker


@router.get(
    "/nasa/apod",
    summary="NASA APOD",
    description="Astronomy Picture of the Day от NASA API",
)
@circuit_breaker(
    name="nasa_api",
    failure_threshold=3,
    recovery_timeout=120,
    fallback={"error": "NASA API unavailable", "fallback": True}
)
async def get_nasa_apod(date: Optional[str] = None):
    """
    Получение Astronomy Picture of the Day от NASA
    
    Args:
        date: Дата в формате YYYY-MM-DD (по умолчанию сегодня)
    
    Returns:
        Данные APOD
    """
    api_key = "DEMO_KEY"  # В production использовать env variable
    url = "https://api.nasa.gov/planetary/apod"
    params = {"api_key": api_key}
    
    if date:
        params["date"] = date
    
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    
    return response.json()


@router.get(
    "/zenodo/search",
    summary="Zenodo Search",
    description="Поиск научных данных на Zenodo",
)
@circuit_breaker(
    name="zenodo_api",
    failure_threshold=5,
    recovery_timeout=60,
    fallback={"hits": {"hits": [], "total": 0}, "fallback": True}
)
async def search_zenodo(
    query: str = Query(..., description="Поисковый запрос"),
    size: int = Query(10, ge=1, le=100, description="Количество результатов")
):
    """
    Поиск научных данных на Zenodo
    
    Args:
        query: Поисковый запрос
        size: Количество результатов
    
    Returns:
        Результаты поиска
    """
    url = "https://zenodo.org/api/records"
    params = {
        "q": query,
        "size": size,
        "sort": "bestmatch",
        "order": "desc"
    }
    
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    
    return response.json()


@router.get(
    "/figodo/search",
    summary="Figshare Search",
    description="Поиск данных на Figshare",
)
@circuit_breaker(
    name="figshare_api",
    failure_threshold=5,
    recovery_timeout=60,
    fallback={"items": [], "total": 0, "fallback": True}
)
async def search_figshare(
    query: str = Query(..., description="Поисковый запрос"),
    limit: int = Query(10, ge=1, le=100)
):
    """
    Поиск данных на Figshare
    
    Args:
        query: Поисковый запрос
        limit: Количество результатов
    
    Returns:
        Результаты поиска
    """
    url = "https://api.figshare.com/v2/articles"
    params = {
        "search": query,
        "limit": limit
    }
    
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    
    return response.json()


@router.get(
    "/circuit-breakers",
    summary="Статус Circuit Breakers",
    description="Получить статус всех circuit breakers",
)
async def get_circuit_breakers_status():
    """Статус всех circuit breakers для внешних сервисов"""
    from utils.circuit_breaker import get_all_circuit_breakers_stats
    return {
        "circuit_breakers": get_all_circuit_breakers_stats()
    }


@router.post(
    "/circuit-breakers/reset",
    summary="Сброс Circuit Breakers",
    description="Принудительный сброс всех circuit breakers",
)
async def reset_circuit_breakers():
    """Сброс всех circuit breakers"""
    from utils.circuit_breaker import reset_all_circuit_breakers
    reset_all_circuit_breakers()
    return {"success": True, "message": "All circuit breakers reset"}


@router.get(
    "/circuit-breaker/{name}",
    summary="Статус Circuit Breaker",
    description="Получить статус конкретного circuit breaker",
)
async def get_circuit_breaker_status(name: str):
    """Статус конкретного circuit breaker"""
    from utils.circuit_breaker import get_circuit_breaker
    
    try:
        breaker = get_circuit_breaker(name)
        return breaker.get_stats()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Circuit breaker '{name}' not found: {str(e)}"
        )


@router.post(
    "/circuit-breaker/{name}/reset",
    summary="Сброс Circuit Breaker",
    description="Принудительный сброс конкретного circuit breaker",
)
async def reset_circuit_breaker(name: str):
    """Сброс конкретного circuit breaker"""
    from utils.circuit_breaker import get_circuit_breaker
    
    try:
        breaker = get_circuit_breaker(name)
        breaker.reset()
        return {"success": True, "message": f"Circuit breaker '{name}' reset"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Circuit breaker '{name}' not found: {str(e)}"
        )


# Health check для внешних сервисов
@router.get(
    "/health",
    summary="Health Check External Services",
    description="Проверка доступности внешних сервисов",
)
async def check_external_services_health():
    """Проверка доступности внешних сервисов"""
    services = {
        "nasa": {"url": "https://api.nasa.gov", "status": "unknown"},
        "zenodo": {"url": "https://zenodo.org", "status": "unknown"},
        "figshare": {"url": "https://api.figshare.com", "status": "unknown"},
    }
    
    for name, service in services.items():
        try:
            response = requests.head(service["url"], timeout=5)
            if response.status_code < 400:
                service["status"] = "healthy"
            else:
                service["status"] = "degraded"
        except Exception as e:
            service["status"] = "unhealthy"
            service["error"] = str(e)
    
    return {"services": services, "timestamp": datetime.now().isoformat()}
