"""
NASA API Routes

Полная интеграция с NASA API:
- APOD (Astronomy Picture of the Day)
- Mars Rover Photos
- Near Earth Objects (Asteroids)
- Earth Imagery (EPIC)
- NASA Image Library
- EONET Natural Events

Требуется API ключ в .env: NASA_API_KEY=your_key
Получить ключ: https://api.nasa.gov/
"""

from fastapi import APIRouter, Query, Depends
from typing import Optional, List
from datetime import datetime, timedelta, timezone
import logging

from utils.api.nasa_api_client import get_nasa_client, NASAAPIClient
from utils.caching.redis_cache import cache
from utils.security.rate_limiter import rate_limit
from api.schemas import APODResponse, MarsPhotosResponse, NEOsResponse
from api.error_handlers import ExternalServiceError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/nasa", tags=["NASA API"])


# ==========================================
# APOD Endpoints
# ==========================================

@router.get(
    "/apod",
    summary="NASA APOD",
    description="Astronomy Picture of the Day - ежедневное изображение космоса",
    response_model=APODResponse,
)
@rate_limit(max_requests=30, window_seconds=60)
async def get_apod(
    date: Optional[str] = Query(None, description="Дата в формате YYYY-MM-DD"),
    count: Optional[int] = Query(None, ge=1, le=100, description="Количество случайных изображений"),
):
    """
    Получение Astronomy Picture of the Day.
    
    - **date**: Конкретная дата (по умолчанию сегодня)
    - **count**: Количество случайных изображений (1-100)
    """
    client = get_nasa_client()
    
    # Проверка кэша
    cache_key = f"nasa:apod:{date or 'today'}:{count or 'single'}"
    cached = cache.get(cache_key)
    if cached:
        cached["cached"] = True
        return cached
    
    try:
        result = await client.get_apod(date=date, count=count)
        
        # Кэширование на 1 час
        cache.set(cache_key, result, expire=3600)
        
        return result
    except Exception as e:
        logger.error(f"APOD fetch error: {e}")
        raise ExternalServiceError("NASA", f"Ошибка получения APOD: {str(e)}")


@router.get(
    "/apod/date-range",
    summary="NASA APOD Диапазон",
    description="Получение APOD за диапазон дат",
)
@rate_limit(max_requests=20, window_seconds=60)
async def get_apod_range(
    start_date: str = Query(..., description="Начальная дата YYYY-MM-DD"),
    end_date: str = Query(..., description="Конечная дата YYYY-MM-DD"),
):
    """Получение APOD за диапазон дат"""
    client = get_nasa_client()
    
    cache_key = f"nasa:apod:range:{start_date}:{end_date}"
    cached = cache.get(cache_key)
    if cached:
        cached["cached"] = True
        return cached
    
    try:
        result = await client.get_apod(start_date=start_date, end_date=end_date)
        cache.set(cache_key, result, expire=7200)  # 2 часа
        return result
    except Exception as e:
        raise ExternalServiceError("NASA", str(e))


# ==========================================
# Mars Rover Photos
# ==========================================

@router.get(
    "/mars/photos",
    summary="Mars Rover Photos",
    description="Фотографии с марсоходов NASA (Curiosity, Opportunity, Spirit, Perseverance)",
)
@rate_limit(max_requests=20, window_seconds=60)
async def get_mars_photos(
    sol: Optional[int] = Query(None, ge=0, description="Марсианский день миссии"),
    earth_date: Optional[str] = Query(None, description="Земная дата YYYY-MM-DD"),
    camera: Optional[str] = Query(None, description="Камера (FHAZ, RHAZ, MAHI, etc.)"),
    rover: Optional[str] = Query(None, description="Ровер (Curiosity, Opportunity, Spirit, Perseverance)"),
    page: int = Query(0, ge=0),
    per_page: int = Query(25, ge=1, le=100),
):
    """Получение фотографий с марсоходов"""
    client = get_nasa_client()
    
    cache_key = f"nasa:mars:{sol or 'any'}:{earth_date or 'any'}:{rover or 'all'}:{page}:{per_page}"
    cached = cache.get(cache_key)
    if cached:
        cached["cached"] = True
        return cached
    
    try:
        result = await client.get_mars_photos(
            sol=sol,
            earth_date=earth_date,
            camera=camera,
            rover=rover,
            page=page,
            per_page=per_page,
        )
        cache.set(cache_key, result, expire=3600)
        return result
    except Exception as e:
        raise ExternalServiceError("NASA", str(e))


@router.get(
    "/mars/rovers",
    summary="Mars Rover Manifest",
    description="Информация о всех марсоходах NASA",
)
@rate_limit(max_requests=30, window_seconds=60)
async def get_mars_rovers():
    """Получение информации о марсоходах"""
    client = get_nasa_client()
    
    cache_key = "nasa:mars:rovers"
    cached = cache.get(cache_key)
    if cached:
        cached["cached"] = True
        return cached
    
    try:
        result = await client.get_mars_rover_manifest()
        cache.set(cache_key, result, expire=86400)  # 24 часа
        return result
    except Exception as e:
        raise ExternalServiceError("NASA", str(e))


# ==========================================
# Near Earth Objects (Asteroids)
# ==========================================

@router.get(
    "/asteroids/feed",
    summary="Near Earth Objects",
    description="Данные о околоземных объектах (астероидах) за период",
)
@rate_limit(max_requests=20, window_seconds=60)
async def get_asteroids(
    start_date: Optional[str] = Query(None, description="Начальная дата YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="Конечная дата YYYY-MM-DD"),
    page: int = Query(0, ge=0),
    per_page: int = Query(25, ge=1, le=100),
):
    """Получение данных об астероидах"""
    client = get_nasa_client()
    
    # Даты по умолчанию - сегодня + 7 дней
    if not start_date:
        start_date = datetime.now().strftime("%Y-%m-%d")
    if not end_date:
        end_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
    
    cache_key = f"nasa:asteroids:{start_date}:{end_date}:{page}:{per_page}"
    cached = cache.get(cache_key)
    if cached:
        cached["cached"] = True
        return cached
    
    try:
        result = await client.get_asteroids(
            start_date=start_date,
            end_date=end_date,
            page=page,
            per_page=per_page,
        )
        cache.set(cache_key, result, expire=3600)
        return result
    except Exception as e:
        raise ExternalServiceError("NASA", str(e))


@router.get(
    "/asteroids/{asteroid_id}",
    summary="Asteroid by ID",
    description="Детальная информация об астероиде по ID",
)
@rate_limit(max_requests=30, window_seconds=60)
async def get_asteroid(asteroid_id: int):
    """Получение данных об астероиде по ID"""
    client = get_nasa_client()
    
    cache_key = f"nasa:asteroid:{asteroid_id}"
    cached = cache.get(cache_key)
    if cached:
        cached["cached"] = True
        return cached
    
    try:
        result = await client.get_asteroid_by_id(asteroid_id)
        cache.set(cache_key, result, expire=86400)  # 24 часа
        return result
    except Exception as e:
        raise ExternalServiceError("NASA", str(e))


# ==========================================
# Earth Imagery (EPIC)
# ==========================================

@router.get(
    "/earth/imagery",
    summary="Earth Imagery (EPIC)",
    description="Изображения Земли со спутника DSCOVR EPIC",
)
@rate_limit(max_requests=20, window_seconds=60)
async def get_earth_imagery(
    date: Optional[str] = Query(None, description="Дата YYYY-MM-DD"),
    lat: Optional[float] = Query(None, ge=-90, le=90, description="Широта"),
    lon: Optional[float] = Query(None, ge=-180, le=180, description="Долгота"),
):
    """Получение изображений Земли"""
    client = get_nasa_client()
    
    coordinates = None
    if lat is not None and lon is not None:
        coordinates = {"lat": lat, "lon": lon}
    
    cache_key = f"nasa:earth:{date or 'latest'}:{lat}:{lon}"
    cached = cache.get(cache_key)
    if cached:
        cached["cached"] = True
        return cached
    
    try:
        result = await client.get_earth_imagery(
            date=date,
            coordinates=coordinates,
        )
        cache.set(cache_key, result, expire=3600)
        return result
    except Exception as e:
        raise ExternalServiceError("NASA", str(e))


# ==========================================
# NASA Image Library
# ==========================================

@router.get(
    "/image-library/search",
    summary="NASA Image Library Search",
    description="Поиск в библиотеке изображений NASA (100,000+ изображений)",
)
@rate_limit(max_requests=30, window_seconds=60)
async def search_images(
    query: str = Query(..., description="Поисковый запрос"),
    media_type: Optional[str] = Query(None, description="Тип: image, video, audio"),
    year_start: Optional[int] = Query(None, ge=1900, le=2100),
    year_end: Optional[int] = Query(None, ge=1900, le=2100),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
):
    """Поиск в библиотеке изображений NASA"""
    client = get_nasa_client()
    
    cache_key = f"nasa:library:{query}:{media_type}:{page}:{page_size}"
    cached = cache.get(cache_key)
    if cached:
        cached["cached"] = True
        return cached
    
    try:
        result = await client.search_images(
            query=query,
            media_type=media_type,
            year_start=year_start,
            year_end=year_end,
            page=page,
            page_size=page_size,
        )
        cache.set(cache_key, result, expire=3600)
        return result
    except Exception as e:
        raise ExternalServiceError("NASA", str(e))


# ==========================================
# EONET - Natural Events
# ==========================================

@router.get(
    "/events/natural",
    summary="EONET Natural Events",
    description="Природные события: пожары, ураганы, извержения, etc.",
)
@rate_limit(max_requests=30, window_seconds=60)
async def get_natural_events(
    status: Optional[str] = Query(None, description="Статус: open, closed"),
    days: Optional[int] = Query(None, ge=1, le=365, description="Количество дней"),
    limit: int = Query(50, ge=1, le=200),
):
    """Получение данных о природных событиях"""
    client = get_nasa_client()
    
    cache_key = f"nasa:events:{status or 'all'}:{days or 'any'}:{limit}"
    cached = cache.get(cache_key)
    if cached:
        cached["cached"] = True
        return cached
    
    try:
        result = await client.get_natural_events(
            status=status,
            days=days,
            limit=limit,
        )
        cache.set(cache_key, result, expire=1800)  # 30 минут
        return result
    except Exception as e:
        raise ExternalServiceError("NASA", str(e))


@router.get(
    "/events/{event_id}",
    summary="EONET Event by ID",
    description="Детальная информация о природном событии",
)
@rate_limit(max_requests=30, window_seconds=60)
async def get_event(event_id: str):
    """Получение данных о событии по ID"""
    client = get_nasa_client()
    
    cache_key = f"nasa:event:{event_id}"
    cached = cache.get(cache_key)
    if cached:
        cached["cached"] = True
        return cached
    
    try:
        result = await client.get_event_by_id(event_id)
        cache.set(cache_key, result, expire=3600)
        return result
    except Exception as e:
        raise ExternalServiceError("NASA", str(e))


# ==========================================
# Health & Info
# ==========================================

@router.get(
    "/health",
    summary="NASA API Health Check",
    description="Проверка доступности NASA API",
)
async def health_check():
    """Проверка доступности NASA API"""
    client = get_nasa_client()
    
    cache_key = "nasa:health"
    cached = cache.get(cache_key)
    if cached:
        return cached
    
    try:
        is_healthy = await client.health_check()
        result = {
            "status": "healthy" if is_healthy else "unhealthy",
            "api": "NASA API",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if is_healthy:
            cache.set(cache_key, result, expire=300)  # 5 минут

        return result
    except Exception as e:
        return {
            "status": "error",
            "api": "NASA API",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
