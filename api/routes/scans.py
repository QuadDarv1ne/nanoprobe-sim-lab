"""
API роуты для управления сканированиями
CRUD операции для результатов сканирований
"""

from fastapi import APIRouter, Depends, Query, Response
from typing import Optional

from api.schemas import (
    ScanCreate,
    ScanResponse,
    ScanListResponse,
    ErrorResponse,
)
from api.dependencies import get_db, get_redis_cache
from api.error_handlers import NotFoundError, DatabaseError
from utils.redis_cache import RedisCache
from utils.database import DatabaseManager


router = APIRouter()


@router.get(
    "",
    response_model=ScanListResponse,
    summary="Получить список сканирований",
    description="Возвращает список результатов сканирований с пагинацией",
    responses={
        200: {"description": "Успешный ответ"},
        400: {"model": ErrorResponse},
    },
)
async def get_scans(
    scan_type: Optional[str] = Query(None, description="Фильтр по типу сканирования"),
    page: int = Query(1, ge=1, description="Номер страницы"),
    page_size: int = Query(20, ge=1, le=100, description="Размер страницы"),
    db: DatabaseManager = Depends(get_db),
    redis_cache: RedisCache = Depends(get_redis_cache),
):
    """Получить список сканирований с пагинацией по страницам"""
    from api.metrics import BusinessMetrics

    offset = (page - 1) * page_size
    cache_key = f"scans:{scan_type or 'all'}:{page}:{page_size}"

    if redis_cache and redis_cache.is_available():
        cached_result = redis_cache.get(cache_key)
        if cached_result:
            BusinessMetrics.inc_cache_hit("scans")
            return ScanListResponse(**cached_result)
        BusinessMetrics.inc_cache_miss("scans")

    scans = db.get_scan_results(scan_type=scan_type, limit=page_size, offset=offset)
    total = db.count_scans(scan_type)

    result = ScanListResponse(
        items=[ScanResponse.model_validate(scan) for scan in scans],
        total=total,
        limit=page_size,
        offset=offset,
    )

    if redis_cache and redis_cache.is_available():
        redis_cache.set(cache_key, result.model_dump(), expire=300)

    return result


@router.get(
    "/{scan_id}",
    response_model=ScanResponse,
    summary="Получить сканирование по ID",
    description="Возвращает детали конкретного сканирования",
    responses={
        200: {"description": "Успешный ответ"},
        404: {"model": ErrorResponse, "description": "Сканирование не найдено"},
    },
)
async def get_scan(
    scan_id: int,
    db: DatabaseManager = Depends(get_db),
):
    """Получить сканирование по ID"""
    from api.main import redis_cache
    from api.metrics import BusinessMetrics

    cache_key = f"scan:{scan_id}"

    # Проверка кэша
    if redis_cache and redis_cache.is_available():
        cached = redis_cache.get(cache_key)
        if cached:
            BusinessMetrics.inc_cache_hit("scan_detail")
            return ScanResponse(**cached)
        BusinessMetrics.inc_cache_miss("scan_detail")

    scan = db.get_scan_by_id(scan_id)

    if not scan:
        raise NotFoundError(f"Сканирование с ID {scan_id} не найдено", resource_type="scan")

    result = ScanResponse.model_validate(scan)

    # Сохранение в кэш
    if redis_cache and redis_cache.is_available():
        redis_cache.set(cache_key, result.model_dump(), expire=600)

    return result


@router.post(
    "",
    response_model=ScanResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Создать сканирование",
    description="Создаёт новую запись о сканировании",
    responses={
        201: {"description": "Сканирование создано"},
        400: {"model": ErrorResponse, "description": "Ошибка валидации"},
    },
)
async def create_scan(
    scan: ScanCreate,
    db: DatabaseManager = Depends(get_db),
):
    """Создать новое сканирование"""
    from api.main import redis_cache
    from api.metrics import BusinessMetrics

    scan_id = db.add_scan_result(
        scan_type=scan.scan_type.value,
        surface_type=scan.surface_type,
        width=scan.width,
        height=scan.height,
        metadata=scan.metadata,
    )

    # Получение созданной записи
    scans = db.get_scan_results(limit=1)
    if not scans:
        raise NotFoundError("Не удалось получить созданную запись", resource_type="scan")

    # Бизнес-метрики
    BusinessMetrics.inc_scan_created(scan.scan_type.value)

    # Инвалидация кэша
    if redis_cache and redis_cache.is_available():
        redis_cache.clear_pattern("scans:*")

    return ScanResponse.model_validate(scans[0])


@router.delete(
    "/{scan_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Удалить сканирование",
    description="Удаляет запись о сканировании",
    responses={
        204: {"description": "Успешно удалено"},
        404: {"model": ErrorResponse, "description": "Сканирование не найдено"},
    },
)
async def delete_scan(
    scan_id: int,
    db: DatabaseManager = Depends(get_db),
):
    """Удалить сканирование"""
    from api.main import redis_cache

    success = db.delete_scan(scan_id)

    if not success:
        raise NotFoundError(f"Сканирование с ID {scan_id} не найдено", resource_type="scan")

    # Инвалидация кэша
    if redis_cache and redis_cache.is_available():
        redis_cache.clear_pattern("scans:*")

    return None


@router.get(
    "/search/{query}",
    response_model=ScanListResponse,
    summary="Поиск сканирований",
    description="Поиск по результатам сканирований",
)
async def search_scans(
    query: str,
    limit: int = Query(50, ge=1, le=500),
    db: DatabaseManager = Depends(get_db),
):
    """Поиск сканирований"""
    try:
        scans = db.search_scans(query=query, limit=limit)

        return ScanListResponse(
            items=[ScanResponse.model_validate(scan) for scan in scans],
            total=len(scans),
            limit=limit,
            offset=0,
        )
    except Exception as e:
        raise DatabaseError(f"Ошибка поиска сканирований: {str(e)}")
