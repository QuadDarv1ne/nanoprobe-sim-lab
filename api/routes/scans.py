# -*- coding: utf-8 -*-
"""
API роуты для управления сканированиями
CRUD операции для результатов сканирований
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional
from datetime import datetime

from api.schemas import (
    ScanCreate,
    ScanResponse,
    ScanListResponse,
    ErrorResponse,
)
from utils.database import DatabaseManager


router = APIRouter()


def get_db() -> DatabaseManager:
    """Зависимость для получения менеджера БД"""
    from api.main import db_manager
    if db_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="База данных недоступна"
        )
    return db_manager


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
    limit: int = Query(100, ge=1, le=1000, description="Лимит записей"),
    offset: int = Query(0, ge=0, description="Смещение"),
    db: DatabaseManager = Depends(get_db),
):
    """Получить список сканирований"""
    from api.main import redis_cache
    
    # Генерация ключа кэша
    cache_key = f"scans:{scan_type or 'all'}:{limit}:{offset}"
    
    # Попытка получить из кэша
    if redis_cache and redis_cache.is_available():
        cached_result = redis_cache.get(cache_key)
        if cached_result:
            return ScanListResponse(**cached_result)
    
    # Получение данных из БД
    scans = db.get_scan_results(scan_type=scan_type, limit=limit, offset=offset)
    
    # Получение общего количества
    stats = db.get_statistics()
    total = stats.get('total_scans', 0)
    
    result = ScanListResponse(
        items=[ScanResponse.model_validate(scan) for scan in scans],
        total=total,
        limit=limit,
        offset=offset,
    )
    
    # Сохранение в кэш (5 минут)
    if redis_cache and redis_cache.is_available():
        redis_cache.set(
            cache_key, 
            result.model_dump(), 
            expire=300
        )
    
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
    
    cache_key = f"scan:{scan_id}"
    
    # Проверка кэша
    if redis_cache and redis_cache.is_available():
        cached = redis_cache.get(cache_key)
        if cached:
            return ScanResponse(**cached)
    
    scan = db.get_scan_by_id(scan_id)
    
    if not scan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Сканирование с ID {scan_id} не найдено",
        )
    
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не удалось получить созданную запись",
        )

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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Сканирование с ID {scan_id} не найдено",
        )

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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка поиска: {str(e)}",
        )
