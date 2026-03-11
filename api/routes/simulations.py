# -*- coding: utf-8 -*-
"""
API роуты для управления симуляциями
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional
from datetime import datetime
import uuid

from api.schemas import (
    SimulationCreate,
    SimulationResponse,
    SimulationListResponse,
    SimulationStatus,
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
    response_model=SimulationListResponse,
    summary="Получить список симуляций",
)
async def get_simulations(
    status: Optional[str] = Query(None, description="Фильтр по статусу"),
    limit: int = Query(50, ge=1, le=500),
    db: DatabaseManager = Depends(get_db),
):
    """Получить список симуляций"""
    from api.main import redis_cache
    
    cache_key = f"simulations:{status or 'all'}:{limit}"
    
    if redis_cache and redis_cache.is_available():
        cached = redis_cache.get(cache_key)
        if cached:
            return SimulationListResponse(**cached)
    
    simulations = db.get_simulations(status=status, limit=limit)
    
    result = SimulationListResponse(
        items=[SimulationResponse.model_validate(sim) for sim in simulations],
        total=len(simulations),
        limit=limit,
    )
    
    if redis_cache and redis_cache.is_available():
        redis_cache.set(cache_key, result.model_dump(), expire=300)
    
    return result


@router.get(
    "/{simulation_id}",
    response_model=SimulationResponse,
    summary="Получить симуляцию по ID",
)
async def get_simulation(
    simulation_id: str,
    db: DatabaseManager = Depends(get_db),
):
    """Получить симуляцию по ID"""
    from api.main import redis_cache
    
    cache_key = f"simulation:{simulation_id}"
    
    if redis_cache and redis_cache.is_available():
        cached = redis_cache.get(cache_key)
        if cached:
            return SimulationResponse(**cached)
    
    simulations = db.get_simulations(limit=100)
    sim = next((s for s in simulations if s.get('simulation_id') == simulation_id), None)
    
    if not sim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Симуляция с ID {simulation_id} не найдена",
        )
    
    result = SimulationResponse.model_validate(sim)
    
    if redis_cache and redis_cache.is_available():
        redis_cache.set(cache_key, result.model_dump(), expire=600)
    
    return result


@router.post(
    "",
    response_model=SimulationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Создать симуляцию",
)
async def create_simulation(
    simulation: SimulationCreate,
    db: DatabaseManager = Depends(get_db),
):
    """Создать новую симуляцию"""
    from api.main import redis_cache
    
    sim_id = f"sim_{uuid.uuid4().hex[:8]}"

    db.add_simulation(
        simulation_id=sim_id,
        simulation_type=simulation.simulation_type,
        parameters=simulation.parameters,
    )

    # Инвалидация кэша
    if redis_cache and redis_cache.is_available():
        redis_cache.clear_pattern("simulations:*")

    simulations = db.get_simulations(limit=1)
    if not simulations:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не удалось получить созданную симуляцию",
        )

    return SimulationResponse.model_validate(simulations[0])


@router.patch(
    "/{simulation_id}",
    response_model=SimulationResponse,
    summary="Обновить статус симуляции",
)
async def update_simulation(
    simulation_id: str,
    status: str = Query(..., description="Новый статус"),
    db: DatabaseManager = Depends(get_db),
):
    """Обновить статус симуляции"""
    from api.main import redis_cache
    
    db.update_simulation(
        simulation_id=simulation_id,
        status=status,
    )

    # Инвалидация кэша
    if redis_cache and redis_cache.is_available():
        redis_cache.clear_pattern("simulations:*")
        redis_cache.delete(f"simulation:{simulation_id}")

    simulations = db.get_simulations(limit=100)
    sim = next((s for s in simulations if s.get('simulation_id') == simulation_id), None)

    if not sim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Симуляция с ID {simulation_id} не найдена",
        )

    return SimulationResponse.model_validate(sim)
