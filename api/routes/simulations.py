"""
API роуты для управления симуляциями
"""

from fastapi import APIRouter, Depends, Query, status
from typing import List, Optional
import uuid

from api.schemas import (
    SimulationCreate,
    SimulationResponse,
    SimulationListResponse,
    SimulationStatus,
    ErrorResponse,
)
from api.dependencies import get_db, get_redis_cache
from api.error_handlers import NotFoundError
from utils.database import DatabaseManager
from utils.redis_cache import RedisCache


router = APIRouter()


@router.get(
    "",
    response_model=SimulationListResponse,
    summary="Получить список симуляций",
)
async def get_simulations(
    status: Optional[str] = Query(None, description="Фильтр по статусу"),
    limit: int = Query(50, ge=1, le=500),
    db: DatabaseManager = Depends(get_db),
    redis_cache: RedisCache = Depends(get_redis_cache),
):
    """Получить список симуляций"""
    from api.metrics import BusinessMetrics

    cache_key = f"simulations:{status or 'all'}:{limit}"

    if redis_cache and redis_cache.is_available():
        cached = redis_cache.get(cache_key)
        if cached:
            BusinessMetrics.inc_cache_hit("simulations")
            return SimulationListResponse(**cached)
        BusinessMetrics.inc_cache_miss("simulations")

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
        raise NotFoundError(f"Симуляция с ID {simulation_id} не найдена", resource_type="simulation")

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
        raise NotFoundError("Не удалось получить созданную симуляцию", resource_type="simulation")

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
        raise NotFoundError(f"Симуляция с ID {simulation_id} не найдена", resource_type="simulation")

    return SimulationResponse.model_validate(sim)
