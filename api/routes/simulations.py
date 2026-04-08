"""
API роуты для управления симуляциями
"""

from fastapi import APIRouter, Depends, Query, status
from typing import Optional
import uuid

from api.schemas import (
    SimulationCreate,
    SimulationResponse,
    SimulationListResponse,
)
from api.dependencies import get_db, get_redis_cache
from api.error_handlers import NotFoundError, ValidationError
from api.state import get_redis
from utils.database import DatabaseManager
from utils.caching.redis_cache import RedisCache


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
    redis = get_redis()
    cache_key = f"simulation:{simulation_id}"

    if redis and redis.is_available():
        cached = redis.get(cache_key)
        if cached:
            return SimulationResponse(**cached)

    simulations = db.get_simulations(limit=100)
    sim = next((s for s in simulations if s.get('simulation_id') == simulation_id), None)

    if not sim:
        raise NotFoundError(f"Симуляция с ID {simulation_id} не найдена", resource_type="simulation")

    result = SimulationResponse.model_validate(sim)

    if redis and redis.is_available():
        redis.set(cache_key, result.model_dump(), expire=600)

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
    sim_id = f"sim_{uuid.uuid4().hex[:8]}"

    db.add_simulation(
        simulation_id=sim_id,
        simulation_type=simulation.simulation_type,
        parameters=simulation.parameters,
    )

    redis = get_redis()
    if redis and redis.is_available():
        redis.clear_pattern("simulations:*")
        redis.clear_pattern("dashboard:*")

    # Получаем именно созданную запись по simulation_id
    simulations = db.get_simulations(limit=500)
    sim = next((s for s in simulations if s.get('simulation_id') == sim_id), None)
    if not sim:
        raise NotFoundError("Не удалось получить созданную симуляцию", resource_type="simulation")

    return SimulationResponse.model_validate(sim)


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
    db.update_simulation(simulation_id=simulation_id, status=status)

    redis = get_redis()
    if redis and redis.is_available():
        redis.clear_pattern("simulations:*")
        redis.clear_pattern("dashboard:*")
        redis.delete(f"simulation:{simulation_id}")

    simulations = db.get_simulations(limit=100)
    sim = next((s for s in simulations if s.get('simulation_id') == simulation_id), None)

    if not sim:
        raise NotFoundError(f"Симуляция с ID {simulation_id} не найдена", resource_type="simulation")

    return SimulationResponse.model_validate(sim)


@router.post(
    "/{simulation_id}/stop",
    response_model=SimulationResponse,
    summary="Остановить симуляцию",
)
async def stop_simulation(
    simulation_id: str,
    db: DatabaseManager = Depends(get_db),
):
    """Остановить запущенную симуляцию (принимает числовой id или simulation_id UUID)"""
    simulations = db.get_simulations(limit=500)

    # Ищем по simulation_id (UUID) или по числовому id
    sim = next(
        (s for s in simulations if
         s.get('simulation_id') == simulation_id or str(s.get('id')) == simulation_id),
        None
    )

    if not sim:
        raise NotFoundError(f"Симуляция с ID {simulation_id} не найдена", resource_type="simulation")

    if sim.get('status') not in ('running', 'pending'):
        raise ValidationError(f"Симуляция уже завершена (статус: {sim.get('status')})")

    real_sim_id = sim['simulation_id']
    db.update_simulation(simulation_id=real_sim_id, status='stopped')

    redis = get_redis()
    if redis and redis.is_available():
        redis.clear_pattern("simulations:*")
        redis.clear_pattern("dashboard:*")
        redis.delete(f"simulation:{real_sim_id}")

    simulations = db.get_simulations(limit=500)
    sim = next((s for s in simulations if s.get('simulation_id') == real_sim_id), None)
    return SimulationResponse.model_validate(sim)


@router.delete(
    "/{simulation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Удалить симуляцию",
)
async def delete_simulation(
    simulation_id: str,
    db: DatabaseManager = Depends(get_db),
):
    """Удалить симуляцию из БД (принимает числовой id или simulation_id UUID)"""
    with db.get_connection() as conn:
        cursor = conn.cursor()
        # Пробуем удалить по simulation_id (UUID) или по числовому id
        cursor.execute(
            "DELETE FROM simulations WHERE simulation_id = ? OR CAST(id AS TEXT) = ?",
            (simulation_id, simulation_id)
        )
        if cursor.rowcount == 0:
            raise NotFoundError(f"Симуляция с ID {simulation_id} не найдена", resource_type="simulation")

    redis = get_redis()
    if redis and redis.is_available():
        redis.clear_pattern("simulations:*")
        redis.clear_pattern("dashboard:*")
        redis.delete(f"simulation:{simulation_id}")
