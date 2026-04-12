"""
SSTV Satellites endpoints

Спутники, расписание МКС, TLE данные,
позиция спутников.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Query

from api.error_handlers import NotFoundError, ServiceUnavailableError
from api.routes.sstv.helpers import REDIS_AVAILABLE, get_redis_cache, get_satellite_tracker

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/iss/schedule")
async def get_iss_schedule(
    hours_ahead: int = Query(default=24, ge=1, le=72),
    min_elevation: float = Query(default=10.0, ge=0, le=90),
):
    """Получает расписание пролётов МКС (ISS)."""
    tracker = get_satellite_tracker()
    if not tracker:
        raise ServiceUnavailableError("Satellite tracker недоступен")

    cache_key = f"iss_schedule:{hours_ahead}:{min_elevation}"
    redis_cache = get_redis_cache()

    if redis_cache and REDIS_AVAILABLE:
        cached = redis_cache.get(cache_key)
        if cached:
            return {"status": "success", "cached": True, "data": cached}

    try:
        passes = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: tracker.get_pass_predictions(
                satellite_name="iss",
                hours_ahead=hours_ahead,
                min_elevation=min_elevation,
            ),
        )

        result = []
        for pass_info in passes:
            result.append(
                {
                    "aos": pass_info["aos"].isoformat(),
                    "los": pass_info["los"].isoformat(),
                    "max_elevation": pass_info["max_elevation"],
                    "frequency_mhz": pass_info["frequency"],
                    "duration_minutes": pass_info["duration_minutes"],
                    "mode": ("SSTV Martin 1" if pass_info["frequency"] == 145.800 else "Unknown"),
                }
            )

        if redis_cache and REDIS_AVAILABLE:
            redis_cache.set(cache_key, result, expire=300)

        return {
            "status": "success",
            "cached": False,
            "count": len(result),
            "data": result,
        }

    except Exception as e:
        logger.exception("Failed to get ISS schedule: %s", e)
        raise ServiceUnavailableError("Не удалось получить расписание МКС")


@router.get("/iss/next-pass")
async def get_iss_next_pass(
    min_elevation: float = Query(default=10.0, ge=0, le=90),
):
    """Получает следующий пролёт МКС."""
    tracker = get_satellite_tracker()
    if not tracker:
        raise ServiceUnavailableError("Satellite tracker недоступен")

    cache_key = "iss_next_pass"
    redis_cache = get_redis_cache()

    if redis_cache and REDIS_AVAILABLE:
        cached = redis_cache.get(cache_key)
        if cached:
            return {"status": "success", "cached": True, "data": cached}

    try:
        next_pass = await asyncio.get_event_loop().run_in_executor(
            None, lambda: tracker.get_next_pass("iss")
        )

        if not next_pass:
            return {
                "status": "success",
                "message": "No passes found in next 24 hours",
                "data": None,
            }

        result = {
            "aos": next_pass["aos"].isoformat(),
            "los": next_pass["los"].isoformat(),
            "max_elevation": round(next_pass["max_elevation"], 1),
            "frequency_mhz": next_pass["frequency"],
            "duration_minutes": round(next_pass["duration_minutes"], 1),
            "time_until_aos": next_pass.get(
                "time_until_aos",
                str(next_pass["aos"] - datetime.now(timezone.utc)),
            ),
        }

        if redis_cache and REDIS_AVAILABLE:
            redis_cache.set(cache_key, result, expire=120)

        return {"status": "success", "cached": False, "data": result}

    except Exception as e:
        logger.exception("Failed to get ISS next pass: %s", e)
        raise ServiceUnavailableError("Не удалось получить данные о пролёте МКС")


@router.get("/iss/position")
async def get_iss_current_position():
    """Получает текущую позицию МКС."""
    tracker = get_satellite_tracker()
    if not tracker:
        raise ServiceUnavailableError("Satellite tracker недоступен")

    cache_key = "iss_position"
    redis_cache = get_redis_cache()

    if redis_cache and REDIS_AVAILABLE:
        cached = redis_cache.get(cache_key)
        if cached:
            return {"status": "success", "cached": True, "data": cached}

    try:
        position = tracker.get_current_position("iss")

        if not position:
            raise NotFoundError("Позиция МКС недоступна")

        result = {
            "latitude": position["latitude"],
            "longitude": position["longitude"],
            "altitude_km": position["altitude_km"],
            "velocity_kmh": position["velocity_kmh"],
            "footprint_km": position["footprint_km"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if redis_cache and REDIS_AVAILABLE:
            redis_cache.set(cache_key, result, expire=30)

        return {"status": "success", "cached": False, "data": result}

    except Exception as e:
        logger.exception("Failed to get ISS position: %s", e)
        raise ServiceUnavailableError("Не удалось получить позицию МКС")


@router.get("/iss/visible")
async def is_iss_visible(
    min_elevation: float = Query(default=10.0, ge=0, le=90),
):
    """Проверяет видимость МКС сейчас."""
    tracker = get_satellite_tracker()
    if not tracker:
        raise ServiceUnavailableError("Satellite tracker недоступен")

    try:
        visible = tracker.is_satellite_visible("iss", min_elevation)
        position = tracker.get_current_position("iss")

        elevation = 0
        if position:
            elevation = tracker._elevation_from_position(position, datetime.now(timezone.utc))

        return {
            "status": "success",
            "visible": visible,
            "elevation": elevation,
            "message": "ISS видна" if visible else "ISS не видна",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.exception("Failed to check ISS visibility: %s", e)
        raise ServiceUnavailableError("Не удалось проверить видимость МКС")


@router.get("/satellites")
async def get_all_satellites():
    """Получает список всех отслеживаемых спутников."""
    tracker = get_satellite_tracker()
    if not tracker:
        raise ServiceUnavailableError("Satellite tracker недоступен")

    satellites = tracker.get_all_satellites()

    return {
        "status": "success",
        "count": len(satellites),
        "satellites": satellites,
    }


@router.get("/satellites/schedule")
async def get_all_satellites_schedule(hours_ahead: int = 24):
    """Получает расписание всех SSTV спутников."""
    tracker = get_satellite_tracker()
    if not tracker:
        raise ServiceUnavailableError("Satellite tracker недоступен")

    hours_ahead = min(hours_ahead, 72)

    try:
        schedule = await asyncio.get_event_loop().run_in_executor(
            None, lambda: tracker.get_sstv_schedule(hours_ahead)
        )

        result = []
        for pass_info in schedule:
            result.append(
                {
                    "satellite": pass_info["satellite"],
                    "aos": pass_info["aos"].isoformat(),
                    "los": pass_info["los"].isoformat(),
                    "max_elevation": pass_info["max_elevation"],
                    "frequency_mhz": pass_info["frequency"],
                    "duration_minutes": pass_info["duration_minutes"],
                }
            )

        return {
            "status": "success",
            "count": len(result),
            "data": result,
        }

    except Exception as e:
        logger.exception("Failed to get satellite schedule: %s", e)
        raise ServiceUnavailableError("Не удалось получить расписание спутников")


@router.post("/tle/refresh")
async def refresh_tle():
    """Принудительное обновление TLE данных с CelesTrak."""
    tracker = get_satellite_tracker()
    if not tracker:
        raise ServiceUnavailableError("Satellite tracker недоступен")

    try:
        updated = tracker.update_tle_from_celestrak()
        if updated > 0:
            tracker.save_tle("data/tle_data.json")
            from api.routes.sstv.helpers import set_app_state

            set_app_state("satellite_tracker", None)

        return {
            "status": "success",
            "updated": updated,
            "message": f"Обновлено TLE: {updated} спутников",
        }
    except Exception as e:
        raise ServiceUnavailableError(f"Ошибка обновления TLE: {str(e)}")


@router.get("/tle/status")
async def get_tle_status():
    """Статус TLE данных (возраст, источник)."""
    tle_file = Path("data/tle_data.json")

    if tle_file.exists():
        age_hours = (time.time() - tle_file.stat().st_mtime) / 3600
        return {
            "status": "cached",
            "age_hours": round(age_hours, 2),
            "fresh": age_hours < 12,
            "file": str(tle_file),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    return {
        "status": "builtin",
        "age_hours": None,
        "fresh": False,
        "message": "Используются встроенные TLE, рекомендуется обновить",
    }
