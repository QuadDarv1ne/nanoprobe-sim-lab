"""
SSTV API модули

Разбитие большого sstv.py файла на отдельные модули:
- health: Health checks и диагностика
- recording: Запись и управление записями
- satellites: Спутники и TLE
- websocket: WebSocket для real-time данных
- helpers: Общие утилиты и хелперы
"""

from fastapi import APIRouter

from api.routes.sstv.health import router as health_router
from api.routes.sstv.helpers import (
    REDIS_AVAILABLE,
    SSTV_AVAILABLE,
    get_redis_cache,
    get_satellite_tracker,
    get_sstv_decoder,
    tracker_module,
)
from api.routes.sstv.recording import get_recording_status, list_recordings
from api.routes.sstv.recording import router as recording_router
from api.routes.sstv.recording import start_sstv_recording, stop_sstv_recording
from api.routes.sstv.satellites import router as satellites_router
from api.routes.sstv.websocket import router as websocket_router

# Создаём единый роутер для обратной совместимости
router = APIRouter()
router.include_router(health_router)
router.include_router(recording_router)
router.include_router(satellites_router)
router.include_router(websocket_router)

__all__ = [
    "router",
    "health_router",
    "recording_router",
    "satellites_router",
    "websocket_router",
    "REDIS_AVAILABLE",
    "SSTV_AVAILABLE",
    "get_redis_cache",
    "get_satellite_tracker",
    "get_sstv_decoder",
    "tracker_module",
    "start_sstv_recording",
    "stop_sstv_recording",
    "get_recording_status",
    "list_recordings",
]
