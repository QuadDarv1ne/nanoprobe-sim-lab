"""SSTV API Router - Unified

Объединяет все SSTV endpoint'ы из отдельных модулей:
- health: Health checks и диагностика устройств
- recording: Управление записью RTL-SDR
- satellites: Спутники, расписание, TLE
- calibration: Автоматическая PPM калибровка
- websocket: Real-time данные

Обратная совместимость: все старые URL'ы работают.
"""

from fastapi import APIRouter

from api.routes.sstv.calibration import router as calibration_router
from api.routes.sstv.health import router as health_router
from api.routes.sstv.recording import get_recording_status, list_recordings
from api.routes.sstv.recording import router as recording_router
from api.routes.sstv.recording import start_sstv_recording, stop_sstv_recording
from api.routes.sstv.satellites import router as satellites_router
from api.routes.sstv.websocket import router as websocket_router

router = APIRouter()

# Подключаем все модули
router.include_router(health_router)
router.include_router(recording_router)
router.include_router(satellites_router)
router.include_router(calibration_router)
router.include_router(websocket_router)

# Экспорт для тестов и прямого импорта
__all__ = [
    "router",
    "start_sstv_recording",
    "stop_sstv_recording",
    "get_recording_status",
    "list_recordings",
]
