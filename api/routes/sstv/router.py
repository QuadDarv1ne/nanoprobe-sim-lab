"""
SSTV Router - обратная совместимость

Этот файл обеспечивает обратную совместимость для кода,
который импортирует router из api.routes.sstv.

Все endpoint'ы теперь находятся в отдельных модулях:
- health: Health checks и диагностика
- recording: Запись и управление записями
- satellites: Спутники и TLE
- websocket: WebSocket для real-time данных
"""

from fastapi import APIRouter, BackgroundTasks, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from api.error_handlers import NotFoundError, ServiceUnavailableError, ValidationError
from api.routes.sstv.health import router as health_router
from api.routes.sstv.recording import router as recording_router
from api.routes.sstv.satellites import router as satellites_router
from api.routes.sstv.websocket import router as websocket_router

# Создаём единый роутер для обратной совместимости
router = APIRouter()

# Подключаем все модули
router.include_router(health_router)
router.include_router(recording_router)
router.include_router(satellites_router)
router.include_router(websocket_router)

# Импортируем helpers для использования
from api.routes.sstv.helpers import (  # noqa: F401
    REDIS_AVAILABLE,
    SSTV_AVAILABLE,
    get_redis_cache,
    get_satellite_tracker,
    get_sstv_decoder,
    tracker_module,
)

# SSTV decode/download endpoints (из recording модуля)
from api.routes.sstv.recording import (  # noqa: F401
    decode_sstv_audio,
    download_sstv_image,
    get_sstv_modes,
)

__all__ = ["router"]
