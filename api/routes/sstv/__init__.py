"""
SSTV API модули

Разбитие большого sstv.py файла на отдельные модули:
- health: Health checks и диагностика
- recording: Запись и управление записями
- satellites: Спутники и TLE
- websocket: WebSocket для real-time данных
- helpers: Общие утилиты и хелперы
"""

from api.routes.sstv.health import router as health_router
from api.routes.sstv.recording import router as recording_router
from api.routes.sstv.satellites import router as satellites_router
from api.routes.sstv.websocket import router as websocket_router

__all__ = [
    "health_router",
    "recording_router",
    "satellites_router",
    "websocket_router",
]
