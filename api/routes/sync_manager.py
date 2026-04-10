"""
Sync Manager Router - эндпоинты для управления синхронизацией Backend ↔ Frontend
"""

from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter

router = APIRouter()

# Глобальное состояние sync manager
_sync_manager_instance: Optional[Any] = None


def set_sync_manager(manager):
    """Установить экземпляр Sync Manager"""
    global _sync_manager_instance
    _sync_manager_instance = manager


def get_sync_manager():
    """Получить экземпляр Sync Manager"""
    return _sync_manager_instance


@router.get("/api/v1/sync/status", tags=["Sync Manager"])
async def get_sync_status():
    """
    Получить статус синхронизации Backend ↔ Frontend

    Возвращает:
    - running: запущен ли sync manager
    - backend_url: URL Backend
    - frontend_url: URL Frontend
    - last_sync_time: время последней синхронизации
    - backend_connections: количество подключений к Backend
    - frontend_connections: количество подключений к Frontend
    """
    manager = get_sync_manager()

    if manager:
        status = manager.get_sync_status()
        return {"status": "ok", **status}
    else:
        return {
            "status": "standby",
            "running": False,
            "backend_url": "http://localhost:8000",
            "frontend_url": "http://localhost:5000",
            "last_sync_time": None,
            "backend_connections": 0,
            "frontend_connections": 0,
            "message": "Sync Manager не запущен",
        }


@router.get("/api/v1/sync/health", tags=["Sync Manager"])
async def get_sync_health():
    """
    Проверка здоровья Sync Manager
    """
    manager = get_sync_manager()

    health = {"status": "unknown", "timestamp": datetime.now(timezone.utc).isoformat()}

    if manager:
        status = manager.get_sync_status()
        if status.get("running"):
            health["status"] = "healthy"
            health["running"] = True
            health["last_sync_time"] = status.get("last_sync_time")
        else:
            health["status"] = "standby"
            health["running"] = False
    else:
        health["status"] = "not_available"
        health["message"] = "Sync Manager не инициализирован"

    return health
