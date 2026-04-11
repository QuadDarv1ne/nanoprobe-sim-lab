"""
Dashboard alerts endpoints

Алерты, конфигурация, история метрик, процессы.
"""

import logging
from pathlib import Path
from typing import Optional

import psutil
from fastapi import APIRouter, Query

from api.error_handlers import DatabaseError, ServiceUnavailableError
from api.routes.dashboard.helpers import get_project_root, get_system_disk_usage
from utils.caching.redis_cache import cached
from utils.monitoring.monitoring import get_monitor

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/alerts/config")
async def get_alerts_config():
    """Get alerts configuration."""
    return {
        "thresholds": {
            "cpu_warning": 70,
            "cpu_critical": 90,
            "memory_warning": 70,
            "memory_critical": 90,
            "disk_warning": 80,
            "disk_critical": 95,
        },
        "notifications": {"enabled": True, "channels": ["ui", "log"]},
    }


@router.get("/alerts/check")
@cached(prefix="dashboard:alerts", expire=5)
async def check_alerts():
    """Check current system alerts (cached for 5 seconds)."""
    try:
        alerts = []
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = get_system_disk_usage()

        if cpu_percent >= 90:
            alerts.append(
                {
                    "level": "critical",
                    "type": "cpu",
                    "message": f"Critical CPU usage: {cpu_percent}%",
                    "value": cpu_percent,
                    "threshold": 90,
                }
            )
        elif cpu_percent >= 70:
            alerts.append(
                {
                    "level": "warning",
                    "type": "cpu",
                    "message": f"High CPU usage: {cpu_percent}%",
                    "value": cpu_percent,
                    "threshold": 70,
                }
            )

        if memory.percent >= 90:
            alerts.append(
                {
                    "level": "critical",
                    "type": "memory",
                    "message": f"Critical memory usage: {memory.percent}%",
                    "value": memory.percent,
                    "threshold": 90,
                }
            )
        elif memory.percent >= 70:
            alerts.append(
                {
                    "level": "warning",
                    "type": "memory",
                    "message": f"High memory usage: {memory.percent}%",
                    "value": memory.percent,
                    "threshold": 70,
                }
            )

        if disk.percent >= 95:
            alerts.append(
                {
                    "level": "critical",
                    "type": "disk",
                    "message": f"Critical disk usage: {disk.percent}%",
                    "value": disk.percent,
                    "threshold": 95,
                }
            )
        elif disk.percent >= 80:
            alerts.append(
                {
                    "level": "warning",
                    "type": "disk",
                    "message": f"High disk usage: {disk.percent}%",
                    "value": disk.percent,
                    "threshold": 80,
                }
            )

        return {
            "active_alerts": len(alerts),
            "alerts": alerts,
            "status": ("critical" if any(a["level"] == "critical" for a in alerts) else "ok"),
        }
    except Exception as e:
        logger.error(f"Error checking alerts: {e}")
        raise ServiceUnavailableError(f"Не удалось проверить алерты: {str(e)}")


@router.get(
    "/metrics/history",
    summary="История метрик",
    description="Получить историю метрик за период",
)
async def get_metrics_history(
    limit: int = Query(60, ge=1, le=300, description="Количество записей"),
    component: Optional[str] = Query(None, description="Фильтр по компоненту"),
):
    """Получить историю метрик."""
    try:
        monitor = get_monitor()
        history = monitor.get_metrics_history(limit=limit)

        if component:
            filtered_history = []
            for entry in history:
                if component == "cpu" and "cpu_percent" in entry:
                    filtered_history.append(
                        {
                            "timestamp": entry["timestamp"],
                            "value": entry["cpu_percent"],
                        }
                    )
                elif component == "memory" and "memory_percent" in entry:
                    filtered_history.append(
                        {
                            "timestamp": entry["timestamp"],
                            "value": entry["memory_percent"],
                        }
                    )
                elif component == "disk" and "disk_percent" in entry:
                    filtered_history.append(
                        {
                            "timestamp": entry["timestamp"],
                            "value": entry["disk_percent"],
                        }
                    )
            return {"component": component, "history": filtered_history}

        return {"history": history}
    except Exception as e:
        logger.error(f"Error getting metrics history: {e}")
        raise DatabaseError(f"Ошибка получения истории: {str(e)}")


@router.get(
    "/alerts",
    summary="Получить алерты",
    description="Получить список алертов системы",
)
async def get_alerts(
    limit: int = Query(50, ge=1, le=200, description="Количество алертов"),
    level: Optional[str] = Query(None, description="Фильтр по уровню"),
):
    """Получить алерты системы."""
    try:
        monitor = get_monitor()
        alerts = monitor.get_alerts(limit=limit, level=level)
        return {"alerts": alerts, "total": len(alerts)}
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise DatabaseError(f"Ошибка получения алертов: {str(e)}")


@router.get(
    "/processes",
    summary="Топ процессов",
    description="Получить топ процессов по использованию ресурсов",
)
async def get_top_processes(
    limit: int = Query(10, ge=1, le=50, description="Количество процессов"),
    sort_by: str = Query("cpu", description="Сортировка (cpu/memory)"),
):
    """Получить топ процессов."""
    try:
        monitor = get_monitor()
        processes = monitor.get_process_list(limit=limit, sort_by=sort_by)
        return {"processes": processes, "total": len(processes)}
    except Exception as e:
        logger.error(f"Error getting processes: {e}")
        raise DatabaseError(f"Ошибка получения процессов: {str(e)}")
