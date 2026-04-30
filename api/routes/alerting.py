"""
API роуты для алертинга и мониторинга
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, Query

from api.alerting import AlertManager
from api.schemas import ErrorResponse

router = APIRouter()


def get_alert_manager() -> AlertManager:
    """Зависимость для получения менеджера алертов"""
    return AlertManager()


@router.post(
    "/send",
    summary="Отправить алерт",
    description="Отправка алерта через настроенные каналы",
    responses={
        200: {"description": "Алерт отправлен"},
        400: {"model": ErrorResponse, "description": "Ошибка валидации"},
    },
)
async def send_alert(
    alert_name: str = Body(..., description="Название алерта"),
    severity: str = Body(..., description="Уровень серьёзности"),
    description: str = Body(..., description="Описание алерта"),
    details: Dict[str, Any] = Body(default={}, description="Дополнительные детали"),
    channels: List[str] = Body(default=None, description="Каналы для отправки"),
    alert_manager: AlertManager = Depends(get_alert_manager),
):
    """Отправить алерт"""
    result = alert_manager.send_alert(
        alert_name=alert_name,
        severity=severity,
        description=description,
        details=details,
        channels=channels,
    )

    if result.get("status") == "duplicate":
        return {"success": False, "reason": "duplicate", "message": "Дубликат алерта"}
    elif result.get("status") == "rate_limited":
        return {"success": False, "reason": "rate_limited", "message": "Превышен лимит"}
    elif result.get("status") == "silenced":
        return {"success": False, "reason": "silenced", "message": "Алерт заглушен"}

    return {"success": True, **result}


@router.post(
    "/send-async",
    summary="Отправить алерт (async)",
    description="Асинхронная отправка алерта",
)
async def send_alert_async(
    alert_name: str = Body(...),
    severity: str = Body(...),
    description: str = Body(...),
    details: Dict[str, Any] = Body(default={}),
    channels: List[str] = Body(default=None),
    alert_manager: AlertManager = Depends(get_alert_manager),
):
    """Асинхронная отправка алерта"""
    result = await alert_manager.send_alert_async(
        alert_name=alert_name,
        severity=severity,
        description=description,
        details=details,
        channels=channels,
    )
    return result


@router.post(
    "/resolve/{alert_id}",
    summary="Закрыть алерт",
    description="Закрытие алерта по ID",
)
async def resolve_alert(
    alert_id: str,
    alert_manager: AlertManager = Depends(get_alert_manager),
):
    """Закрыть алерт"""
    success = alert_manager.resolve_alert(alert_id)
    return {"success": success, "alert_id": alert_id}


@router.post(
    "/acknowledge/{alert_id}",
    summary="Подтвердить алерт",
    description="Подтверждение алерта пользователем",
)
async def acknowledge_alert(
    alert_id: str,
    acknowledged_by: str = Body(..., description="Имя пользователя", embed=True),
    alert_manager: AlertManager = Depends(get_alert_manager),
):
    """Подтвердить алерт"""
    success = alert_manager.acknowledge_alert(alert_id, acknowledged_by)
    return {"success": success, "alert_id": alert_id}


@router.post(
    "/silence/{alert_id}",
    summary="Заглушить алерт",
    description="Временное отключение уведомлений для алерта",
)
async def silence_alert(
    alert_id: str,
    duration_minutes: int = Body(default=60, ge=1, le=1440),
    alert_manager: AlertManager = Depends(get_alert_manager),
):
    """Заглушить алерт"""
    alert_manager.silence_alert(alert_id, duration_minutes)
    return {"success": True, "alert_id": alert_id, "duration_minutes": duration_minutes}


@router.get(
    "/active",
    summary="Активные алерты",
    description="Получение списка активных алертов",
)
async def get_active_alerts(alert_manager: AlertManager = Depends(get_alert_manager)):
    """Получить активные алерты"""
    alerts = alert_manager.get_active_alerts()
    return {"alerts": [a.__dict__ for a in alerts], "total": len(alerts)}


@router.get(
    "/statistics",
    summary="Статистика алертов",
    description="Получение статистики по алертам",
)
async def get_alert_statistics(alert_manager: AlertManager = Depends(get_alert_manager)):
    """Получить статистику"""
    return alert_manager.get_alert_statistics()


@router.get(
    "/history",
    summary="История алертов",
    description="Получение истории алертов с пагинацией",
)
async def get_alert_history(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    severity: Optional[str] = Query(default=None, description="Фильтр по severity"),
    alert_manager: AlertManager = Depends(get_alert_manager),
):
    """Получить историю алертов"""
    history = alert_manager.alert_history

    if severity:
        history = [a for a in history if a.get("severity") == severity]

    total = len(history)
    paginated = history[offset : offset + limit]

    return {
        "alerts": paginated,
        "total": total,
        "limit": limit,
        "offset": offset,
    }
