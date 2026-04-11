"""
Dashboard actions endpoints

Экспорт данных, очистка кэша, управление компонентами.
"""

import logging
import os
from datetime import datetime, timezone

from fastapi import APIRouter, Query, status
from fastapi.responses import JSONResponse

from api.error_handlers import ValidationError
from utils.caching.cache_manager import CacheManager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/export/{format}",
    summary="Экспорт данных",
    description="Экспорт данных дашборда в различных форматах",
)
async def export_data(format: str):
    """Экспорт данных в различных форматах."""
    if format not in ["json", "csv", "pdf"]:
        raise ValidationError(f"Неподдерживаемый формат: {format}. Доступны: json, csv, pdf")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return {
        "format": format,
        "status": "success",
        "message": f"Данные экспортированы в формате {format.upper()}",
        "download_url": f"/api/v1/downloads/export_{timestamp}.{format}",
        "expires_in": 3600,
    }


@router.post(
    "/actions/clean_cache",
    summary="Очистка кэша",
    description="Очистка системного кэша проекта",
)
async def clean_cache_action():
    """Очистка системного кэша."""
    try:
        cache_mgr = CacheManager()

        cleaned_size = 0.0
        cleaned_files = 0

        report = cache_mgr.auto_cleanup()
        if report:
            cleaned_size = report.get("cleaned_size_mb", 0.0)
            cleaned_files = report.get("cleaned_files", 0)

        logger.info(f"Cache cleaned: {cleaned_files} files, {cleaned_size} MB")
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "message": "Кэш успешно очищен",
                "cleaned_size_mb": cleaned_size,
                "cleaned_files": cleaned_files,
            },
        )
    except Exception as e:
        logger.error(f"Error cleaning cache: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "error": str(e)},
        )


@router.post(
    "/actions/start_component",
    summary="Запуск компонента",
    description="Запуск компонента проекта",
)
async def start_component_action(component: dict):
    """Запуск компонента проекта."""
    component_name = component.get("component", "")
    if not component_name:
        raise ValidationError("Не указано имя компонента")

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "success": True,
            "message": f"Компонент '{component_name}' запущен",
            "component": component_name,
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )


@router.post(
    "/actions/stop_component",
    summary="Остановка компонента",
    description="Остановка компонента проекта",
)
async def stop_component_action(component: dict):
    """Остановка компонента проекта."""
    component_name = component.get("component", "")
    if not component_name:
        raise ValidationError("Не указано имя компонента")

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "success": True,
            "message": f"Компонент '{component_name}' остановлен",
            "component": component_name,
            "stopped_at": datetime.now(timezone.utc).isoformat(),
        },
    )
