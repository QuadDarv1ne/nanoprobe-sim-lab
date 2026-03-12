# -*- coding: utf-8 -*-
"""
Dashboard API routes for Nanoprobe Sim Lab
Provides aggregated stats and system health endpoints
"""

from fastapi import APIRouter, HTTPException, status
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import psutil

router = APIRouter()


@router.get("/stats", summary="Получить сводную статистику")
async def get_dashboard_stats():
    """
    Возвращает сводную статистику для дашборда:
    - Количество сканирований
    - Количество симуляций
    - Количество анализов
    - Аптайм системы
    """
    # В реальной реализации эти данные берутся из БД
    stats = {
        "scans_count": 0,
        "simulations_count": 0,
        "analyses_count": 0,
        "uptime_seconds": _get_system_uptime(),
        "uptime_formatted": _format_uptime(_get_system_uptime()),
    }
    return stats


@router.get("/health/detailed", summary="Детальная проверка здоровья системы")
async def get_detailed_health():
    """
    Детальная информация о здоровье системы:
    - Статус CPU
    - Статус памяти
    - Статус диска
    - Статус сервисов
    """
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        health_status = "healthy"
        issues = []
        
        if cpu_percent > 90:
            health_status = "warning"
            issues.append("Высокая загрузка CPU")
        
        if memory.percent > 90:
            health_status = "warning"
            issues.append("Высокое использование памяти")
        
        if disk.percent > 90:
            health_status = "critical"
            issues.append("Критическое заполнение диска")
        
        return {
            "status": health_status,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "metrics": {
                "cpu": {
                    "percent": cpu_percent,
                    "status": "ok" if cpu_percent < 90 else "warning"
                },
                "memory": {
                    "percent": memory.percent,
                    "used_gb": memory.used / (1024 ** 3),
                    "total_gb": memory.total / (1024 ** 3),
                    "status": "ok" if memory.percent < 90 else "warning"
                },
                "disk": {
                    "percent": disk.percent,
                    "used_gb": disk.used / (1024 ** 3),
                    "total_gb": disk.total / (1024 ** 3),
                    "status": "ok" if disk.percent < 90 else "warning"
                }
            },
            "issues": issues,
            "services": {
                "api": "running",
                "database": "running",
                "cache": "disabled"
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка проверки здоровья: {str(e)}"
        )


@router.get("/metrics/realtime", summary="Метрики в реальном времени")
async def get_realtime_metrics():
    """
    Метрики системы в реальном времени для графиков
    """
    try:
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "network": {
                "bytes_sent": psutil.net_io_counters().bytes_sent,
                "bytes_recv": psutil.net_io_counters().bytes_recv,
            } if hasattr(psutil, 'net_io_counters') else None
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка получения метрик: {str(e)}"
        )


@router.get("/export/{format}", summary="Экспорт данных")
async def export_data(format: str):
    """
    Экспорт данных в различных форматах:
    - json: JSON формат
    - csv: CSV формат
    - pdf: PDF отчёт
    """
    if format not in ["json", "csv", "pdf"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Неподдерживаемый формат: {format}. Доступны: json, csv, pdf"
        )
    
    # Временная заглушка
    return {
        "format": format,
        "status": "success",
        "message": f"Данные экспортированы в формате {format.upper()}",
        "download_url": f"/api/v1/downloads/export_{format}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
    }


@router.post("/actions/clean_cache", summary="Очистка кэша")
async def clean_cache_action():
    """
    Очистка системного кэша
    """
    try:
        # В реальной реализации здесь будет вызов CacheManager
        cleaned_size = 0.0
        cleaned_files = 0
        
        return {
            "success": True,
            "message": f"Кэш очищен",
            "cleaned_size_mb": cleaned_size,
            "cleaned_files": cleaned_files
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@router.post("/actions/start_component", summary="Запуск компонента")
async def start_component_action(component: dict):
    """
    Запуск компонента проекта
    """
    component_name = component.get("component", "")
    
    # В реальной реализации здесь будет запуск процесса
    return {
        "success": True,
        "message": f"Компонент '{component_name}' запущен",
        "component": component_name,
        "pid": 12345  # В реальности PID процесса
    }


@router.post("/actions/stop_component", summary="Остановка компонента")
async def stop_component_action(component: dict):
    """
    Остановка компонента проекта
    """
    component_name = component.get("component", "")
    
    # В реальной реализации здесь будет остановка процесса
    return {
        "success": True,
        "message": f"Компонент '{component_name}' остановлен",
        "component": component_name
    }


def _get_system_uptime() -> int:
    """Получить аптайм системы в секундах"""
    try:
        boot_time = psutil.boot_time()
        uptime = datetime.now().timestamp() - boot_time
        return int(uptime)
    except:
        return 0


def _format_uptime(seconds: int) -> str:
    """Форматировать аптайм в человекочитаемый вид"""
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    
    parts = []
    if days > 0:
        parts.append(f"{days} дн")
    if hours > 0:
        parts.append(f"{hours} ч")
    if minutes > 0:
        parts.append(f"{minutes} мин")
    
    return " ".join(parts) if parts else "< 1 мин"
