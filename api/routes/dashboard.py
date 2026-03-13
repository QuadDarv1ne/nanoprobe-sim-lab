# -*- coding: utf-8 -*-
"""
Dashboard API routes for Nanoprobe Sim Lab
Provides aggregated stats and system health endpoints
"""

from fastapi import APIRouter, HTTPException, status, Depends, Query
from fastapi.responses import JSONResponse
from fastapi import Header
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from functools import lru_cache
import psutil
import os
from pathlib import Path

from api.schemas import (
    DashboardStats,
    SystemHealth,
    RealtimeMetrics,
    HealthStatus,
    PaginatedResponse,
    ErrorResponse,
)
from utils.enhanced_monitor import get_monitor, format_uptime

router = APIRouter()

# Кэш для статистики (5 секунд)
_stats_cache = {}
_stats_cache_time = None
CACHE_TTL = 5  # секунд


def get_project_root() -> Path:
    """Получить корень проекта"""
    return Path(__file__).parent.parent.parent


def get_cached_stats() -> Optional[Dict]:
    """Получить кэшированную статистику если не истёк TTL"""
    global _stats_cache, _stats_cache_time
    if _stats_cache_time is None:
        return None
    age = (datetime.now() - _stats_cache_time).total_seconds()
    if age < CACHE_TTL:
        return _stats_cache
    return None


def cache_stats(stats: Dict):
    """Закэшировать статистику"""
    global _stats_cache, _stats_cache_time
    _stats_cache = stats
    _stats_cache_time = datetime.now()


def get_storage_stats() -> Dict[str, float]:
    """Получить статистику хранилища"""
    root = get_project_root()
    data_dir = root / "data"
    output_dir = root / "output"

    used_mb = 0.0
    total_mb = 0.0

    # Подсчёт размера data и output директорий
    for directory in [data_dir, output_dir]:
        if directory.exists():
            for item in directory.rglob("*"):
                if item.is_file():
                    try:
                        used_mb += item.stat().st_size / (1024 * 1024)
                    except (OSError, IOError):
                        continue

    # Общая ёмкость диска
    disk = psutil.disk_usage('/')
    total_mb = disk.total / (1024 * 1024)

    return {
        "used_mb": round(used_mb, 2),
        "total_mb": round(total_mb, 2),
        "percent": round((used_mb / total_mb) * 100, 2) if total_mb > 0 else 0
    }


@router.get(
    "/stats",
    response_model=DashboardStats,
    summary="Получить сводную статистику",
    description="Возвращает агрегированную статистику для дашборда",
    responses={
        200: {"model": DashboardStats, "description": "Успешный ответ"},
        500: {"model": ErrorResponse, "description": "Ошибка сервера"},
    },
)
async def get_dashboard_stats(
    cache_control: Optional[str] = Header(None, alias="Cache-Control")
):
    """
    Возвращает сводную статистику для дашборда:
    - Количество сканирований
    - Количество симуляций
    - Использование хранилища
    - Аптайм системы
    - Расширенная статистика БД
    """
    # Проверка кэша если не требуется свежий ответ
    if cache_control != "no-cache":
        cached = get_cached_stats()
        if cached:
            return DashboardStats(**cached)

    try:
        monitor = get_monitor()
        stats = monitor.get_statistics()
        storage = get_storage_stats()

        # Интеграция с БД для реальных данных
        db_stats = {}
        db_size_mb = 0.0
        try:
            from utils.database import DatabaseManager
            from pathlib import Path
            db = DatabaseManager(db_path="data/nanoprobe.db")
            db_stats = db.get_statistics()
            # Размер БД
            db_path = Path("data/nanoprobe.db")
            if db_path.exists():
                db_size_mb = round(db_path.stat().st_size / (1024 * 1024), 2)
        except Exception:
            # Фоллбэк на заглушки если БД недоступна
            pass

        result = DashboardStats(
            total_scans=db_stats.get('total_scans', 0),
            total_simulations=db_stats.get('total_simulations', 0),
            active_simulations=db_stats.get('active_simulations', 0),
            storage_used_mb=storage["used_mb"],
            storage_total_mb=storage["total_mb"],
            recent_scans_count=db_stats.get('total_scans', 0),
            recent_simulations_count=db_stats.get('total_simulations', 0),
            success_rate=100.0 if db_stats.get('total_scans', 0) == 0 else 100.0,
            # Расширенная статистика
            total_images=db_stats.get('total_images', 0),
            total_exports=db_stats.get('total_exports', 0),
            total_comparisons=db_stats.get('total_comparisons', 0),
            total_defect_analyses=db_stats.get('total_defect_analyses', 0),
            total_pdf_reports=db_stats.get('total_pdf_reports', 0),
            total_batch_jobs=db_stats.get('total_batch_jobs', 0),
            active_batch_jobs=db_stats.get('active_batch_jobs', 0),
            scans_by_type=db_stats.get('scans_by_type', {}),
            db_size_mb=db_size_mb,
        )
        
        # Кэширование результата
        cache_stats(result.model_dump())
        
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка получения статистики: {str(e)}"
        )


@router.get(
    "/health/detailed",
    response_model=SystemHealth,
    summary="Детальная проверка здоровья системы",
    description="Детальная информация о здоровье системы с метриками",
    responses={
        200: {"model": SystemHealth, "description": "Успешный ответ"},
        500: {"model": ErrorResponse, "description": "Ошибка сервера"},
    },
)
async def get_detailed_health():
    """
    Детальная информация о здоровье системы:
    - Статус CPU с историей
    - Статус памяти с деталями
    - Статус диска с деталями
    - Статус сервисов
    - Список проблем
    """
    try:
        monitor = get_monitor()
        metrics = monitor.get_current_metrics()
        alerts = monitor.get_alerts(limit=10)

        health_status = "healthy"
        issues = []

        # Проверка CPU
        if metrics["cpu_percent"] > 90:
            health_status = "critical" if metrics["cpu_percent"] > 95 else "warning"
            issues.append(f"Высокая загрузка CPU: {metrics['cpu_percent']:.1f}%")
        elif metrics["cpu_percent"] > 70:
            if health_status == "healthy":
                health_status = "info"

        # Проверка памяти
        if metrics["memory_percent"] > 90:
            health_status = "critical" if metrics["memory_percent"] > 95 else "warning"
            issues.append(f"Высокое использование памяти: {metrics['memory_percent']:.1f}%")

        # Проверка диска
        if metrics["disk_percent"] > 90:
            health_status = "critical" if metrics["disk_percent"] > 95 else "warning"
            issues.append(f"Критическое заполнение диска: {metrics['disk_percent']:.1f}%")

        # Проверка алертов
        critical_alerts = [a for a in alerts if a.get("level") == "critical"]
        if critical_alerts:
            health_status = "critical"
            issues.extend([a["message"] for a in critical_alerts[:3]])

        return SystemHealth(
            status=health_status,
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            metrics={
                "cpu": {
                    "percent": metrics["cpu_percent"],
                    "cores": metrics["cpu_cores"],
                    "freq_mhz": metrics.get("cpu_freq_mhz"),
                    "status": "ok" if metrics["cpu_percent"] < 80 else "warning"
                },
                "memory": {
                    "percent": metrics["memory_percent"],
                    "used_gb": metrics["memory_used_gb"],
                    "total_gb": metrics["memory_total_gb"],
                    "available_gb": metrics["memory_available_gb"],
                    "status": "ok" if metrics["memory_percent"] < 80 else "warning"
                },
                "disk": {
                    "percent": metrics["disk_percent"],
                    "used_gb": metrics["disk_used_gb"],
                    "total_gb": metrics["disk_total_gb"],
                    "free_gb": metrics["disk_free_gb"],
                    "status": "ok" if metrics["disk_percent"] < 80 else "warning"
                },
                "network": {
                    "bytes_sent": metrics["network_bytes_sent"],
                    "bytes_recv": metrics["network_bytes_recv"],
                    "packets_sent": metrics["network_packets_sent"],
                    "packets_recv": metrics["network_packets_recv"]
                },
                "system": {
                    "uptime_seconds": metrics["uptime_seconds"],
                    "uptime_formatted": format_uptime(metrics["uptime_seconds"]),
                    "boot_time": metrics["boot_time"],
                    "processes_count": metrics["processes_count"]
                }
            },
            issues=issues,
            services={
                "api": "running",
                "database": "running",
                "cache": "disabled",
                "monitoring": "running"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка проверки здоровья: {str(e)}"
        )


@router.get(
    "/metrics/realtime",
    response_model=RealtimeMetrics,
    summary="Метрики в реальном времени",
    description="Метрики системы в реальном времени для графиков",
    responses={
        200: {"model": RealtimeMetrics, "description": "Успешный ответ"},
        500: {"model": ErrorResponse, "description": "Ошибка сервера"},
    },
)
async def get_realtime_metrics(
    include_network: bool = Query(True, description="Включить сетевые метрики"),
    include_history: bool = Query(False, description="Включить историю метрик"),
):
    """
    Метрики системы в реальном времени для графиков
    """
    try:
        monitor = get_monitor()
        metrics = monitor.get_current_metrics()
        network_speed = monitor.get_network_speed()

        result = RealtimeMetrics(
            timestamp=metrics["timestamp"],
            cpu_percent=metrics["cpu_percent"],
            memory_percent=metrics["memory_percent"],
            disk_percent=metrics["disk_percent"],
            network_upload_mbps=network_speed["upload_mbps"],
            network_download_mbps=network_speed["download_mbps"],
        )

        if include_history:
            history = monitor.get_metrics_history(limit=60)
            return {
                "current": result.model_dump(),
                "history": history,
            }

        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка получения метрик: {str(e)}"
        )


@router.get(
    "/metrics/history",
    summary="История метрик",
    description="Получить историю метрик за период",
)
async def get_metrics_history(
    limit: int = Query(60, ge=1, le=300, description="Количество записей"),
    component: Optional[str] = Query(None, description="Фильтр по компоненту (cpu/memory/disk)"),
):
    """Получить историю метрик"""
    try:
        monitor = get_monitor()
        history = monitor.get_metrics_history(limit=limit)

        if component:
            filtered_history = []
            for entry in history:
                if component == "cpu" and "cpu_percent" in entry:
                    filtered_history.append({
                        "timestamp": entry["timestamp"],
                        "value": entry["cpu_percent"]
                    })
                elif component == "memory" and "memory_percent" in entry:
                    filtered_history.append({
                        "timestamp": entry["timestamp"],
                        "value": entry["memory_percent"]
                    })
                elif component == "disk" and "disk_percent" in entry:
                    filtered_history.append({
                        "timestamp": entry["timestamp"],
                        "value": entry["disk_percent"]
                    })
            return {"component": component, "history": filtered_history}

        return {"history": history}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка получения истории: {str(e)}"
        )


@router.get(
    "/alerts",
    summary="Получить алерты",
    description="Получить список алертов системы",
)
async def get_alerts(
    limit: int = Query(50, ge=1, le=200, description="Количество алертов"),
    level: Optional[str] = Query(None, description="Фильтр по уровню (info/warning/critical)"),
):
    """Получить алерты системы"""
    try:
        monitor = get_monitor()
        alerts = monitor.get_alerts(limit=limit, level=level)
        return {"alerts": alerts, "total": len(alerts)}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка получения алертов: {str(e)}"
        )


@router.get(
    "/processes",
    summary="Топ процессов",
    description="Получить топ процессов по использованию ресурсов",
)
async def get_top_processes(
    limit: int = Query(10, ge=1, le=50, description="Количество процессов"),
    sort_by: str = Query("cpu", description="Сортировка (cpu/memory)"),
):
    """Получить топ процессов"""
    try:
        monitor = get_monitor()
        processes = monitor.get_process_list(limit=limit, sort_by=sort_by)
        return {"processes": processes, "total": len(processes)}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка получения процессов: {str(e)}"
        )


@router.post(
    "/export/{format}",
    summary="Экспорт данных",
    description="Экспорт данных дашборда в различных форматах",
)
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

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return {
        "format": format,
        "status": "success",
        "message": f"Данные экспортированы в формате {format.upper()}",
        "download_url": f"/api/v1/downloads/export_{format}_{timestamp}.{format}",
        "expires_in": 3600
    }


@router.post(
    "/actions/clean_cache",
    summary="Очистка кэша",
    description="Очистка системного кэша проекта",
)
async def clean_cache_action():
    """
    Очистка системного кэша
    """
    try:
        from utils.cache_manager import CacheManager
        cache_mgr = CacheManager()

        cleaned_size = 0.0
        cleaned_files = 0

        # Выполнение очистки
        report = cache_mgr.auto_cleanup()
        if report:
            cleaned_size = report.get("cleaned_size_mb", 0.0)
            cleaned_files = report.get("cleaned_files", 0)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "message": "Кэш успешно очищен",
                "cleaned_size_mb": cleaned_size,
                "cleaned_files": cleaned_files
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": str(e)
            }
        )


@router.post(
    "/actions/start_component",
    summary="Запуск компонента",
    description="Запуск компонента проекта",
)
async def start_component_action(component: dict):
    """
    Запуск компонента проекта
    """
    component_name = component.get("component", "")
    if not component_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Не указано имя компонента"
        )

    # В реальной реализации здесь будет запуск процесса
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "success": True,
            "message": f"Компонент '{component_name}' запущен",
            "component": component_name,
            "pid": os.getpid(),  # В реальности PID процесса
            "started_at": datetime.now().isoformat()
        }
    )


@router.post(
    "/actions/stop_component",
    summary="Остановка компонента",
    description="Остановка компонента проекта",
)
async def stop_component_action(component: dict):
    """
    Остановка компонента проекта
    """
    component_name = component.get("component", "")
    if not component_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Не указано имя компонента"
        )

    # В реальной реализации здесь будет остановка процесса
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "success": True,
            "message": f"Компонент '{component_name}' остановлен",
            "component": component_name,
            "stopped_at": datetime.now().isoformat()
        }
    )


@router.get(
    "/storage",
    summary="Статистика хранилища",
    description="Получить детальную статистику хранилища",
)
async def get_storage_stats_endpoint():
    """Получить статистику хранилища"""
    try:
        storage = get_storage_stats()
        disk = psutil.disk_usage('/')

        return {
            "project_storage": storage,
            "disk_info": {
                "total_gb": round(disk.total / (1024 ** 3), 2),
                "used_gb": round(disk.used / (1024 ** 3), 2),
                "free_gb": round(disk.free / (1024 ** 3), 2),
                "percent": disk.percent
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка получения статистики хранилища: {str(e)}"
        )
