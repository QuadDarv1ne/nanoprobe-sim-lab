"""
Dashboard stats endpoints

Статистика дашборда, health checks.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import psutil
from fastapi import APIRouter, Depends, Header

from api.dependencies import get_db
from api.error_handlers import DatabaseError, ServiceUnavailableError
from api.routes.dashboard.helpers import (
    CACHE_PREFIX,
    cache_stats,
    get_cached_stats,
    get_storage_stats,
)
from api.schemas import DashboardStats, ErrorResponse, SystemHealth
from api.state import get_system_disk_usage
from utils.caching.redis_cache import cache, cached
from utils.database import DatabaseManager
from utils.monitoring.monitoring import format_uptime, get_monitor

logger = logging.getLogger(__name__)

router = APIRouter()


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
    cache_control: Optional[str] = Header(None, alias="Cache-Control"),
    db: DatabaseManager = Depends(get_db),
):
    """
    Возвращает сводную статистику для дашборда.
    Кэширование: 5 секунд (Redis + in-memory)
    """
    if cache_control != "no-cache":
        if cache.is_available():
            redis_key = f"{CACHE_PREFIX['stats']}:all"
            cached_data = cache.get(redis_key)
            if cached_data:
                return DashboardStats(**cached_data)

        cached_stats = get_cached_stats()
        if cached_stats:
            return DashboardStats(**cached_stats)

    try:
        storage = get_storage_stats()

        db_stats = {}
        db_size_mb = 0.0
        try:
            db_stats = db.get_statistics()
            db_path = Path("data/nanoprobe.db")
            if db_path.exists():
                db_size_mb = round(db_path.stat().st_size / (1024 * 1024), 2)
        except Exception as e:
            logger.warning(f"Failed to get DB size: {e}")

        result = DashboardStats(
            total_scans=db_stats.get("total_scans", 0),
            total_simulations=db_stats.get("total_simulations", 0),
            active_simulations=db_stats.get("active_simulations", 0),
            storage_used_mb=storage["used_mb"],
            storage_total_mb=storage["total_mb"],
            recent_scans_count=db_stats.get("total_scans", 0),
            recent_simulations_count=db_stats.get("total_simulations", 0),
            success_rate=100.0 if db_stats.get("total_scans", 0) == 0 else 100.0,
            total_images=db_stats.get("total_images", 0),
            total_exports=db_stats.get("total_exports", 0),
            total_comparisons=db_stats.get("total_comparisons", 0),
            total_defect_analyses=db_stats.get("total_defect_analyses", 0),
            total_pdf_reports=db_stats.get("total_pdf_reports", 0),
            total_batch_jobs=db_stats.get("total_batch_jobs", 0),
            active_batch_jobs=db_stats.get("active_batch_jobs", 0),
            scans_by_type=db_stats.get("scans_by_type", {}),
            db_size_mb=db_size_mb,
        )

        cache_stats(result.model_dump())
        if cache.is_available():
            cache.set(f"{CACHE_PREFIX['stats']}:all", result.model_dump(), expire=5)

        logger.debug(
            "Dashboard stats: %d scans, %d simulations",
            result.total_scans,
            result.total_simulations,
        )
        return result
    except Exception as e:
        logger.error(f"Error getting dashboard stats: {e}")
        raise DatabaseError(f"Ошибка получения статистики: {str(e)}")


@router.get("/stats/detailed")
@cached(prefix="dashboard", expire=5)
async def get_detailed_stats(
    db: DatabaseManager = Depends(get_db),
):
    """Get detailed dashboard statistics (cached for 5 seconds)."""
    try:
        scans_count = db.count_scans()
        simulations_count = db.count_simulations()
        analysis_count = db.count_analysis_results()
        comparisons_count = db.count_comparisons()
        reports_count = db.count_reports()

        scans_by_type = db.execute_query(
            "SELECT scan_type, COUNT(*) as count " "FROM scan_results GROUP BY scan_type"
        )

        recent_scans = db.execute_query(
            "SELECT id, scan_type, surface_type, created_at "
            "FROM scan_results ORDER BY created_at DESC LIMIT 5"
        )

        sim_stats = db.execute_query(
            "SELECT simulation_type, COUNT(*) as count, "
            "AVG(duration_seconds) as avg_duration FROM simulations "
            "GROUP BY simulation_type"
        )

        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = get_system_disk_usage()

        boot_time = datetime.fromtimestamp(psutil.boot_time(), tz=timezone.utc)
        uptime = datetime.now(timezone.utc) - boot_time

        return {
            "summary": {
                "total_scans": scans_count,
                "total_simulations": simulations_count,
                "total_analysis": analysis_count,
                "total_comparisons": comparisons_count,
                "total_reports": reports_count,
                "total_items": (
                    scans_count + simulations_count + analysis_count + comparisons_count
                ),
            },
            "scans": {
                "total": scans_count,
                "by_type": {row["scan_type"]: row["count"] for row in (scans_by_type or [])},
                "recent": [
                    {
                        "id": row["id"],
                        "type": row["scan_type"],
                        "surface_type": row["surface_type"],
                        "created_at": row["created_at"],
                    }
                    for row in (recent_scans or [])
                ],
            },
            "simulations": {
                "total": simulations_count,
                "by_type": {
                    row["simulation_type"]: {
                        "count": row["count"],
                        "avg_duration_sec": round(row["avg_duration"] or 0, 2),
                    }
                    for row in (sim_stats or [])
                },
            },
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_used_gb": round(disk.used / (1024**3), 2),
                "disk_total_gb": round(disk.total / (1024**3), 2),
                "uptime": str(uptime).split(".")[0],
            },
        }
    except Exception as e:
        logger.error(f"Error getting detailed stats: {e}")
        raise ServiceUnavailableError(f"Не удалось получить детальную статистику: {str(e)}")


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
    """Детальная информация о здоровье системы."""
    try:
        monitor = get_monitor()
        metrics = monitor.get_current_metrics()
        alerts = monitor.get_alerts(limit=10)

        health_status = "healthy"
        issues = []

        if metrics["cpu_percent"] > 90:
            health_status = "critical" if metrics["cpu_percent"] > 95 else "warning"
            issues.append(f"Высокая загрузка CPU: {metrics['cpu_percent']:.1f}%")
        elif metrics["cpu_percent"] > 70:
            if health_status == "healthy":
                health_status = "info"

        if metrics["memory_percent"] > 90:
            health_status = "critical" if metrics["memory_percent"] > 95 else "warning"
            issues.append(f"Высокое использование памяти: {metrics['memory_percent']:.1f}%")

        if metrics["disk_percent"] > 90:
            health_status = "critical" if metrics["disk_percent"] > 95 else "warning"
            issues.append(f"Критическое заполнение диска: {metrics['disk_percent']:.1f}%")

        critical_alerts = [a for a in alerts if a.get("level") == "critical"]
        if critical_alerts:
            health_status = "critical"
            issues.extend([a["message"] for a in critical_alerts[:3]])

        return SystemHealth(
            status=health_status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            version="1.0.0",
            metrics={
                "cpu": {
                    "percent": metrics["cpu_percent"],
                    "cores": metrics["cpu_cores"],
                    "freq_mhz": metrics.get("cpu_freq_mhz"),
                    "status": "ok" if metrics["cpu_percent"] < 80 else "warning",
                },
                "memory": {
                    "percent": metrics["memory_percent"],
                    "used_gb": metrics["memory_used_gb"],
                    "total_gb": metrics["memory_total_gb"],
                    "available_gb": metrics["memory_available_gb"],
                    "status": "ok" if metrics["memory_percent"] < 80 else "warning",
                },
                "disk": {
                    "percent": metrics["disk_percent"],
                    "used_gb": metrics["disk_used_gb"],
                    "total_gb": metrics["disk_total_gb"],
                    "free_gb": metrics["disk_free_gb"],
                    "status": "ok" if metrics["disk_percent"] < 80 else "warning",
                },
                "network": {
                    "bytes_sent": metrics["network_bytes_sent"],
                    "bytes_recv": metrics["network_bytes_recv"],
                    "packets_sent": metrics["network_packets_sent"],
                    "packets_recv": metrics["network_packets_recv"],
                },
                "system": {
                    "uptime_seconds": metrics["uptime_seconds"],
                    "uptime_formatted": format_uptime(metrics["uptime_seconds"]),
                    "boot_time": metrics["boot_time"],
                    "processes_count": metrics["processes_count"],
                },
            },
            issues=issues,
            services={
                "api": "running",
                "database": "running",
                "cache": "disabled",
                "monitoring": "running",
            },
        )
    except Exception as e:
        logger.error(f"Error getting detailed health: {e}")
        raise DatabaseError(f"Ошибка проверки здоровья: {str(e)}")
