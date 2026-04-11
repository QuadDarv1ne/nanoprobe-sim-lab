"""
Dashboard realtime endpoints

Real-time метрики, WebSocket.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import psutil
from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect

from api.dependencies import get_db
from api.error_handlers import DatabaseError, ServiceUnavailableError
from api.routes.dashboard.helpers import (
    CACHE_PREFIX,
    METRICS_CACHE_TTL,
    get_project_root,
    get_system_disk_usage,
)
from api.schemas import ErrorResponse, RealtimeMetrics
from api.state import get_app_state, set_app_state
from utils.caching.redis_cache import cache, cached
from utils.monitoring.monitoring import get_monitor

logger = logging.getLogger(__name__)

router = APIRouter()


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
    """Метрики системы в реальном времени для графиков."""
    cache_key = f"{CACHE_PREFIX['metrics']}:realtime"
    if include_history:
        cache_key += ":with_history"

    if cache.is_available():
        cached_data = cache.get(cache_key)
        if cached_data:
            return cached_data

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

        response_data = result.model_dump()
        if include_history:
            history = monitor.get_metrics_history(limit=60)
            response_data = {
                "current": result.model_dump(),
                "history": history,
            }

        if cache.is_available():
            cache.set(cache_key, response_data, expire=1)

        logger.debug(f"Realtime metrics retrieved (history={include_history})")
        return response_data
    except Exception as e:
        logger.error(f"Error getting realtime metrics: {e}")
        raise DatabaseError(f"Ошибка получения метрик: {str(e)}")


@router.get("/metrics/realtime/detailed")
async def get_realtime_metrics_detailed():
    """Get real-time system metrics with detailed information."""
    metrics_cache = get_app_state("metrics_cache")
    cache_timestamp = get_app_state("metrics_cache_time")

    if metrics_cache and cache_timestamp:
        age = (datetime.now(timezone.utc) - cache_timestamp).total_seconds()
        if age < METRICS_CACHE_TTL:
            return metrics_cache

    try:
        cpu_percent_per_core = psutil.cpu_percent(percpu=True, interval=0.1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        net_io = psutil.net_io_counters()

        python_processes = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                if "python" in proc.info["name"].lower():
                    python_processes.append(
                        {
                            "pid": proc.info["pid"],
                            "name": proc.info["name"],
                            "cpu_percent": proc.info["cpu_percent"] or 0,
                            "memory_percent": proc.info["memory_percent"] or 0,
                        }
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cpu": {
                "total": psutil.cpu_percent(interval=0.1),
                "per_core": cpu_percent_per_core,
                "cores": len(cpu_percent_per_core),
            },
            "memory": {
                "percent": memory.percent,
                "used_mb": round(memory.used / (1024**2), 2),
                "available_mb": round(memory.available / (1024**2), 2),
                "total_mb": round(memory.total / (1024**2), 2),
            },
            "disk": {
                "percent": get_system_disk_usage().percent,
                "read_mb": round(disk_io.read_bytes / (1024**2), 2) if disk_io else 0,
                "write_mb": round(disk_io.write_bytes / (1024**2), 2) if disk_io else 0,
            },
            "network": {
                "bytes_sent_mb": (round(net_io.bytes_sent / (1024**2), 2) if net_io else 0),
                "bytes_recv_mb": (round(net_io.bytes_recv / (1024**2), 2) if net_io else 0),
                "packets_sent": net_io.packets_sent if net_io else 0,
                "packets_recv": net_io.packets_recv if net_io else 0,
            },
            "python_processes": python_processes[:5],
        }

        set_app_state("metrics_cache", metrics)
        set_app_state("metrics_cache_time", datetime.now(timezone.utc))

        logger.debug(f"Realtime detailed metrics retrieved: CPU {metrics['cpu']['total']}%")
        return metrics

    except Exception as e:
        logger.error(f"Error getting realtime detailed metrics: {e}")
        raise ServiceUnavailableError(f"Не удалось получить метрики: {str(e)}")


@router.get("/activity/timeline")
@cached(prefix="dashboard:activity", expire=60)
async def get_activity_timeline(
    days: int = Query(7, ge=1, le=30),
    db: Any = Depends(get_db),
):
    """Get activity timeline (cached for 60 seconds)."""
    try:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        scans_timeline = db.execute_query(
            "SELECT DATE(created_at) as date, COUNT(*) as count "
            "FROM scan_results WHERE created_at >= ? "
            "GROUP BY DATE(created_at) ORDER BY date",
            (start_date.isoformat(),),
        )

        sims_timeline = db.execute_query(
            "SELECT DATE(created_at) as date, COUNT(*) as count "
            "FROM simulations WHERE created_at >= ? "
            "GROUP BY DATE(created_at) ORDER BY date",
            (start_date.isoformat(),),
        )

        analysis_timeline = db.execute_query(
            "SELECT DATE(created_at) as date, COUNT(*) as count "
            "FROM defect_analysis WHERE created_at >= ? "
            "GROUP BY DATE(created_at) ORDER BY date",
            (start_date.isoformat(),),
        )

        timeline = {}
        for i in range(days):
            date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
            timeline[date] = {"scans": 0, "simulations": 0, "analysis": 0}

        for row in scans_timeline or []:
            date = row["date"]
            if date in timeline:
                timeline[date]["scans"] = row["count"]

        for row in sims_timeline or []:
            date = row["date"]
            if date in timeline:
                timeline[date]["simulations"] = row["count"]

        for row in analysis_timeline or []:
            date = row["date"]
            if date in timeline:
                timeline[date]["analysis"] = row["count"]

        return {
            "period": {
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d"),
                "days": days,
            },
            "timeline": [
                {
                    "date": date,
                    "scans": data["scans"],
                    "simulations": data["simulations"],
                    "analysis": data["analysis"],
                }
                for date, data in timeline.items()
            ],
        }
    except Exception as e:
        logger.error(f"Error getting activity timeline: {e}")
        raise ServiceUnavailableError(f"Не удалось получить активность: {str(e)}")


@router.get(
    "/storage",
    summary="Статистика хранилища",
    description="Получить детальную статистику хранилища по директориям",
)
@cached(prefix="dashboard:storage", expire=30)
async def get_storage_stats_endpoint(
    db: Any = Depends(get_db),
):
    """Получить детальную статистику хранилища."""
    root = get_project_root()
    data_dir = root / "data"
    output_dir = root / "output"
    logs_dir = root / "logs"

    def get_dir_stats(directory):
        if not directory.exists():
            return {"files": 0, "size_mb": 0, "largest_files": []}

        files = []
        total_size = 0

        for item in directory.rglob("*"):
            if item.is_file():
                try:
                    size = item.stat().st_size
                    total_size += size
                    files.append(
                        {
                            "path": str(item.relative_to(root)),
                            "size_mb": round(size / (1024**2), 4),
                            "modified": datetime.fromtimestamp(
                                item.stat().st_mtime, tz=timezone.utc
                            ).isoformat(),
                        }
                    )
                except (OSError, IOError):
                    continue

        files.sort(key=lambda x: x["size_mb"], reverse=True)

        return {
            "files": len(files),
            "size_mb": round(total_size / (1024**2), 2),
            "largest_files": files[:5],
        }

    try:
        disk = get_system_disk_usage()

        db_path = Path("data/nanoprobe.db")
        db_size_mb = round(db_path.stat().st_size / (1024**2), 2) if db_path.exists() else 0

        return {
            "data": get_dir_stats(data_dir),
            "output": get_dir_stats(output_dir),
            "logs": get_dir_stats(logs_dir),
            "database": {"size_mb": db_size_mb, "path": str(db_path)},
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent": disk.percent,
            },
        }
    except Exception as e:
        logger.error(f"Error getting storage stats: {e}")
        raise ServiceUnavailableError(f"Не удалось получить статистику хранилища: {str(e)}")


@router.websocket("/ws/metrics")
async def metrics_websocket(websocket: WebSocket):
    """WebSocket для real-time метрик."""
    from api.websocket_manager import get_connection_manager

    manager = get_connection_manager()

    if not await manager.connect(websocket):
        return

    logger.info(f"WebSocket metrics connected. " f"Active: {manager.connection_count}")
    try:
        while True:
            try:
                try:
                    msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                    if msg == "ping":
                        await websocket.send_text("pong")
                except asyncio.TimeoutError:
                    pass

                metrics = await get_realtime_metrics_detailed()
                await websocket.send_json(metrics)
                await asyncio.sleep(1)
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket metrics error: {e}", exc_info=True)
                await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"WebSocket metrics fatal error: {e}", exc_info=True)
    finally:
        await manager.disconnect(websocket)
        logger.info(f"WebSocket metrics disconnected. " f"Active: {manager.connection_count}")
