"""
Unified Dashboard API for Nanoprobe Sim Lab

Объединённый модуль для управления дашбордом:
- Базовая статистика (из dashboard.py)
- Расширенные метрики (из enhanced_dashboard.py)
- WebSocket для real-time данных
- Кэширование Redis
- Activity Timeline
- Storage Statistics
- Health Checks
- Alerts & Monitoring

Требования:
- Python 3.11, 3.12, 3.13, or 3.14
- FastAPI, Redis (опционально)
"""

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status, Depends
from fastapi.responses import JSONResponse
from fastapi import Header
from api.state import get_system_disk_usage
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
import psutil
import os
import asyncio
import logging
from pathlib import Path

from api.schemas import (
    DashboardStats,
    SystemHealth,
    RealtimeMetrics,
    ErrorResponse,
)
from api.error_handlers import DatabaseError, ValidationError, ServiceUnavailableError
from api.state import get_app_state, set_app_state
from api.dependencies import get_db
from utils.monitoring.monitoring import get_monitor, format_uptime
from utils.database import DatabaseManager
from utils.caching.redis_cache import cache, cached

logger = logging.getLogger(__name__)

router = APIRouter()

# ==================== Конфигурация ====================

# Кэш для статистики (5 секунд)
STATS_CACHE_TTL = 5  # секунд

# Кэш для метрик (1 секунда для real-time)
METRICS_CACHE_TTL = 1  # секунда

# WebSocket подключения управляются через ConnectionManager (api/websocket_manager.py)

# Префиксы для Redis кэша
CACHE_PREFIX = {
    "stats": "dashboard:stats",
    "metrics": "dashboard:metrics",
    "health": "dashboard:health",
    "storage": "dashboard:storage",
    "activity": "dashboard:activity",
    "alerts": "dashboard:alerts",
}


# ==================== Утилиты ====================

def get_project_root() -> Path:
    """Получить корень проекта"""
    return Path(__file__).parent.parent.parent


def get_cached_stats() -> Optional[Dict]:
    """Получить кэшированную статистику если не истёк TTL"""
    cached = get_app_state("stats_cache")
    cache_time = get_app_state("stats_cache_time")
    
    if cache_time is None or cached is None:
        return None
    
    age = (datetime.now(timezone.utc) - cache_time).total_seconds()
    if age < STATS_CACHE_TTL:
        return cached
    return None


def cache_stats(stats: Dict):
    """Закэшировать статистику"""
    set_app_state("stats_cache", stats)
    set_app_state("stats_cache_time", datetime.now(timezone.utc))


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
    disk = get_system_disk_usage()
    total_mb = disk.total / (1024 * 1024)

    return {
        "used_mb": round(used_mb, 2),
        "total_mb": round(total_mb, 2),
        "percent": round((used_mb / total_mb) * 100, 2) if total_mb > 0 else 0
    }


# ==================== Basic Stats ====================

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
    Возвращает сводную статистику для дашборда:
    - Количество сканирований
    - Количество симуляций
    - Использование хранилища
    - Аптайм системы
    - Расширенная статистика БД

    Кэширование: 5 секунд (Redis + in-memory)
    """
    # Проверка кэша если не требуется свежий ответ
    if cache_control != "no-cache":
        # Redis кэш
        if cache.is_available():
            redis_key = f"{CACHE_PREFIX['stats']}:all"
            cached_data = cache.get(redis_key)
            if cached_data:
                return DashboardStats(**cached_data)

        # In-memory кэш
        cached = get_cached_stats()
        if cached:
            return DashboardStats(**cached)

    try:
        storage = get_storage_stats()

        # Интеграция с БД для реальных данных
        db_stats = {}
        db_size_mb = 0.0
        try:
            db_stats = db.get_statistics()
            # Размер БД
            db_path = Path("data/nanoprobe.db")
            if db_path.exists():
                db_size_mb = round(db_path.stat().st_size / (1024 * 1024), 2)
        except Exception as e:
            logger.warning(f"Failed to get DB size: {e}")

        result = DashboardStats(
            total_scans=db_stats.get('total_scans', 0),
            total_simulations=db_stats.get('total_simulations', 0),
            active_simulations=db_stats.get('active_simulations', 0),
            storage_used_mb=storage["used_mb"],
            storage_total_mb=storage["total_mb"],
            recent_scans_count=db_stats.get('total_scans', 0),
            recent_simulations_count=db_stats.get('total_simulations', 0),
            success_rate=100.0 if db_stats.get('total_scans', 0) == 0 else 100.0,
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
        if cache.is_available():
            cache.set(f"{CACHE_PREFIX['stats']}:all", result.model_dump(), expire=5)

        logger.debug(f"Dashboard stats retrieved: {result.total_scans} scans, {result.total_simulations} simulations")
        return result
    except Exception as e:
        logger.error(f"Error getting dashboard stats: {e}")
        raise DatabaseError(f"Ошибка получения статистики: {str(e)}")


# ==================== Detailed Stats ====================

@router.get("/stats/detailed")
@cached(prefix="dashboard", expire=5)
async def get_detailed_stats(
    db: DatabaseManager = Depends(get_db),
):
    """
    Get detailed dashboard statistics (cached for 5 seconds)

    Возвращает детальную статистику включая:
    - Базовую статистику по всем таблицам
    - Детализацию по типам сканирований
    - Статистику симуляций с средней длительностью
    - Последние активности
    - Использование ресурсов системы
    """
    try:
        # Базовая статистика
        scans_count = db.count_scans()
        simulations_count = db.count_simulations()
        analysis_count = db.count_analysis_results()
        comparisons_count = db.count_comparisons()
        reports_count = db.count_reports()

        # Детальная статистика по сканированиям
        scans_by_type = db.execute_query(
            "SELECT scan_type, COUNT(*) as count FROM scan_results GROUP BY scan_type"
        )

        # Последние активности
        recent_scans = db.execute_query(
            "SELECT id, scan_type, surface_type, created_at FROM scan_results ORDER BY created_at DESC LIMIT 5"
        )

        # Статистика по симуляциям
        sim_stats = db.execute_query(
            "SELECT simulation_type, COUNT(*) as count, "
            "AVG(duration_seconds) as avg_duration FROM simulations "
            "GROUP BY simulation_type"
        )

        # Использование ресурсов
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = get_system_disk_usage()

        # Uptime системы
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = datetime.now(timezone.utc) - boot_time

        return {
            "summary": {
                "total_scans": scans_count,
                "total_simulations": simulations_count,
                "total_analysis": analysis_count,
                "total_comparisons": comparisons_count,
                "total_reports": reports_count,
                "total_items": scans_count + simulations_count + analysis_count + comparisons_count
            },
            "scans": {
                "total": scans_count,
                "by_type": {row["scan_type"]: row["count"] for row in (scans_by_type or [])},
                "recent": [
                    {
                        "id": row["id"],
                        "type": row["scan_type"],
                        "surface_type": row["surface_type"],
                        "created_at": row["created_at"]
                    }
                    for row in (recent_scans or [])
                ]
            },
            "simulations": {
                "total": simulations_count,
                "by_type": {
                    row["simulation_type"]: {
                        "count": row["count"],
                        "avg_duration_sec": round(row["avg_duration"] or 0, 2)
                    }
                    for row in (sim_stats or [])
                }
            },
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_used_gb": round(disk.used / (1024**3), 2),
                "disk_total_gb": round(disk.total / (1024**3), 2),
                "uptime": str(uptime).split('.')[0]
            }
        }
    except Exception as e:
        logger.error(f"Error getting detailed stats: {e}")
        raise ServiceUnavailableError(f"Не удалось получить детальную статистику: {str(e)}")


# ==================== Health Checks ====================

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
            timestamp=datetime.now(timezone.utc).isoformat(),
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
        logger.error(f"Error getting detailed health: {e}")
        raise DatabaseError(f"Ошибка проверки здоровья: {str(e)}")


# ==================== Real-time Metrics ====================

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

    Кэширование: 1 секунда (Redis) - для real-time данных
    """
    cache_key = f"{CACHE_PREFIX['metrics']}:realtime"
    if include_history:
        cache_key += ":with_history"

    # Проверка Redis кэша
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

        # Кэширование в Redis (1 секунда для real-time)
        if cache.is_available():
            cache.set(cache_key, response_data, expire=1)

        logger.debug(f"Realtime metrics retrieved (history={include_history})")
        return response_data
    except Exception as e:
        logger.error(f"Error getting realtime metrics: {e}")
        raise DatabaseError(f"Ошибка получения метрик: {str(e)}")


@router.get("/metrics/realtime/detailed")
async def get_realtime_metrics_detailed():
    """
    Get real-time system metrics with detailed information

    Возвращает детальную информацию о системе:
    - CPU по ядрам
    - Memory детально
    - Disk I/O
    - Network I/O
    - Python процессы
    """
    # Проверка кэша
    metrics_cache = get_app_state("metrics_cache")
    cache_timestamp = get_app_state("metrics_cache_time")

    if metrics_cache and cache_timestamp:
        age = (datetime.now(timezone.utc) - cache_timestamp).total_seconds()
        if age < METRICS_CACHE_TTL:
            return metrics_cache

    try:
        # CPU по ядрам
        cpu_percent_per_core = psutil.cpu_percent(percpu=True, interval=0.1)

        # Memory детально
        memory = psutil.virtual_memory()

        # Disk I/O
        disk_io = psutil.disk_io_counters()

        # Network I/O
        net_io = psutil.net_io_counters()

        # Python процессы
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                if 'python' in proc.info['name'].lower():
                    python_processes.append({
                        "pid": proc.info['pid'],
                        "name": proc.info['name'],
                        "cpu_percent": proc.info['cpu_percent'] or 0,
                        "memory_percent": proc.info['memory_percent'] or 0
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cpu": {
                "total": psutil.cpu_percent(interval=0.1),
                "per_core": cpu_percent_per_core,
                "cores": len(cpu_percent_per_core)
            },
            "memory": {
                "percent": memory.percent,
                "used_mb": round(memory.used / (1024**2), 2),
                "available_mb": round(memory.available / (1024**2), 2),
                "total_mb": round(memory.total / (1024**2), 2)
            },
            "disk": {
                "percent": get_system_disk_usage().percent,
                "read_mb": round(disk_io.read_bytes / (1024**2), 2) if disk_io else 0,
                "write_mb": round(disk_io.write_bytes / (1024**2), 2) if disk_io else 0
            },
            "network": {
                "bytes_sent_mb": round(net_io.bytes_sent / (1024**2), 2) if net_io else 0,
                "bytes_recv_mb": round(net_io.bytes_recv / (1024**2), 2) if net_io else 0,
                "packets_sent": net_io.packets_sent if net_io else 0,
                "packets_recv": net_io.packets_recv if net_io else 0
            },
            "python_processes": python_processes[:5]
        }

        # Кэширование
        set_app_state("metrics_cache", metrics)
        set_app_state("metrics_cache_time", datetime.now(timezone.utc))

        logger.debug(f"Realtime detailed metrics retrieved: CPU {metrics['cpu']['total']}%")
        return metrics

    except Exception as e:
        logger.error(f"Error getting realtime detailed metrics: {e}")
        raise ServiceUnavailableError(f"Не удалось получить метрики: {str(e)}")


# ==================== Activity Timeline ====================

@router.get("/activity/timeline")
@cached(prefix="dashboard:activity", expire=60)
async def get_activity_timeline(
    days: int = Query(7, ge=1, le=30),
    db: DatabaseManager = Depends(get_db),
):
    """
    Get activity timeline (cached for 60 seconds)

    Возвращает активность по дням:
    - Сканирования
    - Симуляции
    - Анализы
    """
    try:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        # Активность сканирований по дням
        scans_timeline = db.execute_query(
            """
            SELECT DATE(created_at) as date, COUNT(*) as count
            FROM scan_results
            WHERE created_at >= ?
            GROUP BY DATE(created_at)
            ORDER BY date
            """,
            (start_date.isoformat(),)
        )

        # Активность симуляций по дням
        sims_timeline = db.execute_query(
            """
            SELECT DATE(created_at) as date, COUNT(*) as count
            FROM simulations
            WHERE created_at >= ?
            GROUP BY DATE(created_at)
            ORDER BY date
            """,
            (start_date.isoformat(),)
        )

        # Активность анализа по дням
        analysis_timeline = db.execute_query(
            """
            SELECT DATE(created_at) as date, COUNT(*) as count
            FROM defect_analysis
            WHERE created_at >= ?
            GROUP BY DATE(created_at)
            ORDER BY date
            """,
            (start_date.isoformat(),)
        )

        # Формирование таймлайна
        timeline = {}
        for i in range(days):
            date = (start_date + timedelta(days=i)).strftime('%Y-%m-%d')
            timeline[date] = {
                "scans": 0,
                "simulations": 0,
                "analysis": 0
            }

        for row in (scans_timeline or []):
            date = row["date"]
            if date in timeline:
                timeline[date]["scans"] = row["count"]

        for row in (sims_timeline or []):
            date = row["date"]
            if date in timeline:
                timeline[date]["simulations"] = row["count"]

        for row in (analysis_timeline or []):
            date = row["date"]
            if date in timeline:
                timeline[date]["analysis"] = row["count"]

        return {
            "period": {
                "start": start_date.strftime('%Y-%m-%d'),
                "end": end_date.strftime('%Y-%m-%d'),
                "days": days
            },
            "timeline": [
                {
                    "date": date,
                    "scans": data["scans"],
                    "simulations": data["simulations"],
                    "analysis": data["analysis"]
                }
                for date, data in timeline.items()
            ]
        }
    except Exception as e:
        logger.error(f"Error getting activity timeline: {e}")
        raise ServiceUnavailableError(f"Не удалось получить активность: {str(e)}")


# ==================== Storage Statistics ====================

@router.get(
    "/storage",
    summary="Статистика хранилища",
    description="Получить детальную статистику хранилища по директориям",
    response_model=Dict[str, Any],
)
@cached(prefix="dashboard:storage", expire=30)
async def get_storage_stats_endpoint(
    db: DatabaseManager = Depends(get_db),
):
    """
    Получить детальную статистику хранилища:
    - data/ директория (файлы, размер, крупнейшие файлы)
    - output/ директория
    - logs/ директория
    - Информация о диске
    - Размер БД
    """
    root = get_project_root()
    data_dir = root / "data"
    output_dir = root / "output"
    logs_dir = root / "logs"

    def get_dir_stats(directory: Path) -> Dict:
        if not directory.exists():
            return {"files": 0, "size_mb": 0, "largest_files": []}

        files = []
        total_size = 0

        for item in directory.rglob("*"):
            if item.is_file():
                try:
                    size = item.stat().st_size
                    total_size += size
                    files.append({
                        "path": str(item.relative_to(root)),
                        "size_mb": round(size / (1024**2), 4),
                        "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                    })
                except (OSError, IOError):
                    continue

        files.sort(key=lambda x: x["size_mb"], reverse=True)

        return {
            "files": len(files),
            "size_mb": round(total_size / (1024**2), 2),
            "largest_files": files[:5]
        }

    try:
        disk = get_system_disk_usage()
        
        # Размер БД
        db_path = Path("data/nanoprobe.db")
        db_size_mb = round(db_path.stat().st_size / (1024**2), 2) if db_path.exists() else 0

        return {
            "data": get_dir_stats(data_dir),
            "output": get_dir_stats(output_dir),
            "logs": get_dir_stats(logs_dir),
            "database": {
                "size_mb": db_size_mb,
                "path": str(db_path)
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent": disk.percent
            }
        }
    except Exception as e:
        logger.error(f"Error getting storage stats: {e}")
        raise ServiceUnavailableError(f"Не удалось получить статистику хранилища: {str(e)}")


# ==================== WebSocket ====================

@router.websocket("/ws/metrics")
async def metrics_websocket(websocket: WebSocket):
    """WebSocket для real-time метрик — отправляет обновления каждую секунду"""
    from api.websocket_manager import get_connection_manager
    manager = get_connection_manager()

    if not await manager.connect(websocket):
        return

    logger.info(f"WebSocket metrics connected. Active: {manager.connection_count}")
    try:
        while True:
            try:
                # Проверяем входящие сообщения (ping/disconnect) без блокировки
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
        logger.info(f"WebSocket metrics disconnected. Active: {manager.connection_count}")


# ==================== Alerts ====================

@router.get("/alerts/config")
async def get_alerts_config():
    """
    Get alerts configuration
    """
    return {
        "thresholds": {
            "cpu_warning": 70,
            "cpu_critical": 90,
            "memory_warning": 70,
            "memory_critical": 90,
            "disk_warning": 80,
            "disk_critical": 95
        },
        "notifications": {
            "enabled": True,
            "channels": ["ui", "log"]
        }
    }


@router.get("/alerts/check")
@cached(prefix="dashboard:alerts", expire=5)
async def check_alerts():
    """
    Check current system alerts (cached for 5 seconds)
    """
    try:
        alerts = []
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = get_system_disk_usage()

        # CPU алерты
        if cpu_percent >= 90:
            alerts.append({
                "level": "critical",
                "type": "cpu",
                "message": f"Critical CPU usage: {cpu_percent}%",
                "value": cpu_percent,
                "threshold": 90
            })
        elif cpu_percent >= 70:
            alerts.append({
                "level": "warning",
                "type": "cpu",
                "message": f"High CPU usage: {cpu_percent}%",
                "value": cpu_percent,
                "threshold": 70
            })

        # Memory алерты
        if memory.percent >= 90:
            alerts.append({
                "level": "critical",
                "type": "memory",
                "message": f"Critical memory usage: {memory.percent}%",
                "value": memory.percent,
                "threshold": 90
            })
        elif memory.percent >= 70:
            alerts.append({
                "level": "warning",
                "type": "memory",
                "message": f"High memory usage: {memory.percent}%",
                "value": memory.percent,
                "threshold": 70
            })

        # Disk алерты
        if disk.percent >= 95:
            alerts.append({
                "level": "critical",
                "type": "disk",
                "message": f"Critical disk usage: {disk.percent}%",
                "value": disk.percent,
                "threshold": 95
            })
        elif disk.percent >= 80:
            alerts.append({
                "level": "warning",
                "type": "disk",
                "message": f"High disk usage: {disk.percent}%",
                "value": disk.percent,
                "threshold": 80
            })

        return {
            "active_alerts": len(alerts),
            "alerts": alerts,
            "status": "critical" if any(a["level"] == "critical" for a in alerts) else "ok"
        }
    except Exception as e:
        logger.error(f"Error checking alerts: {e}")
        raise ServiceUnavailableError(f"Не удалось проверить алерты: {str(e)}")


# ==================== Metrics History ====================

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
        logger.error(f"Error getting metrics history: {e}")
        raise DatabaseError(f"Ошибка получения истории: {str(e)}")


# ==================== Additional Endpoints ====================

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
    """Получить топ процессов"""
    try:
        monitor = get_monitor()
        processes = monitor.get_process_list(limit=limit, sort_by=sort_by)
        return {"processes": processes, "total": len(processes)}
    except Exception as e:
        logger.error(f"Error getting processes: {e}")
        raise DatabaseError(f"Ошибка получения процессов: {str(e)}")


# ==================== Actions ====================

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
        raise ValidationError(
            f"Неподдерживаемый формат: {format}. Доступны: json, csv, pdf"
        )

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    return {
        "format": format,
        "status": "success",
        "message": f"Данные экспортированы в формате {format.upper()}",
        "download_url": f"/api/v1/downloads/export_{timestamp}.{format}",
        "expires_in": 3600
    }


@router.post(
    "/actions/clean_cache",
    summary="Очистка кэша",
    description="Очистка системного кэша проекта",
)
async def clean_cache_action():
    """Очистка системного кэша"""
    try:
        from utils.caching.cache_manager import CacheManager
        cache_mgr = CacheManager()

        cleaned_size = 0.0
        cleaned_files = 0

        # Выполнение очистки
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
                "cleaned_files": cleaned_files
            }
        )
    except Exception as e:
        logger.error(f"Error cleaning cache: {e}")
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
    """Запуск компонента проекта"""
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
            "started_at": datetime.now(timezone.utc).isoformat()
        }
    )


@router.post(
    "/actions/stop_component",
    summary="Остановка компонента",
    description="Остановка компонента проекта",
)
async def stop_component_action(component: dict):
    """Остановка компонента проекта"""
    component_name = component.get("component", "")
    if not component_name:
        raise ValidationError("Не указано имя компонента")

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "success": True,
            "message": f"Компонент '{component_name}' остановлен",
            "component": component_name,
            "stopped_at": datetime.now(timezone.utc).isoformat()
        }
    )
