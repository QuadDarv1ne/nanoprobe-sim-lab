"""
Enhanced Dashboard API with advanced metrics and real-time data
"""

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, HTTPException
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import psutil
from pathlib import Path
import asyncio
import json

from utils.database import DatabaseManager
from utils.redis_cache import cache

router = APIRouter()

# WebSocket подключения
active_websockets: List[WebSocket] = []

# Кэш для метрик
_metrics_cache: Optional[Dict[str, Any]] = None
_cache_timestamp: Optional[datetime] = None
CACHE_TTL = 1  # секунда для real-time метрик


@router.get("/stats/detailed")
@cache.cached(timeout=5, key_prefix="dashboard:stats:detailed")
async def get_detailed_stats():
    """
    Get detailed dashboard statistics (cached for 5 seconds)
    """
    db_manager = None
    try:
        db_manager = DatabaseManager("data/nanoprobe.db")

        # Базовая статистика
        scans_count = db_manager.count_scans()
        simulations_count = db_manager.count_simulations()
        analysis_count = db_manager.count_analysis_results()
        comparisons_count = db_manager.count_comparisons()
        reports_count = db_manager.count_reports()

        # Детальная статистика по сканированиям
        scans_by_type = db_manager.execute_query(
            "SELECT scan_type, COUNT(*) as count FROM scans GROUP BY scan_type"
        )

        # Последние активности
        recent_scans = db_manager.execute_query(
            "SELECT id, scan_type, resolution, created_at FROM scans ORDER BY created_at DESC LIMIT 5"
        )

        # Статистика по симуляциям
        sim_stats = db_manager.execute_query(
            "SELECT simulation_type, COUNT(*) as count, "
            "AVG(duration_sec) as avg_duration FROM simulations "
            "GROUP BY simulation_type"
        )

        # Использование ресурсов
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Uptime системы
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = datetime.now() - boot_time

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
                        "resolution": row["resolution"],
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
        raise HTTPException(status_code=500, detail=f"Failed to get detailed stats: {str(e)}")
    finally:
        if db_manager:
            db_manager.close_pool()


@router.get("/metrics/realtime")
async def get_realtime_metrics():
    """
    Get real-time system metrics
    Кэш: 1 секунда для real-time данных
    """
    global _metrics_cache, _cache_timestamp
    
    # Проверка кэша
    if _metrics_cache and _cache_timestamp:
        age = (datetime.now() - _cache_timestamp).total_seconds()
        if age < CACHE_TTL:
            return _metrics_cache
    
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
            "timestamp": datetime.now().isoformat(),
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
                "percent": psutil.disk_usage('/').percent,
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
        _metrics_cache = metrics
        _cache_timestamp = datetime.now()
        
        return metrics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get real-time metrics: {str(e)}")


@router.get("/activity/timeline")
@cache.cached(timeout=60, key_prefix="dashboard:activity:timeline")
async def get_activity_timeline(days: int = Query(7, ge=1, le=30)):
    """
    Get activity timeline (cached for 60 seconds)
    """
    db_manager = None
    try:
        db_manager = DatabaseManager("data/nanoprobe.db")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Активность сканирований по дням
        scans_timeline = db_manager.execute_query(
            """
            SELECT DATE(created_at) as date, COUNT(*) as count
            FROM scans
            WHERE created_at >= ?
            GROUP BY DATE(created_at)
            ORDER BY date
            """,
            (start_date.isoformat(),)
        )

        # Активность симуляций по дням
        sims_timeline = db_manager.execute_query(
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
        analysis_timeline = db_manager.execute_query(
            """
            SELECT DATE(created_at) as date, COUNT(*) as count
            FROM analysis_results
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
        raise HTTPException(status_code=500, detail=f"Failed to get activity timeline: {str(e)}")
    finally:
        if db_manager:
            db_manager.close_pool()


@router.get("/storage/detailed")
@cache.cached(timeout=30, key_prefix="dashboard:storage:detailed")
async def get_detailed_storage():
    """
    Get detailed storage statistics (cached for 30 seconds)
    """
    root = Path(__file__).parent.parent.parent
    data_dir = root / "data"
    output_dir = root / "output"
    logs_dir = root / "logs"

    def get_dir_stats(directory: Path) -> Dict:
        if not directory.exists():
            return {"files": 0, "size_mb": 0, "largest_file": None}

        files = []
        total_size = 0

        for item in directory.rglob("*"):
            if item.is_file():
                try:
                    size = item.stat().st_size
                    total_size += size
                    files.append({
                        "path": str(item.relative_to(root)),
                        "size_mb": round(size / (1024**2), 2),
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
        disk = psutil.disk_usage('/')

        return {
            "data": get_dir_stats(data_dir),
            "output": get_dir_stats(output_dir),
            "logs": get_dir_stats(logs_dir),
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent": disk.percent
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get storage stats: {str(e)}")


@router.websocket("/ws/metrics")
async def metrics_websocket(websocket: WebSocket):
    """
    WebSocket для real-time метрик
    Отправляет обновления каждую секунду
    """
    await websocket.accept()
    active_websockets.append(websocket)

    try:
        while True:
            metrics = await get_realtime_metrics()
            await websocket.send_json(metrics)
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        if websocket in active_websockets:
            active_websockets.remove(websocket)
    except Exception:
        if websocket in active_websockets:
            active_websockets.remove(websocket)
        raise


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
@cache.cached(timeout=5, key_prefix="dashboard:alerts:check")
async def check_alerts():
    """
    Check current system alerts (cached for 5 seconds)
    """
    try:
        alerts = []
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

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
            "timestamp": datetime.now().isoformat(),
            "alerts": alerts,
            "has_critical": any(a["level"] == "critical" for a in alerts),
            "has_warning": any(a["level"] == "warning" for a in alerts)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check alerts: {str(e)}")
