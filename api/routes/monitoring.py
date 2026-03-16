"""
Performance Monitoring API

Endpoints для мониторинга производительности:
- /metrics - Prometheus metrics
- /health/detailed - Детальная информация о здоровье
- /monitoring/stats - Статистика мониторинга
"""

from fastapi import APIRouter, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import psutil
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


@router.get(
    "/metrics",
    summary="Prometheus Metrics",
    description="Получение Prometheus метрик для сбора",
)
async def get_metrics():
    """
    Prometheus metrics endpoint.
    
    Используется Prometheus server для сбора метрик.
    Content-Type: text/plain; version=0.0.4
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@router.get(
    "/health/detailed",
    summary="Detailed Health Check",
    description="Детальная проверка здоровья системы",
)
async def get_detailed_health():
    """
    Детальная информация о здоровье системы.
    
    Включает:
    - CPU usage
    - Memory usage
    - Disk usage
    - Network stats
    - Process info
    """
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_per_core = psutil.cpu_percent(percpu=True)
    cpu_freq = psutil.cpu_freq()
    
    # Memory
    memory = psutil.virtual_memory()
    
    # Disk
    disk = psutil.disk_usage('/')
    
    # Network
    net = psutil.net_io_counters()
    
    # Process
    process = psutil.Process()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": {
            "cpu": {
                "percent": cpu_percent,
                "per_core": cpu_per_core,
                "frequency_mhz": cpu_freq.current if cpu_freq else None,
            },
            "memory": {
                "percent": memory.percent,
                "available_mb": memory.available // (1024 * 1024),
                "total_mb": memory.total // (1024 * 1024),
            },
            "disk": {
                "percent": disk.percent,
                "free_gb": disk.free // (1024 * 1024 * 1024),
                "total_gb": disk.total // (1024 * 1024 * 1024),
            },
            "network": {
                "bytes_sent_mb": net.bytes_sent // (1024 * 1024),
                "bytes_recv_mb": net.bytes_recv // (1024 * 1024),
                "packets_sent": net.packets_sent,
                "packets_recv": net.packets_recv,
            },
        },
        "process": {
            "pid": process.pid,
            "cpu_percent": process.cpu_percent(),
            "memory_percent": process.memory_percent(),
            "num_threads": process.num_threads(),
            "status": process.status(),
            "uptime_seconds": datetime.now(timezone.utc).timestamp() - process.create_time(),
        },
    }


@router.get(
    "/stats",
    summary="Monitoring Statistics",
    description="Статистика мониторинга",
)
async def get_monitoring_stats():
    """
    Статистика мониторинга.

    Включает:
    - Uptime
    - Request counts
    - Error rates
    """
    import time

    # Получаем uptime
    boot_time = datetime.fromtimestamp(psutil.boot_time(), tz=timezone.utc)
    uptime = datetime.now(timezone.utc) - boot_time

    return {
        "uptime": {
            "seconds": uptime.total_seconds(),
            "formatted": str(uptime),
            "boot_time": boot_time.isoformat(),
        },
        "cpu": {
            "cores": psutil.cpu_count(),
            "logical_cores": psutil.cpu_count(logical=True),
        },
        "memory": {
            "total_gb": psutil.virtual_memory().total // (1024 * 1024 * 1024),
        },
        "disk": {
            "partitions": len(psutil.disk_partitions()),
        },
    }
