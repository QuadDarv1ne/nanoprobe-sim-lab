"""
System health checks for Nanoprobe Sim Lab

Единый модуль для проверки состояния системы.
Используется в /health/detailed и /dashboard/health/detailed.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

import psutil

from api.state import get_system_disk_usage

logger = logging.getLogger(__name__)

# Пороги для алертов
CPU_WARNING = 70
CPU_CRITICAL = 90
MEMORY_WARNING = 70
MEMORY_CRITICAL = 90
DISK_WARNING = 80
DISK_CRITICAL = 90


def compute_system_health() -> Dict[str, Any]:
    """
    Проверка состояния системы.

    Returns:
        Dict с полями:
            - status: "healthy" | "info" | "warning" | "critical"
            - cpu_percent: загрузка CPU
            - memory_percent: загрузка памяти
            - disk_percent: загрузка диска
            - issues: список проблем
            - timestamp: время проверки
    """
    try:
        cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = get_system_disk_usage()

        status: str = "healthy"
        issues: List[str] = []

        # CPU проверка
        if cpu > CPU_CRITICAL:
            status = "critical"
            issues.append(f"CPU критическая: {cpu:.1f}%")
        elif cpu > CPU_WARNING:
            if status == "healthy":
                status = "warning"
            issues.append(f"Высокая CPU: {cpu:.1f}%")
        elif cpu > CPU_WARNING * 0.7 and status == "healthy":  # > 49%
            status = "info"

        # Memory проверка
        mem_percent = memory.percent
        if mem_percent > MEMORY_CRITICAL:
            status = "critical"
            issues.append(f"Память критическая: {mem_percent:.1f}%")
        elif mem_percent > MEMORY_WARNING:
            if status == "healthy":
                status = "warning"
            issues.append(f"Высокая память: {mem_percent:.1f}%")

        # Disk проверка
        disk_percent = disk.percent
        if disk_percent > DISK_CRITICAL:
            status = "critical"
            issues.append(f"Диск критический: {disk_percent:.1f}%")
        elif disk_percent > DISK_WARNING:
            if status == "healthy":
                status = "warning"
            issues.append(f"Высокий диск: {disk_percent:.1f}%")

        return {
            "status": status,
            "cpu_percent": cpu,
            "memory_percent": mem_percent,
            "disk_percent": disk_percent,
            "issues": issues,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.exception("System health check failed: %s", e)
        return {
            "status": "critical",
            "cpu_percent": None,
            "memory_percent": None,
            "disk_percent": None,
            "issues": [f"Health check error: {str(e)}"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
