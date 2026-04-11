"""
Dashboard helpers

Утилиты, кэширование, константы.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from api.state import get_app_state, get_system_disk_usage, set_app_state

logger = logging.getLogger(__name__)

# Кэш для статистики (5 секунд)
STATS_CACHE_TTL = 5  # секунд

# Кэш для метрик (1 секунда для real-time)
METRICS_CACHE_TTL = 1  # секунда

# Префиксы для Redis кэша
CACHE_PREFIX = {
    "stats": "dashboard:stats",
    "metrics": "dashboard:metrics",
    "health": "dashboard:health",
    "storage": "dashboard:storage",
    "activity": "dashboard:activity",
    "alerts": "dashboard:alerts",
}


def get_project_root() -> Path:
    """Получить корень проекта."""
    return Path(__file__).parent.parent.parent


def get_cached_stats() -> Optional[Dict]:
    """Получить кэшированную статистику если не истёк TTL."""
    cached_data = get_app_state("stats_cache")
    cache_time = get_app_state("stats_cache_time")

    if cache_time is None or cached_data is None:
        return None

    age = (datetime.now(timezone.utc) - cache_time).total_seconds()
    if age < STATS_CACHE_TTL:
        return cached_data
    return None


def cache_stats(stats: Dict):
    """Закэшировать статистику."""
    set_app_state("stats_cache", stats)
    set_app_state("stats_cache_time", datetime.now(timezone.utc))


def get_storage_stats() -> Dict[str, float]:
    """Получить статистику хранилища."""
    root = get_project_root()
    data_dir = root / "data"
    output_dir = root / "output"

    used_mb = 0.0
    total_mb = 0.0

    for directory in [data_dir, output_dir]:
        if directory.exists():
            for item in directory.rglob("*"):
                if item.is_file():
                    try:
                        used_mb += item.stat().st_size / (1024 * 1024)
                    except (OSError, IOError):
                        continue

    disk = get_system_disk_usage()
    total_mb = disk.total / (1024 * 1024)

    return {
        "used_mb": round(used_mb, 2),
        "total_mb": round(total_mb, 2),
        "percent": round((used_mb / total_mb) * 100, 2) if total_mb > 0 else 0,
    }
