"""
SSTV API helpers

Общие утилиты и хелперы для SSTV модулей.
"""

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from api.state import get_app_state, get_redis, set_app_state

logger = logging.getLogger(__name__)

# Добавляем корень проекта в path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.caching.redis_cache import RedisCache

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Импорт SSTV компонентов
try:
    components_path = PROJECT_ROOT / "components" / "py-sstv-groundstation" / "src"
    sys.path.insert(0, str(components_path))

    import satellite_tracker
    from sstv_decoder import SSTVDecoder

    tracker_module = satellite_tracker
    SSTV_AVAILABLE = True
except ImportError:
    SSTV_AVAILABLE = False
    tracker_module = None
    SSTVDecoder = None


def get_redis_cache() -> Optional[RedisCache]:
    """Получает Redis cache instance из api.state."""
    return get_redis()


def get_satellite_tracker() -> Optional[Any]:
    """Получает SatelliteTracker instance."""
    if tracker_module is None:
        return None

    tracker = get_app_state("satellite_tracker")
    if tracker is not None:
        return tracker

    try:
        lat = float(os.getenv("GROUND_STATION_LAT", "55.75"))
        lon = float(os.getenv("GROUND_STATION_LON", "37.61"))
        tracker = tracker_module.SatelliteTracker(
            ground_station_lat=lat,
            ground_station_lon=lon,
        )
        set_app_state("satellite_tracker", tracker)
    except Exception as e:
        logger.warning(f"SatelliteTracker initialization error: {e}")
        return None

    return tracker


def get_sstv_decoder() -> Optional[Any]:
    """Получает SSTVDecoder instance."""
    if SSTVDecoder is None:
        return None

    decoder = get_app_state("sstv_decoder")
    if decoder is not None:
        return decoder

    try:
        decoder = SSTVDecoder()
        set_app_state("sstv_decoder", decoder)
    except Exception as e:
        logger.warning(f"SSTVDecoder initialization error: {e}")
        return None

    return decoder
