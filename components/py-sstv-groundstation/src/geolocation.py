"""
Модуль автоматического определения координат и часового пояса.
Является обёрткой над utils.location_manager для обеспечения обратной совместимости.
Все данные берутся из единого источника — location_manager.py
"""

# Публичный API модуля (re-export из location_manager)
__all__ = [
    "TZInfo",
    "MSK_TZ",
    "DEFAULT_LAT",
    "DEFAULT_LON",
    "DEFAULT_LOCATION",
    "CACHE_FILE",
    "CACHE_TTL_HOURS",
    "detect_location_by_ip",
    "load_location_cache",
    "save_location_cache",
    "get_location",
    "force_detect_and_save",
    "get_location_info",
    "now_msk",
    "utc_to_msk",
    "now_utc",
    "refresh_msk_data",
]

# Импортируем всё из location_manager для обеспечения обратной совместимости
try:
    from utils.location_manager import (
        CACHE_FILE,
        CACHE_TTL_HOURS,
        DEFAULT_LAT,
        DEFAULT_LON,
        MSK_TZ,
        TZInfo,
        detect_location_by_ip,
        force_detect_and_save,
        get_location,
        get_location_info,
        load_location_cache,
        now_msk,
        now_utc,
        refresh_msk_data,
        save_location_cache,
        utc_to_msk,
    )
except ImportError:
    # Fallback если utils.location_manager недоступен
    import sys
    from pathlib import Path

    _utils_path = Path(__file__).parent.parent.parent.parent / "utils" / "location_manager.py"
    if _utils_path.exists():
        sys.path.insert(0, str(_utils_path.parent.parent))
        from utils.location_manager import (
            CACHE_FILE,
            CACHE_TTL_HOURS,
            DEFAULT_LAT,
            DEFAULT_LON,
            MSK_TZ,
            TZInfo,
            detect_location_by_ip,
            force_detect_and_save,
            get_location,
            get_location_info,
            load_location_cache,
            now_msk,
            now_utc,
            refresh_msk_data,
            save_location_cache,
            utc_to_msk,
        )
    else:
        raise ImportError(
            "Не удалось импортировать utils.location_manager. "
            "Убедитесь, что проект запущен из корневой директории."
        )

# Для обратной совместимости создаём DEFAULT_LOCATION
DEFAULT_LOCATION = {
    "lat": DEFAULT_LAT,
    "lon": DEFAULT_LON,
    "city": "Москва",
    "country": "Россия",
    "timezone": MSK_TZ,
}
