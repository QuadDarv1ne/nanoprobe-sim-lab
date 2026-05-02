"""
Централизованный менеджер местоположения и часового пояса (МСК).
Используется всеми модулями проекта для автоматической актуализации координат и времени.

Приоритет источников (по убыванию):
1. Явные параметры (lat/lon переданные в функцию)
2. Переменные окружения GROUND_STATION_LAT / GROUND_STATION_LON
3. Автоопределение по IP (кэш 24ч)
4. Дефолт: Москва, МСК (UTC+3)
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)


class TZInfo:
    """Простой класс для представления часового пояса."""

    def __init__(self, name: str, utc_offset: int):
        self.name = name
        self.utc_offset = utc_offset

    def to_local(self, dt_utc: datetime) -> datetime:
        """Конвертирует UTC время в локальное (возвращает naive datetime)."""
        if dt_utc.tzinfo is not None:
            dt_utc = dt_utc.replace(tzinfo=None)
        return dt_utc + timedelta(hours=self.utc_offset)

    def now_local(self) -> datetime:
        """Возвращает текущее локальное время."""
        return self.to_local(datetime.now(timezone.utc))

    def __eq__(self, other):
        """Сравнивает два объекта TZInfo по их атрибутам."""
        if not isinstance(other, TZInfo):
            return False
        return self.name == other.name and self.utc_offset == other.utc_offset

    def __repr__(self):
        return f"TZInfo({self.name}, UTC{'+' if self.utc_offset >= 0 else ''}{self.utc_offset})"


# Часовой пояс МСК (UTC+3) — используется для всех расчётов
MSK_TZ = TZInfo("MSK", 3)

# Дефолтные координаты (Москва)
DEFAULT_LAT = 55.7558
DEFAULT_LON = 37.6173

# Кэш геолокации
CACHE_FILE = Path("data/location_cache.json")
CACHE_TTL_HOURS = 24  # Обновлять не чаще раза в сутки


def now_msk() -> datetime:
    """Возвращает текущее время в МСК (UTC+3)."""
    return MSK_TZ.now_local()


def utc_to_msk(dt_utc: datetime) -> datetime:
    """Конвертирует UTC время в МСК."""
    return MSK_TZ.to_local(dt_utc)


def now_utc() -> datetime:
    """Возвращает текущее время UTC (наивное, для совместимости с SGP4)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def detect_location_by_ip() -> Optional[Dict]:
    """
    Определяет местоположение по IP через бесплатные API.
    Пробует несколько сервисов для надёжности.
    """
    # ip-api.com (бесплатный, без ключа)
    try:
        resp = requests.get(
            "http://ip-api.com/json/?fields=lat,lon,city,country,timezone", timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("lat") and data.get("lon"):
            tz_offset = 0
            tz_name = "LOCAL"
            tz_str = data.get("timezone", "")
            if tz_str:
                try:
                    import zoneinfo

                    tz = zoneinfo.ZoneInfo(tz_str)
                    offset = tz.utcoffset(datetime.now(timezone.utc))
                    tz_offset = int(offset.total_seconds() // 3600)
                    tz_name = tz_str.split("/")[-1].replace("_", " ")
                except Exception as e:
                    logger.debug(f"Failed to parse timezone info: {e}")
                    tz_offset = 0
                    tz_name = "LOCAL"
            return {
                "lat": float(data["lat"]),
                "lon": float(data["lon"]),
                "city": data.get("city", "Unknown"),
                "country": data.get("country", "Unknown"),
                "timezone": TZInfo(tz_name, tz_offset),
            }
    except Exception as e:
        logger.debug(f"ip-api.com geolocation failed: {e}")

    # ipapi.co
    try:
        resp = requests.get("https://ipapi.co/json/", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("latitude") and data.get("longitude"):
            tz_offset = int(data.get("utc_offset", 0)[:3]) if data.get("utc_offset") else 0
            return {
                "lat": float(data["latitude"]),
                "lon": float(data["longitude"]),
                "city": data.get("city", "Unknown"),
                "country": data.get("country_name", "Unknown"),
                "timezone": TZInfo(data.get("timezone", "LOCAL"), tz_offset),
            }
    except Exception as e:
        logger.debug(f"ipapi.co geolocation failed: {e}")

    return None


def load_location_cache() -> Optional[Dict]:
    """Загружает местоположение из кэша."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
            cached_time = datetime.fromisoformat(cache.get("timestamp", "")).replace(
                tzinfo=timezone.utc
            )
            if datetime.now(timezone.utc) - cached_time < timedelta(hours=CACHE_TTL_HOURS):
                return cache
        except Exception as e:
            logger.debug(f"Failed to load location cache: {e}")
    return None


def save_location_cache(location: Dict):
    """Сохраняет местоположение в кэш."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    cache = {
        **location,
        "timezone_name": (
            location["timezone"].name if hasattr(location["timezone"], "name") else "MSK"
        ),
        "timezone_offset": (
            location["timezone"].utc_offset if hasattr(location["timezone"], "utc_offset") else 3
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    cache.pop("timezone", None)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def get_location(
    force_detect: bool = False, use_env: bool = True, auto_refresh: bool = True
) -> Dict:
    """
    Получает местоположение с автоматическим определением.

    Приоритет:
    1. Переменные окружения GROUND_STATION_LAT / GROUND_STATION_LON (если use_env=True)
    2. Кэш геолокации (если не force_detect)
    3. Автоопределение по IP
    4. Дефолт: Москва, МСК

    Args:
        force_detect: Принудительно определить по IP
        use_env: Проверять переменные окружения
        auto_refresh: Автоматически обновлять кэш если он устарел

    Returns:
        Dict: lat, lon, city, country, timezone (TZInfo)
    """
    # 1. Переменные окружения
    if use_env:
        env_lat = os.getenv("GROUND_STATION_LAT")
        env_lon = os.getenv("GROUND_STATION_LON")
        if env_lat and env_lon:
            try:
                return {
                    "lat": float(env_lat),
                    "lon": float(env_lon),
                    "city": os.getenv("GROUND_STATION_CITY", "Configured"),
                    "country": os.getenv("GROUND_STATION_COUNTRY", ""),
                    "timezone": MSK_TZ,
                }
            except ValueError:
                pass

    # 2. Кэш
    if not force_detect:
        cached = load_location_cache()
        if cached:
            tz = TZInfo(cached.get("timezone_name", "MSK"), cached.get("timezone_offset", 3))
            location = {
                "lat": cached["lat"],
                "lon": cached["lon"],
                "city": cached.get("city", "Unknown"),
                "country": cached.get("country", "Unknown"),
                "timezone": tz,
            }

            # Если включено автообновление и кэш устарел - обновляем в фоне
            if auto_refresh:
                cached_time = datetime.fromisoformat(cached.get("timestamp", "")).replace(
                    tzinfo=timezone.utc
                )
                if datetime.now(timezone.utc) - cached_time >= timedelta(hours=CACHE_TTL_HOURS):
                    # Запускаем фоновое обновление
                    import threading

                    thread = threading.Thread(target=_background_location_refresh, daemon=True)
                    thread.start()

            return location

    # 3. Автоопределение по IP
    location = detect_location_by_ip()
    if location:
        save_location_cache(location)
        return location

    # 4. Дефолт
    return {
        "lat": DEFAULT_LAT,
        "lon": DEFAULT_LON,
        "city": "Москва",
        "country": "Россия",
        "timezone": MSK_TZ,
    }


def _background_location_refresh():
    """Фоновое обновление кэша геолокации."""
    try:
        location = detect_location_by_ip()
        if location:
            save_location_cache(location)
    except Exception as e:
        logger.debug(f"Background location refresh failed: {e}")


def force_detect_and_save() -> Optional[Dict]:
    """Принудительно определить местоположение и сохранить."""
    logger.info("Opredelenie mestopolozheniya po IP...")
    location = detect_location_by_ip()
    if location:
        save_location_cache(location)
        logger.info("OK: %s, %s", location["city"], location["country"])
        logger.info("   Coords: %.4fN, %.4fE", location["lat"], location["lon"])
        tz = location["timezone"]
        sign = "+" if tz.utc_offset >= 0 else ""
        logger.info("   Timezone: %s (UTC%s%d)", tz.name, sign, tz.utc_offset)
        return location
    else:
        logger.error("ne udalos opredelit mestopolozhenie po IP")
        return None


def get_location_info() -> str:
    """Возвращает читаемую строку с информацией о местоположении."""
    loc = get_location()
    tz = loc["timezone"]
    return (
        f"[LOC] {loc['city']}, {loc['country']}\n"
        f"   Coords: {loc['lat']:.4f}N, {loc['lon']:.4f}E\n"
        f"   Timezone: {tz.name} (UTC{'+' if tz.utc_offset >= 0 else ''}{tz.utc_offset})\n"
        f"   Current time: {tz.now_local().strftime('%H:%M:%S')}"
    )


def refresh_msk_data() -> Optional[Dict]:
    """
    Принудительно обновляет данные МСК (координаты и часовой пояс).
    Используется для гарантированной актуализации всех расчётов.

    Returns:
        Dict: Обновлённые данные местоположения или None при ошибке
    """
    logger.info("[MSK] Обновление данных геолокации...")
    location = detect_location_by_ip()
    if location:
        save_location_cache(location)
        logger.info(
            "[MSK] ✓ Местоположение обновлено: %s, %s", location["city"], location["country"]
        )
        logger.info("[MSK]   Координаты: %.4f°N, %.4f°E", location["lat"], location["lon"])
        logger.info(
            "[MSK]   Часовой пояс: %s (UTC%s%d)",
            location["timezone"].name,
            "+" if location["timezone"].utc_offset >= 0 else "",
            location["timezone"].utc_offset,
        )
        return location
    else:
        logger.error("[MSK] ✗ Не удалось обновить местоположение по IP")
        return None


if __name__ == "__main__":
    import logging
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if len(sys.argv) > 1 and sys.argv[1] == "--force":
        force_detect_and_save()
    else:
        logger.info(get_location_info())
