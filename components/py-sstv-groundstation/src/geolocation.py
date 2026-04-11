"""
Модуль автоматического определения координат и часового пояса.
Использует IP-геолокацию через бесплатные API.
"""

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import requests


class TZInfo:
    """Простой класс для представления часового пояса."""

    def __init__(self, name: str, utc_offset: int):
        """
        Args:
            name: Название часового пояса (например, 'MSK')
            utc_offset: Смещение от UTC в часах (например, 3 для МСК)
        """
        self.name = name
        self.utc_offset = utc_offset

    def to_local(self, dt_utc: datetime) -> datetime:
        """Конвертирует UTC время в локальное."""
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        return dt_utc + timedelta(hours=self.utc_offset)

    def now_local(self) -> datetime:
        """Возвращает текущее локальное время."""
        return self.to_local(datetime.now(timezone.utc))

    def __repr__(self):
        return f"TZInfo({self.name}, UTC{'+' if self.utc_offset >= 0 else ''}{self.utc_offset})"


# Координаты и часовой пояс по умолчанию (Москва, МСК = UTC+3)
DEFAULT_LOCATION = {
    "lat": 55.7558,
    "lon": 37.6173,
    "city": "Москва",
    "country": "Россия",
    "timezone": TZInfo("MSK", 3),
}

# Кэш геолокации
CACHE_FILE = Path("data/location_cache.json")
CACHE_TTL_HOURS = 24  # Обновлять не чаще раза в сутки


def detect_location_by_ip() -> Optional[Dict]:
    """
    Определяет местоположение по IP через бесплатные API.
    Пробует несколько сервисов для надёжности.

    Returns:
        Dict с ключами: lat, lon, city, country, timezone (TZInfo) или None
    """
    # Пробуем ip-api.com (бесплатный, без ключа)
    try:
        resp = requests.get(
            "http://ip-api.com/json/?fields=lat,lon,city,country,timezone", timeout=10
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("lat") and data.get("lon"):
            # Определяем смещение часового пояса
            tz_offset = 0
            tz_name = "LOCAL"
            tz_str = data.get("timezone", "")
            if tz_str:
                try:
                    import zoneinfo

                    tz = zoneinfo.ZoneInfo(tz_str)
                    now = datetime.now(timezone.utc)
                    offset = tz.utcoffset(now)
                    tz_offset = int(offset.total_seconds() // 3600)
                    tz_name = tz_str.split("/")[-1].replace("_", " ")
                except Exception:
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
        print(f"⚠ ip-api.com не доступен: {e}")

    # Пробуем ipapi.co
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
        print(f"⚠ ipapi.co не доступен: {e}")

    return None


def load_location_cache() -> Optional[Dict]:
    """Загружает местоположение из кэша."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)

            # Проверяем возраст кэша
            cached_time = datetime.fromisoformat(cache.get("timestamp", "")).replace(
                tzinfo=timezone.utc
            )
            if datetime.now(timezone.utc) - cached_time < timedelta(hours=CACHE_TTL_HOURS):
                return cache
        except Exception:
            pass
    return None


def save_location_cache(location: Dict):
    """Сохраняет местоположение в кэш."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    cache = {
        **location,
        "timezone_name": location["timezone"].name,
        "timezone_offset": location["timezone"].utc_offset,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    # Удаляем TZInfo объект, т.к. он не сериализуется
    cache.pop("timezone", None)

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def get_location(force_detect: bool = False) -> Dict:
    """
    Получает местоположение с автоопределением.
    Приоритет: кэш → IP-геолокация → дефолт (Москва).

    Args:
        force_detect: Принудительно определить по IP

    Returns:
        Dict с ключами: lat, lon, city, country, timezone (TZInfo)
    """
    # Если не принудительно — пробуем кэш
    if not force_detect:
        cached = load_location_cache()
        if cached:
            tz = TZInfo(cached.get("timezone_name", "LOCAL"), cached.get("timezone_offset", 0))
            return {
                "lat": cached["lat"],
                "lon": cached["lon"],
                "city": cached.get("city", "Unknown"),
                "country": cached.get("country", "Unknown"),
                "timezone": tz,
            }

    # Пробуем IP-геолокацию
    location = detect_location_by_ip()
    if location:
        save_location_cache(location)
        return location

    # Фоллбэк на дефолт
    print("⚠ Не удалось определить местоположение, используется Москва (МСК, UTC+3)")
    return DEFAULT_LOCATION.copy()


def force_detect_and_save():
    """Принудительно определить местоположение и сохранить."""
    print("🌍 Определение местоположения по IP...")
    location = detect_location_by_ip()
    if location:
        save_location_cache(location)
        print(f"✅ Местоположение определено: {location['city']}, {location['country']}")
        print(f"   Координаты: {location['lat']:.4f}°N, {location['lon']:.4f}°E")
        print(
            f"   Часовой пояс: {location['timezone'].name} (UTC{'+' if location['timezone'].utc_offset >= 0 else ''}{location['timezone'].utc_offset})"
        )
        return location
    else:
        print("❌ Не удалось определить местоположение по IP")
        return None


def get_msk_timezone() -> TZInfo:
    """Возвращает часовой пояс МСК (UTC+3)."""
    return TZInfo("MSK", 3)


def now_msk() -> datetime:
    """Возвращает текущее время в МСК."""
    return get_msk_timezone().now_local()


def utc_to_msk(dt_utc: datetime) -> datetime:
    """Конвертирует UTC время в МСК."""
    return get_msk_timezone().to_local(dt_utc)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--force":
        force_detect_and_save()
    else:
        loc = get_location()
        print(f"📍 Местоположение: {loc['city']}, {loc['country']}")
        print(f"   Координаты: {loc['lat']:.4f}°N, {loc['lon']:.4f}°E")
        print(
            f"   Часовой пояс: {loc['timezone'].name} (UTC{'+' if loc['timezone'].utc_offset >= 0 else ''}{loc['timezone'].utc_offset})"
        )
        print(f"   Текущее время: {loc['timezone'].now_local().strftime('%H:%M:%S')}")
