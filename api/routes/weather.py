"""
Weather API for Nanoprobe Sim Lab
Provides weather forecasts for any location using Open-Meteo (free, no API key).
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Query

from api.error_handlers import ServiceUnavailableError

logger = logging.getLogger(__name__)

router = APIRouter()

# Weather codes mapping (WMO)
WEATHER_CODES = {
    0: "Ясно", 1: "Преимущественно ясно", 2: "Переменная облачность",
    3: "Пасмурно", 45: "Туман", 48: "Туман с изморозью",
    51: "Лёгкая морось", 53: "Морось", 55: "Сильная морось",
    56: "Ледяная морось", 57: "Сильная ледяная морось",
    61: "Небольшой дождь", 63: "Дождь", 65: "Сильный дождь",
    66: "Ледяной дождь", 67: "Сильный ледяной дождь",
    71: "Небольшой снег", 73: "Снег", 75: "Сильный снег",
    77: "Снежная крупа", 80: "Небольшой ливень", 81: "Ливень",
    82: "Сильный ливень", 85: "Снегопад", 86: "Сильный снегопад",
    95: "Гроза", 96: "Гроза с градом", 99: "Сильная гроза с градом",
}

# Known locations
LOCATIONS = {
    "odintsovo": {"lat": 55.67, "lon": 37.28, "name": "Одинцово, Московская область"},
    "moscow": {"lat": 55.75, "lon": 37.62, "name": "Москва"},
    "iss": {"lat": 51.6, "lon": -0.1, "name": "МКС (низкая орбита)"},
}


def _get_weather_desc(code: int) -> str:
    return WEATHER_CODES.get(code, f"Код {code}")


@router.get("/weather/{location}")
async def get_weather(
    location: str,
    days: int = Query(default=3, ge=1, le=7, description="Количество дней прогноза (1-7)"),
):
    """
    Прогноз погоды для заданного местоположения.

    - **location**: Название города (odintsovo, moscow) или координаты lat,lon
    - **days**: Количество дней прогноза

    Примеры:
    - /weather/odintsovo
    - /weather/moscow?days=5
    - /weather/55.67,37.28?days=3
    """
    # Resolve location
    loc_key = location.lower().strip()
    if loc_key in LOCATIONS:
        loc = LOCATIONS[loc_key]
        lat, lon = loc["lat"], loc["lon"]
        name = loc["name"]
    elif "," in location:
        try:
            parts = location.split(",")
            lat, lon = float(parts[0].strip()), float(parts[1].strip())
            name = f"{lat:.4f}, {lon:.4f}"
        except (ValueError, IndexError):
            raise ServiceUnavailableError(f"Invalid coordinates: {location}")
    else:
        raise ServiceUnavailableError(
            f"Unknown location: {location}. Available: {', '.join(LOCATIONS.keys())}"
        )

    # Fetch from Open-Meteo (free, no API key)
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum,"
        f"windspeed_10m_max,winddirection_10m_dominant,uv_index_max,sunrise,sunset"
        f"&hourly=temperature_2m,relativehumidity_2m,weathercode,windspeed_10m"
        f"&current=temperature_2m,relativehumidity_2m,apparent_temperature,"
        f"weathercode,windspeed_10m,winddirection_10m,pressure_msl"
        f"&timezone=auto&forecast_days={days}"
    )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        raise ServiceUnavailableError(f"Weather API error: {e}")

    # Build response
    current = data.get("current", {})
    daily = data.get("daily", {})
    hourly = data.get("hourly", {})

    # Current weather
    current_weather = {
        "temperature": current.get("temperature_2m"),
        "feels_like": current.get("apparent_temperature"),
        "humidity": current.get("relativehumidity_2m"),
        "wind_speed": current.get("windspeed_10m"),
        "wind_direction": current.get("winddirection_10m"),
        "pressure": current.get("pressure_msl"),
        "weather_code": current.get("weathercode"),
        "weather_description": _get_weather_desc(current.get("weathercode", -1)),
        "timestamp": current.get("time", datetime.now(timezone.utc).isoformat()),
    }

    # Daily forecasts
    forecasts = []
    dates = daily.get("time", [])
    for i, date_str in enumerate(dates):
        wcode = daily.get("weathercode", [None] * len(dates))
        forecasts.append({
            "date": date_str,
            "temp_max": daily.get("temperature_2m_max", [None] * len(dates))[i],
            "temp_min": daily.get("temperature_2m_min", [None] * len(dates))[i],
            "precipitation": daily.get("precipitation_sum", [None] * len(dates))[i],
            "wind_speed_max": daily.get("windspeed_10m_max", [None] * len(dates))[i],
            "wind_direction": daily.get("winddirection_10m_dominant", [None] * len(dates))[i],
            "uv_index_max": daily.get("uv_index_max", [None] * len(dates))[i],
            "sunrise": daily.get("sunrise", [None] * len(dates))[i],
            "sunset": daily.get("sunset", [None] * len(dates))[i],
            "weather_code": wcode[i] if isinstance(wcode, list) else wcode,
            "weather_description": _get_weather_desc(wcode[i] if isinstance(wcode, list) else wcode),
        })

    # Hourly for today (first 24 hours)
    hourly_times = hourly.get("time", [])[:24]
    hourly_data = []
    for i, t in enumerate(hourly_times):
        hourly_data.append({
            "time": t,
            "temperature": hourly.get("temperature_2m", [None] * len(hourly_times))[i],
            "humidity": hourly.get("relativehumidity_2m", [None] * len(hourly_times))[i],
            "weather_code": hourly.get("weathercode", [None] * len(hourly_times))[i],
            "weather_description": _get_weather_desc(
                hourly.get("weathercode", [None] * len(hourly_times))[i]
            ),
            "wind_speed": hourly.get("windspeed_10m", [None] * len(hourly_times))[i],
        })

    return {
        "status": "success",
        "location": name,
        "coordinates": {"latitude": lat, "longitude": lon},
        "current": current_weather,
        "forecast_days": days,
        "daily": forecasts,
        "hourly": hourly_data[:24],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
