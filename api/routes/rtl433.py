"""
RTL_433 API Routes
Endpoints for weather station and sensor data
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from utils.caching.cache_manager import CacheManager
from utils.database import DatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rtl433", tags=["RTL-433"])

# Global references (set during app startup)
_db_manager: Optional[DatabaseManager] = None
_cache_manager: Optional[CacheManager] = None


def set_managers(db: DatabaseManager, cache: CacheManager):
    """Set database and cache managers"""
    global _db_manager, _cache_manager
    _db_manager = db
    _cache_manager = cache


def _get_db() -> DatabaseManager:
    """Get database manager or raise error"""
    if _db_manager is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    return _db_manager


class RTL433Reading(BaseModel):
    """RTL_433 reading model"""

    id: int
    model: str
    device_id: str = Field(alias="device_id")
    channel: Optional[int] = None
    battery_ok: Optional[int] = None
    temperature_c: Optional[float] = None
    humidity: Optional[float] = None
    pressure_hpa: Optional[float] = None
    wind_speed_kmh: Optional[float] = None
    rain_mm: Optional[float] = None
    created_at: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class RTL433DeviceSummary(BaseModel):
    """RTL_433 device summary model"""

    model: str
    device_id: str
    channel: Optional[int] = None
    reading_count: int
    last_seen: Optional[str] = None
    avg_temperature_c: Optional[float] = None
    avg_humidity: Optional[float] = None


class RTL433ReadingsResponse(BaseModel):
    """Response model for readings list"""

    items: list[RTL433Reading]
    total: int
    limit: int
    offset: int


class RTL433DevicesResponse(BaseModel):
    """Response model for device summary"""

    devices: list[RTL433DeviceSummary]
    total_devices: int


@router.get("/readings", response_model=RTL433ReadingsResponse)
async def get_readings(
    limit: int = Query(50, ge=1, le=500, description="Number of readings to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    model: Optional[str] = Query(None, description="Filter by device model"),
    device_id: Optional[str] = Query(None, description="Filter by device ID"),
):
    """
    Get recent RTL_433 readings.

    Returns sensor data from weather stations, temperature/humidity sensors,
    and other 433/868/915 MHz ISM band devices.
    """
    cache_key = f"rtl433:readings:{limit}:{offset}:{model}:{device_id}"

    # Try cache
    if _cache_manager:
        cached = _cache_manager.get(cache_key)
        if cached:
            return cached

    # Ensure table exists
    _ensure_table()

    # Query database
    readings = _get_readings_db(limit, offset, model, device_id)

    response = RTL433ReadingsResponse(
        items=readings["items"],
        total=readings["total"],
        limit=limit,
        offset=offset,
    )

    # Cache for 5 seconds
    if _cache_manager:
        _cache_manager.set(cache_key, response, expire=5)

    return response


@router.get("/devices", response_model=RTL433DevicesResponse)
async def get_devices():
    """
    Get unique devices summary.

    Returns aggregated data for each detected device including
    reading count, last seen time, and averages.
    """
    cache_key = "rtl433:devices"

    # Try cache
    if _cache_manager:
        cached = _cache_manager.get(cache_key)
        if cached:
            return cached

    # Ensure table exists
    _ensure_table()

    # Query database
    devices = _get_devices_db()

    response = RTL433DevicesResponse(
        devices=devices["devices"],
        total_devices=devices["total"],
    )

    # Cache for 30 seconds
    if _cache_manager:
        _cache_manager.set(cache_key, response, expire=30)

    return response


@router.get("/stats")
async def get_stats():
    """
    Get RTL_433 statistics.

    Returns total readings, unique devices, and time range.
    """
    cache_key = "rtl433:stats"

    # Try cache
    if _cache_manager:
        cached = _cache_manager.get(cache_key)
        if cached:
            return cached

    # Ensure table exists
    _ensure_table()

    db = _get_db()
    with db.get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM rtl433_readings")
        total_readings = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT device_id) FROM rtl433_readings")
        unique_devices = cursor.fetchone()[0]

        cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM rtl433_readings")
        row = cursor.fetchone()
        first_reading = row[0]
        last_reading = row[1]

    stats = {
        "total_readings": total_readings,
        "unique_devices": unique_devices,
        "first_reading": first_reading,
        "last_reading": last_reading,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }

    # Cache for 10 seconds
    if _cache_manager:
        _cache_manager.set(cache_key, stats, expire=10)

    return stats


@router.post("/clear")
async def clear_data():
    """
    Clear all RTL_433 data.

    WARNING: This deletes all sensor readings!
    """
    _ensure_table()

    db = _get_db()
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM rtl433_readings")
        deleted = cursor.rowcount
        conn.commit()

    # Invalidate caches
    if _cache_manager:
        _cache_manager.delete_by_pattern("rtl433:*")

    return {
        "message": f"Cleared {deleted} readings",
        "deleted_count": deleted,
    }


def _ensure_table():
    """Create rtl433_readings table if not exists"""
    db = _get_db()
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS rtl433_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                device_id TEXT NOT NULL,
                channel INTEGER,
                battery_ok INTEGER,
                temperature_c REAL,
                humidity REAL,
                pressure_hpa REAL,
                wind_speed_kmh REAL,
                rain_mm REAL,
                raw_data TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rtl433_model ON rtl433_readings(model)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rtl433_device ON rtl433_readings(device_id)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_rtl433_time ON rtl433_readings(created_at DESC)"
        )
        conn.commit()


def _get_readings_db(
    limit: int, offset: int, model: Optional[str], device_id: Optional[str]
) -> dict:
    """Get readings from database"""
    db = _get_db()
    with db.get_connection() as conn:
        cursor = conn.cursor()

        # Build query with filters
        where = []
        params = []

        if model:
            where.append("model = ?")
            params.append(model)
        if device_id:
            where.append("device_id = ?")
            params.append(device_id)

        where_clause = "WHERE " + " AND ".join(where) if where else ""

        # Get total count
        cursor.execute("SELECT COUNT(*) FROM rtl433_readings " + where_clause, params)
        total = cursor.fetchone()[0]

        # Get readings
        cursor.execute(
            f"""
            SELECT id, model, device_id, channel, battery_ok,
                   temperature_c, humidity, pressure_hpa,
                   wind_speed_kmh, rain_mm, raw_data, created_at
            FROM rtl433_readings
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            params + [limit, offset],
        )

        columns = [desc[0] for desc in cursor.description]
        items = [dict(zip(columns, row)) for row in cursor.fetchall()]

        return {"items": items, "total": total}


def _get_devices_db() -> dict:
    """Get device summary from database"""
    db = _get_db()
    with db.get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT model, device_id, channel,
                   COUNT(*) as reading_count,
                   MAX(created_at) as last_seen,
                   AVG(temperature_c) as avg_temp,
                   AVG(humidity) as avg_humidity
            FROM rtl433_readings
            GROUP BY model, device_id, channel
            ORDER BY last_seen DESC
            """
        )

        devices = []
        for row in cursor.fetchall():
            devices.append(
                RTL433DeviceSummary(
                    model=row[0],
                    device_id=row[1],
                    channel=row[2],
                    reading_count=row[3],
                    last_seen=row[4],
                    avg_temperature_c=round(row[5], 2) if row[5] else None,
                    avg_humidity=round(row[6], 2) if row[6] else None,
                )
            )

        return {"devices": devices, "total": len(devices)}
