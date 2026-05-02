"""
FM Radio API Routes
Endpoints for FM station scanning and recordings
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict

from utils.caching.cache_manager import CacheManager
from utils.database import DatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/fm-radio", tags=["FM Radio"])

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


class FMRecording(BaseModel):
    """FM recording model"""

    id: int
    frequency_mhz: float
    file_path: str
    file_size_bytes: Optional[int] = None
    duration_sec: Optional[float] = None
    created_at: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class FMStation(BaseModel):
    """FM station model"""

    id: int
    frequency_mhz: float
    signal_strength_db: Optional[float] = None
    signal_power: Optional[float] = None
    last_seen: str

    model_config = ConfigDict(from_attributes=True)


class FMRecordingsResponse(BaseModel):
    """Response model for recordings list"""

    items: list[FMRecording]
    total: int
    limit: int
    offset: int


class FMStationsResponse(BaseModel):
    """Response model for known stations"""

    stations: list[FMStation]
    total: int


class FMStatsResponse(BaseModel):
    """Response model for FM statistics"""

    total_recordings: int
    unique_stations: int
    total_storage_bytes: int
    timestamp: str


@router.get("/recordings", response_model=FMRecordingsResponse)
async def get_recordings(
    limit: int = Query(50, ge=1, le=500, description="Number of recordings"),
    offset: int = Query(0, ge=0, description="Offset"),
):
    """
    Get recent FM recordings.

    Returns list of recorded FM radio stations with file metadata.
    """
    cache_key = f"fm:recordings:{limit}:{offset}"

    if _cache_manager:
        cached = _cache_manager.get(cache_key)
        if cached:
            return cached

    _ensure_table()
    recordings = _get_recordings_db(limit, offset)

    response = FMRecordingsResponse(
        items=recordings["items"],
        total=recordings["total"],
        limit=limit,
        offset=offset,
    )

    if _cache_manager:
        _cache_manager.set(cache_key, response, expire=10)

    return response


@router.get("/stations", response_model=FMStationsResponse)
async def get_stations():
    """
    Get known FM stations from scans.

    Returns unique frequencies with signal strength,
    showing most recent scan for each frequency.
    """
    cache_key = "fm:stations"

    if _cache_manager:
        cached = _cache_manager.get(cache_key)
        if cached:
            return cached

    _ensure_table()
    stations = _get_stations_db()

    response = FMStationsResponse(
        stations=stations["stations"],
        total=stations["total"],
    )

    if _cache_manager:
        _cache_manager.set(cache_key, response, expire=30)

    return response


@router.get("/stats", response_model=FMStatsResponse)
async def get_stats():
    """
    Get FM radio statistics.

    Returns total recordings, unique stations, and storage used.
    """
    cache_key = "fm:stats"

    if _cache_manager:
        cached = _cache_manager.get(cache_key)
        if cached:
            return cached

    _ensure_table()

    db = _get_db()
    with db.get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM fm_recordings")
        total_recordings = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT frequency_mhz) FROM fm_stations")
        unique_stations = cursor.fetchone()[0]

        cursor.execute("SELECT SUM(file_size_bytes) FROM fm_recordings")
        total_storage = cursor.fetchone()[0] or 0

    stats = FMStatsResponse(
        total_recordings=total_recordings,
        unique_stations=unique_stations,
        total_storage_bytes=total_storage,
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
    )

    if _cache_manager:
        _cache_manager.set(cache_key, stats, expire=10)

    return stats


@router.post("/clear")
async def clear_data():
    """
    Clear all FM radio data.

    WARNING: This deletes all recordings and station data!
    """
    _ensure_table()

    db = _get_db()
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM fm_recordings")
        deleted_recordings = cursor.rowcount
        cursor.execute("DELETE FROM fm_stations")
        deleted_stations = cursor.rowcount
        conn.commit()

    if _cache_manager:
        _cache_manager.delete_by_pattern("fm:*")

    return {
        "message": f"Cleared {deleted_recordings} recordings and {deleted_stations} stations",
        "deleted_recordings": deleted_recordings,
        "deleted_stations": deleted_stations,
    }


def _ensure_table():
    """Create FM radio tables if not exist"""
    db = _get_db()
    with db.get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS fm_recordings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frequency_mhz REAL NOT NULL,
                file_path TEXT NOT NULL,
                file_size_bytes INTEGER,
                duration_sec REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS fm_stations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frequency_mhz REAL NOT NULL,
                signal_strength_db REAL,
                signal_power REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_fm_rec_time ON fm_recordings(created_at DESC)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_fm_station_freq ON fm_stations(frequency_mhz)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_fm_station_time ON fm_stations(created_at DESC)"
        )
        conn.commit()


def _get_recordings_db(limit: int, offset: int) -> dict:
    """Get recordings from database"""
    db = _get_db()
    with db.get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM fm_recordings")
        total = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT id, frequency_mhz, file_path, file_size_bytes,
                   duration_sec, created_at
            FROM fm_recordings
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )

        columns = [desc[0] for desc in cursor.description]
        items = [dict(zip(columns, row)) for row in cursor.fetchall()]

        return {"items": items, "total": total}


def _get_stations_db() -> dict:
    """Get stations from database"""
    db = _get_db()
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT s1.id, s1.frequency_mhz, s1.signal_strength_db,
                   s1.signal_power, s1.created_at as last_seen
            FROM fm_stations s1
            INNER JOIN (
                SELECT frequency_mhz, MAX(created_at) as max_time
                FROM fm_stations
                GROUP BY frequency_mhz
            ) s2 ON s1.frequency_mhz = s2.frequency_mhz
                AND s1.created_at = s2.max_time
            ORDER BY s1.signal_strength_db DESC
            """
        )

        stations = []
        columns = [desc[0] for desc in cursor.description]
        for row in cursor.fetchall():
            data = dict(zip(columns, row))
            stations.append(
                FMStation(
                    id=data["id"],
                    frequency_mhz=data["frequency_mhz"],
                    signal_strength_db=data.get("signal_strength_db"),
                    signal_power=data.get("signal_power"),
                    last_seen=data["last_seen"],
                )
            )

        return {"stations": stations, "total": len(stations)}
