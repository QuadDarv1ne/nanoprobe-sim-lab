"""
ADS-B Aircraft Tracker API Routes
Endpoints for aircraft tracking data from 1090 MHz
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from utils.caching.cache_manager import CacheManager
from utils.database import DatabaseManager

router = APIRouter(prefix="/adsb", tags=["ADS-B Aircraft"])

# Global references (set during app startup)
_db_manager: Optional[DatabaseManager] = None
_cache_manager: Optional[CacheManager] = None


def set_managers(db: DatabaseManager, cache: CacheManager):
    """Set database and cache managers"""
    global _db_manager, _cache_manager
    _db_manager = db
    _cache_manager = cache


class ADSBSighting(BaseModel):
    """ADS-B sighting model"""

    id: int
    icao: str
    flight: Optional[str] = None
    altitude_ft: Optional[float] = None
    speed_knots: Optional[float] = None
    heading: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    category: Optional[str] = None
    squawk: Optional[str] = None
    rssi_db: Optional[float] = None
    message_count: Optional[int] = None
    created_at: Optional[str] = None

    class Config:
        from_attributes = True


class ADSBActiveAircraft(BaseModel):
    """Active aircraft model"""

    icao: str
    flight: Optional[str] = None
    altitude_ft: Optional[float] = None
    speed_knots: Optional[float] = None
    heading: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    category: Optional[str] = None
    last_seen: str

    class Config:
        from_attributes = True


class ADSBSightingsResponse(BaseModel):
    """Response model for sightings list"""

    items: list[ADSBSighting]
    total: int
    limit: int
    offset: int


class ADSBActiveResponse(BaseModel):
    """Response model for active aircraft"""

    aircraft: list[ADSBActiveAircraft]
    total: int


class ADSBStatsResponse(BaseModel):
    """Response model for ADS-B statistics"""

    total_sightings: int
    unique_aircraft: int
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    timestamp: str


@router.get("/sightings", response_model=ADSBSightingsResponse)
async def get_sightings(
    limit: int = Query(50, ge=1, le=500, description="Number of sightings"),
    offset: int = Query(0, ge=0, description="Offset"),
    icao: Optional[str] = Query(None, description="Filter by ICAO address"),
    flight: Optional[str] = Query(None, description="Filter by callsign"),
):
    """
    Get recent ADS-B sightings.

    Returns aircraft sightings data including flight, altitude, speed,
    and position from 1090 MHz Mode-S transponder signals.
    """
    cache_key = f"adsb:sightings:{limit}:{offset}:{icao}:{flight}"

    if _cache_manager:
        cached = _cache_manager.get(cache_key)
        if cached:
            return cached

    _ensure_table()
    sightings = _get_sightings_db(limit, offset, icao, flight)

    response = ADSBSightingsResponse(
        items=sightings["items"],
        total=sightings["total"],
        limit=limit,
        offset=offset,
    )

    if _cache_manager:
        _cache_manager.set(cache_key, response, expire=5)

    return response


@router.get("/active", response_model=ADSBActiveResponse)
async def get_active_aircraft(
    limit: int = Query(100, ge=1, le=500, description="Maximum aircraft"),
):
    """
    Get currently active aircraft (most recent sighting per ICAO).

    Returns the latest sighting for each unique aircraft.
    """
    cache_key = f"adsb:active:{limit}"

    if _cache_manager:
        cached = _cache_manager.get(cache_key)
        if cached:
            return cached

    _ensure_table()
    aircraft = _get_active_aircraft_db(limit)

    response = ADSBActiveResponse(
        aircraft=aircraft["aircraft"],
        total=aircraft["total"],
    )

    if _cache_manager:
        _cache_manager.set(cache_key, response, expire=10)

    return response


@router.get("/stats", response_model=ADSBStatsResponse)
async def get_stats():
    """
    Get ADS-B tracking statistics.

    Returns total sightings, unique aircraft, and time range.
    """
    cache_key = "adsb:stats"

    if _cache_manager:
        cached = _cache_manager.get(cache_key)
        if cached:
            return cached

    _ensure_table()

    with _db_manager.get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM adsb_sightings")
        total_sightings = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT icao) FROM adsb_sightings")
        unique_aircraft = cursor.fetchone()[0]

        cursor.execute(
            "SELECT MIN(created_at), MAX(created_at) FROM adsb_sightings"
        )
        row = cursor.fetchone()
        first_seen = row[0]
        last_seen = row[1]

    stats = ADSBStatsResponse(
        total_sightings=total_sightings,
        unique_aircraft=unique_aircraft,
        first_seen=first_seen,
        last_seen=last_seen,
        timestamp=datetime.now().isoformat(),
    )

    if _cache_manager:
        _cache_manager.set(cache_key, stats, expire=10)

    return stats


@router.post("/clear")
async def clear_data():
    """
    Clear all ADS-B data.

    WARNING: This deletes all aircraft sightings!
    """
    _ensure_table()

    with _db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM adsb_sightings")
        deleted = cursor.rowcount
        conn.commit()

    if _cache_manager:
        _cache_manager.delete_by_pattern("adsb:*")

    return {
        "message": f"Cleared {deleted} sightings",
        "deleted_count": deleted,
    }


def _ensure_table():
    """Create adsb_sightings table if not exists"""
    with _db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS adsb_sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                icao TEXT NOT NULL,
                flight TEXT,
                altitude_ft REAL,
                speed_knots REAL,
                heading REAL,
                latitude REAL,
                longitude REAL,
                vertical_rate REAL,
                category TEXT,
                squawk TEXT,
                rssi_db REAL,
                message_count INTEGER,
                seconds_ago REAL,
                raw_data TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_adsb_icao ON adsb_sightings(icao)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_adsb_time ON adsb_sightings(created_at DESC)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_adsb_flight ON adsb_sightings(flight)"
        )
        conn.commit()


def _get_sightings_db(
    limit: int, offset: int, icao: Optional[str], flight: Optional[str]
) -> dict:
    """Get sightings from database"""
    with _db_manager.get_connection() as conn:
        cursor = conn.cursor()

        where = []
        params = []

        if icao:
            where.append("icao = ?")
            params.append(icao)
        if flight:
            where.append("flight = ?")
            params.append(flight)

        where_clause = "WHERE " + " AND ".join(where) if where else ""

        cursor.execute(
            f"SELECT COUNT(*) FROM adsb_sightings {where_clause}", params
        )
        total = cursor.fetchone()[0]

        cursor.execute(
            f"""
            SELECT id, icao, flight, altitude_ft, speed_knots, heading,
                   latitude, longitude, category, squawk, rssi_db,
                   message_count, seconds_ago, created_at
            FROM adsb_sightings
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            params + [limit, offset],
        )

        columns = [desc[0] for desc in cursor.description]
        items = [dict(zip(columns, row)) for row in cursor.fetchall()]

        return {"items": items, "total": total}


def _get_active_aircraft_db(limit: int) -> dict:
    """Get active aircraft from database"""
    with _db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT s1.icao, s1.flight, s1.altitude_ft, s1.speed_knots,
                   s1.heading, s1.latitude, s1.longitude, s1.category,
                   s1.created_at as last_seen
            FROM adsb_sightings s1
            INNER JOIN (
                SELECT icao, MAX(created_at) as max_time
                FROM adsb_sightings
                GROUP BY icao
            ) s2 ON s1.icao = s2.icao AND s1.created_at = s2.max_time
            ORDER BY s1.created_at DESC
            LIMIT ?
            """,
            (limit,),
        )

        aircraft = []
        columns = [desc[0] for desc in cursor.description]
        for row in cursor.fetchall():
            data = dict(zip(columns, row))
            aircraft.append(
                ADSBActiveAircraft(
                    icao=data["icao"],
                    flight=data.get("flight"),
                    altitude_ft=data.get("altitude_ft"),
                    speed_knots=data.get("speed_knots"),
                    heading=data.get("heading"),
                    latitude=data.get("latitude"),
                    longitude=data.get("longitude"),
                    category=data.get("category"),
                    last_seen=data["last_seen"],
                )
            )

        return {"aircraft": aircraft, "total": len(aircraft)}
