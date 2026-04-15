"""Database schema initialization."""

import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def init_database_schema(conn) -> None:
    """Создаёт все таблицы и индексы, если они не существуют."""
    cursor = conn.cursor()

    # Таблица результатов сканирований
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS scan_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            scan_type TEXT NOT NULL,
            surface_type TEXT,
            width INTEGER,
            height INTEGER,
            file_path TEXT,
            metadata TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Таблица симуляций
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS simulations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            simulation_id TEXT UNIQUE NOT NULL,
            simulation_type TEXT NOT NULL,
            status TEXT DEFAULT 'running',
            start_time TEXT,
            end_time TEXT,
            duration_seconds REAL,
            parameters TEXT,
            results_summary TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Таблица изображений
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT UNIQUE NOT NULL,
            image_type TEXT,
            source TEXT,
            width INTEGER,
            height INTEGER,
            channels INTEGER,
            metadata TEXT,
            processed INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Таблица экспорта данных
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS exports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            export_path TEXT UNIQUE NOT NULL,
            export_format TEXT NOT NULL,
            source_type TEXT,
            source_id INTEGER,
            file_size_bytes INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Индексы для ускорения поиска
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_scan_timestamp ON scan_results(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_scan_type ON scan_results(scan_type)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_scan_type_timestamp ON scan_results(scan_type, timestamp DESC)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_simulations_status_created ON simulations(status, created_at DESC)"
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_simulation_status ON simulations(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_image_type ON images(image_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_scan_file_path ON scan_results(file_path)")

    # Таблица сравнения изображений поверхностей
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS surface_comparisons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            comparison_id TEXT UNIQUE NOT NULL,
            image1_path TEXT NOT NULL,
            image2_path TEXT NOT NULL,
            similarity_score REAL,
            difference_map_path TEXT,
            metrics TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Таблица AI/ML анализа дефектов
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS defect_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id TEXT UNIQUE NOT NULL,
            image_path TEXT NOT NULL,
            model_name TEXT,
            defects_detected INTEGER DEFAULT 0,
            defects_data TEXT,
            confidence_score REAL,
            processing_time_ms REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Таблица PDF отчётов
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS pdf_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_path TEXT UNIQUE NOT NULL,
            report_type TEXT NOT NULL,
            title TEXT,
            source_ids TEXT,
            file_size_bytes INTEGER,
            pages_count INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Таблица пакетной обработки
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS batch_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT UNIQUE NOT NULL,
            job_type TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            total_items INTEGER DEFAULT 0,
            processed_items INTEGER DEFAULT 0,
            failed_items INTEGER DEFAULT 0,
            parameters TEXT,
            results_summary TEXT,
            started_at TEXT,
            completed_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Таблица метрик производительности
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            metric_type TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            value REAL NOT NULL,
            unit TEXT,
            metadata TEXT
        )
    """
    )

    # Таблица пользователей
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            last_login TEXT
        )
    """
    )

    # Индексы для новых таблиц
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_comparison_timestamp ON surface_comparisons(created_at)"
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_defect_image ON defect_analysis(image_path)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_batch_status ON batch_jobs(status)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON performance_metrics(timestamp)"
    )

    # Таблицы RTL-SDR
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS rtl433_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reading_id TEXT UNIQUE NOT NULL,
            model TEXT NOT NULL,
            device_id TEXT,
            frequency REAL,
            temperature REAL,
            humidity REAL,
            battery_ok INTEGER,
            raw_data TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS adsb_sightings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sighting_id TEXT UNIQUE NOT NULL,
            icao TEXT NOT NULL,
            flight TEXT,
            altitude_ft INTEGER,
            speed_knots REAL,
            heading INTEGER,
            lat REAL,
            lon REAL,
            vertical_rate INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS fm_recordings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recording_id TEXT UNIQUE NOT NULL,
            frequency_mhz REAL NOT NULL,
            station_name TEXT,
            file_path TEXT,
            duration_sec INTEGER,
            file_size_bytes INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS fm_stations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id TEXT UNIQUE NOT NULL,
            frequency_mhz REAL NOT NULL,
            name TEXT,
            signal_strength REAL,
            location TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Индексы для RTL-SDR таблиц
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rtl433_model ON rtl433_readings(model)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rtl433_device ON rtl433_readings(device_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rtl433_time ON rtl433_readings(created_at DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_adsb_icao ON adsb_sightings(icao)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_adsb_time ON adsb_sightings(created_at DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_adsb_flight ON adsb_sightings(flight)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fm_rec_time ON fm_recordings(created_at DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fm_station_freq ON fm_stations(frequency_mhz)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fm_station_time ON fm_stations(created_at DESC)")


def get_database_stats(conn) -> Dict[str, Any]:
    """Получение статистики базы данных."""
    cursor = conn.cursor()

    tables = {}
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    for row in cursor.fetchall():
        table_name = row[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        tables[table_name] = count

    indexes = {}
    cursor.execute(
        "SELECT name, tbl_name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%' ORDER BY tbl_name"
    )
    for row in cursor.fetchall():
        table = row[1]
        if table not in indexes:
            indexes[table] = []
        indexes[table].append(row[0])

    cursor.execute("PRAGMA page_count")
    page_count = cursor.fetchone()[0]
    cursor.execute("PRAGMA page_size")
    page_size = cursor.fetchone()[0]
    total_size = page_count * page_size

    cursor.execute("PRAGMA journal_mode")
    journal_mode = cursor.fetchone()[0]

    return {
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "journal_mode": journal_mode,
        "table_count": len(tables),
        "table_row_counts": tables,
        "index_count": sum(len(v) for v in indexes.values()),
        "indexes_by_table": indexes,
    }
