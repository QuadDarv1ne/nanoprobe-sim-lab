"""
Модуль базы данных для проекта Nanoprobe Simulation Lab
Хранение результатов сканирований, истории симуляций, метаданных
"""

import asyncio
import hashlib
import json
import logging
import sqlite3
import threading
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ConnectionPool:
    """Пул соединений для SQLite"""

    def __init__(self, db_path: str, pool_size: int = 5):
        """
        Инициализация пула соединений.

        Args:
            db_path: Путь к файлу базы данных SQLite
            pool_size: Размер пула соединений (по умолчанию 5)
        """
        self.db_path = db_path
        self.pool_size = pool_size
        self._pool: Queue = Queue(maxsize=pool_size)
        self._lock = threading.Lock()
        self._created = 0
        self._stats = {"hits": 0, "misses": 0, "created": 0}

    def get_connection(self, timeout: float = 30.0) -> sqlite3.Connection:
        """Получение соединения из пула"""
        try:
            conn = self._pool.get_nowait()
            self._stats["hits"] += 1
            return conn
        except Exception:
            with self._lock:
                # Атомарная проверка и инкремент (race condition fix)
                if self._created < self.pool_size:
                    self._created += 1  # Инкрементируем ДО создания
                    self._stats["created"] += 1
                    return self._create_connection()
            self._stats["misses"] += 1
            return self._pool.get(timeout=timeout)

    def return_connection(self, conn: sqlite3.Connection):
        """Возврат соединения в пул"""
        try:
            self._pool.put_nowait(conn)
        except Exception:
            logger.debug("Connection close on return error")
            conn.close()

    def _create_connection(self) -> sqlite3.Connection:
        """Создание нового соединения с оптимизациями"""
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = -64000")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA mmap_size = 268435456")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def close_all(self):
        """Закрытие всех соединений"""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Exception:
                break

    def get_stats(self) -> Dict[str, int]:
        """Получение статистики пула"""
        return {
            **self._stats,
            "pool_size": self.pool_size,
            "current_size": self._pool.qsize(),
        }


class AsyncConnectionPool:
    """Асинхронный пул соединений для SQLite"""

    def __init__(self, db_path: str, pool_size: int = 5):
        """
        Инициализация асинхронного пула соединений.

        Args:
            db_path: Путь к файлу базы данных SQLite
            pool_size: Размер пула соединений (по умолчанию 5)
        """
        self.db_path = db_path
        self.pool_size = pool_size
        self._pool: asyncio.Queue = asyncio.Queue(maxsize=pool_size)
        self._lock = asyncio.Lock()
        self._created = 0

    async def get_connection(self, timeout: float = 30.0) -> sqlite3.Connection:
        """Получение соединения из пула"""
        try:
            return self._pool.get_nowait()
        except Exception:
            async with self._lock:
                if self._created < self.pool_size:
                    conn = self._create_connection()
                    self._created += 1
                    return conn
            return await asyncio.wait_for(self._pool.get(), timeout=timeout)

    def return_connection(self, conn: sqlite3.Connection):
        """Возврат соединения в пул"""
        try:
            self._pool.put_nowait(conn)
        except Exception:
            conn.close()

    def _create_connection(self) -> sqlite3.Connection:
        """Создание нового соединения"""
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = -64000")
        return conn

    async def close_all(self):
        """Закрытие всех соединений"""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Exception:
                break


class DatabaseManager:
    """Менеджер базы данных SQLite с connection pooling и query caching."""

    _pools: Dict[str, ConnectionPool] = {}
    _pool_lock = threading.Lock()

    def __init__(
        self, db_path: str = "data/nanoprobe.db", pool_size: int = 5, enable_cache: bool = True
    ):
        """
        Инициализирует менеджер базы данных.

        Args:
            db_path: Путь к файлу базы данных
            pool_size: Размер пула соединений
            enable_cache: Включить кэширование запросов
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.pool_size = pool_size
        self.enable_cache = enable_cache
        self._query_cache: Dict[str, Tuple[Any, datetime]] = {}
        self._cache_ttl = 60  # секунд
        self._cache_max_size = 100
        self._init_pool()
        self._init_database()

    def _init_pool(self):
        """Инициализация пула соединений"""
        db_path_str = str(self.db_path)
        with self._pool_lock:
            if db_path_str not in self._pools:
                self._pools[db_path_str] = ConnectionPool(db_path_str, self.pool_size)
            self._pool = self._pools[db_path_str]

    @classmethod
    def close_all_pools(cls):
        """Закрыть все пулы соединений"""
        with cls._pool_lock:
            for pool in cls._pools.values():
                pool.close_all()
            cls._pools.clear()

    def close_pool(self):
        """Закрыть пул соединений для текущего экземпляра"""
        db_path_str = str(self.db_path)
        with self._pool_lock:
            if db_path_str in self._pools:
                self._pools[db_path_str].close_all()
                del self._pools[db_path_str]
            if hasattr(self, "_pool") and self._pool:
                self._pool.close_all()

    @contextmanager
    def get_connection(self):
        """Контекстный менеджер для подключения к БД с пулом соединений."""
        conn = self._pool.get_connection(timeout=30.0)
        try:
            yield conn
            conn.commit()
        except Exception as e:
            logger.error(f"Database transaction error: {e}")
            conn.rollback()
            raise e
        finally:
            self._pool.return_connection(conn)

    @asynccontextmanager
    async def get_connection_async(self):
        """Асинхронный контекстный менеджер для подключения к БД."""
        loop = asyncio.get_event_loop()
        conn = await loop.run_in_executor(None, self._pool.get_connection)
        try:
            yield conn
            await loop.run_in_executor(None, conn.commit)
        except Exception as e:
            logger.error(f"Database async transaction error: {e}")
            await loop.run_in_executor(None, conn.rollback)
            raise e
        finally:
            await loop.run_in_executor(None, self._pool.return_connection, conn)

    def get_pool_stats(self) -> Dict[str, Any]:
        """Получение статистики пула соединений"""
        return self._pool.get_stats()

    def optimize_database(self) -> Dict[str, Any]:
        """
        Оптимизация базы данных.

        Выполняет:
        - ANALYZE: обновляет статистику для оптимизатора запросов
        - VACUUM: переупаковывает БД (удаляет мёртвые строки)
        - Проверка целостности (integrity check)

        Returns:
            Dict с результатами операций
        """
        results = {}

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # ANALYZE — обновляет статистику индексов
            try:
                cursor.execute("ANALYZE")
                results["analyze"] = "success"
            except Exception as e:
                results["analyze"] = f"error: {e}"

            # Integrity check
            try:
                cursor.execute("PRAGMA integrity_check")
                check_result = cursor.fetchone()[0]
                results["integrity"] = check_result
            except Exception as e:
                results["integrity"] = f"error: {e}"

            # Database size before vacuum
            cursor.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            cursor.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]
            size_before = page_count * page_size
            results["size_bytes_before"] = size_before

            # VACUUM — переупаковка БД (может занять время на больших БД)
            try:
                cursor.execute("VACUUM")
                results["vacuum"] = "success"
            except Exception as e:
                results["vacuum"] = f"error: {e}"

            # Database size after vacuum
            cursor.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            size_after = page_count * page_size
            results["size_bytes_after"] = size_after

            if size_before > 0:
                saved = size_before - size_after
                results["space_saved_bytes"] = max(0, saved)
                results["space_saved_percent"] = (
                    round((saved / size_before) * 100, 2) if size_before > 0 else 0
                )

            # Table sizes
            cursor.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table'
                ORDER BY name
                """
            )
            tables = {}
            for row in cursor.fetchall():
                table_name = row[0]
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    tables[table_name] = count
                except Exception:
                    tables[table_name] = 0

            results["table_row_counts"] = tables

        return results

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Получение статистики базы данных.

        Returns:
            Dict с информацией о таблицах, индексах и размерах
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Tables and row counts
            cursor.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table'
                ORDER BY name
            """
            )
            tables = {}
            for row in cursor.fetchall():
                table_name = row[0]
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                tables[table_name] = count

            # Indexes
            cursor.execute(
                """
                SELECT name, tbl_name FROM sqlite_master
                WHERE type='index' AND name NOT LIKE 'sqlite_%'
                ORDER BY tbl_name
            """
            )
            indexes = {}
            for row in cursor.fetchall():
                table = row[1]
                if table not in indexes:
                    indexes[table] = []
                indexes[table].append(row[0])

            # Database size
            cursor.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            cursor.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]
            total_size = page_count * page_size

            # WAL mode
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

    def _init_database(self):
        """Инициализация схемы базы данных."""
        with self.get_connection() as conn:
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
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_scan_timestamp
                ON scan_results(timestamp)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_scan_type
                ON scan_results(scan_type)
            """
            )
            # Составные индексы для сложных запросов
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_scan_type_timestamp
                ON scan_results(scan_type, timestamp DESC)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_simulations_status_created
                ON simulations(status, created_at DESC)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_simulation_status
                ON simulations(status)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_image_type
                ON images(image_type)
            """
            )
            # Индекс для поиска по путям
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_scan_file_path
                ON scan_results(file_path)
            """
            )

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

            # Таблица метрик производительности (для real-time визуализации)
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

            # Таблица пользователей (персистентное хранение вместо in-memory)
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
                """
                CREATE INDEX IF NOT EXISTS idx_comparison_timestamp
                ON surface_comparisons(created_at)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_defect_image
                ON defect_analysis(image_path)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_batch_status
                ON batch_jobs(status)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp
                ON performance_metrics(timestamp)
            """
            )

            # Таблицы RTL-SDR (RTL_433, ADS-B, FM Radio)
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
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_rtl433_model
                ON rtl433_readings(model)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_rtl433_device
                ON rtl433_readings(device_id)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_rtl433_time
                ON rtl433_readings(created_at DESC)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_adsb_icao
                ON adsb_sightings(icao)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_adsb_time
                ON adsb_sightings(created_at DESC)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_adsb_flight
                ON adsb_sightings(flight)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_fm_rec_time
                ON fm_recordings(created_at DESC)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_fm_station_freq
                ON fm_stations(frequency_mhz)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_fm_station_time
                ON fm_stations(created_at DESC)
            """
            )

    def add_scan_result(
        self,
        scan_type: str,
        surface_type: str = None,
        width: int = None,
        height: int = None,
        file_path: str = None,
        metadata: Dict = None,
    ) -> int:
        """
        Добавляет результат сканирования.

        Args:
            scan_type: Тип сканирования (spm, image, sstv)
            surface_type: Тип поверхности
            width: Ширина
            height: Высота
            file_path: Путь к файлу
            metadata: Метаданные

        Returns:
            ID записи
        """
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO scan_results
                (timestamp, scan_type, surface_type, width, height, file_path, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    now,
                    scan_type,
                    surface_type,
                    width,
                    height,
                    file_path,
                    json.dumps(metadata) if metadata else None,
                    now,  # Устанавливаем created_at явно
                ),
            )
            scan_id = cursor.lastrowid

        # Инвалидация кэша сканирований
        self.invalidate_cache("scans:")

        return scan_id

    def add_scan_result_batch(self, scan_results: List[Dict]) -> int:
        """
        Добавляет несколько результатов сканирования пакетно.

        Args:
            scan_results: Список словарей с параметрами сканирования

        Returns:
            Количество добавленных записей
        """
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            data = []
            for scan in scan_results:
                data.append(
                    (
                        now,
                        scan.get("scan_type"),
                        scan.get("surface_type"),
                        scan.get("width"),
                        scan.get("height"),
                        scan.get("file_path"),
                        json.dumps(scan.get("metadata")) if scan.get("metadata") else None,
                        now,  # Устанавливаем created_at явно
                    )
                )
            cursor.executemany(
                """
                INSERT INTO scan_results
                (timestamp, scan_type, surface_type, width, height, file_path, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                data,
            )
            # Инвалидация кэша после вставки
            self.invalidate_cache("scans:")
            return len(data)

    def get_scan_results(
        self, scan_type: str = None, limit: int = 100, offset: int = 0
    ) -> List[Dict]:
        """
        Получает результаты сканирований с кэшированием.

        Args:
            scan_type: Фильтр по типу сканирования
            limit: Лимит записей
            offset: Смещение

        Returns:
            Список записей
        """
        cache_key = self._get_cache_key(f"scans:{scan_type}:{limit}:{offset}")
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT id, timestamp, scan_type, surface_type, width, height,
                       file_path, metadata, created_at
                FROM scan_results
            """
            params = []

            if scan_type:
                query += " WHERE scan_type = ?"
                params.append(scan_type)

            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()
            result = [self._row_to_dict(row) for row in rows]

            self._set_cache(cache_key, result)
            return result

    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """
        Выполняет SQL запрос и возвращает результат.

        Args:
            query: SQL запрос
            params: Параметры запроса

        Returns:
            Список словарей с результатами
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in rows]

    def get_scan_by_id(self, scan_id: int) -> Optional[Dict]:
        """Получает результат сканирования по ID с кэшированием."""
        cache_key = self._get_cache_key(f"scan:id:{scan_id}")
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM scan_results WHERE id = ?", (scan_id,))
            row = cursor.fetchone()
            result = self._row_to_dict(row) if row else None

            if result:
                self._set_cache(cache_key, result)
            return result

    def count_scans(self, scan_type: str = None) -> int:
        """Подсчитывает количество сканирований с кэшированием."""
        cache_key = self._get_cache_key(f"scans:count:{scan_type or 'all'}")
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        with self.get_connection() as conn:
            cursor = conn.cursor()
            if scan_type:
                cursor.execute(
                    "SELECT COUNT(*) FROM scan_results WHERE scan_type = ?", (scan_type,)
                )
            else:
                cursor.execute("SELECT COUNT(*) FROM scan_results")
            result = cursor.fetchone()[0]

            self._set_cache(cache_key, result)
            return result

    # ==================== Async Methods ====================

    async def get_scan_results_async(
        self, scan_type: str = None, limit: int = 100, offset: int = 0
    ) -> List[Dict]:
        """Асинхронное получение результатов сканирований"""
        cache_key = self._get_cache_key(f"scans:{scan_type}:{limit}:{offset}")
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        async with self.get_connection_async() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM scan_results"
            params = []

            if scan_type:
                query += " WHERE scan_type = ?"
                params.append(scan_type)

            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()
            result = [self._row_to_dict(row) for row in rows]

            self._set_cache(cache_key, result)
            return result

    async def get_scan_by_id_async(self, scan_id: int) -> Optional[Dict]:
        """Асинхронное получение сканирования по ID"""
        cache_key = self._get_cache_key(f"scan:id:{scan_id}")
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        async with self.get_connection_async() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM scan_results WHERE id = ?", (scan_id,))
            row = cursor.fetchone()
            result = self._row_to_dict(row) if row else None

            if result:
                self._set_cache(cache_key, result)
            return result

    async def count_scans_async(self, scan_type: str = None) -> int:
        """Асинхронный подсчёт количества сканирований"""
        cache_key = self._get_cache_key(f"scans:count:{scan_type or 'all'}")
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        async with self.get_connection_async() as conn:
            cursor = conn.cursor()
            if scan_type:
                cursor.execute(
                    "SELECT COUNT(*) FROM scan_results WHERE scan_type = ?", (scan_type,)
                )
            else:
                cursor.execute("SELECT COUNT(*) FROM scan_results")
            result = cursor.fetchone()[0]

            self._set_cache(cache_key, result)
            return result

    def count_simulations(self, simulation_type: str = None) -> int:
        """Подсчитывает количество симуляций с кэшированием."""
        cache_key = self._get_cache_key(f"simulations:count:{simulation_type or 'all'}")
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        with self.get_connection() as conn:
            cursor = conn.cursor()
            if simulation_type:
                cursor.execute(
                    "SELECT COUNT(*) FROM simulations WHERE simulation_type = ?", (simulation_type,)
                )
            else:
                cursor.execute("SELECT COUNT(*) FROM simulations")
            result = cursor.fetchone()[0]

            self._set_cache(cache_key, result)
            return result

    async def count_simulations_async(self, simulation_type: str = None) -> int:
        """Асинхронный подсчёт количества симуляций"""
        cache_key = self._get_cache_key(f"simulations:count:{simulation_type or 'all'}")
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        async with self.get_connection_async() as conn:
            cursor = conn.cursor()
            if simulation_type:
                cursor.execute(
                    "SELECT COUNT(*) FROM simulations WHERE simulation_type = ?", (simulation_type,)
                )
            else:
                cursor.execute("SELECT COUNT(*) FROM simulations")
            result = cursor.fetchone()[0]

            self._set_cache(cache_key, result)
            return result

    def add_simulation(
        self, simulation_id: str, simulation_type: str, parameters: Dict = None
    ) -> int:
        """
        Добавляет запись о симуляции.

        Args:
            simulation_id: ID симуляции
            simulation_type: Тип симуляции
            parameters: Параметры

        Returns:
            ID записи
        """
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO simulations
                (simulation_id, simulation_type, start_time, status, parameters, created_at)
                VALUES (?, ?, ?, 'running', ?, ?)
            """,
                (
                    simulation_id,
                    simulation_type,
                    now,
                    json.dumps(parameters) if parameters else None,
                    now,  # Устанавливаем created_at явно
                ),
            )
            sim_id = cursor.lastrowid

        # Инвалидация кэша симуляций
        self.invalidate_cache("simulations:")

        return sim_id

    def count_analysis_results(self) -> int:
        """Подсчитывает количество записей AI/ML анализа."""
        cache_key = self._get_cache_key("analysis:count")
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM defect_analysis")
            result = cursor.fetchone()[0]
            self._set_cache(cache_key, result)
            return result

    def count_comparisons(self) -> int:
        """Подсчитывает количество сравнений поверхностей."""
        cache_key = self._get_cache_key("comparisons:count")
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM surface_comparisons")
            result = cursor.fetchone()[0]
            self._set_cache(cache_key, result)
            return result

    def count_reports(self) -> int:
        """Подсчитывает количество отчётов."""
        cache_key = self._get_cache_key("reports:count")
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Отчёты могут храниться в exports или отдельной таблице
            try:
                cursor.execute("SELECT COUNT(*) FROM reports")
            except Exception:
                # Если таблица reports не существует, используем exports
                cursor.execute("SELECT COUNT(*) FROM exports WHERE export_format = 'PDF'")
            result = cursor.fetchone()[0]
            self._set_cache(cache_key, result)
            return result

    async def add_simulation_async(
        self, simulation_id: str, simulation_type: str, parameters: Dict = None
    ) -> int:
        """Асинхронное добавление записи о симуляции"""
        now = datetime.now(timezone.utc).isoformat()
        async with self.get_connection_async() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO simulations
                (simulation_id, simulation_type, start_time, status, parameters, created_at)
                VALUES (?, ?, ?, 'running', ?, ?)
            """,
                (
                    simulation_id,
                    simulation_type,
                    now,
                    json.dumps(parameters) if parameters else None,
                    now,  # Устанавливаем created_at явно
                ),
            )
            sim_id = cursor.lastrowid

        # Инвалидация кэша
        self.invalidate_cache("simulations:")

        return sim_id

    def update_simulation(
        self, simulation_id: str, status: str = None, results_summary: Dict = None
    ):
        """
        Обновляет запись о симуляции.

        Args:
            simulation_id: ID симуляции
            status: Новый статус
            results_summary: Результаты
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            updates = []
            params = []

            if status:
                updates.append("status = ?")
                params.append(status)

            if results_summary:
                updates.append("results_summary = ?")
                params.append(json.dumps(results_summary))

            if status in ("completed", "failed", "stopped"):
                updates.append("end_time = ?")
                params.append(datetime.now(timezone.utc).isoformat())

                # Рассчитываем длительность
                cursor.execute(
                    "SELECT start_time FROM simulations WHERE simulation_id = ?", (simulation_id,)
                )
                row = cursor.fetchone()
                if row and row["start_time"]:
                    start = datetime.fromisoformat(row["start_time"])
                    duration = (datetime.now(timezone.utc) - start).total_seconds()
                    updates.append("duration_seconds = ?")
                    params.append(duration)

            params.append(simulation_id)
            query = f"UPDATE simulations SET {', '.join(updates)} WHERE simulation_id = ?"
            cursor.execute(query, params)

        # Инвалидация кэша
        self.invalidate_cache("simulations:")

    async def update_simulation_async(
        self, simulation_id: str, status: str = None, results_summary: Dict = None
    ):
        """Асинхронное обновление записи о симуляции"""
        async with self.get_connection_async() as conn:
            cursor = conn.cursor()

            updates = []
            params = []

            if status:
                updates.append("status = ?")
                params.append(status)

            if results_summary:
                updates.append("results_summary = ?")
                params.append(json.dumps(results_summary))

            if status in ("completed", "failed", "stopped"):
                updates.append("end_time = ?")
                params.append(datetime.now(timezone.utc).isoformat())

                cursor.execute(
                    "SELECT start_time FROM simulations WHERE simulation_id = ?", (simulation_id,)
                )
                row = cursor.fetchone()
                if row and row["start_time"]:
                    start = datetime.fromisoformat(row["start_time"])
                    duration = (datetime.now(timezone.utc) - start).total_seconds()
                    updates.append("duration_seconds = ?")
                    params.append(duration)

            params.append(simulation_id)
            query = f"UPDATE simulations SET {', '.join(updates)} WHERE simulation_id = ?"
            cursor.execute(query, params)

        # Инвалидация кэша
        self.invalidate_cache("simulations:")

    def get_simulations(self, status: str = None, limit: int = 50) -> List[Dict]:
        """
        Получает список симуляций.

        Args:
            status: Фильтр по статусу
            limit: Лимит записей

        Returns:
            Список записей
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT id, simulation_id, simulation_type, status,
                       start_time, end_time, duration_seconds, parameters,
                       results_summary, created_at
                FROM simulations
            """
            params = []

            if status:
                query += " WHERE status = ?"
                params.append(status)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [self._row_to_dict(row) for row in rows]

    def add_image(
        self,
        image_path: str,
        image_type: str = None,
        source: str = None,
        width: int = None,
        height: int = None,
        channels: int = None,
        metadata: Dict = None,
    ) -> int:
        """
        Добавляет запись об изображении.

        Args:
            image_path: Путь к файлу
            image_type: Тип изображения
            source: Источник (hubble, nasa, local)
            width: Ширина
            height: Высота
            channels: Количество каналов
            metadata: Метаданные

        Returns:
            ID записи
        """
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO images
                (image_path, image_type, source, width, height, channels, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    image_path,
                    image_type,
                    source,
                    width,
                    height,
                    channels,
                    json.dumps(metadata) if metadata else None,
                    now,  # Устанавливаем created_at явно
                ),
            )
            return cursor.lastrowid

    def get_images(
        self, image_type: str = None, source: str = None, limit: int = 100
    ) -> List[Dict]:
        """
        Получает список изображений.

        Args:
            image_type: Фильтр по типу
            source: Фильтр по источнику
            limit: Лимит записей

        Returns:
            Список записей
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT id, image_path, image_type, source, width, height,
                       channels, metadata, created_at
                FROM images
            """
            params = []
            conditions = []

            if image_type:
                conditions.append("image_type = ?")
                params.append(image_type)

            if source:
                conditions.append("source = ?")
                params.append(source)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [self._row_to_dict(row) for row in rows]

    def add_export(
        self,
        export_path: str,
        export_format: str,
        source_type: str = None,
        source_id: int = None,
        file_size_bytes: int = None,
    ) -> int:
        """
        Добавляет запись об экспорте.

        Args:
            export_path: Путь к файлу
            export_format: Формат (csv, hdf5, json)
            source_type: Тип источника
            source_id: ID источника
            file_size_bytes: Размер файла

        Returns:
            ID записи
        """
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO exports
                (export_path, export_format, source_type, source_id, file_size_bytes, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    export_path,
                    export_format,
                    source_type,
                    source_id,
                    file_size_bytes,
                    now,  # Устанавливаем created_at явно
                ),
            )
            return cursor.lastrowid

    def get_statistics(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Получает статистику базы данных с кэшированием.

        Args:
            use_cache: Использовать кэш (по умолчанию True, TTL=10 сек)

        Returns:
            Словарь со статистикой
        """
        # Проверка кэша
        cache_key = "statistics"
        if use_cache:
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached

        with self.get_connection() as conn:
            cursor = conn.cursor()

            stats = {}

            # Количество сканирований
            cursor.execute("SELECT COUNT(*) FROM scan_results")
            stats["total_scans"] = cursor.fetchone()[0]

            # Количество симуляций
            cursor.execute("SELECT COUNT(*) FROM simulations")
            stats["total_simulations"] = cursor.fetchone()[0]

            # Активные симуляции
            cursor.execute("SELECT COUNT(*) FROM simulations WHERE status = 'running'")
            stats["active_simulations"] = cursor.fetchone()[0]

            # Количество изображений
            cursor.execute("SELECT COUNT(*) FROM images")
            stats["total_images"] = cursor.fetchone()[0]

            # Количество экспортов
            cursor.execute("SELECT COUNT(*) FROM exports")
            stats["total_exports"] = cursor.fetchone()[0]

            # Сканирования по типам
            cursor.execute(
                """
                SELECT scan_type, COUNT(*) as count
                FROM scan_results
                GROUP BY scan_type
            """
            )
            stats["scans_by_type"] = {row["scan_type"]: row["count"] for row in cursor.fetchall()}

            # Новая статистика для расширенных функций
            # Сравнения поверхностей
            cursor.execute("SELECT COUNT(*) FROM surface_comparisons")
            stats["total_comparisons"] = cursor.fetchone()[0]

            # AI анализы дефектов
            cursor.execute("SELECT COUNT(*) FROM defect_analysis")
            stats["total_defect_analyses"] = cursor.fetchone()[0]

            # PDF отчёты
            cursor.execute("SELECT COUNT(*) FROM pdf_reports")
            stats["total_pdf_reports"] = cursor.fetchone()[0]

            # Пакетные задания
            cursor.execute("SELECT COUNT(*) FROM batch_jobs")
            stats["total_batch_jobs"] = cursor.fetchone()[0]

            # Активные пакетные задания
            cursor.execute("SELECT COUNT(*) FROM batch_jobs WHERE status = 'running'")
            stats["active_batch_jobs"] = cursor.fetchone()[0]

            # Метрики производительности
            cursor.execute("SELECT COUNT(*) FROM performance_metrics")
            stats["total_metrics"] = cursor.fetchone()[0]

            # Кэширование результата (TTL=10 сек)
            if use_cache:
                self._cache_result(cache_key, stats, ttl=10)

            return stats

    def _get_cached(self, key: str) -> Optional[Any]:
        """Получить из кэша если не истёк TTL"""
        if not self.enable_cache:
            return None
        if key in self._query_cache:
            value, timestamp = self._query_cache[key]
            if (datetime.now(timezone.utc) - timestamp).total_seconds() < self._cache_ttl:
                return value
            del self._query_cache[key]
        return None

    def _cache_result(self, key: str, value: Any, ttl: Optional[int] = None):
        """Закэшировать результат"""
        if not self.enable_cache:
            return
        if len(self._query_cache) >= self._cache_max_size:
            # Очистка старого кэша
            oldest = min(self._query_cache.items(), key=lambda x: x[1][1])
            del self._query_cache[oldest[0]]
        self._query_cache[key] = (value, datetime.now(timezone.utc))

    def cached_query(ttl: int = 300):
        """
        Декоратор для кэширования результатов запросов.

        Args:
            ttl: Время жизни кэша в секундах (по умолчанию 300 = 5 минут)

        Использование:
            @db.cached_query(ttl=600)
            def get_expensive_query(...):
                ...
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Генерация ключа кэша на основе имени функции и аргументов
                cache_key_parts = [func.__name__]
                for arg in args[1:]:  # Пропускаем self
                    cache_key_parts.append(str(arg))
                for k, v in sorted(kwargs.items()):
                    cache_key_parts.append(f"{k}={v}")

                cache_key = hashlib.md5("|".join(cache_key_parts).encode()).hexdigest()

                # Проверка кэша
                cached = self._get_cached(cache_key)  # noqa: F821
                if cached is not None:
                    return cached

                # Выполнение функции
                result = func(*args, **kwargs)

                # Кэширование результата
                self._cache_result(cache_key, result, ttl=ttl)  # noqa: F821
                return result

            return wrapper

        return decorator

    def search_scans(self, query: str, limit: int = 50) -> List[Dict]:
        """
        Поиск по результатам сканирований.

        Args:
            query: Поисковый запрос
            limit: Лимит записей

        Returns:
            Список записей
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            search_pattern = f"%{query}%"
            cursor.execute(
                """
                SELECT * FROM scan_results
                WHERE surface_type LIKE ? OR metadata LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (search_pattern, search_pattern, limit),
            )

            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]

    def delete_scan(self, scan_id: int) -> bool:
        """
        Удаляет запись о сканировании.

        Args:
            scan_id: ID записи

        Returns:
            True если успешно
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM scan_results WHERE id = ?", (scan_id,))
            success = cursor.rowcount > 0

        # Инвалидация кэша сканирований
        if success:
            self.invalidate_cache("scans:")
            self.invalidate_cache(f"scan:id:{scan_id}")

        return success

    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        """Конвертирует строку результата в словарь."""
        result = dict(row)

        # Парсим JSON поля
        for key in [
            "metadata",
            "parameters",
            "results_summary",
            "metrics",
            "defects_data",
            "source_ids",
        ]:
            if key in result and result[key]:
                try:
                    result[key] = json.loads(result[key])
                except (json.JSONDecodeError, TypeError):
                    pass

        return result

    # ==================== Query Cache Methods ====================

    def _get_cache_key(self, query: str, params: tuple = None) -> str:
        """Создание ключа кэша для запроса"""
        return f"{query}:{params}" if params else query

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Получение из кэша"""
        if not self.enable_cache:
            return None

        if key in self._query_cache:
            value, timestamp = self._query_cache[key]
            if (datetime.now(timezone.utc) - timestamp).total_seconds() < self._cache_ttl:
                return value
            else:
                del self._query_cache[key]
        return None

    def _set_cache(self, key: str, value: Any):
        """Сохранение в кэш"""
        if not self.enable_cache:
            return

        if len(self._query_cache) >= self._cache_max_size:
            # Удаляем oldest entry
            oldest_key = min(self._query_cache.keys(), key=lambda k: self._query_cache[k][1])
            del self._query_cache[oldest_key]

        self._query_cache[key] = (value, datetime.now(timezone.utc))

    def invalidate_cache(self, pattern: str = None):
        """
        Инвалидация кэша

        Args:
            pattern: Шаблон для фильтрации ключей (если None, очищается весь кэш)
        """
        if pattern:
            keys_to_delete = [k for k in self._query_cache.keys() if k.startswith(pattern)]
            for key in keys_to_delete:
                del self._query_cache[key]
        else:
            self._query_cache.clear()

    def set_cache_ttl(self, ttl_seconds: int):
        """Установка времени жизни кэша"""
        self._cache_ttl = ttl_seconds

    def get_cache_stats(self) -> Dict:
        """Получение статистики кэша"""
        now = datetime.now(timezone.utc)
        valid_entries = sum(
            1
            for _, ts in self._query_cache.values()
            if (now - ts).total_seconds() < self._cache_ttl
        )
        return {
            "total_entries": len(self._query_cache),
            "valid_entries": valid_entries,
            "max_size": self._cache_max_size,
            "ttl_seconds": self._cache_ttl,
        }

    # Методы для сравнения изображений поверхностей
    def add_surface_comparison(
        self,
        comparison_id: str,
        image1_path: str,
        image2_path: str,
        similarity_score: float,
        difference_map_path: str = None,
        metrics: Dict = None,
    ) -> int:
        """
        Добавляет результат сравнения поверхностей.

        Args:
            comparison_id: ID сравнения
            image1_path: Путь к первому изображению
            image2_path: Путь ко второму изображению
            similarity_score: Оценка схожести (0-1)
            difference_map_path: Путь к карте различий
            metrics: Метрики сравнения

        Returns:
            ID записи
        """
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO surface_comparisons
                (comparison_id, image1_path, image2_path, similarity_score, difference_map_path, metrics, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    comparison_id,
                    image1_path,
                    image2_path,
                    similarity_score,
                    difference_map_path,
                    json.dumps(metrics) if metrics else None,
                    now,  # Устанавливаем created_at явно
                ),
            )
            return cursor.lastrowid

    def get_surface_comparisons(self, limit: int = 50) -> List[Dict]:
        """Получает историю сравнений поверхностей."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM surface_comparisons
                ORDER BY created_at DESC LIMIT ?
            """,
                (limit,),
            )
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    # Методы для AI/ML анализа дефектов
    def add_defect_analysis(
        self,
        analysis_id: str,
        image_path: str,
        model_name: str,
        defects_detected: int,
        defects_data: Dict = None,
        confidence_score: float = None,
        processing_time_ms: float = None,
    ) -> int:
        """
        Добавляет результат AI анализа дефектов.

        Args:
            analysis_id: ID анализа
            image_path: Путь к изображению
            model_name: Название модели
            defects_detected: Количество обнаруженных дефектов
            defects_data: Детальные данные о дефектах
            confidence_score: Оценка достоверности
            processing_time_ms: Время обработки

        Returns:
            ID записи
        """
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO defect_analysis
                (analysis_id, image_path, model_name, defects_detected, defects_data, confidence_score, processing_time_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    analysis_id,
                    image_path,
                    model_name,
                    defects_detected,
                    json.dumps(defects_data) if defects_data else None,
                    confidence_score,
                    processing_time_ms,
                    now,  # Устанавливаем created_at явно
                ),
            )
            return cursor.lastrowid

    def get_defect_analyses(self, image_path: str = None, limit: int = 50) -> List[Dict]:
        """Получает историю AI анализов."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if image_path:
                cursor.execute(
                    """
                    SELECT * FROM defect_analysis
                    WHERE image_path = ?
                    ORDER BY created_at DESC LIMIT ?
                """,
                    (image_path, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM defect_analysis
                    ORDER BY created_at DESC LIMIT ?
                """,
                    (limit,),
                )

            return [self._row_to_dict(row) for row in cursor.fetchall()]

    # Методы для PDF отчётов
    def add_pdf_report(
        self,
        report_path: str,
        report_type: str,
        title: str = None,
        source_ids: List[int] = None,
        file_size_bytes: int = None,
        pages_count: int = None,
    ) -> int:
        """
        Добавляет запись о PDF отчёте.

        Args:
            report_path: Путь к файлу
            report_type: Тип отчёта
            title: Заголовок
            source_ids: ID исходных данных
            file_size_bytes: Размер файла
            pages_count: Количество страниц

        Returns:
            ID записи
        """
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO pdf_reports
                (report_path, report_type, title, source_ids, file_size_bytes, pages_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    report_path,
                    report_type,
                    title,
                    json.dumps(source_ids) if source_ids else None,
                    file_size_bytes,
                    pages_count,
                    now,  # Устанавливаем created_at явно
                ),
            )
            return cursor.lastrowid

    def get_pdf_reports(self, report_type: str = None, limit: int = 50) -> List[Dict]:
        """Получает список PDF отчётов."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if report_type:
                cursor.execute(
                    """
                    SELECT * FROM pdf_reports
                    WHERE report_type = ?
                    ORDER BY created_at DESC LIMIT ?
                """,
                    (report_type, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM pdf_reports
                    ORDER BY created_at DESC LIMIT ?
                """,
                    (limit,),
                )

            return [self._row_to_dict(row) for row in cursor.fetchall()]

    # Методы для пакетной обработки
    def add_batch_job(
        self, job_id: str, job_type: str, total_items: int = 0, parameters: Dict = None
    ) -> int:
        """
        Добавляет задание пакетной обработки.

        Args:
            job_id: ID задания
            job_type: Тип задания
            total_items: Всего элементов
            parameters: Параметры

        Returns:
            ID записи
        """
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO batch_jobs
                (job_id, job_type, total_items, started_at, parameters, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    job_id,
                    job_type,
                    total_items,
                    now,
                    json.dumps(parameters) if parameters else None,
                    now,  # Устанавливаем created_at явно
                ),
            )
            return cursor.lastrowid

    def update_batch_job(
        self,
        job_id: str,
        status: str = None,
        processed_items: int = None,
        failed_items: int = None,
        results_summary: Dict = None,
    ):
        """Обновляет статус задания пакетной обработки."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            updates = []
            params = []

            if status:
                updates.append("status = ?")
                params.append(status)

                if status in ("completed", "failed", "cancelled"):
                    updates.append("completed_at = ?")
                    params.append(datetime.now(timezone.utc).isoformat())

            if processed_items is not None:
                updates.append("processed_items = ?")
                params.append(processed_items)

            if failed_items is not None:
                updates.append("failed_items = ?")
                params.append(failed_items)

            if results_summary:
                updates.append("results_summary = ?")
                params.append(json.dumps(results_summary))

            params.append(job_id)
            query = f"UPDATE batch_jobs SET {', '.join(updates)} WHERE job_id = ?"
            cursor.execute(query, params)

    def get_batch_jobs(self, status: str = None, limit: int = 50) -> List[Dict]:
        """Получает список заданий пакетной обработки."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if status:
                cursor.execute(
                    """
                    SELECT * FROM batch_jobs
                    WHERE status = ?
                    ORDER BY created_at DESC LIMIT ?
                """,
                    (status, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM batch_jobs
                    ORDER BY created_at DESC LIMIT ?
                """,
                    (limit,),
                )

            return [self._row_to_dict(row) for row in cursor.fetchall()]

    # Методы для real-time метрик производительности
    def add_performance_metric(
        self,
        metric_type: str,
        metric_name: str,
        value: float,
        unit: str = None,
        metadata: Dict = None,
    ) -> int:
        """
        Добавляет метрику производительности.

        Args:
            metric_type: Тип метрики (spm, system, analysis)
            metric_name: Название метрики
            value: Значение
            unit: Единица измерения
            metadata: Дополнительные данные

        Returns:
            ID записи
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO performance_metrics
                (timestamp, metric_type, metric_name, value, unit, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    metric_type,
                    metric_name,
                    value,
                    unit,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            return cursor.lastrowid

    def get_performance_metrics(
        self,
        metric_type: str = None,
        metric_name: str = None,
        start_time: str = None,
        end_time: str = None,
        limit: int = 1000,
    ) -> List[Dict]:
        """Получает метрики производительности."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM performance_metrics WHERE 1=1"
            params = []

            if metric_type:
                query += " AND metric_type = ?"
                params.append(metric_type)

            if metric_name:
                query += " AND metric_name = ?"
                params.append(metric_name)

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def cleanup_old_metrics(self, days: int = 7) -> int:
        """
        Очищает старые метрики производительности.

        Args:
            days: Хранить метрики за последние N дней

        Returns:
            Количество удалённых записей
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cutoff = datetime.now(timezone.utc)
            cutoff = cutoff.replace(day=cutoff.day - days)
            cursor.execute(
                "DELETE FROM performance_metrics WHERE timestamp < ?", (cutoff.isoformat(),)
            )
            return cursor.rowcount

    def export_to_json(self, output_path: str) -> Path:
        """
        Экспортирует всю базу данных в JSON.

        Args:
            output_path: Путь к файлу

        Returns:
            Путь к файлу
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            data = {
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "scan_results": [],
                "simulations": [],
                "images": [],
                "exports": [],
            }

            # Экспорт сканирований
            cursor.execute("SELECT * FROM scan_results")
            data["scan_results"] = [self._row_to_dict(row) for row in cursor.fetchall()]

            # Экспорт симуляций
            cursor.execute("SELECT * FROM simulations")
            data["simulations"] = [self._row_to_dict(row) for row in cursor.fetchall()]

            # Экспорт изображений
            cursor.execute("SELECT * FROM images")
            data["images"] = [self._row_to_dict(row) for row in cursor.fetchall()]

            # Экспорт экспортов
            cursor.execute("SELECT * FROM exports")
            data["exports"] = [self._row_to_dict(row) for row in cursor.fetchall()]

            # Сохранение
            output = Path(output_path)
            with open(output, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return output

    def cleanup_old_records(
        self, table: str, days: int = 30, date_column: str = "created_at"
    ) -> int:
        """
        Удаляет старые записи из таблицы.

        Args:
            table: Имя таблицы
            days: Возраст записей в днях
            date_column: Имя колонки с датой

        Returns:
            Количество удалённых записей
        """
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {table} WHERE {date_column} < ?", (cutoff_date,))
            return cursor.rowcount

    # ==================== User Management ====================

    def get_user(self, username: str) -> Optional[Dict]:
        """Получает пользователя по имени."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, username, password_hash, role, created_at, last_login FROM users WHERE username = ?",
                (username,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """Получает пользователя по ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, username, password_hash, role, created_at, last_login FROM users WHERE id = ?",
                (user_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def upsert_user(self, username: str, password_hash: str, role: str = "user") -> int:
        """
        Создаёт или обновляет пользователя (INSERT OR IGNORE + UPDATE hash если изменился).
        Используется при старте для синхронизации паролей из ENV/файлов.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO users (username, password_hash, role)
                VALUES (?, ?, ?)
                ON CONFLICT(username) DO UPDATE SET
                    password_hash = excluded.password_hash,
                    role = excluded.role
            """,
                (username, password_hash, role),
            )
            return cursor.lastrowid

    def update_last_login(self, username: str) -> None:
        """Обновляет время последнего входа."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE users SET last_login = ? WHERE username = ?",
                (datetime.now(timezone.utc).isoformat(), username),
            )

    def update_password_hash(self, username: str, new_hash: str) -> bool:
        """Обновляет хеш пароля пользователя."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE users SET password_hash = ? WHERE username = ?", (new_hash, username)
            )
            return cursor.rowcount > 0


# Глобальный экземляр для удобства
_db_instance: Optional[DatabaseManager] = None


def get_database(db_path: str = "data/nanoprobe.db") -> DatabaseManager:
    """
    Получает экземпляр менеджера базы данных.

    Args:
        db_path: Путь к файлу БД

    Returns:
        DatabaseManager
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager(db_path)
    return _db_instance
