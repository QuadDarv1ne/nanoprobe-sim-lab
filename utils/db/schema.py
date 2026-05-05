"""Database schema and utilities for Nanoprobe Sim Lab."""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List


def get_database_stats(conn) -> Dict[str, Any]:
    """Получение статистики базы данных."""
    cursor = conn.cursor()

    tables: Dict[str, int] = {}
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    for row in cursor.fetchall():
        table_name = row[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        tables[table_name] = count

    indexes: Dict[str, List[str]] = {}
    cursor.execute(
        "SELECT name, tbl_name "
        "FROM sqlite_master "
        "WHERE type='index' "
        "AND name NOT LIKE 'sqlite_%' "
        "ORDER BY tbl_name"
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


def init_database_schema(conn):
    """Инициализация схемы базы данных (создание таблиц и индексов)."""
    cursor = conn.cursor()

    # Таблица результатов сканирования
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
            created_at TEXT NOT NULL,
            updated_at TEXT
        )
        """
    )

    # Таблица результатов симуляций
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS simulation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            simulation_type TEXT NOT NULL,
            parameters TEXT NOT NULL,
            results_summary TEXT NOT NULL,
            metrics TEXT,
            file_path TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT
        )
        """
    )

    # Индексы для ускорения запросов
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_scan_type
        ON scan_results(scan_type)
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_scan_timestamp
        ON scan_results(timestamp)
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_sim_type
        ON simulation_results(simulation_type)
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_sim_timestamp
        ON simulation_results(timestamp)
        """
    )

    conn.commit()


class DBSchema:
    """Класс для управления схемой базы данных."""

    def __init__(self, db_path: str):
        """
        Инициализация схемы БД.

        Args:
            db_path: Путь к файлу SQLite базы данных
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.create_tables()

    def create_tables(self):
        """Создание таблиц БД если они не существуют."""
        with sqlite3.connect(str(self.db_path)) as conn:
            init_database_schema(conn)

    def get_database_stats(self, conn) -> Dict[str, Any]:
        """Получение статистики базы данных."""
        return get_database_stats(conn)

    def backup_database(self, backup_path: str):
        """
        Создать резервную копию БД.

        Args:
            backup_path: Путь для сохранения резервной копии
        """
        import shutil

        shutil.copy2(str(self.db_path), backup_path)

    def vacuum_database(self, conn):
        """Оптимизация БД (VACUUM)."""
        cursor = conn.cursor()
        cursor.execute("VACUUM")
        conn.commit()

    def analyze_database(self, conn):
        """Анализ БД для оптимизации запросов."""
        cursor = conn.cursor()
        cursor.execute("ANALYZE")
        conn.commit()
