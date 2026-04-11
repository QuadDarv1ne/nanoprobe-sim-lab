"""
Database Composite Indexes

Скрипт для создания составных индексов в базе данных.
Оптимизация запросов для часто используемых паттернов.

Usage:
    python apply_indexes.py [--dry-run]
"""

import logging
import sqlite3
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = "data/nanoprobe.db"

# Composite indexes для оптимизации запросов
COMPOSITE_INDEXES = [
    # Scans table
    {
        "name": "idx_scans_user_status",
        "table": "scans",
        "columns": ["user_id", "status"],
        "where": "deleted_at IS NULL",
        "description": "Оптимизация поиска сканов по пользователю и статусу",
    },
    {
        "name": "idx_scans_user_created",
        "table": "scans",
        "columns": ["user_id", "created_at DESC"],
        "description": "Оптимизация получения последних сканов пользователя",
    },
    {
        "name": "idx_scans_status_created",
        "table": "scans",
        "columns": ["status", "created_at DESC"],
        "description": "Оптимизация поиска по статусу с сортировкой по дате",
    },
    # Simulations table
    {
        "name": "idx_simulations_user_status",
        "table": "simulations",
        "columns": ["user_id", "status"],
        "where": "deleted_at IS NULL",
        "description": "Оптимизация поиска симуляций по пользователю и статусу",
    },
    {
        "name": "idx_simulations_user_created",
        "table": "simulations",
        "columns": ["user_id", "created_at DESC"],
        "description": "Оптимизация получения последних симуляций",
    },
    {
        "name": "idx_simulations_type_status",
        "table": "simulations",
        "columns": ["simulation_type", "status"],
        "description": "Оптимизация поиска по типу симуляции",
    },
    # Analysis table
    {
        "name": "idx_analysis_scan_user",
        "table": "analysis",
        "columns": ["scan_id", "user_id"],
        "description": "Оптимизация поиска анализа по скану и пользователю",
    },
    {
        "name": "idx_analysis_created",
        "table": "analysis",
        "columns": ["created_at DESC"],
        "description": "Оптимизация получения последнего анализа",
    },
    # Reports table
    {
        "name": "idx_reports_user_created",
        "table": "reports",
        "columns": ["user_id", "created_at DESC"],
        "description": "Оптимизация получения отчётов пользователя",
    },
    {
        "name": "idx_reports_status",
        "table": "reports",
        "columns": ["status", "created_at DESC"],
        "description": "Оптимизация поиска отчётов по статусу",
    },
    # Users table (если существует)
    {
        "name": "idx_users_email",
        "table": "users",
        "columns": ["email"],
        "unique": True,
        "description": "Уникальный индекс на email для быстрого поиска",
    },
    {
        "name": "idx_users_username",
        "table": "users",
        "columns": ["username"],
        "unique": True,
        "description": "Уникальный индекс на username",
    },
]


def get_existing_indexes(conn: sqlite3.Connection) -> set:
    """Получение списка существующих индексов"""
    cursor = conn.execute(
        """
        SELECT name FROM sqlite_master
        WHERE type='index' AND name NOT LIKE 'sqlite_%'
    """
    )
    return {row[0] for row in cursor.fetchall()}


def create_index(conn: sqlite3.Connection, index_def: dict, dry_run: bool = False) -> bool:
    """
    Создание индекса.

    Args:
        conn: SQLite соединение
        index_def: Определение индекса
        dry_run: Не выполнять, только показать

    Returns:
        True если индекс создан
    """
    table = index_def["table"]
    name = index_def["name"]
    columns = ", ".join(index_def["columns"])
    unique = index_def.get("unique", False)
    where = index_def.get("where")

    # Формирование SQL
    sql = f"CREATE {'UNIQUE ' if unique else ''}INDEX IF NOT EXISTS {name} ON {table} ({columns})"
    if where:
        sql += f" WHERE {where}"

    if dry_run:
        logger.info(f"[DRY RUN] Would create index: {name}")
        logger.info(f"  SQL: {sql}")
        logger.info(f"  Description: {index_def.get('description', 'N/A')}")
        return True

    try:
        conn.execute(sql)
        conn.commit()
        logger.info(f"✓ Created index: {name}")
        logger.info(f"  {index_def.get('description', '')}")
        return True
    except sqlite3.Error as e:
        logger.error(f"✗ Failed to create index {name}: {e}")
        return False


def analyze_index_usage(conn: sqlite3.Connection):
    """Анализ использования индексов"""
    logger.info("\n" + "=" * 60)
    logger.info("Index Usage Statistics")
    logger.info("=" * 60)

    try:
        cursor = conn.execute(
            """
            SELECT
                name,
                tbl_name,
                sql
            FROM sqlite_master
            WHERE type='index' AND name NOT LIKE 'sqlite_%'
            ORDER BY tbl_name, name
        """
        )

        for row in cursor.fetchall():
            logger.info(f"\nIndex: {row[0]}")
            logger.info(f"  Table: {row[1]}")
            logger.info(f"  Definition: {row[2]}")

    except sqlite3.Error as e:
        logger.error(f"Error analyzing indexes: {e}")


def get_database_size(conn: sqlite3.Connection) -> int:
    """Получение размера базы данных"""
    cursor = conn.execute(
        "SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()"
    )
    return cursor.fetchone()[0]


def get_table_sizes(conn: sqlite3.Connection) -> dict:
    """Получение размеров таблиц"""
    sizes = {}

    try:
        cursor = conn.execute(
            """
            SELECT name FROM sqlite_master WHERE type='table'
        """
        )

        for row in cursor.fetchall():
            table_name = row[0]
            if table_name.startswith("sqlite_"):
                continue

            try:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                sizes[table_name] = count
            except sqlite3.Error:
                pass

        return sizes

    except sqlite3.Error as e:
        logger.error(f"Error getting table sizes: {e}")
        return {}


def main():
    """Основная функция"""
    import argparse

    parser = argparse.ArgumentParser(description="Create composite database indexes")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without making changes"
    )
    parser.add_argument("--analyze", action="store_true", help="Analyze index usage after creation")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Database Composite Indexes")
    logger.info("=" * 60)

    # Проверка существования базы данных
    db_path = Path(DB_PATH)
    if not db_path.exists():
        logger.error(f"Database not found: {DB_PATH}")
        logger.info("Run the application first to create the database")
        return

    # Подключение к базе данных
    conn = sqlite3.connect(DB_PATH)

    try:
        # Получение существующих индексов
        existing = get_existing_indexes(conn)
        logger.info(f"\nExisting indexes: {len(existing)}")

        # Получение размеров таблиц
        table_sizes = get_table_sizes(conn)
        logger.info("\nTable sizes:")
        for table, count in table_sizes.items():
            logger.info(f"  {table}: {count} rows")

        # Создание индексов
        created = 0
        skipped = 0

        logger.info("\nCreating indexes...")
        for index_def in COMPOSITE_INDEXES:
            name = index_def["name"]

            if name in existing:
                logger.info(f"⊘ Skip (exists): {name}")
                skipped += 1
                continue

            if create_index(conn, index_def, args.dry_run):
                created += 1

        logger.info(f"\n{'=' * 60}")
        logger.info("Summary")
        logger.info(f"{'=' * 60}")
        logger.info(f"  Created: {created}")
        logger.info(f"  Skipped: {skipped}")
        logger.info(f"  Total: {len(COMPOSITE_INDEXES)}")

        # Анализ использования индексов
        if args.analyze and not args.dry_run:
            analyze_index_usage(conn)

        # Размер базы данных
        db_size = get_database_size(conn)
        logger.info(f"\nDatabase size: {db_size / 1024 / 1024:.2f} MB")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise
    finally:
        conn.close()
        logger.info("\nDone!")


if __name__ == "__main__":
    main()
