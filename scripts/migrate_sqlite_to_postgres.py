#!/usr/bin/env python3
"""
Скрипт миграции SQLite → PostgreSQL

Использование:
    python scripts/migrate_sqlite_to_postgres.py --help
    python scripts/migrate_sqlite_to_postgres.py --dry-run
    python scripts/migrate_sqlite_to_postgres.py --execute --pg-url postgresql://user:pass@localhost:5432/nanoprobe

Требования:
    pip install psycopg2-binary asyncpg sqlalchemy
"""

import argparse
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

try:
    import psycopg2
    from psycopg2.extras import execute_batch
except ImportError:
    psycopg2 = None
    print("⚠️  psycopg2 не установлен. Установите: pip install psycopg2-binary")

try:
    import asyncpg
except ImportError:
    asyncpg = None
    print("⚠️  asyncpg не установлен. Установите: pip install asyncpg")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SQLiteToPostgresMigrator:
    """
    Мигратор данных из SQLite в PostgreSQL
    """

    # Типы данных SQLite → PostgreSQL
    TYPE_MAP = {
        "INTEGER": "INTEGER",
        "INT": "INTEGER",
        "BIGINT": "BIGINT",
        "TEXT": "TEXT",
        "VARCHAR": "VARCHAR",
        "BOOLEAN": "BOOLEAN",
        "REAL": "REAL",
        "FLOAT": "DOUBLE PRECISION",
        "BLOB": "BYTEA",
        "DATETIME": "TIMESTAMP WITH TIME ZONE",
        "TIMESTAMP": "TIMESTAMP WITH TIME ZONE",
        "JSON": "JSONB",
    }

    def __init__(
        self,
        sqlite_path: str,
        postgres_url: str,
        batch_size: int = 1000,
    ):
        self.sqlite_path = Path(sqlite_path)
        self.postgres_url = postgres_url
        self.batch_size = batch_size
        self.stats = {
            "tables_migrated": 0,
            "rows_migrated": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None,
        }

    def get_sqlite_tables(self, conn: sqlite3.Connection) -> List[str]:
        """Получение списка таблиц из SQLite"""
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
        return [row[0] for row in cursor.fetchall()]

    def get_table_schema(self, conn: sqlite3.Connection, table: str) -> List[tuple]:
        """Получение схемы таблицы"""
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table})")
        return cursor.fetchall()

    def convert_column_type(self, sqlite_type: str) -> str:
        """Конвертация типа SQLite → PostgreSQL"""
        sqlite_type_upper = sqlite_type.upper()
        for sqlite_key, pg_value in self.TYPE_MAP.items():
            if sqlite_key in sqlite_type_upper:
                return pg_value
        return "TEXT"  # По умолчанию

    def generate_create_table(self, table: str, schema: List[tuple]) -> str:
        """Генерация CREATE TABLE для PostgreSQL"""
        columns = []
        primary_keys = []

        for col in schema:
            col_id, col_name, col_type, not_null, default_value, is_pk = col

            # Конвертируем тип
            pg_type = self.convert_column_type(col_type)

            # Строим определение колонки
            col_def = f"{col_name} {pg_type}"

            # Добавляем constraints
            if is_pk:
                primary_keys.append(col_name)
            if not_null and not is_pk:  # PRIMARY KEY уже подразумевает NOT NULL
                col_def += " NOT NULL"
            if default_value is not None:
                # Конвертируем boolean
                if default_value in ("0", "1") and pg_type == "BOOLEAN":
                    col_def += f" DEFAULT {bool(int(default_value))}"
                else:
                    col_def += f" DEFAULT {default_value}"

            columns.append(col_def)

        # Добавляем PRIMARY KEY constraint
        if primary_keys:
            columns.append(f"PRIMARY KEY ({', '.join(primary_keys)})")

        create_sql = f"CREATE TABLE IF NOT EXISTS {table} (\n    " + ",\n    ".join(columns) + "\n)"
        return create_sql

    def migrate_schema(self, sqlite_conn: sqlite3.Connection, pg_conn) -> Dict[str, int]:
        """Миграция схемы"""
        tables = self.get_sqlite_tables(sqlite_conn)
        table_row_counts = {}

        logger.info(f"📋 Найдено {len(tables)} таблиц для миграции")

        for table in tables:
            schema = self.get_table_schema(sqlite_conn, table)
            create_sql = self.generate_create_table(table, schema)

            try:
                with pg_conn.cursor() as cur:
                    cur.execute(create_sql)
                    pg_conn.commit()

                # Считаем строки
                cursor = sqlite_conn.cursor()
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]
                table_row_counts[table] = row_count

                logger.info(f"  ✓ {table}: {row_count} строк, {len(schema)} колонок")

            except Exception as e:
                logger.error(f"  ✗ {table}: ошибка создания таблицы - {e}")
                self.stats["errors"] += 1

        return table_row_counts

    def migrate_table_data(
        self, sqlite_conn: sqlite3.Connection, pg_conn, table: str, total_rows: int
    ):
        """Миграция данных одной таблицы"""
        logger.info(f"  📦 Миграция данных {table} ({total_rows} строк)...")

        cursor = sqlite_conn.cursor()
        cursor.execute(f"SELECT * FROM {table}")

        # Получаем имена колонок
        column_names = [desc[0] for desc in cursor.description]

        # Фильтруем rowid (внутренняя колонка SQLite)
        if "rowid" in column_names:
            column_names.remove("rowid")

        # Строим INSERT запрос
        placeholders = ", ".join(["%s"] * len(column_names))
        columns_str = ", ".join(column_names)
        insert_sql = f"INSERT INTO {table} ({columns_str}) VALUES ({placeholders})"

        # Мигрируем батчами
        batch = []
        rows_migrated = 0
        errors = 0

        for row in cursor:
            # Конвертируем данные
            row_dict = dict(zip(cursor.description, row))
            values = []

            for col_name in column_names:
                value = row_dict[col_name]

                # Конвертируем datetime
                if isinstance(value, str) and "T" in value:
                    try:
                        # Проверяем, это datetime строка
                        if value.endswith("Z"):
                            value = value[:-1] + "+00:00"
                    except (ValueError, TypeError):
                        pass

                # Конвертируем boolean
                if value in ("0", "1"):
                    # Пытаемся определить, boolean ли это
                    col_type = None
                    for desc in cursor.description:
                        if desc[0] == col_name:
                            col_type = desc[1]
                            break
                    if col_type and "BOOL" in str(col_type).upper():
                        value = bool(int(value))

                values.append(value)

            batch.append(values)

            # Вставляем батч
            if len(batch) >= self.batch_size:
                try:
                    with pg_conn.cursor() as cur:
                        execute_batch(cur, insert_sql, batch)
                        pg_conn.commit()
                    rows_migrated += len(batch)
                    batch = []

                    # Прогресс
                    if rows_migrated % (self.batch_size * 10) == 0:
                        progress = (rows_migrated / total_rows) * 100 if total_rows > 0 else 0
                        logger.info(f"    ⏳ {rows_migrated}/{total_rows} строк ({progress:.1f}%)")

                except Exception as e:
                    logger.error(f"    ✗ Ошибка вставки батча: {e}")
                    errors += len(batch)
                    batch = []

        # Вставляем остаток
        if batch:
            try:
                with pg_conn.cursor() as cur:
                    execute_batch(cur, insert_sql, batch)
                    pg_conn.commit()
                rows_migrated += len(batch)
            except Exception as e:
                logger.error(f"    ✗ Ошибка вставки последнего батча: {e}")
                errors += len(batch)

        self.stats["rows_migrated"] += rows_migrated
        self.stats["errors"] += errors

        logger.info(f"  ✓ {table}: мигрировано {rows_migrated} строк, ошибок: {errors}")

    def create_indexes(self, pg_conn):
        """Создание индексов после миграции"""
        logger.info("🔧 Создание индексов...")

        indexes = [
            # scans
            (
                "scans",
                "idx_scans_user_id",
                "CREATE INDEX IF NOT EXISTS idx_scans_user_id ON scans (user_id)",
            ),
            (
                "scans",
                "idx_scans_created_at",
                "CREATE INDEX IF NOT EXISTS idx_scans_created_at ON scans (created_at DESC)",
            ),
            (
                "scans",
                "idx_scans_status",
                "CREATE INDEX IF NOT EXISTS idx_scans_status ON scans (status)",
            ),
            # simulations
            (
                "simulations",
                "idx_simulations_user_id",
                "CREATE INDEX IF NOT EXISTS idx_simulations_user_id ON simulations (user_id)",
            ),
            (
                "simulations",
                "idx_simulations_created_at",
                "CREATE INDEX IF NOT EXISTS idx_simulations_created_at ON simulations (created_at DESC)",
            ),
            # users
            (
                "users",
                "idx_users_email",
                "CREATE INDEX IF NOT EXISTS idx_users_email ON users (email)",
            ),
            (
                "users",
                "idx_users_username",
                "CREATE INDEX IF NOT EXISTS idx_users_username ON users (username)",
            ),
        ]

        created = 0
        skipped = 0

        with pg_conn.cursor() as cur:
            for table, idx_name, idx_sql in indexes:
                try:
                    # Проверяем существует ли таблица
                    cur.execute(
                        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
                        (table,),
                    )
                    if cur.fetchone()[0]:
                        cur.execute(idx_sql)
                        pg_conn.commit()
                        created += 1
                    else:
                        skipped += 1
                except Exception as e:
                    logger.warning(f"  ⚠️  {idx_name}: {e}")
                    skipped += 1

        logger.info(f"  ✓ Создано {created} индексов, пропущено {skipped}")

    def run(self, dry_run: bool = False):
        """
        Запуск миграции

        Args:
            dry_run: Если True, только анализ без выполнения
        """
        self.stats["start_time"] = datetime.now(timezone.utc)

        logger.info("=" * 80)
        logger.info("SQLite → PostgreSQL Migration")
        logger.info("=" * 80)
        logger.info(f"SQLite: {self.sqlite_path}")
        logger.info(
            f"PostgreSQL: {self.postgres_url.split('@')[-1] if '@' in self.postgres_url else self.postgres_url}"
        )
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Dry run: {dry_run}")
        logger.info("=" * 80)

        # Проверяем зависимости
        if psycopg2 is None:
            logger.error("❌ psycopg2 не установлен")
            return False

        if not self.sqlite_path.exists():
            logger.error(f"❌ SQLite файл не найден: {self.sqlite_path}")
            return False

        # Подключаемся к SQLite
        try:
            sqlite_conn = sqlite3.connect(str(self.sqlite_path))
            logger.info("✓ Подключено к SQLite")
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к SQLite: {e}")
            return False

        # Подключаемся к PostgreSQL только если не dry-run
        pg_conn = None
        if not dry_run:
            try:
                pg_conn = psycopg2.connect(self.postgres_url)
                logger.info("✓ Подключено к PostgreSQL")
            except Exception as e:
                logger.error(f"❌ Ошибка подключения к PostgreSQL: {e}")
                sqlite_conn.close()
                return False

        try:
            if dry_run:
                logger.info("\n📊 DRY RUN - анализ структуры...")
                tables = self.get_sqlite_tables(sqlite_conn)
                total_rows = 0

                for table in tables:
                    schema = self.get_table_schema(sqlite_conn, table)
                    cursor = sqlite_conn.cursor()
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    row_count = cursor.fetchone()[0]
                    total_rows += row_count

                    logger.info(f"  {table}: {len(schema)} колонок, {row_count} строк")
                    # Показываем схему
                    for col in schema:
                        col_id, col_name, col_type, not_null, default_value, is_pk = col
                        pg_type = self.convert_column_type(col_type)
                        pk_marker = " [PK]" if is_pk else ""
                        nn_marker = " [NOT NULL]" if not_null and not is_pk else ""
                        logger.info(
                            f"    - {col_name}: {col_type} → {pg_type}{pk_marker}{nn_marker}"
                        )

                logger.info(f"\n📈 Итого: {len(tables)} таблиц, ~{total_rows} строк")
                logger.info("✅ Dry run завершён. Для выполнения добавьте --execute")
                return True

            # Выполняем миграцию
            logger.info("\n🚀 Начало миграции...")

            # 1. Миграция схемы
            logger.info("\n📋 Шаг 1: Миграция схемы...")
            table_row_counts = self.migrate_schema(sqlite_conn, pg_conn)

            # 2. Миграция данных
            logger.info("\n📦 Шаг 2: Миграция данных...")
            for table, row_count in table_row_counts.items():
                if row_count > 0:
                    self.migrate_table_data(sqlite_conn, pg_conn, table, row_count)

            # 3. Создание индексов
            logger.info("\n🔧 Шаг 3: Создание индексов...")
            self.create_indexes(pg_conn)

            # Финальная статистика
            self.stats["end_time"] = datetime.now(timezone.utc)
            duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()

            logger.info("\n" + "=" * 80)
            logger.info("✅ МИГРАЦИЯ ЗАВЕРШЕНА")
            logger.info("=" * 80)
            logger.info(f"Таблиц мигрировано: {self.stats['tables_migrated']}")
            logger.info(f"Строк мигрировано: {self.stats['rows_migrated']}")
            logger.info(f"Ошибок: {self.stats['errors']}")
            logger.info(f"Время выполнения: {duration:.2f} секунд")
            logger.info("=" * 80)

            return self.stats["errors"] == 0

        except Exception as e:
            logger.error(f"❌ Ошибка миграции: {e}")
            import traceback

            traceback.print_exc()
            return False

        finally:
            sqlite_conn.close()
            if pg_conn:
                pg_conn.close()
            logger.info("✓ Соединения закрыты")


def main():
    parser = argparse.ArgumentParser(
        description="Миграция SQLite → PostgreSQL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Dry run (анализ без выполнения)
  python scripts/migrate_sqlite_to_postgres.py --dry-run

  # Выполнение миграции
  python scripts/migrate_sqlite_to_postgres.py --execute --pg-url postgresql://user:pass@localhost:5432/nanoprobe

  # С кастомным batch size
  python scripts/migrate_sqlite_to_postgres.py --execute --pg-url postgresql://user:pass@localhost:5432/nanoprobe --batch-size 500
        """,
    )

    parser.add_argument(
        "--sqlite-path",
        default="data/nanoprobe.db",
        help="Путь к SQLite базе (по умолчанию: data/nanoprobe.db)",
    )
    parser.add_argument(
        "--pg-url",
        required=True,
        help="PostgreSQL connection URL (postgresql://user:pass@host:port/dbname)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1000, help="Размер батча для вставки (по умолчанию: 1000)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Только анализ без выполнения")
    parser.add_argument("--execute", action="store_true", help="Выполнить миграцию")

    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        parser.error("Укажите --dry-run или --execute")

    migrator = SQLiteToPostgresMigrator(
        sqlite_path=args.sqlite_path,
        postgres_url=args.pg_url,
        batch_size=args.batch_size,
    )

    success = migrator.run(dry_run=args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
