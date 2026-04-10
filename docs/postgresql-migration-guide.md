# Миграция SQLite → PostgreSQL — Руководство

## Текущее состояние

- **База данных:** SQLite (`data/nanoprobe.db`)
- **Connection Pool:** Кастомный пул для SQLite
- **PRAGMA:** WAL mode, synchronous=NORMAL, cache_size=64MB
- **Миграции:** Alembic (3 миграции)

## Почему PostgreSQL для production?

| Фича | SQLite | PostgreSQL |
|------|--------|------------|
| Конкурентность | Ограниченная | Полная (MVCC) |
| Масштабирование | Single-node | Multi-node, replication |
| Типы данных | Базовые | Расширенные (JSONB, arrays) |
| Индексы | Простые | Partial, expression, GIN |
| Транзакции | Есть | Полные + savepoints |
| Бэкапы | Файл | pg_dump, PITR |

## План миграции

### Шаг 1: Установка PostgreSQL

```bash
# Windows
# Скачать: https://www.postgresql.org/download/windows/

# Linux
sudo apt install postgresql postgresql-contrib

# Docker
docker run -d --name postgres \
  -e POSTGRES_PASSWORD=nanoprobe_secret \
  -e POSTGRES_DB=nanoprobe \
  -p 5432:5432 \
  postgres:16-alpine
```

### Шаг 2: Создание базы

```sql
CREATE DATABASE nanoprobe;
CREATE USER nanoprobe_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE nanoprobe TO nanoprobe_user;
```

### Шаг 3: Установка зависимостей

```bash
pip install psycopg2-binary  # или psycopg2 для production
pip install asyncpg          # для async поддержки
```

### Шаг 4: Обновление .env

```env
# Database
DATABASE_URL=postgresql://nanoprobe_user:secure_password@localhost:5432/nanoprobe
DATABASE_TYPE=postgresql
```

### Шаг 5: Миграция данных

```bash
# Вариант 1: Использовать скрипт
python scripts/migrate_sqlite_to_postgres.py

# Вариант 2: Alembic (рекомендуется)
# 1. Создать новую миграцию
alembic revision --autogenerate -m "migrate_to_postgresql"

# 2. Применить миграцию
alembic upgrade head
```

### Шаг 6: Обновление database.py

Текущий `utils/database.py` использует sqlite3 напрямую.
Необходимо:

1. **SQLAlchemy Core** — абстракция над БД
2. **Connection Pool** — заменить кастомный пул на SQLAlchemy QueuePool
3. **PRAGMA** → **SET** — конвертировать SQLite PRAGMA в PostgreSQL SET
4. **Типы данных** — TEXT → VARCHAR, DATETIME → TIMESTAMPTZ

### Пример обновления Connection Pool

```python
# БЫЛО (SQLite):
import sqlite3
conn = sqlite3.connect(self.db_path)
conn.execute("PRAGMA journal_mode = WAL")

# СТАЛО (PostgreSQL):
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Health check
)
```

### Шаг 7: Alembic миграции

Текущие миграции совместимы с обоими БД благодаря Alembic.
Проверить:

```bash
alembic check  # Проверить статус миграций
alembic current  # Текущая версия
```

### Шаг 8: Тестирование

```bash
# 1. Unit тесты
pytest tests/test_database.py -v

# 2. Integration тесты
pytest tests/test_integration_db.py -v

# 3. Нагрузочные тест
python tests/load_test.py
```

## Альтернатива: Гибридный режим

Разрешить использование обоих БД:

```python
# utils/database.py
import os

DB_TYPE = os.getenv("DATABASE_TYPE", "sqlite")

if DB_TYPE == "postgresql":
    # Использовать SQLAlchemy + psycopg2
    from .db_postgresql import PostgreSQLManager
    DatabaseManager = PostgreSQLManager
else:
    # Использовать sqlite3
    from .db_sqlite import SQLiteManager
    DatabaseManager = SQLiteManager
```

## Оценка сложности

| Компонент | Сложность | Время |
|-----------|-----------|-------|
| Connection Pool refactor | Высокая | 4-6 часов |
| Типы данных | Средняя | 2-3 часа |
| Миграция данных | Низкая | 1-2 часа |
| Тестирование | Средняя | 3-4 часа |
| **Итого** | | **10-15 часов** |

## Рекомендация

**Для production:** PostgreSQL обязательно
**Для development:** SQLite оставить (проще для локальной разработки)

**Минимальная конфигурация PostgreSQL:**
```ini
# postgresql.conf
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
max_connections = 100
```

## Ресурсы

- [Alembic PostgreSQL](https://alembic.sqlalchemy.org/en/latest/)
- [SQLAlchemy PostgreSQL](https://docs.sqlalchemy.org/en/20/dialects/postgresql.html)
- [Миграция данных](https://www.pgloader.io/)
