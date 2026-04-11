# Database Optimization Guide

## Обзор

Оптимизация базы данных критична для производительности API. Этот документ описывает комплексный подход к оптимизации PostgreSQL.

## 1. EXPLAIN ANALYZE

### Базовый анализ запросов

```sql
-- Анализ конкретного запроса
EXPLAIN ANALYZE
SELECT * FROM scans WHERE user_id = 123 AND created_at > '2024-01-01';

-- Подробный анализ с затратами
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT s.*, u.username
FROM scans s
JOIN users u ON s.user_id = u.id
WHERE s.status = 'completed'
ORDER BY s.created_at DESC
LIMIT 50;
```

### Типичные проблемы и решения

| Проблема | Признак | Решение |
|----------|---------|---------|
| Seq Scan | Full table scan | Добавить индекс |
| Nested Loop | Медленный JOIN | Проверить индексы |
| HashAggregate | Большая группировка | Оптимизировать GROUP BY |
| Sort | Долгая сортировка | Индекс по ORDER BY |
| Filter | Много фильтров | Composite index |

### Скрипт автоматического анализа

```python
# utils/database/query_analyzer.py
import asyncpg
from typing import List, Dict, Any
import logging
import json

logger = logging.getLogger(__name__)

class QueryAnalyzer:
    """Анализатор SQL запросов с рекомендациями"""

    def __init__(self, connection: asyncpg.Connection):
        self.conn = connection

    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """Проанализировать запрос и вернуть рекомендации"""

        # Получаем план выполнения
        explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"
        result = await self.conn.fetchval(explain_query)

        plan = json.loads(result)[0]

        analysis = {
            "plan": plan,
            "execution_time_ms": plan.get("Execution Time", 0),
            "planning_time_ms": plan.get("Planning Time", 0),
            "total_cost": plan.get("Plan", {}).get("Total Cost", 0),
            "issues": [],
            "recommendations": []
        }

        # Анализ плана
        self._analyze_plan_node(plan.get("Plan", {}), analysis)

        return analysis

    def _analyze_plan_node(self, node: Dict, analysis: Dict):
        """Рекурсивный анализ узла плана"""

        node_type = node.get("Node Type", "")

        # Проверка Seq Scan
        if node_type == "Seq Scan":
            table = node.get("Relation Name", "unknown")
            analysis["issues"].append({
                "type": "seq_scan",
                "table": table,
                "severity": "high",
                "message": f"Sequential scan on table '{table}'"
            })
            analysis["recommendations"].append({
                "action": "add_index",
                "table": table,
                "suggestion": f"Consider adding an index on frequently filtered columns in '{table}'"
            })

        # Проверка большого количества строк
        actual_rows = node.get("Actual Rows", 0)
        plan_rows = node.get("Plan Rows", 0)

        if actual_rows > 10000:
            analysis["issues"].append({
                "type": "large_result",
                "rows": actual_rows,
                "severity": "medium",
                "message": f"Large result set: {actual_rows} rows"
            })

        # Проверка hash join с большим количеством строк
        if node_type in ["Hash Join", "Hash Aggregate"]:
            if actual_rows > 100000:
                analysis["recommendations"].append({
                    "action": "optimize_join",
                    "suggestion": "Consider adding indexes or restructuring the query"
                })

        # Рекурсивный анализ дочерних узлов
        for key in ["Plans", "plan"]:
            if key in node:
                for child in node[key]:
                    self._analyze_plan_node(child, analysis)

    async def analyze_slow_queries(self, limit: int = 10) -> List[Dict]:
        """Получить самые медленные запросы из pg_stat_statements"""

        try:
            queries = await self.conn.fetch("""
                SELECT
                    query,
                    calls,
                    total_exec_time / 1000 as total_time_sec,
                    mean_exec_time as avg_time_ms,
                    rows,
                    100.0 * total_exec_time / SUM(total_exec_time) OVER() as percentage
                FROM pg_stat_statements
                ORDER BY total_exec_time DESC
                LIMIT $1
            """, limit)

            return [dict(q) for q in queries]
        except Exception as e:
            logger.error(f"Error analyzing slow queries: {e}")
            return []

    async def get_table_stats(self) -> List[Dict]:
        """Получить статистику по таблицам"""

        stats = await self.conn.fetch("""
            SELECT
                schemaname,
                tablename,
                n_live_tup as row_count,
                n_dead_tup as dead_rows,
                last_vacuum,
                last_autovacuum,
                last_analyze,
                last_autoanalyze
            FROM pg_stat_user_tables
            ORDER BY n_live_tup DESC
        """)

        return [dict(s) for s in stats]

    async def get_index_usage(self) -> List[Dict]:
        """Получить статистику использования индексов"""

        stats = await self.conn.fetch("""
            SELECT
                schemaname,
                tablename,
                indexname,
                idx_scan as index_scans,
                idx_tup_read as tuples_read,
                idx_tup_fetch as tuples_fetched,
                pg_size_pretty(pg_relation_size(indexrelid)) as index_size
            FROM pg_stat_user_indexes
            ORDER BY idx_scan DESC
        """)

        return [dict(s) for s in stats]

    async def find_unused_indexes(self) -> List[Dict]:
        """Найти неиспользуемые индексы"""

        indexes = await self.conn.fetch("""
            SELECT
                schemaname,
                tablename,
                indexname,
                pg_size_pretty(pg_relation_size(indexrelid)) as index_size
            FROM pg_stat_user_indexes
            WHERE idx_scan = 0
            AND indexname NOT LIKE '%_pkey'
            ORDER BY pg_relation_size(indexrelid) DESC
        """)

        return [dict(i) for i in indexes]


# Usage
async def main():
    conn = await asyncpg.connect("postgresql://user:pass@localhost/db")
    analyzer = QueryAnalyzer(conn)

    # Analyze a query
    analysis = await analyzer.analyze_query("""
        SELECT * FROM scans WHERE user_id = 123
    """)

    print(f"Execution time: {analysis['execution_time_ms']:.2f}ms")
    print(f"Issues: {len(analysis['issues'])}")
    for issue in analysis["issues"]:
        print(f"  - {issue['message']}")
```

## 2. Composite Indexes

### Стратегия создания индексов

```sql
-- ==========================================
-- COMPOSITE INDEXES FOR NANOPROBE SIM LAB
-- ==========================================

-- Scans table
-- Частые запросы: user_id + status, user_id + created_at
CREATE INDEX CONCURRENTLY idx_scans_user_status
    ON scans(user_id, status)
    WHERE deleted_at IS NULL;

CREATE INDEX CONCURRENTLY idx_scans_user_created
    ON scans(user_id, created_at DESC)
    WHERE deleted_at IS NULL;

CREATE INDEX CONCURRENTLY idx_scans_status_created
    ON scans(status, created_at DESC);

-- Simulations table
CREATE INDEX CONCURRENTLY idx_simulations_user_type_status
    ON simulations(user_id, simulation_type, status);

CREATE INDEX CONCURRENTLY idx_simulations_active
    ON simulations(started_at)
    WHERE status = 'running';

-- Analysis results
CREATE INDEX CONCURRENTLY idx_analysis_scan_defects
    ON analysis_results(scan_id, defect_count DESC);

CREATE INDEX CONCURRENTLY idx_analysis_type_created
    ON analysis_results(analysis_type, created_at DESC);

-- Users table
CREATE INDEX CONCURRENTLY idx_users_email_active
    ON users(email)
    WHERE is_active = TRUE;

CREATE INDEX CONCURRENTLY idx_users_created
    ON users(created_at DESC);

-- Audit log
CREATE INDEX CONCURRENTLY idx_audit_user_action_time
    ON audit_log(user_id, action, created_at DESC);

CREATE INDEX CONCURRENTLY idx_audit_entity
    ON audit_log(entity_type, entity_id);

-- Sessions (for JWT)
CREATE INDEX CONCURRENTLY idx_sessions_user_valid
    ON sessions(user_id, expires_at)
    WHERE revoked = FALSE;

-- Partial indexes для конкретных сценариев
CREATE INDEX CONCURRENTLY idx_scans_completed
    ON scans(completed_at DESC)
    WHERE status = 'completed';

CREATE INDEX CONCURRENTLY idx_scans_failed
    ON scans(created_at DESC)
    WHERE status = 'failed';
```

### Правила создания composite indexes

1. **Порядок колонок важен!**
   ```sql
   -- ПРАВИЛЬНО: WHERE user_id = ? AND status = ?
   CREATE INDEX idx_good ON scans(user_id, status);

   -- НЕПРАВИЛЬНО: WHERE status = ? (user_id не указан)
   -- Индекс не будет использован!
   ```

2. **Equality перед Range**
   ```sql
   -- ПРАВИЛЬНО: equality + range
   CREATE INDEX idx_good ON scans(user_id, created_at DESC);
   -- WHERE user_id = ? AND created_at > ?

   -- МЕНЕЕ ЭФФЕКТИВНО: range + equality
   CREATE INDEX idx_bad ON scans(created_at DESC, user_id);
   ```

3. **Include колонки для Index-Only Scan**
   ```sql
   -- Index-only scan без обращения к таблице
   CREATE INDEX idx_scans_user_status_include
       ON scans(user_id, status)
       INCLUDE (scan_type, file_path);

   -- SELECT scan_type, file_path FROM scans WHERE user_id = ?
   -- Вернёт данные только из индекса!
   ```

## 3. Query Caching

### Redis Query Cache

```python
# utils/database/query_cache.py
import hashlib
import json
from typing import Optional, Any, List, Callable
from functools import wraps
import asyncio
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)

class QueryCache:
    """
    Кэш для результатов SQL запросов.

    Features:
    - Automatic key generation
    - TTL management
    - Cache invalidation patterns
    - Compression for large results
    """

    def __init__(
        self,
        redis_client,
        prefix: str = "db_cache:",
        default_ttl: int = 300,  # 5 minutes
        compress_threshold: int = 1024  # bytes
    ):
        self.redis = redis_client
        self.prefix = prefix
        self.default_ttl = default_ttl
        self.compress_threshold = compress_threshold

    def _generate_key(self, query: str, params: tuple = ()) -> str:
        """Генерация ключа кэша"""
        content = f"{query}:{json.dumps(params, default=str)}"
        hash_key = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"{self.prefix}{hash_key}"

    async def get(self, query: str, params: tuple = ()) -> Optional[Any]:
        """Получить результат из кэша"""
        key = self._generate_key(query, params)

        cached = await self.redis.get(key)
        if cached:
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return json.loads(cached)

        return None

    async def set(
        self,
        query: str,
        params: tuple,
        result: Any,
        ttl: Optional[int] = None
    ):
        """Сохранить результат в кэш"""
        key = self._generate_key(query, params)

        # Serialize result
        serialized = json.dumps(result, default=str)

        # Compress if large
        if len(serialized) > self.compress_threshold:
            import gzip
            serialized = gzip.compress(serialized.encode())
            await self.redis.set(
                f"{key}:compressed",
                serialized,
                ex=ttl or self.default_ttl
            )
        else:
            await self.redis.set(
                key,
                serialized,
                ex=ttl or self.default_ttl
            )

        logger.debug(f"Cached query result: {query[:50]}...")

    async def invalidate_pattern(self, pattern: str):
        """Инвалидация кэша по паттерну"""
        keys = await self.redis.keys(f"{self.prefix}*{pattern}*")
        if keys:
            await self.redis.delete(*keys)
            logger.info(f"Invalidated {len(keys)} cache entries")

    async def invalidate_table(self, table: str):
        """Инвалидация кэша для таблицы"""
        await self.invalidate_pattern(table)


def cached_query(
    ttl: int = 300,
    key_pattern: Optional[str] = None,
    invalidate_on: Optional[List[str]] = None
):
    """
    Декоратор для кэширования результатов запросов.

    Usage:
        @cached_query(ttl=60, invalidate_on=["scans"])
        async def get_user_scans(user_id: int):
            return await db.fetch(...)
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get cache instance
            from utils.redis_client import get_redis
            redis = await get_redis()
            cache = QueryCache(redis)

            # Generate cache key
            if key_pattern:
                key = key_pattern.format(*args, **kwargs)
            else:
                key = f"{func.__name__}:{args}:{kwargs}"

            # Try cache first
            cached = await cache.redis.get(f"{cache.prefix}{key}")
            if cached:
                return json.loads(cached)

            # Execute function
            result = await func(*args, **kwargs)

            # Cache result
            serialized = json.dumps(result, default=str)
            await cache.redis.set(
                f"{cache.prefix}{key}",
                serialized,
                ex=ttl
            )

            return result

        # Add invalidation method
        async def invalidate(*args, **kwargs):
            from utils.redis_client import get_redis
            redis = await get_redis()
            cache = QueryCache(redis)

            if key_pattern:
                key = key_pattern.format(*args, **kwargs)
            else:
                key = f"{func.__name__}:{args}:{kwargs}"

            await cache.redis.delete(f"{cache.prefix}{key}")

        wrapper.invalidate = invalidate

        return wrapper

    return decorator


# Repository with caching
class CachedRepository:
    """Базовый класс для репозиториев с кэшированием"""

    def __init__(self, db, cache: QueryCache):
        self.db = db
        self.cache = cache

    async def _fetch_cached(
        self,
        query: str,
        params: tuple = (),
        ttl: int = 300
    ) -> List[dict]:
        """Выполнить запрос с кэшированием"""

        # Check cache
        cached = await self.cache.get(query, params)
        if cached is not None:
            return cached

        # Execute query
        result = await self.db.fetch(query, *params)
        result_list = [dict(r) for r in result]

        # Cache result
        await self.cache.set(query, params, result_list, ttl)

        return result_list

    async def _execute_and_invalidate(
        self,
        query: str,
        params: tuple = (),
        tables: List[str] = None
    ):
        """Выполнить запрос и инвалидировать кэш"""

        result = await self.db.execute(query, *params)

        # Invalidate related tables
        if tables:
            for table in tables:
                await self.cache.invalidate_table(table)

        return result
```

### Пример использования

```python
# api/repositories/scans.py
from utils.database.query_cache import CachedRepository, cached_query

class ScanRepository(CachedRepository):
    """Репозиторий для работы со сканированиями"""

    @cached_query(ttl=60, key_pattern="user_scans:{0}")
    async def get_user_scans(self, user_id: int, limit: int = 50) -> List[dict]:
        """Получить сканирования пользователя (кэшируется)"""
        return await self._fetch_cached(
            """
            SELECT id, scan_type, status, created_at, file_path
            FROM scans
            WHERE user_id = $1 AND deleted_at IS NULL
            ORDER BY created_at DESC
            LIMIT $2
            """,
            (user_id, limit),
            ttl=60
        )

    async def create_scan(self, user_id: int, scan_type: str, **kwargs) -> dict:
        """Создать новое сканирование"""
        result = await self._execute_and_invalidate(
            """
            INSERT INTO scans (user_id, scan_type, status, created_at)
            VALUES ($1, $2, 'pending', NOW())
            RETURNING *
            """,
            (user_id, scan_type),
            tables=["scans"]
        )
        return dict(result)

    async def update_scan_status(self, scan_id: int, status: str) -> dict:
        """Обновить статус сканирования"""
        result = await self._execute_and_invalidate(
            """
            UPDATE scans
            SET status = $2, updated_at = NOW()
            WHERE id = $1
            RETURNING *
            """,
            (scan_id, status),
            tables=["scans"]
        )
        return dict(result)
```

## 4. Connection Pooling

```python
# utils/database/pool.py
import asyncpg
from typing import Optional
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)

class DatabasePool:
    """Управление пулом соединений PostgreSQL"""

    _instance: Optional['DatabasePool'] = None
    _pool: Optional[asyncpg.Pool] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def initialize(
        self,
        dsn: str,
        min_size: int = 5,
        max_size: int = 20,
        max_queries: int = 50000,
        max_inactive_connection_lifetime: float = 300.0
    ):
        """Инициализация пула соединений"""
        if self._pool is not None:
            return

        self._pool = await asyncpg.create_pool(
            dsn,
            min_size=min_size,
            max_size=max_size,
            max_queries=max_queries,
            max_inactive_connection_lifetime=max_inactive_connection_lifetime,
            command_timeout=60.0,
            statement_cache_size=100,
        )

        logger.info(f"Database pool initialized: {min_size}-{max_size} connections")

    @asynccontextmanager
    async def connection(self):
        """Получить соединение из пула"""
        if self._pool is None:
            raise RuntimeError("Pool not initialized")

        async with self._pool.acquire() as conn:
            yield conn

    @asynccontextmanager
    async def transaction(self):
        """Выполнить операции в транзакции"""
        async with self.connection() as conn:
            async with conn.transaction():
                yield conn

    async def close(self):
        """Закрыть пул соединений"""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Database pool closed")


# Global instance
db_pool = DatabasePool()


# FastAPI dependency
async def get_db():
    """Dependency для получения соединения с БД"""
    async with db_pool.connection() as conn:
        yield conn
```

## 5. Monitoring Dashboard

```sql
-- Monitoring queries

-- Текущие активные запросы
SELECT
    pid,
    now() - pg_stat_activity.query_start AS duration,
    query,
    state,
    usename
FROM pg_stat_activity
WHERE (now() - pg_stat_activity.query_start) > interval '5 seconds'
AND state != 'idle'
ORDER BY duration DESC;

-- Размеры таблиц
SELECT
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
    n_live_tup as rows
FROM pg_stat_user_tables
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
LIMIT 10;

-- Проблемы с индексами
SELECT
    tablename,
    indexname,
    idx_scan as scans,
    pg_size_pretty(pg_relation_size(indexrelid)) as size
FROM pg_stat_user_indexes
WHERE idx_scan = 0
AND indexname NOT LIKE '%_pkey'
ORDER BY pg_relation_size(indexrelid) DESC;
```

## Best Practices Summary

| Практика | Описание |
|----------|----------|
| **EXPLAIN ANALYZE** | Всегда анализируйте запросы перед оптимизацией |
| **Composite Indexes** | Создавайте индексы для частых паттернов запросов |
| **Partial Indexes** | Используйте WHERE в индексах для часто фильтруемых данных |
| **Query Cache** | Кэшируйте результаты частых запросов в Redis |
| **Connection Pool** | Используйте пул соединений с лимитом |
| **Monitor** | Отслеживайте slow queries и неиспользуемые индексы |
| **Vacuum** | Настройте autovacuum для удаления dead tuples |
