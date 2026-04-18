"""Database module — split from monolithic database.py."""

from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from utils.db.connection import ConnectionPool
from utils.db.operations import DatabaseOperations
from utils.db.schema import get_database_stats, init_database_schema


class DatabaseManager(DatabaseOperations):
    """Менеджер базы данных SQLite с connection pooling и query caching."""

    _pools: Dict[str, ConnectionPool] = {}
    _pool_lock = __import__("threading").Lock()

    def __init__(
        self, db_path: str = "data/nanoprobe.db", pool_size: int = 5, enable_cache: bool = True
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.pool_size = pool_size
        self.enable_cache = enable_cache
        self._query_cache: Dict[str, tuple] = {}
        self._cache_ttl = 60
        self._cache_max_size = 100
        self._init_pool()
        self._init_database()

    def _init_pool(self):
        db_path_str = str(self.db_path)
        with self._pool_lock:
            if db_path_str not in self._pools:
                self._pools[db_path_str] = ConnectionPool(db_path_str, self.pool_size)
            self._pool = self._pools[db_path_str]

    @classmethod
    def close_all_pools(cls):
        with cls._pool_lock:
            for pool in cls._pools.values():
                pool.close_all()
            cls._pools.clear()

    def close_pool(self):
        db_path_str = str(self.db_path)
        with self._pool_lock:
            if db_path_str in self._pools:
                self._pools[db_path_str].close_all()
                del self._pools[db_path_str]
            if hasattr(self, "_pool") and self._pool:
                self._pool.close_all()

    @contextmanager
    def get_connection(self):
        conn = self._pool.get_connection(timeout=30.0)
        try:
            yield conn
            conn.commit()
        except Exception as e:
            __import__("logging").getLogger(__name__).error(f"Database transaction error: {e}")
            conn.rollback()
            raise e
        finally:
            self._pool.return_connection(conn)

    @asynccontextmanager
    async def get_connection_async(self):
        import asyncio

        loop = asyncio.get_event_loop()
        conn = await loop.run_in_executor(None, self._pool.get_connection)
        try:
            yield conn
            await loop.run_in_executor(None, conn.commit)
        except Exception as e:
            __import__("logging").getLogger(__name__).error(
                f"Database async transaction error: {e}"
            )
            await loop.run_in_executor(None, conn.rollback)
            raise e
        finally:
            await loop.run_in_executor(None, self._pool.return_connection, conn)

    def get_pool_stats(self) -> Dict[str, Any]:
        return self._pool.get_stats()

    def optimize_database(self) -> Dict[str, Any]:
        results = {}
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("ANALYZE")
                results["analyze"] = "success"
            except Exception as e:
                results["analyze"] = f"error: {e}"
            try:
                cursor.execute("PRAGMA integrity_check")
                results["integrity"] = cursor.fetchone()[0]
            except Exception as e:
                results["integrity"] = f"error: {e}"
            cursor.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            cursor.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]
            size_before = page_count * page_size
            results["size_bytes_before"] = size_before
            try:
                cursor.execute("VACUUM")
                results["vacuum"] = "success"
            except Exception as e:
                results["vacuum"] = f"error: {e}"
            cursor.execute("PRAGMA page_count")
            size_after = cursor.fetchone()[0] * page_size
            results["size_bytes_after"] = size_after
            if size_before > 0:
                saved = size_before - size_after
                results["space_saved_bytes"] = max(0, saved)
                results["space_saved_percent"] = (
                    round((saved / size_before) * 100, 2) if size_before > 0 else 0
                )
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = {}
            for row in cursor.fetchall():
                table_name = row[0]
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    tables[table_name] = count
                except Exception as e:
                    __import__("logging").getLogger(__name__).warning(
                        f"Failed to count rows in {table_name}: {e}"
                    )
                    tables[table_name] = 0
            results["table_row_counts"] = tables
        return results

    def get_database_stats(self) -> Dict[str, Any]:
        with self.get_connection() as conn:
            return get_database_stats(conn)

    def _init_database(self):
        with self.get_connection() as conn:
            init_database_schema(conn)


# Глобальный экземпляр для удобства
_db_instance: Optional[DatabaseManager] = None


def get_database(db_path: str = "data/nanoprobe.db") -> DatabaseManager:
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager(db_path)
    return _db_instance
