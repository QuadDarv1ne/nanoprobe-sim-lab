"""Connection pooling for SQLite databases."""

import asyncio
import logging
import sqlite3
import threading
from queue import Empty, Queue
from typing import Dict

logger = logging.getLogger(__name__)


class ConnectionPool:
    """Пул соединений для SQLite"""

    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self._pool: Queue = Queue(maxsize=pool_size)
        self._lock = threading.Lock()
        self._created = 0
        self._stats = {"hits": 0, "misses": 0, "created": 0}

    def get_connection(self, timeout: float = 30.0) -> sqlite3.Connection:
        try:
            conn = self._pool.get_nowait()
            self._stats["hits"] += 1
            return conn
        except Empty:
            with self._lock:
                if self._created < self.pool_size:
                    self._created += 1
                    self._stats["created"] += 1
                    return self._create_connection()
            self._stats["misses"] += 1
            return self._pool.get(timeout=timeout)

    def return_connection(self, conn: sqlite3.Connection):
        try:
            self._pool.put_nowait(conn)
        except Exception as e:
            logger.debug(f"Connection pool full, closing connection: {e}")
            conn.close()

    def _create_connection(self) -> sqlite3.Connection:
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
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Empty:
                break
            except sqlite3.Error as e:
                logger.warning("Error closing connection: %s", e)

    def get_stats(self) -> Dict[str, int]:
        return {
            **self._stats,
            "pool_size": self.pool_size,
            "current_size": self._pool.qsize(),
        }


class AsyncConnectionPool:
    """Асинхронный пул соединений для SQLite"""

    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self._pool: Queue = Queue(maxsize=pool_size)
        self._lock = threading.Lock()
        self._created = 0

    async def get_connection(self, timeout: float = 30.0) -> sqlite3.Connection:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_connection_sync, timeout)

    def _get_connection_sync(self, timeout: float) -> sqlite3.Connection:
        try:
            conn = self._pool.get_nowait()
            return conn
        except Empty:
            with self._lock:
                if self._created < self.pool_size:
                    self._created += 1
                    return self._create_connection()
            return self._pool.get(timeout=timeout)

    def return_connection(self, conn: sqlite3.Connection):
        try:
            self._pool.put_nowait(conn)
        except Exception as e:
            logger.debug(f"Async connection pool full, closing connection: {e}")
            conn.close()

    def _create_connection(self) -> sqlite3.Connection:
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
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Empty:
                break
            except sqlite3.Error as e:
                logger.warning("Error closing async connection: %s", e)
