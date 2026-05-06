"""Database operations for Nanoprobe Sim Lab."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schema import DBSchema


class DatabaseOperations:
    """
    Класс для выполнения операций с базой данных.

    Методы:
        - add_scan_result: Добавить результат сканирования
        - get_scan_results: Получить результаты сканирования
        - update_scan_result: Обновить результат сканирования
        - delete_scan_result: Удалить результат сканирования
    """

    def __init__(self, db_path: str, enable_cache: bool = True):
        """
        Инициализация подключения к базе данных.

        Args:
            db_path: Путь к файлу SQLite базы данных
            enable_cache: Включить кэширование запросов
        """
        self.db_path = Path(db_path)
        self.enable_cache = enable_cache
        self._query_cache: Dict[str, tuple] = {}
        self._cache_max_size = 1000
        self._cache_ttl = 300  # 5 минут

        # Инициализация схемы БД
        self.schema = DBSchema(str(self.db_path))
        self.schema.create_tables()

    def get_connection(self):
        """Получить подключение к базе данных."""
        import sqlite3

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _row_to_dict(self, row) -> Dict:
        """Конвертирует строку результата в словарь."""
        result = dict(row)
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

    # ==================== Scan Operations ====================

    def add_scan_result(
        self,
        scan_type: str,
        surface_type: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        file_path: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> int:
        """
        Добавить результат сканирования.

        Args:
            scan_type: Тип сканирования (spm, afm, sem, etc.)
            surface_type: Тип поверхности
            width: Ширина изображения в пикселях
            height: Высота изображения в пикселях
            file_path: Путь к файлу с данными
            metadata: Дополнительные метаданные

        Returns:
            ID добавленной записи
        """
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO scan_results
                (timestamp, scan_type, surface_type, width, height, file_path, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    now,
                    scan_type,
                    surface_type,
                    width,
                    height,
                    file_path,
                    json.dumps(metadata) if metadata else None,
                    now,
                ),
            )
            scan_id = cursor.lastrowid
            self.invalidate_cache("scans:")
            return scan_id

    def add_scan_result_batch(self, scan_results: List[Dict]) -> int:
        """
        Добавить несколько результатов сканирования.

        Args:
            scan_results: Список словарей с данными сканирования

        Returns:
            Количество добавленных записей
        """
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """INSERT INTO scan_results
                (timestamp, scan_type, surface_type, width, height, file_path, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        now,
                        r.get("scan_type"),
                        r.get("surface_type"),
                        r.get("width"),
                        r.get("height"),
                        r.get("file_path"),
                        json.dumps(r.get("metadata")) if r.get("metadata") else None,
                        now,
                    )
                    for r in scan_results
                ],
            )
            self.invalidate_cache("scans:")
            return cursor.rowcount

    def get_scan_results(
        self,
        scan_type: Optional[str] = None,
        surface_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict]:
        """
        Получить результаты сканирования.

        Args:
            scan_type: Фильтр по типу сканирования
            surface_type: Фильтр по типу поверхности
            limit: Максимальное количество записей
            offset: Смещение для пагинации

        Returns:
            Список результатов сканирования
        """
        cache_key = f"scans:{scan_type}:{surface_type}:{limit}:{offset}"
        if self.enable_cache and cache_key in self._query_cache:
            cached_time, result = self._query_cache[cache_key]
            if datetime.now(timezone.utc).timestamp() - cached_time < self._cache_ttl:
                return result

        query = "SELECT * FROM scan_results WHERE 1=1"
        params = []

        if scan_type:
            query += " AND scan_type = ?"
            params.append(scan_type)
        if surface_type:
            query += " AND surface_type = ?"
            params.append(surface_type)

        query += f" ORDER BY timestamp DESC LIMIT {limit} OFFSET {offset}"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            results = [self._row_to_dict(row) for row in cursor.fetchall()]

        if self.enable_cache:
            self._query_cache[cache_key] = (
                datetime.now(timezone.utc).timestamp(),
                results,
            )

        return results

    def get_scan_result(self, scan_id: int) -> Optional[Dict]:
        """
        Получить результат сканирования по ID.

        Args:
            scan_id: ID результата сканирования

        Returns:
            Результат сканирования или None
        """
        cache_key = f"scan:{scan_id}"
        if self.enable_cache and cache_key in self._query_cache:
            cached_time, result = self._query_cache[cache_key]
            if datetime.now(timezone.utc).timestamp() - cached_time < self._cache_ttl:
                return result

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM scan_results WHERE id = ?", (scan_id,))
            row = cursor.fetchone()

        if row:
            result = self._row_to_dict(row)
            if self.enable_cache:
                self._query_cache[cache_key] = (
                    datetime.now(timezone.utc).timestamp(),
                    result,
                )
            return result
        return None

    def update_scan_result(
        self,
        scan_id: int,
        surface_type: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        file_path: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Обновить результат сканирования.

        Args:
            scan_id: ID результата сканирования
            surface_type: Новый тип поверхности
            width: Новая ширина
            height: Новая высота
            file_path: Новый путь к файлу
            metadata: Новые метаданные

        Returns:
            True если запись обновлена, False если не найдена
        """
        updates = []
        params = []

        if surface_type is not None:
            updates.append("surface_type = ?")
            params.append(surface_type)
        if width is not None:
            updates.append("width = ?")
            params.append(width)
        if height is not None:
            updates.append("height = ?")
            params.append(height)
        if file_path is not None:
            updates.append("file_path = ?")
            params.append(file_path)
        if metadata is not None:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))

        if not updates:
            return False

        params.append(scan_id)
        query = f"UPDATE scan_results SET {', '.join(updates)} WHERE id = ?"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            self.invalidate_cache(f"scan:{scan_id}")
            return cursor.rowcount > 0

    def delete_scan_result(self, scan_id: int) -> bool:
        """
        Удалить результат сканирования.

        Args:
            scan_id: ID результата сканирования

        Returns:
            True если запись удалена, False если не найдена
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM scan_results WHERE id = ?", (scan_id,))
            self.invalidate_cache(f"scan:{scan_id}")
            return cursor.rowcount > 0

    # ==================== Simulation Operations ====================

    def add_simulation_result(
        self,
        simulation_type: str,
        parameters: Dict,
        results_summary: Dict,
        metrics: Optional[Dict] = None,
        file_path: Optional[str] = None,
    ) -> int:
        """
        Добавить результат симуляции.

        Args:
            simulation_type: Тип симуляции
            parameters: Параметры симуляции
            results_summary: Краткое описание результатов
            metrics: Метрики качества
            file_path: Путь к файлу с результатами

        Returns:
            ID добавленной записи
        """
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO simulation_results
                (timestamp, simulation_type, parameters, results_summary,
                 metrics, file_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    now,
                    simulation_type,
                    json.dumps(parameters),
                    json.dumps(results_summary),
                    json.dumps(metrics) if metrics else None,
                    file_path,
                    now,
                ),
            )
            sim_id = cursor.lastrowid
            self.invalidate_cache("simulations:")
            return sim_id

    def get_simulation_results(
        self, simulation_type: Optional[str] = None, limit: int = 100
    ) -> List[Dict]:
        """Получить результаты симуляций."""
        cache_key = f"simulations:{simulation_type}:{limit}"
        if self.enable_cache and cache_key in self._query_cache:
            cached_time, result = self._query_cache[cache_key]
            if datetime.now(timezone.utc).timestamp() - cached_time < self._cache_ttl:
                return result

        query = "SELECT * FROM simulation_results WHERE 1=1"
        params = []

        if simulation_type:
            query += " AND simulation_type = ?"
            params.append(simulation_type)

        query += f" ORDER BY timestamp DESC LIMIT {limit}"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            results = [self._row_to_dict(row) for row in cursor.fetchall()]

        if self.enable_cache:
            self._query_cache[cache_key] = (
                datetime.now(timezone.utc).timestamp(),
                results,
            )

        return results

    # ==================== Cache Management ====================

    def _cache_result(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Cache a result with optional TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
        """
        if not self.enable_cache:
            return

        ttl = ttl or self._cache_ttl
        self._query_cache[key] = (
            datetime.now(timezone.utc).timestamp(),
            value,
        )
        # Enforce max cache size
        if len(self._query_cache) > self._cache_max_size:
            # Remove oldest entries
            sorted_keys = sorted(self._query_cache.keys(), key=lambda k: self._query_cache[k][0])
            for key_to_remove in sorted_keys[: len(self._query_cache) - self._cache_max_size]:
                del self._query_cache[key_to_remove]

    def _get_from_cache(self, key: str) -> Any:
        """
        Get a value from cache if it exists and is not expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if not self.enable_cache:
            return None

        if key in self._query_cache:
            cached_time, value = self._query_cache[key]
            if datetime.now(timezone.utc).timestamp() - cached_time < self._cache_ttl:
                return value
            else:
                # Remove expired entry
                del self._query_cache[key]
        return None

    def invalidate_cache(self, pattern: str = ""):
        """
        Invalidates cache entries matching the given pattern.
        If pattern is empty, invalidates all cache.

        Args:
            pattern: Cache key pattern (e.g., "scans:", "scan:123")
        """
        if not self.enable_cache:
            return

        if pattern == "":
            self.clear_cache()
            return

        keys_to_remove = [key for key in self._query_cache if key.startswith(pattern.rstrip(":"))]
        for key in keys_to_remove:
            del self._query_cache[key]

    def clear_cache(self):
        """Очистить весь кэш."""
        self._query_cache.clear()

    def set_cache_ttl(self, ttl: int):
        """
        Set cache TTL.

        Args:
            ttl: Time to live in seconds
        """
        self._cache_ttl = ttl

    def get_cache_stats(self) -> Dict[str, Any]:
        """Получить статистику кэша."""
        return {
            "cache_size": len(self._query_cache),
            "cache_max_size": self._cache_max_size,
            "cache_ttl": self._cache_ttl,
        }

    # ==================== Performance Metrics Operations ====================

    def add_performance_metric(
        self,
        metric_type: str,
        metric_name: str,
        value: float,
        unit: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> int:
        """
        Добавить метрику производительности.

        Args:
            metric_type: Тип метрики (api, system, etc.)
            metric_name: Название метрики
            value: Значение метрики
            unit: Единица измерения
            metadata: Дополнительные метаданные

        Returns:
            ID добавленной записи
        """
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO performance_metrics
                (timestamp, metric_type, metric_name, value, unit, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    now,
                    metric_type,
                    metric_name,
                    value,
                    unit,
                    json.dumps(metadata) if metadata else None,
                    now,
                ),
            )
            metric_id = cursor.lastrowid
            self.invalidate_cache("metrics:")
            return metric_id

    def get_performance_metrics(
        self,
        metric_type: Optional[str] = None,
        metric_name: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict]:
        """
        Получить метрики производительности.

        Args:
            metric_type: Фильтр по типу метрики
            metric_name: Фильтр по названию метрики
            limit: Максимальное количество записей
            offset: Смещение для пагинации

        Returns:
            Список метрик производительности
        """
        cache_key = f"metrics:{metric_type}:{metric_name}:{limit}:{offset}"
        if self.enable_cache and cache_key in self._query_cache:
            cached_time, result = self._query_cache[cache_key]
            if datetime.now(timezone.utc).timestamp() - cached_time < self._cache_ttl:
                return result

        query = "SELECT * FROM performance_metrics WHERE 1=1"
        params = []

        if metric_type:
            query += " AND metric_type = ?"
            params.append(metric_type)
        if metric_name:
            query += " AND metric_name = ?"
            params.append(metric_name)

        query += f" ORDER BY timestamp DESC LIMIT {limit} OFFSET {offset}"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            results = [self._row_to_dict(row) for row in cursor.fetchall()]

        if self.enable_cache:
            self._query_cache[cache_key] = (
                datetime.now(timezone.utc).timestamp(),
                results,
            )

        return results

    def cleanup_old_metrics(self, days: int = 30) -> int:
        """
        Очистить старые метрики производительности.

        Args:
            days: Количество дней для хранения метрик

        Returns:
            Количество удаленных записей
        """
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM performance_metrics WHERE timestamp < ?", (cutoff_date,))
            self.invalidate_cache("metrics:")
            return cursor.rowcount

    # ==================== User Operations ====================

    def upsert_user(
        self,
        username: str,
        password_hash: str,
        role: str = "user",
        email: Optional[str] = None,
        full_name: Optional[str] = None,
        is_active: bool = True,
    ) -> int:
        """
        Вставить или обновить пользователя.

        Args:
            username: Имя пользователя
            password_hash: Хеш пароля
            role: Роль пользователя
            email: Email адрес
            full_name: Полное имя
            is_active: Активен ли пользователь

        Returns:
            ID пользователя
        """
        # Check if users table exists, if not we'll need to create it
        # For now, we'll assume it exists based on the test expectations
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Try to update first
            cursor.execute(
                """UPDATE users
                SET password_hash = ?, role = ?, email = ?, full_name = ?,
                    is_active = ?, updated_at = ?
                WHERE username = ?""",
                (
                    password_hash,
                    role,
                    email,
                    full_name,
                    is_active,
                    now,
                    username,
                ),
            )

            if cursor.rowcount == 0:
                # Insert new user
                cursor.execute(
                    """INSERT INTO users
                    (username, password_hash, role, email, full_name, is_active,
                     created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        username,
                        password_hash,
                        role,
                        email,
                        full_name,
                        is_active,
                        now,
                        now,
                    ),
                )
                user_id = cursor.lastrowid
            else:
                # Get the ID of the updated user
                cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
                row = cursor.fetchone()
                user_id = row["id"] if row else None

            self.invalidate_cache("users:")
            return user_id

    def update_last_login(self, username: str) -> None:
        """
        Обновить время последнего входа пользователя.

        Args:
            username: Имя пользователя
        """
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE users
                SET last_login = ?, updated_at = ?
                WHERE username = ?""",
                (now, now, username),
            )
            self.invalidate_cache(f"user:{username}")

    def update_password_hash(self, username: str, new_hash: str) -> bool:
        """
        Обновить хеш пароля пользователя.

        Args:
            username: Имя пользователя
            new_hash: Новый хеш пароля

        Returns:
            True если обновлено, False если пользователь не найден
        """
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE users
                SET password_hash = ?, updated_at = ?
                WHERE username = ?""",
                (new_hash, now, username),
            )
            self.invalidate_cache(f"user:{username}")
            return cursor.rowcount > 0

    def get_user(self, username: str) -> Optional[Dict]:
        """
        Получить пользователя по имени.

        Args:
            username: Имя пользователя

        Returns:
            Данные пользователя или None если не найден
        """
        cache_key = f"user:{username}"
        if self.enable_cache and cache_key in self._query_cache:
            cached_time, result = self._query_cache[cache_key]
            if datetime.now(timezone.utc).timestamp() - cached_time < self._cache_ttl:
                return result

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            row = cursor.fetchone()

        if row:
            result = self._row_to_dict(row)
            if self.enable_cache:
                self._query_cache[cache_key] = (
                    datetime.now(timezone.utc).timestamp(),
                    result,
                )
            return result
        return None

    # ==================== Export Operations ====================

    def export_to_json(self, file_path: str) -> bool:
        """
        Экспортировать данные в JSON файл.

        Args:
            file_path: Путь к файлу для сохранения

        Returns:
            True если экспорт успешен
        """
        import json
        from pathlib import Path

        try:
            data = {
                "scan_results": self.get_scan_results(limit=10000),  # Get all scans
                "simulation_results": self.get_simulation_results(
                    limit=10000
                ),  # Get all simulations
                "performance_metrics": self.get_performance_metrics(limit=10000),  # Get all metrics
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return True
        except Exception:
            # Log the error in a real implementation
            return False
