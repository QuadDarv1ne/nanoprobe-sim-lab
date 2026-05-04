"""Database operations for Nanoprobe Sim Lab."""

import json
from datetime import datetime, timezone
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
                (timestamp, simulation_type, parameters, results_summary, metrics, file_path, created_at)
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

    def invalidate_cache(self, pattern: str):
        """
        Invalidates cache entries matching the given pattern.

        Args:
            pattern: Cache key pattern (e.g., "scans:", "scan:123")
        """
        if not self.enable_cache:
            return

        keys_to_remove = [key for key in self._query_cache if key.startswith(pattern.rstrip(":"))]
        for key in keys_to_remove:
            del self._query_cache[key]

    def clear_cache(self):
        """Очистить весь кэш."""
        self._query_cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Получить статистику кэша."""
        return {
            "cache_size": len(self._query_cache),
            "cache_max_size": self._cache_max_size,
            "cache_ttl": self._cache_ttl,
        }
