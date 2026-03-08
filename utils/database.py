# -*- coding: utf-8 -*-
"""
Модуль базы данных для проекта Nanoprobe Simulation Lab
Хранение результатов сканирований, истории симуляций, метаданных
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager
import numpy as np


class DatabaseManager:
    """Менеджер базы данных SQLite."""

    def __init__(self, db_path: str = "data/nanoprobe.db"):
        """
        Инициализирует менеджер базы данных.

        Args:
            db_path: Путь к файлу базы данных
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    @contextmanager
    def get_connection(self):
        """Контекстный менеджер для подключения к БД."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _init_database(self):
        """Инициализация схемы базы данных."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Таблица результатов сканирований
            cursor.execute("""
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
            """)

            # Таблица симуляций
            cursor.execute("""
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
            """)

            # Таблица изображений
            cursor.execute("""
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
            """)

            # Таблица экспорта данных
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS exports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    export_path TEXT UNIQUE NOT NULL,
                    export_format TEXT NOT NULL,
                    source_type TEXT,
                    source_id INTEGER,
                    file_size_bytes INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Индексы для ускорения поиска
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_scan_timestamp 
                ON scan_results(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_scan_type 
                ON scan_results(scan_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_simulation_status 
                ON simulations(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_image_type 
                ON images(image_type)
            """)

    def add_scan_result(
        self,
        scan_type: str,
        surface_type: str = None,
        width: int = None,
        height: int = None,
        file_path: str = None,
        metadata: Dict = None
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
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO scan_results 
                (timestamp, scan_type, surface_type, width, height, file_path, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                scan_type,
                surface_type,
                width,
                height,
                file_path,
                json.dumps(metadata) if metadata else None
            ))
            return cursor.lastrowid

    def get_scan_results(
        self,
        scan_type: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """
        Получает результаты сканирований.

        Args:
            scan_type: Фильтр по типу сканирования
            limit: Лимит записей
            offset: Смещение

        Returns:
            Список записей
        """
        with self.get_connection() as conn:
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
            
            return [self._row_to_dict(row) for row in rows]

    def add_simulation(
        self,
        simulation_id: str,
        simulation_type: str,
        parameters: Dict = None
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
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO simulations 
                (simulation_id, simulation_type, start_time, status, parameters)
                VALUES (?, ?, ?, 'running', ?)
            """, (
                simulation_id,
                simulation_type,
                datetime.now().isoformat(),
                json.dumps(parameters) if parameters else None
            ))
            return cursor.lastrowid

    def update_simulation(
        self,
        simulation_id: str,
        status: str = None,
        results_summary: Dict = None
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
            
            if status in ('completed', 'failed', 'stopped'):
                updates.append("end_time = ?")
                params.append(datetime.now().isoformat())
                
                # Рассчитываем длительность
                cursor.execute(
                    "SELECT start_time FROM simulations WHERE simulation_id = ?",
                    (simulation_id,)
                )
                row = cursor.fetchone()
                if row and row['start_time']:
                    start = datetime.fromisoformat(row['start_time'])
                    duration = (datetime.now() - start).total_seconds()
                    updates.append("duration_seconds = ?")
                    params.append(duration)
            
            params.append(simulation_id)
            query = f"UPDATE simulations SET {', '.join(updates)} WHERE simulation_id = ?"
            cursor.execute(query, params)

    def get_simulations(
        self,
        status: str = None,
        limit: int = 50
    ) -> List[Dict]:
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
            
            query = "SELECT * FROM simulations"
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
        metadata: Dict = None
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
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO images 
                (image_path, image_type, source, width, height, channels, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                image_path,
                image_type,
                source,
                width,
                height,
                channels,
                json.dumps(metadata) if metadata else None
            ))
            return cursor.lastrowid

    def get_images(
        self,
        image_type: str = None,
        source: str = None,
        limit: int = 100
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
            
            query = "SELECT * FROM images"
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
        file_size_bytes: int = None
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
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO exports 
                (export_path, export_format, source_type, source_id, file_size_bytes)
                VALUES (?, ?, ?, ?, ?)
            """, (
                export_path,
                export_format,
                source_type,
                source_id,
                file_size_bytes
            ))
            return cursor.lastrowid

    def get_statistics(self) -> Dict[str, Any]:
        """
        Получает статистику базы данных.

        Returns:
            Словарь со статистикой
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Количество сканирований
            cursor.execute("SELECT COUNT(*) FROM scan_results")
            stats['total_scans'] = cursor.fetchone()[0]
            
            # Количество симуляций
            cursor.execute("SELECT COUNT(*) FROM simulations")
            stats['total_simulations'] = cursor.fetchone()[0]
            
            # Активные симуляции
            cursor.execute("SELECT COUNT(*) FROM simulations WHERE status = 'running'")
            stats['active_simulations'] = cursor.fetchone()[0]
            
            # Количество изображений
            cursor.execute("SELECT COUNT(*) FROM images")
            stats['total_images'] = cursor.fetchone()[0]
            
            # Количество экспортов
            cursor.execute("SELECT COUNT(*) FROM exports")
            stats['total_exports'] = cursor.fetchone()[0]
            
            # Сканирования по типам
            cursor.execute("""
                SELECT scan_type, COUNT(*) as count 
                FROM scan_results 
                GROUP BY scan_type
            """)
            stats['scans_by_type'] = {
                row['scan_type']: row['count'] 
                for row in cursor.fetchall()
            }
            
            return stats

    def search_scans(
        self,
        query: str,
        limit: int = 50
    ) -> List[Dict]:
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
            cursor.execute("""
                SELECT * FROM scan_results 
                WHERE surface_type LIKE ? OR metadata LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (search_pattern, search_pattern, limit))
            
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
            cursor.execute(
                "DELETE FROM scan_results WHERE id = ?",
                (scan_id,)
            )
            return cursor.rowcount > 0

    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        """Конвертирует строку результата в словарь."""
        result = dict(row)
        
        # Парсим JSON поля
        for key in ['metadata', 'parameters', 'results_summary']:
            if key in result and result[key]:
                try:
                    result[key] = json.loads(result[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        
        return result

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
                'export_timestamp': datetime.now().isoformat(),
                'scan_results': [],
                'simulations': [],
                'images': [],
                'exports': []
            }
            
            # Экспорт сканирований
            cursor.execute("SELECT * FROM scan_results")
            data['scan_results'] = [self._row_to_dict(row) for row in cursor.fetchall()]
            
            # Экспорт симуляций
            cursor.execute("SELECT * FROM simulations")
            data['simulations'] = [self._row_to_dict(row) for row in cursor.fetchall()]
            
            # Экспорт изображений
            cursor.execute("SELECT * FROM images")
            data['images'] = [self._row_to_dict(row) for row in cursor.fetchall()]
            
            # Экспорт экспортов
            cursor.execute("SELECT * FROM exports")
            data['exports'] = [self._row_to_dict(row) for row in cursor.fetchall()]
            
            # Сохранение
            output = Path(output_path)
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return output


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
