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

            # Таблица сравнения изображений поверхностей
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS surface_comparisons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    comparison_id TEXT UNIQUE NOT NULL,
                    image1_path TEXT NOT NULL,
                    image2_path TEXT NOT NULL,
                    similarity_score REAL,
                    difference_map_path TEXT,
                    metrics TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Таблица AI/ML анализа дефектов
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS defect_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id TEXT UNIQUE NOT NULL,
                    image_path TEXT NOT NULL,
                    model_name TEXT,
                    defects_detected INTEGER DEFAULT 0,
                    defects_data TEXT,
                    confidence_score REAL,
                    processing_time_ms REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Таблица PDF отчётов
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pdf_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_path TEXT UNIQUE NOT NULL,
                    report_type TEXT NOT NULL,
                    title TEXT,
                    source_ids TEXT,
                    file_size_bytes INTEGER,
                    pages_count INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Таблица пакетной обработки
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS batch_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT UNIQUE NOT NULL,
                    job_type TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    total_items INTEGER DEFAULT 0,
                    processed_items INTEGER DEFAULT 0,
                    failed_items INTEGER DEFAULT 0,
                    parameters TEXT,
                    results_summary TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Таблица метрик производительности (для real-time визуализации)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT,
                    metadata TEXT
                )
            """)

            # Индексы для новых таблиц
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_comparison_timestamp
                ON surface_comparisons(created_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_defect_image
                ON defect_analysis(image_path)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_batch_status
                ON batch_jobs(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp
                ON performance_metrics(timestamp)
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

            # Новая статистика для расширенных функций
            # Сравнения поверхностей
            cursor.execute("SELECT COUNT(*) FROM surface_comparisons")
            stats['total_comparisons'] = cursor.fetchone()[0]

            # AI анализы дефектов
            cursor.execute("SELECT COUNT(*) FROM defect_analysis")
            stats['total_defect_analyses'] = cursor.fetchone()[0]

            # PDF отчёты
            cursor.execute("SELECT COUNT(*) FROM pdf_reports")
            stats['total_pdf_reports'] = cursor.fetchone()[0]

            # Пакетные задания
            cursor.execute("SELECT COUNT(*) FROM batch_jobs")
            stats['total_batch_jobs'] = cursor.fetchone()[0]

            # Активные пакетные задания
            cursor.execute("SELECT COUNT(*) FROM batch_jobs WHERE status = 'running'")
            stats['active_batch_jobs'] = cursor.fetchone()[0]

            # Метрики производительности
            cursor.execute("SELECT COUNT(*) FROM performance_metrics")
            stats['total_metrics'] = cursor.fetchone()[0]

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
        for key in ['metadata', 'parameters', 'results_summary', 'metrics', 'defects_data', 'source_ids']:
            if key in result and result[key]:
                try:
                    result[key] = json.loads(result[key])
                except (json.JSONDecodeError, TypeError):
                    pass

        return result

    # Методы для сравнения изображений поверхностей
    def add_surface_comparison(
        self,
        comparison_id: str,
        image1_path: str,
        image2_path: str,
        similarity_score: float,
        difference_map_path: str = None,
        metrics: Dict = None
    ) -> int:
        """
        Добавляет результат сравнения поверхностей.

        Args:
            comparison_id: ID сравнения
            image1_path: Путь к первому изображению
            image2_path: Путь ко второму изображению
            similarity_score: Оценка схожести (0-1)
            difference_map_path: Путь к карте различий
            metrics: Метрики сравнения

        Returns:
            ID записи
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO surface_comparisons
                (comparison_id, image1_path, image2_path, similarity_score, difference_map_path, metrics)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                comparison_id,
                image1_path,
                image2_path,
                similarity_score,
                difference_map_path,
                json.dumps(metrics) if metrics else None
            ))
            return cursor.lastrowid

    def get_surface_comparisons(self, limit: int = 50) -> List[Dict]:
        """Получает историю сравнений поверхностей."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM surface_comparisons
                ORDER BY created_at DESC LIMIT ?
            """, (limit,))
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    # Методы для AI/ML анализа дефектов
    def add_defect_analysis(
        self,
        analysis_id: str,
        image_path: str,
        model_name: str,
        defects_detected: int,
        defects_data: Dict = None,
        confidence_score: float = None,
        processing_time_ms: float = None
    ) -> int:
        """
        Добавляет результат AI анализа дефектов.

        Args:
            analysis_id: ID анализа
            image_path: Путь к изображению
            model_name: Название модели
            defects_detected: Количество обнаруженных дефектов
            defects_data: Детальные данные о дефектах
            confidence_score: Оценка достоверности
            processing_time_ms: Время обработки

        Returns:
            ID записи
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO defect_analysis
                (analysis_id, image_path, model_name, defects_detected, defects_data, confidence_score, processing_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                analysis_id,
                image_path,
                model_name,
                defects_detected,
                json.dumps(defects_data) if defects_data else None,
                confidence_score,
                processing_time_ms
            ))
            return cursor.lastrowid

    def get_defect_analyses(self, image_path: str = None, limit: int = 50) -> List[Dict]:
        """Получает историю AI анализов."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if image_path:
                cursor.execute("""
                    SELECT * FROM defect_analysis
                    WHERE image_path = ?
                    ORDER BY created_at DESC LIMIT ?
                """, (image_path, limit))
            else:
                cursor.execute("""
                    SELECT * FROM defect_analysis
                    ORDER BY created_at DESC LIMIT ?
                """, (limit,))
            
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    # Методы для PDF отчётов
    def add_pdf_report(
        self,
        report_path: str,
        report_type: str,
        title: str = None,
        source_ids: List[int] = None,
        file_size_bytes: int = None,
        pages_count: int = None
    ) -> int:
        """
        Добавляет запись о PDF отчёте.

        Args:
            report_path: Путь к файлу
            report_type: Тип отчёта
            title: Заголовок
            source_ids: ID исходных данных
            file_size_bytes: Размер файла
            pages_count: Количество страниц

        Returns:
            ID записи
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO pdf_reports
                (report_path, report_type, title, source_ids, file_size_bytes, pages_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                report_path,
                report_type,
                title,
                json.dumps(source_ids) if source_ids else None,
                file_size_bytes,
                pages_count
            ))
            return cursor.lastrowid

    def get_pdf_reports(self, report_type: str = None, limit: int = 50) -> List[Dict]:
        """Получает список PDF отчётов."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if report_type:
                cursor.execute("""
                    SELECT * FROM pdf_reports
                    WHERE report_type = ?
                    ORDER BY created_at DESC LIMIT ?
                """, (report_type, limit))
            else:
                cursor.execute("""
                    SELECT * FROM pdf_reports
                    ORDER BY created_at DESC LIMIT ?
                """, (limit,))
            
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    # Методы для пакетной обработки
    def add_batch_job(
        self,
        job_id: str,
        job_type: str,
        total_items: int = 0,
        parameters: Dict = None
    ) -> int:
        """
        Добавляет задание пакетной обработки.

        Args:
            job_id: ID задания
            job_type: Тип задания
            total_items: Всего элементов
            parameters: Параметры

        Returns:
            ID записи
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO batch_jobs
                (job_id, job_type, total_items, started_at, parameters)
                VALUES (?, ?, ?, ?, ?)
            """, (
                job_id,
                job_type,
                total_items,
                datetime.now().isoformat(),
                json.dumps(parameters) if parameters else None
            ))
            return cursor.lastrowid

    def update_batch_job(
        self,
        job_id: str,
        status: str = None,
        processed_items: int = None,
        failed_items: int = None,
        results_summary: Dict = None
    ):
        """Обновляет статус задания пакетной обработки."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            updates = []
            params = []
            
            if status:
                updates.append("status = ?")
                params.append(status)
                
                if status in ('completed', 'failed', 'cancelled'):
                    updates.append("completed_at = ?")
                    params.append(datetime.now().isoformat())
            
            if processed_items is not None:
                updates.append("processed_items = ?")
                params.append(processed_items)
            
            if failed_items is not None:
                updates.append("failed_items = ?")
                params.append(failed_items)
            
            if results_summary:
                updates.append("results_summary = ?")
                params.append(json.dumps(results_summary))
            
            params.append(job_id)
            query = f"UPDATE batch_jobs SET {', '.join(updates)} WHERE job_id = ?"
            cursor.execute(query, params)

    def get_batch_jobs(self, status: str = None, limit: int = 50) -> List[Dict]:
        """Получает список заданий пакетной обработки."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if status:
                cursor.execute("""
                    SELECT * FROM batch_jobs
                    WHERE status = ?
                    ORDER BY created_at DESC LIMIT ?
                """, (status, limit))
            else:
                cursor.execute("""
                    SELECT * FROM batch_jobs
                    ORDER BY created_at DESC LIMIT ?
                """, (limit,))
            
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    # Методы для real-time метрик производительности
    def add_performance_metric(
        self,
        metric_type: str,
        metric_name: str,
        value: float,
        unit: str = None,
        metadata: Dict = None
    ) -> int:
        """
        Добавляет метрику производительности.

        Args:
            metric_type: Тип метрики (spm, system, analysis)
            metric_name: Название метрики
            value: Значение
            unit: Единица измерения
            metadata: Дополнительные данные

        Returns:
            ID записи
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO performance_metrics
                (timestamp, metric_type, metric_name, value, unit, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                metric_type,
                metric_name,
                value,
                unit,
                json.dumps(metadata) if metadata else None
            ))
            return cursor.lastrowid

    def get_performance_metrics(
        self,
        metric_type: str = None,
        metric_name: str = None,
        start_time: str = None,
        end_time: str = None,
        limit: int = 1000
    ) -> List[Dict]:
        """Получает метрики производительности."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM performance_metrics WHERE 1=1"
            params = []
            
            if metric_type:
                query += " AND metric_type = ?"
                params.append(metric_type)
            
            if metric_name:
                query += " AND metric_name = ?"
                params.append(metric_name)
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def cleanup_old_metrics(self, days: int = 7) -> int:
        """
        Очищает старые метрики производительности.

        Args:
            days: Хранить метрики за последние N дней

        Returns:
            Количество удалённых записей
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cutoff = datetime.now()
            cutoff = cutoff.replace(day=cutoff.day - days)
            cursor.execute(
                "DELETE FROM performance_metrics WHERE timestamp < ?",
                (cutoff.isoformat(),)
            )
            return cursor.rowcount

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
