# -*- coding: utf-8 -*-
"""
Модуль пакетной обработки данных для проекта Nanoprobe Simulation Lab
Автоматизация обработки множественных файлов и заданий
"""

import json
import time
import asyncio
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Callable, Optional, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue, Empty
import traceback
from dataclasses import dataclass, field
from enum import Enum

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class JobStatus(Enum):
    """Статусы задания"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class BatchJobStats:
    """Статистика задания"""
    job_id: str
    job_type: str
    status: str
    total_items: int
    processed_items: int
    failed_items: int
    progress_percent: float
    success_rate: float
    started_at: Optional[str]
    completed_at: Optional[str]
    duration_seconds: Optional[float]
    priority: int
    errors_count: int


class BatchJob:
    """Класс отдельного задания пакетной обработки"""

    def __init__(
        self,
        job_id: str,
        job_type: str,
        items: List[Any],
        processor: Callable,
        parameters: Dict = None,
        priority: int = 0,
        callback: Optional[Callable] = None
    ):
        """
        Инициализация задания

        Args:
            job_id: ID задания
            job_type: Тип задания
            items: Список элементов для обработки
            processor: Функция обработки
            parameters: Параметры обработки
            priority: Приоритет (чем выше, тем важнее)
            callback: Callback для обновления прогресса
        """
        self.job_id = job_id
        self.job_type = job_type
        self.items = items
        self.processor = processor
        self.parameters = parameters or {}
        self.priority = priority
        self.callback = callback

        self.status = 'pending'
        self.total_items = len(items)
        self.processed_items = 0
        self.failed_items = 0
        self.results = []
        self.errors = []
        self.started_at = None
        self.completed_at = None
        self.progress_callback = callback

    def set_progress_callback(self, callback: Callable):
        """Установка callback для обновления прогресса"""
        self.progress_callback = callback
        self.callback = callback

    def update_progress(self, processed: int, success: bool = True):
        """Обновление прогресса"""
        self.processed_items = processed
        if not success:
            self.failed_items += 1
        if self.progress_callback:
            self.progress_callback(self.job_id, self.progress_percent)
        if self.callback:
            self.callback(self.job_id, {
                'processed': processed,
                'total': self.total_items,
                'percent': self.progress_percent,
                'status': self.status
            })

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            'job_id': self.job_id,
            'job_type': self.job_type,
            'status': self.status,
            'total_items': self.total_items,
            'processed_items': self.processed_items,
            'failed_items': self.failed_items,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'progress': self.progress_percent,
            'priority': self.priority,
            'success_rate': self.success_rate,
        }

    @property
    def progress_percent(self) -> float:
        """Процент выполнения"""
        if self.total_items == 0:
            return 0
        return 100 * self.processed_items / self.total_items

    @property
    def success_rate(self) -> float:
        """Процент успешных операций"""
        if self.processed_items == 0:
            return 0
        return 100 * (self.processed_items - self.failed_items) / self.processed_items


class BatchProcessor:
    """
    Менеджер пакетной обработки
    Управление очередями, выполнение заданий, отслеживание прогресса
    """

    def __init__(
        self,
        max_workers: int = 4,
        output_dir: str = "output/batch",
        db_manager=None
    ):
        """
        Инициализация процессора

        Args:
            max_workers: Максимальное количество потоков
            output_dir: Директория для результатов
            db_manager: Менеджер базы данных
        """
        self.max_workers = max_workers
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_manager = db_manager

        self.jobs: Dict[str, BatchJob] = {}
        self.job_queue: Queue = Queue()
        self.active_jobs: set = set()
        self.lock = threading.Lock()

        # Статистика
        self.total_jobs_completed = 0
        self.total_items_processed = 0

    def create_job(
        self,
        job_type: str,
        items: List[Any],
        processor: Callable,
        parameters: Dict = None,
        priority: int = 0
    ) -> str:
        """
        Создание нового задания

        Args:
            job_type: Тип задания
            items: Элементы для обработки
            processor: Функция обработки
            parameters: Параметры
            priority: Приоритет (чем выше, тем важнее)

        Returns:
            ID созданного задания
        """
        job_id = f"batch_{job_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        job = BatchJob(job_id, job_type, items, processor, parameters, priority, callback)

        with self.lock:
            self.jobs[job_id] = job
            self.job_queue.put(job)

        # Сохранение в БД
        if self.db_manager:
            self.db_manager.add_batch_job(
                job_id=job_id,
                job_type=job_type,
                total_items=len(items),
                parameters=parameters
            )

        return job_id

    def process_sstv_batch(
        self,
        audio_files: List[str],
        output_dir: str = "output/sstv_batch",
        parameters: Dict = None,
        callback: Callable = None
    ) -> str:
        """
        Пакетное декодирование SSTV файлов

        Args:
            audio_files: Пути к аудио файлам
            output_dir: Директория для результатов
            parameters: Параметры
            callback: Callback для прогресса

        Returns:
            ID задания
        """
        from components.py_sstv_groundstation.src.sstv_decoder import SSTVDecoder
        
        def decode_sstv_file(path: str) -> Dict:
            """Декодирование одного SSTV файла"""
            result = {
                'path': path,
                'name': Path(path).name,
                'status': 'success',
                'image_path': None,
                'mode': None,
                'error': None
            }
            
            try:
                decoder = SSTVDecoder()
                image = decoder.decode_from_audio(path)
                
                if image:
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    output_path = f"{output_dir}/sstv_{Path(path).stem}_{timestamp}.png"
                    image.save(output_path)
                    
                    result['image_path'] = output_path
                    result['mode'] = decoder.metadata.get('mode', 'unknown')
                else:
                    result['status'] = 'failed'
                    result['error'] = 'Decoding failed'
                    
            except Exception as e:
                result['status'] = 'failed'
                result['error'] = str(e)
            
            return result
        
        return self.create_job(
            job_type='sstv_decode',
            items=audio_files,
            processor=decode_sstv_file,
            parameters={'output_dir': output_dir, **(parameters or {})},
            callback=callback
        )

    def process_satellite_passes(
        self,
        satellite_names: List[str],
        hours_ahead: int = 24,
        ground_station_lat: float = 55.75,
        ground_station_lon: float = 37.61,
        callback: Callable = None
    ) -> str:
        """
        Пакетный расчёт пролётов спутников

        Args:
            satellite_names: Названия спутников
            hours_ahead: На сколько часов вперёд
            ground_station_lat: Широта наземной станции
            ground_station_lon: Долгота наземной станции
            callback: Callback для прогресса

        Returns:
            ID задания
        """
        from components.py_sstv_groundstation.src.satellite_tracker import SatelliteTracker
        
        def calculate_passes(sat_name: str) -> Dict:
            """Расчёт пролётов для одного спутника"""
            result = {
                'satellite': sat_name,
                'status': 'success',
                'passes': [],
                'error': None
            }
            
            try:
                tracker = SatelliteTracker(
                    ground_station_lat=ground_station_lat,
                    ground_station_lon=ground_station_lon
                )
                passes = tracker.get_pass_predictions(
                    sat_name,
                    hours_ahead=hours_ahead
                )
                result['passes'] = passes
                
            except Exception as e:
                result['status'] = 'failed'
                result['error'] = str(e)
            
            return result
        
        return self.create_job(
            job_type='satellite_passes',
            items=satellite_names,
            processor=calculate_passes,
            parameters={
                'hours_ahead': hours_ahead,
                'ground_station_lat': ground_station_lat,
                'ground_station_lon': ground_station_lon
            },
            callback=callback
        )

    def process_image_batch(
        self,
        image_paths: List[str],
        operation: str = 'analyze',
        parameters: Dict = None
    ) -> str:
        """
        Пакетная обработка изображений

        Args:
            image_paths: Пути к изображениям
            operation: Тип операции
            parameters: Параметры

        Returns:
            ID задания
        """
        def process_single_image(path: str) -> Dict:
            """Обработка одного изображения"""
            result = {
                'path': path,
                'name': Path(path).name,
                'status': 'success',
                'data': None,
                'error': None
            }

            try:
                if operation == 'analyze':
                    # Анализ изображения
                    img = Image.open(path)
                    result['data'] = {
                        'size': img.size,
                        'mode': img.mode,
                        'format': img.format,
                        'width': img.width,
                        'height': img.height,
                    }
                elif operation == 'resize':
                    # Изменение размера
                    size = parameters.get('size', (256, 256))
                    img = Image.open(path)
                    resized = img.resize(size, Image.Resampling.LANCZOS)
                    output_path = self.output_dir / f"resized_{Path(path).name}"
                    resized.save(output_path)
                    result['data'] = {'output_path': str(output_path)}
                elif operation == 'convert':
                    # Конвертация формата
                    format_out = parameters.get('format', 'PNG')
                    img = Image.open(path)
                    output_path = self.output_dir / f"{Path(path).stem}.{format_out.lower()}"
                    img.save(output_path, format=format_out)
                    result['data'] = {'output_path': str(output_path)}
                else:
                    result['error'] = f"Неизвестная операция: {operation}"
                    result['status'] = 'error'

            except Exception as e:
                result['error'] = str(e)
                result['status'] = 'error'

            return result

        return self.create_job(
            job_type=f'image_{operation}',
            items=image_paths,
            processor=process_single_image,
            parameters=parameters
        )

    def process_surface_analysis_batch(
        self,
        surface_data_list: List[Dict[str, Any]],
        analysis_type: str = 'statistics'
    ) -> str:
        """
        Пакетный анализ поверхностей

        Args:
            surface_data_list: Список данных поверхностей
            analysis_type: Тип анализа

        Returns:
            ID задания
        """
        def analyze_surface(data: Dict) -> Dict:
            """Анализ одной поверхности"""
            import numpy as np

            result = {
                'id': data.get('id', 'unknown'),
                'status': 'success',
                'data': None,
                'error': None
            }

            try:
                surface = data.get('surface', np.zeros((100, 100)))

                if analysis_type == 'statistics':
                    result['data'] = {
                        'mean': float(np.mean(surface)),
                        'std': float(np.std(surface)),
                        'min': float(np.min(surface)),
                        'max': float(np.max(surface)),
                        'rms': float(np.sqrt(np.mean(surface**2))),
                    }
                elif analysis_type == 'full':
                    result['data'] = {
                        'mean': float(np.mean(surface)),
                        'std': float(np.std(surface)),
                        'min': float(np.min(surface)),
                        'max': float(np.max(surface)),
                        'rms': float(np.sqrt(np.mean(surface**2))),
                        'skewness': float(self._calculate_skewness(surface)),
                        'kurtosis': float(self._calculate_kurtosis(surface)),
                    }

            except Exception as e:
                result['error'] = str(e)
                result['status'] = 'error'

            return result

        return self.create_job(
            job_type=f'surface_{analysis_type}',
            items=surface_data_list,
            processor=analyze_surface
        )

    def _calculate_skewness(self, data) -> float:
        """Расчёт асимметрии"""
        import numpy as np
        from scipy import stats
        return float(stats.skew(data.flatten()))

    def _calculate_kurtosis(self, data) -> float:
        """Расчёт эксцесса"""
        import numpy as np
        from scipy import stats
        return float(stats.kurtosis(data.flatten()))

    def run_job(self, job_id: str, parallel: bool = True) -> Dict[str, Any]:
        """
        Выполнение задания

        Args:
            job_id: ID задания
            parallel: Использовать параллельное выполнение

        Returns:
            Результаты выполнения
        """
        with self.lock:
            if job_id not in self.jobs:
                return {'error': f'Задание {job_id} не найдено'}

            job = self.jobs[job_id]

            if job.status == 'running':
                return {'error': 'Задание уже выполняется'}

            job.status = 'running'
            job.started_at = datetime.now()
            self.active_jobs.add(job_id)

        # Обновление в БД
        if self.db_manager:
            self.db_manager.update_batch_job(job_id, status='running')

        try:
            if parallel and len(job.items) > 1:
                results = self._run_parallel(job)
            else:
                results = self._run_sequential(job)

            job.results = results
            job.status = 'completed'
            job.completed_at = datetime.now()

            self.total_jobs_completed += 1
            self.total_items_processed += job.processed_items

            # Обновление в БД
            if self.db_manager:
                self.db_manager.update_batch_job(
                    job_id,
                    status='completed',
                    processed_items=job.processed_items,
                    failed_items=job.failed_items,
                    results_summary={'results_count': len(results)}
                )

            # Сохранение результатов
            self._save_job_results(job)

        except Exception as e:
            job.status = 'failed'
            job.completed_at = datetime.now()
            job.errors.append(str(e))

            if self.db_manager:
                self.db_manager.update_batch_job(job_id, status='failed')

        finally:
            with self.lock:
                self.active_jobs.discard(job_id)

        return job.to_dict()

    def _run_sequential(self, job: BatchJob) -> List[Dict]:
        """Последовательное выполнение"""
        results = []

        for item in job.items:
            try:
                result = job.processor(item)
                results.append(result)
                job.processed_items += 1
            except Exception as e:
                job.failed_items += 1
                results.append({'error': str(e), 'item': str(item)})

        return results

    def _run_parallel(self, job: BatchJob) -> List[Dict]:
        """Параллельное выполнение"""
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_item = {executor.submit(job.processor, item): item for item in job.items}

            for future in as_completed(future_to_item):
                try:
                    result = future.result()
                    results.append(result)
                    job.processed_items += 1
                except Exception as e:
                    job.failed_items += 1
                    results.append({'error': str(e)})

        return results

    def run_all_pending(self, parallel: bool = True):
        """Запуск всех ожидающих заданий"""
        with self.lock:
            pending_jobs = [
                job_id for job_id, job in self.jobs.items()
                if job.status == 'pending' and job_id not in self.active_jobs
            ]

        # Сортировка по приоритету
        pending_jobs.sort(key=lambda jid: self.jobs[jid].priority, reverse=True)

        for job_id in pending_jobs:
            self.run_job(job_id, parallel)

    def get_enhanced_report(self, job_id: str) -> Dict[str, Any]:
        """
        Получение расширенного отчёта по заданию

        Args:
            job_id: ID задания

        Returns:
            Расширенный отчёт
        """
        job = self.jobs.get(job_id)
        if not job:
            return {'error': 'Job not found'}

        # Детальная статистика
        error_types = {}
        for err in job.errors:
            error_type = type(err).__name__ if isinstance(err, Exception) else 'Unknown'
            error_types[error_type] = error_types.get(error_type, 0) + 1

        # Распределение результатов по времени
        processing_time = None
        if job.started_at and job.completed_at:
            processing_time = (job.completed_at - job.started_at).total_seconds()

        return {
            **job.to_dict(),
            'detailed_stats': {
                'processing_time_sec': processing_time,
                'avg_time_per_item': processing_time / job.total_items if processing_time and job.total_items > 0 else None,
                'error_types': error_types,
                'errors_list': job.errors[:10],  # Первые 10 ошибок
            },
            'sample_results': job.results[:5],  # Первые 5 результатов
            'recommendations': self._generate_job_recommendations(job),
        }

    def _generate_job_recommendations(self, job: BatchJob) -> List[str]:
        """Генерация рекомендаций по заданию"""
        recommendations = []

        if job.failed_items > 0:
            fail_rate = job.failed_items / job.total_items * 100
            if fail_rate > 20:
                recommendations.append(f"Высокий процент ошибок ({fail_rate:.1f}%) - проверьте входные данные")
            elif fail_rate > 5:
                recommendations.append(f"Замечены ошибки ({fail_rate:.1f}%) - рекомендуется анализ логов")

        if job.progress_percent < 100 and job.status == 'running':
            recommendations.append("Задание выполняется - ожидайте завершения")

        if job.priority < 5 and job.total_items > 100:
            recommendations.append("Для больших заданий рассмотрите увеличение приоритета")

        if not recommendations:
            recommendations.append("Задание выполнено успешно - рекомендаций нет")

        return recommendations

    def get_queue_summary(self) -> Dict[str, Any]:
        """
        Получение сводки по очереди заданий

        Returns:
            Сводка по очереди
        """
        with self.lock:
            jobs_by_status = {}
            jobs_by_priority = {}
            total_items = 0

            for job in self.jobs.values():
                status = job.status
                priority = job.priority

                jobs_by_status[status] = jobs_by_status.get(status, 0) + 1
                jobs_by_priority[priority] = jobs_by_priority.get(priority, 0) + 1
                total_items += job.total_items

            return {
                'total_jobs': len(self.jobs),
                'jobs_by_status': jobs_by_status,
                'jobs_by_priority': jobs_by_priority,
                'total_items': total_items,
                'active_jobs': len(self.active_jobs),
                'queue_size': self.job_queue.qsize(),
            }

    def export_job_report(self, job_id: str, output_path: str = None) -> str:
        """
        Экспорт отчёта по заданию в JSON

        Args:
            job_id: ID задания
            output_path: Путь для сохранения

        Returns:
            Путь к файлу отчёта
        """
        report = self.get_enhanced_report(job_id)

        if output_path is None:
            output_path = self.output_dir / f"report_{job_id}.json"
        else:
            output_path = Path(output_path)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        return str(output_path)

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Получение статуса задания"""
        with self.lock:
            if job_id not in self.jobs:
                return {'error': f'Задание {job_id} не найдено'}
            return self.jobs[job_id].to_dict()

    def get_job_stats(self, job_id: str) -> Optional[BatchJobStats]:
        """Получение статистики задания"""
        with self.lock:
            if job_id not in self.jobs:
                return None
            job = self.jobs[job_id]
            duration = None
            if job.started_at:
                end = job.completed_at or datetime.now()
                duration = (end - job.started_at).total_seconds()
            return BatchJobStats(
                job_id=job.job_id,
                job_type=job.job_type,
                status=job.status,
                total_items=job.total_items,
                processed_items=job.processed_items,
                failed_items=job.failed_items,
                progress_percent=job.progress_percent,
                success_rate=job.success_rate,
                started_at=job.started_at.isoformat() if job.started_at else None,
                completed_at=job.completed_at.isoformat() if job.completed_at else None,
                duration_seconds=duration,
                priority=job.priority,
                errors_count=len(job.errors),
            )

    def get_all_jobs(self, status: str = None) -> List[Dict[str, Any]]:
        """Получение списка заданий"""
        with self.lock:
            jobs = list(self.jobs.values())

            if status:
                jobs = [job for job in jobs if job.status == status]

            return [job.to_dict() for job in jobs]

    def cancel_job(self, job_id: str) -> bool:
        """Отмена задания"""
        with self.lock:
            if job_id not in self.jobs:
                return False

            job = self.jobs[job_id]

            if job.status == 'running':
                return False

            job.status = 'cancelled'
            job.completed_at = datetime.now()

            if self.db_manager:
                self.db_manager.update_batch_job(job_id, status='cancelled')

            return True

    def pause_job(self, job_id: str) -> bool:
        """Приостановка задания"""
        with self.lock:
            if job_id not in self.jobs:
                return False
            job = self.jobs[job_id]
            if job.status != 'running':
                return False
            job.status = 'paused'
            return True

    def resume_job(self, job_id: str) -> bool:
        """Возобновление задания"""
        with self.lock:
            if job_id not in self.jobs:
                return False
            job = self.jobs[job_id]
            if job.status != 'paused':
                return False
            job.status = 'running'
            self.job_queue.put(job)
            return True

    def _save_job_results(self, job: BatchJob):
        """Сохранение результатов задания"""
        results_path = self.output_dir / f"{job.job_id}_results.json"

        output_data = {
            'job_info': job.to_dict(),
            'results': job.results,
            'errors': job.errors,
        }

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)

    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики обработки"""
        with self.lock:
            total_jobs = len(self.jobs)
            completed = sum(1 for j in self.jobs.values() if j.status == 'completed')
            failed = sum(1 for j in self.jobs.values() if j.status == 'failed')
            running = len(self.active_jobs)
            pending = total_jobs - completed - failed - running
            paused = sum(1 for j in self.jobs.values() if j.status == 'paused')

            return {
                'total_jobs': total_jobs,
                'completed': completed,
                'failed': failed,
                'running': running,
                'pending': pending,
                'paused': paused,
                'total_items_processed': self.total_items_processed,
                'success_rate': completed / (completed + failed) * 100 if (completed + failed) > 0 else 0,
            }

    async def process_item_async(self, item: Any, processor: Callable) -> Any:
        """Асинхронная обработка одного элемента"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, processor, item)

    async def process_batch_async(
        self,
        items: List[Any],
        processor: Callable,
        max_concurrent: int = 4
    ) -> List[Any]:
        """
        Асинхронная пакетная обработка

        Args:
            items: Элементы для обработки
            processor: Функция обработки
            max_concurrent: Максимум одновременных задач

        Returns:
            Список результатов
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(item: Any) -> Any:
            async with semaphore:
                return await self.process_item_async(item, processor)

        tasks = [process_with_semaphore(item) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=True)


class FolderWatcher:
    """
    Наблюдатель за папкой
    Автоматическая обработка новых файлов в папке
    """

    def __init__(
        self,
        watch_folder: str,
        processor: BatchProcessor,
        pattern: str = "*.png",
        callback: Callable = None
    ):
        """
        Инициализация наблюдателя

        Args:
            watch_folder: Папка для наблюдения
            processor: Процессор для обработки
            pattern: Шаблон файлов
            callback: Функция обратного вызова
        """
        self.watch_folder = Path(watch_folder)
        self.watch_folder.mkdir(parents=True, exist_ok=True)
        self.processor = processor
        self.pattern = pattern
        self.callback = callback

        self.is_running = False
        self.processed_files: set = set()
        self.thread: Optional[threading.Thread] = None

    def start(self, interval: float = 5.0):
        """
        Запуск наблюдения

        Args:
            interval: Интервал проверки (секунды)
        """
        self.is_running = True
        self.thread = threading.Thread(target=self._watch_loop, args=(interval,))
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Остановка наблюдения"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=10)

    def _watch_loop(self, interval: float):
        """Цикл наблюдения"""
        while self.is_running:
            try:
                self._check_new_files()
            except Exception as e:
                print(f"Ошибка в наблюдателе: {e}")

            time.sleep(interval)

    def _check_new_files(self):
        """Проверка новых файлов"""
        files = list(self.watch_folder.glob(self.pattern))

        for file_path in files:
            if file_path not in self.processed_files:
                self._process_new_file(file_path)

    def _process_new_file(self, file_path: Path):
        """Обработка нового файла"""
        print(f"Обнаружен новый файл: {file_path}")

        # Добавление в пакетную обработку
        job_id = self.processor.process_image_batch(
            image_paths=[str(file_path)],
            operation='analyze'
        )

        self.processed_files.add(file_path)

        if self.callback:
            self.callback(job_id, str(file_path))


# Глобальная функция для быстрой пакетной обработки
def batch_process_images(
    folder: str,
    operation: str = 'analyze',
    output_dir: str = "output/batch",
    max_workers: int = 4
) -> Dict[str, Any]:
    """
    Быстрая пакетная обработка изображений

    Args:
        folder: Папка с изображениями
        operation: Операция
        output_dir: Директория результатов
        max_workers: Количество потоков

    Returns:
        Результаты обработки
    """
    folder_path = Path(folder)
    images = list(folder_path.glob("*.png")) + list(folder_path.glob("*.jpg"))

    processor = BatchProcessor(max_workers=max_workers, output_dir=output_dir)

    job_id = processor.process_image_batch(
        image_paths=[str(img) for img in images],
        operation=operation
    )

    result = processor.run_job(job_id)

    return {
        'job_id': job_id,
        'status': result,
        'statistics': processor.get_statistics()
    }


if __name__ == "__main__":
    # Тестирование
    print("=== Тестирование пакетной обработки ===")

    # Создание процессора
    processor = BatchProcessor(max_workers=2, output_dir="output/batch_test")

    # Создание тестовых изображений
    if PIL_AVAILABLE:
        test_folder = Path("output/batch_test/input")
        test_folder.mkdir(parents=True, exist_ok=True)

        for i in range(5):
            img = Image.new('RGB', (100, 100), color=(i*50, 100, 150))
            img.save(test_folder / f"test_{i}.png")

        # Пакетная обработка
        images = list(test_folder.glob("*.png"))

        job_id = processor.process_image_batch(
            image_paths=[str(img) for img in images],
            operation='analyze'
        )

        print(f"Создано задание: {job_id}")

        # Выполнение
        result = processor.run_job(job_id)
        print(f"Статус: {result}")

        # Статистика
        stats = processor.get_statistics()
        print(f"\nСтатистика:")
        print(f"  Всего заданий: {stats['total_jobs']}")
        print(f"  Выполнено: {stats['completed']}")
        print(f"  Успешность: {stats['success_rate']:.1f}%")

        # Статус задания
        status = processor.get_job_status(job_id)
        print(f"\nСтатус задания: {status}")
