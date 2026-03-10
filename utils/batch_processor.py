# -*- coding: utf-8 -*-
"""
Модуль пакетной обработки данных для проекта Nanoprobe Simulation Lab
Автоматизация обработки множественных файлов и заданий
"""

import os
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue, Empty
import traceback

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class BatchJob:
    """Класс отдельного задания пакетной обработки"""

    def __init__(
        self,
        job_id: str,
        job_type: str,
        items: List[Any],
        processor: Callable,
        parameters: Dict = None
    ):
        """
        Инициализация задания

        Args:
            job_id: ID задания
            job_type: Тип задания
            items: Список элементов для обработки
            processor: Функция обработки
            parameters: Параметры обработки
        """
        self.job_id = job_id
        self.job_type = job_type
        self.items = items
        self.processor = processor
        self.parameters = parameters or {}

        self.status = 'pending'
        self.total_items = len(items)
        self.processed_items = 0
        self.failed_items = 0
        self.results = []
        self.errors = []
        self.started_at = None
        self.completed_at = None

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
        }

    @property
    def progress_percent(self) -> float:
        """Процент выполнения"""
        if self.total_items == 0:
            return 0
        return 100 * self.processed_items / self.total_items


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
        parameters: Dict = None
    ) -> str:
        """
        Создание нового задания

        Args:
            job_type: Тип задания
            items: Элементы для обработки
            processor: Функция обработки
            parameters: Параметры

        Returns:
            ID созданного задания
        """
        job_id = f"batch_{job_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        job = BatchJob(job_id, job_type, items, processor, parameters)

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

        for job_id in pending_jobs:
            self.run_job(job_id, parallel)

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Получение статуса задания"""
        with self.lock:
            if job_id not in self.jobs:
                return {'error': f'Задание {job_id} не найдено'}
            return self.jobs[job_id].to_dict()

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

            return {
                'total_jobs': total_jobs,
                'completed': completed,
                'failed': failed,
                'running': running,
                'pending': pending,
                'total_items_processed': self.total_items_processed,
                'success_rate': completed / (completed + failed) * 100 if (completed + failed) > 0 else 0,
            }


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
