"""Тесты для batch processor."""

import unittest
import tempfile
import shutil
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.batch_processor import BatchProcessor, BatchJob


class TestBatchJob(unittest.TestCase):
    """Тесты для класса BatchJob"""

    def setUp(self):
        """Подготовка"""
        self.job = BatchJob(
            job_id="test_001",
            job_type="test",
            items=[1, 2, 3, 4, 5],
            processor=lambda x: x * 2,
            priority=5
        )

    def test_initialization(self):
        """Тест инициализации"""
        self.assertEqual(self.job.job_id, "test_001")
        self.assertEqual(self.job.job_type, "test")
        self.assertEqual(self.job.total_items, 5)
        self.assertEqual(self.job.priority, 5)
        self.assertEqual(self.job.status, "pending")

    def test_progress_percent(self):
        """Тест процента выполнения"""
        self.assertEqual(self.job.progress_percent, 0)

        self.job.processed_items = 3
        self.assertEqual(self.job.progress_percent, 60)

    def test_success_rate(self):
        """Тест процента успеха"""
        self.assertEqual(self.job.success_rate, 0)

        self.job.processed_items = 10
        self.job.failed_items = 2
        self.assertEqual(self.job.success_rate, 80)

    def test_to_dict(self):
        """Тест конвертации в словарь"""
        self.job.processed_items = 3
        result = self.job.to_dict()

        self.assertIn('job_id', result)
        self.assertIn('progress', result)
        self.assertIn('priority', result)
        self.assertIn('success_rate', result)
        self.assertEqual(result['priority'], 5)

    def test_progress_callback(self):
        """Тест callback прогресса"""
        callback_called = []

        def callback(job_id, progress):
            """TODO: Add description"""
            callback_called.append((job_id, progress))

        self.job.set_progress_callback(callback)
        self.job.update_progress(2)

        self.assertEqual(len(callback_called), 1)
        self.assertEqual(callback_called[0][0], "test_001")
        self.assertEqual(callback_called[0][1], 40)


class TestBatchProcessor(unittest.TestCase):
    """Тесты для BatchProcessor"""

    def setUp(self):
        """Подготовка"""
        self.temp_dir = tempfile.mkdtemp()
        self.processor = BatchProcessor(max_workers=2, output_dir=self.temp_dir)

    def tearDown(self):
        """Очистка"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_job_with_priority(self):
        """Тест создания задания с приоритетом"""
        job_id = self.processor.create_job(
            job_type="test",
            items=[1, 2, 3],
            processor=lambda x: x * 2,
            priority=10
        )

        self.assertIn(job_id, self.processor.jobs)
        self.assertEqual(self.processor.jobs[job_id].priority, 10)

    def test_create_job_default_priority(self):
        """Тест создания задания с приоритетом по умолчанию"""
        job_id = self.processor.create_job(
            job_type="test",
            items=[1, 2, 3],
            processor=lambda x: x * 2
        )

        self.assertEqual(self.processor.jobs[job_id].priority, 0)

    def test_run_job_sequential(self):
        """Тест последовательного выполнения"""
        job_id = self.processor.create_job(
            job_type="test",
            items=[1, 2, 3, 4, 5],
            processor=lambda x: x * 2
        )

        result = self.processor.run_job(job_id, parallel=False)

        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['processed_items'], 5)
        self.assertEqual(result['failed_items'], 0)

    def test_run_job_with_errors(self):
        """Тест выполнения с ошибками"""
        def faulty_processor(x):
            """TODO: Add description"""
            if x == 3:
                raise ValueError("Test error")
            return x * 2

        job_id = self.processor.create_job(
            job_type="test",
            items=[1, 2, 3, 4, 5],
            processor=faulty_processor
        )

        result = self.processor.run_job(job_id, parallel=False)

        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['failed_items'], 1)

    def test_get_enhanced_report(self):
        """Тест расширенного отчёта"""
        job_id = self.processor.create_job(
            job_type="test",
            items=[1, 2, 3],
            processor=lambda x: x * 2,
            priority=5
        )

        self.processor.run_job(job_id, parallel=False)
        report = self.processor.get_enhanced_report(job_id)

        self.assertIn('job_id', report)
        self.assertIn('detailed_stats', report)
        self.assertIn('recommendations', report)
        self.assertIn('sample_results', report)

    def test_get_queue_summary(self):
        """Тест сводки очереди"""
        self.processor.create_job("test1", [1, 2], lambda x: x, priority=1)
        self.processor.create_job("test2", [3, 4], lambda x: x, priority=5)
        self.processor.create_job("test3", [5, 6], lambda x: x, priority=1)

        summary = self.processor.get_queue_summary()

        self.assertEqual(summary['total_jobs'], 3)
        self.assertEqual(summary['total_items'], 6)
        self.assertIn(1, summary['jobs_by_priority'])
        self.assertIn(5, summary['jobs_by_priority'])

    def test_export_job_report(self):
        """Тест экспорта отчёта"""
        job_id = self.processor.create_job(
            job_type="test",
            items=[1, 2, 3],
            processor=lambda x: x * 2
        )

        self.processor.run_job(job_id, parallel=False)
        report_path = self.processor.export_job_report(job_id)

        self.assertTrue(Path(report_path).exists())

    def test_run_all_pending_priority_order(self):
        """Тест выполнения по приоритету"""
        execution_order = []

        def make_processor(value):
            """TODO: Add description"""
            def processor(x):
                """TODO: Add description"""
                execution_order.append(value)
                return x * 2
            return processor

        # Создание заданий с разным приоритетом
        self.processor.create_job("low", [1], make_processor("low"), priority=1)
        self.processor.create_job("high", [1], make_processor("high"), priority=10)
        self.processor.create_job("medium", [1], make_processor("medium"), priority=5)

        self.processor.run_all_pending(parallel=False)

        # Высокий приоритет должен выполниться первым
        self.assertEqual(execution_order[0], "high")
        self.assertEqual(execution_order[1], "medium")
        self.assertEqual(execution_order[2], "low")

    def test_get_job_status(self):
        """Тест получения статуса задания"""
        job_id = self.processor.create_job(
            job_type="test",
            items=[1, 2, 3],
            processor=lambda x: x * 2
        )

        status = self.processor.get_job_status(job_id)

        self.assertEqual(status['job_id'], job_id)
        self.assertEqual(status['total_items'], 3)

    def test_get_job_status_not_found(self):
        """Тест статуса несуществующего задания"""
        status = self.processor.get_job_status("nonexistent")

        self.assertIn('error', status)

    def test_get_all_jobs(self):
        """Тест получения всех заданий"""
        self.processor.create_job("test1", [1, 2], lambda x: x)
        self.processor.create_job("test2", [3, 4], lambda x: x)

        all_jobs = self.processor.get_all_jobs()
        self.assertEqual(len(all_jobs), 2)

    def test_cancel_job(self):
        """Тест отмены задания"""
        job_id = self.processor.create_job(
            job_type="test",
            items=[1, 2, 3],
            processor=lambda x: x * 2
        )

        # Отмена до выполнения
        result = self.processor.cancel_job(job_id)
        self.assertTrue(result)

        # Отмена несуществующего
        result = self.processor.cancel_job("nonexistent")
        self.assertFalse(result)


class TestBatchProcessorIntegration(unittest.TestCase):
    """Интеграционные тесты"""

    def setUp(self):
        """Подготовка"""
        self.temp_dir = tempfile.mkdtemp()
        self.processor = BatchProcessor(max_workers=4, output_dir=self.temp_dir)

    def tearDown(self):
        """Очистка"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_batch_workflow(self):
        """Тест полного рабочего процесса"""
        # Создание партии изображений (симуляция)
        items = [{"id": i, "data": f"item_{i}"} for i in range(10)]

        def process_item(item):
            """TODO: Add description"""
            time.sleep(0.01)  # Имитация работы
            return {"id": item["id"], "processed": True, "result": item["data"] + "_done"}

        job_id = self.processor.create_job(
            job_type="image_processing",
            items=items,
            processor=process_item,
            priority=7
        )

        # Выполнение
        result = self.processor.run_job(job_id, parallel=True)

        # Проверка
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['processed_items'], 10)
        self.assertEqual(result['priority'], 7)

        # Расширенный отчёт
        report = self.processor.get_enhanced_report(job_id)
        self.assertIn('recommendations', report)
        self.assertTrue(len(report['recommendations']) > 0)

    def test_multiple_jobs_with_different_priorities(self):
        """Тест нескольких заданий с разными приоритетами"""
        results = {'order': []}

        def make_processor(name):
            """TODO: Add description"""
            def processor(x):
                """TODO: Add description"""
                results['order'].append(name)
                return x
            return processor

        # Создание 5 заданий с разными приоритетами
        for i in range(5):
            self.processor.create_job(
                f"job_{i}",
                [1],
                make_processor(f"job_{i}"),
                priority=i
            )

        # Выполнение всех
        self.processor.run_all_pending(parallel=False)

        # Проверка порядка выполнения (обратный приоритету)
        expected_order = [f"job_{i}" for i in range(4, -1, -1)]
        self.assertEqual(results['order'], expected_order)


if __name__ == '__main__':
    unittest.main()
