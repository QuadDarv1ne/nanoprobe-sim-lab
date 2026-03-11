# -*- coding: utf-8 -*-
"""Тесты для новых функций проекта."""

import unittest
import sys
import tempfile
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataExporter(unittest.TestCase):
    """Тесты для экспортера данных."""

    def test_export_csv(self):
        """Тестирует экспорт в CSV."""
        from utils.data_exporter import DataExporter
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = DataExporter(tmpdir)
            data = {'x': [1, 2, 3], 'y': [4, 5, 6]}
            filepath = exporter.export(data, 'test.csv', fmt='csv')
            
            self.assertTrue(filepath.exists())
            self.assertEqual(filepath.suffix, '.csv')

    def test_export_json(self):
        """Тестирует экспорт в JSON."""
        from utils.data_exporter import DataExporter
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = DataExporter(tmpdir)
            data = {'key': 'value', 'number': 42}
            filepath = exporter.export(data, 'test.json', fmt='json')
            
            self.assertTrue(filepath.exists())
            self.assertEqual(filepath.suffix, '.json')

    def test_export_npy(self):
        """Тестирует экспорт в NumPy формат."""
        from utils.data_exporter import DataExporter
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = DataExporter(tmpdir)
            data = np.array([[1, 2], [3, 4]])
            filepath = exporter.export(data, 'test.npy', fmt='npy')
            
            self.assertTrue(filepath.exists())
            self.assertEqual(filepath.suffix, '.npy')

    def test_export_surface_data(self):
        """Тестирует экспорт данных поверхности."""
        from utils.data_exporter import DataExporter
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = DataExporter(tmpdir)
            surface_data = np.random.rand(10, 10)
            filepath = exporter.export_surface_data(surface_data, fmt='json')
            
            self.assertTrue(filepath.exists())

    def test_import_json(self):
        """Тестирует импорт из JSON."""
        from utils.data_exporter import DataExporter, DataImporter
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = DataExporter(tmpdir)
            importer = DataImporter(tmpdir)
            
            data = {'key': 'value', 'number': 42}
            filepath = exporter.export(data, 'test.json', fmt='json')
            
            imported = importer.import_file(filepath)
            self.assertEqual(imported['key'], 'value')
            self.assertEqual(imported['number'], 42)


class TestCLIUtils(unittest.TestCase):
    """Тесты для CLI утилит."""

    def test_colors_basic(self):
        """Тестирует базовые цвета."""
        from utils.cli_utils import Colors, colorize
        
        red_text = colorize("test", Colors.RED)
        self.assertIn(Colors.RED, red_text)
        self.assertIn(Colors.RESET, red_text)

    def test_colorize_function(self):
        """Тестирует функцию colorize."""
        from utils.cli_utils import colorize, Colors
        
        result = colorize("hello", Colors.GREEN)
        self.assertIsInstance(result, str)
        self.assertIn("hello", result)

    def test_progress_bar_creation(self):
        """Тестирует создание progress bar."""
        from utils.cli_utils import ProgressBar
        
        pb = ProgressBar(total=10, desc="Test")
        self.assertEqual(pb.total, 10)
        self.assertEqual(pb.desc, "Test")

    def test_progress_bar_iteration(self):
        """Тестирует итерацию progress bar."""
        from utils.cli_utils import ProgressBar
        
        pb = ProgressBar(total=5, desc="Test")
        count = 0
        for i in pb:
            count += 1
        
        self.assertEqual(count, 5)


class TestSPMMultiprocessing(unittest.TestCase):
    """Тесты для multiprocessing в СЗМ."""

    def setUp(self):
        """Подготовка тестового окружения."""
        spm_path = Path(__file__).parent.parent / "components" / "cpp-spm-hardware-sim" / "src"
        sys.path.insert(0, str(spm_path))

    def test_parallel_scan(self):
        """Тестирует параллельное сканирование."""
        from spm_simulator import SurfaceModel, SPMController
        
        surface = SurfaceModel(20, 20)
        controller = SPMController()
        controller.set_surface(surface)
        
        # Параллельное сканирование
        controller.scan_surface(parallel=True, num_processes=2)
        
        self.assertIsNotNone(controller.scan_data)
        self.assertEqual(controller.scan_data.shape, (20, 20))

    def test_sequential_scan(self):
        """Тестирует последовательное сканирование."""
        from spm_simulator import SurfaceModel, SPMController
        
        surface = SurfaceModel(10, 10)
        controller = SPMController()
        controller.set_surface(surface)
        
        # Последовательное сканирование
        controller.scan_surface(parallel=False)
        
        self.assertIsNotNone(controller.scan_data)
        self.assertEqual(controller.scan_data.shape, (10, 10))

    def test_scan_data_range(self):
        """Тестирует диапазон данных сканирования."""
        from spm_simulator import SurfaceModel, SPMController
        
        surface = SurfaceModel(15, 15)
        controller = SPMController()
        controller.set_surface(surface)
        controller.scan_surface(parallel=False)
        
        # Все значения должны быть в разумном диапазоне
        self.assertTrue(np.all(controller.scan_data >= 0))
        self.assertTrue(np.all(controller.scan_data <= 2))


def run_tests():
    """Запускает все тесты."""
    print("=" * 60)
    print("ЗАПУСК ТЕСТОВ НОВЫХ ФУНКЦИЙ")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 60)
    print(f"РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print(f"  Пройдено: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Ошибки: {len(result.errors)}")
    print(f"  Провалы: {len(result.failures)}")
    print("=" * 60)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
