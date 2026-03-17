"""Тесты для экспортера данных."""

import unittest
import numpy as np
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.data.data_exporter import DataExporter


class TestDataExporter(unittest.TestCase):
    """Тесты для класса DataExporter"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = DataExporter(output_dir=self.temp_dir)
        self.sample_data = np.random.rand(10, 10)
        self.sample_dict = {'a': [1, 2, 3], 'b': [4, 5, 6]}

    def tearDown(self):
        """Очистка после теста"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass

    def test_export_csv(self):
        """Тестирует экспорт в CSV"""
        filepath = self.exporter.export(self.sample_data, 'test.csv', fmt='csv')
        self.assertTrue(filepath.exists())
        self.assertTrue(filepath.stat().st_size > 0)

    def test_export_json(self):
        """Тестирует экспорт в JSON"""
        filepath = self.exporter.export(self.sample_dict, 'test.json', fmt='json')
        self.assertTrue(filepath.exists())
        self.assertTrue(filepath.stat().st_size > 0)

    def test_export_npy(self):
        """Тестирует экспорт в NPY"""
        filepath = self.exporter.export(self.sample_data, 'test.npy', fmt='npy')
        self.assertTrue(filepath.exists())
        self.assertTrue(filepath.stat().st_size > 0)

    def test_export_invalid_format(self):
        """Тестирует экспорт с неверным форматом"""
        with self.assertRaises(ValueError):
            self.exporter.export(self.sample_data, 'test.xyz', fmt='xyz')

    def test_export_empty_filename(self):
        """Тестирует экспорт с пустым именем файла"""
        with self.assertRaises(ValueError):
            self.exporter.export(self.sample_data, '', fmt='csv')

    def test_export_none_data(self):
        """Тестирует экспорт с пустыми данными"""
        with self.assertRaises(ValueError):
            self.exporter.export(None, 'test.csv', fmt='csv')

    def test_export_invalid_extension(self):
        """Тестирует экспорт с неверным расширением"""
        with self.assertRaises(ValueError):
            self.exporter.export(self.sample_data, 'test.xyz', fmt='csv')

    def test_export_surface_data(self):
        """Тестирует экспорт данных поверхности"""
        filepath = self.exporter.export_surface_data(self.sample_data, fmt='json')
        self.assertTrue(filepath.exists())

    def test_export_scan_results_csv(self):
        """Тестирует экспорт результатов сканирования в CSV"""
        filepath = self.exporter.export_scan_results(self.sample_data, fmt='csv')
        self.assertTrue(filepath.exists())

    def test_export_scan_results_json(self):
        """Тестирует экспорт результатов сканирования в JSON"""
        filepath = self.exporter.export_scan_results(self.sample_data, fmt='json')
        self.assertTrue(filepath.exists())


def run_tests():
    """Запускает все тесты."""
    print("=" * 60)
    print("ЗАПУСК ТЕСТОВ ДЛЯ DATA EXPORTER")
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
