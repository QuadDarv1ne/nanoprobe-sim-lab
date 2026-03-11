# -*- coding: utf-8 -*-
"""
Unit-тесты для модуля управления данными
"""

import unittest
import tempfile
import shutil
import json
import numpy as np
import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_manager import DataManager


class TestDataManager(unittest.TestCase):
    """Тесты для класса DataManager"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "data"
        self.output_dir = Path(self.temp_dir) / "output"
        self.data_manager = DataManager(
            data_dir=str(self.data_dir), 
            output_dir=str(self.output_dir)
        )

    def tearDown(self):
        """Очистка после теста"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Тестирует инициализацию DataManager"""
        self.assertEqual(self.data_manager.data_dir, self.data_dir)
        self.assertEqual(self.data_manager.output_dir, self.output_dir)
        self.assertTrue(self.data_dir.exists())
        self.assertTrue(self.output_dir.exists())

    def test_save_and_load_surface_data(self):
        """Тестирует сохранение и загрузку данных поверхности"""
        test_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        # Сохраняем
        success = self.data_manager.save_surface_data(test_data, "surface.txt")
        self.assertTrue(success)
        
        # Загружаем
        loaded_data = self.data_manager.load_surface_data("surface.txt")
        
        self.assertIsNotNone(loaded_data)
        np.testing.assert_array_equal(loaded_data, test_data)

    def test_save_and_load_scan_results(self):
        """Тестирует сохранение и загрузку результатов сканирования"""
        test_data = np.linspace(0, 100, 50)
        
        # Сохраняем
        success = self.data_manager.save_scan_results(test_data, "scan.txt")
        self.assertTrue(success)
        
        # Загружаем
        loaded_data = self.data_manager.load_scan_results("scan.txt")
        
        self.assertIsNotNone(loaded_data)
        np.testing.assert_array_equal(loaded_data, test_data)

    def test_save_and_load_image_analysis_results(self):
        """Тестирует сохранение и загрузку результатов анализа изображений"""
        test_results = {
            "roughness": 0.5,
            "defects": 3,
            "quality": 0.95
        }
        
        # Сохраняем
        success = self.data_manager.save_image_analysis_results(
            test_results, "analysis.json"
        )
        self.assertTrue(success)
        
        # Загружаем
        loaded_results = self.data_manager.load_image_analysis_results("analysis.json")
        
        self.assertIsNotNone(loaded_results)
        self.assertEqual(loaded_results["roughness"], 0.5)
        self.assertEqual(loaded_results["defects"], 3)

    def test_save_simulation_metadata(self):
        """Тестирует сохранение метаданных симуляции"""
        metadata = {
            "duration": 3600,
            "parameters": {"temp": 25, "pressure": 101325}
        }
        
        success = self.data_manager.save_simulation_metadata(metadata)
        self.assertTrue(success)
        
        # Проверяем, что файл существует и содержит timestamp
        metadata_file = self.output_dir / "simulation_metadata.json"
        self.assertTrue(metadata_file.exists())
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        self.assertIn("timestamp", saved_data)
        self.assertEqual(saved_data["duration"], 3600)

    def test_load_simulation_metadata_nonexistent(self):
        """Тестирует загрузку несуществующих метаданных"""
        result = self.data_manager.load_simulation_metadata("nonexistent.json")
        self.assertIsNone(result)

    def test_export_to_csv_from_numpy(self):
        """Тестирует экспорт из numpy массива в CSV"""
        test_data = np.array([[1, 2, 3], [4, 5, 6]])
        
        success = self.data_manager.export_to_csv(test_data, "export.csv")
        self.assertTrue(success)
        
        # Проверяем файл
        export_file = self.output_dir / "export.csv"
        self.assertTrue(export_file.exists())

    def test_export_to_csv_from_dataframe(self):
        """Тестирует экспорт из DataFrame в CSV"""
        import pandas as pd
        
        test_df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6]
        })
        
        success = self.data_manager.export_to_csv(test_df, "dataframe.csv")
        self.assertTrue(success)
        
        export_file = self.output_dir / "dataframe.csv"
        self.assertTrue(export_file.exists())

    def test_get_recent_files(self):
        """Тестирует получение последних файлов"""
        # Создаём тестовые файлы
        (self.output_dir / "file1.txt").touch()
        (self.output_dir / "file2.txt").touch()
        (self.output_dir / "file3.txt").touch()
        
        recent = self.data_manager.get_recent_files(".txt", count=2)
        
        self.assertEqual(len(recent), 2)

    def test_get_recent_files_no_extension(self):
        """Тестирует получение последних файлов без фильтра по расширению"""
        (self.output_dir / "file1.txt").touch()
        (self.output_dir / "file2.json").touch()
        
        recent = self.data_manager.get_recent_files(count=5)
        
        self.assertGreaterEqual(len(recent), 2)

    def test_cleanup_old_files(self):
        """Тестирует удаление старых файлов"""
        import time
        import os
        
        # Создаём старый файл
        old_file = self.output_dir / "old_file.txt"
        old_file.write_text("old content")
        
        # Меняем время модификации на 31 день назад
        old_time = time.time() - (31 * 24 * 60 * 60)
        os.utime(old_file, (old_time, old_time))
        
        # Создаём новый файл
        new_file = self.output_dir / "new_file.txt"
        new_file.write_text("new content")
        
        # Запускаем очистку
        deleted_count = self.data_manager.cleanup_old_files(days_old=30)
        
        self.assertEqual(deleted_count, 1)
        self.assertFalse(old_file.exists())
        self.assertTrue(new_file.exists())

    def test_load_nonexistent_surface_data(self):
        """Тестирует загрузку несуществующих данных поверхности"""
        result = self.data_manager.load_surface_data("nonexistent.txt")
        self.assertIsNone(result)


class TestDataManagerEdgeCases(unittest.TestCase):
    """Тесты для граничных случаев DataManager"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Очистка после теста"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_corrupted_json(self):
        """Тестирует загрузку повреждённого JSON"""
        data_dir = Path(self.temp_dir) / "data"
        output_dir = Path(self.temp_dir) / "output"
        data_dir.mkdir()
        output_dir.mkdir()
        
        # Создаём повреждённый файл
        corrupted_file = data_dir / "corrupted.json"
        corrupted_file.write_text("{ invalid json }")
        
        dm = DataManager(str(data_dir), str(output_dir))
        result = dm.load_image_analysis_results("corrupted.json")
        
        self.assertIsNone(result)

    def test_export_unsupported_type(self):
        """Тестирует экспорт неподдерживаемого типа данных"""
        data_dir = Path(self.temp_dir) / "data"
        output_dir = Path(self.temp_dir) / "output"
        data_dir.mkdir()
        output_dir.mkdir()
        
        dm = DataManager(str(data_dir), str(output_dir))
        
        # Пробуем экспортировать неподдерживаемый тип
        success = dm.export_to_csv("not a dataframe", "test.csv")
        
        self.assertFalse(success)


if __name__ == '__main__':
    unittest.main()
