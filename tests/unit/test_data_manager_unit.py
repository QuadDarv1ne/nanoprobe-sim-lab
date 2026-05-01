"""Unit-тесты для модуля управления данными."""

import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.data.data_manager import DataManager


class TestDataManager(unittest.TestCase):
    """Тесты для класса DataManager"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "data"
        self.output_dir = Path(self.temp_dir) / "output"
        self.data_manager = DataManager(
            data_dir=str(self.data_dir), output_dir=str(self.output_dir)
        )

    def tearDown(self):
        """Очистка после тестов"""
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        """Тест инициализации менеджера данных"""
        self.assertTrue(self.data_dir.exists())
        self.assertTrue(self.output_dir.exists())
        self.assertEqual(self.data_manager.data_dir, self.data_dir)
        self.assertEqual(self.data_manager.output_dir, self.output_dir)

    def test_save_and_load_surface_data(self):
        """Тест сохранения и загрузки данных поверхности"""
        # Создаем тестовые данные
        test_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)

        # Сохраняем данные
        result = self.data_manager.save_surface_data(test_data, "surface.txt")
        self.assertTrue(result)

        # Проверяем, что файл создан
        expected_path = self.output_dir / "surface.txt"
        self.assertTrue(expected_path.exists())

        # Загружаем данные
        loaded_data = self.data_manager.load_surface_data("surface.txt")
        self.assertIsNotNone(loaded_data)
        np.testing.assert_array_equal(loaded_data, test_data)

    def test_save_and_load_surface_data_not_found(self):
        """Тест загрузки несуществующего файла поверхности"""
        loaded_data = self.data_manager.load_surface_data("nonexistent.txt")
        self.assertIsNone(loaded_data)

    def test_save_and_load_scan_results(self):
        """Тест сохранения и загрузки результатов сканирования"""
        # Создаем тестовые данные
        test_data = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=float)

        # Сохраняем данные
        result = self.data_manager.save_scan_results(test_data, "scan.txt")
        self.assertTrue(result)

        # Проверяем, что файл создан
        expected_path = self.output_dir / "scan.txt"
        self.assertTrue(expected_path.exists())

        # Загружаем данные
        loaded_data = self.data_manager.load_scan_results("scan.txt")
        self.assertIsNotNone(loaded_data)
        np.testing.assert_array_equal(loaded_data, test_data)

    def test_save_and_load_image_analysis_results(self):
        """Тест сохранения и загрузки результатов анализа изображений"""
        # Создаем тестовые результаты
        test_results = {
            "surface_roughness": 0.123,
            "defect_count": 5,
            "analysis_date": datetime.now(timezone.utc).isoformat(),
            "quality_score": 0.85,
        }

        # Сохраняем результаты
        result = self.data_manager.save_image_analysis_results(test_results, "analysis.json")
        self.assertTrue(result)

        # Проверяем, что файл создан
        expected_path = self.output_dir / "analysis.json"
        self.assertTrue(expected_path.exists())

        # Загружаем результаты
        loaded_results = self.data_manager.load_image_analysis_results("analysis.json")
        self.assertIsNotNone(loaded_results)
        self.assertEqual(loaded_results["surface_roughness"], 0.123)
        self.assertEqual(loaded_results["defect_count"], 5)
        self.assertEqual(loaded_results["quality_score"], 0.85)

    def test_save_and_load_image_analysis_results_not_found(self):
        """Тест загрузки несуществующего файла анализа изображений"""
        loaded_results = self.data_manager.load_image_analysis_results("nonexistent.json")
        self.assertIsNone(loaded_results)

    def test_save_sstv_results(self):
        """Тест сохранения результатов SSTV декодирования"""
        # Создаем тестовые данные изображения (как numpy массив)
        test_image = np.random.rand(100, 100)

        # Сохраняем результаты
        result = self.data_manager.save_sstv_results(test_image, "sstv_image.npy")
        self.assertTrue(result)

        # Проверяем, что файл создан
        expected_path = self.output_dir / "sstv_image.npy"
        self.assertTrue(expected_path.exists())

        # Проверяем, что файл можно загрузить как numpy массив
        loaded_image = np.load(expected_path)
        np.testing.assert_array_equal(loaded_image, test_image)

    def test_save_and_load_simulation_metadata(self):
        """Тест сохранения и загрузки метаданных симуляции"""
        # Создаем тестовые метаданные
        test_metadata = {
            "simulation_id": "test_123",
            "parameters": {"voltage": 5.0, "current": 0.1},
            "status": "completed",
        }

        # Сохраняем метаданные
        result = self.data_manager.save_simulation_metadata(test_metadata, "sim_meta.json")
        self.assertTrue(result)

        # Проверяем, что файл создан
        expected_path = self.output_dir / "sim_meta.json"
        self.assertTrue(expected_path.exists())

        # Загружаем метаданные
        loaded_metadata = self.data_manager.load_simulation_metadata("sim_meta.json")
        self.assertIsNotNone(loaded_metadata)
        self.assertEqual(loaded_metadata["simulation_id"], "test_123")
        self.assertEqual(loaded_metadata["parameters"]["voltage"], 5.0)
        self.assertIn("timestamp", loaded_metadata)  # Добавляется автоматически

    def test_export_to_csv_numpy(self):
        """Тест экспорта данных numpy в CSV"""
        # Создаем тестовые данные
        test_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)

        # Экспортируем в CSV
        result = self.data_manager.export_to_csv(test_data, "data.csv")
        self.assertTrue(result)

        # Проверяем, что файл создан
        expected_path = self.output_dir / "data.csv"
        self.assertTrue(expected_path.exists())

        # Проверяем содержимое CSV
        import pandas as pd

        df = pd.read_csv(expected_path)
        # При чтении из CSV имена колонок становятся строками, поэтому сравниваем значения
        expected_df = pd.DataFrame(test_data)
        expected_df.columns = [str(i) for i in range(expected_df.shape[1])]
        pd.testing.assert_frame_equal(df, expected_df)

    def test_export_to_csv_dataframe(self):
        """Тест экспорта данных pandas DataFrame в CSV"""
        # Создаем тестовые данные
        import pandas as pd

        test_data = pd.DataFrame({"A": [1, 2, 3], "B": [4.0, 5.0, 6.0], "C": ["x", "y", "z"]})

        # Экспортируем в CSV
        result = self.data_manager.export_to_csv(test_data, "dataframe.csv")
        self.assertTrue(result)

        # Проверяем, что файл создан
        expected_path = self.output_dir / "dataframe.csv"
        self.assertTrue(expected_path.exists())

        # Проверяем содержимое CSV
        loaded_data = pd.read_csv(expected_path)
        pd.testing.assert_frame_equal(loaded_data, test_data)

    def test_get_recent_files(self):
        """Тест получения списка последних файлов"""
        # Создаем несколько файлов с разным временем модификации
        file1 = self.output_dir / "file1.txt"
        file2 = self.output_dir / "file2.txt"
        file3 = self.output_dir / "file3.log"

        file1.touch()
        file2.touch()
        file3.touch()

        # Получаем последние 2 файла с расширением .txt
        recent_files = self.data_manager.get_recent_files(".txt", 2)
        self.assertEqual(len(recent_files), 2)
        self.assertIn(file1, recent_files)
        self.assertIn(file2, recent_files)
        self.assertNotIn(file3, recent_files)  # Не .txt файл

    def test_cleanup_old_files(self):
        """Тест удаления старых файлов"""
        import time

        # Создаем старый и новый файлы
        old_file = self.output_dir / "old.txt"
        new_file = self.output_dir / "new.txt"

        old_file.touch()
        new_file.touch()

        # Делаем старый файл действительно старым (изменяем время модификации)
        old_time = time.time() - (31 * 24 * 60 * 60)  # 31 день назад
        os.utime(old_file, (old_time, old_time))

        # Запускаем очистку файлов старше 30 дней
        deleted_count = self.data_manager.cleanup_old_files(days_old=30)

        # Проверяем, что старый файл удален, а новый остался
        self.assertEqual(deleted_count, 1)
        self.assertFalse(old_file.exists())
        self.assertTrue(new_file.exists())


if __name__ == "__main__":
    unittest.main()
