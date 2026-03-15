"""Тесты для модуля базы данных."""

import unittest
import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.database.database import DatabaseManager, get_database


class TestDatabaseManager(unittest.TestCase):
    """Тесты для DatabaseManager."""

    def setUp(self):
        """Подготовка тестового окружения."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.db = DatabaseManager(self.db_path)

    def tearDown(self):
        """Очистка после теста."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_scan_result(self):
        """Тестирует добавление результата сканирования."""
        scan_id = self.db.add_scan_result(
            scan_type="spm",
            surface_type="graphite",
            width=100,
            height=100,
            file_path="data/scan_001.txt",
            metadata={"voltage": 5.0, "current": 1.2}
        )

        self.assertIsInstance(scan_id, int)
        self.assertGreater(scan_id, 0)

    def test_get_scan_results(self):
        """Тестирует получение результатов сканирований."""
        # Добавляем тестовые данные
        for i in range(5):
            self.db.add_scan_result(
                scan_type="spm",
                surface_type=f"type_{i}",
                width=50,
                height=50
            )

        results = self.db.get_scan_results(limit=10)

        self.assertEqual(len(results), 5)
        self.assertIn('scan_type', results[0])

    def test_get_scan_results_filtered(self):
        """Тестирует фильтрацию результатов."""
        self.db.add_scan_result(scan_type="spm", surface_type="type_a")
        self.db.add_scan_result(scan_type="image", surface_type="type_b")

        spm_results = self.db.get_scan_results(scan_type="spm")
        image_results = self.db.get_scan_results(scan_type="image")

        self.assertEqual(len(spm_results), 1)
        self.assertEqual(len(image_results), 1)

    def test_add_simulation(self):
        """Тестирует добавление симуляции."""
        sim_id = self.db.add_simulation(
            simulation_id="sim_001",
            simulation_type="spm_scan",
            parameters={"speed": 1.0, "resolution": "high"}
        )

        self.assertIsInstance(sim_id, int)

    def test_update_simulation(self):
        """Тестирует обновление симуляции."""
        self.db.add_simulation(
            simulation_id="sim_002",
            simulation_type="image_analysis"
        )

        # Обновляем статус
        self.db.update_simulation("sim_002", status="completed")

        simulations = self.db.get_simulations(status="completed")
        # Проверяем, что симуляция обновлена
        self.assertEqual(len(simulations), 1)

    def test_get_simulations(self):
        """Тестирует получение списка симуляций."""
        for i in range(3):
            self.db.add_simulation(
                simulation_id=f"sim_{i:03d}",
                simulation_type="test"
            )

        simulations = self.db.get_simulations(limit=10)

        self.assertEqual(len(simulations), 3)

    def test_add_image(self):
        """Тестирует добавление изображения."""
        image_id = self.db.add_image(
            image_path="data/images/test.png",
            image_type="surface",
            source="hubble",
            width=1024,
            height=768,
            channels=3
        )

        self.assertIsInstance(image_id, int)
        self.assertGreater(image_id, 0)

    def test_get_images(self):
        """Тестирует получение изображений."""
        for i in range(4):
            self.db.add_image(
                image_path=f"data/image_{i}.png",
                image_type="surface" if i % 2 == 0 else "space",
                source="local"
            )

        all_images = self.db.get_images(limit=10)
        surface_images = self.db.get_images(image_type="surface")

        self.assertEqual(len(all_images), 4)
        self.assertEqual(len(surface_images), 2)

    def test_add_export(self):
        """Тестирует добавление записи об экспорте."""
        export_id = self.db.add_export(
            export_path="output/export_001.csv",
            export_format="csv",
            source_type="scan",
            source_id=1,
            file_size_bytes=1024
        )

        self.assertIsInstance(export_id, int)

    def test_get_statistics(self):
        """Тестирует получение статистики."""
        # Добавляем тестовые данные
        self.db.add_scan_result(scan_type="spm")
        self.db.add_scan_result(scan_type="image")
        self.db.add_simulation(simulation_id="sim_1", simulation_type="test")
        self.db.add_image(image_path="test.png")

        stats = self.db.get_statistics()

        self.assertIn('total_scans', stats)
        self.assertIn('total_simulations', stats)
        self.assertIn('total_images', stats)
        self.assertEqual(stats['total_scans'], 2)
        self.assertEqual(stats['total_simulations'], 1)
        self.assertEqual(stats['total_images'], 1)

    def test_search_scans(self):
        """Тестирует поиск по сканированиям."""
        self.db.add_scan_result(scan_type="test", surface_type="graphite", metadata={"note": "sample_1"})
        self.db.add_scan_result(scan_type="test", surface_type="silicon", metadata={"note": "sample_2"})

        results = self.db.search_scans(query="graphite")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['surface_type'], "graphite")

    def test_delete_scan(self):
        """Тестирует удаление сканирования."""
        scan_id = self.db.add_scan_result(scan_type="test")

        # Удаляем
        deleted = self.db.delete_scan(scan_id)
        self.assertTrue(deleted)

        # Проверяем, что удалено
        results = self.db.get_scan_results()
        self.assertEqual(len(results), 0)

    def test_export_to_json(self):
        """Тестирует экспорт БД в JSON."""
        self.db.add_scan_result(scan_type="test")
        self.db.add_simulation(simulation_id="sim_test", simulation_type="test")

        output_path = os.path.join(self.temp_dir, "export.json")
        result_path = self.db.export_to_json(output_path)

        self.assertTrue(result_path.exists())

        import json
        with open(result_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.assertIn('scan_results', data)
        self.assertIn('simulations', data)

    def test_get_database_singleton(self):
        """Тестирует singleton паттерн get_database."""
        db1 = get_database()
        db2 = get_database()

        self.assertIs(db1, db2)


def run_tests():
    """Запускает все тесты."""
    print("=" * 60)
    print("ЗАПУСК ТЕСТОВ БАЗЫ ДАННЫХ")
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
