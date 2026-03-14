"""Тесты для анализатора изображений поверхности."""

import unittest
import numpy as np
import sys
import os
import tempfile

# Добавляем путь к исходному коду
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../components/py-surface-image-analyzer/src'))

from image_processor import ImageProcessor, calculate_surface_roughness


class TestImageProcessor(unittest.TestCase):
    """Тесты для класса ImageProcessor"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.processor = ImageProcessor()
        # Создаем тестовое изображение для всех тестов
        self.test_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        from PIL import Image
        img = Image.fromarray(self.test_image)
        img.save(self.temp_file.name)
        self.temp_file.close()
        # Загружаем изображение в процессор
        self.processor.load_image(self.temp_file.name)

    def tearDown(self):
        """Очистка после теста"""
        try:
            os.unlink(self.temp_file.name)
        except Exception:
            pass

    def test_initialization(self):
        """Тестирует инициализацию процессора изображений"""
        processor = ImageProcessor()
        self.assertIsNone(processor.image)
        self.assertIsNone(processor.processed_image)

    def test_load_image(self):
        """Тестирует загрузку изображения"""
        self.assertTrue(self.processor.image is not None)

    def test_apply_noise_reduction_gaussian(self):
        """Тестирует применение гауссового фильтра"""
        result = self.processor.apply_noise_reduction("gaussian")
        self.assertIsNotNone(result)

    def test_apply_noise_reduction_median(self):
        """Тестирует применение медианного фильтра"""
        result = self.processor.apply_noise_reduction("median")
        self.assertIsNotNone(result)

    def test_apply_noise_reduction_bilateral(self):
        """Тестирует применение билатерального фильтра"""
        result = self.processor.apply_noise_reduction("bilateral")
        self.assertIsNotNone(result)

    def test_apply_noise_reduction_invalid_method(self):
        """Тестирует применение неверного метода фильтрации"""
        result = self.processor.apply_noise_reduction("invalid")
        self.assertIsNone(result)

    def test_detect_edges(self):
        """Тестирует обнаружение краев"""
        self.processor.apply_noise_reduction("gaussian")
        result = self.processor.detect_edges()
        self.assertIsNotNone(result)

    def test_detect_edges_invalid_thresholds(self):
        """Тестирует обнаружение краев с некорректными порогами"""
        self.processor.apply_noise_reduction("gaussian")
        result = self.processor.detect_edges(threshold1=200, threshold2=100)
        self.assertIsNone(result)

    def test_load_image_nonexistent_file(self):
        """Тестирует загрузку несуществующего файла"""
        result = self.processor.load_image("/nonexistent/path/image.png")
        self.assertFalse(result)

    def test_load_image_invalid_format(self):
        """Тестирует загрузку файла с неподдерживаемым форматом"""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b"test")
            tmp_name = tmp.name
        result = self.processor.load_image(tmp_name)
        self.assertFalse(result)
        try:
            os.unlink(tmp_name)
        except Exception:
            pass


class TestUtilityFunctions(unittest.TestCase):
    """Тесты для вспомогательных функций"""

    def test_calculate_surface_roughness(self):
        """Тестирует вычисление шероховатости поверхности"""
        # Используем uint8 изображение для совместимости с cv2
        test_image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        roughness = calculate_surface_roughness(test_image)

        self.assertIsInstance(roughness, dict)
        self.assertIn('ra', roughness)
        self.assertIn('rq', roughness)
        self.assertIn('rz', roughness)
        self.assertGreaterEqual(roughness['ra'], 0)

    def test_calculate_surface_roughness_grayscale(self):
        """Тестирует вычисление шероховатости для Ч/Б изображения"""
        # Используем uint8 изображение для совместимости с cv2
        test_image = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        roughness = calculate_surface_roughness(test_image)

        self.assertIsInstance(roughness, dict)
        self.assertIn('ra', roughness)
        self.assertGreaterEqual(roughness['ra'], 0)

    def test_calculate_surface_roughness_empty_image(self):
        """Тестирует обработку пустого изображения"""
        empty_image = np.array([])
        with self.assertRaises(ValueError):
            calculate_surface_roughness(empty_image)

    def test_calculate_surface_roughness_invalid_channels(self):
        """Тестирует обработку изображения с неверным количеством каналов"""
        invalid_image = np.random.randint(0, 255, (10, 10, 2), dtype=np.uint8)
        with self.assertRaises(ValueError):
            calculate_surface_roughness(invalid_image)


class TestImageProcessorExtended(unittest.TestCase):
    """Расширенные тесты для ImageProcessor"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.processor = ImageProcessor()
        # Создаем тестовое изображение
        self.test_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

        import tempfile
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        from PIL import Image
        img = Image.fromarray(self.test_image)
        img.save(self.temp_file.name)
        self.temp_file.close()

    def tearDown(self):
        """Очистка после теста"""
        import os
        try:
            os.unlink(self.temp_file.name)
        except Exception:
            pass

    def test_load_image_actual(self):
        """Тестирует загрузку реального изображения"""
        result = self.processor.load_image(self.temp_file.name)
        self.assertTrue(result)
        self.assertIsNotNone(self.processor.image)
        self.assertEqual(self.processor.image.shape, (50, 50, 3))

    def test_get_statistics(self):
        """Тестирует получение статистики изображения"""
        self.processor.load_image(self.temp_file.name)
        stats = self.processor.get_statistics()

        self.assertIsNotNone(stats)
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
        self.assertIn('min', stats)
        self.assertIn('max', stats)
        self.assertIn('shape', stats)

    def test_get_histogram(self):
        """Тестирует получение гистограммы"""
        self.processor.load_image(self.temp_file.name)
        hist = self.processor.get_histogram()

        self.assertIsNotNone(hist)
        self.assertEqual(hist.shape, (256, 1))

    def test_save_image(self):
        """Тестирует сохранение изображения"""
        import tempfile
        import time
        self.processor.load_image(self.temp_file.name)
        self.processor.apply_noise_reduction('gaussian')

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            save_path = tmp.name

        result = self.processor.save_image(save_path)
        self.assertTrue(result)

        # Даем файлу закрыться перед удалением
        time.sleep(0.1)
        try:
            os.unlink(save_path)
        except Exception:
            pass

    def test_get_metadata(self):
        """Тестирует получение метаданных"""
        self.processor.load_image(self.temp_file.name)
        self.processor.apply_noise_reduction('median')

        metadata = self.processor.get_metadata()

        self.assertIn('filepath', metadata)
        self.assertIn('filter_applied', metadata)
        self.assertEqual(metadata['filter_applied'], 'median')


def run_tests():
    """Запускает все тесты."""
    print("=" * 60)
    print("ЗАПУСК ТЕСТОВ ДЛЯ АНАЛИЗАТОРА ИЗОБРАЖЕНИЙ")
    print("=" * 60)

    # Создаем тестовый набор
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    # Запускаем тесты
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

