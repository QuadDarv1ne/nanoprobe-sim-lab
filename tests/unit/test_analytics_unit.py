"""Unit-тесты для модуля аналитики."""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from utils.analytics import ImageAnalytics, ProjectAnalytics, SSTVAnalytics, SurfaceAnalytics


class TestSurfaceAnalytics(unittest.TestCase):
    """Тесты для класса SurfaceAnalytics"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.surface_analytics = SurfaceAnalytics()
        # Создаем тестовые данные поверхности
        self.test_surface = np.array([[1, 2], [3, 4]], dtype=float)

    def test_init(self):
        """Тест инициализации аналитика поверхности"""
        self.assertIsInstance(self.surface_analytics.scaler, type(self.surface_analytics.scaler))
        self.assertIsInstance(self.surface_analytics.pca, type(self.surface_analytics.pca))
        self.assertIsInstance(self.surface_analytics.kmeans, type(self.surface_analytics.kmeans))
        self.assertIsInstance(
            self.surface_analytics.regressor, type(self.surface_analytics.regressor)
        )

    def test_calculate_surface_properties(self):
        """Тест вычисления свойств поверхности"""
        properties = self.surface_analytics.calculate_surface_properties(self.test_surface)
        self.assertIsInstance(properties, dict)
        self.assertIn("mean_height", properties)
        self.assertIn("std_height", properties)
        self.assertIn("min_height", properties)
        self.assertIn("max_height", properties)
        self.assertIn("height_range", properties)
        self.assertIn("surface_roughness_rms", properties)
        self.assertIn("skewness", properties)
        self.assertIn("kurtosis", properties)
        self.assertIn("surface_area", properties)
        self.assertIn("volume", properties)

        # Проверяем, что значения являются числами с плавающей точкой
        for key, value in properties.items():
            self.assertIsInstance(value, float, f"Property {key} should be float")

    def test_cluster_surface_regions(self):
        """Тест кластеризации областей поверхности"""
        cluster_map = self.surface_analytics.cluster_surface_regions(
            self.test_surface, n_clusters=2
        )
        self.assertIsInstance(cluster_map, np.ndarray)
        self.assertEqual(cluster_map.shape, self.test_surface.shape)
        # Проверяем, что метки кластеров являются целыми числами в ожидаемом диапазоне
        self.assertTrue(np.all(cluster_map >= 0))
        self.assertTrue(np.all(cluster_map < 2))

    def test_dimensionality_reduction(self):
        """Тест понижения размерности поверхности"""
        reduced_data, explained_variance = self.surface_analytics.dimensionality_reduction(
            self.test_surface
        )
        self.assertIsInstance(reduced_data, np.ndarray)
        self.assertIsInstance(explained_variance, np.ndarray)
        # Для наших тестовых данных 2x2, после преобразования должно быть 4 строки и 2 столбца
        self.assertEqual(reduced_data.shape, (4, 2))
        self.assertEqual(explained_variance.shape, (2,))
        # Сумма объясненной дисперсии должна быть <= 1
        self.assertLessEqual(np.sum(explained_variance), 1.0)

    def test_predict_surface_properties(self):
        """Тест предсказания свойств поверхности"""
        # Создаем тестовые признаки
        features = np.random.rand(10, 3)
        predictions, mse, r2 = self.surface_analytics.predict_surface_properties(features)
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(predictions.shape, (2,))  # 20% от 10 = 2
        self.assertIsInstance(mse, float)
        self.assertIsInstance(r2, float)


class TestImageAnalytics(unittest.TestCase):
    """Тесты для класса ImageAnalytics"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.image_analytics = ImageAnalytics()
        # Создаем тестовые данные изображения
        self.test_image = np.random.rand(10, 10)
        self.test_color_image = np.random.rand(10, 10, 3)

    def test_init(self):
        """Тест инициализации аналитика изображений"""
        # Просто проверяем, что объект создан
        self.assertIsInstance(self.image_analytics, ImageAnalytics)

    def test_calculate_image_features(self):
        """Тест вычисления признаков изображения"""
        features = self.image_analytics.calculate_image_features(self.test_image)
        self.assertIsInstance(features, dict)
        expected_keys = [
            "mean_intensity",
            "std_intensity",
            "min_intensity",
            "max_intensity",
            "contrast",
            "entropy",
            "homogeneity",
            "energy",
            "correlation",
        ]
        for key in expected_keys:
            self.assertIn(key, features)
            self.assertIsInstance(features[key], float, f"Feature {key} should be float")

    def test_calculate_image_features_color(self):
        """Тест вычисления признаков цветного изображения"""
        features = self.image_analytics.calculate_image_features(self.test_color_image)
        self.assertIsInstance(features, dict)
        # Для цветного изображения должно быть преобразовано в оттенки серого
        self.assertIn("mean_intensity", features)
        self.assertIsInstance(features["mean_intensity"], float)

    def test_detect_patterns(self):
        """Тест обнаружения паттернов в изображении"""
        patterns = self.image_analytics.detect_patterns(self.test_image)
        self.assertIsInstance(patterns, dict)
        expected_keys = [
            "edge_density",
            "texture_complexity",
            "average_edge_strength",
            "pattern_regions",
        ]
        for key in expected_keys:
            self.assertIn(key, patterns)
            self.assertIsInstance(patterns[key], float, f"Pattern {key} should be float")


class TestSSTVAnalytics(unittest.TestCase):
    """Тесты для класса SSTVAnalytics"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.sstv_analytics = SSTVAnalytics()
        # Создаем тестовые данные сигнала
        self.test_signal = np.random.rand(1000)
        self.test_image = np.random.rand(50, 50)

    def test_init(self):
        """Тест инициализации аналитика SSTV"""
        self.assertIsInstance(self.sstv_analytics, SSTVAnalytics)

    def test_analyze_signal_quality(self):
        """Тест анализа качества сигнала"""
        quality_metrics = self.sstv_analytics.analyze_signal_quality(
            self.test_signal, sample_rate=44100
        )
        self.assertIsInstance(quality_metrics, dict)
        expected_keys = [
            "rms_amplitude",
            "peak_amplitude",
            "signal_power",
            "dominant_frequency",
            "signal_to_noise_ratio_db",
            "total_energy",
            "zero_crossing_rate",
            "spectral_centroid",
        ]
        for key in expected_keys:
            self.assertIn(key, quality_metrics)
            self.assertIsInstance(quality_metrics[key], float, f"Metric {key} should be float")

    def test_evaluate_decoding_quality(self):
        """Тест оценки качества декодирования"""
        # Создаем два похожих изображения
        original = self.test_image
        decoded = original + np.random.rand(*original.shape) * 0.1  # Добавляем небольшой шум

        quality_metrics = self.sstv_analytics.evaluate_decoding_quality(original, decoded)
        self.assertIsInstance(quality_metrics, dict)
        expected_keys = ["mse", "psnr", "ssim", "correlation", "mean_difference"]
        for key in expected_keys:
            self.assertIn(key, quality_metrics)
            self.assertIsInstance(quality_metrics[key], float, f"Metric {key} should be float")


class TestProjectAnalytics(unittest.TestCase):
    """Тесты для класса ProjectAnalytics"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.temp_dir = tempfile.mkdtemp()
        self.project_analytics = ProjectAnalytics()
        # Создаем тестовые данные
        self.test_surface = np.random.rand(20, 20)
        self.test_image = np.random.rand(20, 20, 3)
        self.test_signal = np.random.rand(1000)

    def tearDown(self):
        """Очистка после тестов"""
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        """Тест инициализации центрального аналитика"""
        self.assertIsInstance(self.project_analytics.surface_analytics, SurfaceAnalytics)
        self.assertIsInstance(self.project_analytics.image_analytics, ImageAnalytics)
        self.assertIsInstance(self.project_analytics.sstv_analytics, SSTVAnalytics)

    def test_generate_comprehensive_report(self):
        """Тест генерации комплексного аналитического отчета"""
        report = self.project_analytics.generate_comprehensive_report(
            surface_data=self.test_surface,
            image_data=self.test_image,
            signal_data=self.test_signal,
        )
        self.assertIsInstance(report, dict)
        self.assertIn("timestamp", report)
        self.assertIn("analyses_performed", report)
        self.assertIn("surface_analysis", report)
        self.assertIn("image_analysis", report)
        self.assertIn("sstv_analysis", report)

        # Проверяем, что анализы были выполнены
        self.assertIn("surface", report["analyses_performed"])
        self.assertIn("image", report["analyses_performed"])
        self.assertIn("sstv", report["analyses_performed"])

        # Проверяем, что каждый анализ содержит данные
        self.assertIsInstance(report["surface_analysis"], dict)
        self.assertIsInstance(report["image_analysis"], dict)
        self.assertIsInstance(report["sstv_analysis"], dict)

    def test_save_analytics_report(self):
        """Тест сохранения аналитического отчета"""
        report = {
            "timestamp": "2023-01-01T00:00:00",
            "analyses_performed": ["surface"],
            "surface_analysis": {"mean_height": 1.0},
        }
        filename = Path(self.temp_dir) / "test_report.json"
        self.project_analytics.save_analytics_report(report, str(filename))
        self.assertTrue(filename.exists())

        # Проверяем, что файл содержит правильные данные
        with open(filename, "r", encoding="utf-8") as f:
            loaded_report = json.load(f)
        self.assertEqual(loaded_report["timestamp"], "2023-01-01T00:00:00")
        self.assertEqual(loaded_report["surface_analysis"]["mean_height"], 1.0)


if __name__ == "__main__":
    unittest.main()
