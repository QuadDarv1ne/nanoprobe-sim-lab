"""Тесты для AI/ML анализатора дефектов."""

import unittest
import tempfile
import shutil
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from utils.ai.defect_analyzer import AdvancedDefectAnalyzer, DefectDetector


class TestDefectDetector(unittest.TestCase):
    """Тесты для базового детектора дефектов"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.test_image = np.random.normal(128, 10, (128, 128)).astype(np.uint8)

    def test_isolation_forest_detector(self):
        """Тест детектора Isolation Forest"""
        detector = DefectDetector('isolation_forest')
        result = detector.detect_defects(self.test_image)

        self.assertIn('defects', result)
        self.assertIn('defects_count', result)

    def test_kmeans_detector(self):
        """Тест детектора K-Means"""
        detector = DefectDetector('kmeans')
        result = detector.detect_defects(self.test_image)

        self.assertIn('defects', result)

    def test_invalid_model(self):
        """Тест неверной модели"""
        with self.assertRaises(ValueError):
            DefectDetector('invalid_model')


class TestAdvancedDefectAnalyzer(unittest.TestCase):
    """Тесты для продвинутого анализатора дефектов"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image = np.random.normal(128, 10, (128, 128)).astype(np.uint8)
        self.analyzer = AdvancedDefectAnalyzer(confidence_threshold=0.5)
        self.analyzer.output_dir = Path(self.temp_dir)

    def tearDown(self):
        """Очистка после теста"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ensemble_detect(self):
        """Тест ансамблевого детектирования"""
        result = self.analyzer.ensemble_detect(self.test_image)

        self.assertIn('defects', result)
        self.assertIn('defects_count', result)
        self.assertIn('if_defects_count', result)
        self.assertIn('km_defects_count', result)
        self.assertTrue(result['ensemble'])

    def test_analyze_with_stats(self):
        """Тест анализа со статистикой"""
        result = self.analyzer.analyze_with_stats(self.test_image)

        self.assertIn('statistics', result)
        stats = result['statistics']

        self.assertIn('total_area', stats)
        self.assertIn('defect_density', stats)
        self.assertIn('severity', stats)
        self.assertIn('defect_types', stats)

    def test_generate_defect_map(self):
        """Тест генерации карты дефектов"""
        defects = [
            {'x': 50, 'y': 50, 'width': 20, 'height': 20},
            {'x': 100, 'y': 100, 'width': 15, 'height': 15},
        ]

        defect_map = self.analyzer.generate_defect_map(self.test_image, defects)

        self.assertEqual(defect_map.shape, self.test_image.shape)
        self.assertEqual(defect_map.dtype, np.uint8)

    def test_save_analysis_report(self):
        """Тест сохранения отчёта"""
        result = self.analyzer.analyze_with_stats(self.test_image)
        report_path = self.analyzer.save_analysis_report(result, "test_image.png")

        self.assertTrue(Path(report_path).exists())

    def test_generate_recommendations_high_severity(self):
        """Тест рекомендаций - высокая серьёзность"""
        result = {
            'statistics': {
                'severity': 'high',
                'defect_density': 6.0,
                'defect_types': {'scratch': 3, 'crack': 1},
            }
        }

        recommendations = self.analyzer._generate_recommendations(result)

        self.assertGreater(len(recommendations), 0)
        self.assertTrue(any('критический' in r.lower() for r in recommendations))

    def test_generate_recommendations_low_severity(self):
        """Тест рекомендаций - низкая серьёзность"""
        result = {
            'statistics': {
                'severity': 'low',
                'defect_density': 0.5,
                'defect_types': {},
            }
        }

        recommendations = self.analyzer._generate_recommendations(result)

        self.assertGreater(len(recommendations), 0)

    def test_combine_detections_overlapping(self):
        """Тест объединения перекрывающихся детектирований"""
        defects1 = [{'x': 50, 'y': 50, 'width': 20, 'height': 20, 'confidence': 0.9, 'type': 'pit'}]
        defects2 = [{'x': 55, 'y': 55, 'width': 18, 'height': 18, 'confidence': 0.85, 'type': 'pit'}]

        combined = self.analyzer._combine_detections(defects1, defects2)

        # Должно быть одно объединённое детектирование
        self.assertEqual(len(combined), 1)
        self.assertAlmostEqual(combined[0]['x'], 52.5, places=1)
        self.assertAlmostEqual(combined[0]['y'], 52.5, places=1)

    def test_combine_detections_non_overlapping(self):
        """Тест объединения неперекрывающихся детектирований"""
        defects1 = [{'x': 20, 'y': 20, 'width': 10, 'height': 10, 'confidence': 0.9, 'type': 'pit'}]
        defects2 = [{'x': 100, 'y': 100, 'width': 10, 'height': 10, 'confidence': 0.85, 'type': 'pit'}]

        combined = self.analyzer._combine_detections(defects1, defects2)

        # Должно быть два отдельных детектирования
        self.assertEqual(len(combined), 2)

    def test_confidence_threshold_filtering(self):
        """Тест фильтрации по порогу уверенности"""
        analyzer_high = AdvancedDefectAnalyzer(confidence_threshold=0.9)
        analyzer_low = AdvancedDefectAnalyzer(confidence_threshold=0.3)

        # Создаём тестовое изображение с явными дефектами
        test_img = np.random.normal(128, 10, (128, 128)).astype(np.uint8)
        test_img[50:60, 50:60] = 200  # Явный дефект

        result_high = analyzer_high.ensemble_detect(test_img)
        result_low = analyzer_low.ensemble_detect(test_img)

        # Низкий порог должен найти больше дефектов
        self.assertGreaterEqual(result_low['defects_count'], result_high['defects_count'])


@unittest.skipUnless(PIL_AVAILABLE, "PIL required for image tests")
class TestDefectAnalyzerWithImages(unittest.TestCase):
    """Тесты с реальными изображениями"""

    def setUp(self):
        """Подготовка тестовых изображений"""
        self.temp_dir = tempfile.mkdtemp()
        self.analyzer = AdvancedDefectAnalyzer(confidence_threshold=0.5)
        self.analyzer.output_dir = Path(self.temp_dir)

        # Создание тестового изображения с дефектами
        self.test_image = np.random.normal(128, 10, (256, 256)).astype(np.uint8)
        self.test_image[50:70, 100:150] = 200  # Выступ
        self.test_image[150:155, 50:200] = 50  # Царапина
        self.test_image[200:220, 200:220] = 30  # Впадина

        self.test_path = Path(self.temp_dir) / "test_surface.png"
        Image.fromarray(self.test_image).save(str(self.test_path))

    def tearDown(self):
        """Очистка"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_analysis_pipeline(self):
        """Тест полного цикла анализа"""
        result = self.analyzer.analyze_with_stats(self.test_image)

        self.assertGreater(result['defects_count'], 0)
        self.assertIn('statistics', result)
        self.assertIn('severity', result['statistics'])

    def test_report_generation(self):
        """Тест генерации отчёта"""
        result = self.analyzer.analyze_with_stats(self.test_image)
        report_path = self.analyzer.save_analysis_report(result, str(self.test_path))

        self.assertTrue(Path(report_path).exists())

        # Проверка содержимого отчёта
        import json
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)

        self.assertIn('id', report)
        self.assertIn('timestamp', report)
        self.assertIn('analysis', report)
        self.assertIn('recommendations', report)


if __name__ == '__main__':
    unittest.main()
