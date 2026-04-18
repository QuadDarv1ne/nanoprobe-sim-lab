"""Тесты для SignalClassifier."""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.ml.signal_classifier import SignalClassifier, SUPPORTED_CLASSES


class TestSignalClassifierInit(unittest.TestCase):
    """Тесты инициализации SignalClassifier."""

    def test_init_no_model(self):
        """Тест инициализации без модели."""
        classifier = SignalClassifier()
        self.assertFalse(classifier.available)
        self.assertIsNone(classifier._interpreter)

    def test_init_with_nonexistent_model(self):
        """Тест инициализации с несуществующей моделью."""
        classifier = SignalClassifier(model_path="/nonexistent/model.tflite")
        # Должен упасть gracefully
        self.assertFalse(classifier.available)


class TestSignalClassifierClassification(unittest.TestCase):
    """Тесты классификации сигналов."""

    def setUp(self):
        """Подготовка тестового окружения."""
        self.classifier = SignalClassifier()

    def test_classify_empty_samples(self):
        """Тест классификации пустых сэмплов."""
        samples = np.array([], dtype=np.complex64)
        label, confidence = self.classifier.classify(samples, 2400000)
        self.assertEqual(label, "unknown")
        self.assertEqual(confidence, 0.0)

    def test_classify_none_samples(self):
        """Тест классификации None сэмплов."""
        label, confidence = self.classifier.classify(None, 2400000)
        self.assertEqual(label, "unknown")
        self.assertEqual(confidence, 0.0)

    def test_classify_noise_signal(self):
        """Тест классификации шума."""
        # Шум - низкая энергия
        samples = np.random.randn(1000).astype(np.complex64) * 0.001
        label, confidence = self.classifier.classify(samples, 2400000)
        self.assertIn(label, SUPPORTED_CLASSES)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_classify_strong_signal(self):
        """Тест классификации сильного сигнала."""
        # Сильный сигнал - высокая энергия
        samples = np.random.randn(1000).astype(np.complex64) * 10.0
        label, confidence = self.classifier.classify(samples, 2400000)
        self.assertIn(label, SUPPORTED_CLASSES)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_classify_fm_signal(self):
        """Тест классификации FM сигнала."""
        # FM сигнал - широкая полоса
        t = np.linspace(0, 0.01, 24000)
        samples = (np.exp(1j * 2 * np.pi * 5000 * np.sin(2 * np.pi * 1000 * t)) * 0.5 +
                   np.random.randn(len(t)).astype(np.complex64) * 0.1)
        label, confidence = self.classifier.classify(samples, 2400000)
        self.assertIn(label, SUPPORTED_CLASSES)

    def test_classify_cw_signal(self):
        """Тест классификации CW сигнала."""
        # CW сигнал - узкая полоса
        t = np.linspace(0, 0.01, 24000)
        samples = (np.exp(1j * 2 * np.pi * 700 * t).astype(np.complex64) * 0.8 +
                   np.random.randn(len(t)).astype(np.complex64) * 0.1)
        label, confidence = self.classifier.classify(samples, 2400000)
        self.assertIn(label, SUPPORTED_CLASSES)


class TestSignalClassifierHeuristic(unittest.TestCase):
    """Тесты эвристической классификации."""

    def setUp(self):
        """Подготовка тестового окружения."""
        self.classifier = SignalClassifier()

    def test_heuristic_noise_detection(self):
        """Тест детекции шума эвристиками."""
        # Шум ниже порога
        samples = np.random.randn(1000).astype(np.complex64) * 0.001
        label, confidence = self.classifier._classify_heuristic(samples, 2400000)
        self.assertEqual(label, "noise")
        self.assertGreater(confidence, 0.0)

    def test_heuristic_signal_detection(self):
        """Тест детекции сигнала эвристиками."""
        # Сигнал выше порога
        samples = np.random.randn(1000).astype(np.complex64) * 0.5
        label, confidence = self.classifier._classify_heuristic(samples, 2400000)
        self.assertNotEqual(label, "noise")
        self.assertGreater(confidence, 0.0)

    def test_spectral_width_calculation(self):
        """Тест расчёта ширины спектра."""
        # Узкий сигнал
        samples = np.exp(1j * 2 * np.pi * 0.1 * np.arange(1000)).astype(np.complex64)
        width = self.classifier._calculate_spectral_width(samples)
        self.assertGreater(width, 0)

    def test_energy_calculation(self):
        """Тест расчёта энергии."""
        samples = np.random.randn(1000).astype(np.complex64) * 2.0
        energy = self.classifier._calculate_energy(samples)
        self.assertGreater(energy, 0)


class TestSignalClassifierPreprocessing(unittest.TestCase):
    """Тесты предобработки сигналов."""

    def setUp(self):
        """Подготовка тестового окружения."""
        self.classifier = SignalClassifier()

    def test_preprocess_samples(self):
        """Тест предобработки сэмплов."""
        samples = np.random.randn(1024).astype(np.complex64)
        result = self.classifier._preprocess_samples(samples, 2400000)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, np.ndarray)

    def test_preprocess_small_samples(self):
        """Тест предобработки малых сэмплов."""
        samples = np.random.randn(100).astype(np.complex64)
        result = self.classifier._preprocess_samples(samples, 2400000)
        self.assertIsNotNone(result)


class TestSignalClassifierSupportedClasses(unittest.TestCase):
    """Тесты поддерживаемых классов."""

    def test_supported_classes_content(self):
        """Тест содержимого SUPPORTED_CLASSES."""
        self.assertIn("sstv", SUPPORTED_CLASSES)
        self.assertIn("cw", SUPPORTED_CLASSES)
        self.assertIn("rtty", SUPPORTED_CLASSES)
        self.assertIn("fm", SUPPORTED_CLASSES)
        self.assertIn("noise", SUPPORTED_CLASSES)
        self.assertIn("unknown", SUPPORTED_CLASSES)

    def test_supported_classes_count(self):
        """Тест количества поддерживаемых классов."""
        self.assertEqual(len(SUPPORTED_CLASSES), 6)


class TestSignalClassifierEdgeCases(unittest.TestCase):
    """Тесты граничных случаев."""

    def setUp(self):
        """Подготовка тестового окружения."""
        self.classifier = SignalClassifier()

    def test_classify_single_sample(self):
        """Тест классификации одного сэмпла."""
        samples = np.array([1.0 + 1.0j], dtype=np.complex64)
        label, confidence = self.classifier.classify(samples, 2400000)
        self.assertIn(label, SUPPORTED_CLASSES)

    def test_classify_very_long_signal(self):
        """Тест классификации очень длинного сигнала."""
        samples = np.random.randn(100000).astype(np.complex64) * 0.5
        label, confidence = self.classifier.classify(samples, 2400000)
        self.assertIn(label, SUPPORTED_CLASSES)

    def test_classify_zero_samples(self):
        """Тест классификации нулевых сэмплов."""
        samples = np.zeros(1000, dtype=np.complex64)
        label, confidence = self.classifier.classify(samples, 2400000)
        # Нулевые сэмплы должны быть классифицированы как шум или неизвестно
        self.assertIn(label, ["noise", "unknown"])

    def test_classify_different_sample_rates(self):
        """Тест классификации с разными частотами дискретизации."""
        samples = np.random.randn(1000).astype(np.complex64) * 0.5
        for rate in [48000, 240000, 2400000]:
            label, confidence = self.classifier.classify(samples, rate)
            self.assertIn(label, SUPPORTED_CLASSES)


if __name__ == "__main__":
    unittest.main()
