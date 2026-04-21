"""Unit-тесты для модуля визуализации."""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np

# Используем бэкенд Agg для matplotlib, который не требует GUI
matplotlib.use("Agg")

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.visualizer import (
    ImageAnalyzerVisualizer,
    ProjectVisualizer,
    SSTVVisualizer,
    SurfaceVisualizer,
)


class TestSurfaceVisualizer(unittest.TestCase):
    """Тесты для класса SurfaceVisualizer"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.viz = SurfaceVisualizer()
        self.test_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)

    def test_init(self):
        """Тест инициализации визуализатора поверхности"""
        self.assertEqual(self.viz.figsize, (12, 8))

    def test_plot_surface_2d(self):
        """Тест создания 2D визуализации поверхности"""
        fig = self.viz.plot_surface_2d(self.test_data, "Test 2D Surface")
        self.assertIsNotNone(fig)
        # Закрываем фигуру чтобы избежать предупреждений
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_surface_3d(self):
        """Тест создания 3D визуализации поверхности"""
        fig = self.viz.plot_surface_3d(self.test_data, "Test 3D Surface")
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_animate_scan_process(self):
        """Тест создания анимации процесса сканирования"""
        scan_data_list = [
            np.array([[1, 2], [3, 4]]),
            np.array([[2, 3], [4, 5]]),
            np.array([[3, 4], [5, 6]]),
        ]
        ani = self.viz.animate_scan_process(scan_data_list, "Test Animation")
        self.assertIsNotNone(ani)
        # Анимация не требует закрытия, но мы можем проверить, что она создана


class TestImageAnalyzerVisualizer(unittest.TestCase):
    """Тесты для класса ImageAnalyzerVisualizer"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.viz = ImageAnalyzerVisualizer()
        self.test_image = np.random.rand(10, 10)
        self.processed_image = np.random.rand(10, 10)

    def test_init(self):
        """Тест инициализации визуализатора анализа изображений"""
        self.assertEqual(self.viz.figsize, (12, 8))

    def test_plot_comparison(self):
        """Тест сравнения оригинального и обработанного изображений"""
        fig = self.viz.plot_comparison(
            self.test_image,
            self.processed_image,
            "Original",
            "Processed",
        )
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_histograms(self):
        """Тест построения гистограмм изображений"""
        fig = self.viz.plot_histograms(
            self.test_image,
            self.processed_image,
            "Test Histograms",
        )
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_highlight_defects(self):
        """Тест выделения дефектов на изображении"""
        defects_coords = [(2, 2), (5, 5), (8, 8)]
        fig = self.viz.highlight_defects(
            self.test_image,
            defects_coords,
            "Test Defects",
        )
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestSSTVVisualizer(unittest.TestCase):
    """Тесты для класса SSTVVisualizer"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.viz = SSTVVisualizer()
        self.test_image = np.random.rand(20, 20)
        self.test_signal = np.random.rand(1000)

    def test_init(self):
        """Тест инициализации визуализатора SSTV"""
        self.assertEqual(self.viz.figsize, (12, 8))

    def test_plot_decoded_image(self):
        """Тест отображения декодированного SSTV изображения"""
        fig = self.viz.plot_decoded_image(self.test_image, "Test SSTV Image")
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_signal_spectrum(self):
        """Тест отображения спектра сигнала SSTV"""
        fig = self.viz.plot_signal_spectrum(
            self.test_signal,
            sample_rate=44100,
            title="Test Signal Spectrum",
        )
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestProjectVisualizer(unittest.TestCase):
    """Тесты для класса ProjectVisualizer"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.temp_dir = tempfile.mkdtemp()
        self.viz = ProjectVisualizer()
        self.surface_data = np.random.rand(20, 20)
        self.original_image = np.random.rand(20, 20)
        self.processed_image = np.random.rand(20, 20)
        self.sstv_image = np.random.rand(20, 20, 3)

    def tearDown(self):
        """Очистка после тестов"""
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        """Тест инициализации центрального визуализатора проекта"""
        self.assertIsInstance(self.viz.surface_viz, SurfaceVisualizer)
        self.assertIsInstance(self.viz.analyzer_viz, ImageAnalyzerVisualizer)
        self.assertIsInstance(self.viz.sstv_viz, SSTVVisualizer)

    def test_visualize_all_for_report(self):
        """Тест создания полного отчета визуализации"""
        success = self.viz.visualize_all_for_report(
            surface_data=self.surface_data,
            original_image=self.original_image,
            processed_image=self.processed_image,
            sstv_image=self.sstv_image,
            output_dir=self.temp_dir,
        )
        self.assertTrue(success)

        # Проверяем, что файлы были созданы
        output_path = Path(self.temp_dir)
        expected_files = [
            "spm_surface_2d.png",
            "spm_surface_3d.png",
            "image_comparison.png",
            "image_histograms.png",
            "sstv_decoded.png",
        ]
        for filename in expected_files:
            self.assertTrue((output_path / filename).exists(), f"File {filename} not found")


if __name__ == "__main__":
    unittest.main()
