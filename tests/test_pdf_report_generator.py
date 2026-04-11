"""Тесты для PDF report generator."""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.reporting.pdf_report_generator import ScientificPDFReport, generate_pdf_report


class TestPDFReportGenerator(unittest.TestCase):
    """Тесты для генератора PDF отчётов"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "pdf_reports"

    def tearDown(self):
        """Очистка после теста"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_comparison_report_generation(self):
        """Тест генерации отчёта о сравнении поверхностей"""
        generator = ScientificPDFReport(str(self.output_dir))

        comparison_data = {
            "ssim": 0.92,
            "psnr": 35.5,
            "similarity": 0.88,
            "mse": 0.005,
            "pearson": 0.95,
            "mean_diff": 0.03,
            "max_diff": 0.12,
            "std_diff": 0.02,
        }

        filepath = generator.generate_comparison_report(
            comparison_data, title="Тестовое сравнение поверхностей"
        )

        self.assertTrue(Path(filepath).exists())
        self.assertTrue(filepath.endswith(".pdf"))

    def test_simulation_report_generation(self):
        """Тест генерации отчёта о симуляции"""
        generator = ScientificPDFReport(str(self.output_dir))

        simulation_data = {
            "simulation_type": "SPM Contact Mode",
            "scan_size": "10x10 мкм",
            "resolution": "512x512",
            "probe_radius": "10 нм",
            "scan_speed": "1 Гц",
            "duration": "45.3 с",
            "points_count": 262144,
            "data_size": "2.1 MB",
            "results_summary": "Симуляция завершена успешно. Получены данные топографии поверхности.",
        }

        filepath = generator.generate_simulation_report(
            simulation_data, title="Тестовая симуляция СЗМ"
        )

        self.assertTrue(Path(filepath).exists())
        self.assertTrue(filepath.endswith(".pdf"))

    def test_batch_report_generation(self):
        """Тест генерации отчёта о пакетной обработке"""
        generator = ScientificPDFReport(str(self.output_dir))

        batch_data = {
            "total_items": 100,
            "processed_items": 98,
            "success_count": 95,
            "error_count": 3,
            "duration": "15 мин 32 с",
            "error_details": [
                "Файл не найден: sample_042.dat",
                "Ошибка чтения: sample_067.dat",
                "Неверный формат: sample_089.dat",
            ],
        }

        filepath = generator.generate_batch_report(batch_data, title="Тестовая пакетная обработка")

        self.assertTrue(Path(filepath).exists())
        self.assertTrue(filepath.endswith(".pdf"))

    def test_comparison_conclusions_high_similarity(self):
        """Тест выводов для высокого сходства"""
        generator = ScientificPDFReport(str(self.output_dir))

        data = {"similarity": 0.92, "ssim": 0.95}
        content = generator._create_comparison_conclusions(data)

        self.assertGreater(len(content), 0)

    def test_comparison_conclusions_low_similarity(self):
        """Тест выводов для низкого сходства"""
        generator = ScientificPDFReport(str(self.output_dir))

        data = {"similarity": 0.35, "ssim": 0.40}
        content = generator._create_comparison_conclusions(data)

        self.assertGreater(len(content), 0)

    def test_difference_analysis_identical(self):
        """Тест анализа различий - идентичные поверхности"""
        generator = ScientificPDFReport(str(self.output_dir))

        data = {"mean_diff": 0.02, "max_diff": 0.05, "std_diff": 0.01}
        content = generator._create_difference_analysis(data)

        self.assertGreater(len(content), 0)

    def test_difference_analysis_different(self):
        """Тест анализа различий - различные поверхности"""
        generator = ScientificPDFReport(str(self.output_dir))

        data = {"mean_diff": 0.25, "max_diff": 0.45, "std_diff": 0.15}
        content = generator._create_difference_analysis(data)

        self.assertGreater(len(content), 0)

    def test_batch_statistics_with_errors(self):
        """Тест статистики пакетной обработки с ошибками"""
        generator = ScientificPDFReport(str(self.output_dir))

        data = {
            "total_items": 100,
            "success_count": 85,
            "error_details": ["Error 1", "Error 2"],
        }
        content = generator._create_batch_statistics(data)

        self.assertGreater(len(content), 0)
        self.assertIn("85.0%", str(content))

    def test_global_generate_pdf_report_comparison(self):
        """Тест глобальной функции для comparison отчёта"""
        filepath = generate_pdf_report(
            "comparison",
            {
                "ssim": 0.85,
                "psnr": 32.0,
                "similarity": 0.80,
                "mse": 0.01,
                "pearson": 0.88,
                "mean_diff": 0.05,
                "max_diff": 0.15,
                "std_diff": 0.03,
            },
            output_dir=str(self.output_dir),
        )

        self.assertTrue(Path(filepath).exists())

    def test_global_generate_pdf_report_simulation(self):
        """Тест глобальной функции для simulation отчёта"""
        filepath = generate_pdf_report(
            "simulation",
            {
                "simulation_type": "Test",
                "duration": "10 с",
                "points_count": 1000,
                "data_size": "1 MB",
            },
            output_dir=str(self.output_dir),
        )

        self.assertTrue(Path(filepath).exists())

    def test_global_generate_pdf_report_batch(self):
        """Тест глобальной функции для batch отчёта"""
        filepath = generate_pdf_report(
            "batch",
            {"total_items": 50, "success_count": 48, "error_count": 2, "duration": "5 мин"},
            output_dir=str(self.output_dir),
        )

        self.assertTrue(Path(filepath).exists())

    def test_invalid_report_type(self):
        """Тест неверного типа отчёта"""
        with self.assertRaises(ValueError):
            generate_pdf_report("invalid_type", {}, output_dir=str(self.output_dir))


if __name__ == "__main__":
    unittest.main()
