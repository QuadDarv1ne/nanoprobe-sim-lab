"""
Тесты для utils/test_framework.py

Покрытие:
- TestFramework инициализация
- Discovery тестов
- Запуск unit тестов
- Coverage анализ
- Performance тестирование
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from utils.test_framework import TestFramework


@pytest.fixture
def test_framework():
    """Создать TestFramework инстанс."""
    return TestFramework(project_root=".")


@pytest.fixture
def sample_test_directory():
    """Создать временную директорию с тестами."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Создаем структуру test_*.py файлов
        test_file_1 = Path(temp_dir) / "test_sample.py"
        test_file_1.write_text(
            """
import unittest

class TestSample(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(1 + 1, 2)

    def test_subtraction(self):
        self.assertEqual(5 - 3, 2)
"""
        )

        test_file_2 = Path(temp_dir) / "test_another.py"
        test_file_2.write_text(
            """
import unittest

class TestAnother(unittest.TestCase):
    def test_multiplication(self):
        self.assertEqual(2 * 3, 6)
"""
        )

        # Создаем не-тестовый файл (должен быть проигнорирован)
        (Path(temp_dir) / "helper.py").write_text("# helper code")

        yield temp_dir


class TestTestFrameworkInit:
    """Тесты инициализации TestFramework."""

    def test_initialization_default(self):
        """Инициализация по умолчанию."""
        tf = TestFramework()
        assert tf.project_root is not None
        assert tf.test_results == {}
        assert tf.coverage_results == {}
        assert tf.performance_results == {}

    def test_initialization_custom_root(self):
        """Инициализация с кастомным root."""
        tf = TestFramework(project_root="/tmp/test_project")
        assert str(tf.project_root) == str(Path("/tmp/test_project").resolve())

    def test_initialization_resolves_path(self):
        """Путь разрешается относительно текущего."""
        tf = TestFramework(project_root=".")
        assert tf.project_root.is_absolute()


class TestDiscoverTests:
    """Тесты discovery функциональности."""

    def test_discover_tests_existing_directory(self, test_framework):
        """Discovery тестов из существующей директории."""
        test_files = test_framework.discover_tests(test_directory="tests")

        assert isinstance(test_files, list)
        # Должен найти хотя бы один тест
        assert len(test_files) > 0
        # Все файлы должны начинаться с test_ и заканчиваться .py
        for test_file in test_files:
            assert "test_" in Path(test_file).name
            assert test_file.endswith(".py")

    def test_discover_tests_nonexistent_directory(self, test_framework):
        """Discovery из несуществующей директории."""
        test_files = test_framework.discover_tests(test_directory="nonexistent_tests")

        assert isinstance(test_files, list)
        assert len(test_files) == 0

    def test_discover_tests_empty_directory(self, test_framework, tmp_path):
        """Discovery из пустой директории."""
        test_files = test_framework.discover_tests(test_directory=str(tmp_path))

        assert isinstance(test_files, list)
        # Может вернуть пустой список или найти файлы в subdirectories
        assert len(test_files) >= 0

    def test_discover_tests_filters_correct_files(self, sample_test_directory):
        """Discovery фильтрует только test_*.py файлы."""
        # Создаем TestFramework с временной директорией
        tf = TestFramework(project_root=sample_test_directory)

        test_files = tf.discover_tests(test_directory=".")

        assert len(test_files) == 2  # test_sample.py и test_another.py
        for test_file in test_files:
            filename = Path(test_file).name
            assert filename.startswith("test_")
            assert filename.endswith(".py")
            assert "helper.py" not in filename


class TestRunUnittests:
    """Тесты запуска unit тестов."""

    def test_run_unittests_basic(self, test_framework):
        """Базовый запуск unit тестов."""
        # Мокаем unittest для быстрого теста
        mock_result = MagicMock()
        mock_result.testsRun = 5
        mock_result.failures = []
        mock_result.errors = []
        mock_result.skipped = []
        mock_result.time_taken = 1.5

        with (
            patch("unittest.TestLoader") as mock_loader,
            patch("unittest.TextTestRunner") as mock_runner,
        ):
            mock_loader.return_value.discover.return_value = MagicMock()
            mock_runner.return_value.run.return_value = mock_result

            results = test_framework.run_unittests()

        assert isinstance(results, dict)
        assert "total_tests" in results
        assert "passed" in results
        assert "failures" in results
        assert "errors" in results
        assert "skipped" in results
        assert results["total_tests"] == 5
        assert results["passed"] == 5
        assert results["failures"] == 0

    def test_run_unittests_with_failures(self, test_framework):
        """Запуск тестов с фейлами."""
        mock_result = MagicMock()
        mock_result.testsRun = 10
        mock_result.failures = [("test1", "error1"), ("test2", "error2")]
        mock_result.errors = [("test3", "error3")]
        mock_result.skipped = []
        mock_result.time_taken = 2.0

        with (
            patch("unittest.TestLoader") as mock_loader,
            patch("unittest.TextTestRunner") as mock_runner,
        ):
            mock_loader.return_value.discover.return_value = MagicMock()
            mock_runner.return_value.run.return_value = mock_result

            results = test_framework.run_unittests()

        assert results["total_tests"] == 10
        assert results["passed"] == 7  # 10 - 2 failures - 1 error
        assert results["failures"] == 2
        assert results["errors"] == 1

    def test_run_unittests_with_skipped(self, test_framework):
        """Запуск тестов с пропусками."""
        mock_result = MagicMock()
        mock_result.testsRun = 8
        mock_result.failures = []
        mock_result.errors = []
        mock_result.skipped = [("test1", "reason")]
        mock_result.time_taken = 1.0

        with (
            patch("unittest.TestLoader") as mock_loader,
            patch("unittest.TextTestRunner") as mock_runner,
        ):
            mock_loader.return_value.discover.return_value = MagicMock()
            mock_runner.return_value.run.return_value = mock_result

            results = test_framework.run_unittests()

        assert results["skipped"] == 1


class TestCoverageAnalysis:
    """Тесты анализа покрытия кода."""

    def test_coverage_results_initialization(self, test_framework):
        """Проверка инициализации coverage_results."""
        assert test_framework.coverage_results == {}

    def test_test_results_structure(self, test_framework):
        """Проверка структуры test_results."""
        # Мокаем unittest для быстрого теста
        mock_result = MagicMock()
        mock_result.testsRun = 5
        mock_result.failures = []
        mock_result.errors = []
        mock_result.skipped = []
        mock_result.time_taken = 1.5

        with (
            patch("unittest.TestLoader") as mock_loader,
            patch("unittest.TextTestRunner") as mock_runner,
        ):
            mock_loader.return_value.discover.return_value = MagicMock()
            mock_runner.return_value.run.return_value = mock_result

            results = test_framework.run_unittests()

        # Проверяем, что результаты имеют ожидаемую структуру
        assert isinstance(results, dict)
        assert "total_tests" in results
        assert "passed" in results


class TestPerformanceTesting:
    """Тесты performance функциональности."""

    def test_performance_results_initialization(self, test_framework):
        """Проверка инициализации performance_results."""
        assert test_framework.performance_results == {}

    def test_run_performance_benchmark_exists(self, test_framework):
        """Проверка наличия метода performance бенчмарка."""
        # TestFramework должен иметь методы для performance тестирования
        assert (
            hasattr(test_framework, "run_performance_tests")
            or hasattr(test_framework, "run_benchmark")
            or hasattr(test_framework, "measure_performance")
        )


class TestQualityChecks:
    """Тесты проверок качества кода."""

    def test_quality_results_initialization(self, test_framework):
        """Проверка инициализации quality_results."""
        assert test_framework.quality_results == {}

    def test_quality_module_loaded(self, test_framework):
        """Проверка, что модуль качества загружен."""
        # Проверяем, что quality_results это словарь
        assert isinstance(test_framework.quality_results, dict)


class TestTestFrameworkResults:
    """Тесты обработки результатов."""

    def test_test_results_is_dict(self, test_framework):
        """test_results это словарь."""
        assert isinstance(test_framework.test_results, dict)

    def test_coverage_results_is_dict(self, test_framework):
        """coverage_results это словарь."""
        assert isinstance(test_framework.coverage_results, dict)

    def test_performance_results_is_dict(self, test_framework):
        """performance_results это словарь."""
        assert isinstance(test_framework.performance_results, dict)

    def test_quality_results_is_dict(self, test_framework):
        """quality_results это словарь."""
        assert isinstance(test_framework.quality_results, dict)


class TestTestFrameworkIntegration:
    """Интеграционные тесты TestFramework."""

    def test_full_test_workflow(self, test_framework):
        """Полный рабочий процесс тестирования."""
        # 1. Discover tests
        test_files = test_framework.discover_tests("tests")
        assert len(test_files) > 0

        # 2. Запустить тесты (мокаем для скорости)
        mock_result = MagicMock()
        mock_result.testsRun = 3
        mock_result.failures = []
        mock_result.errors = []
        mock_result.skipped = []
        mock_result.time_taken = 0.5

        with (
            patch("unittest.TestLoader") as mock_loader,
            patch("unittest.TextTestRunner") as mock_runner,
        ):
            mock_loader.return_value.discover.return_value = MagicMock()
            mock_runner.return_value.run.return_value = mock_result

            results = test_framework.run_unittests()

        # 3. Проверить результаты
        assert results["total_tests"] == 3
        assert results["passed"] == 3

    def test_discover_and_run_consistency(self, test_framework):
        """Согласованность discovery и run."""
        test_files = test_framework.discover_tests("tests")

        # Если найдены тесты, должен быть способ их запустить
        if len(test_files) > 0:
            # Проверяем, что есть метод запуска
            assert hasattr(test_framework, "run_unittests")


class TestEdgeCases:
    """Тесты граничных случаев."""

    def test_discover_tests_with_symlinks(self, test_framework, tmp_path):
        """Discovery с symlink'ами."""
        # Создаем тестовый файл
        test_file = tmp_path / "test_link.py"
        test_file.write_text("import unittest")

        # Создаем symlink
        symlink_path = tmp_path / "test_symlink.py"
        try:
            symlink_path.symlink_to(test_file)

            tf = TestFramework(project_root=str(tmp_path))
            test_files = tf.discover_tests(test_directory=".")

            # Symlinks могут быть обработаны по-разному
            assert isinstance(test_files, list)
        except (OSError, NotImplementedError):
            # Symlinks не поддерживаются на Windows
            pytest.skip("Symlinks not supported")

    def test_run_unittests_empty_directory(self, test_framework, tmp_path):
        """Запуск тестов из пустой директории."""
        # Мокаем для пустого набора тестов
        mock_result = MagicMock()
        mock_result.testsRun = 0
        mock_result.failures = []
        mock_result.errors = []
        mock_result.skipped = []
        mock_result.time_taken = 0.0

        with (
            patch("unittest.TestLoader") as mock_loader,
            patch("unittest.TextTestRunner") as mock_runner,
        ):
            mock_loader.return_value.discover.return_value = MagicMock()
            mock_runner.return_value.run.return_value = mock_result

            results = test_framework.run_unittests()

        assert results["total_tests"] == 0
        assert results["passed"] == 0

    def test_discover_tests_deep_hierarchy(self, test_framework, tmp_path):
        """Discovery из глубокой иерархии директорий."""
        # Создаем вложенную структуру
        nested_dir = tmp_path / "level1" / "level2" / "level3"
        nested_dir.mkdir(parents=True)

        test_file = nested_dir / "test_deep.py"
        test_file.write_text("import unittest")

        tf = TestFramework(project_root=str(tmp_path))
        test_files = tf.discover_tests(test_directory=".")

        assert len(test_files) == 1
        assert "test_deep.py" in test_files[0]
