#!/usr/bin/env python
"""
Тесты для utils/performance/performance_benchmark.py

Покрытие:
- BenchmarkResult
- PerformanceComparison
- PerformanceBenchmarkSuite (init, measure, results)
- Краевые случаи
"""

import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.performance.performance_benchmark import (
    BenchmarkResult,
    PerformanceBenchmarkSuite,
    PerformanceComparison,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_output_dir():
    """Создает временную директорию для вывода бенчмарков."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mocked_psutil_process():
    """Мок для psutil.Process."""
    mock_process = MagicMock()
    mock_process.memory_info.return_value = MagicMock(rss=50 * 1024 * 1024)  # 50 MB
    mock_process.cpu_percent.return_value = 5.0
    return mock_process


@pytest.fixture
def mocked_perf_counter():
    """Мок для time.perf_counter с последовательными значениями."""
    with patch("utils.performance.performance_benchmark.time.perf_counter") as mock_pc:
        mock_pc.side_effect = [0.0, 0.5]  # start=0.0, end=0.5 -> delta=0.5
        yield mock_pc


@pytest.fixture
def benchmark_suite(tmp_output_dir, mocked_psutil_process):
    """Создает PerformanceBenchmarkSuite с мокированным процессом."""
    with patch(
        "utils.performance.performance_benchmark.psutil.Process", return_value=mocked_psutil_process
    ):
        suite = PerformanceBenchmarkSuite(output_dir=str(tmp_output_dir))
        suite.current_process = mocked_psutil_process
        yield suite


# ---------------------------------------------------------------------------
# 1. TestBenchmarkResult
# ---------------------------------------------------------------------------


class TestBenchmarkResult:
    """Тесты для dataclass BenchmarkResult."""

    def test_create_result_basic(self):
        """Создание результата с основными полями."""
        result = BenchmarkResult(
            name="test_func",
            execution_time=0.5,
            memory_used_mb=10.0,
            cpu_percent=5.0,
            iterations=10,
            timestamp=datetime.now(timezone.utc),
            parameters={},
        )
        assert result.name == "test_func"
        assert result.execution_time == 0.5
        assert result.memory_used_mb == 10.0
        assert result.cpu_percent == 5.0
        assert result.iterations == 10

    def test_create_result_with_all_fields(self):
        """Создание результата со всеми полями включая опциональные."""
        ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = BenchmarkResult(
            name="full_test",
            execution_time=1.23,
            memory_used_mb=25.5,
            cpu_percent=10.0,
            iterations=50,
            timestamp=ts,
            parameters={"key": "value"},
            result_value="success",
        )
        assert result.name == "full_test"
        assert result.execution_time == 1.23
        assert result.memory_used_mb == 25.5
        assert result.cpu_percent == 10.0
        assert result.iterations == 50
        assert result.timestamp == ts
        assert result.parameters == {"key": "value"}
        assert result.result_value == "success"

    def test_result_timestamp_auto(self):
        """Timestamp автоматически устанавливается при создании."""
        before = datetime.now(timezone.utc)
        result = BenchmarkResult(
            name="auto_ts",
            execution_time=0.1,
            memory_used_mb=1.0,
            cpu_percent=1.0,
            iterations=1,
            timestamp=datetime.now(timezone.utc),
            parameters={},
        )
        after = datetime.now(timezone.utc)
        assert before <= result.timestamp <= after

    def test_result_default_values(self):
        """Проверка значений по умолчанию для result_value."""
        result = BenchmarkResult(
            name="default_test",
            execution_time=0.0,
            memory_used_mb=0.0,
            cpu_percent=0.0,
            iterations=1,
            timestamp=datetime.now(timezone.utc),
            parameters={},
        )
        assert result.result_value is None
        assert result.parameters == {}


# ---------------------------------------------------------------------------
# 2. TestPerformanceComparison
# ---------------------------------------------------------------------------


class TestPerformanceComparison:
    """Тесты для dataclass PerformanceComparison."""

    def test_create_comparison(self):
        """Создание объекта сравнения."""
        ts = datetime.now(timezone.utc)
        baseline = BenchmarkResult(
            name="baseline",
            execution_time=1.0,
            memory_used_mb=20.0,
            cpu_percent=10.0,
            iterations=10,
            timestamp=ts,
            parameters={},
        )
        comparison = BenchmarkResult(
            name="improved",
            execution_time=0.8,
            memory_used_mb=15.0,
            cpu_percent=8.0,
            iterations=10,
            timestamp=ts,
            parameters={},
        )
        perf_comp = PerformanceComparison(
            test_name="baseline_vs_improved",
            baseline_result=baseline,
            comparison_result=comparison,
            improvement_percent=20.0,
            is_significant=True,
        )
        assert perf_comp.test_name == "baseline_vs_improved"
        assert perf_comp.improvement_percent == 20.0
        assert perf_comp.is_significant is True
        assert perf_comp.baseline_result.execution_time == 1.0
        assert perf_comp.comparison_result.execution_time == 0.8

    def test_comparison_improvement_positive(self):
        """Сравнение с положительным улучшением (ускорение)."""
        ts = datetime.now(timezone.utc)
        baseline = BenchmarkResult(
            name="slow",
            execution_time=2.0,
            memory_used_mb=30.0,
            cpu_percent=20.0,
            iterations=10,
            timestamp=ts,
            parameters={},
        )
        faster = BenchmarkResult(
            name="fast",
            execution_time=1.0,
            memory_used_mb=20.0,
            cpu_percent=10.0,
            iterations=10,
            timestamp=ts,
            parameters={},
        )
        comp = PerformanceComparison(
            test_name="slow_vs_fast",
            baseline_result=baseline,
            comparison_result=faster,
            improvement_percent=50.0,
            is_significant=True,
        )
        assert comp.improvement_percent > 0
        assert comp.is_significant is True

    def test_comparison_improvement_negative(self):
        """Сравнение с отрицательным улучшением (замедление)."""
        ts = datetime.now(timezone.utc)
        baseline = BenchmarkResult(
            name="fast",
            execution_time=1.0,
            memory_used_mb=10.0,
            cpu_percent=5.0,
            iterations=10,
            timestamp=ts,
            parameters={},
        )
        slower = BenchmarkResult(
            name="slow",
            execution_time=3.0,
            memory_used_mb=40.0,
            cpu_percent=30.0,
            iterations=10,
            timestamp=ts,
            parameters={},
        )
        comp = PerformanceComparison(
            test_name="fast_vs_slow",
            baseline_result=baseline,
            comparison_result=slower,
            improvement_percent=-200.0,
            is_significant=True,
        )
        assert comp.improvement_percent < 0
        assert comp.is_significant is True


# ---------------------------------------------------------------------------
# 3. TestPerformanceBenchmarkSuiteInit
# ---------------------------------------------------------------------------


class TestPerformanceBenchmarkSuiteInit:
    """Тесты инициализации PerformanceBenchmarkSuite."""

    def test_init_creates_output_dir(self, tmp_output_dir):
        """Инициализация создает директорию вывода."""
        nested = tmp_output_dir / "benchmarks_output"
        assert not nested.exists()
        with patch("utils.performance.performance_benchmark.psutil.Process"):
            PerformanceBenchmarkSuite(output_dir=str(nested))
        assert nested.exists()
        assert nested.is_dir()

    def test_init_empty_results(self):
        """Инициализация с пустым списком результатов."""
        with patch("utils.performance.performance_benchmark.psutil.Process"):
            suite = PerformanceBenchmarkSuite()
        assert suite.results == []
        assert isinstance(suite.results, list)

    def test_init_empty_comparisons(self):
        """Инициализация с пустым списком сравнений."""
        with patch("utils.performance.performance_benchmark.psutil.Process"):
            suite = PerformanceBenchmarkSuite()
        assert suite.comparisons == []
        assert isinstance(suite.comparisons, list)

    def test_init_monitoring_false(self):
        """Инициализация с monitoring=False по умолчанию."""
        with patch("utils.performance.performance_benchmark.psutil.Process"):
            suite = PerformanceBenchmarkSuite()
        assert suite.monitoring is False

    def test_init_monitoring_data_structure(self):
        """Инициализация создает правильную структуру monitoring_data."""
        with patch("utils.performance.performance_benchmark.psutil.Process"):
            suite = PerformanceBenchmarkSuite()
        expected_keys = {"timestamps", "cpu_percent", "memory_mb", "disk_io", "network_io"}
        assert set(suite.monitoring_data.keys()) == expected_keys
        assert suite.monitoring_data["timestamps"] == []
        assert suite.monitoring_data["cpu_percent"] == []
        assert suite.monitoring_data["memory_mb"] == []
        assert suite.monitoring_data["disk_io"] == []
        assert suite.monitoring_data["network_io"] == []


# ---------------------------------------------------------------------------
# 4. TestMeasurePerformance
# ---------------------------------------------------------------------------


class TestMeasurePerformance:
    """Тесты метода measure_performance."""

    def test_measure_basic_function(self, benchmark_suite, mocked_perf_counter):
        """Измерение базовой функции."""

        def simple_func():
            return 42

        with patch("utils.performance.performance_benchmark.gc.collect"):
            results = benchmark_suite.measure_performance(simple_func, iterations=1, warmup=0)

        assert len(results) == 1
        assert results[0].name == "simple_func"
        assert results[0].result_value == 42
        assert results[0].execution_time == 0.5  # из мока perf_counter

    def test_measure_with_iterations(self, benchmark_suite):
        """Измерение с несколькими итерациями."""

        def dummy():
            pass

        with (
            patch("utils.performance.performance_benchmark.gc.collect"),
            patch(
                "utils.performance.performance_benchmark.time.perf_counter",
                side_effect=[0.0, 0.1] * 3,
            ),
        ):
            results = benchmark_suite.measure_performance(dummy, iterations=3, warmup=0)

        assert len(results) == 3
        for r in results:
            assert r.execution_time == 0.1

    def test_measure_with_warmup(self, benchmark_suite):
        """Проверка вызова warmup итераций."""
        call_count = 0

        def warmable():
            nonlocal call_count
            call_count += 1

        warmup_count = 3
        iter_count = 2

        with (
            patch("utils.performance.performance_benchmark.gc.collect"),
            patch(
                "utils.performance.performance_benchmark.time.perf_counter",
                side_effect=[0.0, 0.1] * iter_count,
            ),
        ):
            benchmark_suite.measure_performance(
                warmable, iterations=iter_count, warmup=warmup_count
            )

        # warmup + iterations
        assert call_count == warmup_count + iter_count

    def test_measure_with_args_kwargs(self, benchmark_suite):
        """Измерение функции с аргументами и ключевыми аргументами."""
        captured_args = None
        captured_kwargs = None

        def func_with_params(*args, **kwargs):
            nonlocal captured_args, captured_kwargs
            captured_args = args
            captured_kwargs = kwargs
            return sum(args) + sum(kwargs.values())

        with (
            patch("utils.performance.performance_benchmark.gc.collect"),
            patch(
                "utils.performance.performance_benchmark.time.perf_counter", side_effect=[0.0, 0.2]
            ),
        ):
            results = benchmark_suite.measure_performance(
                func_with_params, 1, 2, 3, a=10, b=20, iterations=1, warmup=0
            )

        assert len(results) == 1
        assert captured_args == (1, 2, 3)
        assert captured_kwargs == {"a": 10, "b": 20}
        assert results[0].result_value == 36  # 1+2+3+10+20
        assert results[0].parameters["args_count"] == 3
        assert results[0].parameters["kwargs_count"] == 2

    def test_measure_returns_list_of_benchmark_result(self, benchmark_suite):
        """measure_performance возвращает список BenchmarkResult."""

        def ret_val():
            return "ok"

        # Нужно 4 значения: start/end для каждой из 2 итераций
        with (
            patch("utils.performance.performance_benchmark.gc.collect"),
            patch(
                "utils.performance.performance_benchmark.time.perf_counter",
                side_effect=[0.0, 0.3, 0.3, 0.6],
            ),
        ):
            results = benchmark_suite.measure_performance(ret_val, iterations=2, warmup=0)

        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, BenchmarkResult)


# ---------------------------------------------------------------------------
# 5. TestResultsManagement
# ---------------------------------------------------------------------------


class TestResultsManagement:
    """Тесты управления результатами."""

    def test_results_initially_empty(self, benchmark_suite):
        """Результаты изначально пусты."""
        assert len(benchmark_suite.results) == 0

    def test_results_append(self, benchmark_suite):
        """Добавление результата в список."""
        ts = datetime.now(timezone.utc)
        result = BenchmarkResult(
            name="append_test",
            execution_time=0.1,
            memory_used_mb=1.0,
            cpu_percent=1.0,
            iterations=1,
            timestamp=ts,
            parameters={},
        )
        benchmark_suite.results.append(result)
        assert len(benchmark_suite.results) == 1
        assert benchmark_suite.results[0].name == "append_test"

    def test_results_multiple_additions(self, benchmark_suite):
        """Несколько добавлений результатов."""
        for i in range(5):
            ts = datetime.now(timezone.utc)
            result = BenchmarkResult(
                name=f"test_{i}",
                execution_time=float(i),
                memory_used_mb=float(i),
                cpu_percent=float(i),
                iterations=1,
                timestamp=ts,
                parameters={},
            )
            benchmark_suite.results.append(result)
        assert len(benchmark_suite.results) == 5
        assert benchmark_suite.results[0].name == "test_0"
        assert benchmark_suite.results[4].name == "test_4"

    def test_get_average_performance(self, benchmark_suite):
        """Расчет средней производительности."""
        for i in range(3):
            ts = datetime.now(timezone.utc)
            result = BenchmarkResult(
                name=f"avg_test_{i}",
                execution_time=float(i + 1),  # 1.0, 2.0, 3.0
                memory_used_mb=float(i + 1) * 10,
                cpu_percent=float(i + 1) * 5,
                iterations=1,
                timestamp=ts,
                parameters={},
            )
            benchmark_suite.results.append(result)

        stats = benchmark_suite._calculate_aggregated_stats()
        assert stats["execution_time"]["avg"] == 2.0  # (1+2+3)/3
        assert stats["memory_usage"]["avg_mb"] == 20.0  # (10+20+30)/3
        assert stats["cpu_usage"]["avg_percent"] == 10.0  # (5+10+15)/3


# ---------------------------------------------------------------------------
# 6. TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Тесты краевых случаев."""

    def test_benchmark_simple_function(self, benchmark_suite):
        """Бенчмарк простой функции."""

        def add(a, b):
            return a + b

        with (
            patch.object(benchmark_suite.current_process, "memory_info") as mock_mem,
            patch.object(benchmark_suite.current_process, "cpu_percent", return_value=2.0),
            patch(
                "utils.performance.performance_benchmark.time.perf_counter", side_effect=[0.0, 0.01]
            ),
            patch("utils.performance.performance_benchmark.gc.collect"),
        ):
            mock_mem.return_value = MagicMock(rss=10 * 1024 * 1024)
            results = benchmark_suite.measure_performance(add, 2, 3, iterations=1, warmup=0)

        assert len(results) == 1
        assert results[0].result_value == 5

    def test_benchmark_with_exception_handling(self, benchmark_suite):
        """Бенчмарк функции с исключением (функция выбрасывает ошибку)."""

        def failing_func():
            raise ValueError("test error")

        # measure_performance не перехватывает исключения — они пробрасываются
        with (
            patch("utils.performance.performance_benchmark.gc.collect"),
            patch(
                "utils.performance.performance_benchmark.time.perf_counter", side_effect=[0.0, 0.1]
            ),
        ):
            with pytest.raises(ValueError, match="test error"):
                benchmark_suite.measure_performance(failing_func, iterations=1, warmup=0)

    def test_benchmark_concurrent_measurements(self, benchmark_suite):
        """Бенчмарк нескольких измерений подряд (имитация параллельных)."""

        def quick():
            return True

        timings = [0.0, 0.1, 0.0, 0.2, 0.0, 0.15]
        with (
            patch("utils.performance.performance_benchmark.gc.collect"),
            patch("utils.performance.performance_benchmark.time.perf_counter", side_effect=timings),
        ):
            results = benchmark_suite.measure_performance(quick, iterations=3, warmup=0)

        assert len(results) == 3
        assert results[0].execution_time == 0.1
        assert results[1].execution_time == 0.2
        assert results[2].execution_time == 0.15
        assert all(r.result_value is True for r in results)

    def test_benchmark_large_iterations(self, benchmark_suite):
        """Бенчмарк с большим количеством итераций."""
        call_count = 0

        def counter():
            nonlocal call_count
            call_count += 1
            return call_count

        large_iterations = 100
        # Генерируем тайминги: pairs of (start, end)
        timing_values = []
        for i in range(large_iterations):
            timing_values.extend([float(i), float(i) + 0.01])

        with (
            patch("utils.performance.performance_benchmark.gc.collect"),
            patch(
                "utils.performance.performance_benchmark.time.perf_counter",
                side_effect=timing_values,
            ),
        ):
            results = benchmark_suite.measure_performance(
                counter, iterations=large_iterations, warmup=0
            )

        assert len(results) == large_iterations
        assert call_count == large_iterations
        for r in results:
            assert r.execution_time == pytest.approx(0.01, rel=1e-9)
            assert isinstance(r, BenchmarkResult)
