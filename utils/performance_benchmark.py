# -*- coding: utf-8 -*-
#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3

"""
Модуль бенчмаркинга производительности для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для бенчмаркинга, тестирования производительности
и сравнения производительности различных компонентов проекта.
"""

import time
import statistics
import threading
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dataclasses import dataclass
from functools import wraps
import psutil
import gc
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np

@dataclass
class BenchmarkResult:
    """Результат бенчмарка"""
    name: str
    execution_time: float
    memory_used_mb: float
    cpu_percent: float
    iterations: int
    timestamp: datetime
    parameters: Dict[str, Any]
    result_value: Any = None

@dataclass
class PerformanceComparison:
    """Сравнение производительности"""
    test_name: str
    baseline_result: BenchmarkResult
    comparison_result: BenchmarkResult
    improvement_percent: float
    is_significant: bool

class PerformanceBenchmarkSuite:
    """
    Класс набора бенчмарков производительности
    Обеспечивает тестирование производительности, сравнение и анализ результатов.
    """


    def __init__(self, output_dir: str = "benchmarks"):
        """
        Инициализирует набор бенчмарков

        Args:
            output_dir: Директория для сохранения результатов
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        self.comparisons = []
        self.current_process = psutil.Process()
        self.baseline_measurements = {}
        self.monitoring = False
        self.monitoring_thread = None
        self.monitoring_data = {
            'timestamps': [],
            'cpu_percent': [],
            'memory_mb': [],
            'disk_io': [],
            'network_io': []
        }


    def measure_performance(self, func: Callable, *args,
    """TODO: Add description"""

                          iterations: int = 10, warmup: int = 2,
                          **kwargs) -> List[BenchmarkResult]:
        """
        Измеряет производительность функции

        Args:
            func: Функция для тестирования
            *args: Аргументы для функции
            iterations: Количество итераций
            warmup: Количество прогревочных итераций
            **kwargs: Ключевые аргументы для функции

        Returns:
            Список результатов бенчмарков
        """
        results = []

        # Прогрев
        for _ in range(warmup):
            func(*args, **kwargs)

        # Замеры
        for i in range(iterations):
            # Сбросим сборку мусора перед каждым замером
            gc.collect()

            # Замеряем память до
            memory_before = self.current_process.memory_info().rss / (1024 * 1024)
            cpu_before = self.current_process.cpu_percent()

            # Замеряем время выполнения
            start_time = time.perf_counter()
            result_value = func(*args, **kwargs)
            end_time = time.perf_counter()

            # Замеряем память после
            memory_after = self.current_process.memory_info().rss / (1024 * 1024)
            cpu_after = self.current_process.cpu_percent()

            execution_time = end_time - start_time
            memory_used = memory_after - memory_before

            result = BenchmarkResult(
                name=func.__name__,
                execution_time=execution_time,
                memory_used_mb=memory_used,
                cpu_percent=cpu_after,
                iterations=1,
                timestamp=datetime.now(),
                parameters={'iteration': i, 'args_count': len(args), 'kwargs_count': len(kwargs)},
                result_value=result_value
            )

            results.append(result)
            self.results.append(result)

        return results

    """TODO: Add description"""


    def benchmark_function(self, name: str, func: Callable, *args,
                          iterations: int = 100, warmup: int = 10,
                          **kwargs) -> Dict[str, Any]:
        """
        Выполняет бенчмарк функции

        Args:
            name: Имя теста
            func: Функция для бенчмарка
            *args: Аргументы для функции
            iterations: Количество итераций
            warmup: Количество прогревочных итераций
            **kwargs: Ключевые аргументы для функции

        Returns:
            Результаты бенчмарка
        """
        print(f"Запуск бенчмарка: {name}")

        # Прогрев
        for _ in range(warmup):
            func(*args, **kwargs)

        # Замеры
        times = []
        memory_usages = []
        cpu_usages = []

        for _ in range(iterations):
            gc.collect()

            # Замеряем память и CPU до
            memory_before = self.current_process.memory_info().rss / (1024 * 1024)
            cpu_before = self.current_process.cpu_percent()

            # Выполняем функцию
            start_time = time.perf_counter()
            result_value = func(*args, **kwargs)
            end_time = time.perf_counter()

            # Замеряем после
            memory_after = self.current_process.memory_info().rss / (1024 * 1024)
            cpu_after = self.current_process.cpu_percent()

            times.append(end_time - start_time)
            memory_usages.append(memory_after - memory_before)
            cpu_usages.append(cpu_after)

        # Статистика
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0

        avg_memory = statistics.mean(memory_usages)
        max_memory = max(memory_usages)

        avg_cpu = statistics.mean(cpu_usages)

        result = {
            'name': name,
            'function': func.__name__,
            'iterations': iterations,
            'warmup': warmup,
            'timing_stats': {
                'avg_seconds': avg_time,
                'min_seconds': min_time,
                'max_seconds': max_time,
                'std_dev_seconds': std_dev,
                'total_seconds': sum(times)
            },
            'memory_stats': {
                'avg_mb': avg_memory,
                'max_mb': max_memory,
                'measurements_count': len(memory_usages)
            },
            'cpu_stats': {
                'avg_percent': avg_cpu,
                'measurements_count': len(cpu_usages)
            },
            'throughput_ops_per_second': 1 / avg_time if avg_time > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }

        # Сохраняем как BenchmarkResult для внутреннего использования
        benchmark_result = BenchmarkResult(
            name=name,
            execution_time=avg_time,
            memory_used_mb=avg_memory,
            cpu_percent=avg_cpu,
            iterations=iterations,
            timestamp=datetime.now(),
            parameters={
                'function': func.__name__,
                'iterations': iterations,
                'warmup': warmup
            }
        )
        self.results.append(benchmark_result)

        return result
    """TODO: Add description"""


    def benchmark_parallel_execution(self, name: str, func: Callable, *args,
                                   thread_counts: List[int] = [1, 2, 4, 8],
                                   iterations_per_thread: int = 10,
                                   **kwargs) -> Dict[str, Any]:
        """
        Бенчмарка параллельного выполнения

        Args:
            name: Имя теста
            func: Функция для тестирования
            *args: Аргументы для функции
            thread_counts: Количество потоков для тестирования
            iterations_per_thread: Итераций на поток
            **kwargs: Ключевые аргументы для функции

        Returns:
            Результаты бенчмарка параллельного выполнения
        """
        print(f"Запуск бенчмарка параллелизма: {name}")

        results = {}

        for num_threads in thread_counts:
            start_time = time.time()

            # Прогрев
            for _ in range(2):
                func(*args, **kwargs)

            # Тестирование
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(func, *args, **kwargs) for _ in range(iterations_per_thread)]
                # Ждем завершения всех задач
                for future in futures:
                    future.result()

            end_time = time.time()
            execution_time = end_time - start_time

            results[num_threads] = {
                'execution_time': execution_time,
                'throughput': iterations_per_thread / execution_time if execution_time > 0 else 0,
                'threads': num_threads
            }

        parallel_result = {
            'name': f"{name}_parallel",
            'results': results,
            'thread_counts': thread_counts,
            'iterations_per_thread': iterations_per_thread,
            'timestamp': datetime.now().isoformat()
        }

        return parallel_result


    def benchmark_memory_usage(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Бенчмарка использования памяти

        Args:
            func: Функция для тестирования
            *args: Аргументы для функции
            **kwargs: Ключевые аргументы для функции

        Returns:
            Результаты бенчмарка памяти
        """
        print(f"Запуск бенчмарка памяти для: {func.__name__}")

        # Начинаем трассировку памяти
        tracemalloc.start()

        # Выполняем функцию
        result = func(*args, **kwargs)

        # Получаем статистику
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        memory_result = {
            'function': func.__name__,
            'current_memory_mb': current / (1024 * 1024),
            'peak_memory_mb': peak / (1024 * 1024),
            'timestamp': datetime.now().isoformat()
        }

    """TODO: Add description"""

        return memory_result


    def compare_algorithms(self, algorithms: Dict[str, Callable],
                          test_data: Any, iterations: int = 50, **kwargs) -> List[PerformanceComparison]:
        """
        Сравнивает производительность алгоритмов

        Args:
            algorithms: Словарь с названиями и функциями алгоритмов
            test_data: Данные для тестирования
            iterations: Количество итераций

        Returns:
            Список сравнений производительности
        """
        print("Сравнение производительности алгоритмов...")

        algorithm_results = {}

        for name, func in algorithms.items():
            # Выполняем бенчмарк для каждого алгоритма
            result = self.benchmark_function(
                f"algorithm_{name}",
                func,
                test_data,
                iterations=iterations,
                **kwargs
            )
            algorithm_results[name] = result

        # Создаем сравнения (берем первый как базовый)
        base_name = list(algorithm_results.keys())[0]
        base_result = algorithm_results[base_name]

        comparisons = []

        for name, result in algorithm_results.items():
            if name != base_name:
                # Сравниваем по времени выполнения
                base_time = base_result['timing_stats']['avg_seconds']
                current_time = result['timing_stats']['avg_seconds']

                improvement = ((base_time - current_time) / base_time) * 100 if base_time > 0 else 0
                is_significant = abs(improvement) > 5  # Различие более 5% считаем значимым

                comparison = PerformanceComparison(
                    test_name=f"{base_name}_vs_{name}",
                    baseline_result=BenchmarkResult(
                        name=base_name,
                        execution_time=base_time,
                        memory_used_mb=base_result['memory_stats']['avg_mb'],
                        cpu_percent=base_result['cpu_stats']['avg_percent'],
                        iterations=iterations,
                        timestamp=datetime.now(),
                        parameters={}
                    ),
                    comparison_result=BenchmarkResult(
                        name=name,
                        execution_time=current_time,
                        memory_used_mb=result['memory_stats']['avg_mb'],
                        cpu_percent=result['cpu_stats']['avg_percent'],
                        iterations=iterations,
                        timestamp=datetime.now(),
                        parameters={}
                    ),
                    improvement_percent=improvement,
                    is_significant=is_significant
                )

                comparisons.append(comparison)
                self.comparisons.append(comparison)

        return comparisons


    def start_system_monitoring(self, interval: float = 1.0):
        """
        Начинает мониторинг системы во время бенчмарков

        Args:
            interval: Интервал между замерами
        """
        if self.monitoring:
            return
    """TODO: Add description"""

        self.monitoring = True

        def monitor():
            """TODO: Add description"""
            while self.monitoring:
                try:
                    # Замеряем системные показатели
                    cpu_percent = psutil.cpu_percent()
                    memory_info = psutil.virtual_memory()

                    self.monitoring_data['timestamps'].append(datetime.now())
                    self.monitoring_data['cpu_percent'].append(cpu_percent)
                    self.monitoring_data['memory_mb'].append(memory_info.used / (1024 * 1024))

                    time.sleep(interval)

                except Exception as e:
                    print(f"Ошибка мониторинга системы: {e}")
                    time.sleep(interval)

        self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self.monitoring_thread.start()


    def stop_system_monitoring(self):
        """Останавливает мониторинг системы"""
        self.monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2)


    def generate_performance_report(self, output_path: str = None) -> str:
        """
        Генерирует отчет о производительности

        Args:
            output_path: Путь для сохранения отчета

        Returns:
            Путь к сохраненному отчету
        """
        if output_path is None:
            output_path = str(self.output_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        # Подготавливаем данные для отчета
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'total_benchmarks': len(self.results),
            'total_comparisons': len(self.comparisons),
            'benchmark_results': [
                {
                    'name': r.name,
                    'execution_time': r.execution_time,
                    'memory_used_mb': r.memory_used_mb,
                    'cpu_percent': r.cpu_percent,
                    'iterations': r.iterations,
                    'timestamp': r.timestamp.isoformat(),
                    'parameters': r.parameters
                }
                for r in self.results
            ],
            'comparisons': [
                {
                    'test_name': c.test_name,
                    'baseline': {
                        'name': c.baseline_result.name,
                        'execution_time': c.baseline_result.execution_time,
                        'memory_used_mb': c.baseline_result.memory_used_mb,
                        'cpu_percent': c.baseline_result.cpu_percent
                    },
                    'comparison': {
                        'name': c.comparison_result.name,
                        'execution_time': c.comparison_result.execution_time,
                        'memory_used_mb': c.comparison_result.memory_used_mb,
                        'cpu_percent': c.comparison_result.cpu_percent
                    },
                    'improvement_percent': c.improvement_percent,
                    'is_significant': c.is_significant
                }
                for c in self.comparisons
            ],
            'aggregated_stats': self._calculate_aggregated_stats(),
            'monitoring_data': {
                'samples_count': len(self.monitoring_data['timestamps']),
                'avg_cpu_percent': statistics.mean(self.monitoring_data['cpu_percent']) if self.monitoring_data['cpu_percent'] else 0,
                'avg_memory_mb': statistics.mean(self.monitoring_data['memory_mb']) if self.monitoring_data['memory_mb'] else 0
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)

        return output_path


    def _calculate_aggregated_stats(self) -> Dict[str, Any]:
        """Рассчитывает агрегированные статистики"""
        if not self.results:
            return {}

        execution_times = [r.execution_time for r in self.results]
        memory_usages = [r.memory_used_mb for r in self.results]
        cpu_percentages = [r.cpu_percent for r in self.results]

        return {
            'execution_time': {
                'avg': statistics.mean(execution_times),
                'min': min(execution_times),
                'max': max(execution_times),
                'median': statistics.median(execution_times),
                'std_dev': statistics.stdev(execution_times) if len(execution_times) > 1 else 0
            },
            'memory_usage': {
                'avg_mb': statistics.mean(memory_usages),
                'min_mb': min(memory_usages),
                'max_mb': max(memory_usages),
                'median_mb': statistics.median(memory_usages)
            },
            'cpu_usage': {
                'avg_percent': statistics.mean(cpu_percentages),
                'min_percent': min(cpu_percentages),
                'max_percent': max(cpu_percentages)
            }
        }


    def visualize_benchmark_results(self, output_path: str = None) -> str:
        """
        Визуализирует результаты бенчмарков

        Args:
            output_path: Путь для сохранения визуализации

        Returns:
            Путь к сохраненной визуализации
        """
        if not self.results:
            print("Нет данных для визуализации")
            return ""

        if output_path is None:
            output_path = str(self.output_dir / f"benchmark_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

        # Подготовка данных
        df = pd.DataFrame([
            {
                'name': r.name,
                'execution_time': r.execution_time,
                'memory_used_mb': r.memory_used_mb,
                'cpu_percent': r.cpu_percent,
                'iterations': r.iterations
            }
            for r in self.results
        ])

        # Создаем фигуру с несколькими подграфиками
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Результаты бенчмарков производительности', fontsize=16)

        # 1. Время выполнения
        axes[0, 0].bar(range(len(df)), df['execution_time'], color='skyblue')
        axes[0, 0].set_title('Время выполнения по тестам')
        axes[0, 0].set_ylabel('Время (сек)')
        axes[0, 0].set_xticks(range(len(df)))
        axes[0, 0].set_xticklabels(df['name'], rotation=45, ha='right')

        # 2. Использование памяти
        axes[0, 1].bar(range(len(df)), df['memory_used_mb'], color='lightgreen')
        axes[0, 1].set_title('Использование памяти по тестам')
        axes[0, 1].set_ylabel('Память (MB)')
        axes[0, 1].set_xticks(range(len(df)))
        axes[0, 1].set_xticklabels(df['name'], rotation=45, ha='right')

        # 3. Использование CPU
        axes[1, 0].bar(range(len(df)), df['cpu_percent'], color='coral')
        axes[1, 0].set_title('Использование CPU по тестам')
        axes[1, 0].set_ylabel('CPU (%)')
        axes[1, 0].set_xticks(range(len(df)))
        axes[1, 0].set_xticklabels(df['name'], rotation=45, ha='right')

        # 4. Сравнение времени выполнения и памяти (scatter plot)
        scatter = axes[1, 1].scatter(df['execution_time'], df['memory_used_mb'],
                                     c=df['cpu_percent'], cmap='viridis', alpha=0.7)
        axes[1, 1].set_title('Сравнение времени и памяти')
        axes[1, 1].set_xlabel('Время выполнения (сек)')
        axes[1, 1].set_ylabel('Использование памяти (MB)')
        plt.colorbar(scatter, ax=axes[1, 1], label='CPU (%)')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path


    def get_performance_recommendations(self) -> List[str]:
        """
        Получает рекомендации по производительности

        Returns:
            Список рекомендаций
        """
        recommendations = []

        if not self.results:
            return ["Нет данных для анализа производительности"]

        # Анализируем средние значения
        avg_time = statistics.mean(r.execution_time for r in self.results)
        avg_memory = statistics.mean(r.memory_used_mb for r in self.results)
        avg_cpu = statistics.mean(r.cpu_percent for r in self.results)

        if avg_time > 1.0:  # Если среднее время больше 1 секунды
            recommendations.append("Среднее время выполнения превышает 1 секунду. Рассмотрите оптимизацию.")

        if avg_memory > 100:  # Если среднее использование памяти больше 100MB
            recommendations.append("Среднее использование памяти превышает 100MB. Рассмотрите оптимизацию.")

        if avg_cpu > 80:  # Если среднее использование CPU больше 80%
            recommendations.append("Средняя загрузка CPU превышает 80%. Рассмотрите оптимизацию.")

        # Проверяем сравнения
        significant_comparisons = [c for c in self.comparisons if c.is_significant]
        if significant_comparisons:
            better_performers = [c for c in significant_comparisons if c.improvement_percent > 0]
            worse_performers = [c for c in significant_comparisons if c.improvement_percent < 0]

            if better_performers:
                recommendations.append(f"Найдено {len(better_performers)} алгоритмов с лучшей производительностью.")

            if worse_performers:
                recommendations.append(f"Найдено {len(worse_performers)} алгоритмов с худшей производительностью.")

        if not recommendations:
            recommendations.append("Производительность системы в норме. Рекомендаций по оптимизации нет.")

        return recommendations

class BenchmarkDecorator:
    """
    Декоратор для бенчмаркинга функций
    """


    def __init__(self, benchmark_suite: PerformanceBenchmarkSuite):
        """
        Инициализирует декоратор

        Args:
            benchmark_suite: Экземпляр набора бенчмарков
        """
        self.benchmark_suite = benchmark_suite


    def __call__(self, func: Callable, iterations: int = 10, warmup: int = 2) -> Callable:
        """
        Декорирует функцию для бенчмаркинга

        Args:
            func: Функция для декорирования
            iterations: Количество итераций
            warmup: Количество прогревочных итераций

    """TODO: Add description"""

        Returns:
            Обернутая функция
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Выполняем бенчмарк
            """TODO: Add description"""
            result = self.benchmark_suite.benchmark_function(
                name=f"decorated_{func.__name__}",
                func=func,
                *args,
                iterations=iterations,
                warmup=warmup,
                **kwargs
            )

            print(f"\n=== Результаты бенчмарка для {func.__name__} ===")
            print(f"Среднее время: {result['timing_stats']['avg_seconds']:.6f} сек")
            print(f"Минимальное время: {result['timing_stats']['min_seconds']:.6f} сек")
            print(f"Максимальное время: {result['timing_stats']['max_seconds']:.6f} сек")
            print(f"Среднее использование памяти: {result['memory_stats']['avg_mb']:.2f} MB")
            print(f"Пропускная способность: {result['throughput_ops_per_second']:.2f} ops/sec")

            return func(*args, **kwargs)

        return wrapper

def main():
    """Главная функция для демонстрации возможностей бенчмарка"""
    print("=== НАБОР БЕНЧМАРКОВ ПРОИЗВОДИТЕЛЬНОСТИ ===")

    # Создаем набор бенчмарков
    benchmark_suite = PerformanceBenchmarkSuite()

    print("✓ Набор бенчмарков инициализирован")
    print(f"✓ Директория вывода: {benchmark_suite.output_dir}")

    # Начинаем мониторинг системы
    print("\nЗапуск мониторинга системы...")
    benchmark_suite.start_system_monitoring(interval=0.5)

    # Примеры функций для бенчмарка

    def slow_calculation(n):
        """Медленное вычисление для тестирования"""
        result = 0
        for i in range(n):
            result += i ** 2
        return result


    def fast_calculation(n):
        """Быстрое вычисление для тестирования"""
        return sum(i ** 2 for i in range(n))


    def memory_intensive_operation(size):
        """Операция с интенсивным использованием памяти"""
        data = [i for i in range(size)]
        processed = [x * 2 for x in data]
        return len(processed)

    # Выполняем бенчмарк для медленной функции
    print("\nБенчмарк медленной функции...")
    slow_result = benchmark_suite.benchmark_function(
        "slow_calculation",
        slow_calculation,
        10000,
        iterations=50
    )
    print(f"Среднее время: {slow_result['timing_stats']['avg_seconds']:.6f} сек")

    # Выполняем бенчмарк для быстрой функции
    print("\nБенчмарк быстрой функции...")
    fast_result = benchmark_suite.benchmark_function(
        "fast_calculation",
        fast_calculation,
        10000,
        iterations=50
    )
    print(f"Среднее время: {fast_result['timing_stats']['avg_seconds']:.6f} сек")

    # Бенчмарк использования памяти
    print("\nБенчмарк использования памяти...")
    memory_result = benchmark_suite.benchmark_memory_usage(
        memory_intensive_operation,
        50000
    )
    print(f"Пик использования памяти: {memory_result['peak_memory_mb']:.2f} MB")

    # Бенчмарк параллельного выполнения
    print("\nБенчмарк параллельного выполнения...")
    parallel_result = benchmark_suite.benchmark_parallel_execution(
        "parallel_processing",
        slow_calculation,
        5000,
        thread_counts=[1, 2, 4],
        iterations_per_thread=10
    )

    for threads, result in parallel_result['results'].items():
        print(f"  {threads} потоков: {result['execution_time']:.4f} сек, "
              f"пропускная способность: {result['throughput']:.2f} ops/sec")

    # Сравниваем алгоритмы
    print("\nСравнение алгоритмов...")
    algorithms = {
        'slow': slow_calculation,
        'fast': fast_calculation
    }

    comparisons = benchmark_suite.compare_algorithms(algorithms, 5000, iterations=20)
    print(f"Выполнено {len(comparisons)} сравнений")

    for comparison in comparisons:
        print(f"  {comparison.test_name}: {comparison.improvement_percent:+.2f}% "
              f"({'значимо' if comparison.is_significant else 'незначимо'})")

    # Останавливаем мониторинг
    benchmark_suite.stop_system_monitoring()
    print("\nМониторинг системы остановлен")

    # Генерируем отчет
    print("\nГенерация отчета...")
    report_path = benchmark_suite.generate_performance_report()
    print(f"✓ Отчет сохранен: {report_path}")

    # Создаем визуализацию
    print("\nСоздание визуализации...")
    viz_path = benchmark_suite.visualize_benchmark_results()
    if viz_path:
        print(f"✓ Визуализация сохранена: {viz_path}")

    # Получаем рекомендации
    print("\nПолучение рекомендаций по производительности...")
    recommendations = benchmark_suite.get_performance_recommendations()
    print("Рекомендации:")
    for rec in recommendations:
        print(f"  - {rec}")

    """TODO: Add description"""

    # Пример использования декоратора
    print("\nПример использования декоратора бенчмарка...")
    benchmark_decorator = BenchmarkDecorator(benchmark_suite)

    @benchmark_decorator

    def decorated_function(x):
        """TODO: Add description"""
        return x ** 2

    result = decorated_function(100)
    print(f"Результат функции: {result}")

    print("\nНабор бенчмарков успешно протестирован")
    print("\nДоступные функции:")
    print("- Бенчмарк функций: benchmark_suite.benchmark_function()")
    print("- Бенчмарк памяти: benchmark_suite.benchmark_memory_usage()")
    print("- Бенчмарк параллелизма: benchmark_suite.benchmark_parallel_execution()")
    print("- Сравнение алгоритмов: benchmark_suite.compare_algorithms()")
    print("- Мониторинг системы: benchmark_suite.start_system_monitoring()")
    print("- Отчеты: benchmark_suite.generate_performance_report()")
    print("- Визуализация: benchmark_suite.visualize_benchmark_results()")
    print("- Рекомендации: benchmark_suite.get_performance_recommendations()")
    print("- Декоратор для функций: BenchmarkDecorator")

if __name__ == "__main__":
    main()

