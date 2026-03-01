# -*- coding: utf-8 -*-
#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3

"""
Модуль профилирования производительности для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для детального анализа производительности,
профилирования функций и оптимизации ресурсов проекта.
"""

import cProfile
import pstats
import io
import time
import threading
import psutil
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dataclasses import dataclass
from functools import wraps
try:
    import memory_profiler
except ImportError:
    memory_profiler = None
    print("Warning: memory_profiler not installed. Install with 'pip install memory-profiler'")
import tracemalloc
from contextlib import contextmanager
import gc

@dataclass
class PerformanceMetric:
    """Метрика производительности"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    context: str

@dataclass
class ResourceUsage:
    """Использование ресурсов"""
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_io_read: float
    disk_io_write: float
    network_sent: float
    network_recv: float
    timestamp: datetime

class PerformanceProfiler:
    """
    Класс профилировщика производительности
    Обеспечивает детальный анализ производительности,
    профилирование функций и оптимизацию ресурсов.
    """


    def __init__(self, output_dir: str = "profiles"):
        """
        Инициализирует профилировщик производительности

        Args:
            output_dir: Директория для сохранения результатов профилирования
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics = []
        self.profile_stats = {}
        self.resource_usage_history = []
        self.cpu_monitoring = False
        self.memory_monitoring = False
        self.monitoring_thread = None
        self.monitoring_data = {
            'timestamps': [],
            'cpu_percent': [],
            'memory_percent': [],
            'memory_mb': [],
            'disk_io_read': [],
            'disk_io_write': [],
            'network_sent': [],
            'network_recv': []
        }
        self.current_process = psutil.Process()
        self.benchmark_results = {}


    def start_resource_monitoring(self, interval: float = 1.0):
        """
        Запускает мониторинг ресурсов

        Args:
            interval: Интервал между измерениями (в секундах)
        """
        if self.cpu_monitoring:
            return

        self.cpu_monitoring = True

        def monitor():

            while self.cpu_monitoring:
                try:
                    # Сбор метрик
                    cpu_percent = psutil.cpu_percent()
                    memory_info = psutil.virtual_memory()
                    disk_io = psutil.disk_io_counters()
                    net_io = psutil.net_io_counters()

                    # Получаем значения IO
                    disk_read = disk_io.read_bytes if disk_io else 0
                    disk_write = disk_io.write_bytes if disk_io else 0
                    net_sent = net_io.bytes_sent if net_io else 0
                    net_recv = net_io.bytes_recv if net_io else 0

                    # Добавляем данные
                    self.monitoring_data['timestamps'].append(datetime.now())
                    self.monitoring_data['cpu_percent'].append(cpu_percent)
                    self.monitoring_data['memory_percent'].append(memory_info.percent)
                    self.monitoring_data['memory_mb'].append(memory_info.used / (1024*1024))
                    self.monitoring_data['disk_io_read'].append(disk_read)
                    self.monitoring_data['disk_io_write'].append(disk_write)
                    self.monitoring_data['network_sent'].append(net_sent)
                    self.monitoring_data['network_recv'].append(net_recv)

                    # Сохраняем в историю
                    resource_usage = ResourceUsage(
                        cpu_percent=cpu_percent,
                        memory_percent=memory_info.percent,
                        memory_mb=memory_info.used / (1024*1024),
                        disk_io_read=disk_read,
                        disk_io_write=disk_write,
                        network_sent=net_sent,
                        network_recv=net_recv,
                        timestamp=datetime.now()
                    )
                    self.resource_usage_history.append(resource_usage)

                except Exception as e:
                    print(f"Ошибка мониторинга ресурсов: {e}")

                time.sleep(interval)

        self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self.monitoring_thread.start()


    def stop_resource_monitoring(self):
        """Останавливает мониторинг ресурсов"""
        self.cpu_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2)


    def profile_function(self, func: Callable) -> Callable:
        """
        Декоратор для профилирования функции

        Args:
            func: Функция для профилирования

        Returns:
            Профилированная функция
        """
        @wraps(func)

        def wrapper(*args, **kwargs):
            # Запускаем трассировку памяти
                    tracemalloc.start()
            start_time = time.time()
            start_cpu = self.current_process.cpu_percent()
            start_memory = self.current_process.memory_info().rss / (1024 * 1024)  # MB

            try:
                result = func(*args, **kwargs)

                # Завершаем трассировку памяти
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                end_time = time.time()
                end_cpu = self.current_process.cpu_percent()
                end_memory = self.current_process.memory_info().rss / (1024 * 1024)  # MB

                # Рассчитываем метрики
                execution_time = end_time - start_time
                memory_used = end_memory - start_memory
                cpu_usage = end_cpu - start_cpu

                # Сохраняем метрики
                metric = PerformanceMetric(
                    name=f"{func.__name__}_execution_time",
                    value=execution_time,
                    unit="seconds",
                    timestamp=datetime.now(),
                    context=f"Function: {func.__name__}"
                )
                self.metrics.append(metric)

                metric = PerformanceMetric(
                    name=f"{func.__name__}_memory_used",
                    value=memory_used,
                    unit="MB",
                    timestamp=datetime.now(),
                    context=f"Function: {func.__name__}"
                )
                self.metrics.append(metric)

                # Выводим результаты
                print(f"\n=== Профилирование функции: {func.__name__} ===")
                print(f"Время выполнения: {execution_time:.4f} сек")
                print(f"Использование памяти: {memory_used:.2f} MB")
                print(f"Пиковое использование памяти: {peak / (1024 * 1024):.2f} MB")
                print(f"Использование CPU: {cpu_usage}%")

                return result

            except Exception as e:
                tracemalloc.stop()
                raise e

        return wrapper


    def profile_code_block(self, code: str, name: str = "code_block") -> Dict[str, Any]:
        """
        Профилирует блок кода

        Args:
            code: Код для профилирования
            name: Имя блока кода

        Returns:
            Результаты профилирования
        """
        pr = cProfile.Profile()
        pr.enable()

        start_time = time.time()
        exec(code, globals(), locals())
        end_time = time.time()

        pr.disable()

        # Сохраняем статистику
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats()

        # Сохраняем в файл
        profile_file = self.output_dir / f"profile_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(profile_file, 'w', encoding='utf-8') as f:
            f.write(s.getvalue())

        execution_time = end_time - start_time

        # Добавляем метрики
        metric = PerformanceMetric(
            name=f"{name}_execution_time",
            value=execution_time,
            unit="seconds",
            timestamp=datetime.now(),
            context=f"Code block: {name}"
        )
        self.metrics.append(metric)

        return {
            'execution_time': execution_time,
            'profile_file': str(profile_file),
            'call_count': ps.total_calls,
            'primitive_calls': ps.total_calls  # primitive_calls is not a valid attribute, using total_calls instead
        }


    def memory_profile_function(self, func: Callable) -> Callable:
        """
        Декоратор для профилирования памяти функции

        Args:
            func: Функция для профилирования памяти

        Returns:
            Профилированная функция
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
                    if memory_profiler is None:
                print(f"\n=== Memory profiler not available for: {func.__name__} ===")
                print("Install memory-profiler to enable memory profiling: pip install memory-profiler")
                return func(*args, **kwargs)

            # Запускаем профилирование памяти
            mem_usage = memory_profiler.memory_usage(
                (func, args, kwargs),
                retval=True,
                timeout=None,
                repeat=1,
                precision=2
            )

            # Получаем результат функции и использование памяти
            result, mem_usage_values = mem_usage[0][0], mem_usage[0][1:]

            avg_memory = sum(mem_usage_values) / len(mem_usage_values)
            peak_memory = max(mem_usage_values)

            # Добавляем метрики
            metric = PerformanceMetric(
                name=f"{func.__name__}_avg_memory",
                value=avg_memory,
                unit="MB",
                timestamp=datetime.now(),
                context=f"Function: {func.__name__}"
            )
            self.metrics.append(metric)

            metric = PerformanceMetric(
                name=f"{func.__name__}_peak_memory",
                value=peak_memory,
                unit="MB",
                timestamp=datetime.now(),
                context=f"Function: {func.__name__}"
            )
            self.metrics.append(metric)

            print(f"\n=== Профилирование памяти функции: {func.__name__} ===")
            print(f"Среднее использование памяти: {avg_memory:.2f} MB")
            print(f"Пиковое использование памяти: {peak_memory:.2f} MB")

            return result

        return wrapper


    def benchmark_function(self, func: Callable, iterations: int = 100, warmup: int = 10) -> Dict[str, Any]:
        """
        Бенчмаркинг функции

        Args:
            func: Функция для бенчмаркинга
            iterations: Количество итераций
            warmup: Количество прогревочных итераций

        Returns:
            Результаты бенчмаркинга
        """
        # Прогрев
        for _ in range(warmup):
            func()

        # Замер времени
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            end = time.perf_counter()
            times.append(end - start)

        # Статистика
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

        # Сохраняем результаты
        benchmark_key = f"{func.__name__}_benchmark"
        self.benchmark_results[benchmark_key] = {
            'function': func.__name__,
            'iterations': iterations,
            'warmup': warmup,
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'std_dev': std_dev,
            'total_time': sum(times)
        }

        # Добавляем метрики
        metric = PerformanceMetric(
            name=f"{func.__name__}_avg_execution_time",
            value=avg_time,
            unit="seconds",
            timestamp=datetime.now(),
            context=f"Benchmark: {func.__name__}"
        )
        self.metrics.append(metric)

        return {
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'std_dev': std_dev,
            'total_time': sum(times),
            'iterations': iterations
        }


    def generate_performance_report(self, output_path: str = None) -> str:
        """
        Генерирует полный отчет о производительности

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
            'total_metrics': len(self.metrics),
            'total_resource_samples': len(self.resource_usage_history),
            'benchmark_results': self.benchmark_results,
            'metrics': [
                {
                    'name': metric.name,
                    'value': metric.value,
                    'unit': metric.unit,
                    'timestamp': metric.timestamp.isoformat(),
                    'context': metric.context
                }
                for metric in self.metrics
            ],
            'resource_usage': [
                {
                    'cpu_percent': usage.cpu_percent,
                    'memory_percent': usage.memory_percent,
                    'memory_mb': usage.memory_mb,
                    'disk_io_read': usage.disk_io_read,
                    'disk_io_write': usage.disk_io_write,
                    'network_sent': usage.network_sent,
                    'network_recv': usage.network_recv,
                    'timestamp': usage.timestamp.isoformat()
                }
                for usage in self.resource_usage_history
            ],
            'aggregated_stats': self._calculate_aggregated_stats()
        }

        # Сохраняем отчет
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)

        return output_path


    def _calculate_aggregated_stats(self) -> Dict[str, Any]:
        """Рассчитывает агрегированные статистики"""
        if not self.metrics:
            return {}

        # Группируем метрики по типу
        metric_types = {}
        for metric in self.metrics:
            if metric.name not in metric_types:
                metric_types[metric.name] = []
            metric_types[metric.name].append(metric.value)

        # Рассчитываем статистики для каждого типа
        stats = {}
        for metric_type, values in metric_types.items():
            stats[metric_type] = {
                'count': len(values),
                'avg': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'sum': sum(values)
            }

        return stats


    def visualize_performance(self, output_path: str = None) -> str:
        """
        Визуализирует производительность

        Args:
            output_path: Путь для сохранения визуализации

        Returns:
            Путь к сохраненной визуализации
        """
        if output_path is None:
            output_path = str(self.output_dir / f"performance_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

        if not self.monitoring_data['timestamps']:
            print("Нет данных для визуализации")
            return ""

        # Создаем фигуру с несколькими подграфиками
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Анализ производительности', fontsize=16)

        timestamps = self.monitoring_data['timestamps']

        # График CPU
        axes[0, 0].plot(timestamps, self.monitoring_data['cpu_percent'], label='CPU %', color='blue')
        axes[0, 0].set_title('Загрузка CPU')
        axes[0, 0].set_ylabel('Проценты')
        axes[0, 0].grid(True)

        # График памяти
        axes[0, 1].plot(timestamps, self.monitoring_data['memory_mb'], label='Память (MB)', color='red')
        axes[0, 1].set_title('Использование памяти')
        axes[0, 1].set_ylabel('MB')
        axes[0, 1].grid(True)

        # График дискового ввода-вывода
        axes[1, 0].plot(timestamps, self.monitoring_data['disk_io_read'], label='Диск чтение', color='green')
        axes[1, 0].plot(timestamps, self.monitoring_data['disk_io_write'], label='Диск запись', color='orange')
        axes[1, 0].set_title('Дисковый I/O')
        axes[1, 0].set_ylabel('Байты')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # График сетевого ввода-вывода
        axes[1, 1].plot(timestamps, self.monitoring_data['network_sent'], label='Сеть отправлено', color='purple')
        axes[1, 1].plot(timestamps, self.monitoring_data['network_recv'], label='Сеть получено', color='brown')
        axes[1, 1].set_title('Сетевой I/O')
        axes[1, 1].set_ylabel('Байты')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        # Настройка подписей
        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path


    def get_optimization_recommendations(self) -> List[str]:
        """
        Получает рекомендации по оптимизации

        Returns:
            Список рекомендаций
        """
        recommendations = []

        if not self.resource_usage_history:
            return ["Нет данных для анализа оптимизации"]

        # Анализ средних значений
        avg_cpu = sum(r.cpu_percent for r in self.resource_usage_history) / len(self.resource_usage_history)
        avg_memory = sum(r.memory_mb for r in self.resource_usage_history) / len(self.resource_usage_history)

        if avg_cpu > 80:
            recommendations.append("Высокая загрузка CPU (>80%). Рассмотрите оптимизацию алгоритмов или параллелизацию.")

        if avg_memory > 1000:  # больше 1GB
            recommendations.append("Высокое использование памяти (>1GB). Рассмотрите оптимизацию использования памяти.")

        # Анализ пиковых значений
        peak_cpu = max(r.cpu_percent for r in self.resource_usage_history)
        peak_memory = max(r.memory_mb for r in self.resource_usage_history)

        if peak_cpu > 95:
            recommendations.append("Пиковая загрузка CPU очень высока (>95%). Необходима срочная оптимизация.")

        if peak_memory > 2000:  # больше 2GB
            recommendations.append("Пиковое использование памяти чрезмерно (>2GB). Проверьте утечки памяти.")

        if not recommendations:
            recommendations.append("Производительность системы в норме. Рекомендаций по оптимизации нет.")

        return recommendations

def main():
    """Главная функция для демонстрации возможностей профилировщика производительности"""
    print("=== ПРОФИЛИРОВЩИК ПРОИЗВОДИТЕЛЬНОСТИ ===")

    # Создаем профилировщик
    profiler = PerformanceProfiler()

    print("✓ Профилировщик производительности инициализирован")
    print(f"✓ Директория вывода: {profiler.output_dir}")

    # Запускаем мониторинг ресурсов
    print("\nЗапуск мониторинга ресурсов...")
    profiler.start_resource_monitoring(interval=0.5)

    # Пример функции для профилирования
    @profiler.profile_function

    def example_heavy_function():
        """Пример тяжелой функции для профилирования"""
        result = 0
        for i in range(1000000):
            result += i ** 2
        return result

    # Пример функции для профилирования памяти
    @profiler.memory_profile_function

    def example_memory_function():
        """Пример функции для профилирования памяти"""
        # Создаем большой список
        big_list = [i for i in range(100000)]
        return len(big_list)

    # Запускаем примеры
    print("\nПрофилирование тяжелой функции...")
    result1 = example_heavy_function()
    print(f"Результат: {result1}")

    print("\nПрофилирование функции памяти...")
    result2 = example_memory_function()
    print(f"Результат: {result2}")

    # Бенчмаркинг
    print("\nБенчмаркинг функции...")
    benchmark_result = profiler.benchmark_function(example_heavy_function, iterations=10)
    print(f"Среднее время выполнения: {benchmark_result['avg_time']:.6f} сек")

    # Пример профилирования блока кода

    print("\nПрофилирование блока кода...")
    code_block = """
def test_function():
    return sum(i**2 for i in range(100000))

result = test_function()
"""
    code_profile_result = profiler.profile_code_block(code_block, "test_code_block")
    print(f"Время выполнения блока кода: {code_profile_result['execution_time']:.4f} сек")

    # Останавливаем мониторинг
    print("\nОстановка мониторинга ресурсов...")
    profiler.stop_resource_monitoring()

    # Генерируем отчет
    print("\nГенерация отчета о производительности...")
    report_path = profiler.generate_performance_report()
    print(f"✓ Отчет сохранен: {report_path}")

    # Создаем визуализацию
    print("\nСоздание визуализации производительности...")
    viz_path = profiler.visualize_performance()
    if viz_path:
        print(f"✓ Визуализация сохранена: {viz_path}")

    # Получаем рекомендации
    print("\nПолучение рекомендаций по оптимизации...")
    recommendations = profiler.get_optimization_recommendations()
    print("Рекомендации:")
    for rec in recommendations:
        print(f"  - {rec}")

    print("\nПрофилировщик производительности успешно протестирован")
    print("\nДоступные функции:")
    print("- Профилирование функций: @profiler.profile_function")
    print("- Профилирование памяти: @profiler.memory_profile_function")
    print("- Бенчмаркинг: profiler.benchmark_function()")
    print("- Профилирование кода: profiler.profile_code_block()")
    print("- Мониторинг ресурсов: profiler.start_resource_monitoring()")
    print("- Отчет о производительности: profiler.generate_performance_report()")
    print("- Визуализация: profiler.visualize_performance()")
    print("- Рекомендации: profiler.get_optimization_recommendations()")

if __name__ == "__main__":
    main()

