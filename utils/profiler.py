#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль профилирования производительности для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для профилирования, 
анализа производительности и оптимизации кода проекта.
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
import memory_profiler
import line_profiler
import tracemalloc
from contextlib import contextmanager


@dataclass
class PerformanceMetric:
    """Метрика производительности"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    context: str


class Profiler:
    """
    Класс профилировщика
    Обеспечивает профилирование, анализ 
    производительности и оптимизацию кода.
    """
    
    def __init__(self, output_dir: str = "profiles"):
        """
        Инициализирует профилировщик
        
        Args:
            output_dir: Директория для сохранения результатов профилирования
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics = []
        self.profile_stats = {}
        self.memory_snapshots = []
        self.cpu_monitoring = False
        self.memory_monitoring = False
        self.monitoring_thread = None
        self.monitoring_data = {
            'timestamps': [],
            'cpu_percent': [],
            'memory_percent': [],
            'memory_mb': []
        }
    
    def start_cpu_monitoring(self):
        """Запускает мониторинг CPU"""
        if self.cpu_monitoring:
            return
        
        self.cpu_monitoring = True
        
        def monitor():
            while self.cpu_monitoring:
                self.monitoring_data['timestamps'].append(datetime.now())
                self.monitoring_data['cpu_percent'].append(psutil.cpu_percent())
                self.monitoring_data['memory_percent'].append(psutil.virtual_memory().percent)
                self.monitoring_data['memory_mb'].append(psutil.virtual_memory().used / (1024*1024))
                time.sleep(1)
        
        self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self.monitoring_thread.start()
    
    def stop_cpu_monitoring(self):
        """Останавливает мониторинг CPU"""
        self.cpu_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Профилирует выполнение функции
        
        Args:
            func: Функция для профилирования
            *args: Аргументы функции
            **kwargs: Именованные аргументы функции
            
        Returns:
            Словарь с результатами профилирования
        """
        profiler = cProfile.Profile()
        
        start_time = time.time()
        profiler.enable()
        
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()
            end_time = time.time()
        
        # Сохраняем статистику
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('cumulative')
        stats.print_stats()
        
        execution_time = end_time - start_time
        
        profile_data = {
            'function_name': func.__name__,
            'execution_time': execution_time,
            'profile_stats': stream.getvalue(),
            'call_count': stats.total_calls,
            'primitive_calls': stats.primitive_calls
        }
        
        return profile_data
    
    def profile_memory_usage(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Профилирует использование памяти функцией
        
        Args:
            func: Функция для профилирования
            *args: Аргументы функции
            **kwargs: Именованные аргументы функции
            
        Returns:
            Словарь с результатами профилирования памяти
        """
        # Начинаем отслеживание памяти
        tracemalloc.start()
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        memory_data = {
            'function_name': func.__name__,
            'execution_time': end_time - start_time,
            'current_memory_mb': current / (1024 * 1024),
            'peak_memory_mb': peak / (1024 * 1024),
            'result': result
        }
        
        return memory_data
    
    def profile_line_by_line(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Профилирует функцию построчно
        
        Args:
            func: Функция для профилирования
            *args: Аргументы функции
            **kwargs: Именованные аргументы функции
            
        Returns:
            Словарь с результатами построчного профилирования
        """
        # Создаем временную функцию для профилирования
        def temp_func():
            return func(*args, **kwargs)
        
        # Используем line_profiler
        profiler = line_profiler.LineProfiler()
        profiler.add_function(temp_func)
        profiler.enable_by_count()
        
        result = temp_func()
        
        profiler.disable_by_count()
        
        # Получаем результаты
        stream = io.StringIO()
        profiler.print_stats(stream=stream)
        
        line_profile_data = {
            'function_name': func.__name__,
            'line_profile_stats': stream.getvalue(),
            'result': result
        }
        
        return line_profile_data
    
    def benchmark_function(self, func: Callable, iterations: int = 100, *args, **kwargs) -> Dict[str, Any]:
        """
        Бенчмаркинг функции
        
        Args:
            func: Функция для бенчмаркинга
            iterations: Количество итераций
            *args: Аргументы функции
            **kwargs: Именованные аргументы функции
            
        Returns:
            Словарь с результатами бенчмаркинга
        """
        times = []
        
        for i in range(iterations):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        
        benchmark_data = {
            'function_name': func.__name__,
            'iterations': iterations,
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'std_dev': std_dev,
            'total_time': sum(times),
            'ops_per_sec': iterations / sum(times) if sum(times) > 0 else 0
        }
        
        return benchmark_data
    
    def analyze_system_resources(self) -> Dict[str, Any]:
        """
        Анализирует системные ресурсы
        
        Returns:
            Словарь с информацией о системных ресурсах
        """
        process = psutil.Process(os.getpid())
        
        system_info = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'disk_usage_percent': psutil.disk_usage('/').percent,
            'process_memory_mb': process.memory_info().rss / (1024 * 1024),
            'process_cpu_percent': process.cpu_percent(),
            'num_threads': process.num_threads(),
            'num_fds': process.num_fds() if os.name != 'nt' else 'N/A',
            'timestamp': datetime.now().isoformat()
        }
        
        return system_info
    
    def generate_performance_report(self, profile_data: Dict[str, Any], 
                                 output_path: str = None) -> str:
        """
        Генерирует отчет о производительности
        
        Args:
            profile_data: Данные профилирования
            output_path: Путь для сохранения отчета (если None, генерируется автоматически)
            
        Returns:
            Путь к созданному отчету
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"performance_report_{timestamp}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'profile_data': profile_data,
            'system_info': self.analyze_system_resources(),
            'summary': self._generate_summary(profile_data)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        return str(output_path)
    
    def _generate_summary(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Генерирует сводку по данным профилирования
        
        Args:
            profile_data: Данные профилирования
            
        Returns:
            Сводка по профилированию
        """
        summary = {}
        
        if 'execution_time' in profile_data:
            summary['execution_time'] = profile_data['execution_time']
        
        if 'avg_time' in profile_data:
            summary['avg_execution_time'] = profile_data['avg_time']
            summary['operations_per_second'] = profile_data.get('ops_per_sec', 0)
        
        if 'current_memory_mb' in profile_data:
            summary['memory_usage_mb'] = profile_data['current_memory_mb']
            summary['peak_memory_mb'] = profile_data['peak_memory_mb']
        
        if 'call_count' in profile_data:
            summary['function_calls'] = profile_data['call_count']
        
        return summary
    
    def visualize_performance_data(self, profile_data: Dict[str, Any], 
                                output_path: str = None) -> str:
        """
        Визуализирует данные производительности
        
        Args:
            profile_data: Данные профилирования
            output_path: Путь для сохранения графика (если None, генерируется автоматически)
            
        Returns:
            Путь к созданному графику
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"performance_chart_{timestamp}.png"
        
        plt.figure(figsize=(12, 8))
        
        # Создаем подграфики для разных метрик
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Анализ производительности', fontsize=16)
        
        # График времени выполнения (если есть)
        if 'avg_time' in profile_data and 'min_time' in profile_data and 'max_time' in profile_data:
            labels = ['Avg', 'Min', 'Max']
            values = [profile_data['avg_time'], profile_data['min_time'], profile_data['max_time']]
            axes[0, 0].bar(labels, values)
            axes[0, 0].set_title('Время выполнения (с)')
            axes[0, 0].set_ylabel('Время (с)')
        
        # График использования памяти (если есть)
        if 'current_memory_mb' in profile_data and 'peak_memory_mb' in profile_data:
            labels = ['Current', 'Peak']
            values = [profile_data['current_memory_mb'], profile_data['peak_memory_mb']]
            axes[0, 1].bar(labels, values)
            axes[0, 1].set_title('Использование памяти (MB)')
            axes[0, 1].set_ylabel('Память (MB)')
        
        # График количества вызовов (если есть)
        if 'call_count' in profile_data:
            axes[1, 0].pie([profile_data['call_count'], 100 - profile_data['call_count']], 
                          labels=['Function Calls', 'Other'], autopct='%1.1f%%')
            axes[1, 0].set_title('Распределение вызовов функций')
        
        # График производительности (если есть)
        if 'ops_per_sec' in profile_data:
            axes[1, 1].bar(['Ops/sec'], [profile_data['ops_per_sec']])
            axes[1, 1].set_title('Операций в секунду')
            axes[1, 1].set_ylabel('Операций/с')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)


def profile_performance(func: Callable) -> Callable:
    """
    Декоратор для профилирования производительности функции
    
    Args:
        func: Функция для профилирования
        
    Returns:
        Обернутая функция с профилированием
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = Profiler()
        
        print(f"Начинаем профилирование функции {func.__name__}...")
        
        # Профилируем выполнение
        profile_result = profiler.profile_function(func, *args, **kwargs)
        
        # Генерируем отчет
        report_path = profiler.generate_performance_report(profile_result)
        
        print(f"Профилирование завершено. Отчет сохранен: {report_path}")
        
        # Возвращаем результат функции
        return profile_result.get('result', None) if 'result' in profile_result else func(*args, **kwargs)
    
    return wrapper


def benchmark_function(iterations: int = 100):
    """
    Декоратор для бенчмаркинга функции
    
    Args:
        iterations: Количество итераций для бенчмаркинга
        
    Returns:
        Декоратор для бенчмаркинга
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = Profiler()
            
            print(f"Запускаем бенчмарк функции {func.__name__} ({iterations} итераций)...")
            
            benchmark_result = profiler.benchmark_function(func, iterations, *args, **kwargs)
            
            print(f"Среднее время: {benchmark_result['avg_time']:.6f} с")
            print(f"Минимальное время: {benchmark_result['min_time']:.6f} с")
            print(f"Максимальное время: {benchmark_result['max_time']:.6f} с")
            print(f"Операций в секунду: {benchmark_result['ops_per_sec']:.2f}")
            
            # Генерируем отчет
            report_path = profiler.generate_performance_report(benchmark_result)
            print(f"Отчет бенчмарка сохранен: {report_path}")
            
            return benchmark_result.get('result', None) if 'result' in benchmark_result else func(*args, **kwargs)
        
        return wrapper
    return decorator


@contextmanager
def performance_monitor(name: str = "Operation"):
    """
    Контекстный менеджер для мониторинга производительности
    
    Args:
        name: Название операции для мониторинга
    """
    profiler = Profiler()
    
    start_time = time.time()
    tracemalloc.start()
    
    print(f"Начинаем мониторинг производительности: {name}")
    
    try:
        yield
    finally:
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        execution_time = end_time - start_time
        
        print(f"Мониторинг завершен: {name}")
        print(f"Время выполнения: {execution_time:.4f} с")
        print(f"Пиковое использование памяти: {peak / (1024*1024):.2f} MB")
        
        # Сохраняем метрики
        metric = PerformanceMetric(
            name=f"{name}_execution_time",
            value=execution_time,
            unit="seconds",
            timestamp=datetime.now(),
            context=name
        )
        
        # Здесь можно добавить сохранение метрики в файл или базу данных


def main():
    """Главная функция для демонстрации возможностей профилировщика"""
    print("=== ПРОФИЛИРОВЩИК ПРОИЗВОДИТЕЛЬНОСТИ ПРОЕКТА ===")
    
    # Создаем профилировщик
    profiler = Profiler()
    
    print("✓ Профилировщик инициализирован")
    print(f"✓ Директория вывода: {profiler.output_dir}")
    
    # Пример функции для профилирования
    def sample_function(n: int = 10000):
        """Пример функции для профилирования"""
        result = []
        for i in range(n):
            result.append(i ** 2)
        return sum(result)
    
    # Профилируем функцию
    print("\nПрофилирование функции...")
    profile_result = profiler.profile_function(sample_function, 10000)
    print(f"  - Время выполнения: {profile_result['execution_time']:.4f} с")
    print(f"  - Количество вызовов: {profile_result['call_count']}")
    
    # Профилируем использование памяти
    print("\nПрофилирование использования памяти...")
    memory_result = profiler.profile_memory_usage(sample_function, 10000)
    print(f"  - Текущее использование памяти: {memory_result['current_memory_mb']:.2f} MB")
    print(f"  - Пиковое использование памяти: {memory_result['peak_memory_mb']:.2f} MB")
    
    # Бенчмаркинг функции
    print("\nБенчмаркинг функции...")
    benchmark_result = profiler.benchmark_function(sample_function, iterations=10, n=5000)
    print(f"  - Среднее время: {benchmark_result['avg_time']:.6f} с")
    print(f"  - Операций в секунду: {benchmark_result['ops_per_sec']:.2f}")
    
    # Анализ системных ресурсов
    print("\nАнализ системных ресурсов...")
    system_info = profiler.analyze_system_resources()
    print(f"  - Загрузка CPU: {system_info['cpu_percent']}%")
    print(f"  - Использование памяти: {system_info['memory_percent']}%")
    print(f"  - Память процесса: {system_info['process_memory_mb']:.2f} MB")
    
    # Генерируем отчет
    print("\nГенерация отчета о производительности...")
    report_path = profiler.generate_performance_report(profile_result)
    print(f"  - Отчет сохранен: {report_path}")
    
    # Создаем визуализацию
    print("\nСоздание визуализации производительности...")
    chart_path = profiler.visualize_performance_data(benchmark_result)
    print(f"  - График сохранен: {chart_path}")
    
    # Демонстрируем декоратор профилирования
    print("\nДемонстрация декоратора профилирования...")
    
    @profile_performance
    def decorated_function():
        time.sleep(0.1)  # Имитация работы
        return "Результат функции"
    
    result = decorated_function()
    
    # Демонстрируем декоратор бенчмаркинга
    print("\nДемонстрация декоратора бенчмаркинга...")
    
    @benchmark_function(iterations=5)
    def benchmarked_function():
        return sum(i**2 for i in range(1000))
    
    result = benchmarked_function()
    
    # Демонстрируем контекстный менеджер
    print("\nДемонстрация контекстного менеджера производительности...")
    with performance_monitor("Тестовая операция"):
        time.sleep(0.05)  # Имитация работы
        result = sum(i**3 for i in range(1000))
    
    print("\nПрофилировщик производительности успешно протестирован")
    print("\nДоступные функции:")
    print("- Профилирование функций: profile_function()")
    print("- Профилирование памяти: profile_memory_usage()")
    print("- Бенчмаркинг: benchmark_function()")
    print("- Анализ системных ресурсов: analyze_system_resources()")
    print("- Генерация отчетов: generate_performance_report()")
    print("- Визуализация данных: visualize_performance_data()")
    print("- Декоратор профилирования: @profile_performance")
    print("- Декоратор бенчмаркинга: @benchmark_function")
    print("- Контекстный менеджер: performance_monitor")


if __name__ == "__main__":
    main()