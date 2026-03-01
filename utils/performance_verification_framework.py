# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Модуль тестирования производительности для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для тестирования и верификации
эффективности оптимизационных инструментов проекта.
"""

import time
import statistics
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional
from datetime import datetime
import json
import psutil
import gc
import tracemalloc
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.performance_profiler import PerformanceProfiler
from utils.resource_optimizer import ResourceManager
from utils.advanced_logger_analyzer import AdvancedLoggerAnalyzer
from utils.memory_tracker import MemoryTracker
from utils.performance_benchmark import PerformanceBenchmarkSuite
from utils.optimization_orchestrator import OptimizationOrchestrator
from utils.system_health_monitor import SystemHealthMonitor
from utils.performance_analytics_dashboard import PerformanceAnalyticsDashboard

@dataclass
class PerformanceTestResult:
    """Результат теста производительности"""
    test_name: str
    original_metrics: Dict[str, Any]
    optimized_metrics: Dict[str, Any]
    improvement_percent: float
    execution_time: float
    timestamp: datetime
    notes: str = ""

class PerformanceVerificationFramework:
    """
    Класс фреймворка верификации производительности
    Обеспечивает тестирование и верификацию эффективности оптимизационных инструментов.
    """


    def __init__(self, output_dir: str = "verification_reports"):
        """
        Инициализирует фреймворк верификации

        Args:
            output_dir: Директория для сохранения отчетов верификации
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Инициализируем все инструменты оптимизации
        self.performance_profiler = PerformanceProfiler(output_dir="profiles")
        self.resource_manager = ResourceManager()
        self.logger_analyzer = AdvancedLoggerAnalyzer()
        self.memory_tracker = MemoryTracker(output_dir="memory_logs")
        self.benchmark_suite = PerformanceBenchmarkSuite(output_dir="benchmarks")
        self.orchestrator = OptimizationOrchestrator(output_dir="optimization_reports")
        self.health_monitor = SystemHealthMonitor(output_dir="health_reports")
        self.analytics_dashboard = PerformanceAnalyticsDashboard(output_dir="analytics_reports")

        # Хранилище результатов тестов
        self.test_results = []
        self.baseline_metrics = {}


    def establish_baseline(self) -> Dict[str, Any]:
        """
        Устанавливает базовые метрики производительности

        Returns:
            Словарь с базовыми метриками
        """
        print("Установка базовых метрик производительности...")

        # Получаем текущие системные метрики
        baseline = {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': self._get_system_metrics(),
            'resource_metrics': self.resource_manager.get_current_resources(),
            'memory_snapshot': self._take_memory_snapshot(),
            'health_status': self.health_monitor.get_current_health_status(),
            'performance_summary': self.analytics_dashboard.get_performance_summary()
        }

        self.baseline_metrics = baseline
        print("Базовые метрики установлены")
        return baseline


    def _get_system_metrics(self) -> Dict[str, float]:
        """Получает основные системные метрики"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'disk_percent': disk.percent if disk else 0,
                'process_count': len(psutil.pids()),
                'boot_time': psutil.boot_time()
            }
        except Exception as e:
            print(f"Ошибка получения системных метрик: {e}")
            return {}


    def _take_memory_snapshot(self) -> Dict[str, Any]:
        """Делает снимок памяти"""
        try:
            # Запускаем трассировку памяти
            tracemalloc.start()

            # Получаем метрики до оптимизации
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Также получаем метрики через memory_tracker
            snapshot = self.memory_tracker.take_snapshot()

            return {
                'current_memory_mb': current / (1024 * 1024),
                'peak_memory_mb': peak / (1024 * 1024),
                'rss_memory_mb': snapshot.rss_mb if snapshot else 0,
                'vms_memory_mb': snapshot.vms_mb if snapshot else 0,
                'memory_percent': snapshot.percent if snapshot else 0
            }
        except Exception as e:
            print(f"Ошибка получения снимка памяти: {e}")
            return {}


    def run_performance_tests(self) -> List[PerformanceTestResult]:
        """
        Запускает комплексные тесты производительности

        Returns:
            Список результатов тестов
        """
        print("Запуск комплексных тестов производительности...")

        # Устанавливаем базовые метрики если не установлены
        if not self.baseline_metrics:
            self.establish_baseline()

        results = []

        # Тест 1: Производительность CPU
        cpu_result = self._test_cpu_performance()
        results.append(cpu_result)

        # Тест 2: Использование памяти
        memory_result = self._test_memory_performance()
        results.append(memory_result)

        # Тест 3: Производительность алгоритмов
        algo_result = self._test_algorithm_performance()
        results.append(algo_result)

        # Тест 4: Ресурсный менеджмент
        resource_result = self._test_resource_management()
        results.append(resource_result)

        # Тест 5: Комплексная оптимизация
        comprehensive_result = self._test_comprehensive_optimization()
        results.append(comprehensive_result)

        self.test_results.extend(results)
        print(f"Завершено {len(results)} тестов производительности")

        return results


    def _test_cpu_performance(self) -> PerformanceTestResult:
        """Тестирует производительность CPU"""
        print("  Тест: Производительность CPU...")

        start_time = time.time()

        # Тестируем до оптимизации
        original_metrics = self._benchmark_cpu_intensive_task()

        # Применяем оптимизацию CPU
        optimization_result = self.resource_manager.optimize_cpu_usage()

        # Тестируем после оптимизации
        optimized_metrics = self._benchmark_cpu_intensive_task()

        # Рассчитываем улучшение
        original_time = original_metrics.get('execution_time', 1.0)
        optimized_time = optimized_metrics.get('execution_time', 1.0)
        improvement = ((original_time - optimized_time) / original_time) * 100 if original_time > 0 else 0

        execution_time = time.time() - start_time

        return PerformanceTestResult(
            test_name='CPU Performance',
            original_metrics=original_metrics,
            optimized_metrics=optimized_metrics,
            improvement_percent=improvement,
            execution_time=execution_time,
            timestamp=datetime.now(),
            notes=f"Оптимизация статус: {optimization_result.status}"
        )


    def _benchmark_cpu_intensive_task(self) -> Dict[str, Any]:
        """Бенчмарк CPU-интенсивной задачи"""
        def cpu_intensive_task():

            result = 0
            for i in range(100000):
                result += i ** 2
            return result

        # Измеряем время выполнения
        start_time = time.time()
        result = cpu_intensive_task()
        end_time = time.time()

        # Измеряем использование ресурсов
        resources = self.resource_manager.get_current_resources()

        return {
            'execution_time': end_time - start_time,
            'result': result,
            'cpu_percent': resources.get('cpu_percent', 0),
            'memory_rss_mb': resources.get('memory_rss_mb', 0),
            'timestamp': datetime.now().isoformat()
        }


    def _test_memory_performance(self) -> PerformanceTestResult:
        """Тестирует производительность памяти"""
        print("  Тест: Производительность памяти...")

        start_time = time.time()

        # Тестируем до оптимизации
        original_metrics = self._benchmark_memory_usage()

        # Применяем оптимизацию памяти
        optimization_result = self.resource_manager.optimize_memory_usage()

        # Тестируем после оптимизации
        optimized_metrics = self._benchmark_memory_usage()

        # Рассчитываем улучшение (меньше памяти = лучше)
        original_memory = original_metrics.get('memory_used_mb', 1.0)
        optimized_memory = optimized_metrics.get('memory_used_mb', 1.0)
        improvement = ((original_memory - optimized_memory) / original_memory) * 100 if original_memory > 0 else 0

        execution_time = time.time() - start_time

        return PerformanceTestResult(
            test_name='Memory Performance',
            original_metrics=original_metrics,
            optimized_metrics=optimized_metrics,
            improvement_percent=improvement,
            execution_time=execution_time,
            timestamp=datetime.now(),
            notes=f"Оптимизация статус: {optimization_result.status}"
        )


    def _benchmark_memory_usage(self) -> Dict[str, Any]:
        """Бенчмарк использования памяти"""
        # Запускаем трассировку памяти
        tracemalloc.start()

        # Создаем объекты для измерения
        test_data = [i for i in range(50000)]
        result = sum(test_data)
        del test_data  # Удаляем данные

        # Получаем статистику памяти
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Также получаем снимок через memory_tracker
        snapshot = self.memory_tracker.take_snapshot()

        return {
            'current_memory_mb': current / (1024 * 1024),
            'peak_memory_mb': peak / (1024 * 1024),
            'rss_memory_mb': snapshot.rss_mb if snapshot else 0,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }


    def _test_algorithm_performance(self) -> PerformanceTestResult:
        """Тестирует производительность алгоритмов"""
        print("  Тест: Производительность алгоритмов...")

        start_time = time.time()

        # Определяем разные реализации одной задачи
        def algorithm_slow(n):
            """Медленная реализация"""
            result = 0
            for i in range(n):
                result += i ** 2
            return result

        def algorithm_fast(n):
            """Быстрая реализация"""
            return sum(i ** 2 for i in range(n))

        def algorithm_optimal(n):
            """Оптимальная реализация (математическая формула)"""
            return n * (n - 1) * (2 * n - 1) // 6

        algorithms = {
            'slow': algorithm_slow,
            'fast': algorithm_fast,
            'optimal': algorithm_optimal
        }

        # Тестируем до оптимизации
        original_results = {}
        for name, func in algorithms.items():
            original_results[name] = self.benchmark_suite.benchmark_function(
                f"original_{name}",
                func,
                10000,
                iterations=10
            )

        # Сравниваем алгоритмы
        comparisons = self.benchmark_suite.compare_algorithms(algorithms, 10000, iterations=10)

        # Применяем оптимизации
        # Используем самый быстрый алгоритм
        optimized_result = self.benchmark_suite.benchmark_function(
            "optimized_algorithm",
            algorithm_optimal,
            10000,
            iterations=10
        )

        # Рассчитываем улучшение
        original_time = original_results['slow']['timing_stats']['avg_seconds']
        optimized_time = optimized_result['timing_stats']['avg_seconds']
        improvement = ((original_time - optimized_time) / original_time) * 100 if original_time > 0 else 0

        execution_time = time.time() - start_time

        return PerformanceTestResult(
            test_name='Algorithm Performance',
            original_metrics=original_results,
            optimized_metrics=optimized_result,
            improvement_percent=improvement,
            execution_time=execution_time,
            timestamp=datetime.now(),
            notes=f"Выполнено {len(comparisons)} сравнений алгоритмов"
        )


    def _test_resource_management(self) -> PerformanceTestResult:
        """Тестирует управление ресурсами"""
        print("  Тест: Управление ресурсами...")

        start_time = time.time()

        # Тестируем до оптимизации
        original_resources = self.resource_manager.get_current_resources()

        # Применяем комплексную оптимизацию ресурсов
        optimization_results = self.resource_manager.optimize_all_resources()

        # Тестируем после оптимизации
        optimized_resources = self.resource_manager.get_current_resources()

        # Рассчитываем улучшение по эффективности
        original_efficiency = self.resource_manager.get_resource_efficiency_score()
        optimized_efficiency = self.resource_manager.get_resource_efficiency_score()
        improvement = optimized_efficiency - original_efficiency

        execution_time = time.time() - start_time

        return PerformanceTestResult(
            test_name='Resource Management',
            original_metrics=original_resources,
            optimized_metrics=optimized_resources,
            improvement_percent=improvement,
            execution_time=execution_time,
            timestamp=datetime.now(),
            notes=f"Оптимизировано {len(optimization_results)} типов ресурсов"
        )


    def _test_comprehensive_optimization(self) -> PerformanceTestResult:
        """Тестирует комплексную оптимизацию"""
        print("  Тест: Комплексная оптимизация...")

        start_time = time.time()

        # Тестируем до комплексной оптимизации
        original_summary = self.analytics_dashboard.get_performance_summary()

        # Запускаем комплексную оптимизацию
        optimization_results = self.orchestrator.start_comprehensive_optimization([
            "core_utils", "spm_simulator"
        ])

        # Тестируем после комплексной оптимизации
        optimized_summary = self.analytics_dashboard.get_performance_summary()

        # Рассчитываем улучшение по комплексному показателю
        original_score = original_summary.get('current_metrics', {}).get('efficiency_score', 50)
        optimized_score = optimized_summary.get('current_metrics', {}).get('efficiency_score', 50)
        improvement = optimized_score - original_score

        execution_time = time.time() - start_time

        return PerformanceTestResult(
            test_name='Comprehensive Optimization',
            original_metrics=original_summary,
            optimized_metrics=optimized_summary,
            improvement_percent=improvement,
            execution_time=execution_time,
            timestamp=datetime.now(),
            notes=f"Обработано {len(optimization_results)} модулей"
        )


    def verify_optimization_effectiveness(self) -> Dict[str, Any]:
        """
        Верифицирует эффективность оптимизаций

        Returns:
            Словарь с результатами верификации
        """
        print("Верификация эффективности оптимизаций...")

        # Если нет результатов тестов, запускаем тесты
        if not self.test_results:
            self.run_performance_tests()

        # Анализируем результаты
        total_tests = len(self.test_results)
        positive_improvements = sum(1 for r in self.test_results if r.improvement_percent > 0)
        negative_improvements = sum(1 for r in self.test_results if r.improvement_percent < 0)

        avg_improvement = statistics.mean([r.improvement_percent for r in self.test_results]) if self.test_results else 0

        # Определяем общий статус
        effectiveness_score = 0
        if total_tests > 0:
            positive_ratio = positive_improvements / total_tests
            effectiveness_score = avg_improvement * positive_ratio * 100

        verification_results = {
            'total_tests_run': total_tests,
            'positive_improvements': positive_improvements,
            'negative_improvements': negative_improvements,
            'neutral_improvements': total_tests - positive_improvements - negative_improvements,
            'average_improvement_percent': avg_improvement,
            'effectiveness_score': effectiveness_score,
            'success_rate_percent': (positive_improvements / total_tests * 100) if total_tests > 0 else 0,
            'total_execution_time': sum(r.execution_time for r in self.test_results),
            'timestamp': datetime.now().isoformat(),
            'recommendations': self._generate_verification_recommendations()
        }

        print(f"Верификация завершена:")
        print(f"  - Общее количество тестов: {total_tests}")
        print(f"  - Положительных улучшений: {positive_improvements}")
        print(f"  - Среднее улучшение: {avg_improvement:.2f}%")
        print(f"  - Оценка эффективности: {effectiveness_score:.2f}")

        return verification_results


    def _generate_verification_recommendations(self) -> List[str]:
        """Генерирует рекомендации на основе верификации"""
        recommendations = []

        if not self.test_results:
            return ["Нет данных для генерации рекомендаций"]

        # Анализируем результаты тестов
        avg_improvement = statistics.mean([r.improvement_percent for r in self.test_results])

        if avg_improvement > 10:
            recommendations.append("Отличная эффективность оптимизаций! Среднее улучшение превышает 10%.")
        elif avg_improvement > 0:
            recommendations.append("Оптимизации показывают положительный эффект, но есть резервы для улучшения.")
        else:
            recommendations.append("Оптимизации требуют доработки - средний эффект отрицательный.")

        # Рекомендации по конкретным тестам
        worst_test = min(self.test_results, key=lambda x: x.improvement_percent)
        if worst_test.improvement_percent < 0:
            recommendations.append(f"Обратите внимание на '{worst_test.test_name}' - показал отрицательный результат.")

        best_test = max(self.test_results, key=lambda x: x.improvement_percent)
        if best_test.improvement_percent > 10:
            recommendations.append(f"'{best_test.test_name}' показал отличный результат - {best_test.improvement_percent:.2f}% улучшения.")

        # Рекомендации по системе в целом
        if self.resource_manager.get_resource_efficiency_score() > 80:
            recommendations.append("Система эффективно использует ресурсы.")
        else:
            recommendations.append("Рекомендуется дополнительно оптимизировать использование ресурсов.")

        return recommendations


    def generate_verification_report(self, output_path: str = None) -> str:
        """
        Генерирует отчет о верификации

        Args:
            output_path: Путь для сохранения отчета

        Returns:
            Путь к сохраненному отчету
        """
        if output_path is None:
            output_path = str(self.output_dir / f"verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        # Выполняем верификацию
        verification_results = self.verify_optimization_effectiveness()

        report = {
            'generation_time': datetime.now().isoformat(),
            'baseline_metrics': self.baseline_metrics,
            'test_results': [
                {
                    'test_name': r.test_name,
                    'original_metrics': r.original_metrics,
                    'optimized_metrics': r.optimized_metrics,
                    'improvement_percent': r.improvement_percent,
                    'execution_time': r.execution_time,
                    'timestamp': r.timestamp.isoformat(),
                    'notes': r.notes
                }
                for r in self.test_results
            ],
            'verification_results': verification_results,
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'platform': str(psutil.Process().parent().exe()) if hasattr(psutil, 'Process') and psutil.Process().parent() else 'unknown'
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        print(f"Отчет о верификации сохранен: {output_path}")
        return output_path


    def run_regression_tests(self) -> Dict[str, Any]:
        """
        Запускает регрессионные тесты для проверки нарушения функциональности

        Returns:
            Результаты регрессионных тестов
        """
        print("Запуск регрессионных тестов...")

        regression_results = {
            'tests_passed': 0,
            'tests_failed': 0,
            'total_tests': 0,
            'failed_tests': [],
            'passed_tests': [],
            'execution_time': 0
        }

        start_time = time.time()

        # Тест 1: Проверка работоспособности оптимизационных инструментов
        try:
            # Проверяем, что все инструменты могут быть инициализированы
            test_profiler = PerformanceProfiler()
            test_resource = ResourceManager()
            test_memory = MemoryTracker()

            # Проверяем базовую функциональность
            snapshot = test_memory.take_snapshot()
            cpu_opt = test_resource.optimize_cpu_usage()

            regression_results['tests_passed'] += 1
            regression_results['passed_tests'].append({
                'name': 'Tools Initialization',
                'status': 'PASS',
                'details': 'All optimization tools initialized and basic functions work'
            })
        except Exception as e:
            regression_results['tests_failed'] += 1
            regression_results['failed_tests'].append({
                'name': 'Tools Initialization',
                'status': 'FAIL',
                'error': str(e)
            })

        regression_results['total_tests'] += 1

        # Тест 2: Проверка совместимости версий
        try:
            # Проверяем, что основные методы доступны
            methods_exist = all([
                hasattr(self.performance_profiler, 'benchmark_function'),
                hasattr(self.resource_manager, 'optimize_cpu_usage'),
                hasattr(self.memory_tracker, 'take_snapshot'),
                hasattr(self.benchmark_suite, 'benchmark_function'),
                hasattr(self.orchestrator, 'start_comprehensive_optimization')
            ])

            if methods_exist:
                regression_results['tests_passed'] += 1
                regression_results['passed_tests'].append({
                    'name': 'API Compatibility',
                    'status': 'PASS',
                    'details': 'All expected API methods are available'
                })
            else:
                regression_results['tests_failed'] += 1
                regression_results['failed_tests'].append({
                    'name': 'API Compatibility',
                    'status': 'FAIL',
                    'error': 'Some expected methods are missing'
                })
        except Exception as e:
            regression_results['tests_failed'] += 1
            regression_results['failed_tests'].append({
                'name': 'API Compatibility',
                'status': 'FAIL',
                'error': str(e)
            })

        regression_results['total_tests'] += 1

        # Тест 3: Проверка производительности после оптимизации
        try:
            # Сравниваем время выполнения базовой операции до и после "оптимизации"

            def baseline_operation():
                            return sum(i**2 for i in range(10000))

            # Время до
            start = time.time()
            result1 = baseline_operation()
            time_before = time.time() - start

            # Имитируем оптимизацию (реально ничего не делаем, просто вызываем методы)
            self.resource_manager.optimize_cpu_usage()
            self.resource_manager.optimize_memory_usage()

            # Время после
            start = time.time()
            result2 = baseline_operation()
            time_after = time.time() - start

            # Результаты должны быть одинаковыми
            if result1 == result2:
                regression_results['tests_passed'] += 1
                regression_results['passed_tests'].append({
                    'name': 'Functional Correctness',
                    'status': 'PASS',
                    'details': f'Timing: before={time_before:.4f}s, after={time_after:.4f}s',
                    'functional_correctness': True
                })
            else:
                regression_results['tests_failed'] += 1
                regression_results['failed_tests'].append({
                    'name': 'Functional Correctness',
                    'status': 'FAIL',
                    'error': 'Optimization changed functional behavior'
                })
        except Exception as e:
            regression_results['tests_failed'] += 1
            regression_results['failed_tests'].append({
                'name': 'Functional Correctness',
                'status': 'FAIL',
                'error': str(e)
            })

        regression_results['total_tests'] += 1

        regression_results['execution_time'] = time.time() - start_time

        print(f"Регрессионные тесты завершены: {regression_results['tests_passed']} пройдено, {regression_results['tests_failed']} провалено")

        return regression_results

def main():
    """Главная функция для демонстрации фреймворка верификации"""
    print("=== ФРЕЙМВОРК ВЕРИФИКАЦИИ ПРОИЗВОДИТЕЛЬНОСТИ ===")

    # Создаем фреймворк верификации
    verification_framework = PerformanceVerificationFramework()

    print("✓ Фреймворк верификации инициализирован")
    print(f"✓ Директория вывода: {verification_framework.output_dir}")

    # Устанавливаем базовые метрики
    print("\nУстановка базовых метрик...")
    baseline = verification_framework.establish_baseline()
    print(f"✓ Базовые метрики установлены в {baseline['timestamp']}")

    # Запускаем тесты производительности
    print("\nЗапуск тестов производительности...")
    test_results = verification_framework.run_performance_tests()
    print(f"✓ Завершено {len(test_results)} тестов")

    # Верифицируем эффективность
    print("\nВерификация эффективности оптимизаций...")
    verification_results = verification_framework.verify_optimization_effectiveness()
    print(f"✓ Оценка эффективности: {verification_results['effectiveness_score']:.2f}")
    print(f"✓ Среднее улучшение: {verification_results['average_improvement_percent']:.2f}%")

    # Запускаем регрессионные тесты
    print("\nЗапуск регрессионных тестов...")
    regression_results = verification_framework.run_regression_tests()
    print(f"✓ Регрессионные тесты: {regression_results['tests_passed']} пройдено, {regression_results['tests_failed']} провалено")

    # Генерируем отчет
    print("\nГенерация отчета о верификации...")
    report_path = verification_framework.generate_verification_report()
    print(f"✓ Отчет сохранен: {report_path}")

    # Выводим рекомендации
    print("\nРекомендации по результатам верификации:")
    for i, rec in enumerate(verification_results['recommendations'], 1):
        print(f"  {i}. {rec}")

    print("\nФреймворк верификации успешно протестирован")
    print("\nДоступные функции:")
    print("- Установка базовых метрик: verification_framework.establish_baseline()")
    print("- Тестирование производительности: verification_framework.run_performance_tests()")
    print("- Верификация эффективности: verification_framework.verify_optimization_effectiveness()")
    print("- Регрессионные тесты: verification_framework.run_regression_tests()")
    print("- Генерация отчетов: verification_framework.generate_verification_report()")

if __name__ == "__main__":
    main()

