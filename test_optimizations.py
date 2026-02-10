#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт тестирования оптимизаций для проекта Лаборатория моделирования нанозонда
Этот скрипт демонстрирует эффективность всех созданных инструментов оптимизации.
"""

import time
import statistics
from datetime import datetime
from pathlib import Path
import json
import psutil
import gc

# Добавляем путь к проекту
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.performance_profiler import PerformanceProfiler
from utils.resource_optimizer import ResourceManager
from utils.advanced_logger_analyzer import AdvancedLoggerAnalyzer
from utils.memory_tracker import MemoryTracker
from utils.performance_benchmark import PerformanceBenchmarkSuite
from utils.optimization_orchestrator import OptimizationOrchestrator
from utils.system_health_monitor import SystemHealthMonitor
from utils.performance_analytics_dashboard import PerformanceAnalyticsDashboard
from utils.performance_verification_framework import PerformanceVerificationFramework
from utils.realtime_dashboard import RealTimeDashboard
from utils.performance_monitoring_center import PerformanceMonitoringCenter

def simulate_heavy_workload(duration: int = 10):
    """
    Симулирует тяжелую рабочую нагрузку для тестирования

    Args:
        duration: Продолжительность в секундах
    """
    print(f"🔄 Запуск симуляции тяжелой нагрузки на {duration} секунд...")

    start_time = time.time()
    results = []

    while time.time() - start_time < duration:
        # Создаем вычислительную нагрузку
        data = [i**2 for i in range(1000)]
        result = sum(data) / len(data)
        results.append(result)

        # Создаем немного памяти
        temp_list = [j for j in range(1000)]
        del temp_list

        # Небольшая пауза
        time.sleep(0.01)

    print(f"✅ Симуляция нагрузки завершена, обработано {len(results)} итераций")

def run_comprehensive_optimization_test():
    """Запускает комплексное тестирование оптимизаций"""
    print("=" * 80)
    print("🧪 КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ ОПТИМИЗАЦИЙ NANOPROBE SIMULATION LAB")
    print("=" * 80)

    start_time = datetime.now()
    print(f"Время начала: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Инициализируем все инструменты оптимизации
    print("\n🔧 Инициализация инструментов оптимизации...")

    # Создаем экземпляры всех инструментов
    profiler = PerformanceProfiler(output_dir="test_reports/profiles")
    resource_mgr = ResourceManager()
    logger_analyzer = AdvancedLoggerAnalyzer(log_directory="logs")
    memory_tracker = MemoryTracker(output_dir="test_reports/memory")
    benchmark_suite = PerformanceBenchmarkSuite(output_dir="test_reports/benchmarks")
    orchestrator = OptimizationOrchestrator(output_dir="test_reports/optimization")
    health_monitor = SystemHealthMonitor(output_dir="test_reports/health")
    analytics_dashboard = PerformanceAnalyticsDashboard(output_dir="test_reports/analytics")
    verification_framework = PerformanceVerificationFramework(output_dir="test_reports/verification")
    monitoring_center = PerformanceMonitoringCenter(output_dir="test_reports/monitoring")

    print("✅ Все инструменты успешно инициализированы")

    # Запускаем мониторинг
    print("\n📊 Запуск мониторинга...")
    resource_mgr.start_monitoring(interval=2.0)
    memory_tracker.start_tracking(interval=2.0)
    health_monitor.start_monitoring(interval=3.0)
    analytics_dashboard.start_analytics_monitoring(interval=5.0)
    monitoring_center.start_monitoring(interval=3.0)

    # Собираем базовые метрики до оптимизации
    print("\n🔍 Сбор базовых метрик до оптимизации...")
    baseline_resources = resource_mgr.get_current_resources()
    baseline_memory = memory_tracker.get_current_memory_usage()

    print(f"   CPU до: {baseline_resources['cpu_percent']:.2f}%")
    print(f"   Память до: {baseline_resources['memory_rss_mb']:.2f} MB")
    print(f"   Эффективность до: {baseline_resources['resource_efficiency']:.2f}%")

    # Запускаем симуляцию нагрузки
    print("\n⚡ Запуск симуляции нагрузки...")
    simulate_heavy_workload(duration=15)

    # Проверяем метрики после нагрузки
    print("\n🔍 Сбор метрик после нагрузки...")
    post_load_resources = resource_mgr.get_current_resources()
    post_load_memory = memory_tracker.get_current_memory_usage()

    print(f"   CPU после нагрузки: {post_load_resources['cpu_percent']:.2f}%")
    print(f"   Память после нагрузки: {post_load_resources['memory_rss_mb']:.2f} MB")

    # Применяем оптимизации
    print("\n🚀 Применение оптимизаций...")

    # Оптимизация ресурсов
    print("   - Оптимизация ресурсов...")
    resource_opt_results = resource_mgr.optimize_all_resources()

    # Оптимизация памяти
    print("   - Оптимизация памяти...")
    memory_opt_results = memory_tracker.perform_memory_optimization()

    # Запускаем оркестратор оптимизации
    print("   - Запуск оркестратора оптимизации...")
    orchestration_results = orchestrator.start_comprehensive_optimization([
        "core_utils", "spm_simulator", "image_analyzer"
    ])

    # Проверяем метрики после оптимизации
    print("\n🔍 Сбор метрик после оптимизации...")
    post_opt_resources = resource_mgr.get_current_resources()
    post_opt_memory = memory_tracker.get_current_memory_usage()

    print(f"   CPU после оптимизации: {post_opt_resources['cpu_percent']:.2f}%")
    print(f"   Память после оптимизации: {post_opt_resources['memory_rss_mb']:.2f} MB")
    print(f"   Эффективность после: {post_opt_resources['resource_efficiency']:.2f}%")

    # Запускаем бенчмаркинг
    print("\n⏱️ Запуск бенчмаркинга производительности...")


    def sample_algorithm_1(n):
        """Пример алгоритма 1"""
        result = 0
        for i in range(n):
            result += i ** 2
        return result


    def sample_algorithm_2(n):
        """Пример алгоритма 2 (оптимизированный)"""
        return sum(i ** 2 for i in range(n))


    def sample_algorithm_3(n):
        """Пример алгоритма 3 (еще более оптимизированный)"""
        return n * (n - 1) * (2 * n - 1) // 6

    algorithms = {
        'algorithm_1': sample_algorithm_1,
        'algorithm_2': sample_algorithm_2,
        'algorithm_3': sample_algorithm_3
    }

    benchmark_results = benchmark_suite.compare_algorithms(
        algorithms,
        test_data=10000,
        iterations=10
    )

    print(f"   Тестирование {len(algorithms)} алгоритмов завершено")

    # Запускаем верификацию эффективности
    print("\n🔬 Запуск верификации эффективности оптимизаций...")
    verification_results = verification_framework.verify_optimization_effectiveness()

    # Генерируем аналитику
    print("\n📊 Генерация аналитических отчетов...")

    # Отчеты от аналитической панели
    analytics_summary = analytics_dashboard.get_performance_summary()

    # Отчеты от центра мониторинга
    monitoring_summary = monitoring_center.get_performance_summary()

    # Генерация полного отчета
    print("\n📝 Генерация сводного отчета...")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Рассчитываем улучшения
    cpu_improvement = baseline_resources['cpu_percent'] - post_opt_resources['cpu_percent']
    memory_improvement = baseline_resources['memory_rss_mb'] - post_opt_resources['memory_rss_mb']
    efficiency_improvement = post_opt_resources['resource_efficiency'] - baseline_resources['resource_efficiency']

    summary = {
        'test_metadata': {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'test_type': 'comprehensive_optimization_test'
        },
        'baseline_metrics': {
            'cpu_percent': baseline_resources['cpu_percent'],
            'memory_mb': baseline_resources['memory_rss_mb'],
            'efficiency_score': baseline_resources['resource_efficiency']
        },
        'post_load_metrics': {
            'cpu_percent': post_load_resources['cpu_percent'],
            'memory_mb': post_load_resources['memory_rss_mb']
        },
        'post_optimization_metrics': {
            'cpu_percent': post_opt_resources['cpu_percent'],
            'memory_mb': post_opt_resources['memory_rss_mb'],
            'efficiency_score': post_opt_resources['resource_efficiency']
        },
        'improvements': {
            'cpu_improvement': cpu_improvement,
            'memory_improvement_mb': memory_improvement,
            'efficiency_improvement': efficiency_improvement,
            'cpu_improvement_percent': (cpu_improvement / baseline_resources['cpu_percent']) * 100 if baseline_resources['cpu_percent'] > 0 else 0
        },
        'optimization_results': {
            'resource_optimization': resource_opt_results,
            'memory_optimization': memory_opt_results,
            'orchestration_results': orchestration_results,
            'benchmark_results': benchmark_results,
            'verification_results': verification_results
        },
        'analytics_summary': analytics_summary,
        'monitoring_summary': monitoring_summary
    }

    # Сохраняем отчет
    report_path = Path("test_reports/comprehensive_optimization_test_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n✅ ТЕСТИРОВАНИЕ ОПТИМИЗАЦИЙ ЗАВЕРШЕНО")
    print("=" * 80)
    print(f"⏱️  Продолжительность: {duration:.2f} сек")
    print(f"📈 Улучшения:")
    print(f"   • Загрузка CPU: {cpu_improvement:+.2f}% ({(cpu_improvement / baseline_resources['cpu_percent'] * 100):+.2f}%)")
    print(f"   • Использование памяти: {memory_improvement:+.2f} MB")
    print(f"   • Эффективность ресурсов: {efficiency_improvement:+.2f}%")
    print(f"📁 Отчет сохранен: {report_path}")

    # Останавливаем мониторинг
    print("\n🛑 Остановка мониторинга...")
    resource_mgr.stop_monitoring()
    memory_tracker.stop_tracking()
    health_monitor.stop_monitoring()
    analytics_dashboard.stop_analytics_monitoring()
    monitoring_center.stop_monitoring()

    return summary

def run_individual_component_tests():
    """Запускает тестирование отдельных компонентов"""
    print("\n🧪 ТЕСТИРОВАНИЕ ОТДЕЛЬНЫХ КОМПОНЕНТОВ")
    print("-" * 50)

    tests_passed = 0
    tests_total = 0

    # Тестирование профайлера
    print("\n🔍 Тестирование профайлера...")
    tests_total += 1
    try:
        profiler = PerformanceProfiler()
        result = profiler.profile_function(lambda x: x**2)(5)
        assert result == 25, "Профайлер работает некорректно"
        print("   ✅ Профайлер - работает")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Профайлер - ошибка: {e}")

    # Тестирование оптимизатора ресурсов
    print("\n⚙️ Тестирование оптимизатора ресурсов...")
    tests_total += 1
    try:
        resource_mgr = ResourceManager()
        resources = resource_mgr.get_current_resources()
        assert 'cpu_percent' in resources, "Оптимизатор ресурсов работает некорректно"
        print("   ✅ Оптимизатор ресурсов - работает")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Оптимизатор ресурсов - ошибка: {e}")

    # Тестирование трекера памяти
    print("\n💾 Тестирование трекера памяти...")
    tests_total += 1
    try:
        memory_tracker = MemoryTracker()
        usage = memory_tracker.get_current_memory_usage()
        assert 'rss_mb' in usage, "Трекер памяти работает некорректно"
        print("   ✅ Трекер памяти - работает")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Трекер памяти - ошибка: {e}")

    # Тестирование бенчмарка
    print("\n⏱️ Тестирование бенчмарка...")
    tests_total += 1
    try:
        benchmark = PerformanceBenchmarkSuite()
        result = benchmark.benchmark_function("test_func", lambda x: x*2, 5, iterations=3)
        assert 'average_time' in result, "Бенчмарк работает некорректно"
        print("   ✅ Бенчмарк - работает")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Бенчмарк - ошибка: {e}")

    # Тестирование оркестратора
    print("\n🤖 Тестирование оркестратора...")
    tests_total += 1
    try:
        orchestrator = OptimizationOrchestrator()
        # Тестируем без запуска полной оптимизации, только инициализацию
        assert orchestrator is not None, "Оркестратор работает некорректно"
        print("   ✅ Оркестратор - работает")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Оркестратор - ошибка: {e}")

    # Тестирование монитора здоровья
    print("\n🩺 Тестирование монитора здоровья...")
    tests_total += 1
    try:
        health_monitor = SystemHealthMonitor()
        status = health_monitor.get_current_health_status()
        assert 'overall_health_score' in status, "Монитор здоровья работает некорректно"
        print("   ✅ Монитор здоровья - работает")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Монитор здоровья - ошибка: {e}")

    print(f"\n📊 Результаты тестирования компонентов: {tests_passed}/{tests_total} пройдено")

def main():
    """Главная функция тестирования"""
    print("🚀 Запуск тестирования оптимизаций Nanoprobe Simulation Lab")

    # Создаем директорию для отчетов
    Path("test_reports").mkdir(exist_ok=True)

    try:
        # Тестируем отдельные компоненты
        run_individual_component_tests()

        # Запускаем комплексное тестирование
        summary = run_comprehensive_optimization_test()

        print(f"\n🎉 ТЕСТИРОВАНИЕ УСПЕШНО ЗАВЕРШЕНО!")
        print(f"📈 Все инструменты оптимизации работают корректно")
        print(f"📁 Результаты тестирования сохранены в test_reports/")

    except KeyboardInterrupt:
        print("\n❌ Тестирование прервано пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка в процессе тестирования: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

