# -*- coding: utf-8 -*-
#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3

"""
Мастер-скрипт комплексной оптимизации для проекта Лаборатория моделирования нанозонда
Этот скрипт демонстрирует интеграцию всех инструментов оптимизации
и предоставляет единый интерфейс для комплексной оптимизации системы.
"""

import time
from pathlib import Path
from datetime import datetime
import json

# Импортируем все инструменты оптимизации
from utils.performance_profiler import PerformanceProfiler
from utils.resource_optimizer import ResourceManager
from utils.advanced_logger_analyzer import AdvancedLoggerAnalyzer
from utils.memory_tracker import MemoryTracker
from utils.performance_benchmark import PerformanceBenchmarkSuite
from utils.optimization_orchestrator import OptimizationOrchestrator
from utils.system_health_monitor import SystemHealthMonitor
from utils.performance_analytics_dashboard import PerformanceAnalyticsDashboard

def run_comprehensive_optimization():
    """Запускает комплексную оптимизацию системы"""
    print("=" * 80)
    print("🚀 КОМПЛЕКСНАЯ ОПТИМИЗАЦИЯ СИСТЕМЫ NANOPROBE SIMULATION LAB")
    print("=" * 80)

    start_time = datetime.now()
    print(f"Время начала: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Инициализируем все инструменты
    print("\n🔧 Инициализация инструментов оптимизации...")
    profiler = PerformanceProfiler(output_dir="reports/profiles")
    resource_mgr = ResourceManager()
    logger_analyzer = AdvancedLoggerAnalyzer(log_directory="logs")
    memory_tracker = MemoryTracker(output_dir="reports/memory")
    benchmark_suite = PerformanceBenchmarkSuite(output_dir="reports/benchmarks")
    orchestrator = OptimizationOrchestrator(output_dir="reports/optimization")
    health_monitor = SystemHealthMonitor(output_dir="reports/health")
    analytics_dashboard = PerformanceAnalyticsDashboard(output_dir="reports/analytics")

    print("✅ Все инструменты успешно инициализированы")

    # Запускаем мониторинг здоровья системы
    print("\n🩺 Запуск мониторинга здоровья системы...")
    health_monitor.start_monitoring(interval=5.0)

    # Запускаем аналитическую панель
    print("📊 Запуск аналитической панели...")
    analytics_dashboard.start_analytics_monitoring(interval=10.0)

    # Запускаем мониторинг ресурсов
    print("📈 Запуск мониторинга ресурсов...")
    resource_mgr.start_monitoring(interval=3.0)

    # Запускаем мониторинг памяти
    print("💾 Запуск мониторинга памяти...")
    memory_tracker.start_tracking(interval=3.0)

    print("\n⏳ Сбор начальных метрик...")
    time.sleep(5)  # Даём время для сбора начальных данных

    # Запускаем оркестратор оптимизации
    print("\n🎯 Запуск оркестратора оптимизации...")
    optimization_results = orchestrator.start_comprehensive_optimization([
        "core_utils", "spm_simulator", "image_analyzer"
    ])

    print("✅ Оптимизация завершена")

    # Останавливаем мониторинг
    print("\n🛑 Остановка мониторинга...")
    health_monitor.stop_monitoring()
    analytics_dashboard.stop_analytics_monitoring()
    resource_mgr.stop_monitoring()
    memory_tracker.stop_tracking()

    # Генерируем отчеты
    print("\n📝 Генерация отчетов...")

    # Отчет оркестратора
    orchestrator_report = orchestrator.generate_optimization_report()
    print(f"  - Отчет оркестратора: {orchestrator_report}")

    # Отчет здоровья системы
    health_report = health_monitor.generate_health_report()
    print(f"  - Отчет здоровья: {health_report}")

    # Аналитический отчет
    analytics_report = analytics_dashboard.generate_analytics_report()
    print(f"  - Аналитический отчет: {analytics_report}")

    # Отчет производительности
    perf_report = profiler.generate_performance_report()
    print(f"  - Отчет производительности: {perf_report}")

    # Отчет памяти
    mem_report = memory_tracker.save_memory_report()
    print(f"  - Отчет памяти: {mem_report}")

    # Отчет бенчмарков
    bench_report = benchmark_suite.generate_performance_report()
    print(f"  - Отчет бенчмарков: {bench_report}")

    # Получаем рекомендации
    print("\n💡 Рекомендации по оптимизации:")

    orchestrator_recs = orchestrator.get_optimization_recommendations()
    health_recs = health_monitor.get_health_recommendations()
    analytics_recs = analytics_dashboard.get_optimization_suggestions()

    print(f"  Оркестратор: {len(orchestrator_recs)} рекомендаций")
    print(f"  Монитор здоровья: {len(health_recs)} рекомендаций")
    print(f"  Аналитическая панель: {len(analytics_recs)} категорий рекомендаций")

    # Показываем топ-5 критических инсайтов
    critical_insights = analytics_dashboard.get_actionable_insights('high')
    print(f"\n🔍 Топ-5 критических инсайтов:")
    for i, insight in enumerate(critical_insights[:5], 1):
        print(f"  {i}. [{insight.severity.upper()}] {insight.title}")
        print(f"     Рекомендация: {insight.recommendation}")

    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n🏁 КОМПЛЕКСНАЯ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
    print(f"Время окончания: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Общая продолжительность: {duration}")

    # Сводка
    summary = {
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_seconds': duration.total_seconds(),
        'reports_generated': 7,
        'recommendations_provided': len(orchestrator_recs) + len(health_recs),
        'critical_insights_found': len(critical_insights),
        'optimization_modules_used': [
            'Performance Profiler',
            'Resource Optimizer',
            'Logger Analyzer',
            'Memory Tracker',
            'Benchmark Suite',
            'Optimization Orchestrator',
            'Health Monitor',
            'Analytics Dashboard'
        ]
    }

    # Сохраняем сводку
    summary_path = f"reports/comprehensive_optimization_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n📋 Сводка сохранена: {summary_path}")

    return summary

def main():
    """Основная функция запуска комплексной оптимизации"""
    # Создаем директорию для отчетов
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    try:
        summary = run_comprehensive_optimization()

        print("\n" + "=" * 80)
        print("🎉 КОМПЛЕКСНАЯ ОПТИМИЗАЦИЯ УСПЕШНО ЗАВЕРШЕНА!")
        print("=" * 80)

        print("\n📦 СОЗДАННЫЕ ИНСТРУМЕНТЫ ОПТИМИЗАЦИИ:")
        print("   1. Профилировщик производительности - Глубокий анализ функций и кода")
        print("   2. Оптимизатор ресурсов - Управление CPU, памятью и другими ресурсами")
        print("   3. Анализатор логов - Обнаружение аномалий и проблем в логах")
        print("   4. Трекер памяти - Мониторинг утечек и потребления памяти")
        print("   5. Бенчмаркинг - Сравнение алгоритмов и производительности")
        print("   6. Оркестратор - Координация всех инструментов оптимизации")
        print("   7. Монитор здоровья - Постоянный контроль состояния системы")
        print("   8. Аналитическая панель - Комплексный анализ и визуализация")

        print("\n📊 РЕЗУЛЬТАТЫ:")
        print(f"   • Сгенерировано отчетов: {summary['reports_generated']}")
        print(f"   • Найдено инсайтов: {summary['critical_insights_found']}")
        print(f"   • Продолжительность: {summary['duration_seconds']:.1f} сек")

        print("\n📈 РЕКОМЕНДАЦИИ:")
        print("   • Регулярно запускайте комплексную оптимизацию")
        print("   • Мониторьте критические инсайты и рекомендации")
        print("   • Используйте аналитическую панель для принятия решений")
        print("   • Применяйте оптимизации по результатам бенчмарков")

    except KeyboardInterrupt:
        print("\n❌ Оптимизация прервана пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка в процессе оптимизации: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

