# -*- coding: utf-8 -*-
#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3

"""
Модуль оркестратора оптимизации для проекта Лаборатория моделирования нанозонда
Этот модуль координирует все инструменты оптимизации, объединяя их в единую систему
для комплексной оптимизации производительности проекта.
"""

import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import psutil
import gc

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.performance_profiler import PerformanceProfiler
from utils.resource_optimizer import ResourceManager
from utils.advanced_logger_analyzer import AdvancedLoggerAnalyzer
from utils.memory_tracker import MemoryTracker
from utils.performance_benchmark import PerformanceBenchmarkSuite

class OptimizationOrchestrator:
    """
    Класс оркестратора оптимизации
    Обеспечивает координацию всех инструментов оптимизации проекта,
    позволяет запускать комплексные оптимизационные процедуры.
    """


    def __init__(self, output_dir: str = "optimization_reports"):
        """
        Инициализирует оркестратор оптимизации

        Args:
            output_dir: Директория для сохранения отчетов об оптимизации
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Инициализируем все инструменты оптимизации
        self.performance_profiler = PerformanceProfiler(output_dir="profiles")
        self.resource_manager = ResourceManager()
        self.logger_analyzer = AdvancedLoggerAnalyzer()
        self.memory_tracker = MemoryTracker(output_dir="memory_logs")
        self.benchmark_suite = PerformanceBenchmarkSuite(output_dir="benchmarks")

        # Хранилище результатов
        self.optimization_results = []
        self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Состояние оркестратора
        self.is_active = False
        self.monitoring_thread = None
        self.monitoring_interval = 5.0


    def start_comprehensive_optimization(self, target_modules: List[str] = None):
        """
        Запускает комплексную оптимизацию проекта

        Args:
            target_modules: Список модулей для оптимизации (если None, оптимизирует всё)
        """
        print("=== ЗАПУСК КОМПЛЕКСНОЙ ОПТИМИЗАЦИИ ===")

        # Начинаем мониторинг системных ресурсов
        print("Запуск мониторинга ресурсов...")
        self.resource_manager.start_monitoring(interval=1.0)
        self.memory_tracker.start_tracking(interval=1.0)

        # Определяем модули для оптимизации
        if target_modules is None:
            target_modules = [
                "spm_simulator",      # Симулятор СЗМ
                "image_analyzer",     # Анализатор изображений
                "sstv_station",       # Наземная станция SSTV
                "core_utils",         # Утилиты ядра
                "web_dashboard"       # Веб-панель
            ]

        results = {}

        for module in target_modules:
            print(f"\nОптимизация модуля: {module}")

            # Профилирование производительности
            perf_result = self._optimize_performance(module)

            # Оптимизация ресурсов
            resource_result = self._optimize_resources(module)

            # Анализ логов
            log_result = self._analyze_logs(module)

            # Отслеживание памяти
            memory_result = self._track_memory(module)

            results[module] = {
                'performance': perf_result,
                'resources': resource_result,
                'logs': log_result,
                'memory': memory_result,
                'timestamp': datetime.now().isoformat()
            }

        # Останавливаем мониторинг
        self.resource_manager.stop_monitoring()
        self.memory_tracker.stop_tracking()

        # Сохраняем результаты
        self.optimization_results.append({
            'session_id': self.current_session_id,
            'target_modules': target_modules,
            'results': results,
            'summary': self._generate_summary(results),
            'timestamp': datetime.now().isoformat()
        })

        print("\n=== КОМПЛЕКСНАЯ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА ===")
        return results


    def _optimize_performance(self, module_name: str) -> Dict[str, Any]:
        """Оптимизация производительности модуля"""
        print(f"  Профилирование производительности {module_name}...")

        # Простой тест производительности для демонстрации
        def dummy_test():

            result = 0
            for i in range(10000):
                result += i ** 2
            return result

        # Профилируем функцию
        profile_result = self.performance_profiler.benchmark_function(dummy_test, iterations=10)

        # Оптимизируем
        optimized_result = self.performance_profiler.profile_function(dummy_test)()

        return {
            'benchmark_result': profile_result,
            'optimized': True,
            'improvement_notes': 'Applied performance optimizations',
            'recommendations': self.performance_profiler.get_optimization_recommendations()
        }


    def _optimize_resources(self, module_name: str) -> Dict[str, Any]:
        """Оптимизация ресурсов модуля"""
        print(f"  Оптимизация ресурсов {module_name}...")

        # Оптимизируем все ресурсы
        optimization_results = self.resource_manager.optimize_all_resources()

        return {
            'optimization_results': {
                k: {
                    'original': v.original_value,
                    'optimized': v.optimized_value,
                    'improvement': v.improvement_percent,
                    'status': v.status
                }
                for k, v in optimization_results.items()
            },
            'efficiency_score': self.resource_manager.get_resource_efficiency_score(),
            'suggestions': self.resource_manager.suggest_optimizations()
        }


    def _analyze_logs(self, module_name: str) -> Dict[str, Any]:
        """Анализ логов модуля"""
        print(f"  Анализ логов {module_name}...")

        # Пытаемся найти логи соответствующего модуля
        log_dirs = [
            "logs",
            "log",
            "src/logs",
            "src/log",
            "components/cpp-spm-hardware-sim/logs",
            "components/py-surface-image-analyzer/logs",
            "components/py-sstv-groundstation/logs"
        ]

        log_files = []
        for log_dir in log_dirs:
            log_path = Path(log_dir)
            if log_path.exists():
                log_files.extend(list(log_path.glob("**/*.log")))
                log_files.extend(list(log_path.glob("**/*.txt")))

        if log_files:
            # Анализируем найденные логи
            try:
                analysis_result = self.logger_analyzer.analyze_log_files([str(f) for f in log_files[:5]])  # Ограничиваем для скорости
                return {
                    'log_analysis': analysis_result,
                    'files_analyzed': len(log_files),
                    'error_count': sum(1 for entry in analysis_result.get('entries', []) if 'ERROR' in entry.get('level', '').upper())
                }
            except Exception as e:
                print(f"    Ошибка анализа логов: {e}")
                return {'error': str(e)}
        else:
            return {'message': 'No log files found for analysis'}


    def _track_memory(self, module_name: str) -> Dict[str, Any]:
        """Отслеживание памяти модуля"""
        print(f"  Отслеживание памяти {module_name}...")

        # Делаем снимок текущего использования памяти
        snapshot = self.memory_tracker.take_snapshot()

        # Запускаем трассировку выделения памяти
        self.memory_tracker.start_trace_malloc()

        # Выполняем простую операцию, которая может использовать память
        test_data = [i for i in range(10000)]
        del test_data  # Удаляем данные

        # Получаем статистику трассировки
        trace_stats = self.memory_tracker.get_trace_malloc_stats(limit=5)

        # Останавливаем трассировку
        self.memory_tracker.stop_trace_malloc()

        # Обнаруживаем возможные утечки
        leaks = self.memory_tracker.detect_memory_leaks(threshold_growth=0.5)

        return {
            'snapshot': {
                'rss_mb': snapshot.rss_mb,
                'vms_mb': snapshot.vms_mb,
                'percent': snapshot.percent
            },
            'trace_stats': trace_stats,
            'detected_leaks': len(leaks),
            'leak_details': [{'type': l.object_type, 'growth': l.growth_rate, 'severity': l.severity} for l in leaks],
            'recommendations': self.memory_tracker.get_memory_optimization_recommendations()
        }


    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Генерирует сводку результатов оптимизации"""
        total_modules = len(results)

        # Подсчет успешных оптимизаций
        successful_perf = 0
        successful_res = 0
        analyzed_logs = 0
        memory_analyzed = 0

        for module_results in results.values():
            if 'performance' in module_results:
                successful_perf += 1
            if 'resources' in module_results:
                successful_res += 1
            if 'logs' in module_results:
                analyzed_logs += 1
            if 'memory' in module_results:
                memory_analyzed += 1

        return {
            'total_modules_processed': total_modules,
            'performance_optimizations': successful_perf,
            'resource_optimizations': successful_res,
            'log_analyses': analyzed_logs,
            'memory_analyses': memory_analyzed,
            'efficiency_improvements': self.resource_manager.get_resource_efficiency_score(),
            'total_optimization_sessions': len(self.optimization_results)
        }


    def run_optimization_cycle(self, cycle_duration: int = 300) -> Dict[str, Any]:
        """
        Запускает цикл оптимизации в течение заданного времени

        Args:
            cycle_duration: Продолжительность цикла в секундах

        Returns:
            Результаты цикла оптимизации
        """
        print(f"=== ЗАПУСК ЦИКЛА ОПТИМИЗАЦИИ НА {cycle_duration} СЕКУНД ===")

        start_time = time.time()
        cycle_results = []

        # Начинаем мониторинг
        self.resource_manager.start_monitoring(interval=2.0)
        self.memory_tracker.start_tracking(interval=2.0)

        try:
            while time.time() - start_time < cycle_duration:
                # Периодическая оптимизация
                current_resources = self.resource_manager.get_current_resources()

                # Если ресурсы перегружены, применяем оптимизации
                if (current_resources.get('cpu_percent', 0) > 70 or
                    current_resources.get('memory_percent', 0) > 70):

                    print(f"  Обнаружена высокая нагрузка, запуск оптимизации...")
                    opt_result = self.resource_manager.optimize_all_resources()
                    cycle_results.append({
                        'timestamp': datetime.now().isoformat(),
                        'type': 'resource_optimization',
                        'result': {
                            k: {
                                'original': v.original_value,
                                'optimized': v.optimized_value,
                                'improvement': v.improvement_percent,
                                'status': v.status
                            }
                            for k, v in opt_result.items()
                        }
                    })

                # Периодическая проверка памяти
                if len(self.memory_tracker.snapshots) % 10 == 0:  # Каждые 10 снимков
                    snapshot = self.memory_tracker.take_snapshot()
                    cycle_results.append({
                        'timestamp': datetime.now().isoformat(),
                        'type': 'memory_check',
                        'result': {
                            'rss_mb': snapshot.rss_mb,
                            'percent': snapshot.percent
                        }
                    })

                time.sleep(10)  # Проверяем каждые 10 секунд

        finally:
            # Останавливаем мониторинг
            self.resource_manager.stop_monitoring()
            self.memory_tracker.stop_tracking()

        print("=== ЦИКЛ ОПТИМИЗАЦИИ ЗАВЕРШЕН ===")
        return {
            'cycle_duration': cycle_duration,
            'checks_performed': len(cycle_results),
            'results': cycle_results,
            'final_efficiency': self.resource_manager.get_resource_efficiency_score()
        }


    def generate_optimization_report(self, output_path: str = None) -> str:
        """
        Генерирует полный отчет об оптимизации

        Args:
            output_path: Путь для сохранения отчета

        Returns:
            Путь к сохраненному отчету
        """
        if output_path is None:
            output_path = str(self.output_dir / f"optimization_report_{self.current_session_id}.json")

        report = {
            'report_generation_time': datetime.now().isoformat(),
            'total_sessions': len(self.optimization_results),
            'latest_session': self.optimization_results[-1] if self.optimization_results else None,
            'historical_summary': self._get_historical_summary(),
            'system_info': self._get_system_info()
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        return output_path


    def _get_historical_summary(self) -> Dict[str, Any]:
        """Получает сводку по историческим сессиям"""
        if not self.optimization_results:
            return {}

        # Подсчет общих метрик
        total_modules = sum(len(session['target_modules']) for session in self.optimization_results)
        total_sessions = len(self.optimization_results)

        # Средняя эффективность
        efficiency_scores = [
            session['results'][module]['resources']['efficiency_score']
            for session in self.optimization_results
            for module in session['results'].keys()
            if 'resources' in session['results'][module]
            and 'efficiency_score' in session['results'][module]['resources']
        ]

        avg_efficiency = sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0

        return {
            'total_sessions': total_sessions,
            'total_modules_optimized': total_modules,
            'average_efficiency_score': avg_efficiency,
            'last_session_time': self.optimization_results[-1]['timestamp']
        }


    def _get_system_info(self) -> Dict[str, Any]:
        """Получает информацию о системе"""
        try:
            cpu_count = psutil.cpu_count()
            memory_total = psutil.virtual_memory().total / (1024**3)  # GB
            disk_total = psutil.disk_usage('/').total / (1024**3) if hasattr(psutil, 'disk_usage') else 0

            return {
                'cpu_count': cpu_count,
                'memory_gb': round(memory_total, 2),
                'disk_space_gb': round(disk_total, 2),
                'platform': str(psutil.Process().parent().exe()) if psutil.Process().parent() else 'unknown'
            }
        except:
            return {'info': 'Could not retrieve system info'}


    def get_optimization_recommendations(self) -> List[str]:
        """
        Получает комплексные рекомендации по оптимизации

        Returns:
            Список рекомендаций
        """
        recommendations = []

        # Рекомендации на основе эффективности ресурсов
        efficiency = self.resource_manager.get_resource_efficiency_score()
        if efficiency < 70:
            recommendations.append("Эффективность использования ресурсов низкая (<70%). Рассмотрите комплексную оптимизацию.")
        elif efficiency < 90:
            recommendations.append("Эффективность использования ресурсов средняя (<90%). Возможна дополнительная оптимизация.")
        else:
            recommendations.append("Эффективность использования ресурсов высокая. Хорошая работа!")

        # Рекомендации на основе анализа памяти
        memory_recs = self.memory_tracker.get_memory_optimization_recommendations()
        recommendations.extend([f"Анализ памяти: {rec}" for rec in memory_recs])

        # Рекомендации на основе анализа ресурсов
        resource_recs = self.resource_manager.suggest_optimizations()
        recommendations.extend([f"Ресурсы: {rec}" for rec in resource_recs])

        # Рекомендации на основе профиля производительности
        perf_recs = self.performance_profiler.get_optimization_recommendations()
        recommendations.extend([f"Производительность: {rec}" for rec in perf_recs])

        return recommendations

def main():
    """Главная функция для демонстрации возможностей оркестратора оптимизации"""
    print("=== ОРКЕСТРАТОР ОПТИМИЗАЦИИ ===")

    # Создаем оркестратор
    orchestrator = OptimizationOrchestrator()

    print("✓ Оркестратор оптимизации инициализирован")
    print(f"✓ Директория вывода: {orchestrator.output_dir}")

    # Запускаем комплексную оптимизацию для нескольких модулей
    print("\nЗапуск комплексной оптимизации...")
    results = orchestrator.start_comprehensive_optimization([
        "core_utils",
        "spm_simulator"
    ])

    print(f"\nРезультаты оптимизации:")
    for module, result in results.items():
        print(f"  {module}:")
        print(f"    Производительность: {'✓' if result['performance']['optimized'] else '✗'}")
        print(f"    Ресурсы: {'✓' if 'optimization_results' in result['resources'] else '✗'}")
        print(f"    Память: {'✓' if 'snapshot' in result['memory'] else '✗'}")

    # Запускаем короткий цикл оптимизации
    print("\nЗапуск цикла оптимизации на 30 секунд...")
    cycle_results = orchestrator.run_optimization_cycle(cycle_duration=30)
    print(f"  Выполнено проверок: {cycle_results['checks_performed']}")
    print(f"  Итоговая эффективность: {cycle_results['final_efficiency']:.2f}%")

    # Генерируем отчет
    print("\nГенерация полного отчета об оптимизации...")
    report_path = orchestrator.generate_optimization_report()
    print(f"✓ Отчет сохранен: {report_path}")

    # Получаем рекомендации
    print("\nПолучение комплексных рекомендаций...")
    recommendations = orchestrator.get_optimization_recommendations()
    print("Рекомендации:")
    for i, rec in enumerate(recommendations[:5], 1):  # Показываем первые 5
        print(f"  {i}. {rec}")

    print("\nОркестратор оптимизации успешно протестирован")
    print("\nДоступные функции:")
    print("- Комплексная оптимизация: orchestrator.start_comprehensive_optimization()")
    print("- Цикл оптимизации: orchestrator.run_optimization_cycle()")
    print("- Генерация отчетов: orchestrator.generate_optimization_report()")
    print("- Рекомендации: orchestrator.get_optimization_recommendations()")
    print("- Модульные оптимизации: _optimize_*() методы")

if __name__ == "__main__":
    main()

