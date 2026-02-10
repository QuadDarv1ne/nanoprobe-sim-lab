# -*- coding: utf-8 -*-
#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3

"""
Модуль оптимизации ресурсов для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для оптимизации системных ресурсов,
управления памятью, CPU и другими ресурсами проекта.
"""

import psutil
import gc
import threading
import time
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
import heapq

@dataclass
class ResourceAllocation:
    """Выделение ресурсов"""
    cpu_percent: float
    memory_mb: float
    disk_io_priority: int
    network_priority: int
    timestamp: datetime

@dataclass
class OptimizationResult:
    """Результат оптимизации"""
    resource_type: str
    original_value: float
    optimized_value: float
    improvement_percent: float
    timestamp: datetime
    status: str

class ResourceManager:
    """
    Класс менеджера ресурсов
    Обеспечивает оптимизацию системных ресурсов,
    управление памятью и распределение ресурсов.
    """


    def __init__(self):
        """Инициализирует менеджер ресурсов"""
        self.process = psutil.Process()
        self.resources_history = []
        self.optimization_results = []
        self.resource_limits = {
            'cpu_percent': 80.0,
            'memory_mb': 2048.0,
            'disk_io_priority': 3,
            'network_priority': 3
        }
        self.monitoring = False
        self.monitoring_thread = None
        self.resource_allocations = []


    def set_resource_limits(self, cpu_percent: float = None, memory_mb: float = None,
    """TODO: Add description"""

                          disk_io_priority: int = None, network_priority: int = None):
        """
        Устанавливает ограничения ресурсов

        Args:
            cpu_percent: Максимальный процент CPU
            memory_mb: Максимальное количество памяти в MB
            disk_io_priority: Приоритет дискового ввода-вывода
            network_priority: Приоритет сети
        """
        if cpu_percent is not None:
            self.resource_limits['cpu_percent'] = cpu_percent
        if memory_mb is not None:
            self.resource_limits['memory_mb'] = memory_mb
        if disk_io_priority is not None:
            self.resource_limits['disk_io_priority'] = disk_io_priority
        if network_priority is not None:
            self.resource_limits['network_priority'] = network_priority


    def get_current_resources(self) -> Dict[str, float]:
        """Получает текущее использование ресурсов"""
        try:
            memory_info = self.process.memory_info()
            return {
                'cpu_percent': self.process.cpu_percent(),
                'memory_rss_mb': memory_info.rss / (1024 * 1024),
                'memory_vms_mb': memory_info.vms / (1024 * 1024),
                'memory_percent': self.process.memory_percent(),
                'num_threads': self.process.num_threads(),
                'num_fds': self.process.num_fds() if hasattr(self.process, 'num_fds') else 0,
                'io_counters': self.process.io_counters()._asdict() if self.process.io_counters() else {},
                'connections': len(self.process.connections()),
                'open_files': len(self.process.open_files())
            }
        except Exception as e:
            print(f"Ошибка получения ресурсов: {e}")
            return {}


    def optimize_cpu_usage(self) -> OptimizationResult:
        """
        Оптимизирует использование CPU

        Returns:
            Результат оптимизации
        """
        current_cpu = self.process.cpu_percent()

        # Пытаемся понизить приоритет процесса если CPU загрузка высокая
        if current_cpu > self.resource_limits['cpu_percent']:
            try:
                # Понижаем приоритет процесса
                if sys.platform.startswith('win'):
                    import ctypes
                    # LOW_PRIORITY_CLASS = 0x00000040
                    ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(os.getpid()), 0x00000040)
                else:
                    os.nice(10)  # Повышаем nice значение (понижаем приоритет)

                # Ждем немного и проверяем результат
                time.sleep(0.1)
                new_cpu = self.process.cpu_percent()

                improvement = ((current_cpu - new_cpu) / current_cpu) * 100 if current_cpu > 0 else 0

                result = OptimizationResult(
                    resource_type='cpu',
                    original_value=current_cpu,
                    optimized_value=new_cpu,
                    improvement_percent=improvement,
                    timestamp=datetime.now(),
                    status='success' if new_cpu < current_cpu else 'no_improvement'
                )

                self.optimization_results.append(result)
                return result

            except Exception as e:
                print(f"Ошибка оптимизации CPU: {e}")
                result = OptimizationResult(
                    resource_type='cpu',
                    original_value=current_cpu,
                    optimized_value=current_cpu,
                    improvement_percent=0,
                    timestamp=datetime.now(),
                    status='error'
                )
                self.optimization_results.append(result)
                return result

        # Если CPU загрузка в норме
        result = OptimizationResult(
            resource_type='cpu',
            original_value=current_cpu,
            optimized_value=current_cpu,
            improvement_percent=0,
            timestamp=datetime.now(),
            status='normal'
        )
        self.optimization_results.append(result)
        return result


    def optimize_memory_usage(self) -> OptimizationResult:
        """
        Оптимизирует использование памяти

        Returns:
            Результат оптимизации
        """
        current_memory = self.process.memory_info().rss / (1024 * 1024)  # MB

        if current_memory > self.resource_limits['memory_mb']:
            # Пытаемся освободить память
            original_memory = current_memory

            # Вызываем garbage collector
            collected = gc.collect()

            # Ждем немного и проверяем результат
            time.sleep(0.1)
            new_memory = self.process.memory_info().rss / (1024 * 1024)  # MB

            improvement = ((original_memory - new_memory) / original_memory) * 100 if original_memory > 0 else 0

            result = OptimizationResult(
                resource_type='memory',
                original_value=original_memory,
                optimized_value=new_memory,
                improvement_percent=improvement,
                timestamp=datetime.now(),
                status='success' if new_memory < original_memory else 'partial_success'
            )

            self.optimization_results.append(result)
            return result

        # Если память в норме
        result = OptimizationResult(
            resource_type='memory',
            original_value=current_memory,
            optimized_value=current_memory,
            improvement_percent=0,
            timestamp=datetime.now(),
            status='normal'
        )
        self.optimization_results.append(result)
        return result


    def optimize_disk_io(self) -> OptimizationResult:
        """
        Оптимизирует дисковый ввод-вывод

        Returns:
            Результат оптимизации
        """
        # Получаем текущие счетчики IO
        io_counters = self.process.io_counters()
        if io_counters:
            original_write_bytes = io_counters.write_bytes
            original_read_bytes = io_counters.read_bytes
        else:
            original_write_bytes = 0
            original_read_bytes = 0

        # В этой реализации мы просто регистрируем текущее состояние
        # Реальная оптимизация IO зависит от конкретных операций

        result = OptimizationResult(
            resource_type='disk_io',
            original_value=original_write_bytes + original_read_bytes,
            optimized_value=original_write_bytes + original_read_bytes,
            improvement_percent=0,
            timestamp=datetime.now(),
            status='monitored'
        )

        self.optimization_results.append(result)
        return result


    def optimize_all_resources(self) -> Dict[str, OptimizationResult]:
        """
        Оптимизирует все ресурсы

        Returns:
            Результаты оптимизации для каждого типа ресурсов
        """
        results = {}

        results['cpu'] = self.optimize_cpu_usage()
        results['memory'] = self.optimize_memory_usage()
        results['disk_io'] = self.optimize_disk_io()

        return results


    def start_monitoring(self, interval: float = 1.0):
        """
        Запускает мониторинг ресурсов

        Args:
            interval: Интервал между измерениями (в секундах)
        """
        if self.monitoring:
            return

        self.monitoring = True

    """TODO: Add description"""

        def monitor():
            """TODO: Add description"""
            while self.monitoring:
                try:
                    # Получаем текущие ресурсы
                    resources = self.get_current_resources()

                    # Создаем выделение ресурсов
                    allocation = ResourceAllocation(
                        cpu_percent=resources.get('cpu_percent', 0),
                        memory_mb=resources.get('memory_rss_mb', 0),
                        disk_io_priority=self.resource_limits['disk_io_priority'],
                        network_priority=self.resource_limits['network_priority'],
                        timestamp=datetime.now()
                    )

                    self.resource_allocations.append(allocation)

                    # Автоматически оптимизируем если нужно
                    if resources.get('cpu_percent', 0) > self.resource_limits['cpu_percent'] or \
                       resources.get('memory_rss_mb', 0) > self.resource_limits['memory_mb']:
                        self.optimize_all_resources()

                    time.sleep(interval)

                except Exception as e:
                    print(f"Ошибка мониторинга ресурсов: {e}")
                    time.sleep(interval)

        self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self.monitoring_thread.start()


    def stop_monitoring(self):
        """Останавливает мониторинг ресурсов"""
        self.monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2)


    def get_resource_efficiency_score(self) -> float:
        """
        Получает оценку эффективности использования ресурсов

        Returns:
            Оценка эффективности (0-100)
        """
        if not self.resource_allocations:
            return 100.0  # Если нет данных, считаем идеальным

        # Рассчитываем средние значения
        avg_cpu = sum(a.cpu_percent for a in self.resource_allocations) / len(self.resource_allocations)
        avg_memory = sum(a.memory_mb for a in self.resource_allocations) / len(self.resource_allocations)

        # Рассчитываем эффективность (чем ниже использование, тем выше эффективность)
        cpu_efficiency = max(0, 100 - (avg_cpu / self.resource_limits['cpu_percent']) * 100)
        memory_efficiency = max(0, 100 - (avg_memory / self.resource_limits['memory_mb']) * 100)

        # Средняя эффективность
        efficiency = (cpu_efficiency + memory_efficiency) / 2

        return max(0, min(100, efficiency))


    def suggest_optimizations(self) -> List[str]:
        """
        Предлагает оптимизации

        Returns:
            Список предложений по оптимизации
        """
        suggestions = []

        current_resources = self.get_current_resources()
        cpu_usage = current_resources.get('cpu_percent', 0)
        memory_usage = current_resources.get('memory_rss_mb', 0)

        if cpu_usage > self.resource_limits['cpu_percent'] * 0.8:
            suggestions.append("Рассмотрите оптимизацию алгоритмов или использование многопоточности")
            suggestions.append("Проверьте наличие бесконечных циклов или частых операций")

        if memory_usage > self.resource_limits['memory_mb'] * 0.8:
            suggestions.append("Рассмотрите оптимизацию использования памяти")
            suggestions.append("Проверьте наличие утечек памяти")
            suggestions.append("Используйте генераторы вместо списков где возможно")

        if not suggestions:
            suggestions.append("Ресурсы используются эффективно")

        return suggestions


    def get_resource_report(self) -> Dict[str, Any]:
        """
        Получает отчет о ресурсах

        Returns:
            Отчет о ресурсах
        """
        current_resources = self.get_current_resources()

        return {
            'timestamp': datetime.now().isoformat(),
            'current_resources': current_resources,
            'limits': self.resource_limits,
            'efficiency_score': self.get_resource_efficiency_score(),
            'optimization_results_count': len(self.optimization_results),
            'resource_allocations_count': len(self.resource_allocations),
            'suggestions': self.suggest_optimizations(),
            'recent_optimizations': [
                {
                    'type': r.resource_type,
                    'original': r.original_value,
                    'optimized': r.optimized_value,
                    'improvement': r.improvement_percent,
                    'status': r.status,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in self.optimization_results[-5:]  # Последние 5 результатов
            ]
        }


    def save_report(self, output_path: str = None) -> str:
        """
        Сохраняет отчет о ресурсах

        Args:
            output_path: Путь для сохранения отчета

        Returns:
            Путь к сохраненному отчету
        """
        if output_path is None:
            output_path = f"resource_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        report = self.get_resource_report()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        return output_path

class AdaptiveResourceOptimizer:
    """
    Адаптивный оптимизатор ресурсов
    Автоматически адаптирует использование ресурсов
    на основе текущей нагрузки и требований.
    """


    def __init__(self):
        """Инициализирует адаптивный оптимизатор"""
        self.resource_manager = ResourceManager()
        self.adaptation_history = []
        self.performance_goals = {
            'response_time_ms': 100,
            'throughput_ops_per_sec': 1000,
            'error_rate_percent': 1.0,
            'resource_utilization_percent': 75.0
        }
        self.current_strategy = 'balanced'
        self.strategies = {
            'performance': {'cpu_priority': 1, 'memory_priority': 1, 'efficiency': 0.3},
            'efficiency': {'cpu_priority': 3, 'memory_priority': 3, 'efficiency': 0.9},
            'balanced': {'cpu_priority': 2, 'memory_priority': 2, 'efficiency': 0.6}
        }
    """TODO: Add description"""


    def set_performance_goals(self, response_time_ms: float = None,
                            throughput_ops_per_sec: float = None,
                            error_rate_percent: float = None,
                            resource_utilization_percent: float = None):
        """
        Устанавливает цели производительности

        Args:
            response_time_ms: Целевое время ответа в миллисекундах
            throughput_ops_per_sec: Целевая пропускная способность
            error_rate_percent: Целевой процент ошибок
            resource_utilization_percent: Целевое использование ресурсов
        """
        if response_time_ms is not None:
            self.performance_goals['response_time_ms'] = response_time_ms
        if throughput_ops_per_sec is not None:
            self.performance_goals['throughput_ops_per_sec'] = throughput_ops_per_sec
        if error_rate_percent is not None:
            self.performance_goals['error_rate_percent'] = error_rate_percent
        if resource_utilization_percent is not None:
            self.performance_goals['resource_utilization_percent'] = resource_utilization_percent


    def evaluate_performance(self) -> Dict[str, float]:
        """
        Оценивает текущую производительность

        Returns:
            Результаты оценки производительности
        """
        resources = self.resource_manager.get_current_resources()

        # Простая оценка на основе использования ресурсов
        cpu_score = min(100, (resources.get('cpu_percent', 0) / 100) * 100)
        memory_score = min(100, (resources.get('memory_percent', 0) / 100) * 100)

        # В реальной системе здесь будет более сложная логика оценки
        # на основе реальных метрик производительности

        return {
            'cpu_utilization': resources.get('cpu_percent', 0),
            'memory_utilization': resources.get('memory_percent', 0),
            'cpu_score': cpu_score,
            'memory_score': memory_score,
            'overall_score': (cpu_score + memory_score) / 2
        }


    def select_optimization_strategy(self) -> str:
        """
        Выбирает стратегию оптимизации на основе текущей ситуации

        Returns:
            Название выбранной стратегии
        """
        performance = self.evaluate_performance()

        # Если высокая загрузка CPU или памяти, выбираем стратегию эффективности
        if performance['cpu_utilization'] > 80 or performance['memory_utilization'] > 80:
            self.current_strategy = 'efficiency'
        # Если низкая загрузка и нужно повысить производительность
        elif performance['overall_score'] < 30:
            self.current_strategy = 'performance'
        # В остальных случаях используем сбалансированную стратегию
        else:
            self.current_strategy = 'balanced'

        return self.current_strategy


    def adapt_resources(self) -> Dict[str, Any]:
        """
        Адаптирует ресурсы на основе текущей ситуации

        Returns:
            Результат адаптации
        """
        strategy = self.select_optimization_strategy()
        strategy_config = self.strategies[strategy]

        # Применяем конфигурацию стратегии
        cpu_limit = 90 if strategy == 'performance' else (70 if strategy == 'balanced' else 50)
        memory_limit = 2048 if strategy == 'performance' else (1024 if strategy == 'balanced' else 512)

        self.resource_manager.set_resource_limits(
            cpu_percent=cpu_limit,
            memory_mb=memory_limit
        )

        # Выполняем оптимизацию
        optimization_results = self.resource_manager.optimize_all_resources()

        adaptation_record = {
            'timestamp': datetime.now(),
            'strategy': strategy,
            'strategy_config': strategy_config,
            'optimization_results': {
                k: {
                    'original': v.original_value,
                    'optimized': v.optimized_value,
                    'improvement': v.improvement_percent,
                    'status': v.status
                }
                for k, v in optimization_results.items()
            },
            'performance_before': self.evaluate_performance()
        }

        self.adaptation_history.append(adaptation_record)

        return {
            'strategy_applied': strategy,
            'adaptation_successful': True,
            'optimization_results': optimization_results,
            'new_limits': {
                'cpu_percent': cpu_limit,
                'memory_mb': memory_limit
            }
        }


    def start_adaptive_optimization(self, interval: float = 5.0):
        """
        Запускает адаптивную оптимизацию

        Args:
    """TODO: Add description"""

            interval: Интервал между адаптациями (в секундах)
        """
        def adaptive_loop():
            """TODO: Add description"""
            while True:
                try:
                    self.adapt_resources()
                    time.sleep(interval)
                except Exception as e:
                    print(f"Ошибка адаптивной оптимизации: {e}")
                    time.sleep(interval)

        adaptive_thread = threading.Thread(target=adaptive_loop, daemon=True)
        adaptive_thread.start()


    def get_adaptation_report(self) -> Dict[str, Any]:
        """
        Получает отчет об адаптации

        Returns:
            Отчет об адаптации
        """
        return {
            'current_strategy': self.current_strategy,
            'strategies_available': list(self.strategies.keys()),
            'adaptation_history_count': len(self.adaptation_history),
            'recent_adaptations': self.adaptation_history[-5:],  # Последние 5 адаптаций
            'current_limits': self.resource_manager.resource_limits,
            'current_performance': self.evaluate_performance()
        }

def main():
    """Главная функция для демонстрации возможностей менеджера ресурсов"""
    print("=== МЕНЕДЖЕР РЕСУРСОВ И АДАПТИВНЫЙ ОПТИМИЗАТОР ===")

    # Создаем менеджер ресурсов
    resource_manager = ResourceManager()

    print("✓ Менеджер ресурсов инициализирован")

    # Устанавливаем ограничения
    resource_manager.set_resource_limits(
        cpu_percent=85.0,
        memory_mb=1024.0
    )
    print("✓ Ограничения ресурсов установлены")

    # Получаем текущие ресурсы
    current_resources = resource_manager.get_current_resources()
    print(f"✓ Текущее использование ресурсов получено")
    print(f"  CPU: {current_resources.get('cpu_percent', 0):.2f}%")
    print(f"  Память: {current_resources.get('memory_rss_mb', 0):.2f} MB")

    # Запускаем оптимизацию
    print("\nЗапуск оптимизации ресурсов...")
    optimization_results = resource_manager.optimize_all_resources()

    for resource_type, result in optimization_results.items():
        print(f"  {resource_type}: {result.improvement_percent:.2f}% улучшение")

    # Запускаем мониторинг
    print("\nЗапуск мониторинга ресурсов...")
    resource_manager.start_monitoring(interval=0.5)

    # Ждем немного
    time.sleep(2)

    # Останавливаем мониторинг
    resource_manager.stop_monitoring()
    print("✓ Мониторинг остановлен")

    # Проверяем эффективность
    efficiency = resource_manager.get_resource_efficiency_score()
    print(f"✓ Эффективность использования ресурсов: {efficiency:.2f}%")

    # Получаем предложения по оптимизации
    suggestions = resource_manager.suggest_optimizations()
    print(f"\nПредложения по оптимизации ({len(suggestions)}):")
    for suggestion in suggestions:
        print(f"  - {suggestion}")

    # Создаем адаптивный оптимизатор
    print("\nСоздание адаптивного оптимизатора...")
    adaptive_optimizer = AdaptiveResourceOptimizer()

    print("✓ Адаптивный оптимизатор инициализирован")

    # Устанавливаем цели производительности
    adaptive_optimizer.set_performance_goals(
        response_time_ms=150,
        resource_utilization_percent=70.0
    )
    print("✓ Цели производительности установлены")

    # Выполняем адаптацию
    adaptation_result = adaptive_optimizer.adapt_resources()
    print(f"✓ Адаптация выполнена: {adaptation_result['strategy_applied']}")

    # Получаем отчет об адаптации
    adaptation_report = adaptive_optimizer.get_adaptation_report()
    print(f"✓ Отчет об адаптации получен")
    print(f"  Текущая стратегия: {adaptation_report['current_strategy']}")
    print(f"  История адаптаций: {adaptation_report['adaptation_history_count']}")

    # Получаем полный отчет
    report_path = resource_manager.save_report()
    print(f"\n✓ Полный отчет сохранен: {report_path}")

    print("\nМенеджер ресурсов и адаптивный оптимизатор успешно протестированы")
    print("\nДоступные функции:")
    print("- Оптимизация CPU: resource_manager.optimize_cpu_usage()")
    print("- Оптимизация памяти: resource_manager.optimize_memory_usage()")
    print("- Оптимизация всех ресурсов: resource_manager.optimize_all_resources()")
    print("- Мониторинг ресурсов: resource_manager.start_monitoring()")
    print("- Оценка эффективности: resource_manager.get_resource_efficiency_score()")
    print("- Адаптивная оптимизация: adaptive_optimizer.adapt_resources()")
    print("- Отчеты: resource_manager.get_resource_report(), resource_manager.save_report()")

if __name__ == "__main__":
    main()

