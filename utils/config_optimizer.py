# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Модуль оптимизации конфигурации для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для оптимизации
конфигурации и параметров проекта.
"""

import json
import yaml
import toml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import copy
import math
import numpy as np
from dataclasses import dataclass, asdict
import configparser
import threading
import time
import psutil
import gc

@dataclass
class OptimizationParams:
    """Параметры оптимизации"""
    cpu_threshold: float = 80.0
    memory_threshold: float = 80.0
    disk_threshold: float = 80.0
    max_threads: int = 4
    batch_size: int = 100
    cache_size: int = 1000
    timeout: int = 30
    retry_attempts: int = 3

class ConfigOptimizer:
    """
    Класс оптимизатора конфигурации
    Обеспечивает оптимизацию параметров конфигурации
    на основе текущего состояния системы.
    """


    def __init__(self, config_path: str = "config.json"):
        """
        Инициализирует оптимизатор конфигурации

        Args:
            config_path: Путь к файлу конфигурации
        """
        self.config_path = Path(config_path)
        self.original_config = {}
        self.optimized_config = {}
        self.optimization_params = OptimizationParams()
        self.system_metrics = {}
        self.lock = threading.Lock()

        # Загружаем исходную конфигурацию
        self.load_config()


    def load_config(self) -> bool:
        """
        Загружает конфигурацию из файла

        Returns:
            True если загрузка успешна, иначе False
        """
        try:
            if self.config_path.suffix.lower() == '.json':
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.original_config = json.load(f)
            elif self.config_path.suffix.lower() in ['.yml', '.yaml']:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.original_config = yaml.safe_load(f)
            elif self.config_path.suffix.lower() == '.toml':
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.original_config = toml.load(f)
            elif self.config_path.suffix.lower() == '.ini':
                config = configparser.ConfigParser()
                config.read(self.config_path, encoding='utf-8')
                self.original_config = {section: dict(config[section]) for section in config.sections()}
            else:
                # Пытаемся определить формат по содержимому
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip().startswith('{'):
                        self.original_config = json.loads(content)
                    elif content.strip().startswith('['):
                        config = configparser.ConfigParser()
                        config.read_string(content)
                        self.original_config = {section: dict(config[section]) for section in config.sections()}
                    else:
                        # Предполагаем YAML
                        self.original_config = yaml.safe_load(content)

            self.optimized_config = copy.deepcopy(self.original_config)
            return True

        except Exception as e:
            print(f"Ошибка загрузки конфигурации: {e}")
            return False


    def save_config(self, config: Dict[str, Any], output_path: str = None) -> bool:
        """
        Сохраняет конфигурацию в файл

        Args:
            config: Конфигурация для сохранения
            output_path: Путь для сохранения (если None, используется исходный путь)

        Returns:
            True если сохранение успешно, иначе False
        """
        if output_path is None:
            output_path = self.config_path

        try:
            output_path = Path(output_path)

            if output_path.suffix.lower() == '.json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False, default=str)
            elif output_path.suffix.lower() in ['.yml', '.yaml']:
                with open(output_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            elif output_path.suffix.lower() == '.toml':
                with open(output_path, 'w', encoding='utf-8') as f:
                    toml.dump(config, f)
            elif output_path.suffix.lower() == '.ini':
                config_parser = configparser.ConfigParser()
                for section, section_data in config.items():
                    config_parser[section] = section_data
                with open(output_path, 'w', encoding='utf-8') as f:
                    config_parser.write(f)
            else:
                # По умолчанию сохраняем как JSON
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False, default=str)

            return True

        except Exception as e:
            print(f"Ошибка сохранения конфигурации: {e}")
            return False


    def collect_system_metrics(self) -> Dict[str, Any]:
        """
        Собирает метрики системы

        Returns:
            Словарь с метриками системы
        """
        with self.lock:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_available_gb': psutil.virtual_memory().available / (1024**3),
                'disk_usage_percent': psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else 0,
                'disk_free_gb': psutil.disk_usage('/').free / (1024**3) if hasattr(psutil, 'disk_usage') else 0,
                'process_count': len(psutil.pids()),
                'thread_count': threading.active_count(),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'cpu_count_physical': psutil.cpu_count(logical=False),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3)
            }

            self.system_metrics = metrics
            return metrics


    def optimize_for_performance(self) -> Dict[str, Any]:
        """
        Оптимизирует конфигурацию для производительности

        Returns:
            Оптимизированная конфигурация
        """
        optimized = copy.deepcopy(self.original_config)

        # Собираем метрики системы
        metrics = self.collect_system_metrics()

        # Оптимизация на основе доступных ресурсов
        available_memory_gb = metrics['memory_available_gb']
        cpu_count = metrics['cpu_count_logical']

        # Оптимизация количества потоков
        if 'threads' in optimized:
            max_threads = min(cpu_count, self.optimization_params.max_threads)
            if available_memory_gb > 4:  # Если доступно больше 4GB RAM
                optimized['threads'] = max_threads
            else:
                optimized['threads'] = max(1, max_threads // 2)

        # Оптимизация размера батча
        if 'batch_size' in optimized:
            if available_memory_gb > 8:
                optimized['batch_size'] = min(1000, self.optimization_params.batch_size * 2)
            elif available_memory_gb > 4:
                optimized['batch_size'] = self.optimization_params.batch_size
            else:
                optimized['batch_size'] = max(10, self.optimization_params.batch_size // 2)

        # Оптимизация размера кэша
        if 'cache_size' in optimized:
            if available_memory_gb > 8:
                optimized['cache_size'] = min(10000, self.optimization_params.cache_size * 5)
            elif available_memory_gb > 4:
                optimized['cache_size'] = min(5000, self.optimization_params.cache_size * 2)
            else:
                optimized['cache_size'] = self.optimization_params.cache_size

        # Оптимизация таймаутов
        if 'timeout' in optimized:
            if metrics['cpu_percent'] > self.optimization_params.cpu_threshold:
                optimized['timeout'] = max(60, self.optimization_params.timeout * 2)
            else:
                optimized['timeout'] = self.optimization_params.timeout

        # Оптимизация попыток повтора
        if 'retry_attempts' in optimized:
            if metrics['memory_percent'] > self.optimization_params.memory_threshold:
                optimized['retry_attempts'] = max(1, self.optimization_params.retry_attempts - 1)
            else:
                optimized['retry_attempts'] = self.optimization_params.retry_attempts

        return optimized


    def optimize_for_resource_efficiency(self) -> Dict[str, Any]:
        """
        Оптимизирует конфигурацию для эффективного использования ресурсов

        Returns:
            Оптимизированная конфигурация
        """
        optimized = copy.deepcopy(self.original_config)

        # Собираем метрики системы
        metrics = self.collect_system_metrics()

        # Уменьшаем использование ресурсов если нагрузка высока
        if metrics['cpu_percent'] > self.optimization_params.cpu_threshold:
            if 'threads' in optimized:
                optimized['threads'] = max(1, optimized['threads'] // 2)

        if metrics['memory_percent'] > self.optimization_params.memory_threshold:
            if 'cache_size' in optimized:
                optimized['cache_size'] = max(100, optimized['cache_size'] // 2)

            if 'batch_size' in optimized:
                optimized['batch_size'] = max(10, optimized['batch_size'] // 2)

        if metrics['disk_usage_percent'] > self.optimization_params.disk_threshold:
            # Уменьшаем размер временных файлов и кэшей
            if 'temp_file_size_limit' in optimized:
                optimized['temp_file_size_limit'] = min(
                    optimized['temp_file_size_limit'],
                    metrics['disk_free_gb'] * 0.1  # 10% от свободного места
                )

        return optimized


    def optimize_for_stability(self) -> Dict[str, Any]:
        """
        Оптимизирует конфигурацию для стабильности

        Returns:
            Оптимизированная конфигурация
        """
        optimized = copy.deepcopy(self.original_config)

        # Собираем метрики системы
        metrics = self.collect_system_metrics()

        # Увеличиваем таймауты и попытки для стабильности
        if 'timeout' in optimized:
            optimized['timeout'] = max(optimized['timeout'], 60)

        if 'retry_attempts' in optimized:
            optimized['retry_attempts'] = max(optimized['retry_attempts'], 3)

        # Ограничиваем использование ресурсов для стабильности
        if 'threads' in optimized:
            optimized['threads'] = min(optimized['threads'], metrics['cpu_count_logical'])

        if 'cache_size' in optimized:
            max_cache_size = int(metrics['memory_available_gb'] * 0.1 * 1000)  # 10% от доступной памяти
            optimized['cache_size'] = min(optimized['cache_size'], max_cache_size)

        # Добавляем параметры стабильности
        if 'gc_threshold' not in optimized:
            optimized['gc_threshold'] = [700, 10, 10]  # Уровни сборки мусора

        if 'connection_pool_size' not in optimized:
            optimized['connection_pool_size'] = 5  # Ограничение пула соединений

        return optimized


    def optimize_config(self, strategy: str = "balanced") -> Dict[str, Any]:
        """
        Оптимизирует конфигурацию по заданной стратегии

        Args:
            strategy: Стратегия оптимизации ("performance", "efficiency", "stability", "balanced")

        Returns:
            Оптимизированная конфигурация
        """
        if strategy == "performance":
            optimized = self.optimize_for_performance()
        elif strategy == "efficiency":
            optimized = self.optimize_for_resource_efficiency()
        elif strategy == "stability":
            optimized = self.optimize_for_stability()
        elif strategy == "balanced":
            # Комбинированная стратегия
            perf_config = self.optimize_for_performance()
            eff_config = self.optimize_for_resource_efficiency()
            stab_config = self.optimize_for_stability()

            optimized = copy.deepcopy(self.original_config)

            # Среднее арифметическое для числовых значений
            for key in set(perf_config.keys()) | set(eff_config.keys()) | set(stab_config.keys()):
                if key in perf_config and key in eff_config and key in stab_config:
                    perf_val = perf_config[key]
                    eff_val = eff_config[key]
                    stab_val = stab_config[key]

                    if isinstance(perf_val, (int, float)) and isinstance(eff_val, (int, float)) and isinstance(stab_val, (int, float)):
                        # Среднее из трех значений
                        avg_val = (perf_val + eff_val + stab_val) / 3
                        optimized[key] = int(avg_val) if isinstance(perf_val, int) else avg_val
                    else:
                        # Берем значение из стабильной конфигурации как наиболее консервативное
                        optimized[key] = stab_val
                elif key in perf_config:
                    optimized[key] = perf_config[key]
                elif key in eff_config:
                    optimized[key] = eff_config[key]
                elif key in stab_config:
                    optimized[key] = stab_config[key]
        else:
            raise ValueError(f"Неизвестная стратегия оптимизации: {strategy}")

        self.optimized_config = optimized
        return optimized


    def apply_optimization(self, strategy: str = "balanced", save_to_file: bool = True) -> bool:
        """
        Применяет оптимизацию к конфигурации

        Args:
            strategy: Стратегия оптимизации
            save_to_file: Сохранять ли результат в файл

        Returns:
            True если применение успешно, иначе False
        """
        try:
            optimized_config = self.optimize_config(strategy)

            if save_to_file:
                return self.save_config(optimized_config)
            else:
                self.optimized_config = optimized_config
                return True

        except Exception as e:
            print(f"Ошибка применения оптимизации: {e}")
            return False


    def get_optimization_report(self) -> Dict[str, Any]:
        """
        Генерирует отчет об оптимизации

        Returns:
            Словарь с отчетом об оптимизации
        """
        changes = {}
        for key in set(self.original_config.keys()) | set(self.optimized_config.keys()):
            orig_val = self.original_config.get(key, 'NOT_FOUND')
            opt_val = self.optimized_config.get(key, 'NOT_FOUND')

            if orig_val != opt_val:
                changes[key] = {
                    'original': orig_val,
                    'optimized': opt_val,
                    'changed': True
                }
            else:
                changes[key] = {
                    'original': orig_val,
                    'optimized': opt_val,
                    'changed': False
                }

        report = {
            'timestamp': datetime.now().isoformat(),
            'original_config_path': str(self.config_path),
            'changes_made': len([c for c in changes.values() if c['changed']]),
            'total_parameters': len(changes),
            'system_metrics': self.system_metrics,
            'optimization_params': asdict(self.optimization_params),
            'parameter_changes': changes
        }

        return report


    def auto_optimize(self, strategy: str = "balanced") -> Dict[str, Any]:
        """
        Автоматически оптимизирует конфигурацию

        Args:
            strategy: Стратегия оптимизации

        Returns:
            Отчет об оптимизации
        """
        # Собираем метрики до оптимизации
        self.collect_system_metrics()

        # Применяем оптимизацию
        success = self.apply_optimization(strategy)

        if success:
            return self.get_optimization_report()
        else:
            return {
                'success': False,
                'error': 'Failed to apply optimization',
                'timestamp': datetime.now().isoformat()
            }


    def optimize_multiple_configs(self, config_paths: List[str], strategy: str = "balanced") -> Dict[str, Any]:
        """
        Оптимизирует несколько конфигурационных файлов

        Args:
            config_paths: Список путей к конфигурационным файлам
            strategy: Стратегия оптимизации

        Returns:
            Словарь с результатами оптимизации для каждого файла
        """
        results = {}

        for config_path in config_paths:
            try:
                # Создаем временный оптимизатор для каждого файла
                temp_optimizer = ConfigOptimizer(config_path)

                # Применяем оптимизацию
                report = temp_optimizer.auto_optimize(strategy)

                results[config_path] = report

            except Exception as e:
                results[config_path] = {
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }

        return results

class AdaptiveConfigManager:
    """
    Класс адаптивного управления конфигурацией
    Обеспечивает динамическую адаптацию параметров
    конфигурации в зависимости от условий выполнения.
    """


    def __init__(self, base_config_path: str = "config.json"):
        """
        Инициализирует адаптивный менеджер конфигурации

        Args:
            base_config_path: Базовый путь к конфигурации
        """
        self.base_config_path = Path(base_config_path)
        self.optimizer = ConfigOptimizer(base_config_path)
        self.adaptation_history = []
        self.is_monitoring = False
        self.monitoring_thread = None
        self.adaptation_callback = None


    def start_adaptive_monitoring(self, interval: float = 60.0, strategy: str = "balanced"):
        """
        Запускает адаптивный мониторинг и оптимизацию

        Args:
            interval: Интервал между проверками (в секундах)
            strategy: Стратегия оптимизации
        """
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval, strategy),
            daemon=True
        )
        self.monitoring_thread.start()


    def stop_adaptive_monitoring(self):
        """Останавливает адаптивный мониторинг"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)


    def _monitoring_loop(self, interval: float, strategy: str):
        """
        Цикл адаптивного мониторинга

        Args:
            interval: Интервал между проверками
            strategy: Стратегия оптимизации
        """
        while self.is_monitoring:
            try:
                # Проверяем необходимость адаптации
                if self.should_adapt():
                    # Применяем адаптацию
                    report = self.optimizer.auto_optimize(strategy)

                    # Сохраняем в историю
                    self.adaptation_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'report': report,
                        'trigger': 'adaptive_monitoring'
                    })

                    # Вызываем callback если установлен
                    if self.adaptation_callback:
                        self.adaptation_callback(report)

                time.sleep(interval)

            except Exception as e:
                print(f"Ошибка в цикле адаптивного мониторинга: {e}")
                time.sleep(interval)


    def should_adapt(self) -> bool:
        """
        Проверяет, нужно ли применять адаптацию

        Returns:
            True если адаптация необходима, иначе False
        """
        metrics = self.optimizer.collect_system_metrics()

        # Адаптируем если хотя бы один показатель превышает порог
        return (
            metrics['cpu_percent'] > self.optimizer.optimization_params.cpu_threshold or
            metrics['memory_percent'] > self.optimizer.optimization_params.memory_threshold or
            metrics['disk_usage_percent'] > self.optimizer.optimization_params.disk_threshold
        )


    def set_adaptation_callback(self, callback: callable):
        """
        Устанавливает callback для уведомлений об адаптации

        Args:
            callback: Функция обратного вызова
        """
        self.adaptation_callback = callback


    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """
        Возвращает историю адаптаций

        Returns:
            Список записей истории адаптаций
        """
        return self.adaptation_history.copy()


    def clear_adaptation_history(self):
        """Очищает историю адаптаций"""
        self.adaptation_history.clear()

def main():
    """Главная функция для демонстрации возможностей оптимизатора конфигурации"""
    print("=== ОПТИМИЗАТОР КОНФИГУРАЦИИ ПРОЕКТА ===")

    # Создаем оптимизатор конфигурации
    optimizer = ConfigOptimizer()

    print("✓ Оптимизатор конфигурации инициализирован")
    print(f"✓ Путь к конфигурации: {optimizer.config_path}")

    # Собираем метрики системы
    print("\nСбор метрик системы...")
    metrics = optimizer.collect_system_metrics()
    print(f"  - Загрузка CPU: {metrics['cpu_percent']}%")
    print(f"  - Использование памяти: {metrics['memory_percent']}%")
    print(f"  - Доступно памяти: {metrics['memory_available_gb']:.2f} GB")
    print(f"  - Использование диска: {metrics['disk_usage_percent']}%")
    print(f"  - Количество процессоров: {metrics['cpu_count_logical']}")

    # Применяем различные стратегии оптимизации
    strategies = ["performance", "efficiency", "stability", "balanced"]

    for strategy in strategies:
        print(f"\nПрименение стратегии оптимизации: {strategy}")
        report = optimizer.auto_optimize(strategy)

        print(f"  - Параметров изменено: {report['changes_made']}")
        print(f"  - Всего параметров: {report['total_parameters']}")

        if report['changes_made'] > 0:
            print("  - Измененные параметры:")
            for param, change in report['parameter_changes'].items():
                if change['changed']:
                    print(f"    * {param}: {change['original']} → {change['optimized']}")

    # Создаем адаптивный менеджер
    print("\nСоздание адаптивного менеджера конфигурации...")
    adaptive_manager = AdaptiveConfigManager()

    # Показываем пример ручной адаптации
    print("\nРучная проверка необходимости адаптации...")
    should_adapt = adaptive_manager.should_adapt()
    print(f"  - Требуется адаптация: {should_adapt}")

    # Показываем историю адаптаций
    history = adaptive_manager.get_adaptation_history()
    print(f"  - Записей в истории адаптаций: {len(history)}")

    print("\nОптимизатор конфигурации успешно протестирован")
    print("\nДоступные функции:")
    print("- Оптимизация по стратегии: optimize_config()")
    print("- Автоматическая оптимизация: auto_optimize()")
    print("- Адаптивный мониторинг: start_adaptive_monitoring()")
    print("- Отчет об оптимизации: get_optimization_report()")
    print("- Оптимизация нескольких файлов: optimize_multiple_configs()")

if __name__ == "__main__":
    main()

