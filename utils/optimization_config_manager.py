# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Модуль централизованного управления конфигурацией для системы оптимизации
Проекта Лаборатория моделирования нанозонда
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
import configparser

@dataclass
class OptimizationConfig:
    """Конфигурация системы оптимизации"""
    # Общие настройки
    enabled: bool = True
    log_level: str = "INFO"
    output_directory: str = "optimization_logs"

    # Настройки профилирования
    profiling_enabled: bool = True
    profiling_interval: float = 1.0
    profiling_output_dir: str = "profiles"

    # Настройки мониторинга ресурсов
    resource_monitoring_enabled: bool = True
    resource_monitoring_interval: float = 5.0
    cpu_threshold_warning: float = 70.0
    cpu_threshold_error: float = 85.0
    memory_threshold_warning: float = 75.0
    memory_threshold_error: float = 85.0

    # Настройки отслеживания памяти
    memory_tracking_enabled: bool = True
    memory_tracking_interval: float = 3.0
    leak_detection_threshold: float = 1.0  # MB/мин

    # Настройки бенчмаркинга
    benchmarking_enabled: bool = True
    benchmark_iterations: int = 100
    benchmark_warmup: int = 10

    # Настройки аналитики
    analytics_enabled: bool = True
    analytics_refresh_interval: float = 60.0
    trend_analysis_window_hours: int = 24

    # Настройки уведомлений
    notifications_enabled: bool = True
    alert_email_recipients: list = None
    console_notifications: bool = True

class OptimizationConfigManager:
    """
    Класс управления конфигурацией системы оптимизации
    Обеспечивает загрузку, сохранение и валидацию конфигурации.
    """


    def __init__(self, config_path: str = "config/optimization_config.json"):
        """
        Инициализирует менеджер конфигурации

        Args:
            config_path: Путь к файлу конфигурации
        """
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(exist_ok=True)

        # Загружаем или создаем конфигурацию по умолчанию
        self.config = self.load_config()

        # Настраиваем логирование
        self.setup_logging()


    def load_config(self) -> OptimizationConfig:
        """
        Загружает конфигурацию из файла

        Returns:
            Объект конфигурации
        """
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)

                # Создаем конфигурацию с учетом возможных новых полей
                config = OptimizationConfig()

                # Обновляем значения из файла
                for key, value in config_dict.items():
                    if hasattr(config, key):
                        setattr(config, key, value)

                return config
            except Exception as e:
                print(f"Ошибка загрузки конфигурации: {e}")
                # Возвращаем конфигурацию по умолчанию
                return OptimizationConfig()
        else:
            # Создаем конфигурацию по умолчанию и сохраняем
            config = OptimizationConfig()
            self.save_config(config)
            return config


    def save_config(self, config: OptimizationConfig = None):
        """
        Сохраняет конфигурацию в файл

        Args:
            config: Объект конфигурации для сохранения
        """
        if config is None:
            config = self.config

        config_dict = asdict(config)

        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)


    def update_config(self, **kwargs):
        """
        Обновляет параметры конфигурации

        Args:
            **kwargs: Параметры для обновления
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self.save_config()


    def get_config(self) -> OptimizationConfig:
        """
        Возвращает текущую конфигурацию

        Returns:
            Объект конфигурации
        """
        return self.config


    def setup_logging(self):
        """Настраивает централизованное логирование для системы оптимизации"""
        log_dir = Path(self.config.output_directory)
        log_dir.mkdir(exist_ok=True)

        # Определяем уровень логирования
        level = getattr(logging, self.config.log_level.upper(), logging.INFO)

        # Создаем форматтер
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Настраиваем корневой логгер для системы оптимизации
        logger = logging.getLogger('optimization_system')
        logger.setLevel(level)

        # Удаляем существующие обработчики
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Файловый обработчик
        file_handler = logging.FileHandler(
            log_dir / f"optimization_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Консольный обработчик
        if self.config.console_notifications:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)


    def get_logger(self, name: str) -> logging.Logger:
        """
        Возвращает логгер для указанного компонента

        Args:
            name: Название компонента

        Returns:
            Объект логгера
        """
        return logging.getLogger(f'optimization_system.{name}')


    def validate_config(self) -> Dict[str, Any]:
        """
        Валидирует конфигурацию и возвращает результаты проверки

        Returns:
            Словарь с результатами валидации
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'timestamp': datetime.now().isoformat()
        }

        config = self.config

        # Проверяем пороговые значения
        if config.cpu_threshold_warning >= config.cpu_threshold_error:
            validation_results['errors'].append(
                "Порог предупреждения CPU должен быть меньше порога ошибки"
            )
            validation_results['valid'] = False

        if config.memory_threshold_warning >= config.memory_threshold_error:
            validation_results['errors'].append(
                "Порог предупреждения памяти должен быть меньше порога ошибки"
            )
            validation_results['valid'] = False

        # Проверяем интервалы
        if config.profiling_interval <= 0:
            validation_results['errors'].append("Интервал профилирования должен быть положительным")
            validation_results['valid'] = False

        if config.resource_monitoring_interval <= 0:
            validation_results['errors'].append("Интервал мониторинга ресурсов должен быть положительным")
            validation_results['valid'] = False

        if config.analytics_refresh_interval <= 0:
            validation_results['errors'].append("Интервал обновления аналитики должен быть положительным")
            validation_results['valid'] = False

        # Проверяем критические значения
        if config.cpu_threshold_error > 100:
            validation_results['errors'].append("Порог ошибки CPU не должен превышать 100%")
            validation_results['valid'] = False

        if config.memory_threshold_error > 100:
            validation_results['errors'].append("Порог ошибки памяти не должен превышать 100%")
            validation_results['valid'] = False

        # Предупреждения
        if config.profiling_interval > 5:
            validation_results['warnings'].append(
                "Большой интервал профилирования может пропустить кратковременные пики"
            )

        if config.leak_detection_threshold < 0.1:
            validation_results['warnings'].append(
                "Слишком низкий порог обнаружения утечек может давать ложные срабатывания"
            )

        return validation_results

def create_default_config() -> OptimizationConfig:
    """Создает конфигурацию по умолчанию для системы оптимизации"""
    return OptimizationConfig(
        enabled=True,
        log_level="INFO",
        output_directory="optimization_logs",
        profiling_enabled=True,
        profiling_interval=1.0,
        profiling_output_dir="profiles",
        resource_monitoring_enabled=True,
        resource_monitoring_interval=5.0,
        cpu_threshold_warning=70.0,
        cpu_threshold_error=85.0,
        memory_threshold_warning=75.0,
        memory_threshold_error=85.0,
        memory_tracking_enabled=True,
        memory_tracking_interval=3.0,
        leak_detection_threshold=1.0,
        benchmarking_enabled=True,
        benchmark_iterations=100,
        benchmark_warmup=10,
        analytics_enabled=True,
        analytics_refresh_interval=60.0,
        trend_analysis_window_hours=24,
        notifications_enabled=True,
        console_notifications=True
    )

def main():
    """Главная функция для демонстрации возможностей менеджера конфигурации"""
    print("=== МЕНЕДЖЕР КОНФИГУРАЦИИ СИСТЕМЫ ОПТИМИЗАЦИИ ===")

    # Создаем менеджер конфигурации
    config_manager = OptimizationConfigManager()

    print("✓ Менеджер конфигурации инициализирован")
    print(f"✓ Путь к конфигурации: {config_manager.config_path}")

    # Получаем текущую конфигурацию
    config = config_manager.get_config()
    print(f"✓ Система оптимизации {'включена' if config.enabled else 'отключена'}")
    print(f"✓ Уровень логирования: {config.log_level}")

    # Валидируем конфигурацию
    print("\nПроверка конфигурации...")
    validation_results = config_manager.validate_config()
    print(f"✓ Валидация: {'успешна' if validation_results['valid'] else 'с ошибками'}")
    print(f"✓ Ошибок: {len(validation_results['errors'])}")
    print(f"✓ Предупреждений: {len(validation_results['warnings'])}")

    if validation_results['errors']:
        print("Ошибки:")
        for error in validation_results['errors']:
            print(f"  - {error}")

    if validation_results['warnings']:
        print("Предупреждения:")
        for warning in validation_results['warnings']:
            print(f"  - {warning}")

    # Создаем логгер и тестируем логирование
    logger = config_manager.get_logger("test_component")
    logger.info("Тестовое сообщение от компонента тестирования")
    logger.warning("Тестовое предупреждение")

    print("\n✓ Тестовое логирование выполнено")

    # Пример обновления конфигурации
    print("\nОбновление конфигурации...")
    config_manager.update_config(
        cpu_threshold_warning=65.0,
        memory_threshold_warning=70.0
    )

    updated_config = config_manager.get_config()
    print(f"✓ Новый порог CPU предупреждения: {updated_config.cpu_threshold_warning}%")
    print(f"✓ Новый порог памяти предупреждения: {updated_config.memory_threshold_warning}%")

    print("\nМенеджер конфигурации успешно протестирован")
    print("\nДоступные функции:")
    print("- Загрузка конфигурации: config_manager.load_config()")
    print("- Сохранение конфигурации: config_manager.save_config()")
    print("- Обновление конфигурации: config_manager.update_config()")
    print("- Валидация конфигурации: config_manager.validate_config()")
    print("- Получение логгера: config_manager.get_logger()")

if __name__ == "__main__":
    main()

