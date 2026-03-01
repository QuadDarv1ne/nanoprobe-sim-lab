# -*- coding: utf-8 -*-
#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3

"""
Модуль централизованного логирования для системы оптимизации
Проекта Лаборатория моделирования нанозонда
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json
import threading
from enum import Enum
import traceback
from dataclasses import dataclass

class LogLevel(Enum):
    """Уровни логирования для системы оптимизации"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class OptimizationComponent(Enum):
    """Компоненты системы оптимизации"""
    PROFILER = "profiler"
    RESOURCE_MANAGER = "resource_manager"
    LOGGER_ANALYZER = "logger_analyzer"
    MEMORY_TRACKER = "memory_tracker"
    BENCHMARK_SUITE = "benchmark_suite"
    ORCHESTRATOR = "orchestrator"
    HEALTH_MONITOR = "health_monitor"
    ANALYTICS_DASHBOARD = "analytics_dashboard"
    VERIFICATION_FRAMEWORK = "verification_framework"
    CONFIG_MANAGER = "config_manager"

@dataclass
class OptimizationLogRecord:
    """Запись лога оптимизации"""
    timestamp: datetime
    component: OptimizationComponent
    level: LogLevel
    message: str
    details: Dict[str, Any] = None
    exception_info: str = None
    thread_id: str = None
    session_id: str = None

class OptimizationLogger:
    """
    Класс централизованного логирования системы оптимизации
    Обеспечивает унифицированное логирование для всех компонентов оптимизации.
    """


    def __init__(self, log_directory: str = "optimization_logs", session_id: str = None):
        """
        Инициализирует централизованный логгер

        Args:
            log_directory: Директория для логов
            session_id: Идентификатор сессии
        """
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(exist_ok=True)

        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Настройка центрального логгера
        self.logger = logging.getLogger('optimization_system')
        self.logger.setLevel(logging.DEBUG)

        # Очищаем существующие обработчики
        self.logger.handlers.clear()

        # Форматтер для логов
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(threadName)-15s | %(message)s'
        )

        # Файловый обработчик
        log_file = self.log_directory / f"optimization_{self.session_id}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Обработчик для ошибок
        error_file = self.log_directory / f"optimization_errors_{self.session_id}.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        self.logger.addHandler(error_handler)

        # Консольный обработчик
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Буфер для асинхронного логирования
        self.log_buffer = []
        self.buffer_lock = threading.Lock()
        self.max_buffer_size = 100


    def log(self, component: OptimizationComponent, level: LogLevel, message: str,

            details: Dict[str, Any] = None, exception: Exception = None):
        """
        Логирует сообщение от компонента системы оптимизации

        Args:
            component: Компонент системы
            level: Уровень логирования
            message: Сообщение
            details: Дополнительные данные
            exception: Исключение для логирования
        """
        # Создаем запись лога
        record = OptimizationLogRecord(
            timestamp=datetime.now(),
            component=component,
            level=level,
            message=message,
            details=details or {},
            exception_info=traceback.format_exc() if exception else None,
            thread_id=threading.current_thread().name,
            session_id=self.session_id
        )

        # Формируем сообщение для стандартного логгера
        log_msg = f"[{component.value}] {message}"
        if details:
            log_msg += f" | Details: {json.dumps(details, ensure_ascii=False, default=str)}"

        # Логируем через стандартный логгер
        log_level = getattr(logging, level.value)
        if exception:
            self.logger.log(log_level, log_msg, exc_info=exception)
        else:
            self.logger.log(log_level, log_msg)

        # Добавляем в буфер (для возможного дальнейшего использования)
        with self.buffer_lock:
            self.log_buffer.append(record)
            if len(self.log_buffer) > self.max_buffer_size:
                self.log_buffer.pop(0)



    def debug(self, component: OptimizationComponent, message: str,
              details: Dict[str, Any] = None):
        """Логирует отладочное сообщение"""
        self.log(component, LogLevel.DEBUG, message, details)


    def info(self, component: OptimizationComponent, message: str,
             details: Dict[str, Any] = None):
        """Логирует информационное сообщение"""

        self.log(component, LogLevel.INFO, message, details)


    def warning(self, component: OptimizationComponent, message: str,
                details: Dict[str, Any] = None):

        """Логирует предупреждение"""
        self.log(component, LogLevel.WARNING, message, details)


    def error(self, component: OptimizationComponent, message: str,

              details: Dict[str, Any] = None, exception: Exception = None):
        """Логирует ошибку"""
        self.log(component, LogLevel.ERROR, message, details, exception)


    def critical(self, component: OptimizationComponent, message: str,
                 details: Dict[str, Any] = None, exception: Exception = None):
        """Логирует критическую ошибку"""
        self.log(component, LogLevel.CRITICAL, message, details, exception)


    def get_component_logger(self, component: OptimizationComponent) -> 'ComponentLogger':
        """
        Возвращает логгер для конкретного компонента

        Args:
            component: Компонент системы

        Returns:
            Логгер компонента
        """
        return ComponentLogger(self, component)


    def get_log_statistics(self) -> Dict[str, int]:
        """
        Получает статистику по логам

        Returns:
            Словарь со статистикой
        """
        stats = {
            'total_records': len(self.log_buffer),
            'debug_count': 0,
            'info_count': 0,
            'warning_count': 0,
            'error_count': 0,
            'critical_count': 0,
            'components': {}
        }

        for record in self.log_buffer:
            stats[f'{record.level.value.lower()}_count'] += 1

            comp_name = record.component.value
            if comp_name not in stats['components']:
                stats['components'][comp_name] = {
                    'debug': 0, 'info': 0, 'warning': 0, 'error': 0, 'critical': 0
                }

            stats['components'][comp_name][record.level.value.lower()] += 1

        return stats


    def export_logs(self, output_path: str = None) -> str:
        """
        Экспортирует логи в файл

        Args:
            output_path: Путь для экспорта

        Returns:
            Путь к экспортированному файлу
        """
        if output_path is None:
            output_path = str(self.log_directory / f"exported_logs_{self.session_id}.json")

        logs_data = []
        for record in self.log_buffer:
            logs_data.append({
                'timestamp': record.timestamp.isoformat(),
                'component': record.component.value,
                'level': record.level.value,
                'message': record.message,
                'details': record.details,
                'exception_info': record.exception_info,
                'thread_id': record.thread_id,
                'session_id': record.session_id
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(logs_data, f, indent=2, ensure_ascii=False, default=str)

        return output_path

class ComponentLogger:
    """
    Логгер для конкретного компонента системы оптимизации
    """


    def __init__(self, central_logger: OptimizationLogger, component: OptimizationComponent):
        """
        Инициализирует логгер компонента

        Args:
            central_logger: Центральный логгер
            component: Компонент системы
        """
        self.central_logger = central_logger
        self.component = component


    def debug(self, message: str, details: Dict[str, Any] = None):
        """Логирует отладочное сообщение"""
        self.central_logger.debug(self.component, message, details)


    def info(self, message: str, details: Dict[str, Any] = None):
        """Логирует информационное сообщение"""
        self.central_logger.info(self.component, message, details)


    def warning(self, message: str, details: Dict[str, Any] = None):
        """Логирует предупреждение"""
        self.central_logger.warning(self.component, message, details)


    def error(self, message: str, details: Dict[str, Any] = None, exception: Exception = None):
        """Логирует ошибку"""
        self.central_logger.error(self.component, message, details, exception)


    def critical(self, message: str, details: Dict[str, Any] = None, exception: Exception = None):
        """Логирует критическую ошибку"""
        self.central_logger.critical(self.component, message, details, exception)

class OptimizationLoggingManager:
    """
    Менеджер централизованного логирования
    Обеспечивает управление логированием для всей системы оптимизации.
    """


    def __init__(self, log_directory: str = "optimization_logs"):
        """
        Инициализирует менеджер логирования

        Args:
            log_directory: Директория для логов
        """
        self.log_directory = Path(log_directory)
        self.loggers: Dict[OptimizationComponent, ComponentLogger] = {}
        self.central_logger = OptimizationLogger(log_directory)

        # Создаем логгеры для всех компонентов
        for component in OptimizationComponent:
            self.loggers[component] = self.central_logger.get_component_logger(component)


    def get_logger(self, component: OptimizationComponent) -> ComponentLogger:
        """
        Возвращает логгер для указанного компонента

        Args:
            component: Компонент системы

        Returns:
            Логгер компонента
        """
        return self.loggers.get(component)


    def get_central_logger(self) -> OptimizationLogger:
        """
        Возвращает центральный логгер

        Returns:
            Центральный логгер
        """
        return self.central_logger


    def get_component_logger(self, component: OptimizationComponent) -> ComponentLogger:
        """
        Возвращает логгер для компонента (альтернативный метод)

        Args:
            component: Компонент системы

        Returns:
            Логгер компонента
        """
        return self.get_logger(component)


    def get_statistics(self) -> Dict[str, int]:
        """
        Получает статистику по всем логам

        Returns:
            Словарь со статистикой
        """
        return self.central_logger.get_log_statistics()


    def export_all_logs(self, output_path: str = None) -> str:
        """
        Экспортирует все логи

        Args:
            output_path: Путь для экспорта

        Returns:
            Путь к экспортированному файлу
        """
        return self.central_logger.export_logs(output_path)

def main():
    """Главная функция для демонстрации возможностей централизованного логирования"""
    print("=== ЦЕНТРАЛИЗОВАННОЕ ЛОГИРОВАНИЕ СИСТЕМЫ ОПТИМИЗАЦИИ ===")

    # Создаем менеджер логирования
    log_manager = OptimizationLoggingManager()

    print("✓ Менеджер логирования инициализирован")
    print(f"✓ Директория логов: {log_manager.central_logger.log_directory}")
    print(f"✓ ID сессии: {log_manager.central_logger.session_id}")

    # Получаем логгеры для разных компонентов
    profiler_logger = log_manager.get_logger(OptimizationComponent.PROFILER)
    resource_logger = log_manager.get_logger(OptimizationComponent.RESOURCE_MANAGER)
    memory_logger = log_manager.get_logger(OptimizationComponent.MEMORY_TRACKER)
    health_logger = log_manager.get_logger(OptimizationComponent.HEALTH_MONITOR)

    print("\nТестирование логирования для разных компонентов...")

    # Тестируем логирование
    profiler_logger.info("Начало профилирования функции calculate_matrix")
    profiler_logger.debug("Параметры профилирования", {
        'iterations': 1000,
        'warmup': 10,
        'function_name': 'calculate_matrix'
    })

    resource_logger.warning("Высокое использование CPU", {
        'current_usage': 85.5,
        'threshold': 80.0,
        'process_id': 12345
    })

    memory_logger.error("Обнаружена утечка памяти", {
        'growth_rate': 2.5,
        'object_type': 'numpy.ndarray',
        'duration_minutes': 10
    })

    health_logger.critical("Критическое состояние системы", {
        'cpu_usage': 95.0,
        'memory_usage': 92.0,
        'disk_usage': 98.0
    })

    print("✓ Тестовое логирование выполнено")

    # Получаем статистику
    print("\nПолучение статистики логирования...")
    stats = log_manager.get_statistics()
    print(f"✓ Всего записей: {stats['total_records']}")
    print(f"✓ Информационных: {stats['info_count']}")
    print(f"✓ Предупреждений: {stats['warning_count']}")
    print(f"✓ Ошибок: {stats['error_count']}")
    print(f"✓ Критических: {stats['critical_count']}")

    # Экспортируем логи
    print("\nЭкспорт логов...")
    export_path = log_manager.export_all_logs()
    print(f"✓ Логи экспортированы в: {export_path}")

    print("\nЦентральное логирование успешно протестировано")
    print("\nДоступные функции:")
    print("- Получение логгера: log_manager.get_logger(component)")
    print("- Логирование информации: logger.info(message, details)")
    print("- Логирование ошибок: logger.error(message, details, exception)")
    print("- Получение статистики: log_manager.get_statistics()")
    print("- Экспорт логов: log_manager.export_all_logs()")

if __name__ == "__main__":
    main()

