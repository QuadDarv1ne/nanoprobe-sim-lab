# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Модуль обработки ошибок для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет систему обработки ошибок,
логирования и восстановления для всего проекта.
"""

import logging
import traceback
import sys
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Callable, TypeVar, Generic
from functools import wraps
import threading
import queue
import time
from enum import Enum
from dataclasses import dataclass

class ErrorSeverity(Enum):
    """Перечисление уровней важности ошибок"""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

@dataclass
class ErrorInfo:
    """Информация об ошибке"""
    timestamp: datetime
    severity: ErrorSeverity
    message: str
    exception_type: str
    exception_message: str
    traceback_info: str
    component: str
    user_context: Optional[Dict[str, Any]] = None

class ErrorHandler:
    """
    Класс обработки ошибок
    Обеспечивает централизованную обработку,
    логирование и восстановление после ошибок.
    """


    def __init__(self, log_file: str = "error_log.json", max_log_size: int = 1000):
        """
        Инициализирует обработчик ошибок

        Args:
            log_file: Файл для логирования ошибок
            max_log_size: Максимальный размер лога (количество записей)
        """
        self.log_file = Path(log_file)
        self.max_log_size = max_log_size
        self.error_queue = queue.Queue()
        self.error_history = []
        self.lock = threading.Lock()

        # Создаем файл лога если не существует
        if not self.log_file.exists():
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self.log_file.touch()
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, default=str)

        # Загружаем историю ошибок
        self.load_error_history()


    def load_error_history(self):
        """Загружает историю ошибок из файла"""
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.error_history = [ErrorInfo(**item) if isinstance(item, dict) else item for item in data]
        except Exception:
            self.error_history = []


    def save_error_history(self):
        """Сохраняет историю ошибок в файл"""
        try:
            # Сохраняем только последние max_log_size ошибок
            recent_errors = self.error_history[-self.max_log_size:]

            # Преобразуем объекты ErrorInfo в словари
            serializable_errors = []
            for error in recent_errors:
                if isinstance(error, ErrorInfo):
                    serializable_error = {
                        'timestamp': error.timestamp.isoformat(),
                        'severity': error.severity.name,
                        'message': error.message,
                        'exception_type': error.exception_type,
                        'exception_message': error.exception_message,
                        'traceback_info': error.traceback_info,
                        'component': error.component,
                        'user_context': error.user_context
                    }
                    serializable_errors.append(serializable_error)
                else:
                    serializable_errors.append(error)

            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_errors, f, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"Ошибка сохранения истории ошибок: {e}")

    def log_error(self,
                  message: str,
                  exception: Exception = None,
                  component: str = "Unknown",
                  severity: ErrorSeverity = ErrorSeverity.ERROR,
                  user_context: Optional[Dict[str, Any]] = None) -> ErrorInfo:
        """
        Логирует ошибку

        Args:
            message: Сообщение об ошибке
            exception: Объект исключения (если есть)
            component: Компонент, в котором произошла ошибка
            severity: Уровень важности ошибки
            user_context: Контекст пользователя (опционально)

        Returns:
            Объект информации об ошибке
        """
        timestamp = datetime.now()

        if exception:
            exception_type = type(exception).__name__
            exception_message = str(exception)
            traceback_info = traceback.format_exc()
        else:
            exception_type = "Manual"
            exception_message = message
            traceback_info = "".join(traceback.format_stack())

        error_info = ErrorInfo(
            timestamp=timestamp,
            severity=severity,
            message=message,
            exception_type=exception_type,
            exception_message=exception_message,
            traceback_info=traceback_info,
            component=component,
            user_context=user_context
        )

        with self.lock:
            self.error_history.append(error_info)
            self.save_error_history()

        # Логируем в стандартный логгер
        logging.log(severity.value, f"[{component}] {message}")

        return error_info

    def handle_exception(self,
                        func: Callable,
                        component: str = "Unknown",
                        fallback_return: Any = None,
                        suppress_exception: bool = False) -> Callable:
        """
        Декоратор для обработки исключений в функциях

        Args:
            func: Функция для оборачивания
            component: Компонент, в котором происходит вызов
            fallback_return: Значение по умолчанию при ошибке
            suppress_exception: Подавлять ли исключение (возвращать fallback_return)

        Returns:
            Обернутая функция с обработкой исключений
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_info = self.log_error(
                    f"Ошибка в функции {func.__name__}",
                    e,
                    component,
                    ErrorSeverity.ERROR,
                    user_context={'function_args': str(args), 'function_kwargs': str(kwargs)}
                )

                if suppress_exception:
                    return fallback_return
                else:
                    raise e
        return wrapper


    def get_recent_errors(self, count: int = 10) -> list:
        """
        Возвращает последние ошибки

        Args:
            count: Количество ошибок для возврата

        Returns:
            Список последних ошибок
        """
        with self.lock:
            return self.error_history[-count:]


    def get_errors_by_severity(self, severity: ErrorSeverity) -> list:
        """
        Возвращает ошибки по уровню важности

        Args:
            severity: Уровень важности

        Returns:
            Список ошибок с указанным уровнем важности
        """
        with self.lock:
            return [error for error in self.error_history if error.severity == severity]


    def get_errors_by_component(self, component: str) -> list:
        """
        Возвращает ошибки по компоненту

        Args:
            component: Название компонента

        Returns:
            Список ошибок для указанного компонента
        """
        with self.lock:
            return [error for error in self.error_history if error.component == component]


    def clear_error_history(self):
        """Очищает историю ошибок"""
        with self.lock:
            self.error_history.clear()
            self.save_error_history()


    def export_error_report(self, output_path: str = None) -> str:
        """
        Экспортирует отчет об ошибках

        Args:
            output_path: Путь для сохранения отчета (если None, генерируется автоматически)

        Returns:
            Путь к экспортированному отчету
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"error_report_{timestamp}.json"

        report = {
            'export_timestamp': datetime.now().isoformat(),
            'total_errors': len(self.error_history),
            'errors': []
        }

        for error in self.error_history:
            report['errors'].append({
                'timestamp': error.timestamp.isoformat(),
                'severity': error.severity.name,
                'message': error.message,
                'exception_type': error.exception_type,
                'exception_message': error.exception_message,
                'component': error.component,
                'has_user_context': error.user_context is not None
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        return output_path

class RecoveryManager:
    """
    Класс менеджера восстановления
    Обеспечивает восстановление после ошибок
    и управление состоянием системы.
    """


    def __init__(self, error_handler: ErrorHandler):
        """
        Инициализирует менеджер восстановления

        Args:
            error_handler: Обработчик ошибок
        """
        self.error_handler = error_handler
        self.state_backups = {}
        self.recovery_strategies = {}
        self.lock = threading.Lock()


    def create_state_backup(self, state_id: str, state_data: Any):
        """
        Создает резервную копию состояния

        Args:
            state_id: Идентификатор состояния
            state_data: Данные состояния
        """
        with self.lock:
            self.state_backups[state_id] = {
                'timestamp': datetime.now(),
                'data': state_data
            }


    def restore_state(self, state_id: str) -> Optional[Any]:
        """
        Восстанавливает состояние из резервной копии

        Args:
            state_id: Идентификатор состояния

        Returns:
            Восстановленные данные состояния или None если не найдено
        """
        with self.lock:
            if state_id in self.state_backups:
                return self.state_backups[state_id]['data']
            return None


    def register_recovery_strategy(self, error_type: str, strategy: Callable):
        """
        Регистрирует стратегию восстановления для типа ошибки

        Args:
            error_type: Тип ошибки
            strategy: Функция стратегии восстановления
        """
        with self.lock:
            self.recovery_strategies[error_type] = strategy


    def attempt_recovery(self, error_info: ErrorInfo) -> bool:
        """
        Пытается восстановиться после ошибки

        Args:
            error_info: Информация об ошибке

        Returns:
            True если восстановление прошло успешно, иначе False
        """
        with self.lock:
            if error_info.exception_type in self.recovery_strategies:
                try:
                    strategy = self.recovery_strategies[error_info.exception_type]
                    return strategy(error_info)
                except Exception as e:
                    self.error_handler.log_error(
                        f"Ошибка при попытке восстановления после {error_info.exception_type}",
                        e,
                        "RecoveryManager",
                        ErrorSeverity.WARNING
                    )
                    return False
            return False


    def cleanup_old_backups(self, retention_hours: int = 24):
        """
        Удаляет старые резервные копии состояния

        Args:
            retention_hours: Время хранения в часах
        """
        with self.lock:
            cutoff_time = datetime.now().timestamp() - (retention_hours * 3600)
            old_backups = []

            for state_id, backup in self.state_backups.items():
                if backup['timestamp'].timestamp() < cutoff_time:
                    old_backups.append(state_id)

            for state_id in old_backups:
                del self.state_backups[state_id]

class SafeExecutor:
    """
    Класс безопасного исполнителя
    Обеспечивает безопасное выполнение кода
    с перехватом и обработкой исключений.
    """


    def __init__(self, error_handler: ErrorHandler):
        """
        Инициализирует безопасный исполнитель

        Args:
            error_handler: Обработчик ошибок
        """
        self.error_handler = error_handler

    def execute_with_retry(self,
                          func: Callable,
                          max_retries: int = 3,
                          retry_delay: float = 1.0,
                          component: str = "SafeExecutor") -> Any:
        """
        Выполняет функцию с повторными попытками при ошибках

        Args:
            func: Функция для выполнения
            max_retries: Максимальное количество попыток
            retry_delay: Задержка между попытками в секундах
            component: Компонент для логирования

        Returns:
            Результат выполнения функции
        """
        for attempt in range(max_retries + 1):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries:
                    # Последняя попытка, регистрируем ошибку и выбрасываем
                    self.error_handler.log_error(
                        f"Ошибка после {max_retries} попыток выполнения {func.__name__}",
                        e,
                        component,
                        ErrorSeverity.ERROR
                    )
                    raise e
                else:
                    # Ждем перед следующей попыткой
                    time.sleep(retry_delay)
                    self.error_handler.log_error(
                        f"Ошибка при попытке {attempt + 1}/{max_retries} выполнения {func.__name__}",
                        e,
                        component,
                        ErrorSeverity.WARNING
                    )

    def execute_with_timeout(self,
                           func: Callable,
                           timeout: float,
                           fallback_return: Any = None,
                           component: str = "SafeExecutor") -> Any:
        """
        Выполняет функцию с таймаутом

        Args:
            func: Функция для выполнения
            timeout: Таймаут в секундах
            fallback_return: Значение по умолчанию при таймауте
            component: Компонент для логирования

        Returns:
            Результат выполнения функции или fallback_return при таймауте
        """
        def target(queue_obj):
            try:
                result = func()
                queue_obj.put(('success', result))
            except Exception as e:
                queue_obj.put(('error', e))

        q = queue.Queue()
        thread = threading.Thread(target=target, args=(q,))
        thread.daemon = True
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            # Таймаут
            self.error_handler.log_error(
                f"Таймаут выполнения функции {func.__name__} (>{timeout}s)",
                component=component,
                severity=ErrorSeverity.WARNING
            )
            return fallback_return
        else:
            try:
                status, result = q.get_nowait()
                if status == 'error':
                    raise result
                return result
            except queue.Empty:
                return fallback_return

def main():
    """Главная функция для демонстрации возможностей обработчика ошибок"""
    print("=== ОБРАБОТЧИК ОШИБОК ПРОЕКТА ===")

    # Создаем обработчик ошибок
    error_handler = ErrorHandler("test_error_log.json")
    recovery_manager = RecoveryManager(error_handler)
    safe_executor = SafeExecutor(error_handler)

    print("✓ Обработчик ошибок инициализирован")
    print(f"✓ Файл лога: {error_handler.log_file}")

    # Демонстрация логирования ошибок
    print("\nТестирование логирования ошибок...")
    try:
        # Искусственно вызываем ошибку
        result = 10 / 0
    except ZeroDivisionError as e:
        error_info = error_handler.log_error(
            "Демонстрационная ошибка деления на ноль",
            e,
            "TestComponent",
            ErrorSeverity.ERROR,
            user_context={"operation": "division", "operands": [10, 0]}
        )
        print(f"✓ Ошибка залогирована: {error_info.message}")

    # Демонстрация декоратора обработки исключений
    print("\nТестирование декоратора обработки исключений...")

    @error_handler.handle_exception(component="TestFunction", fallback_return="default_value")
    def test_function():
        raise ValueError("Тестовая ошибка в функции")

    result = test_function()
    print(f"✓ Функция вернула: {result} (ожидаем значение по умолчанию)")

    # Тестирование безопасного исполнителя с повторными попытками
    print("\nТестирование безопасного исполнителя с повторными попытками...")

    counter = 0

    def flaky_function():
        nonlocal counter
        counter += 1
        if counter < 3:
            raise ConnectionError("Соединение потеряно")
        return "Успех!"

    try:
        result = safe_executor.execute_with_retry(flaky_function, max_retries=5, retry_delay=0.1)
        print(f"✓ Функция выполнена успешно после {counter} попыток: {result}")
    except Exception as e:
        print(f"✗ Ошибка выполнения: {e}")

    # Тестирование безопасного исполнителя с таймаутом
    print("\nТестирование безопасного исполнителя с таймаутом...")

    def slow_function():
        time.sleep(2)
        return "Медленный результат"

    result = safe_executor.execute_with_timeout(slow_function, timeout=1.0, fallback_return="Таймаут!")
    print(f"✓ Результат с таймаутом: {result}")

    # Тестирование восстановления состояния
    print("\nТестирование восстановления состояния...")

    # Создаем резервную копию состояния
    recovery_manager.create_state_backup("test_state", {"data": "important_data", "value": 42})
    print("✓ Создана резервная копия состояния")

    # Восстанавливаем состояние
    restored_data = recovery_manager.restore_state("test_state")
    print(f"✓ Восстановленные данные: {restored_data}")

    # Показываем последние ошибки
    print("\nПоследние ошибки:")
    recent_errors = error_handler.get_recent_errors(5)
    for error in recent_errors:
        print(f"  - [{error.severity.name}] {error.component}: {error.message}")

    # Экспортируем отчет об ошибках
    report_path = error_handler.export_error_report()
    print(f"\n✓ Отчет об ошибках экспортирован: {report_path}")

    print("\nОбработчик ошибок успешно протестирован")
    print("\nДоступные функции:")
    print("- Логирование ошибок: log_error()")
    print("- Декоратор обработки исключений: handle_exception()")
    print("- Безопасное выполнение с повторами: execute_with_retry()")
    print("- Безопасное выполнение с таймаутом: execute_with_timeout()")
    print("- Восстановление состояния: create_state_backup(), restore_state()")
    print("- Отчеты об ошибках: export_error_report()")

if __name__ == "__main__":
    main()

