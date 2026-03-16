"""
Production логирование для Nanoprobe Simulation Lab
Структурированное логирование с ротацией и различными уровнями
"""

import logging
import sys
from pathlib import Path
from datetime import datetime, timezone
from logging.handlers import (
    RotatingFileHandler,
    TimedRotatingFileHandler
)
import json
import traceback


class JSONFormatter(logging.Formatter):
    """
    JSON форматтер для структурированного логирования
    Подходит для отправки в ELK Stack, Splunk, etc.
    """

    def format(self, record):
        log_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Добавляем exception если есть
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info)
            }

        # Добавляем extra поля
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs',
                          'pathname', 'process', 'processName', 'relativeCreated',
                          'stack_info', 'exc_info', 'thread', 'threadName'):
                log_data[key] = value

        return json.dumps(log_data, ensure_ascii=False, default=str)


class ProductionLogger:
    """
    Менеджер production логирования
    """

    def __init__(
        self,
        name: str = 'nanoprobe',
        log_dir: str = 'logs',
        level: int = logging.INFO,
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 10,
        enable_json: bool = False
    ):
        """
        Инициализация production логгера

        Args:
            name: Имя логгера
            log_dir: Директория для логов
            level: Уровень логирования
            max_bytes: Максимальный размер файла до ротации
            backup_count: Количество резервных файлов
            enable_json: Использовать ли JSON формат
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Создание директории для логов
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Форматтеры
        if enable_json:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Rotating file handler (основной лог)
        rotating_handler = RotatingFileHandler(
            log_path / f'{name}.log',
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        rotating_handler.setLevel(level)
        rotating_handler.setFormatter(formatter)
        self.logger.addHandler(rotating_handler)

        # Timed rotating handler (для ежедневных логов)
        timed_handler = TimedRotatingFileHandler(
            log_path / f'{name}_daily.log',
            when='D',
            interval=1,
            backupCount=backup_count,
            encoding='utf-8'
        )
        timed_handler.setLevel(level)
        timed_handler.setFormatter(formatter)
        self.logger.addHandler(timed_handler)

        # Error file handler (только ошибки)
        error_handler = RotatingFileHandler(
            log_path / f'{name}_errors.log',
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        self.logger.addHandler(error_handler)

        # Предотвращение дублирования в родительских логгерах
        self.logger.propagate = False

    def get_logger(self) -> logging.Logger:
        """Получение настроенного логгера"""
        return self.logger

    def debug(self, message: str, **kwargs):
        self.logger.debug(message, extra=kwargs if kwargs else {})

    def info(self, message: str, **kwargs):
        self.logger.info(message, extra=kwargs if kwargs else {})

    def warning(self, message: str, **kwargs):
        self.logger.warning(message, extra=kwargs if kwargs else {})

    def error(self, message: str, **kwargs):
        self.logger.error(message, extra=kwargs if kwargs else {})

    def critical(self, message: str, **kwargs):
        self.logger.critical(message, extra=kwargs if kwargs else {})

    def exception(self, message: str, **kwargs):
        self.logger.exception(message, extra=kwargs if kwargs else {})


# Глобальные логгеры для компонентов
_api_logger = None
_flask_logger = None
_system_logger = None


def get_api_logger() -> ProductionLogger:
    """Получение логгера для FastAPI"""
    global _api_logger
    if _api_logger is None:
        _api_logger = ProductionLogger(
            name='nanoprobe_api',
            log_dir='logs/api',
            enable_json=True  # JSON для API удобно для мониторинга
        )
    return _api_logger


def get_flask_logger() -> ProductionLogger:
    """Получение логгера для Flask"""
    global _flask_logger
    if _flask_logger is None:
        _flask_logger = ProductionLogger(
            name='nanoprobe_flask',
            log_dir='logs/flask',
            enable_json=False  # Человекочитаемый формат для Flask
        )
    return _flask_logger


def get_system_logger() -> ProductionLogger:
    """Получение системного логгера"""
    global _system_logger
    if _system_logger is None:
        _system_logger = ProductionLogger(
            name='nanoprobe_system',
            log_dir='logs/system',
            enable_json=False
        )
    return _system_logger


# Логгер по умолчанию
_default_logger = None


def get_logger(name: str = 'nanoprobe') -> ProductionLogger:
    """Получение логгера по имени"""
    global _default_logger
    if _default_logger is None:
        _default_logger = ProductionLogger(name=name)
    return _default_logger


# Middleware для логирования HTTP запросов (FastAPI/Starlette)
class HTTPLoggingMiddleware:
    """
    Middleware для логирования HTTP запросов
    Использование в FastAPI:
        app.add_middleware(HTTPLoggingMiddleware)
    """

    def __init__(self, app):
        self.app = app
        self.logger = get_api_logger().get_logger()

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)

        # Начало запроса
        method = scope['method']
        path = scope['path']
        start_time = datetime.now()

        self.logger.info(f"→ {method} {path}")

        # Обработка запроса
        try:
            await self.app(scope, receive, send)
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.error(f"✗ {method} {path} - {duration:.2f}ms - {str(e)}")
            raise

        # Завершение (логирование в send не работает для response status)
        # Для полного логирования нужно кастомизировать send


# Контекстный менеджер для логирования времени выполнения
class LogExecutionTime:
    """
    Контекстный менеджер для логирования времени выполнения
    Использование:
        with LogExecutionTime("Операция", logger):
            # код
    """

    def __init__(self, operation: str, logger: logging.Logger = None):
        self.operation = operation
        self.logger = logger or get_logger().get_logger()
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"Начало: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()

        if exc_type:
            self.logger.error(
                f"Ошибка: {self.operation} - {duration:.3f}s - {str(exc_val)}",
                exc_info=True
            )
        else:
            self.logger.debug(f"Завершено: {self.operation} - {duration:.3f}s")

        return False  # Не подавляем исключения


# Декоратор для логирования вызовов функций
def log_function_call(logger: logging.Logger = None):
    """
    Декоратор для логирования вызовов функций
    Использование:
        @log_function_call(api_logger.get_logger())
        def my_function():
            pass
    """
    def decorator(func):
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log = logger or get_logger().get_logger()
            log.debug(f"Вызов {func.__name__} с args={args}, kwargs={kwargs}")

            start = datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start).total_seconds() * 1000
                log.debug(f"{func.__name__} завершён за {duration:.2f}ms")
                return result
            except Exception as e:
                duration = (datetime.now() - start).total_seconds() * 1000
                log.error(f"{func.__name__} ошибка через {duration:.2f}ms: {e}", exc_info=True)
                raise

        return wrapper
    return decorator


# Пример использования
if __name__ == "__main__":
    print("=== Тестирование Production логирования ===\n")

    # Создание логгера
    logger = get_logger('test')

    # Тестирование различных уровней
    logger.debug("Debug сообщение", extra={'user_id': 123, 'action': 'test'})
    logger.info("Info сообщение")
    logger.warning("Warning сообщение")
    logger.error("Error сообщение")

    # Тест exception
    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception("Произошло исключение деления на ноль")

    # Тест JSON форматтера
    json_logger = ProductionLogger(name='json_test', enable_json=True)
    json_logger.info("JSON сообщение", extra={'data': {'key': 'value'}})

    # Тест контекстного менеджера
    with LogExecutionTime("Тестовая операция", logger.get_logger()):
        import time
        time.sleep(0.1)

    print("\n✓ Тестирование завершено")
    print("Проверьте логи в директории logs/")
