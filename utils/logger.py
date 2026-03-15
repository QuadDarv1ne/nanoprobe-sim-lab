"""Модуль ведения логов для проекта Лаборатория моделирования нанозонда."""

import logging
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import threading


class JsonFormatter(logging.Formatter):
    """Форматтер для JSON логов"""

    def __init__(self, name: str):
        """
        Инициализация JSON форматтера.

        Args:
            name: Имя логгера
        """
        super().__init__()
        self.name = name

    def format(self, record: logging.LogRecord) -> str:
        """
        Форматирование записи лога в JSON.

        Args:
            record: Запись лога для форматирования

        Returns:
            JSON строка с данными лога
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


class LoggerSetup:
    """
    Класс для настройки системы ведения логов
    Обеспечивает централизованную настройку логирования для всех
    компонентов проекта.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        log_level: str = "INFO",
        max_bytes: int = 10*1024*1024,
        backup_count: int = 5,
        enable_json: bool = False
    ):
        """
        Инициализирует настройщик логов

        Args:
            log_dir: Директория для файлов логов
            log_level: Уровень логирования
            max_bytes: Максимальный размер файла до ротации
            backup_count: Количество резервных файлов
            enable_json: Включить JSON форматирование
        """
        self.log_dir = Path(log_dir)
        self.log_level = getattr(logging, log_level.upper())
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.enable_json = enable_json
        self.setup_logging_directory()

    def setup_logging_directory(self):
        """Создает директорию для логов если она не существует"""
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def create_logger(self, name: str, log_file: Optional[str] = None) -> logging.Logger:
        """
        Создает экземпляр логгера с заданными параметрами

        Args:
            name: Имя логгера
            log_file: Имя файла для логирования (опционально)

        Returns:
            Настроенный экземпляр логгера
        """
        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)

        # Очищаем существующие обработчики чтобы избежать дублирования
        logger.handlers.clear()

        # Формат сообщений
        if self.enable_json:
            formatter = JsonFormatter(name)
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )

        # Обработчик для консоли
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Обработчик для файла с ротацией
        if log_file is None:
            log_file = f"{name}.log"

        file_path = self.log_dir / log_file

        # RotatingFileHandler для управления размером файлов
        file_handler = RotatingFileHandler(
            file_path,
            encoding="utf-8",
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger


class NanoprobeLogger:
    """
    Класс для централизованного ведения логов проекта
    Предоставляет удобный интерфейс для логирования событий
    в различных компонентах проекта.
    """

    def __init__(
        self,
        config_manager=None,
        enable_json: bool = False,
        max_bytes: int = 10*1024*1024,
        backup_count: int = 5
    ):
        """
        Инициализирует логгер проекта

        Args:
            config_manager: Экземпляр менеджера конфигурации (опционально)
            enable_json: Включить JSON форматирование
            max_bytes: Максимальный размер файла до ротации
            backup_count: Количество резервных файлов
        """
        if config_manager:
            log_dir = config_manager.get("paths.log_dir", "logs")
            log_level = config_manager.get("logging.level", "INFO")
        else:
            log_dir = "logs"
            log_level = "INFO"

        self.logger_setup = LoggerSetup(
            log_dir=log_dir,
            log_level=log_level,
            max_bytes=max_bytes,
            backup_count=backup_count,
            enable_json=enable_json
        )
        self.loggers = {}
        self._context: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def get_logger(self, name: str) -> logging.Logger:
        """
        Получает или создает логгер с заданным именем

        Args:
            name: Имя логгера

        Returns:
            Экземпляр логгера
        """
        if name not in self.loggers:
            self.loggers[name] = self.logger_setup.create_logger(name)
        return self.loggers[name]

    def set_context(self, **kwargs):
        """
        Устанавливает контекст для логирования

        Args:
            **kwargs: Пары ключ-значение для контекста
        """
        with self._lock:
            self._context.update(kwargs)

    def clear_context(self):
        """Очищает контекст логирования"""
        with self._lock:
            self._context.clear()

    def _format_message(self, message: str) -> str:
        """Форматирует сообщение с контекстом"""
        if self._context:
            context_str = " | ".join(f"{k}={v}" for k, v in self._context.items())
            return f"[{context_str}] {message}"
        return message

    def get_logger(self, name: str) -> logging.Logger:
        """
        Получает или создает логгер с заданным именем

        Args:
            name: Имя логгера

        Returns:
            Экземпляр логгера
        """
        if name not in self.loggers:
            self.loggers[name] = self.logger_setup.create_logger(name)
        return self.loggers[name]

    def log_spm_event(self, message: str, level: str = "INFO"):
        """
        Логирует событие связанное с СЗМ симулятором

        Args:
            message: Сообщение для логирования
            level: Уровень логирования
        """
        logger = self.get_logger("spm_simulator")
        getattr(logger, level.lower())(message)

    def log_analyzer_event(self, message: str, level: str = "INFO"):
        """
        Логирует событие связанное с анализатором изображений

        Args:
            message: Сообщение для логирования
            level: Уровень логирования
        """
        logger = self.get_logger("image_analyzer")
        getattr(logger, level.lower())(message)

    def log_sstv_event(self, message: str, level: str = "INFO"):
        """
        Логирует событие связанное с SSTV станцией

        Args:
            message: Сообщение для логирования
            level: Уровень логирования
        """
        logger = self.get_logger("sstv_station")
        getattr(logger, level.lower())(message)

    def log_system_event(self, message: str, level: str = "INFO"):
        """
        Логирует системное событие проекта

        Args:
            message: Сообщение для логирования
            level: Уровень логирования
        """
        logger = self.get_logger("nanoprobe_system")
        getattr(logger, level.lower())(self._format_message(message))

    def log_simulation_event(self, message: str, level: str = "INFO"):
        """
        Логирует событие связанное с симуляцией

        Args:
            message: Сообщение для логирования
            level: Уровень логирования
        """
        logger = self.get_logger("simulation")
        getattr(logger, level.lower())(self._format_message(message))

    def log_api_event(self, message: str, level: str = "INFO"):
        """
        Логирует событие API

        Args:
            message: Сообщение для логирования
            level: Уровень логирования
        """
        logger = self.get_logger("api")
        getattr(logger, level.lower())(self._format_message(message))

    def log_error(self, message: str, exc_info: Exception = None):
        """
        Логирует ошибку

        Args:
            message: Сообщение об ошибке
            exc_info: Исключение для трассировки
        """
        logger = self.get_logger("error")
        logger.error(self._format_message(message), exc_info=exc_info)


def setup_project_logging(config_manager=None) -> NanoprobeLogger:
    """
    Настраивает централизованное логирование для проекта

    Args:
        config_manager: Экземпляр менеджера конфигурации (опционально)

    Returns:
        Настроенный экземпляр NanoprobeLogger
    """
    return NanoprobeLogger(config_manager)


def main():
    """Главная функция для демонстрации работы системы логирования"""
    print("=== СИСТЕМА ВЕДЕНИЯ ЛОГОВ ПРОЕКТА ===")

    # Создаем логгер
    logger_manager = NanoprobeLogger()

    # Тестируем различные типы логирования
    logger_manager.log_system_event("Инициализация системы логирования", "INFO")
    logger_manager.log_spm_event("Запуск симуляции СЗМ", "INFO")
    logger_manager.log_analyzer_event("Загрузка изображения для анализа", "INFO")
    logger_manager.log_sstv_event("Поиск SSTV сигнала", "INFO")
    logger_manager.log_simulation_event("Начало симуляции нанозонда", "INFO")

    # Создаем специфический логгер
    custom_logger = logger_manager.get_logger("custom_module")
    custom_logger.info("Сообщение от пользовательского модуля")

    print("✓ Система логирования успешно настроена")
    print(f"✓ Логи сохраняются в директорию: {logger_manager.logger_setup.log_dir}")


if __name__ == "__main__":
    main()
