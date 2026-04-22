"""Модуль ведения логов для проекта Лаборатория моделирования нанозонда."""

import logging
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .structured_logger import HumanReadableFormatter, JSONFormatter


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
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        enable_json: bool = False,
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

        # Хранилище для отслеживания созданных логгеров для возможности перенастройки
        self._created_loggers: Dict[str, logging.Logger] = {}
        self._lock = threading.Lock()

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
            formatter = JSONFormatter()
        else:
            formatter = HumanReadableFormatter()

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
            file_path, encoding="utf-8", maxBytes=self.max_bytes, backupCount=self.backup_count
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Предотвращение дублирования в родительских логгерах
        logger.propagate = False

        # Отслеживаем созданный логгер для возможности перенастройки
        with self._lock:
            self._created_loggers[name] = logger

        return logger

    def update_level(self, level: Union[str, int]):
        """
        Обновляет уровень логирования для всехManaged логгеров

        Args:
            level: Новый уровень логирования (строка или константа logging)
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper())

        with self._lock:
            self.log_level = level
            # Обновляем уровень для всехManaged логгеров и их обработчиков
            for logger in self._created_loggers.values():
                logger.setLevel(level)
                for handler in logger.handlers:
                    handler.setLevel(level)

    def update_format(self, enable_json: bool):
        """
        Обновляет формат логов для всехManaged логгеров

        Args:
            enable_json: Если True - использовать JSON формат, иначе человекочитаемый
        """
        with self._lock:
            self.enable_json = enable_json
            # Обновляем форматтер для всехManaged логгеров
            for logger in self._created_loggers.values():
                # Очищаем существующие обработчики
                logger.handlers.clear()
                # Создаем новые обработчики с обновленным форматом
                formatter = JSONFormatter() if enable_json else HumanReadableFormatter()

                # Консольный обработчик
                console_handler = logging.StreamHandler()
                console_handler.setLevel(self.log_level)
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)

                # Файловые обработчики (восстанавливаем их)
                for handler in list(
                    logger.handlers
                ):  # Копируем список, так как будем модифицировать
                    if isinstance(handler, RotatingFileHandler):
                        # Оставляем файловые обработчики как есть, но обновляем форматтер
                        handler.setFormatter(formatter)

    def add_file_handler(
        self,
        logger_name: str,
        log_file: str,
        level: Optional[Union[str, int]] = None,
        max_bytes: Optional[int] = None,
        backup_count: Optional[int] = None,
    ) -> bool:
        """
        Добавляет дополнительный файловый обработчик к указанному логгеру

        Args:
            logger_name: Имя логгера
            log_file: Путь к файлу лога
            level: Уровень логирования для этого обработчика (опционально)
            max_bytes: Максимальный размер файла до ротации (опционально)
            backup_count: Количество резервных файлов (опционально)

        Returns:
            True если обработчик добавлен успешно, False если логгер не найден
        """
        if level is None:
            level = self.log_level
        elif isinstance(level, str):
            level = getattr(logging, level.upper())

        if max_bytes is None:
            max_bytes = self.max_bytes
        if backup_count is None:
            backup_count = self.backup_count

        with self._lock:
            logger = self._created_loggers.get(logger_name)
            if logger is None:
                return False

            formatter = JSONFormatter() if self.enable_json else HumanReadableFormatter()
            file_path = self.log_dir / log_file

            file_handler = RotatingFileHandler(
                file_path, encoding="utf-8", maxBytes=max_bytes, backupCount=backup_count
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            return True

    def remove_handlers_by_type(self, logger_name: str, handler_type: type) -> int:
        """
        Удаляет обработчики определенного типа у указанного логгера

        Args:
            logger_name: Имя логгера
            handler_type: Тип обработчика для удаления (например, RotatingFileHandler)

        Returns:
            Количество удаленных обработчиков
        """
        with self._lock:
            logger = self._created_loggers.get(logger_name)
            if logger is None:
                return 0

            removed_count = 0
            handlers_to_remove = [h for h in logger.handlers if isinstance(h, handler_type)]

            for handler in handlers_to_remove:
                logger.removeHandler(handler)
                handler.close()
                removed_count += 1

            return removed_count

    def get_logger_names(self) -> list[str]:
        """
        Возвращает список всехManaged имен логгеров

        Returns:
            Список имен логгеров
        """
        with self._lock:
            return list(self._created_loggers.keys())


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
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
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
            enable_json=enable_json,
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
        """Форматирует сообщение с контекстом и маскирует чувствительную информацию"""
        if self._context:
            masked_context = {}
            for k, v in self._context.items():
                # Маскируем значения для ключей, которые могут содержать чувствительную информацию
                if any(
                    secret_key in k.lower()
                    for secret_key in [
                        "password",
                        "pass",
                        "pwd",
                        "secret",
                        "key",
                        "token",
                        "auth",
                        "authorization",
                    ]
                ):
                    masked_context[k] = "*****"
                else:
                    masked_context[k] = v
            context_str = " | ".join(f"{k}={v}" for k, v in masked_context.items())
            return f"[{context_str}] {message}"
        return message

    def log_spm_event(self, message: str, level: str = "INFO"):
        """
        Логирует событие связанное с СЗМ симулятором

        Args:
            message: Сообщение для логирования
            level: Уровень логирования
        """
        logger = self.get_logger("spm_simulator")
        getattr(logger, level.lower())(self._format_message(message))

    def log_analyzer_event(self, message: str, level: str = "INFO"):
        """
        Логирует событие связанное с анализатором изображений

        Args:
            message: Сообщение для логирования
            level: Уровень логирования
        """
        logger = self.get_logger("image_analyzer")
        getattr(logger, level.lower())(self._format_message(message))

    def log_sstv_event(self, message: str, level: str = "INFO"):
        """
        Логирует событие связанное с SSTV станцией

        Args:
            message: Сообщение для логирования
            level: Уровень логирования
        """
        logger = self.get_logger("sstv_station")
        getattr(logger, level.lower())(self._format_message(message))

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

    # Новые методы для улучшенного управления логированием

    def update_global_level(self, level: Union[str, int]):
        """
        Обновляет уровень логирования глобально для всехManaged логгеров

        Args:
            level: Новый уровень логирования (строка или константа logging)
        """
        self.logger_setup.update_level(level)

    def update_global_format(self, enable_json: bool):
        """
        Обновляет формат логов глобально для всехManaged логгеров

        Args:
            enable_json: Если True - использовать JSON формат, иначе человекочитаемый
        """
        self.logger_setup.update_format(enable_json)

    def add_global_file_handler(
        self,
        log_file: str,
        level: Optional[Union[str, int]] = None,
        max_bytes: Optional[int] = None,
        backup_count: Optional[int] = None,
    ) -> bool:
        """
        Добавляет глобальный файловый обработчик ко всемManaged логгерам

        Args:
            log_file: Путь к файлу лога (относительно log_dir)
            level: Уровень логирования для этого обработчика (опционально)
            max_bytes: Максимальный размер файла до ротации (опционально)
            backup_count: Количество резервных файлов (опционально)

        Returns:
            True если обработчик добавлен успешно ко всем логгерам
        """
        success = True
        logger_names = self.logger_setup.get_logger_names()
        for logger_name in logger_names:
            if not self.logger_setup.add_file_handler(
                logger_name, log_file, level, max_bytes, backup_count
            ):
                success = False
        return success

    def get_logging_status(self) -> Dict[str, Any]:
        """
        Возвращает текущий статус системы логирования

        Returns:
            Словарь с информацией о текущей конфигурации логирования
        """
        with self._lock:
            return {
                "log_dir": str(self.logger_setup.log_dir),
                "log_level": logging.getLevelName(self.logger_setup.log_level),
                "enable_json": self.logger_setup.enable_json,
                "managed_loggers": self.logger_setup.get_logger_names(),
                "context": dict(self._context),
            }


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

    # Демонстрация новых возможностей
    print("\n--- Демонстрация новых возможностей ---")

    # Изменение уровня логирования
    logger_manager.update_global_level("DEBUG")
    logger_manager.log_system_event("Уровень изменен на DEBUG", "DEBUG")

    # Изменение формата
    logger_manager.update_global_format(True)
    logger_manager.log_system_event("Формат изменен на JSON", "INFO")

    # Добавление дополнительного файла логов
    logger_manager.add_global_file_handler("debug.log", "DEBUG")
    logger_manager.log_system_event("Добавлен дополнительный обработчик для debug.log", "INFO")

    # Показ статуса
    status = logger_manager.get_logging_status()
    print(f"Текущий статус логирования: {status}")

    print("✓ Система логирования успешно настроена")
    print(f"✓ Логи сохраняются в директорию: {logger_manager.logger_setup.log_dir}")


if __name__ == "__main__":
    main()
