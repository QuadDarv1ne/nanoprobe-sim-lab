# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Модуль ведения логов для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет централизованную систему ведения логов
для всех компонентов проекта.
"""

import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

class LoggerSetup:
    """
    Класс для настройки системы ведения логов
    Обеспечивает централизованную настройку логирования для всех
    компонентов проекта.
    """


    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        """
        Инициализирует настройщик логов

        Args:
            log_dir: Директория для файлов логов
            log_level: Уровень логирования
        """
        self.log_dir = Path(log_dir)
        self.log_level = getattr(logging, log_level.upper())
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
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Обработчик для консоли
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Обработчик для файла
        if log_file is None:
            log_file = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        file_path = self.log_dir / log_file
        file_handler = logging.FileHandler(file_path, encoding='utf-8')
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


    def __init__(self, config_manager=None):
        """
        Инициализирует логгер проекта

        Args:
            config_manager: Экземпляр менеджера конфигурации (опционально)
        """
        if config_manager:
            log_dir = config_manager.get("paths.log_dir", "logs")
            log_level = config_manager.get("logging.level", "INFO")
        else:
            log_dir = "logs"
            log_level = "INFO"

        self.logger_setup = LoggerSetup(log_dir, log_level)
        self.loggers = {}


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
        getattr(logger, level.lower())(message)


    def log_simulation_event(self, message: str, level: str = "INFO"):
        """
        Логирует событие связанное с симуляцией

        Args:
            message: Сообщение для логирования
            level: Уровень логирования
        """
        logger = self.get_logger("simulation")
        getattr(logger, level.lower())(message)

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

