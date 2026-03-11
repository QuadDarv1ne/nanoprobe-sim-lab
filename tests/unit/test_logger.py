# -*- coding: utf-8 -*-
"""Unit-тесты для модуля логирования."""

import unittest
import tempfile
import shutil
import logging
import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import LoggerSetup, NanoprobeLogger


class TestLoggerSetup(unittest.TestCase):
    """Тесты для класса LoggerSetup"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / "logs"

    def tearDown(self):
        """Очистка после теста"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Тестирует инициализацию LoggerSetup"""
        logger_setup = LoggerSetup(str(self.log_dir), "INFO")
        
        self.assertEqual(logger_setup.log_dir, self.log_dir)
        self.assertEqual(logger_setup.log_level, logging.INFO)
        self.assertTrue(self.log_dir.exists())

    def test_create_logger(self):
        """Тестирует создание логгера"""
        logger_setup = LoggerSetup(str(self.log_dir), "DEBUG")
        logger = logger_setup.create_logger("test_logger")
        
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "test_logger")
        self.assertEqual(logger.level, logging.DEBUG)

    def test_create_logger_with_custom_file(self):
        """Тестирует создание логгера с кастомным файлом"""
        logger_setup = LoggerSetup(str(self.log_dir), "INFO")
        logger = logger_setup.create_logger("custom", "custom_log.log")
        
        log_file = self.log_dir / "custom_log.log"
        self.assertTrue(log_file.exists())

    def test_log_levels(self):
        """Тестирует разные уровни логирования"""
        for level_name, expected_level in [
            ("DEBUG", logging.DEBUG),
            ("INFO", logging.INFO),
            ("WARNING", logging.WARNING),
            ("ERROR", logging.ERROR),
            ("CRITICAL", logging.CRITICAL),
        ]:
            logger_setup = LoggerSetup(str(self.log_dir), level_name)
            self.assertEqual(logger_setup.log_level, expected_level)


class TestNanoprobeLogger(unittest.TestCase):
    """Тесты для класса NanoprobeLogger"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / "logs"

    def tearDown(self):
        """Очистка после теста"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization_default(self):
        """Тестирует инициализацию с параметрами по умолчанию"""
        logger = NanoprobeLogger()
        
        self.assertIsNotNone(logger.logger_setup)
        self.assertIsInstance(logger.loggers, dict)

    def test_get_logger(self):
        """Тестирует получение логгера"""
        logger = NanoprobeLogger()
        
        log1 = logger.get_logger("module1")
        log2 = logger.get_logger("module1")
        
        # Должен возвращаться тот же экземпляр
        self.assertEqual(log1, log2)

    def test_get_different_loggers(self):
        """Тестирует получение разных логгеров"""
        logger = NanoprobeLogger()
        
        log1 = logger.get_logger("module1")
        log2 = logger.get_logger("module2")
        
        self.assertNotEqual(log1, log2)
        self.assertEqual(log1.name, "module1")
        self.assertEqual(log2.name, "module2")

    def test_log_spm_event(self):
        """Тестирует логирование событий СЗМ"""
        logger = NanoprobeLogger()
        
        # Не должно вызвать исключения
        logger.log_spm_event("Тестовое сообщение СЗМ", "INFO")
        
        # Проверяем, что логгер создался
        self.assertIn("spm_simulator", logger.loggers)

    def test_log_analyzer_event(self):
        """Тестирует логирование событий анализатора"""
        logger = NanoprobeLogger()
        
        logger.log_analyzer_event("Тестовое сообщение анализатора", "WARNING")
        
        self.assertIn("image_analyzer", logger.loggers)

    def test_log_sstv_event(self):
        """Тестирует логирование событий SSTV"""
        logger = NanoprobeLogger()
        
        logger.log_sstv_event("Тестовое сообщение SSTV", "ERROR")
        
        self.assertIn("sstv_station", logger.loggers)

    def test_log_system_event(self):
        """Тестирует логирование системных событий"""
        logger = NanoprobeLogger()
        
        logger.log_system_event("Тестовое системное сообщение", "INFO")
        
        self.assertIn("nanoprobe_system", logger.loggers)

    def test_log_simulation_event(self):
        """Тестирует логирование событий симуляции"""
        logger = NanoprobeLogger()
        
        logger.log_simulation_event("Тестовое сообщение симуляции", "DEBUG")
        
        self.assertIn("simulation", logger.loggers)


class TestLoggerIntegration(unittest.TestCase):
    """Интеграционные тесты для логирования"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / "logs"

    def tearDown(self):
        """Очистка после теста"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_logging_to_file(self):
        """Тестирует запись логов в файл"""
        logger_setup = LoggerSetup(str(self.log_dir), "INFO")
        logger = logger_setup.create_logger("file_test", "test.log")
        
        test_message = "Test message for file logging"
        logger.info(test_message)
        
        # Проверяем, что сообщение записалось в файл
        log_file = self.log_dir / "test.log"
        self.assertTrue(log_file.exists())
        
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn(test_message, content)

    def test_multiple_handlers(self):
        """Тестирует несколько обработчиков логов"""
        logger_setup = LoggerSetup(str(self.log_dir), "DEBUG")
        logger = logger_setup.create_logger("multi_handler")
        
        # У логгера должны быть обработчики для консоли и файла
        self.assertGreaterEqual(len(logger.handlers), 1)


if __name__ == '__main__':
    unittest.main()
