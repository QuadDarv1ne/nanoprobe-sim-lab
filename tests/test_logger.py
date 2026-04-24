#!/usr/bin/env python3
"""
Тесты для улучшенной системы логирования
"""

import gc
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

from utils.logger import LoggerSetup, NanoprobeLogger


def test_logger_setup():
    """Тест настройки логгера"""
    print("Тест LoggerSetup...")

    with tempfile.TemporaryDirectory() as tmpdir:
        setup = LoggerSetup(
            log_dir=tmpdir, log_level="DEBUG", max_bytes=1024 * 1024, backup_count=3
        )

        logger = setup.create_logger("test_logger")

        assert logger.level == logging.DEBUG
        assert len(logger.handlers) == 2  # Console + File

        # Закрываем обработчики
        for handler in logger.handlers:
            handler.close()

        print("[PASS] LoggerSetup")
        return True


def test_rotating_file_handler():
    """Тест ротации файлов"""
    print("Тест ротации файлов...")

    with tempfile.TemporaryDirectory() as tmpdir:
        setup = LoggerSetup(
            log_dir=tmpdir, max_bytes=500, backup_count=2  # Маленький размер для теста
        )

        logger = setup.create_logger("rotate_test", log_file="rotate.log")

        # Пишем много данных
        for i in range(50):
            logger.info(f"Test message {i}" * 10)

        # Закрываем обработчики
        for handler in logger.handlers:
            handler.close()

        # Проверяем наличие файлов ротации
        log_files = list(Path(tmpdir).glob("rotate.log*"))
        assert len(log_files) >= 1, "Файл лога должен существовать"

        print(f"[PASS] Ротация файлов (создано файлов: {len(log_files)})")
        return True


def test_json_formatter():
    """Тест JSON форматтера"""
    print("Тест JSON formatter...")

    with tempfile.TemporaryDirectory() as tmpdir:
        setup = LoggerSetup(log_dir=tmpdir, enable_json=True)

        logger = setup.create_logger("json_test", log_file="json.log")

        logger.info("Test JSON message")

        # Закрываем обработчики
        for handler in logger.handlers:
            handler.close()

        # Читаем лог файл
        log_file = Path(tmpdir) / "json.log"
        with open(log_file, "r", encoding="utf-8") as f:
            line = f.readline()

        import json

        log_data = json.loads(line)

        assert "timestamp" in log_data
        assert "level" in log_data
        assert "message" in log_data
        assert log_data["message"] == "Test JSON message"
        assert log_data["level"] == "INFO"

        print("[PASS] JSON formatter")
        return True


def test_context_logging():
    """Тест контекстного логирования"""
    print("Тест контекстного логирования...")

    tmpdir = Path("tests/temp_logger_test")
    tmpdir.mkdir(exist_ok=True)

    try:
        # Создаем фиктивный config_manager
        class FakeConfig:
            """Фиктивный конфигурационный менеджер для тестирования"""

            def get(self, key, default):
                """Получение параметра конфигурации по ключу"""
                if key == "paths.log_dir":
                    return str(tmpdir)
                if key == "logging.level":
                    return "INFO"
                return default

        logger_mgr = NanoprobeLogger(config_manager=FakeConfig())

        # Устанавливаем контекст
        logger_mgr.set_context(user="admin", action="test", request_id="12345")

        logger_mgr.log_system_event("Test event with context")

        # Закрываем все обработчики
        for logger in logger_mgr.loggers.values():
            for handler in logger.handlers:
                handler.close()

        # Принудительная очистка для Windows
        gc.collect()
        time.sleep(0.2)

        # Проверяем лог
        log_file = tmpdir / "nanoprobe_system.log"
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()

        assert "user=admin" in content
        assert "action=test" in content
        assert "request_id=12345" in content

        # Очищаем контекст
        logger_mgr.clear_context()
        logger_mgr.log_system_event("Test event without context")

        print("[PASS] Контекстное логирование")
        return True
    finally:
        # Очистка тестовой директории
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


def test_error_logging_with_exception():
    """Тест логирования ошибок с исключением"""
    print("Тест логирования ошибок...")

    with tempfile.TemporaryDirectory() as tmpdir:

        class FakeConfig:
            """Фиктивный конфигурационный менеджер для тестирования"""

            def get(self, key, default):
                """Получение параметра конфигурации по ключу"""
                if key == "paths.log_dir":
                    return tmpdir
                if key == "logging.level":
                    return "INFO"
                return default

        logger_mgr = NanoprobeLogger(config_manager=FakeConfig())

        try:
            raise ValueError("Test error message")
        except Exception as e:
            logger_mgr.log_error("Error occurred", exc_info=e)

        # Закрываем обработчики
        for logger in logger_mgr.loggers.values():
            for handler in logger.handlers:
                handler.close()

        # Проверяем лог
        log_file = Path(tmpdir) / "error.log"
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()

        assert "Error occurred" in content
        assert "ValueError" in content

        print("[PASS] Логирование ошибок")
        return True


def test_multiple_loggers():
    """Тест множественных логгеров"""
    print("Тест множественных логгеров...")

    with tempfile.TemporaryDirectory() as tmpdir:

        class FakeConfig:
            """Фиктивный конфигурационный менеджер для тестирования"""

            def get(self, key, default):
                """Получение параметра конфигурации по ключу"""
                if key == "paths.log_dir":
                    return tmpdir
                if key == "logging.level":
                    return "INFO"
                return default

        logger_mgr = NanoprobeLogger(config_manager=FakeConfig())

        # Получаем разные логгеры
        logger1 = logger_mgr.get_logger("module1")
        logger2 = logger_mgr.get_logger("module2")
        logger3 = logger_mgr.get_logger("module1")  # Должен вернуть тот же

        assert logger1 is logger3, "Одинаковые логгеры должны быть одним объектом"
        assert logger1 is not logger2, "Разные логгеры должны быть разными объектами"

        logger1.info("Message from module1")
        logger2.info("Message from module2")

        # Закрываем обработчики
        for logger in logger_mgr.loggers.values():
            for handler in logger.handlers:
                handler.close()

        # Проверяем файлы
        log1 = Path(tmpdir) / "module1.log"
        log2 = Path(tmpdir) / "module2.log"

        assert log1.exists()
        assert log2.exists()

        print("[PASS] Множественные логгеры")
        return True


def test_log_api_event():
    """Тест логирования API событий"""
    print("Тест API событий...")

    with tempfile.TemporaryDirectory() as tmpdir:

        class FakeConfig:
            """Фиктивный конфигурационный менеджер для тестирования"""

            def get(self, key, default):
                """Получение параметра конфигурации по ключу"""
                if key == "paths.log_dir":
                    return tmpdir
                if key == "logging.level":
                    return "INFO"
                return default

        logger_mgr = NanoprobeLogger(config_manager=FakeConfig())

        logger_mgr.log_api_event("API request received", "INFO")
        logger_mgr.log_api_event("Authentication failed", "WARNING")

        # Закрываем обработчики
        for logger in logger_mgr.loggers.values():
            for handler in logger.handlers:
                handler.close()

        log_file = Path(tmpdir) / "api.log"
        assert log_file.exists()

        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()

        assert "API request received" in content
        assert "Authentication failed" in content

        print("[PASS] API события")
        return True


def main():
    """Запуск всех тестов"""
    print("=" * 60)
    print("ТЕСТЫ СИСТЕМЫ ЛОГИРОВАНИЯ")
    print("=" * 60)

    tests = [
        test_logger_setup,
        test_rotating_file_handler,
        test_json_formatter,
        test_context_logging,
        test_error_logging_with_exception,
        test_multiple_loggers,
        test_log_api_event,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"ИТОГИ: {passed}/{len(tests)} тестов пройдено ({passed/len(tests)*100:.1f}%)")

    if passed == len(tests):
        print("Все тесты пройдены!")
        return 0
    else:
        print(f"{failed} тест(а) провалено")
        return 1


if __name__ == "__main__":
    sys.exit(main())
