"""Unit-тесты для модуля обработки ошибок."""

import unittest
import tempfile
import shutil
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.core.error_handler import (
    ErrorHandler, ErrorSeverity, ErrorInfo,
    RecoveryManager, SafeExecutor
)


class TestErrorHandler(unittest.TestCase):
    """Тесты для класса ErrorHandler"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = Path(self.temp_dir) / "errors.json"

    def tearDown(self):
        """Очистка после теста"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Тестирует инициализацию ErrorHandler"""
        handler = ErrorHandler(str(self.log_file))

        self.assertEqual(handler.log_file, self.log_file)
        self.assertEqual(handler.max_log_size, 1000)
        self.assertIsInstance(handler.error_history, list)

    def test_log_error_without_exception(self):
        """Тестирует логирование ошибки без исключения"""
        handler = ErrorHandler(str(self.log_file))

        error_info = handler.log_error(
            message="Test error message",
            component="TestComponent",
            severity=ErrorSeverity.ERROR
        )

        self.assertIsInstance(error_info, ErrorInfo)
        self.assertEqual(error_info.message, "Test error message")
        self.assertEqual(error_info.component, "TestComponent")
        self.assertEqual(error_info.severity, ErrorSeverity.ERROR)

    def test_log_error_with_exception(self):
        """Тестирует логирование ошибки с исключением"""
        handler = ErrorHandler(str(self.log_file))

        try:
            raise ValueError("Test exception")
        except ValueError as e:
            error_info = handler.log_error(
                message="Error with exception",
                exception=e,
                component="TestComponent",
                severity=ErrorSeverity.ERROR
            )

        self.assertEqual(error_info.exception_type, "ValueError")
        self.assertEqual(error_info.exception_message, "Test exception")

    def test_get_recent_errors(self):
        """Тестирует получение последних ошибок"""
        handler = ErrorHandler(str(self.log_file))

        # Логируем несколько ошибок
        for i in range(5):
            handler.log_error(f"Error {i}", component=f"Component{i}")

        recent = handler.get_recent_errors(3)

        self.assertEqual(len(recent), 3)
        self.assertEqual(recent[-1].message, "Error 4")

    def test_get_errors_by_severity(self):
        """Тестирует фильтрацию ошибок по уровню"""
        handler = ErrorHandler(str(self.log_file))

        handler.log_error("Error 1", severity=ErrorSeverity.INFO)
        handler.log_error("Error 2", severity=ErrorSeverity.ERROR)
        handler.log_error("Error 3", severity=ErrorSeverity.ERROR)

        errors = handler.get_errors_by_severity(ErrorSeverity.ERROR)

        self.assertEqual(len(errors), 2)

    def test_get_errors_by_component(self):
        """Тестирует фильтрацию ошибок по компоненту"""
        handler = ErrorHandler(str(self.log_file))

        handler.log_error("Error 1", component="ComponentA")
        handler.log_error("Error 2", component="ComponentA")
        handler.log_error("Error 3", component="ComponentB")

        errors = handler.get_errors_by_component("ComponentA")

        self.assertEqual(len(errors), 2)

    def test_clear_error_history(self):
        """Тестирует очистку истории ошибок"""
        handler = ErrorHandler(str(self.log_file))

        handler.log_error("Error 1")
        handler.log_error("Error 2")

        self.assertEqual(len(handler.error_history), 2)

        handler.clear_error_history()

        self.assertEqual(len(handler.error_history), 0)

    def test_export_error_report(self):
        """Тестирует экспорт отчёта об ошибках"""
        handler = ErrorHandler(str(self.log_file))

        handler.log_error("Test error 1", component="Test")
        handler.log_error("Test error 2", component="Test")

        report_path = handler.export_error_report()

        self.assertTrue(Path(report_path).exists())

        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)

        self.assertEqual(report['total_errors'], 2)
        self.assertEqual(len(report['errors']), 2)


class TestRecoveryManager(unittest.TestCase):
    """Тесты для класса RecoveryManager"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = Path(self.temp_dir) / "errors.json"
        self.error_handler = ErrorHandler(str(self.log_file))
        self.recovery_manager = RecoveryManager(self.error_handler)

    def tearDown(self):
        """Очистка после теста"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Тестирует инициализацию RecoveryManager"""
        self.assertEqual(self.recovery_manager.error_handler, self.error_handler)
        self.assertIsInstance(self.recovery_manager.state_backups, dict)
        self.assertIsInstance(self.recovery_manager.recovery_strategies, dict)

    def test_create_state_backup(self):
        """Тестирует создание резервной копии состояния"""
        test_state = {"key": "value", "count": 42}

        self.recovery_manager.create_state_backup("test_state", test_state)

        self.assertIn("test_state", self.recovery_manager.state_backups)
        self.assertEqual(
            self.recovery_manager.state_backups["test_state"]["data"],
            test_state
        )

    def test_restore_state_existing(self):
        """Тестирует восстановление существующего состояния"""
        test_state = {"data": "important"}

        self.recovery_manager.create_state_backup("state1", test_state)
        restored = self.recovery_manager.restore_state("state1")

        self.assertEqual(restored, test_state)

    def test_restore_state_nonexistent(self):
        """Тестирует восстановление несуществующего состояния"""
        restored = self.recovery_manager.restore_state("nonexistent")

        self.assertIsNone(restored)

    def test_register_recovery_strategy(self):
        """Тестирует регистрацию стратегии восстановления"""
        def recovery_func(error_info):
            """TODO: Add description"""
            return True

        self.recovery_manager.register_recovery_strategy("ValueError", recovery_func)

        self.assertIn("ValueError", self.recovery_manager.recovery_strategies)


class TestSafeExecutor(unittest.TestCase):
    """Тесты для класса SafeExecutor"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = Path(self.temp_dir) / "errors.json"
        self.error_handler = ErrorHandler(str(self.log_file))
        self.executor = SafeExecutor(self.error_handler)

    def tearDown(self):
        """Очистка после теста"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_execute_successful_function(self):
        """Тестирует выполнение успешной функции"""
        def successful_func():
            """TODO: Add description"""
            return 42

        result = self.executor.execute_with_retry(successful_func)

        self.assertEqual(result, 42)

    def test_execute_with_retry_success_after_failures(self):
        """Тестирует выполнение с повторными попытками"""
        attempts = [0]

        def flaky_function():
            """TODO: Add description"""
            attempts[0] += 1
            if attempts[0] < 3:
                raise ConnectionError("Connection failed")
            return "success"

        result = self.executor.execute_with_retry(
            flaky_function,
            max_retries=5,
            retry_delay=0.01
        )

        self.assertEqual(result, "success")
        self.assertEqual(attempts[0], 3)

    def test_execute_with_retry_all_failures(self):
        """Тестирует выполнение, когда все попытки неудачны"""
        def always_fails():
            """TODO: Add description"""
            raise ValueError("Always fails")

        with self.assertRaises(ValueError):
            self.executor.execute_with_retry(
                always_fails,
                max_retries=2,
                retry_delay=0.01
            )

    def test_execute_with_timeout_success(self):
        """Тестирует выполнение с таймаутом (успех)"""
        def quick_function():
            """TODO: Add description"""
            return "fast result"

        result = self.executor.execute_with_timeout(quick_function, timeout=1.0)

        self.assertEqual(result, "fast result")

    def test_execute_with_timeout_fallback(self):
        """Тестирует выполнение с таймаутом (возврат fallback)"""
        def slow_function():
            """TODO: Add description"""
            import time
            time.sleep(2)
            return "slow result"

        result = self.executor.execute_with_timeout(
            slow_function,
            timeout=0.1,
            fallback_return="timeout fallback"
        )

        self.assertEqual(result, "timeout fallback")


class TestErrorInfo(unittest.TestCase):
    """Тесты для класса ErrorInfo"""

    def test_error_info_creation(self):
        """Тестирует создание ErrorInfo"""
        error_info = ErrorInfo(
            timestamp=datetime.now(timezone.utc),
            severity=ErrorSeverity.ERROR,
            message="Test message",
            exception_type="ValueError",
            exception_message="Test exception",
            traceback_info="Traceback",
            component="TestComponent"
        )

        self.assertEqual(error_info.message, "Test message")
        self.assertEqual(error_info.severity, ErrorSeverity.ERROR)
        self.assertEqual(error_info.component, "TestComponent")

    def test_error_info_with_context(self):
        """Тестирует создание ErrorInfo с контекстом"""
        context = {"user_id": 123, "action": "test"}

        error_info = ErrorInfo(
            timestamp=datetime.now(timezone.utc),
            severity=ErrorSeverity.WARNING,
            message="Warning message",
            exception_type="Warning",
            exception_message="Warning text",
            traceback_info="",
            component="Test",
            user_context=context
        )

        self.assertEqual(error_info.user_context, context)


if __name__ == '__main__':
    unittest.main()
