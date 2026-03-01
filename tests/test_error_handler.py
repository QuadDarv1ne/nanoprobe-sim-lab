#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тесты для модуля обработки ошибок
"""

import unittest
import tempfile
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

from error_handler import ErrorHandler, ErrorSeverity, ErrorInfo, RecoveryManager, SafeExecutor


class TestErrorSeverity(unittest.TestCase):
    """Тесты для перечисления ErrorSeverity"""

    def test_severity_values(self):
        """Тестирует значения уровней важности"""
        self.assertEqual(ErrorSeverity.DEBUG.value, 10)
        self.assertEqual(ErrorSeverity.INFO.value, 20)
        self.assertEqual(ErrorSeverity.WARNING.value, 30)
        self.assertEqual(ErrorSeverity.ERROR.value, 40)
        self.assertEqual(ErrorSeverity.CRITICAL.value, 50)


class TestErrorHandler(unittest.TestCase):
    """Тесты для класса ErrorHandler"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = Path(self.temp_dir) / "error_log.json"

    def tearDown(self):
        """Очистка после теста"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Тестирует инициализацию ErrorHandler"""
        handler = ErrorHandler(str(self.log_file))
        self.assertEqual(handler.log_file, self.log_file)
        self.assertEqual(handler.max_log_size, 1000)

    def test_log_error(self):
        """Тестирует логирование ошибки"""
        handler = ErrorHandler(str(self.log_file))
        
        error_info = handler.log_error(
            "Test error message",
            ValueError("Test exception"),
            "TestComponent",
            ErrorSeverity.ERROR
        )
        
        self.assertEqual(error_info.message, "Test error message")
        self.assertEqual(error_info.component, "TestComponent")
        self.assertEqual(error_info.severity, ErrorSeverity.ERROR)

    def test_log_error_without_exception(self):
        """Тестирует логирование без исключения"""
        handler = ErrorHandler(str(self.log_file))
        
        error_info = handler.log_error(
            "Manual error",
            component="ManualComponent"
        )
        
        self.assertEqual(error_info.message, "Manual error")
        self.assertEqual(error_info.exception_type, "Manual")

    def test_get_recent_errors(self):
        """Тестирует получение последних ошибок"""
        handler = ErrorHandler(str(self.log_file))
        
        handler.log_error("Error 1", component="Test")
        handler.log_error("Error 2", component="Test")
        
        recent = handler.get_recent_errors(5)
        self.assertGreaterEqual(len(recent), 2)

    def test_get_errors_by_severity(self):
        """Тестирует получение ошибок по уровню"""
        handler = ErrorHandler(str(self.log_file))
        
        handler.log_error("Warning", severity=ErrorSeverity.WARNING)
        handler.log_error("Error", severity=ErrorSeverity.ERROR)
        
        warnings = handler.get_errors_by_severity(ErrorSeverity.WARNING)
        self.assertEqual(len(warnings), 1)

    def test_get_errors_by_component(self):
        """Тестирует получение ошибок по компоненту"""
        handler = ErrorHandler(str(self.log_file))
        
        handler.log_error("Error 1", component="ComponentA")
        handler.log_error("Error 2", component="ComponentB")
        
        errors_a = handler.get_errors_by_component("ComponentA")
        self.assertEqual(len(errors_a), 1)

    def test_clear_error_history(self):
        """Тестирует очистку истории ошибок"""
        handler = ErrorHandler(str(self.log_file))
        
        handler.log_error("Error 1", component="Test")
        handler.clear_error_history()
        
        recent = handler.get_recent_errors()
        self.assertEqual(len(recent), 0)

    def test_export_error_report(self):
        """Тестирует экспорт отчета об ошибках"""
        handler = ErrorHandler(str(self.log_file))
        
        handler.log_error("Test error", component="Test")
        
        report_path = handler.export_error_report()
        self.assertTrue(Path(report_path).exists())


class TestRecoveryManager(unittest.TestCase):
    """Тесты для класса RecoveryManager"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = Path(self.temp_dir) / "error_log.json"
        self.error_handler = ErrorHandler(str(self.log_file))
        self.recovery_mgr = RecoveryManager(self.error_handler)

    def tearDown(self):
        """Очистка после теста"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_state_backup(self):
        """Тестирует создание резервной копии состояния"""
        state_data = {"key": "value", "number": 42}
        self.recovery_mgr.create_state_backup("test_state", state_data)
        
        self.assertIn("test_state", self.recovery_mgr.state_backups)

    def test_restore_state(self):
        """Тестирует восстановление состояния"""
        state_data = {"key": "value"}
        self.recovery_mgr.create_state_backup("test_state", state_data)
        
        restored = self.recovery_mgr.restore_state("test_state")
        self.assertEqual(restored, state_data)

    def test_restore_missing_state(self):
        """Тестирует восстановление несуществующего состояния"""
        restored = self.recovery_mgr.restore_state("missing_state")
        self.assertIsNone(restored)

    def test_register_recovery_strategy(self):
        """Тестирует регистрацию стратегии восстановления"""
        def mock_strategy(error_info):
            return True
        
        self.recovery_mgr.register_recovery_strategy("ValueError", mock_strategy)
        self.assertIn("ValueError", self.recovery_mgr.recovery_strategies)


class TestSafeExecutor(unittest.TestCase):
    """Тесты для класса SafeExecutor"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = Path(self.temp_dir) / "error_log.json"
        self.error_handler = ErrorHandler(str(self.log_file))
        self.executor = SafeExecutor(self.error_handler)

    def tearDown(self):
        """Очистка после теста"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_execute_with_retry_success(self):
        """Тестирует успешное выполнение с повторами"""
        counter = {"attempts": 0}
        
        def success_func():
            counter["attempts"] += 1
            return "success"
        
        result = self.executor.execute_with_retry(success_func, max_retries=3)
        self.assertEqual(result, "success")
        self.assertEqual(counter["attempts"], 1)

    def test_execute_with_retry_failure(self):
        """Тестирует неудачное выполнение с повторами"""
        def fail_func():
            raise ValueError("Always fails")
        
        with self.assertRaises(ValueError):
            self.executor.execute_with_retry(fail_func, max_retries=2)

    def test_execute_with_timeout_success(self):
        """Тестирует успешное выполнение с таймаутом"""
        def quick_func():
            return "quick result"
        
        result = self.executor.execute_with_timeout(quick_func, timeout=5.0)
        self.assertEqual(result, "quick result")

    def test_execute_with_timeout_expired(self):
        """Тестирует истечение таймаута"""
        def slow_func():
            time.sleep(2)
            return "slow result"
        
        result = self.executor.execute_with_timeout(slow_func, timeout=0.1, fallback_return="timeout")
        self.assertEqual(result, "timeout")


if __name__ == '__main__':
    unittest.main()
