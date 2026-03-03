#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тесты для API проекта Nanoprobe Simulation Lab
"""

import unittest
import json
import sys
import os
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.validators import DataValidator, ResponseBuilder, ValidationError


class TestDataValidator(unittest.TestCase):
    """Тесты для валидатора данных."""

    def test_validate_surface_params_valid(self):
        """Тестирует валидацию корректных параметров поверхности."""
        data = {'width': 100, 'height': 100, 'type': 'random'}
        is_valid, error = DataValidator.validate_surface_params(data)
        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_validate_surface_params_default_values(self):
        """Тестирует валидацию параметров со значениями по умолчанию."""
        data = {}
        is_valid, error = DataValidator.validate_surface_params(data)
        self.assertTrue(is_valid)

    def test_validate_surface_params_invalid_width(self):
        """Тестирует валидацию некорректной ширины."""
        data = {'width': 2000, 'height': 100}
        is_valid, error = DataValidator.validate_surface_params(data)
        self.assertFalse(is_valid)
        self.assertIn('width', error)

    def test_validate_surface_params_invalid_height(self):
        """Тестирует валидацию некорректной высоты."""
        data = {'width': 100, 'height': -5}
        is_valid, error = DataValidator.validate_surface_params(data)
        self.assertFalse(is_valid)
        self.assertIn('height', error)

    def test_validate_surface_params_invalid_type(self):
        """Тестирует валидацию некорректного типа поверхности."""
        data = {'width': 100, 'height': 100, 'type': 'invalid_type'}
        is_valid, error = DataValidator.validate_surface_params(data)
        self.assertFalse(is_valid)
        self.assertIn('type', error)

    def test_validate_surface_params_none_data(self):
        """Тестирует валидацию отсутствующих данных."""
        is_valid, error = DataValidator.validate_surface_params(None)
        self.assertTrue(is_valid)  # None = используем значения по умолчанию
        self.assertIsNone(error)

    def test_validate_scan_params_valid(self):
        """Тестирует валидацию корректных параметров сканирования."""
        data = {'surface_id': 'test_surface.txt', 'scan_speed': 2.0}
        is_valid, error = DataValidator.validate_scan_params(data)
        self.assertTrue(is_valid)

    def test_validate_scan_params_missing_surface_id(self):
        """Тестирует валидацию при отсутствии surface_id."""
        data = {'scan_speed': 1.0}
        is_valid, error = DataValidator.validate_scan_params(data)
        self.assertFalse(is_valid)
        self.assertEqual(error, "surface_id обязателен")

    def test_validate_scan_params_invalid_scan_speed(self):
        """Тестирует валидацию некорректной скорости сканирования."""
        data = {'surface_id': 'test.txt', 'scan_speed': -1.0}
        is_valid, error = DataValidator.validate_scan_params(data)
        self.assertFalse(is_valid)
        self.assertIn('scan_speed', error)

    def test_validate_image_data_valid(self):
        """Тестирует валидацию корректных данных изображения."""
        data = {'image_data': 'base64string', 'filter': 'gaussian'}
        is_valid, error = DataValidator.validate_image_data(data)
        self.assertTrue(is_valid)

    def test_validate_image_data_missing_image(self):
        """Тестирует валидацию при отсутствии изображения."""
        data = {'filter': 'gaussian'}
        is_valid, error = DataValidator.validate_image_data(data)
        self.assertFalse(is_valid)
        self.assertEqual(error, "image_data обязателен")

    def test_validate_image_data_invalid_filter(self):
        """Тестирует валидацию некорректного фильтра."""
        data = {'image_data': 'base64string', 'filter': 'invalid_filter'}
        is_valid, error = DataValidator.validate_image_data(data)
        self.assertFalse(is_valid)
        self.assertIn('filter', error)

    def test_validate_audio_data_valid(self):
        """Тестирует валидацию корректных аудио данных."""
        data = {'audio_data': 'base64audio'}
        is_valid, error = DataValidator.validate_audio_data(data)
        self.assertTrue(is_valid)

    def test_validate_audio_data_with_path(self):
        """Тестирует валидацию аудио данных с путем."""
        data = {'audio_path': '/path/to/audio.wav'}
        is_valid, error = DataValidator.validate_audio_data(data)
        self.assertTrue(is_valid)

    def test_validate_audio_data_missing(self):
        """Тестирует валидацию при отсутствии аудио данных."""
        data = {}
        is_valid, error = DataValidator.validate_audio_data(data)
        self.assertFalse(is_valid)
        self.assertIn('Отсутствуют данные запроса', error)

    def test_validate_simulation_params_valid(self):
        """Тестирует валидацию корректных параметров симуляции."""
        data = {'simulation_type': 'spm', 'duration': 120}
        is_valid, error = DataValidator.validate_simulation_params(data)
        self.assertTrue(is_valid)

    def test_validate_simulation_params_missing_type(self):
        """Тестирует валидацию при отсутствии типа симуляции."""
        data = {'duration': 60}
        is_valid, error = DataValidator.validate_simulation_params(data)
        self.assertFalse(is_valid)
        self.assertEqual(error, "simulation_type обязателен")

    def test_validate_simulation_params_invalid_type(self):
        """Тестирует валидацию некорректного типа симуляции."""
        data = {'simulation_type': 'invalid', 'duration': 60}
        is_valid, error = DataValidator.validate_simulation_params(data)
        self.assertFalse(is_valid)
        self.assertIn('simulation_type', error)

    def test_validate_simulation_params_invalid_duration(self):
        """Тестирует валидацию некорректной длительности."""
        data = {'simulation_type': 'spm', 'duration': 5000}
        is_valid, error = DataValidator.validate_simulation_params(data)
        self.assertFalse(is_valid)
        self.assertIn('duration', error)


class TestResponseBuilder(unittest.TestCase):
    """Тесты для конструктора ответов API."""

    def test_success_basic(self):
        """Тестирует базовый успешный ответ."""
        response = ResponseBuilder.success()
        self.assertEqual(response['status'], 'success')
        self.assertIn('timestamp', response)

    def test_success_with_data(self):
        """Тестирует успешный ответ с данными."""
        data = {'result': 'test_data'}
        response = ResponseBuilder.success(data=data)
        self.assertEqual(response['status'], 'success')
        self.assertEqual(response['data'], data)

    def test_success_with_message(self):
        """Тестирует успешный ответ с сообщением."""
        message = "Операция выполнена"
        response = ResponseBuilder.success(message=message)
        self.assertEqual(response['status'], 'success')
        self.assertEqual(response['message'], message)

    def test_error_basic(self):
        """Тестирует базовый ответ об ошибке."""
        response = ResponseBuilder.error("Test error")
        self.assertEqual(response['status'], 'error')
        self.assertEqual(response['message'], "Test error")
        self.assertIn('timestamp', response)

    def test_error_with_code(self):
        """Тестирует ответ об ошибке с кодом."""
        response = ResponseBuilder.error("Error", "ERROR_CODE")
        self.assertEqual(response['error_code'], "ERROR_CODE")

    def test_error_with_details(self):
        """Тестирует ответ об ошибке с деталями."""
        details = {'field': 'test', 'value': 'invalid'}
        response = ResponseBuilder.error("Error", details=details)
        self.assertEqual(response['details'], details)

    def test_validation_error(self):
        """Тестирует ответ об ошибке валидации."""
        response = ResponseBuilder.validation_error("test_field", "Invalid value")
        self.assertEqual(response['status'], 'error')
        self.assertEqual(response['error_code'], "VALIDATION_ERROR")
        self.assertIn('test_field', response['message'])
        self.assertIn('Invalid value', response['message'])
        self.assertEqual(response['details']['field'], 'test_field')


class TestValidationError(unittest.TestCase):
    """Тесты для исключения валидации."""

    def test_validation_error_basic(self):
        """Тестирует базовое исключение валидации."""
        exc = ValidationError("Test error")
        self.assertEqual(exc.message, "Test error")
        self.assertIsNone(exc.field)

    def test_validation_error_with_field(self):
        """Тестирует исключение валидации с полем."""
        exc = ValidationError("Invalid value", "test_field")
        self.assertEqual(exc.message, "Invalid value")
        self.assertEqual(exc.field, "test_field")


def run_tests():
    """Запускает все тесты."""
    print("=" * 60)
    print("ЗАПУСК ТЕСТОВ ДЛЯ API")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 60)
    print(f"РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print(f"  Пройдено: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Ошибки: {len(result.errors)}")
    print(f"  Провалы: {len(result.failures)}")
    print("=" * 60)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
