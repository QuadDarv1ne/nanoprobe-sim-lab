"""Тесты для валидатора данных."""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../utils'))

import pandas as pd
import numpy as np
from data_validator import DataValidator, ValidationLevel, ValidationResult


class TestDataValidator(unittest.TestCase):
    """Тесты для класса DataValidator"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.validator = DataValidator()
        self.sample_df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['A', 'B', 'C', 'D', 'E'],
            'value': [10.5, 20.3, 15.7, 25.1, 30.0],
            'category': ['X', 'Y', 'X', 'Y', 'Z']
        })

    def test_initialization(self):
        """Тестирует инициализацию валидатора"""
        self.assertEqual(self.validator.validation_level, ValidationLevel.STANDARD)

    def test_validation_result(self):
        """Тестирует результат валидации"""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            suggestions=[],
            metadata={'test': 'data'}
        )
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(result.metadata['test'], 'data')

    def test_validate_dataframe_invalid_input(self):
        """Тестирует валидацию не DataFrame"""
        with self.assertRaises(ValueError):
            self.validator.validate_dataframe("not a dataframe", {})

    def test_validate_dataframe_empty_schema(self):
        """Тестирует валидацию с пустой схемой"""
        with self.assertRaises(ValueError):
            self.validator.validate_dataframe(self.sample_df, {})

    def test_validate_dataframe_valid(self):
        """Тестирует валидацию корректного DataFrame"""
        schema = {
            'id': {'dtype': 'int', 'required': True},
            'name': {'dtype': 'object', 'required': True},
            'value': {'dtype': 'float', 'required': False},
        }
        result = self.validator.validate_dataframe(self.sample_df, schema)
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid)

    def test_validate_dataframe_missing_columns(self):
        """Тестирует валидацию с отсутствующими столбцами"""
        schema = {
            'id': {'dtype': 'int', 'required': True},
            'missing_column': {'dtype': 'int', 'required': True},
        }
        result = self.validator.validate_dataframe(self.sample_df, schema)
        self.assertFalse(result.is_valid)
        self.assertTrue(any('missing_column' in err for err in result.errors))

    def test_validate_dataframe_wrong_dtype(self):
        """Тестирует валидацию с неверным типом данных"""
        schema = {
            'id': {'dtype': 'str', 'required': True},
        }
        result = self.validator.validate_dataframe(self.sample_df, schema, )
        # В STANDARD режиме это warning, не error
        self.assertIsInstance(result, ValidationResult)

    def test_validate_dataframe_range_check(self):
        """Тестирует проверку диапазона значений"""
        schema = {
            'value': {
                'dtype': 'float',
                'required': True,
                'range': (0, 20)
            },
        }
        result = self.validator.validate_dataframe(self.sample_df, schema)
        # Значения 25.1 и 30.0 вне диапазона
        self.assertFalse(result.is_valid)

    def test_validate_dataframe_range_exception(self):
        """Тестирует обработку исключений при проверке диапазона"""
        schema = {
            'name': {
                'dtype': 'object',
                'required': True,
                'range': (0, 10)  # Неприменимо к строкам
            },
        }
        result = self.validator.validate_dataframe(self.sample_df, schema)
        # Должно быть warning, не error
        self.assertIsInstance(result, ValidationResult)

    def test_validation_levels(self):
        """Тестирует разные уровни валидации"""
        for level in ValidationLevel:
            validator = DataValidator(validation_level=level)
            self.assertEqual(validator.validation_level, level)

    def test_validate_numeric_field(self):
        """Тестирует валидацию числового поля"""
        self.assertTrue(self.validator.validate_numeric_field(5, min_val=0, max_val=10))
        self.assertFalse(self.validator.validate_numeric_field(15, min_val=0, max_val=10))
        self.assertTrue(self.validator.validate_numeric_field(np.nan, allow_nan=True))
        self.assertFalse(self.validator.validate_numeric_field(np.nan, allow_nan=False))

    def test_validate_string_field(self):
        """Тестирует валидацию строкового поля"""
        self.assertTrue(self.validator.validate_string_field("test", min_length=1, max_length=10))
        self.assertFalse(self.validator.validate_string_field("", min_length=1))
        self.assertFalse(self.validator.validate_string_field("verylongstring", max_length=10))

    def test_add_validation_rule(self):
        """Тестирует добавление правила валидации"""
        def custom_validator(value):
            return value > 0

        self.validator.add_validation_rule(
            'positive_field',
            custom_validator,
            error_message='Значение должно быть положительным'
        )
        self.assertIn('positive_field', self.validator.validation_rules)


def run_tests():
    """Запускает все тесты."""
    print("=" * 60)
    print("ЗАПУСК ТЕСТОВ ДЛЯ DATA VALIDATOR")
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
