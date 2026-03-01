# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Модуль валидации данных для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для проверки,
валидации и обеспечения качества данных проекта.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime
import json
import hashlib
import re
from dataclasses import dataclass
import warnings
from enum import Enum
import logging
from functools import wraps

class ValidationLevel(Enum):
    """Уровни валидации"""
    BASIC = 1
    STANDARD = 2
    STRICT = 3
    COMPREHENSIVE = 4

@dataclass
class ValidationResult:
    """Результат валидации"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    metadata: Dict[str, Any]

class DataValidator:
    """
    Класс валидатора данных
    Обеспечивает проверку, валидацию и
    обеспечение качества данных проекта.
    """


    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        """
        Инициализирует валидатор данных

        Args:
            validation_level: Уровень строгости валидации
        """
        self.validation_level = validation_level
        self.validation_rules = {}
        self.custom_validators = {}
        self.logger = logging.getLogger(__name__)


    def add_validation_rule(self, field_name: str, validator_func: Callable,

                          error_message: str = None, warning: bool = False):
        """
        Добавляет правило валидации

        Args:
            field_name: Имя поля для валидации
            validator_func: Функция валидации
            error_message: Сообщение об ошибке
            warning: Является ли предупреждением вместо ошибки
        """
        if field_name not in self.validation_rules:
            self.validation_rules[field_name] = []

        self.validation_rules[field_name].append({
            'validator': validator_func,
            'error_message': error_message or f"Неверное значение для поля {field_name}",
            'warning': warning
        })



    def validate_numeric_field(self, value: Any, min_val: float = None,
                             max_val: float = None, allow_nan: bool = True) -> bool:
        """
        Валидирует числовое поле

        Args:
            value: Значение для валидации
            min_val: Минимальное значение
            max_val: Максимальное значение
            allow_nan: Разрешать ли NaN значения

        Returns:
            True если значение валидно, иначе False
        """
        try:
            numeric_val = float(value)

            if not allow_nan and np.isnan(numeric_val):
                return False

            if min_val is not None and numeric_val < min_val:
                return False

            if max_val is not None and numeric_val > max_val:
                return False

            return True

        except (ValueError, TypeError):
            return False


    def validate_string_field(self, value: Any, min_length: int = 1,
                            max_length: int = None, pattern: str = None,
                            allowed_values: List[str] = None) -> bool:
        """
        Валидирует строковое поле

        Args:
            value: Значение для валидации
            min_length: Минимальная длина
            max_length: Максимальная длина
            pattern: Регулярное выражение для проверки
            allowed_values: Список разрешенных значений

        Returns:
            True если значение валидно, иначе False
        """
        if not isinstance(value, str):
            try:
                value = str(value)
            except:
                return False

        if len(value) < min_length:
            return False

        if max_length and len(value) > max_length:
            return False

        if pattern and not re.match(pattern, value):
            return False

        if allowed_values and value not in allowed_values:
            return False


        return True


    def validate_array_field(self, arr: Any, min_length: int = 0,
                           max_length: int = None, element_validator: Callable = None,
                           allow_empty: bool = True) -> bool:
        """
        Валидирует массив

        Args:
            arr: Массив для валидации
            min_length: Минимальная длина массива
            max_length: Максимальная длина массива
            element_validator: Валидатор элементов массива
            allow_empty: Разрешать ли пустые массивы

        Returns:
            True если массив валидный, иначе False
        """
        if not isinstance(arr, (list, tuple, np.ndarray)):
            return False

        if not allow_empty and len(arr) == 0:
            return False

        if len(arr) < min_length:
            return False

        if max_length and len(arr) > max_length:
            return False

        if element_validator:
            for element in arr:
                if not element_validator(element):
                    return False

        return True


    def validate_dataframe(self, df: pd.DataFrame, schema: Dict[str, Dict[str, Any]]) -> ValidationResult:
        """
        Валидирует DataFrame согласно схеме

        Args:
            df: DataFrame для валидации
            schema: Схема валидации

        Returns:
            Результат валидации
        """
        errors = []
        warnings_list = []
        suggestions = []

        # Проверяем наличие обязательных столбцов
        required_columns = [col for col, props in schema.items() if props.get('required', False)]
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"Отсутствуют обязательные столбцы: {missing_columns}")

        # Проверяем типы данных
        for column, props in schema.items():
            if column in df.columns:
                expected_dtype = props.get('dtype')
                if expected_dtype:
                    actual_dtype = str(df[column].dtype)
                    if expected_dtype not in actual_dtype:
                        if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.COMPREHENSIVE]:
                            errors.append(f"Неверный тип данных для столбца '{column}': ожидается {expected_dtype}, получено {actual_dtype}")
                        else:
                            warnings_list.append(f"Потенциально неверный тип данных для столбца '{column}': ожидается {expected_dtype}, получено {actual_dtype}")

                # Проверяем диапазон значений для числовых столбцов
                if expected_dtype in ['int', 'float', 'double'] and props.get('range'):
                    min_val, max_val = props['range']
                    invalid_values = df[(df[column] < min_val) | (df[column] > max_val)][column]
                    if not invalid_values.empty:
                        errors.append(f"Найдены значения вне диапазона [{min_val}, {max_val}] в столбце '{column}': {invalid_values.tolist()}")

                # Проверяем уникальность
                if props.get('unique', False):
                    duplicates = df[df.duplicated(subset=[column])]
                    if not duplicates.empty:
                        warnings_list.append(f"Найдены дубликаты в столбце '{column}'")

                # Проверяем наличие null значений
                null_count = df[column].isnull().sum()
                if null_count > 0 and not props.get('nullable', True):
                    if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.COMPREHENSIVE]:
                        errors.append(f"Найдены null значения в столбце '{column}': {null_count}")
                    else:
                        warnings_list.append(f"Найдены null значения в столбце '{column}': {null_count}")

        # Проверяем целостность данных
        if self.validation_level in [ValidationLevel.COMPREHENSIVE]:
            # Проверяем дубликаты строк
            duplicate_rows = df.duplicated().sum()
            if duplicate_rows > 0:
                warnings_list.append(f"Найдено {duplicate_rows} дублирующихся строк")

            # Проверяем пустые строки
            empty_rows = df.isnull().all(axis=1).sum()
            if empty_rows > 0:
                warnings_list.append(f"Найдено {empty_rows} полностью пустых строк")

        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings_list,
            suggestions=suggestions,
            metadata={
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'validation_level': self.validation_level.name

            }
        )


    def validate_numpy_array(self, arr: np.ndarray, shape: tuple = None,
                           dtype: str = None, min_val: float = None,
                           max_val: float = None, allow_nan: bool = True) -> ValidationResult:
        """
        Валидирует numpy массив

        Args:
            arr: Массив для валидации
            shape: Ожидаемая форма массива
            dtype: Ожидаемый тип данных
            min_val: Минимальное значение
            max_val: Максимальное значение
            allow_nan: Разрешать ли NaN значения

        Returns:
            Результат валидации
        """
        errors = []
        warnings_list = []
        suggestions = []

        # Проверяем форму массива
        if shape and arr.shape != shape:
            errors.append(f"Неверная форма массива: ожидается {shape}, получено {arr.shape}")

        # Проверяем тип данных
        if dtype and str(arr.dtype) != dtype:
            if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.COMPREHENSIVE]:
                errors.append(f"Неверный тип данных: ожидается {dtype}, получено {arr.dtype}")
            else:
                warnings_list.append(f"Потенциально неверный тип данных: ожидается {dtype}, получено {arr.dtype}")

        # Проверяем значения
        if not allow_nan and np.isnan(arr).any():
            errors.append("Найдены NaN значения в массиве")

        if min_val is not None:
            min_found = np.min(arr)
            if min_found < min_val:
                errors.append(f"Найдены значения меньше минимального ({min_val}): {min_found}")

        if max_val is not None:
            max_found = np.max(arr)
            if max_found > max_val:
                errors.append(f"Найдены значения больше максимального ({max_val}): {max_found}")

        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings_list,
            suggestions=suggestions,
            metadata={
                'shape': arr.shape,
                'dtype': str(arr.dtype),
                'size': arr.size,
                'validation_level': self.validation_level.name
            }
        )


    def calculate_data_quality_score(self, data: Union[pd.DataFrame, np.ndarray, Dict]) -> Dict[str, float]:
        """
        Рассчитывает оценку качества данных

        Args:
            data: Данные для оценки

        Returns:
            Словарь с метриками качества данных
        """
        if isinstance(data, pd.DataFrame):
            total_cells = data.size
            null_cells = data.isnull().sum().sum()
            duplicate_rows = data.duplicated().sum()
            numeric_cols = data.select_dtypes(include=[np.number]).columns

            completeness = (total_cells - null_cells) / total_cells if total_cells > 0 else 1.0
            uniqueness = (len(data) - duplicate_rows) / len(data) if len(data) > 0 else 1.0

            # Для числовых данных проверяем диапазон значений
            if len(numeric_cols) > 0:
                outliers_count = 0
                for col in numeric_cols:
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers_count += ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()

                validity = 1.0 - (outliers_count / total_cells) if total_cells > 0 else 1.0
            else:
                validity = 1.0

        elif isinstance(data, np.ndarray):
            total_elements = data.size
            null_elements = np.isnan(data).sum() if np.issubdtype(data.dtype, np.number) else 0
            completeness = (total_elements - null_elements) / total_elements if total_elements > 0 else 1.0
            uniqueness = len(np.unique(data)) / total_elements if total_elements > 0 else 1.0
            validity = 1.0  # Для массивов не проверяем валидность как для DF
        else:
            # Для словарей или других типов
            total_items = len(data) if hasattr(data, '__len__') else 1
            completeness = 1.0 if total_items > 0 else 0.0
            uniqueness = 1.0
            validity = 1.0

        overall_score = (completeness + uniqueness + validity) / 3.0

        return {
            'completeness': completeness,
            'uniqueness': uniqueness,
            'validity': validity,

            'overall_score': overall_score,
            'total_items': total_items if 'total_items' in locals() else 1
        }


    def generate_data_report(self, data: Union[pd.DataFrame, np.ndarray],
                           output_path: str = None) -> str:
        """
        Генерирует отчет о данных

        Args:
            data: Данные для анализа
            output_path: Путь для сохранения отчета (если None, генерируется автоматически)

        Returns:
            Путь к созданному отчету
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data_report_{timestamp}.json"

        if isinstance(data, pd.DataFrame):
            report = {
                'timestamp': datetime.now().isoformat(),
                'data_type': 'DataFrame',
                'shape': data.shape,
                'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
                'descriptive_stats': data.describe().to_dict() if len(data.select_dtypes(include=[np.number])) > 0 else {},
                'null_counts': data.isnull().sum().to_dict(),
                'duplicates_count': int(data.duplicated().sum()),
                'quality_metrics': self.calculate_data_quality_score(data),
                'memory_usage': int(data.memory_usage(deep=True).sum())
            }
        elif isinstance(data, np.ndarray):
            report = {
                'timestamp': datetime.now().isoformat(),
                'data_type': 'numpy_array',
                'shape': data.shape,
                'dtype': str(data.dtype),
                'size': int(data.size),
                'itemsize': int(data.itemsize),
                'nbytes': int(data.nbytes),
                'quality_metrics': self.calculate_data_quality_score(data),
                'stats': {
                    'mean': float(np.mean(data)) if np.issubdtype(data.dtype, np.number) else None,
                    'std': float(np.std(data)) if np.issubdtype(data.dtype, np.number) else None,
                    'min': float(np.min(data)) if np.issubdtype(data.dtype, np.number) else None,
                    'max': float(np.max(data)) if np.issubdtype(data.dtype, np.number) else None
                } if np.issubdtype(data.dtype, np.number) else {}
            }
        else:
            report = {
                'timestamp': datetime.now().isoformat(),
                'data_type': str(type(data)),
                'quality_metrics': self.calculate_data_quality_score(data)
            }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        return output_path


    def validate_file_integrity(self, file_path: str, expected_hash: str = None) -> ValidationResult:
        """
        Проверяет целостность файла

        Args:
            file_path: Путь к файлу
            expected_hash: Ожидаемый хеш файла

        Returns:
            Результат валидации
        """
        errors = []
        warnings_list = []
        suggestions = []

        try:
            file_path = Path(file_path)

            if not file_path.exists():
                errors.append(f"Файл не существует: {file_path}")
                return ValidationResult(False, errors, warnings_list, suggestions, {})

            # Проверяем размер файла
            file_size = file_path.stat().st_size
            if file_size == 0:
                warnings_list.append("Файл пустой")

            # Вычисляем хеш файла
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            # Сравниваем с ожидаемым хешем
            if expected_hash and file_hash != expected_hash:
                errors.append(f"Хеш файла не совпадает: ожидается {expected_hash[:8]}..., получено {file_hash[:8]}...")

            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings_list,
                suggestions=suggestions,
                metadata={
                    'file_path': str(file_path),
                    'file_size': file_size,
                    'calculated_hash': file_hash,
                    'expected_hash': expected_hash
                }
            )

        except Exception as e:
            errors.append(f"Ошибка при проверке целостности файла: {str(e)}")
            return ValidationResult(False, errors, warnings_list, suggestions, {})


    def validate_json_schema(self, data: Dict, schema: Dict) -> ValidationResult:
        """
        Валидирует JSON данные по схеме

        Args:
            data: JSON данные
            schema: Схема валидации

        Returns:
            Результат валидации

        """
        errors = []
        warnings_list = []
        suggestions = []

        def validate_recursive(data_item, schema_item, path=""):
                    current_path = path

            # Проверяем тип
            if 'type' in schema_item:
                expected_type = schema_item['type']
                actual_type = type(data_item).__name__

                if expected_type == 'array' and isinstance(data_item, (list, tuple)):
                    pass  # Массивы допустимы
                elif expected_type == 'object' and isinstance(data_item, dict):
                    pass  # Объекты допустимы
                elif expected_type != actual_type:
                    errors.append(f"Неверный тип по пути {current_path}: ожидается {expected_type}, получено {actual_type}")
                    return

            # Проверяем обязательные поля
            if 'required' in schema_item and schema_item['required']:
                if isinstance(data_item, dict):
                    for req_field in schema_item['required']:
                        if req_field not in data_item:
                            errors.append(f"Отсутствует обязательное поле: {current_path}.{req_field}")

            # Проверяем значения
            if isinstance(data_item, dict) and isinstance(schema_item, dict):
                properties = schema_item.get('properties', {})
                for key, value in data_item.items():
                    if key in properties:
                        validate_recursive(value, properties[key], f"{current_path}.{key}")
                    elif 'additionalProperties' in schema_item and not schema_item['additionalProperties']:
                        warnings_list.append(f"Дополнительное поле не разрешено: {current_path}.{key}")

        validate_recursive(data, schema, "")

        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings_list,
            suggestions=suggestions,
            metadata={
                'validation_type': 'json_schema',
                'validation_level': self.validation_level.name
            }
        )

def validate_data(validation_level: ValidationLevel = ValidationLevel.STANDARD):

    """

    Декоратор для валидации данных

    Args:
        validation_level: Уровень строгости валидации
    """

    def decorator(func):
            @wraps(func)
        def wrapper(*args, **kwargs):
                    validator = DataValidator(validation_level)

            # Здесь мы могли бы добавить логику проверки входных данных
            # в зависимости от сигнатуры функции

            result = func(*args, **kwargs)

            # Валидируем результат если он является структурой данных
            if isinstance(result, (pd.DataFrame, np.ndarray, dict)):
                if isinstance(result, pd.DataFrame):
                    validation_result = validator.validate_dataframe(
                        result,
                        {}  # Пустая схема для базовой валидации
                    )
                elif isinstance(result, np.ndarray):
                    validation_result = validator.validate_numpy_array(result)
                else:
                    # Для словарей пока просто проверяем целостность
                    validation_result = ValidationResult(
                        is_valid=True,
                        errors=[],
                        warnings=[],
                        suggestions=[],
                        metadata={}
                    )

                if not validation_result.is_valid:
                    warnings.warn(f"Валидация данных не прошла: {validation_result.errors}")

            return result
        return wrapper
    return decorator

def main():
    """Главная функция для демонстрации возможностей валидатора данных"""
    print("=== ВАЛИДАТОР ДАННЫХ ПРОЕКТА ===")

    # Создаем валидатор данных
    validator = DataValidator()

    print("✓ Валидатор данных инициализирован")
    print(f"✓ Уровень валидации: {validator.validation_level.name}")

    # Создаем тестовые данные
    test_data = {
        'temperature': [20.5, 21.0, 19.8, 22.1, 18.9],
        'pressure': [1013.25, 1012.80, 1014.10, 1011.90, 1015.05],
        'timestamp': pd.date_range('2023-01-01', periods=5, freq='H'),
        'status': ['OK', 'OK', 'WARNING', 'OK', 'ERROR']
    }

    df = pd.DataFrame(test_data)
    print(f"\nСоздан DataFrame с {len(df)} строками и {len(df.columns)} столбцами")

    # Валидируем DataFrame
    schema = {
        'temperature': {
            'dtype': 'float',
            'range': (-50, 50),
            'required': True
        },
        'pressure': {
            'dtype': 'float',
            'range': (900, 1100),
            'required': True
        },
        'status': {
            'dtype': 'object',
            'required': True,
            'nullable': False
        }
    }

    print("\nВалидация DataFrame по схеме...")
    df_result = validator.validate_dataframe(df, schema)
    print(f"  - Валидация пройдена: {df_result.is_valid}")
    print(f"  - Ошибок: {len(df_result.errors)}")
    print(f"  - Предупреждений: {len(df_result.warnings)}")

    if df_result.errors:
        print("  - Ошибки:")
        for error in df_result.errors:
            print(f"    * {error}")

    if df_result.warnings:
        print("  - Предупреждения:")
        for warning in df_result.warnings:
            print(f"    * {warning}")

    # Валидируем numpy массив
    print("\nВалидация numpy массива...")
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    arr_result = validator.validate_numpy_array(arr, shape=(2, 3), dtype='float64', min_val=0, max_val=10)
    print(f"  - Валидация пройдена: {arr_result.is_valid}")
    print(f"  - Ошибок: {len(arr_result.errors)}")

    # Рассчитываем оценку качества данных
    print("\nРасчет оценки качества данных...")
    quality_score = validator.calculate_data_quality_score(df)
    print(f"  - Полнота: {quality_score['completeness']:.2f}")
    print(f"  - Уникальность: {quality_score['uniqueness']:.2f}")
    print(f"  - Валидность: {quality_score['validity']:.2f}")
    print(f"  - Общая оценка: {quality_score['overall_score']:.2f}")

    # Генерируем отчет о данных
    print("\nГенерация отчета о данных...")
    report_path = validator.generate_data_report(df)
    print(f"  - Отчет сохранен: {report_path}")

    # Проверяем целостность файла

    print("\nПроверка целостности файла отчета...")
    file_result = validator.validate_file_integrity(report_path)
    print(f"  - Файл целостен: {file_result.is_valid}")

    # Демонстрируем декоратор валидации
    print("\nДемонстрация декоратора валидации...")

    @validate_data(ValidationLevel.STANDARD)

    def sample_data_processing():
            return pd.DataFrame({'value': [1, 2, 3]})

    result = sample_data_processing()
    print(f"  - Функция с декоратором выполнена успешно: {type(result).__name__}")

    print("\nВалидатор данных успешно протестирован")
    print("\nДоступные функции:")
    print("- Валидация DataFrame: validate_dataframe()")
    print("- Валидация numpy массива: validate_numpy_array()")
    print("- Расчет оценки качества: calculate_data_quality_score()")
    print("- Генерация отчета: generate_data_report()")
    print("- Проверка целостности файла: validate_file_integrity()")
    print("- Валидация JSON схемы: validate_json_schema()")
    print("- Декоратор валидации: @validate_data")

if __name__ == "__main__":
    main()

