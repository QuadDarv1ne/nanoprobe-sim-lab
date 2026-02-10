#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль валидации конфигурации для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для проверки 
и валидации конфигурационных файлов проекта.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import re
import jsonschema
from jsonschema import Draft7Validator
import inspect


class ConfigValidator:
    """
    Класс валидации конфигурации
    Обеспечивает проверку корректности конфигурационных 
    файлов и параметров проекта.
    """
    
    def __init__(self):
        """Инициализирует валидатор конфигурации"""
        self.validation_results = {}
        self.errors = []
        self.warnings = []
    
    def validate_json_config(self, config_path: str, schema: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Валидирует JSON конфигурационный файл
        
        Args:
            config_path: Путь к конфигурационному файлу
            schema: JSON схема для валидации (если None, используется схема по умолчанию)
            
        Returns:
            Словарь с результатами валидации
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Используем схему по умолчанию, если не предоставлена
            if schema is None:
                schema = self.get_default_config_schema()
            
            # Создаем валидатор
            validator = Draft7Validator(schema)
            
            # Проверяем конфигурацию
            errors = list(validator.iter_errors(config))
            valid = len(errors) == 0
            
            result = {
                'valid': valid,
                'config_path': config_path,
                'errors': [str(error) for error in errors],
                'config': config,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except json.JSONDecodeError as e:
            return {
                'valid': False,
                'config_path': config_path,
                'errors': [f"JSON decode error: {str(e)}"],
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'valid': False,
                'config_path': config_path,
                'errors': [f"Validation error: {str(e)}"],
                'timestamp': datetime.now().isoformat()
            }
    
    def validate_yaml_config(self, config_path: str, schema: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Валидирует YAML конфигурационный файл
        
        Args:
            config_path: Путь к конфигурационному файлу
            schema: JSON схема для валидации (если None, используется схема по умолчанию)
            
        Returns:
            Словарь с результатами валидации
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Используем схему по умолчанию, если не предоставлена
            if schema is None:
                schema = self.get_default_config_schema()
            
            # Создаем валидатор
            validator = Draft7Validator(schema)
            
            # Проверяем конфигурацию
            errors = list(validator.iter_errors(config))
            valid = len(errors) == 0
            
            result = {
                'valid': valid,
                'config_path': config_path,
                'errors': [str(error) for error in errors],
                'config': config,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except yaml.YAMLError as e:
            return {
                'valid': False,
                'config_path': config_path,
                'errors': [f"YAML parse error: {str(e)}"],
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'valid': False,
                'config_path': config_path,
                'errors': [f"Validation error: {str(e)}"],
                'timestamp': datetime.now().isoformat()
            }
    
    def get_default_config_schema(self) -> Dict[str, Any]:
        """
        Возвращает схему по умолчанию для валидации конфигурации проекта
        
        Returns:
            Словарь с JSON схемой
        """
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "project_name": {
                    "type": "string",
                    "description": "Название проекта"
                },
                "version": {
                    "type": "string",
                    "pattern": r"^\d+\.\d+\.\d+$",
                    "description": "Версия проекта в формате X.Y.Z"
                },
                "description": {
                    "type": "string",
                    "description": "Описание проекта"
                },
                "authors": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Список авторов"
                },
                "license": {
                    "type": "string",
                    "description": "Лицензия проекта"
                },
                "dependencies": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "string"
                    },
                    "description": "Зависимости проекта"
                },
                "paths": {
                    "type": "object",
                    "properties": {
                        "data_dir": {"type": "string"},
                        "output_dir": {"type": "string"},
                        "backup_dir": {"type": "string"},
                        "log_dir": {"type": "string"}
                    },
                    "additionalProperties": True
                },
                "simulation_settings": {
                    "type": "object",
                    "properties": {
                        "precision": {
                            "type": "string",
                            "enum": ["low", "medium", "high", "ultra"]
                        },
                        "real_time": {"type": "boolean"},
                        "max_threads": {"type": "integer", "minimum": 1}
                    },
                    "additionalProperties": True
                },
                "security": {
                    "type": "object",
                    "properties": {
                        "encryption_enabled": {"type": "boolean"},
                        "auth_required": {"type": "boolean"},
                        "api_rate_limit": {"type": "integer", "minimum": 1}
                    },
                    "additionalProperties": True
                }
            },
            "required": ["project_name", "version", "description"],
            "additionalProperties": True
        }
    
    def validate_config_against_schema(self, config: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Валидирует конфигурацию по заданной схеме
        
        Args:
            config: Конфигурация для валидации
            schema: Схема валидации
            
        Returns:
            Словарь с результатами валидации
        """
        validator = Draft7Validator(schema)
        errors = list(validator.iter_errors(config))
        valid = len(errors) == 0
        
        return {
            'valid': valid,
            'errors': [str(error) for error in errors],
            'config': config,
            'schema': schema,
            'timestamp': datetime.now().isoformat()
        }
    
    def validate_project_structure(self, project_root: str = ".") -> Dict[str, Any]:
        """
        Валидирует структуру проекта
        
        Args:
            project_root: Корневая директория проекта
            
        Returns:
            Словарь с результатами валидации
        """
        project_path = Path(project_root)
        
        # Ожидаемые директории
        expected_dirs = [
            "cpp-spm-hardware-sim",
            "py-surface-image-analyzer", 
            "py-sstv-groundstation",
            "utils",
            "api",
            "security",
            "tests",
            "docs",
            "backups"
        ]
        
        # Ожидаемые файлы
        expected_files = [
            "config.json",
            "requirements.txt",
            "CMakeLists.txt",
            "README.md",
            "LICENCE"
        ]
        
        results = {
            'project_root': str(project_path.absolute()),
            'directories': {
                'found': [],
                'missing': [],
                'unexpected': []
            },
            'files': {
                'found': [],
                'missing': [],
                'unexpected': []
            },
            'valid': True,
            'timestamp': datetime.now().isoformat()
        }
        
        # Проверяем директории
        actual_dirs = [item.name for item in project_path.iterdir() if item.is_dir()]
        results['directories']['found'] = [d for d in expected_dirs if d in actual_dirs]
        results['directories']['missing'] = [d for d in expected_dirs if d not in actual_dirs]
        results['directories']['unexpected'] = [d for d in actual_dirs if d not in expected_dirs and not d.startswith('.')]
        
        # Проверяем файлы
        actual_files = [item.name for item in project_path.iterdir() if item.is_file()]
        results['files']['found'] = [f for f in expected_files if f in actual_files]
        results['files']['missing'] = [f for f in expected_files if f not in actual_files]
        results['files']['unexpected'] = [f for f in actual_files if f not in expected_files and not f.startswith('.')]
        
        # Определяем валидность
        results['valid'] = len(results['directories']['missing']) == 0 and len(results['files']['missing']) == 0
        
        return results
    
    def validate_dependencies(self, requirements_path: str = "requirements.txt") -> Dict[str, Any]:
        """
        Валидирует зависимости проекта
        
        Args:
            requirements_path: Путь к файлу зависимостей
            
        Returns:
            Словарь с результатами валидации
        """
        try:
            req_path = Path(requirements_path)
            if not req_path.exists():
                return {
                    'valid': False,
                    'errors': [f"Файл зависимостей не найден: {requirements_path}"],
                    'timestamp': datetime.now().isoformat()
                }
            
            with open(req_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            valid_deps = []
            invalid_deps = []
            warnings = []
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Проверяем формат зависимости
                if '==' in line or '>=' in line or '<=' in line or '>' in line or '<' in line:
                    # Проверяем формат имени пакета
                    package_name = line.split('==')[0].split('>')[0].split('<')[0].strip()
                    if re.match(r'^[a-zA-Z0-9_-]+$', package_name):
                        valid_deps.append(line)
                    else:
                        invalid_deps.append({
                            'line': line_num,
                            'dependency': line,
                            'error': 'Invalid package name format'
                        })
                else:
                    warnings.append({
                        'line': line_num,
                        'dependency': line,
                        'warning': 'Version specification missing'
                    })
            
            result = {
                'valid': len(invalid_deps) == 0,
                'valid_dependencies': valid_deps,
                'invalid_dependencies': invalid_deps,
                'warnings': warnings,
                'total_dependencies': len(valid_deps) + len(invalid_deps),
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Dependency validation error: {str(e)}"],
                'timestamp': datetime.now().isoformat()
            }
    
    def generate_validation_report(self, validation_results: Dict[str, Any], output_path: str = None) -> str:
        """
        Генерирует отчет о валидации
        
        Args:
            validation_results: Результаты валидации
            output_path: Путь для сохранения отчета (если None, генерируется автоматически)
            
        Returns:
            Путь к созданному отчету
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"config_validation_report_{timestamp}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'report_type': 'config_validation',
            'validation_results': validation_results,
            'summary': self._generate_validation_summary(validation_results)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        return output_path
    
    def _generate_validation_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Генерирует сводку по результатам валидации
        
        Args:
            validation_results: Результаты валидации
            
        Returns:
            Словарь со сводкой
        """
        summary = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'overall_status': 'PASS',
            'critical_issues': [],
            'warnings': []
        }
        
        # Подсчитываем результаты
        if isinstance(validation_results, dict):
            if 'valid' in validation_results:
                summary['total_validations'] = 1
                if validation_results['valid']:
                    summary['passed_validations'] = 1
                else:
                    summary['failed_validations'] = 1
                    if 'errors' in validation_results:
                        summary['critical_issues'].extend(validation_results['errors'])
            elif 'directories' in validation_results and 'files' in validation_results:
                # Это результат валидации структуры проекта
                summary['total_validations'] = 1
                if validation_results['valid']:
                    summary['passed_validations'] = 1
                else:
                    summary['failed_validations'] = 1
                    missing_dirs = validation_results.get('directories', {}).get('missing', [])
                    missing_files = validation_results.get('files', {}).get('missing', [])
                    summary['critical_issues'].extend([
                        f"Missing directory: {d}" for d in missing_dirs
                    ])
                    summary['critical_issues'].extend([
                        f"Missing file: {f}" for f in missing_files
                    ])
        
        # Определяем общий статус
        if summary['failed_validations'] > 0:
            summary['overall_status'] = 'FAIL'
        
        return summary


class OptimizationAdvisor:
    """
    Класс советника по оптимизации
    Предоставляет рекомендации по оптимизации 
    производительности и ресурсов проекта.
    """
    
    def __init__(self):
        """Инициализирует советника по оптимизации"""
        self.advice_history = []
    
    def analyze_performance_data(self, performance_metrics: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Анализирует данные о производительности и дает рекомендации
        
        Args:
            performance_metrics: Метрики производительности
            
        Returns:
            Список рекомендаций по оптимизации
        """
        recommendations = []
        
        # Анализ загрузки CPU
        avg_cpu = performance_metrics.get('avg_cpu_percent', 0)
        if avg_cpu > 80:
            recommendations.append({
                'category': 'cpu',
                'severity': 'high',
                'recommendation': 'Рассмотрите оптимизацию алгоритмов для снижения загрузки CPU',
                'details': f'Средняя загрузка CPU {avg_cpu}% превышает рекомендуемый порог 80%'
            })
        elif avg_cpu > 60:
            recommendations.append({
                'category': 'cpu',
                'severity': 'medium',
                'recommendation': 'Рассмотрите оптимизацию вычислительно интенсивных операций',
                'details': f'Средняя загрузка CPU {avg_cpu}% находится в зоне внимания'
            })
        
        # Анализ использования памяти
        avg_memory = performance_metrics.get('avg_memory_percent', 0)
        if avg_memory > 80:
            recommendations.append({
                'category': 'memory',
                'severity': 'high',
                'recommendation': 'Рассмотрите оптимизацию использования памяти и управление объектами',
                'details': f'Среднее использование памяти {avg_memory}% превышает рекомендуемый порог 80%'
            })
        elif avg_memory > 60:
            recommendations.append({
                'category': 'memory',
                'severity': 'medium',
                'recommendation': 'Рассмотрите оптимизацию использования памяти',
                'details': f'Среднее использование памяти {avg_memory}% находится в зоне внимания'
            })
        
        # Анализ времени выполнения
        execution_time = performance_metrics.get('execution_time_sec', 0)
        if execution_time > 60:  # Более 1 минуты
            recommendations.append({
                'category': 'performance',
                'severity': 'medium',
                'recommendation': 'Рассмотрите возможность параллелизации или оптимизации алгоритмов',
                'details': f'Время выполнения {execution_time} секунд превышает рекомендуемый порог 60 секунд'
            })
        
        # Анализ дискового I/O
        avg_disk_write_rate = performance_metrics.get('avg_disk_write_rate_bps', 0)
        if avg_disk_write_rate > 100 * 1024 * 1024:  # Более 100 MB/s
            recommendations.append({
                'category': 'disk_io',
                'severity': 'medium',
                'recommendation': 'Рассмотрите оптимизацию записи данных и буферизацию',
                'details': f'Средняя скорость записи на диск {avg_disk_write_rate / (1024*1024):.2f} MB/s высока'
            })
        
        return recommendations
    
    def analyze_code_complexity(self, code_path: str) -> List[Dict[str, str]]:
        """
        Анализирует сложность кода и дает рекомендации
        
        Args:
            code_path: Путь к файлу кода
            
        Returns:
            Список рекомендаций по оптимизации
        """
        recommendations = []
        
        try:
            with open(code_path, 'r', encoding='utf-8') as f:
                code_lines = f.readlines()
            
            # Подсчет длинных функций (более 50 строк)
            long_functions = []
            current_function = None
            function_start = 0
            current_line_count = 0
            
            for i, line in enumerate(code_lines, 1):
                stripped_line = line.strip()
                
                # Проверяем начало функции
                if stripped_line.startswith('def ') or stripped_line.startswith('def\t') or \
                   stripped_line.startswith('class ') or stripped_line.startswith('class\t'):
                    if current_function and current_line_count > 50:
                        long_functions.append({
                            'function': current_function,
                            'start_line': function_start,
                            'lines': current_line_count
                        })
                    
                    current_function = stripped_line
                    function_start = i
                    current_line_count = 1
                elif current_function:
                    if stripped_line:  # Не считаем пустые строки
                        current_line_count += 1
            
            # Проверяем последнюю функцию
            if current_function and current_line_count > 50:
                long_functions.append({
                    'function': current_function,
                    'start_line': function_start,
                    'lines': current_line_count
                })
            
            # Добавляем рекомендации
            for func in long_functions:
                recommendations.append({
                    'category': 'code_quality',
                    'severity': 'medium',
                    'recommendation': f'Разделите функцию {func["function"]} на более мелкие функции',
                    'details': f'Функция {func["function"]} занимает {func["lines"]} строк (рекомендуемое максимум 50)'
                })
            
            # Проверяем глубину вложенности (слишком много if/for/while)
            nesting_levels = []
            current_nesting = 0
            
            for i, line in enumerate(code_lines, 1):
                stripped_line = line.lstrip()
                indent_level = len(line) - len(stripped_line)
                
                # Проверяем начало блока
                if any(stripped_line.startswith(keyword) for keyword in ['if ', 'for ', 'while ', 'def ', 'class ', 'try:', 'with ']):
                    nesting_levels.append({
                        'line': i,
                        'indent': indent_level,
                        'statement': stripped_line
                    })
            
            # Находим слишком глубокие вложения
            max_indent = max([nl['indent'] for nl in nesting_levels], default=0)
            if max_indent > 12:  # Предполагаем 4 пробела на уровень
                recommendations.append({
                    'category': 'code_quality',
                    'severity': 'medium',
                    'recommendation': 'Рассмотрите рефакторинг для уменьшения глубины вложенности',
                    'details': f'Найдена глубокая вложенность кода (максимальный отступ {max_indent} символов)'
                })
        
        except Exception as e:
            recommendations.append({
                'category': 'analysis_error',
                'severity': 'low',
                'recommendation': f'Ошибка анализа кода: {str(e)}',
                'details': f'Не удалось проанализировать файл {code_path}'
            })
        
        return recommendations
    
    def generate_optimization_report(self, recommendations: List[Dict[str, str]], output_path: str = None) -> str:
        """
        Генерирует отчет по оптимизации
        
        Args:
            recommendations: Рекомендации по оптимизации
            output_path: Путь для сохранения отчета (если None, генерируется автоматически)
            
        Returns:
            Путь к созданному отчету
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"optimization_report_{timestamp}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'report_type': 'optimization_advice',
            'recommendations': recommendations,
            'summary': {
                'total_recommendations': len(recommendations),
                'by_category': self._categorize_recommendations(recommendations)
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        return output_path
    
    def _categorize_recommendations(self, recommendations: List[Dict[str, str]]) -> Dict[str, int]:
        """
        Категоризирует рекомендации по типам
        
        Args:
            recommendations: Список рекомендаций
            
        Returns:
            Словарь с количеством рекомендаций по категориям
        """
        categories = {}
        for rec in recommendations:
            category = rec['category']
            categories[category] = categories.get(category, 0) + 1
        return categories


def main():
    """Главная функция для демонстрации возможностей валидации конфигурации"""
    print("=== ВАЛИДАЦИЯ КОНФИГУРАЦИИ И ОПТИМИЗАЦИЯ ПРОЕКТА ===")
    
    # Создаем валидатор конфигурации
    config_validator = ConfigValidator()
    
    print("✓ Валидатор конфигурации инициализирован")
    
    # Валидируем структуру проекта
    project_validation = config_validator.validate_project_structure()
    print(f"✓ Валидация структуры проекта: {'Успешна' if project_validation['valid'] else 'С ошибками'}")
    print(f"  - Найдено директорий: {len(project_validation['directories']['found'])}")
    print(f"  - Найдено файлов: {len(project_validation['files']['found'])}")
    
    if not project_validation['directories']['missing'] and not project_validation['files']['missing']:
        print("  - Все обязательные элементы на месте")
    else:
        print(f"  - Отсутствующие директории: {project_validation['directories']['missing']}")
        print(f"  - Отсутствующие файлы: {project_validation['files']['missing']}")
    
    # Валидируем зависимости
    deps_validation = config_validator.validate_dependencies()
    print(f"✓ Валидация зависимостей: {'Успешна' if deps_validation['valid'] else 'С ошибками'}")
    print(f"  - Всего зависимостей: {deps_validation['total_dependencies']}")
    print(f"  - Ошибок: {len(deps_validation['invalid_dependencies'])}")
    print(f"  - Предупреждений: {len(deps_validation['warnings'])}")
    
    # Создаем советника по оптимизации
    optimizer = OptimizationAdvisor()
    
    # Пример анализа производительности
    sample_performance = {
        'avg_cpu_percent': 75.0,
        'avg_memory_percent': 65.0,
        'execution_time_sec': 45.0,
        'avg_disk_write_rate_bps': 50 * 1024 * 1024  # 50 MB/s
    }
    
    recommendations = optimizer.analyze_performance_data(sample_performance)
    print(f"✓ Анализ производительности: {len(recommendations)} рекомендаций")
    
    for rec in recommendations:
        print(f"  - [{rec['severity']}] {rec['recommendation']}")
    
    # Генерируем отчеты
    config_report = config_validator.generate_validation_report(project_validation)
    opt_report = optimizer.generate_optimization_report(recommendations)
    
    print(f"✓ Отчет о валидации конфигурации: {config_report}")
    print(f"✓ Отчет по оптимизации: {opt_report}")
    
    print("Валидация конфигурации и оптимизация успешно протестированы")


if __name__ == "__main__":
    main()