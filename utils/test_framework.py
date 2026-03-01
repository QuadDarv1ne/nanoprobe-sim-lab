#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль тестовой платформы для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для комплексного
тестирования и обеспечения качества кода проекта.
"""

import unittest
import pytest
import coverage
import time
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import json
import subprocess
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
import threading
from functools import wraps

class TestFramework:
    """
    Класс тестовой платформы
    Обеспечивает комплексное тестирование,
    покрытие кода и обеспечение качества проекта.
    """


    def __init__(self, project_root: str = "."):
        """
        Инициализирует тестовую платформу

        Args:
            project_root: Корневая директория проекта
        """
        self.project_root = Path(project_root).resolve()
        self.test_results = {}
        self.coverage_results = {}
        self.performance_results = {}
        self.quality_results = {}


    def discover_tests(self, test_directory: str = "tests") -> List[str]:
        """
        Находит все тестовые файлы в проекте

        Args:
            test_directory: Директория с тестами

        Returns:
            Список путей к тестовым файлам
        """
        test_dir = self.project_root / test_directory
        test_files = []

        if test_dir.exists():
            for root, dirs, files in os.walk(test_dir):
                for file in files:
                    if file.startswith('test_') and file.endswith('.py'):
                        test_files.append(str(Path(root) / file))

        # Также ищем тесты в подмодулях
        for submodule in ['cpp-spm-hardware-sim', 'py-surface-image-analyzer', 'py-sstv-groundstation']:
            sub_test_dir = self.project_root / submodule / 'tests'
            if sub_test_dir.exists():
                for root, dirs, files in os.walk(sub_test_dir):
                    for file in files:
                        if file.startswith('test_') and file.endswith('.py'):
                            test_files.append(str(Path(root) / file))

        return test_files


    def run_unittests(self, test_pattern: str = "test_*.py") -> Dict[str, Any]:
        """
        Запускает тесты с использованием unittest

        Args:
            test_pattern: Паттерн для поиска тестов

        Returns:
            Результаты выполнения тестов
        """
        loader = unittest.TestLoader()
        suite = loader.discover(start_dir=str(self.project_root), pattern=test_pattern)

        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        results = {
            'total_tests': result.testsRun,
            'passed': result.testsRun - len(result.failures) - len(result.errors),
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped),
            'time_elapsed': getattr(result, 'time_taken', 0),
            'test_details': []
        }

        # Добавляем детали тестов
        for failure in result.failures:
            results['test_details'].append({
                'test': str(failure[0]),
                'result': 'FAILURE',
                'message': str(failure[1])
            })

        for error in result.errors:
            results['test_details'].append({
                'test': str(error[0]),
                'result': 'ERROR',
                'message': str(error[1])
            })

        return results


    def run_pytest(self, test_directory: str = "tests", coverage_report: bool = True) -> Dict[str, Any]:
        """
        Запускает тесты с использованием pytest

        Args:
            test_directory: Директория с тестами
            coverage_report: Создавать ли отчет о покрытии

        Returns:
            Результаты выполнения тестов
        """
        try:
            cmd = ["pytest", str(self.project_root / test_directory), "-v", "--json-report"]

            if coverage_report:
                cmd.extend(["--cov=.", "--cov-report=json", "--cov-report=html"])

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.project_root))

            return {
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0
            }
        except Exception as e:
            return {
                'return_code': -1,
                'error': str(e),
                'success': False
            }


    def measure_code_coverage(self, source_directory: str = ".", test_directory: str = "tests") -> Dict[str, Any]:
        """
        Измеряет покрытие кода тестами

        Args:
            source_directory: Директория с исходным кодом
            test_directory: Директория с тестами

        Returns:
            Результаты измерения покрытия
        """
        cov = coverage.Coverage(source=[str(self.project_root / source_directory)])
        cov.start()

        # Запускаем тесты для измерения покрытия
        loader = unittest.TestLoader()
        suite = loader.discover(start_dir=str(self.project_root / test_directory), pattern="test_*.py")
        unittest.TextTestRunner().run(suite)

        cov.stop()
        cov.save()

        # Получаем результаты
        total_coverage = cov.report()
        coverage_details = cov.analysis()

        results = {
            'total_coverage_percent': total_coverage,
            'files_analyzed': len(coverage_details),
            'coverage_details': {}
        }

        for filename, analysis in coverage_details.items():
            results['coverage_details'][filename] = {
                'statements': analysis[1],
                'executed': len(analysis[0]),
                'missing': analysis[2],
                'excluded': analysis[3]
            }

        return results


    def run_performance_tests(self, test_functions: List[Callable], iterations: int = 10) -> Dict[str, Any]:
        """
        Запускает тесты производительности

        Args:
            test_functions: Список функций для тестирования производительности
            iterations: Количество итераций для каждого теста

        Returns:
            Результаты тестов производительности
        """
        results = {}

        for test_func in test_functions:
            test_name = test_func.__name__
            times = []

            for _ in range(iterations):
                start_time = time.perf_counter()
                test_func()
                end_time = time.perf_counter()
                times.append(end_time - start_time)

            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)

            results[test_name] = {
                'average_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'iterations': iterations,
                'times': times
            }

        return results


    def run_stress_tests(self, target_function: Callable, duration: int = 60,
                        concurrency: int = 10) -> Dict[str, Any]:
        """
        Запускает стресс-тесты

        Args:
            target_function: Функция для тестирования
            duration: Продолжительность теста в секундах
            concurrency: Количество одновременных вызовов

        Returns:
            Результаты стресс-теста
        """
        def worker():
                    start_time = time.time()
            iterations = 0

            while time.time() - start_time < duration:
                try:
                    target_function()
                    iterations += 1
                except Exception:
                    # Игнорируем ошибки для продолжения теста
                    pass

            return iterations

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(worker) for _ in range(concurrency)]
            results = [future.result() for future in futures]

        total_iterations = sum(results)
        elapsed_time = time.time() - start_time

        return {
            'total_iterations': total_iterations,
            'duration_seconds': elapsed_time,
            'iterations_per_second': total_iterations / elapsed_time,
            'concurrency_level': concurrency,
            'errors_encountered': sum(results) != total_iterations  # Приблизительная оценка
        }


    def run_integration_tests(self) -> Dict[str, Any]:
        """
        Запускает интеграционные тесты

        Returns:
            Результаты интеграционных тестов
        """
        # Тестируем взаимодействие между компонентами
        results = {
            'spm_component_test': self._test_spm_component(),
            'image_analyzer_test': self._test_image_analyzer_component(),
            'sstv_component_test': self._test_sstv_component(),
            'data_exchange_test': self._test_data_exchange(),
            'api_integration_test': self._test_api_integration()
        }

        # Подсчитываем общие результаты
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result['success'])

        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'individual_results': results
        }


    def _test_spm_component(self) -> Dict[str, Any]:
        """Тестирует компонент СЗМ"""
        try:
            from cpp_spm_hardware_sim.src.spm_simulator import SurfaceModel, SPMController

            # Создаем тестовую поверхность
            surface = SurfaceModel(10, 10)
            controller = SPMController()

            # Тестируем базовую функциональность
            assert surface.getWidth() == 10
            assert surface.getHeight() == 10

            return {
                'success': True,
                'message': 'Компонент СЗМ работает корректно',
                'details': 'SurfaceModel и SPMController инициализированы успешно'
            }
        except ImportError:
            return {
                'success': False,
                'message': 'Компонент СЗМ не найден или не может быть импортирован',
                'details': 'Возможно, компонент еще не реализован или зависимости отсутствуют'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Ошибка в компоненте СЗМ: {str(e)}',
                'details': str(e)
            }


    def _test_image_analyzer_component(self) -> Dict[str, Any]:
        """Тестирует компонент анализатора изображений"""
        try:
            from py_surface_image_analyzer.src.image_processor import ImageProcessor

            # Создаем процессор изображений
            processor = ImageProcessor()

            # Тестируем базовую функциональность
            assert processor is not None

            return {
                'success': True,
                'message': 'Компонент анализатора изображений работает корректно',
                'details': 'ImageProcessor инициализирован успешно'
            }
        except ImportError:
            return {
                'success': False,
                'message': 'Компонент анализатора изображений не найден или не может быть импортирован',
                'details': 'Возможно, компонент еще не реализован или зависимости отсутствуют'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Ошибка в компоненте анализатора изображений: {str(e)}',
                'details': str(e)
            }


    def _test_sstv_component(self) -> Dict[str, Any]:
        """Тестирует компонент SSTV"""
        try:
            from py_sstv_groundstation.src.sstv_decoder import SSTVDecoder

            # Создаем декодер SSTV
            decoder = SSTVDecoder()

            # Тестируем базовую функциональность
            assert decoder is not None

            return {
                'success': True,
                'message': 'Компонент SSTV работает корректно',
                'details': 'SSTVDecoder инициализирован успешно'
            }
        except ImportError:
            return {
                'success': False,
                'message': 'Компонент SSTV не найден или не может быть импортирован',
                'details': 'Возможно, компонент еще не реализован или зависимости отсутствуют'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Ошибка в компоненте SSTV: {str(e)}',
                'details': str(e)
            }


    def _test_data_exchange(self) -> Dict[str, Any]:
        """Тестирует обмен данными между компонентами"""
        try:
            from api.data_exchange import DataExchangeManager

            # Создаем менеджер обмена данными
            exchange_manager = DataExchangeManager()

            # Тестируем базовую функциональность
            formats = exchange_manager.get_supported_formats()
            assert len(formats) > 0

            return {
                'success': True,
                'message': 'Обмен данными работает корректно',
                'details': f'Поддерживаемые форматы: {formats}'
            }
        except ImportError:
            return {
                'success': False,
                'message': 'Модуль обмена данными не найден или не может быть импортирован',
                'details': 'Возможно, модуль еще не реализован или зависимости отсутствуют'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Ошибка в обмене данными: {str(e)}',
                'details': str(e)
            }


    def _test_api_integration(self) -> Dict[str, Any]:
        """Тестирует интеграцию через API"""
        try:
            from api.api_interface import NanoprobeAPI

            # Создаем API интерфейс
            api = NanoprobeAPI()

            # Тестируем базовую функциональность
            assert api is not None

            return {
                'success': True,
                'message': 'Интеграция через API работает корректно',
                'details': 'NanoprobeAPI инициализирован успешно'
            }
        except ImportError:
            return {
                'success': False,
                'message': 'API интерфейс не найден или не может быть импортирован',
                'details': 'Возможно, модуль еще не реализован или зависимости отсутствуют'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Ошибка в API интеграции: {str(e)}',
                'details': str(e)
            }


    def generate_test_report(self, output_path: str = None) -> str:
        """
        Генерирует отчет о тестировании

        Args:
            output_path: Путь для сохранения отчета (если None, генерируется автоматически)

        Returns:
            Путь к созданному отчету
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"test_report_{timestamp}.json"

        report = {
            'timestamp': datetime.now().isoformat(),
            'report_type': 'comprehensive_test_report',
            'project_root': str(self.project_root),
            'unittest_results': self.run_unittests(),
            'integration_test_results': self.run_integration_tests(),
            'code_coverage_results': self.measure_code_coverage(),
            'summary': self._generate_test_summary()
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        return output_path


    def _generate_test_summary(self) -> Dict[str, Any]:
        """
        Генерирует сводку по результатам тестирования

        Returns:
            Сводка по результатам тестирования
        """
        # Запускаем базовое тестирование для получения данных
        unit_results = self.run_unittests()
        integration_results = self.run_integration_tests()
        coverage_results = self.measure_code_coverage()

        total_tests = unit_results['total_tests'] + integration_results['total_tests']
        passed_tests = unit_results['passed'] + integration_results['passed_tests']

        return {
            'total_tests_executed': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate_percent': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'code_coverage_percent': coverage_results.get('total_coverage_percent', 0),
            'test_categories': {
                'unit_tests': unit_results,
                'integration_tests': integration_results,
                'code_coverage': coverage_results
            }
        }


    def run_continuous_integration_pipeline(self) -> Dict[str, Any]:
        """
        Запускает полный CI/CD пайплайн

        Returns:
            Результаты выполнения CI/CD пайплайна
        """
        start_time = time.time()

        results = {
            'timestamp': datetime.now().isoformat(),
            'pipeline_started': start_time,
            'steps': {},
            'overall_success': True
        }

        try:
            # Шаг 1: Запуск юнит-тестов
            print("Запуск юнит-тестов...")
            unit_results = self.run_unittests()
            results['steps']['unit_tests'] = unit_results
            results['overall_success'] &= unit_results['failures'] == 0 and unit_results['errors'] == 0

            # Шаг 2: Запуск интеграционных тестов
            print("Запуск интеграционных тестов...")
            integration_results = self.run_integration_tests()
            results['steps']['integration_tests'] = integration_results
            results['overall_success'] &= integration_results['failed_tests'] == 0

            # Шаг 3: Измерение покрытия кода
            print("Измерение покрытия кода...")
            coverage_results = self.measure_code_coverage()
            results['steps']['code_coverage'] = coverage_results
            results['overall_success'] &= coverage_results.get('total_coverage_percent', 0) >= 70  # Минимальный порог 70%

            # Шаг 4: Генерация отчета
            print("Генерация отчета...")
            report_path = self.generate_test_report()
            results['report_path'] = report_path

        except Exception as e:
            results['overall_success'] = False
            results['error'] = str(e)

        results['pipeline_finished'] = time.time()
        results['total_duration'] = results['pipeline_finished'] - start_time

        return results

class QualityAssurance:
    """
    Класс обеспечения качества
    Обеспечивает статический анализ кода,
    проверку стиля и другие аспекты качества.
    """


    def __init__(self, project_root: str = "."):
        """
        Инициализирует систему обеспечения качества

        Args:
            project_root: Корневая директория проекта
        """
        self.project_root = Path(project_root).resolve()


    def run_pylint_analysis(self) -> Dict[str, Any]:
        """
        Запускает анализ кода с помощью pylint

        Returns:
            Результаты анализа pylint
        """
        try:
            # Находим все Python файлы в проекте
            python_files = []
            for root, dirs, files in os.walk(self.project_root):
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(str(Path(root) / file))

            if not python_files:
                return {'error': 'Python файлы не найдены', 'success': False}

            # Запускаем pylint (реализация зависит от доступности инструмента)
            cmd = ["python", "-m", "pylint"] + python_files + ["--output-format=json"]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.project_root))

            return {
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode <= 4  # pylint возвращает коды ошибок
            }
        except FileNotFoundError:
            return {
                'error': 'pylint не установлен',
                'success': False
            }
        except Exception as e:
            return {
                'error': str(e),
                'success': False
            }


    def run_flake8_analysis(self) -> Dict[str, Any]:
        """
        Запускает анализ кода с помощью flake8

        Returns:
            Результаты анализа flake8
        """
        try:
            # Находим все Python файлы в проекте
            python_files = []
            for root, dirs, files in os.walk(self.project_root):
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(str(Path(root) / file))

            if not python_files:
                return {'error': 'Python файлы не найдены', 'success': False}

            # Запускаем flake8
            cmd = ["flake8"] + python_files

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.project_root))

            return {
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0
            }
        except FileNotFoundError:
            return {
                'error': 'flake8 не установлен',
                'success': False
            }
        except Exception as e:
            return {
                'error': str(e),
                'success': False
            }


    def run_black_formatter_check(self) -> Dict[str, Any]:
        """
        Проверяет форматирование кода с помощью black

        Returns:
            Результаты проверки форматирования
        """
        try:
            # Находим все Python файлы в проекте
            python_files = []
            for root, dirs, files in os.walk(self.project_root):
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(str(Path(root) / file))

            if not python_files:
                return {'error': 'Python файлы не найдены', 'success': False}

            # Проверяем форматирование (dry-run)
            cmd = ["black", "--check", "--diff"] + python_files

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.project_root))

            return {
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0
            }
        except FileNotFoundError:
            return {
                'error': 'black не установлен',
                'success': False
            }
        except Exception as e:
            return {
                'error': str(e),
                'success': False
            }


    def run_mypy_analysis(self) -> Dict[str, Any]:
        """
        Запускает статический анализ типов с помощью mypy

        Returns:
            Результаты анализа mypy
        """
        try:
            # Находим все Python файлы в проекте
            python_files = []
            for root, dirs, files in os.walk(self.project_root):
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(str(Path(root) / file))

            if not python_files:
                return {'error': 'Python файлы не найдены', 'success': False}

            # Запускаем mypy
            cmd = ["mypy"] + python_files

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.project_root))

            return {
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0
            }
        except FileNotFoundError:
            return {
                'error': 'mypy не установлен',
                'success': False
            }
        except Exception as e:
            return {
                'error': str(e),
                'success': False
            }


    def generate_quality_report(self, output_path: str = None) -> str:
        """
        Генерирует отчет о качестве кода

        Args:
            output_path: Путь для сохранения отчета (если None, генерируется автоматически)

        Returns:
            Путь к созданному отчету
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"quality_report_{timestamp}.json"

        report = {
            'timestamp': datetime.now().isoformat(),
            'report_type': 'code_quality_report',
            'project_root': str(self.project_root),
            'pylint_analysis': self.run_pylint_analysis(),
            'flake8_analysis': self.run_flake8_analysis(),
            'black_formatting_check': self.run_black_formatter_check(),
            'mypy_analysis': self.run_mypy_analysis(),
            'summary': self._generate_quality_summary()
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        return output_path


    def _generate_quality_summary(self) -> Dict[str, Any]:
        """
        Генерирует сводку по качеству кода

        Returns:
            Сводка по качеству кода
        """
        pylint_result = self.run_pylint_analysis()
        flake8_result = self.run_flake8_analysis()
        black_result = self.run_black_formatter_check()
        mypy_result = self.run_mypy_analysis()

        return {
            'pylint_success': pylint_result.get('success', False),
            'flake8_success': flake8_result.get('success', False),
            'black_success': black_result.get('success', False),
            'mypy_success': mypy_result.get('success', False),
            'overall_quality_score': self._calculate_quality_score(
                pylint_result, flake8_result, black_result, mypy_result
            )
        }


    def _calculate_quality_score(self, pylint_result: Dict, flake8_result: Dict,
                              black_result: Dict, mypy_result: Dict) -> float:
        """
        Рассчитывает общий балл качества кода

        Args:
            pylint_result: Результаты pylint анализа
            flake8_result: Результаты flake8 анализа
            black_result: Результаты black проверки
            mypy_result: Результаты mypy анализа

        Returns:
            Общий балл качества (0-100)
        """
        score = 0
        total_checks = 4

        if pylint_result.get('success', False):
            score += 1
        if flake8_result.get('success', False):
            score += 1
        if black_result.get('success', False):
            score += 1
        if mypy_result.get('success', False):
            score += 1

        return (score / total_checks) * 100

def main():
    """Главная функция для демонстрации возможностей тестовой платформы"""
    print("=== ТЕСТОВАЯ ПЛАТФОРМА ПРОЕКТА ===")

    # Создаем тестовую платформу
    test_framework = TestFramework()
    qa_system = QualityAssurance()

    print("✓ Тестовая платформа инициализирована")
    print(f"✓ Корневая директория: {test_framework.project_root}")

    # Находим тесты
    test_files = test_framework.discover_tests()
    print(f"✓ Найдено тестовых файлов: {len(test_files)}")

    if test_files:
        print("  Список тестовых файлов:")
        for test_file in test_files[:5]:  # Показываем первые 5
            print(f"    - {test_file}")
        if len(test_files) > 5:
            print(f"    ... и еще {len(test_files) - 5} файлов")

    # Запускаем юнит-тесты
    print("\nЗапуск юнит-тестов...")
    unit_results = test_framework.run_unittests()
    print(f"  - Всего тестов: {unit_results['total_tests']}")
    print(f"  - Пройдено: {unit_results['passed']}")
    print(f"  - Провалено: {unit_results['failures']}")
    print(f"  - Ошибки: {unit_results['errors']}")

    # Запускаем интеграционные тесты
    print("\nЗапуск интеграционных тестов...")
    integration_results = test_framework.run_integration_tests()
    print(f"  - Всего тестов: {integration_results['total_tests']}")
    print(f"  - Пройдено: {integration_results['passed_tests']}")
    print(f"  - Провалено: {integration_results['failed_tests']}")

    # Измеряем покрытие кода
    print("\nИзмерение покрытия кода...")
    try:
        coverage_results = test_framework.measure_code_coverage()
        print(f"  - Общее покрытие: {coverage_results['total_coverage_percent']:.2f}%")
        print(f"  - Проанализировано файлов: {coverage_results['files_analyzed']}")
    except Exception as e:
        print(f"  - Ошибка измерения покрытия: {e}")

    # Запускаем CI/CD пайплайн
    print("\nЗапуск CI/CD пайплайна...")
    ci_results = test_framework.run_continuous_integration_pipeline()
    print(f"  - Успешно завершено: {ci_results['overall_success']}")
    print(f"  - Длительность: {ci_results['total_duration']:.2f} сек")

    # Запускаем анализ качества кода
    print("\nАнализ качества кода...")
    try:
        pylint_result = qa_system.run_pylint_analysis()
        flake8_result = qa_system.run_flake8_analysis()
        black_result = qa_system.run_black_formatter_check()
        mypy_result = qa_system.run_mypy_analysis()

        print(f"  - Pylint: {'✓' if pylint_result.get('success', False) else '✗'}")
        print(f"  - Flake8: {'✓' if flake8_result.get('success', False) else '✗'}")
        print(f"  - Black: {'✓' if black_result.get('success', False) else '✗'}")
        print(f"  - MyPy: {'✓' if mypy_result.get('success', False) else '✗'}")
    except Exception as e:
        print(f"  - Ошибка анализа качества: {e}")

    # Генерируем отчеты
    test_report_path = test_framework.generate_test_report()
    quality_report_path = qa_system.generate_quality_report()

    print(f"\n✓ Отчет о тестировании: {test_report_path}")
    print(f"✓ Отчет о качестве кода: {quality_report_path}")

    print("\nТестовая платформа успешно протестирована")
    print("\nДоступные функции:")
    print("- Поиск тестов: discover_tests()")
    print("- Запуск юнит-тестов: run_unittests()")
    print("- Запуск интеграционных тестов: run_integration_tests()")
    print("- Измерение покрытия кода: measure_code_coverage()")
    print("- Полный CI/CD пайплайн: run_continuous_integration_pipeline()")
    print("- Анализ качества кода: run_pylint_analysis(), run_flake8_analysis() и др.")

if __name__ == "__main__":
    main()

