# -*- coding: utf-8 -*-
#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3

"""
Модуль проверки целостности данных для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для проверки
целостности и корректности данных проекта.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import csv

class DataIntegrityChecker:
    """
    Класс проверки целостности данных
    Обеспечивает проверку целостности, корректности и
    валидность данных проекта.
    """


    def __init__(self):
        """Инициализирует проверяльщик целостности данных"""
        self.check_results = {}


    def calculate_checksum(self, data: bytes) -> str:
        """
        Вычисляет контрольную сумму данных

        Args:
            data: Данные для вычисления контрольной суммы

        Returns:
            Контрольная сумма в формате hex
        """
        digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
        digest.update(data)
        return digest.finalize().hex()


    def calculate_file_checksum(self, file_path: str) -> Optional[str]:
        """
        Вычисляет контрольную сумму файла

        Args:
            file_path: Путь к файлу

        Returns:
            Контрольная сумма в формате hex или None при ошибке
        """
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            return self.calculate_checksum(data)
        except Exception as e:
            print(f"Ошибка чтения файла {file_path}: {e}")
            return None


    def verify_file_integrity(self, file_path: str, expected_checksum: str) -> bool:
        """
        Проверяет целостность файла

        Args:
            file_path: Путь к файлу
            expected_checksum: Ожидаемая контрольная сумма

        Returns:
            True если файл цел, иначе False
        """
        actual_checksum = self.calculate_file_checksum(file_path)
        return actual_checksum == expected_checksum


    def check_numpy_array_integrity(self, array: np.ndarray) -> Dict[str, any]:
        """
        Проверяет целостность numpy массива

        Args:
            array: Numpy массив для проверки

        Returns:
            Словарь с результатами проверки
        """
        results = {
            'shape': array.shape,
            'dtype': str(array.dtype),
            'size': array.size,
            'ndim': array.ndim,
            'has_nan': np.isnan(array).any(),
            'has_inf': np.isinf(array).any(),
            'min_value': float(np.nanmin(array)) if array.size > 0 else None,
            'max_value': float(np.nanmax(array)) if array.size > 0 else None,
            'mean_value': float(np.nanmean(array)) if array.size > 0 else None,
            'std_value': float(np.nanstd(array)) if array.size > 0 else None
        }

        # Проверяем на наличие некорректных значений
        results['valid'] = not (results['has_nan'] or results['has_inf'])

        return results


    def check_csv_integrity(self, file_path: str) -> Dict[str, any]:
        """
        Проверяет целостность CSV файла

        Args:
            file_path: Путь к CSV файлу

        Returns:
            Словарь с результатами проверки
        """
        try:
            df = pd.read_csv(file_path)

            results = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'null_counts': {col: int(df[col].isnull().sum()) for col in df.columns},
                'duplicate_rows': int(df.duplicated().sum()),
                'file_size': Path(file_path).stat().st_size
            }

            # Проверяем, есть ли нулевые значения
            total_nulls = sum(results['null_counts'].values())
            results['has_nulls'] = total_nulls > 0
            results['valid'] = not results['has_nulls'] and results['duplicate_rows'] == 0

            return results
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }


    def check_json_integrity(self, file_path: str) -> Dict[str, any]:
        """
        Проверяет целостность JSON файла

        Args:
            file_path: Путь к JSON файлу

        Returns:
            Словарь с результатами проверки
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            results = {
                'keys_count': len(data.keys()) if isinstance(data, dict) else 'Not a dict',
                'type': type(data).__name__,
                'size_bytes': Path(file_path).stat().st_size,
                'valid': True
            }

            return results
        except json.JSONDecodeError as e:
            return {
                'valid': False,
                'error': f"JSON decode error: {str(e)}"
            }
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }


    def generate_data_fingerprint(self, data: any) -> str:
        """
        Генерирует уникальный отпечаток данных

        Args:
            data: Данные для генерации отпечатка

        Returns:
            Строка отпечатка
        """
        if isinstance(data, np.ndarray):
            data_str = str(data.shape) + str(data.dtype) + str(data.flat[0] if data.size > 0 else "")
        elif isinstance(data, pd.DataFrame):
            data_str = str(data.shape) + str(list(data.columns)) + str(len(data))
        elif isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True, default=str)
        else:
            data_str = str(data)

        return hashlib.sha256(data_str.encode('utf-8')).hexdigest()


    def create_data_manifest(self, directory: str, recursive: bool = True) -> Dict[str, any]:
        """
        Создает манифест данных для директории

        Args:
            directory: Директория для сканирования
            recursive: Рекурсивно сканировать поддиректории

        Returns:
            Словарь с манифестом данных
        """
        dir_path = Path(directory)
        manifest = {
            'directory': str(dir_path.absolute()),
            'timestamp': datetime.now().isoformat(),
            'files': [],
            'subdirectories': []
        }

        for item in dir_path.iterdir():
            if item.is_file():
                file_info = {
                    'name': item.name,
                    'path': str(item),
                    'size': item.stat().st_size,
                    'modified': datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
                    'checksum': self.calculate_file_checksum(str(item))
                }
                manifest['files'].append(file_info)
            elif item.is_dir() and recursive:
                subdir_manifest = self.create_data_manifest(str(item), recursive=False)
                manifest['subdirectories'].append(subdir_manifest)

        return manifest


    def verify_data_manifest(self, manifest: Dict[str, any]) -> Dict[str, any]:
        """
        Проверяет манифест данных

        Args:
            manifest: Манифест данных для проверки

        Returns:
            Словарь с результатами проверки
        """
        results = {
            'directory': manifest['directory'],
            'timestamp': manifest['timestamp'],
            'verified_files': [],
            'missing_files': [],
            'corrupted_files': [],
            'valid': True
        }

        for file_info in manifest['files']:
            file_path = Path(file_info['path'])

            if not file_path.exists():
                results['missing_files'].append(file_info)
                results['valid'] = False
                continue

            current_checksum = self.calculate_file_checksum(str(file_path))
            if current_checksum != file_info['checksum']:
                results['corrupted_files'].append({
                    'file': file_info,
                    'current_checksum': current_checksum
                })
                results['valid'] = False
            else:
                results['verified_files'].append(file_info)

        return results


    def check_simulation_data_integrity(self, data_dict: Dict[str, any]) -> Dict[str, any]:
        """
        Проверяет целостность данных симуляции

        Args:
            data_dict: Словарь с данными симуляции

        Returns:
            Словарь с результатами проверки
        """
        results = {
            'checks_performed': [],
            'errors': [],
            'warnings': [],
            'valid': True
        }

        # Проверяем наличие обязательных ключей
        required_keys = ['timestamp', 'data', 'metadata']
        for key in required_keys:
            if key not in data_dict:
                results['errors'].append(f"Missing required key: {key}")
                results['valid'] = False

        # Проверяем данные
        if 'data' in data_dict:
            data = data_dict['data']
            if isinstance(data, np.ndarray):
                array_check = self.check_numpy_array_integrity(data)
                results['checks_performed'].append(('numpy_array', array_check))

                if not array_check['valid']:
                    results['errors'].append("Array contains invalid values (NaN or Inf)")
                    results['valid'] = False
            elif isinstance(data, dict):
                # Рекурсивная проверка вложенных данных
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        sub_check = self.check_numpy_array_integrity(value)
                        results['checks_performed'].append((f'data.{key}', sub_check))

                        if not sub_check['valid']:
                            results['errors'].append(f"Array at data.{key} contains invalid values")
                            results['valid'] = False

        # Проверяем метаданные
        if 'metadata' in data_dict:
            metadata = data_dict['metadata']
            if not isinstance(metadata, dict):
                results['errors'].append("Metadata must be a dictionary")
                results['valid'] = False

        return results

class IntegrityReportGenerator:
    """
    Класс генерации отчетов о целостности
    Создает отчеты о проверке целостности данных проекта.
    """


    def __init__(self):
        """Инициализирует генератор отчетов о целостности"""
        self.reports = []


    def generate_integrity_report(self, check_results: Dict[str, any],
    """TODO: Add description"""

                               output_path: str = None) -> str:
        """
        Генерирует отчет о целостности данных

        Args:
            check_results: Результаты проверки целостности
            output_path: Путь для сохранения отчета (если None, генерируется автоматически)

        Returns:
            Путь к созданному отчету
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data_integrity_report_{timestamp}.json"

        report = {
            'timestamp': datetime.now().isoformat(),
            'report_type': 'data_integrity',
            'check_results': check_results,
            'summary': self._generate_summary(check_results)
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        self.reports.append(report)
        return output_path


    def _generate_summary(self, check_results: Dict[str, any]) -> Dict[str, any]:
        """
        Генерирует сводку по результатам проверки

        Args:
            check_results: Результаты проверки целостности

        Returns:
            Словарь со сводкой
        """
        summary = {
            'total_checks': 0,
            'passed_checks': 0,
            'failed_checks': 0,
            'overall_status': 'PASS',
            'critical_issues': [],
            'warnings': []
        }

        # Подсчитываем результаты
        if 'checks_performed' in check_results:
            summary['total_checks'] = len(check_results['checks_performed'])
            for check_name, check_result in check_results['checks_performed']:
                if check_result.get('valid', False):
                    summary['passed_checks'] += 1
                else:
                    summary['failed_checks'] += 1

        # Определяем общий статус
        if summary['failed_checks'] > 0:
            summary['overall_status'] = 'FAIL'

        # Собираем критические проблемы
        if 'errors' in check_results:
            summary['critical_issues'] = check_results['errors']

        if 'warnings' in check_results:
            summary['warnings'] = check_results['warnings']

        return summary

def main():
    """Главная функция для демонстрации возможностей проверки целостности данных"""
    print("=== ПРОВЕРКА ЦЕЛОСТНОСТИ ДАННЫХ ПРОЕКТА ===")

    # Создаем проверяльщик целостности
    checker = DataIntegrityChecker()
    reporter = IntegrityReportGenerator()

    print("✓ Проверяльщик целостности данных инициализирован")

    # Тестируем проверку numpy массива
    test_array = np.random.rand(10, 10)
    array_check = checker.check_numpy_array_integrity(test_array)
    print(f"✓ Проверка numpy массива: {'Успешна' if array_check['valid'] else 'Ошибка'}")

    # Тестируем генерацию отпечатка данных
    fingerprint = checker.generate_data_fingerprint(test_array)
    print(f"✓ Отпечаток данных: {fingerprint[:16]}...")

    # Тестируем проверку данных симуляции
    sim_data = {
        'timestamp': datetime.now().isoformat(),
        'data': test_array,
        'metadata': {
            'type': 'surface_data',
            'source': 'spm_simulation'
        }
    }

    sim_check = checker.check_simulation_data_integrity(sim_data)
    print(f"✓ Проверка данных симуляции: {'Успешна' if sim_check['valid'] else 'Ошибка'}")

    # Генерируем отчет
    report_path = reporter.generate_integrity_report(sim_check)
    print(f"✓ Отчет о целостности сохранен: {report_path}")

    print("Проверка целостности данных успешно протестирована")

if __name__ == "__main__":
    main()

