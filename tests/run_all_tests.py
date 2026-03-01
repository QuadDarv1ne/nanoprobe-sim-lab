# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Скрипт запуска всех тестов для проекта Лаборатория моделирования нанозонда
Этот скрипт запускает все доступные тесты для всех компонентов проекта.
"""

import unittest
import sys
import os
import subprocess
from pathlib import Path

def run_specific_test_suite(test_file_path):
    """
    Запускает конкретный тестовый набор

    Args:
        test_file_path: Путь к файлу с тестами

    Returns:
        bool: True если тесты прошли успешно, иначе False
    """
    try:
        # Запускаем тесты с помощью subprocess для изоляции
        result = subprocess.run([
            sys.executable, test_file_path
        ], capture_output=True, text=True, timeout=60)

        print(f"Результаты тестов из {test_file_path}:")
        print(result.stdout)
        if result.stderr:
            print(f"Ошибки: {result.stderr}")

        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"Тесты из {test_file_path} превысили время ожидания")
        return False
    except Exception as e:
        print(f"Ошибка при запуске тестов из {test_file_path}: {str(e)}")
        return False

def discover_and_run_tests():
    """
    Находит и запускает все тестовые файлы в директории tests
    """
    tests_dir = Path(__file__).parent
    test_files = list(tests_dir.glob("test_*.py"))

    if not test_files:
        print("Не найдено тестовых файлов в директории tests/")
        return False

    print(f"Найдено {len(test_files)} тестовых файлов:")
    for test_file in test_files:
        print(f"  - {test_file.name}")

    print("\n" + "="*60)
    print("ЗАПУСК ВСЕХ ТЕСТОВ ПРОЕКТА")
    print("="*60)

    results = {}
    all_passed = True

    for test_file in test_files:
        print(f"\n--- Запуск тестов из {test_file.name} ---")
        success = run_specific_test_suite(str(test_file))
        results[test_file.name] = success
        if not success:
            all_passed = False

    print("\n" + "="*60)
    print("ОБОБЩЕННЫЕ РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print("="*60)

    for test_file, result in results.items():
        status = "ПРОЙДЕН" if result else "НЕ ПРОЙДЕН"
        print(f"{test_file:<30} {status}")

    print("-"*60)
    passed_count = sum(1 for r in results.values() if r)
    total_count = len(results)
    print(f"Пройдено: {passed_count}/{total_count}")

    if all_passed:
        print("[SUCCESS] Все тесты успешно пройдены!")
    else:
        print("[WARNING] Некоторые тесты не прошли. Проверьте вывод выше.")

    return all_passed

def run_with_unittest():
    """
    Альтернативный способ запуска тестов с использованием unittest
    """
    print("Запуск тестов с использованием unittest framework...")

    # Добавляем директорию tests в путь Python
    tests_dir = Path(__file__).parent
    sys.path.insert(0, str(tests_dir))

    # Создаем тестовый набор
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Находим все тестовые файлы и добавляем их в набор
    for test_file in tests_dir.glob("test_*.py"):
        if test_file.name != "run_all_tests.py":  # Исключаем сам этот файл
            # Импортируем модуль и добавляем тесты
            module_name = test_file.stem
            try:
                module = __import__(module_name)
                suite.addTests(loader.loadTestsFromModule(module))
            except ImportError as e:
                print(f"Не удалось импортировать {module_name}: {e}")

    # Запускаем тесты
    if suite.countTestCases() > 0:
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        success = result.wasSuccessful()
        print(f"\nТестирование завершено. Успешно: {'Да' if success else 'Нет'}")
        return success
    else:
        print("Не найдено тестов для запуска")
        return False

def main():
    """Главная функция запуска тестов"""
    print("Лаборатория моделирования нанозонда")
    print("Система тестирования проекта")
    print(f"Текущая директория: {os.getcwd()}")
    print(f"Версия Python: {sys.version}")

    # Запускаем тесты
    success = discover_and_run_tests()

    # Возвращаем код завершения
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

