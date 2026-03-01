# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Главная консольная утилита проекта Лаборатория моделирования нанозонда
Этот скрипт предоставляет интерактивный интерфейс для запуска
всех компонентов проекта и управления ими.
"""

import sys
import os
import atexit
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configuration paths
CONFIG_PATH = project_root / "config" / "config.json"

# Активные процессы
_active_processes: Dict[str, subprocess.Popen] = {}

def show_header():
    """Отображает заголовок программы"""
    print("="*80)
    print("           ЛАБОРАТОРИЯ МОДЕЛИРОВАНИЯ НАНОЗОНДА")
    print("        Nanoprobe Simulation Lab - Main Console")
    print("="*80)
    print(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def show_project_overview():
    """Отображает обзор проекта"""
    print("Проект включает три взаимосвязанных модуля:")
    print("  1. Симулятор аппаратного обеспечения СЗМ на C++")
    print("  2. Анализатор изображений поверхности на Python")
    print("  3. Наземная станция SSTV на Python/C++")
    print()

def show_menu():
    """Отображает главное меню"""
    print("ДОСТУПНЫЕ ОПЕРАЦИИ:")
    print("  1. Запустить симулятор СЗМ (C++)")
    print("  2. Запустить анализатор изображений (Python)")
    print("  3. Запустить наземную станцию SSTV (Python/C++)")
    print("  4. Показать информацию о проекте")
    print("  5. Показать текущую лицензию")
    print("  6. Очистить кэш проекта")
    print("  0. Выход")
    print()

def run_spm_simulator() -> bool:
    """Запускает симулятор СЗМ"""
    print("Запуск симулятора СЗМ...")
    try:
        cpp_path = project_root / "components" / "cpp-spm-hardware-sim" / "build" / "spm-simulator"
        python_spm = project_root / "components" / "cpp-spm-hardware-sim" / "src" / "spm_simulator.py"
        
        if cpp_path.exists():
            print(f"Запуск C++ версии: {cpp_path}")
            process = subprocess.Popen([str(cpp_path)], cwd=str(project_root))
            _active_processes['spm'] = process
            process.wait()
            return True
        elif python_spm.exists():
            print(f"Запуск Python версии: {python_spm}")
            process = subprocess.Popen(
                [sys.executable, str(python_spm)],
                cwd=str(project_root)
            )
            _active_processes['spm'] = process
            process.wait()
            return True
        else:
            print("Файлы симулятора СЗМ не найдены")
            return False
    except Exception as e:
        print(f"Ошибка при запуске симулятора СЗМ: {e}")
        return False

def run_surface_analyzer() -> bool:
    """Запускает анализатор изображений"""
    print("Запуск анализатора изображений поверхности...")
    try:
        analyzer_path = project_root / "components" / "py-surface-image-analyzer" / "src" / "main.py"
        if analyzer_path.exists():
            print(f"Запуск: {analyzer_path}")
            process = subprocess.Popen(
                [sys.executable, str(analyzer_path)],
                cwd=str(project_root)
            )
            _active_processes['analyzer'] = process
            process.wait()
            return True
        else:
            print("Файл анализатора изображений не найден")
            return False
    except Exception as e:
        print(f"Ошибка при запуске анализатора изображений: {e}")
        return False

def run_sstv_groundstation() -> bool:
    """Запускает наземную станцию SSTV"""
    print("Запуск наземной станции SSTV...")
    try:
        station_path = project_root / "components" / "py-sstv-groundstation" / "src" / "main.py"
        if station_path.exists():
            print(f"Запуск: {station_path}")
            process = subprocess.Popen(
                [sys.executable, str(station_path)],
                cwd=str(project_root)
            )
            _active_processes['sstv'] = process
            process.wait()
            return True
        else:
            print("Файл наземной станции SSTV не найден")
            return False
    except Exception as e:
        print(f"Ошибка при запуске наземной станции SSTV: {e}")
        return False

def show_project_info():
    """Показывает информацию о проекте"""
    print("ИНФОРМАЦИЯ О ПРОЕКТЕ:")
    print("-" * 40)
    print("Название: Лаборатория моделирования нанозонда")
    print("Версия: 1.0.0")
    print("Описание: Комплексный проект для моделирования")
    print("          сканирующей зондовой микроскопии")
    print("          и обработки изображений поверхности")
    print("Автор: Школа программирования Maestro7IT")
    print("Лицензия: Проприетарная (ограниченные права)")
    print("-" * 40)

def show_license():
    """Показывает информацию о лицензии"""
    print("ИНФОРМАЦИЯ О ЛИЦЕНЗИИ:")
    print("-" * 40)
    print("Лицензия: Проприетарная лицензия")
    print("Владелец: Школа программирования Maestro7IT")
    print("Все права защищены")
    print()
    print("ОГРАНИЧЕННЫЕ ПРАВА:")
    print("• Использование в образовательных целях")
    print("• Использование в научных исследованиях")
    print("• Использование в некоммерческих проектах")
    print("• Модификация исходного кода")
    print("• Распространение в неизменном виде")
    print()
    print("ЗАПРЕЩЕНО:")
    print("• Коммерческое использование")
    print("• Распространение с удалением авторских прав")
    print("• Использование в военных целях")
    print("• Использование в проектах с закрытым исходным кодом")
    print("-" * 40)

def clean_project_cache() -> bool:
    """Очищает кэш проекта"""
    print("Очистка кэша проекта...")
    try:
        from utils.cache_manager import CacheManager
        cache_manager = CacheManager(str(project_root))

        stats = cache_manager.get_cache_statistics()
        print(f"Текущий размер кэша: {stats['total_cache_size_mb']} MB")
        print(f"Всего файлов в кэше: {stats['total_files']}")

        result = cache_manager.auto_cleanup()

        if "status" in result:
            print(f"Статус: {result['status']}")
        else:
            print(f"Удалено файлов: {result['deleted_files']}")
            print(f"Освобождено места: {result['freed_space_mb']} MB")

        memory_result = cache_manager.optimize_memory_usage()
        print(f"Освобождено памяти: {memory_result['memory_freed_mb']} MB")
        print("Очистка кэша завершена успешно!")
        return True

    except ImportError:
        print("Модуль управления кэшем не найден")
        return False
    except Exception as e:
        print(f"Ошибка при очистке кэша: {e}")
        return False

def _cleanup_processes():
    """Очищает все активные процессы"""
    for name, process in _active_processes.items():
        try:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=3)
                print(f"✓ Процесс {name} остановлен")
        except Exception:
            try:
                process.kill()
                print(f"✓ Процесс {name} уничтожен")
            except Exception:
                pass
    _active_processes.clear()

def auto_cleanup_on_exit():
    """Автоматическая очистка кэша при завершении программы"""
    print("\n" + "="*50)
    print("Завершение работы...")
    
    # Останавливаем активные процессы
    if _active_processes:
        print("Остановка активных процессов...")
        _cleanup_processes()
    
    # Очистка кэша
    print("Автоматическая очистка кэша...")
    try:
        cleanup_success = clean_project_cache()
        if cleanup_success:
            print("✓ Завершение работы выполнено успешно")
        else:
            print("⚠ Завершение работы завершено с предупреждениями")
    except Exception as e:
        print(f"❌ Ошибка при завершении работы: {e}")
    print("="*50)

def main():
    """Главная функция программы"""
    atexit.register(auto_cleanup_on_exit)

    # Автоочистка кэша при старте
    print("Инициализация проекта...")
    try:
        from utils.cache_manager import CacheManager
        cache_manager = CacheManager(str(project_root))
        cache_manager.auto_cleanup()
    except Exception:
        pass

    show_header()
    show_project_overview()

    while True:
        show_menu()
        try:
            choice = input("Выберите действие (0-6): ").strip()

            if choice == '1':
                run_spm_simulator()
            elif choice == '2':
                run_surface_analyzer()
            elif choice == '3':
                run_sstv_groundstation()
            elif choice == '4':
                show_project_info()
            elif choice == '5':
                show_license()
            elif choice == '6':
                clean_project_cache()
            elif choice == '0':
                print("\nСпасибо за использование Лаборатории моделирования нанозонда")
                print("До новых встреч :)")
                break
            else:
                print("Неверный выбор. Пожалуйста, выберите от 0 до 6.")

        except KeyboardInterrupt:
            print("\n\nРабота программы прервана пользователем.")
            break
        except Exception as e:
            print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    main()

