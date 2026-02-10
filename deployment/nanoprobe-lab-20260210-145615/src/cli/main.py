#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Главная консольная утилита проекта Лаборатория моделирования нанозонда
Этот скрипт предоставляет интерактивный интерфейс для запуска 
всех компонентов проекта и управления ими.
"""

import sys
import os
import atexit
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configuration paths
CONFIG_PATH = project_root / "config" / "config.json"


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


def run_spm_simulator():
    """Запускает симулятор СЗМ"""
    print("Запуск симулятора СЗМ...")
    try:
        # Пытаемся запустить C++ версию
        cpp_path = project_root / "components" / "cpp-spm-hardware-sim" / "build" / "spm-simulator"
        if cpp_path.exists():
            os.system(str(cpp_path))
        else:
            print("C++ версия не найдена. Попробуйте Python-реализацию...")
            # Запускаем Python-реализацию
            python_spm = project_root / "components" / "cpp-spm-hardware-sim" / "src" / "spm_simulator.py"
            if python_spm.exists():
                os.system(f"{sys.executable} {python_spm}")
            else:
                print("Файлы симулятора СЗМ не найдены")
    except Exception as e:
        print(f"Ошибка при запуске симулятора СЗМ: {e}")


def run_surface_analyzer():
    """Запускает анализатор изображений"""
    print("Запуск анализатора изображений поверхности...")
    try:
        analyzer_path = project_root / "components" / "py-surface-image-analyzer" / "src" / "main.py"
        if analyzer_path.exists():
            os.system(f"{sys.executable} {analyzer_path}")
        else:
            print("Файл анализатора изображений не найден")
    except Exception as e:
        print(f"Ошибка при запуске анализатора изображений: {e}")


def run_sstv_groundstation():
    """Запускает наземную станцию SSTV"""
    print("Запуск наземной станции SSTV...")
    try:
        station_path = project_root / "components" / "py-sstv-groundstation" / "src" / "main.py"
        if station_path.exists():
            os.system(f"{sys.executable} {station_path}")
        else:
            print("Файл наземной станции SSTV не найден")
    except Exception as e:
        print(f"Ошибка при запуске наземной станции SSTV: {e}")


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


def clean_project_cache():
    """Очищает кэш проекта"""
    print("Очистка кэша проекта...")
    try:
        from utils.cache_manager import CacheManager
        cache_manager = CacheManager(str(project_root))
        
        # Показываем текущую статистику
        stats = cache_manager.get_cache_statistics()
        print(f"Текущий размер кэша: {stats['total_cache_size_mb']} MB")
        print(f"Всего файлов в кэше: {stats['total_files']}")
        
        # Выполняем очистку
        result = cache_manager.auto_cleanup()
        
        if "status" in result:
            print(f"Статус: {result['status']}")
        else:
            print(f"Удалено файлов: {result['deleted_files']}")
            print(f"Освобождено места: {result['freed_space_mb']} MB")
        
        # Оптимизация памяти
        memory_result = cache_manager.optimize_memory_usage()
        print(f"Освобождено памяти: {memory_result['memory_freed_mb']} MB")
        
        print("Очистка кэша завершена успешно!")
        
        return True
        
    except ImportError:
        print("Модуль управления кэшем не найден")
        print("Установите необходимые зависимости или создайте модуль cache_manager")
        return False
    except Exception as e:
        print(f"Ошибка при очистке кэша: {e}")
        return False


def auto_cleanup_on_exit():
    """Автоматическая очистка кэша при завершении программы"""
    print("\n" + "="*50)
    print("Автоматическая очистка кэша при завершении...")
    try:
        cleanup_success = clean_project_cache()
        if cleanup_success:
            print("✓ Автоматическая очистка кэша выполнена успешно")
        else:
            print("⚠ Автоматическая очистка кэша завершена с предупреждениями")
    except Exception as e:
        print(f"❌ Ошибка при автоматической очистке кэша: {e}")
    print("="*50)


def main():
    """Главная функция программы"""
    # Регистрируем функцию автоматической очистки
    atexit.register(auto_cleanup_on_exit)
    
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