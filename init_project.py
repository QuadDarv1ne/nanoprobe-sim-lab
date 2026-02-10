#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт инициализации проекта Лаборатория моделирования нанозонда
Этот скрипт инициализирует все компоненты проекта и устанавливает необходимые зависимости.
"""

import os
import sys
import subprocess
from pathlib import Path
import importlib.util
from utils.config_manager import ConfigManager
from utils.logger import setup_project_logging
from utils.data_manager import DataManager


def check_python_dependencies():
    """Проверяет наличие необходимых Python зависимостей"""
    required_packages = [
        'numpy',
        'matplotlib',
        'pandas',
        'PIL',  # Pillow
        'yaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        if package == 'PIL':
            # Pillow использует PIL как внутреннее имя
            spec = importlib.util.find_spec("PIL")
        else:
            spec = importlib.util.find_spec(package)
        
        if spec is None:
            missing_packages.append(package)
    
    return missing_packages


def install_missing_dependencies(missing_packages):
    """Устанавливает отсутствующие зависимости"""
    if not missing_packages:
        print("✓ Все зависимости уже установлены")
        return True
    
    print(f"Установка отсутствующих зависимостей: {missing_packages}")
    
    # Сопоставляем пакеты с именами для pip
    package_mapping = {
        'PIL': 'Pillow'
    }
    
    for package in missing_packages:
        pip_package = package_mapping.get(package, package)
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_package])
            print(f"✓ Установлен {pip_package}")
        except subprocess.CalledProcessError:
            print(f"✗ Не удалось установить {pip_package}")
            return False
    
    return True


def initialize_directories():
    """Инициализирует необходимые директории проекта"""
    directories = [
        'data',
        'output',
        'logs',
        'temp',
        'samples'
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(exist_ok=True)
        print(f"✓ Директория создана: {dir_path}")
    
    # Создаем поддиректории для каждого компонента
    component_dirs = [
        'cpp-spm-hardware-sim/data',
        'cpp-spm-hardware-sim/output',
        'py-surface-image-analyzer/data',
        'py-surface-image-analyzer/output',
        'py-sstv-groundstation/data',
        'py-sstv-groundstation/output'
    ]
    
    for comp_dir in component_dirs:
        dir_path = Path(comp_dir)
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Компонентная директория создана: {dir_path}")


def initialize_config():
    """Инициализирует конфигурацию проекта"""
    config_manager = ConfigManager()
    print("✓ Конфигурация проекта инициализирована")
    return config_manager


def initialize_logging():
    """Инициализирует систему логирования"""
    logger_manager = setup_project_logging()
    logger_manager.log_system_event("Инициализация проекта", "INFO")
    print("✓ Система логирования инициализирована")
    return logger_manager


def initialize_data_manager():
    """Инициализирует менеджер данных"""
    data_manager = DataManager()
    print("✓ Менеджер данных инициализирован")
    return data_manager


def create_sample_data():
    """Создает примеры данных для тестирования"""
    import numpy as np
    
    # Создаем пример данных поверхности
    surface_data = np.random.rand(20, 20) * 0.5
    surface_data[5:10, 5:10] += 0.3  # Добавляем "гору"
    surface_data[15:18, 2:5] -= 0.4  # Добавляем "кратер"
    
    data_manager = DataManager()
    data_manager.save_surface_data(surface_data, "sample_surface.txt")
    print("✓ Пример данных поверхности создан")
    
    # Создаем пример результатов анализа
    analysis_results = {
        "surface_roughness": float(np.std(surface_data)),
        "max_height": float(np.max(surface_data)),
        "min_height": float(np.min(surface_data)),
        "analysis_date": "2023-12-01",
        "quality_score": 0.92
    }
    
    data_manager.save_image_analysis_results(analysis_results, "sample_analysis.json")
    print("✓ Пример результатов анализа создан")


def display_project_info():
    """Отображает информацию о проекте"""
    print("\n" + "="*60)
    print("ЛАБОРАТОРИЯ МОДЕЛИРОВАНИЯ НАНОЗОНДА")
    print("Проект успешно инициализирован")
    print("="*60)
    print("Доступные компоненты:")
    print("  1. Симулятор СЗМ (cpp-spm-hardware-sim)")
    print("  2. Анализатор изображений (py-surface-image-analyzer)")
    print("  3. Наземная станция SSTV (py-sstv-groundstation)")
    print("\nДоступные утилиты:")
    print("  - project_manager.py: Управление проектом")
    print("  - utils/config_manager.py: Управление конфигурацией")
    print("  - utils/logger.py: Система логирования")
    print("  - utils/data_manager.py: Управление данными")
    print("  - tests/: Тестирование компонентов")
    print("\nДля запуска проекта используйте:")
    print("  python project_manager.py")
    print("="*60)


def main():
    """Главная функция инициализации проекта"""
    print("ИНИЦИАЛИЗАЦИЯ ПРОЕКТА ЛАБОРАТОРИИ МОДЕЛИРОВАНИЯ НАНОЗОНДА")
    print("-" * 60)
    
    # Проверяем зависимости
    print("Проверка зависимостей...")
    missing_deps = check_python_dependencies()
    
    if missing_deps:
        print(f"Найдены отсутствующие зависимости: {missing_deps}")
        response = input("Установить отсутствующие зависимости? (y/n): ")
        if response.lower() in ['y', 'yes', 'да']:
            if install_missing_dependencies(missing_deps):
                print("✓ Зависимости успешно установлены")
            else:
                print("✗ Ошибка установки зависимостей")
                return False
        else:
            print("Продолжение без установки зависимостей может привести к ошибкам")
    else:
        print("✓ Все зависимости установлены")
    
    # Инициализируем директории
    print("\nИнициализация директорий...")
    initialize_directories()
    
    # Инициализируем компоненты
    print("\nИнициализация компонентов проекта...")
    config_manager = initialize_config()
    logger_manager = initialize_logging()
    data_manager = initialize_data_manager()
    
    # Создаем примеры данных
    print("\nСоздание примеров данных...")
    create_sample_data()
    
    # Логируем завершение инициализации
    logger_manager.log_system_event("Проект успешно инициализирован", "INFO")
    
    # Отображаем информацию
    display_project_info()
    
    print("\n🎉 Инициализация завершена успешно!")
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)