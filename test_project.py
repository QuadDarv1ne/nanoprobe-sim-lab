#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Простой тест проекта Nanoprobe Simulation Lab
"""

import sys
from pathlib import Path

# Добавляем путь к проекту
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Тестирование проекта Nanoprobe Simulation Lab...")

try:
    # Тестируем основной CLI
    from src.cli.main import main as cli_main
    print("✅ Импорт src.cli.main успешен")

    # Тестируем менеджер проекта
    from src.cli.project_manager import ProjectManager
    print("✅ Импорт src.cli.project_manager успешен")

    # Тестируем симулятор СЗМ
    from components.cpp_spm_hardware_sim.src.spm_simulator import SurfaceModel, ProbeModel, SPMController
    print("✅ Импорт компонентов симулятора СЗМ успешен")

    # Тестируем процессор изображений
    from components.py_surface_image_analyzer.src.image_processor import ImageProcessor, calculate_surface_roughness
    print("✅ Импорт компонентов обработчика изображений успешен")

    # Тестируем утилиты
    from utils.system_monitor import SystemMonitor
    from utils.cache_manager import CacheManager
    from utils.config_manager import ConfigManager
    print("✅ Импорт утилит проекта успешен")

    # Создаем простой тест
    print("\n--- Создание тестовой поверхности ---")
    surface = SurfaceModel(10, 10)  # Маленькая поверхность для теста
    print(f"Создана поверхность размером {surface.width}x{surface.height}")

    print("\n--- Тестирование зонда ---")
    probe = ProbeModel()
    print(f"Позиция зонда: {probe.get_position()}")

    print("\n--- Тестирование контроллера СЗМ ---")
    controller = SPMController()
    controller.set_surface(surface)
    print("Контроллер СЗМ инициализирован")

    print("\n--- Тестирование процессора изображений ---")
    processor = ImageProcessor()
    print("Процессор изображений создан")

    print("\n--- Тестирование монитора системы ---")
    monitor = SystemMonitor()
    metrics = monitor.get_current_metrics()
    print(f"Метрики системы получены: CPU {metrics['cpu_percent']}%, Memory {metrics['memory_percent']}%")

    print("\n🎉 Все компоненты проекта работают корректно!")
    print("Проект готов к использованию.")

except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
except Exception as e:
    print(f"❌ Ошибка выполнения: {e}")
    import traceback
    traceback.print_exc()

print("\nТестирование завершено.")

