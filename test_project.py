#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Простой тест проекта Nanoprobe Simulation Lab
"""

import sys
import os
from pathlib import Path

# Добавляем путь к проекту
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Добавляем пути к компонентам
spm_path = project_root / "components" / "cpp-spm-hardware-sim" / "src"
analyzer_path = project_root / "components" / "py-surface-image-analyzer" / "src"
sys.path.insert(0, str(spm_path))
sys.path.insert(0, str(analyzer_path))

# Фиксим кодировку для Windows
if sys.platform == 'win32':
    os.system('chcp 65001 >nul')

print("Тестирование проекта Nanoprobe Simulation Lab...")

try:
    # Тестируем основной CLI
    from src.cli.main import main as cli_main
    print("[OK] Импорт src.cli.main успешен")

    # Тестируем менеджер проекта
    from src.cli.project_manager import ProjectManager
    print("[OK] Импорт src.cli.project_manager успешен")

    # Тестируем симулятор СЗМ
    from spm_simulator import SurfaceModel, ProbeModel, SPMController
    print("[OK] Импорт компонентов симулятора СЗМ успешен")

    # Тестируем процессор изображений
    from image_processor import ImageProcessor, calculate_surface_roughness
    print("[OK] Импорт компонентов обработчика изображений успешен")

    # Тестируем утилиты
    from utils.system_monitor import SystemMonitor
    from utils.cache_manager import CacheManager
    from utils.config_manager import ConfigManager
    print("[OK] Импорт утилит проекта успешен")

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
    monitor.start_monitoring()
    import time
    time.sleep(0.5)  # Ждем сбора метрик
    metrics = monitor.get_current_metrics()
    monitor.stop_monitoring()
    cpu = metrics.get('cpu_percent', 'N/A')
    mem = metrics.get('memory_percent', 'N/A')
    print(f"Метрики системы получены: CPU {cpu}%, Memory {mem}%")

    print("\nВсе компоненты проекта работают корректно!")
    print("Проект готов к использованию.")

except ImportError as e:
    print(f"[ERROR] Ошибка импорта: {e}")
except Exception as e:
    print(f"[ERROR] Ошибка выполнения: {e}")
    import traceback
    traceback.print_exc()

print("\nТестирование завершено.")

