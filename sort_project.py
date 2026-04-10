#!/usr/bin/env python3
"""
Скрипт сортировки проекта Nanoprobe Sim Lab

Перемещает файлы в правильные директории:
- scripts/ — утилиты и скрипты
- tests/ — тестовые файлы
- legacy/ — старые/дублирующие файлы
- components/ — научные компоненты
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).parent

# Классификация файлов
CATEGORIES: Dict[str, List[str]] = {
    # scripts/ — полезные скрипты
    "scripts/active": [
        "migrate_datetime.py",
        "migrate.py",
        "apply_indexes.py",
    ],
    # tests/ — тестовые файлы
    "tests/active": [
        "test_api_init.py",
        "test_backend_api.py",
        "test_created_at.py",
        "test_rtlsdr_full_power.py",
        "test_rtlsdr_sstv.py",
    ],
    # legacy/ — старые файлы (не удалять, но не использовать)
    "legacy/": [
        "listen_fm_radio.py",  # дубль fm_radio.py
        "listen_airband.py",  # дубль am_airband.py
        "quick_scan_airband.py",  # дубль am_airband.py --scan
        "check_pysstv.py",  # одноразовая проверка
        "capture_sstv_mmsstv.py",  # старый SSTV скрипт
        "build_cpp.py",  # не используется
        "format_code.py",  # есть pre-commit
        "cleanup_project.py",  #已完成
        "improve_project.py",  #已完成
        "validate_project.py",  # есть check_cyclic_imports.py
        "run_api_no_redis.py",  # не используется
        "run_monitoring_and_improvements.py",  # одноразовый
        "sstv_ground_station.py",  # старый, есть main.py
        "iss_tracker.py",  # есть satellite_tracker.py
        "rtl_sdr_sstv_capture.py",  # есть components/
        "rtl_sdr_noaa_capture.py",  # есть components/
        "rtl_sdr_visualizer.py",  # есть components/
        "rtlsdr_control_panel.py",  # есть components/
        "test_rtlsdr_full_power.py",  # дубль
    ],
}


def sort_files():
    """Сортировка файлов по директориям"""
    moved = 0
    errors = 0

    print("=" * 60)
    print("📂 Сортировка проекта Nanoprobe Sim Lab")
    print("=" * 60)

    for dest_dir, files in CATEGORIES.items():
        dest_path = PROJECT_ROOT / dest_dir
        dest_path.mkdir(parents=True, exist_ok=True)

        print(f"\n📁 {dest_dir}")

        for filename in files:
            src = PROJECT_ROOT / filename
            if not src.exists():
                print(f"  ⚠️  Не найден: {filename}")
                continue

            dst = dest_path / filename
            try:
                shutil.move(str(src), str(dst))
                print(f"  ✅ {filename} → {dest_dir}")
                moved += 1
            except Exception as e:
                print(f"  ❌ Ошибка {filename}: {e}")
                errors += 1

    print("\n" + "=" * 60)
    print(f"✅ Перемещено: {moved} файлов")
    print(f"❌ Ошибок: {errors}")
    print("=" * 60)

    # Показать что осталось в корне
    print("\n📋 Оставшиеся .py файлы в корне:")
    remaining = [
        f for f in PROJECT_ROOT.glob("*.py") if f.name not in [p for files in CATEGORIES.values() for p in files]
    ]
    for f in sorted(remaining):
        print(f"  📄 {f.name}")


if __name__ == "__main__":
    sort_files()
