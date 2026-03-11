#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Проверка импорта основных модулей на Python 3.13"""

import sys
print(f"Python: {sys.executable}")
print()

modules_to_test = [
    ("numpy", "numpy"),
    ("matplotlib", "matplotlib.pyplot"),
    ("PIL", "PIL.Image"),
    ("sklearn", "sklearn"),
    ("scipy", "scipy"),
    ("reportlab", "reportlab.lib"),
    ("plotly", "plotly"),
    ("pandas", "pandas"),
]

print("Проверка зависимостей:")
for name, import_name in modules_to_test:
    try:
        __import__(import_name)
        print(f"  ✓ {name}")
    except Exception as e:
        print(f"  ✗ {name}: {e}")

print()
print("Проверка модулей проекта:")

try:
    from utils.pdf_report_generator import ScientificPDFReport
    print("  ✓ pdf_report_generator")
except Exception as e:
    print(f"  ✗ pdf_report_generator: {e}")

try:
    from utils.defect_analyzer import DefectDetector
    print("  ✓ defect_analyzer")
except Exception as e:
    print(f"  ✗ defect_analyzer: {e}")

try:
    from utils.batch_processor import BatchJob
    print("  ✓ batch_processor")
except Exception as e:
    print(f"  ✗ batch_processor: {e}")

try:
    from utils.spm_realtime_visualizer import StreamingDataBuffer
    print("  ✓ spm_realtime_visualizer")
except Exception as e:
    print(f"  ✗ spm_realtime_visualizer: {e}")

try:
    from utils.surface_comparator import SurfaceComparator
    print("  ✓ surface_comparator")
except Exception as e:
    print(f"  ✗ surface_comparator: {e}")

try:
    from utils.database import DatabaseManager
    print("  ✓ database")
except Exception as e:
    print(f"  ✗ database: {e}")

print()
print("Проверка завершена!")
