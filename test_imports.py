#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Проверка импорта основных модулей"""

print("Проверка импорта модулей...")

try:
    from utils.pdf_report_generator import ScientificPDFReport
    print("✓ pdf_report_generator")
except Exception as e:
    print(f"✗ pdf_report_generator: {e}")

try:
    from utils.defect_analyzer import DefectDetector
    print("✓ defect_analyzer")
except Exception as e:
    print(f"✗ defect_analyzer: {e}")

try:
    from utils.batch_processor import BatchJob
    print("✓ batch_processor")
except Exception as e:
    print(f"✗ batch_processor: {e}")

try:
    from utils.spm_realtime_visualizer import StreamingDataBuffer
    print("✓ spm_realtime_visualizer")
except Exception as e:
    print(f"✗ spm_realtime_visualizer: {e}")

try:
    from utils.surface_comparator import SurfaceComparator
    print("✓ surface_comparator")
except Exception as e:
    print(f"✗ surface_comparator: {e}")

try:
    from utils.database import ScanDatabase
    print("✓ database")
except Exception as e:
    print(f"✗ database: {e}")

print("\nВсе основные модули проверены!")
