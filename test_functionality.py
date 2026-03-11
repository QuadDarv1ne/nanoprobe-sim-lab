#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Тестирование основных функций проекта"""

import sys
import numpy as np
from pathlib import Path

print("=" * 60)
print("ТЕСТИРОВАНИЕ ФУНКЦИОНАЛЬНОСТИ NANOPROBE SIM LAB")
print("=" * 60)
print()

# 1. Тест PDF отчётов
print("1. Тест PDF отчётов...")
try:
    from utils.pdf_report_generator import ScientificPDFReport
    
    generator = ScientificPDFReport(output_dir="reports/test")
    
    surface_data = {
        'surface_type': 'Silicon Wafer',
        'scan_size': '100x100 нм',
        'resolution': '512x512',
        'scan_date': '2026-03-11',
        'method': 'AFM',
        'mean_height': 15.5,
        'std_deviation': 3.2,
        'max_height': 45.8,
        'skewness': 0.15,
        'kurtosis': 2.8,
        'rms': 3.5
    }
    
    filepath = generator.generate_surface_analysis_report(
        surface_data=surface_data,
        title="Тестовый анализ поверхности",
        author="Test User"
    )
    
    if Path(filepath).exists():
        print(f"   ✓ PDF отчёт создан: {filepath}")
    else:
        print(f"   ✗ Файл не создан")
except Exception as e:
    print(f"   ✗ Ошибка: {e}")

print()

# 2. Тест AI/ML анализа дефектов
print("2. Тест AI/ML анализа дефектов...")
try:
    from utils.defect_analyzer import DefectDetector
    
    detector = DefectDetector(model_name="isolation_forest")
    
    test_data = np.random.rand(100, 100) * 100
    test_data[50:55, 50:55] = 200
    
    results = detector.detect_defects(test_data)
    
    print(f"   ✓ Дефектов обнаружено: {results.get('defects_count', 0)}")
    print(f"   ✓ Время обработки: {results.get('processing_time', 0):.3f} сек")
except Exception as e:
    print(f"   ✗ Ошибка: {e}")

print()

# 3. Тест пакетной обработки
print("3. Тест пакетной обработки...")
try:
    from utils.batch_processor import BatchProcessor
    
    processor = BatchProcessor(output_dir="output/batch_test")
    
    test_items = [f"item_{i}.dat" for i in range(5)]
    
    def dummy_processor(item, parameters=None):
        return f"Processed {item}"
    
    results = processor.process_surface_analysis_batch(test_items, dummy_processor)
    
    print(f"   ✓ Обработано элементов: {len(results)}")
except Exception as e:
    print(f"   ✗ Ошибка: {e}")

print()

# 4. Тест сравнения поверхностей
print("4. Тест сравнения поверхностей...")
try:
    from utils.surface_comparator import SurfaceComparator
    
    comparator = SurfaceComparator()
    
    surface1 = np.random.rand(100, 100)
    surface2 = surface1 + np.random.rand(100, 100) * 0.1
    
    comparison = comparator.compare_surfaces(surface1, surface2)
    
    print(f"   ✓ SSIM: {comparison.get('ssim', 0):.4f}")
    print(f"   ✓ PSNR: {comparison.get('psnr', 0):.2f} dB")
except Exception as e:
    print(f"   ✗ Ошибка: {e}")

print()

# 5. Тест БД сканирований
print("5. Тест БД сканирований...")
try:
    from utils.database import DatabaseManager
    
    db = DatabaseManager(db_path="data/test_scans.db")
    
    scan_id = db.add_scan_result(
        scan_type='AFM',
        surface_type='Test Surface',
        width=256,
        height=256,
        file_path='test_data.dat',
        metadata={'test': True}
    )
    
    scans = db.get_scan_results(limit=10)
    
    print(f"   ✓ Запись добавлена: ID={scan_id}")
    print(f"   ✓ Всего записей в БД: {len(scans)}")
except Exception as e:
    print(f"   ✗ Ошибка: {e}")

print()
print("=" * 60)
print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
print("=" * 60)
