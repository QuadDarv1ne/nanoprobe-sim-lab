#!/usr/bin/env python
"""
Скрипт для добавления индексов в существующую БД
"""

import sqlite3
from pathlib import Path

DB_PATH = Path("data/nanoprobe.db")

if not DB_PATH.exists():
    print(f"❌ БД не найдена: {DB_PATH}")
    exit(1)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Список индексов для создания
indexes = [
    # scan_results
    ("idx_scan_created_at", "scan_results", "created_at"),
    ("idx_scan_type_count", "scan_results", "scan_type, id"),
    
    # simulations
    ("idx_simulations_created_at", "simulations", "created_at"),
    ("idx_simulations_status_date", "simulations", "status, start_time"),
    
    # images
    ("idx_image_created_at", "images", "created_at"),
    ("idx_image_processed", "images", "processed"),
    ("idx_image_type_created", "images", "image_type, created_at"),
    
    # exports
    ("idx_export_created_at", "exports", "created_at"),
    ("idx_export_source", "exports", "source_type, source_id"),
    
    # surface_comparisons
    ("idx_comparison_created_at_full", "surface_comparisons", "created_at"),
]

created = 0
skipped = 0
errors = 0

print("📊 Добавление индексов в БД...")
print(f"📁 Файл: {DB_PATH.absolute()}")
print()

for idx_name, table_name, columns in indexes:
    try:
        # Проверка существования индекса
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
            (idx_name,)
        )
        if cursor.fetchone():
            print(f"⏭️  {idx_name} (уже существует)")
            skipped += 1
            continue
        
        # Создание индекса
        sql = f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table_name} ({columns})"
        cursor.execute(sql)
        print(f"✅ {idx_name} на {table_name} ({columns})")
        created += 1
        
    except Exception as e:
        print(f"❌ {idx_name}: {e}")
        errors += 1

conn.commit()
conn.close()

print()
print("=" * 50)
print(f"✅ Создано индексов: {created}")
print(f"⏭️  Пропущено: {skipped}")
print(f"❌ Ошибок: {errors}")
print("=" * 50)

if errors == 0:
    print("🎉 Все индексы успешно добавлены!")
else:
    print(f"⚠️  Завершено с ошибками: {errors}")
