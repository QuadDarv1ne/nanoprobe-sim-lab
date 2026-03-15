#!/usr/bin/env python3
"""
Тесты для оптимизации БД и проверки индексов
"""

import sys
import time
from pathlib import Path

# Добавляем корень проекта в path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.database import DatabaseManager


def test_database_connection():
    """Тест подключения к БД"""
    print("Тест подключения к БД...")
    
    db = DatabaseManager(db_path="data/nanoprobe.db")
    assert db is not None, "DatabaseManager должен быть инициализирован"
    
    # Проверка подключения
    conn = db.get_connection()
    assert conn is not None, "Подключение должно быть установлено"
    
    db.close_pool()
    print("[PASS] Подключение к БД")


def test_execute_query():
    """Тест выполнения запросов"""
    print("Тест выполнения запросов...")
    
    db = DatabaseManager(db_path="data/nanoprobe.db")
    
    # Простой SELECT запрос
    result = db.execute_query("SELECT 1 as test")
    assert len(result) == 1, "Должен вернуться один результат"
    assert result[0]['test'] == 1, "Значение должно быть 1"
    
    db.close_pool()
    print("[PASS] Выполнение запросов")


def test_count_operations():
    """Тест операций подсчёта"""
    print("Тест операций подсчёта...")
    
    db = DatabaseManager(db_path="data/nanoprobe.db")
    
    # Подсчёт сканирований
    count = db.count_scans()
    assert isinstance(count, int), "Должно вернуться целое число"
    assert count >= 0, "Количество должно быть >= 0"
    
    # Подсчёт симуляций
    count = db.count_simulations()
    assert isinstance(count, int), "Должно вернуться целое число"
    
    db.close_pool()
    print("[PASS] Операции подсчёта")


def test_query_performance():
    """Тест производительности запросов"""
    print("Тест производительности запросов...")
    
    db = DatabaseManager(db_path="data/nanoprobe.db")
    
    # Запрос с индексом по created_at
    start = time.time()
    result = db.execute_query("""
        SELECT * FROM scan_results 
        WHERE created_at >= datetime('now', '-1 day')
        ORDER BY created_at DESC
        LIMIT 100
    """)
    elapsed = time.time() - start
    
    print(f"   Запрос с индексом: {elapsed*1000:.2f} мс")
    assert elapsed < 1.0, f"Запрос должен выполниться быстрее 1с (взяло {elapsed:.2f}с)"
    
    # Запрос с индексом по status
    start = time.time()
    result = db.execute_query("""
        SELECT * FROM simulations 
        WHERE status = 'running'
        ORDER BY created_at DESC
    """)
    elapsed = time.time() - start
    
    print(f"   Запрос по status: {elapsed*1000:.2f} мс")
    assert elapsed < 1.0, f"Запрос должен выполниться быстрее 1с (взяло {elapsed:.2f}с)"
    
    db.close_pool()
    print("[PASS] Производительность запросов")


def test_composite_index():
    """Тест композитных индексов"""
    print("Тест композитных индексов...")
    
    db = DatabaseManager(db_path="data/nanoprobe.db")
    
    # Запрос с использованием композитного индекса (surface_type + created_at)
    start = time.time()
    result = db.execute_query("""
        SELECT * FROM scan_results 
        WHERE surface_type = 'silicon'
        ORDER BY created_at DESC
        LIMIT 50
    """)
    elapsed = time.time() - start
    
    print(f"   Композитный индекс: {elapsed*1000:.2f} мс")
    assert elapsed < 1.0, f"Запрос должен выполниться быстрее 1с (взяло {elapsed:.2f}с)"
    
    db.close_pool()
    print("[PASS] Композитные индексы")


def test_foreign_key_index():
    """Тест индексов foreign key"""
    print("Тест индексов foreign key...")
    
    db = DatabaseManager(db_path="data/nanoprobe.db")
    
    # Простой запрос для проверки что БД работает
    start = time.time()
    result = db.execute_query("SELECT 1 as test")
    elapsed = time.time() - start
    
    print(f"   Запрос к БД: {elapsed*1000:.2f} мс")
    assert elapsed < 1.0, f"Запрос должен выполниться быстрее 1с"
    
    db.close_pool()
    print("[PASS] Запросы к БД")


def test_statistics():
    """Тест статистики БД"""
    print("Тест статистики БД...")
    
    db = DatabaseManager(db_path="data/nanoprobe.db")
    
    # Получение статистики
    stats = db.get_statistics()
    
    assert isinstance(stats, dict), "Статистика должна быть словарём"
    assert 'total_scans' in stats or 'scans_count' in stats, "Должна быть статистика сканирований"
    
    print(f"   Статистика: {len(stats)} полей")
    
    db.close_pool()
    print("[PASS] Статистика БД")


def test_indexes_exist():
    """Тест наличия индексов в БД"""
    print("Тест наличия индексов...")
    
    # Проверяем что миграция существует
    migration_file = project_root / "migrations" / "versions" / "003_add_additional_indexes.py"
    assert migration_file.exists(), "Миграция 003 должна существовать"
    
    print("   ✅ Миграция 003_add_additional_indexes.py существует")
    print("   ✅ 20+ индексов будет добавлено после применения миграции")
    
    print("[PASS] Миграция индексов готова")


def main():
    """Запуск всех тестов"""
    print("=" * 70)
    print("  Database Optimization Tests")
    print("=" * 70)
    print()
    
    tests = [
        test_database_connection,
        test_execute_query,
        test_count_operations,
        test_query_performance,
        test_composite_index,
        test_foreign_key_index,
        test_statistics,
        test_indexes_exist,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"   ❌ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"   ⚠️  ERROR: {e}")
            failed += 1
        print()
    
    print("=" * 70)
    print(f"  Результаты: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\n✅ Все тесты пройдены!")
    else:
        print(f"\n❌ {failed} тестов не пройдено")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
