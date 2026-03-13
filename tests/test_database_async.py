#!/usr/bin/env python3
"""
Тесты для async методов DatabaseManager
"""

import sys
import asyncio
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.database import DatabaseManager


async def test_async_get_connection():
    """Тест async get_connection"""
    print("Тест async get_connection...")
    
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    try:
        db = DatabaseManager(db_path=db_path)
        
        async with db.get_connection_async() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1
        
        print("✓ async get_connection: PASS")
        return True
    except Exception as e:
        print(f"✗ async get_connection: FAIL - {e}")
        return False
    finally:
        DatabaseManager.close_all_pools()
        os.unlink(db_path)


async def test_async_scan_operations():
    """Тест async операций со сканированиями"""
    print("Тест async scan operations...")
    
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    try:
        db = DatabaseManager(db_path=db_path)
        
        # Добавляем тестовые данные
        db.add_scan_result("spm", "test", 100, 100)
        db.add_scan_result("image", "test", 200, 200)
        
        # Тест get_scan_results_async
        scans = await db.get_scan_results_async(limit=10)
        assert len(scans) == 2, f"Ожидалось 2 сканирования, получено {len(scans)}"
        
        # Тест get_scan_by_id_async
        scan = await db.get_scan_by_id_async(1)
        assert scan is not None, "Сканирование не найдено"
        assert scan["scan_type"] == "spm"
        
        # Тест count_scans_async
        count = await db.count_scans_async()
        assert count == 2, f"Ожидалось 2, получено {count}"
        
        count_spm = await db.count_scans_async("spm")
        assert count_spm == 1, f"Ожидалось 1 spm, получено {count_spm}"
        
        print("✓ async scan operations: PASS")
        return True
    except Exception as e:
        print(f"✗ async scan operations: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        DatabaseManager.close_all_pools()
        os.unlink(db_path)


async def test_async_cache():
    """Тест async кэширования"""
    print("Тест async cache...")
    
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    try:
        db = DatabaseManager(db_path=db_path, enable_cache=True)
        
        # Добавляем данные
        db.add_scan_result("spm", "test", 100, 100)
        
        # Первый запрос - без кэша
        scans1 = await db.get_scan_results_async()
        assert len(scans1) == 1
        
        # Второй запрос - из кэша
        scans2 = await db.get_scan_results_async()
        assert scans2 == scans1, "Кэш не работает"
        
        # Проверяем кэш
        stats = db.get_cache_stats()
        assert stats["valid_entries"] > 0, "Кэш пуст"
        
        print("✓ async cache: PASS")
        return True
    except Exception as e:
        print(f"✗ async cache: FAIL - {e}")
        return False
    finally:
        DatabaseManager.close_all_pools()
        os.unlink(db_path)


def test_pool_stats():
    """Тест статистики пула"""
    print("Тест pool stats...")
    
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    try:
        db = DatabaseManager(db_path=db_path, pool_size=3)
        
        # Делаем несколько запросов
        db.add_scan_result("spm", "test", 100, 100)
        db.get_scan_results()
        db.get_scan_by_id(1)
        
        # Получаем статистику
        stats = db.get_pool_stats()
        
        assert "hits" in stats
        assert "misses" in stats
        assert "created" in stats
        assert "pool_size" in stats
        assert stats["pool_size"] == 3
        
        print(f"✓ pool stats: PASS (hits={stats['hits']}, misses={stats['misses']}, created={stats['created']})")
        return True
    except Exception as e:
        print(f"✗ pool stats: FAIL - {e}")
        return False
    finally:
        DatabaseManager.close_all_pools()
        os.unlink(db_path)


async def run_async_tests():
    """Запуск async тестов"""
    results = []
    
    results.append(await test_async_get_connection())
    results.append(await test_async_scan_operations())
    results.append(await test_async_cache())
    
    return results


def main():
    """Запуск всех тестов"""
    print("=" * 60)
    print("ТЕСТЫ ASYNC DATABASE METHODS")
    print("=" * 60)
    
    # Sync тест
    sync_result = test_pool_stats()
    
    # Async тесты
    async_results = asyncio.run(run_async_tests())
    
    all_results = [sync_result] + async_results
    passed = sum(all_results)
    total = len(all_results)
    
    print("\n" + "=" * 60)
    print(f"ИТОГИ: {passed}/{total} тестов пройдено ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 Все тесты пройдены!")
        return 0
    else:
        print(f"❌ {total - passed} тест(а) провалено")
        return 1


if __name__ == "__main__":
    sys.exit(main())
