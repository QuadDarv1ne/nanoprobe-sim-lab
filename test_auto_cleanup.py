#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестовый скрипт для демонстрации автоматической очистки кэша
"""

import atexit
import time
from utils.cache_manager import CacheManager


def auto_cleanup_demo():
    """Демонстрация автоматической очистки кэша"""
    print("=== ДЕМОНСТРАЦИЯ АВТОМАТИЧЕСКОЙ ОЧИСТКИ КЭША ===")
    
    # Создаем менеджер кэша
    cache_manager = CacheManager()
    
    print("1. Проверка текущего состояния кэша...")
    stats = cache_manager.get_cache_statistics()
    print(f"   Размер кэша: {stats['total_cache_size_mb']} MB")
    print(f"   Файлов в кэше: {stats['total_files']}")
    
    print("\n2. Создание тестовых файлов кэша...")
    # В реальной ситуации здесь будут создаваться файлы кэша
    
    print("\n3. Регистрация функции автоматической очистки...")
    def cleanup_on_exit():
        print("\n" + "="*50)
        print("Автоматическая очистка кэша при завершении программы...")
        try:
            result = cache_manager.auto_cleanup()
            if "status" in result:
                print(f"Статус: {result['status']}")
            else:
                print(f"Удалено файлов: {result['deleted_files']}")
                print(f"Освобождено места: {result['freed_space_mb']} MB")
            
            memory_result = cache_manager.optimize_memory_usage()
            print(f"Освобождено памяти: {memory_result['memory_freed_mb']} MB")
            print("✓ Автоматическая очистка выполнена успешно")
        except Exception as e:
            print(f"❌ Ошибка при очистке: {e}")
        print("="*50)
    
    # Регистрируем функцию очистки
    atexit.register(cleanup_on_exit)
    
    print("\n4. Программа работает...")
    print("   (имитация работы программы)")
    time.sleep(2)
    
    print("\n5. Завершение программы...")
    print("   Автоматическая очистка кэша будет выполнена при выходе")
    
    # Программа завершится и автоматически вызовет cleanup_on_exit


if __name__ == "__main__":
    auto_cleanup_demo()