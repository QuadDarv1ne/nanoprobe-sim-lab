#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тесты для модуля управления кэшем
"""

import unittest
import tempfile
import os
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

from cache_manager import CacheManager, CacheInfo


class TestCacheManager(unittest.TestCase):
    """Тесты для класса CacheManager"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir) / "cache"
        self.cache_dir.mkdir()
        
        self.config = {
            "cache_directories": ["cache"],
            "max_age_days": 7,
            "max_size_mb": 100,
            "auto_cleanup": True
        }
        
        config_file = Path(self.temp_dir) / "config" / "cache_config.json"
        config_file.parent.mkdir()
        import json
        with open(config_file, 'w') as f:
            json.dump(self.config, f)

    def tearDown(self):
        """Очистка после теста"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Тестирует инициализацию CacheManager"""
        cache_mgr = CacheManager(project_root=self.temp_dir)
        self.assertIsNotNone(cache_mgr.cache_config)
        self.assertIsInstance(cache_mgr.cache_directories, list)

    def test_get_cache_statistics(self):
        """Тестирует получение статистики кэша"""
        cache_mgr = CacheManager(project_root=self.temp_dir)
        
        stats = cache_mgr.get_cache_statistics()
        self.assertIsInstance(stats, dict)

    def test_cleanup_cache(self):
        """Тестирует очистку кэша"""
        cache_mgr = CacheManager(project_root=self.temp_dir)
        
        result = cache_mgr.cleanup_cache()
        self.assertIsInstance(result, dict)

    def test_auto_cleanup(self):
        """Тестирует автоматическую очистку"""
        cache_mgr = CacheManager(project_root=self.temp_dir)
        
        result = cache_mgr.auto_cleanup()
        self.assertIsInstance(result, dict)

    def test_optimize_memory_usage(self):
        """Тестирует оптимизацию памяти"""
        cache_mgr = CacheManager(project_root=self.temp_dir)

        result = cache_mgr.optimize_memory_usage()
        self.assertIsInstance(result, dict)

    def test_cache_directories(self):
        """Тестирует получение директорий кэша"""
        cache_mgr = CacheManager(project_root=self.temp_dir)

        self.assertIsInstance(cache_mgr.cache_directories, list)


class TestCacheInfo(unittest.TestCase):
    """Тесты для класса CacheInfo"""

    def test_cache_info_creation(self):
        """Тестирует создание CacheInfo"""
        from datetime import datetime
        
        info = CacheInfo(
            path=Path("/test/cache"),
            size_bytes=1024,
            file_count=10,
            last_accessed=datetime.now(),
            cache_type="temp"
        )
        
        self.assertEqual(info.size_bytes, 1024)
        self.assertEqual(info.file_count, 10)
        self.assertEqual(info.cache_type, "temp")


if __name__ == '__main__':
    unittest.main()
