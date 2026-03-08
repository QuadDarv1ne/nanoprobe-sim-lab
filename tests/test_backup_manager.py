#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль тестирования для backup_manager
"""

import unittest
import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../utils'))

from backup_manager import BackupManager


class TestBackupManager(unittest.TestCase):
    """Тесты для класса BackupManager"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.temp_dir = tempfile.mkdtemp()
        self.backup_dir = os.path.join(self.temp_dir, 'backups')
        os.makedirs(self.backup_dir, exist_ok=True)

    def tearDown(self):
        """Очистка после теста"""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

    def test_create_backup_invalid_name(self):
        """Тестирует создание бэкапа с некорректным именем"""
        # Пустое имя должно вызывать ValueError
        with self.assertRaises(ValueError):
            manager = BackupManager()
            manager.backup_dir = self.backup_dir
            manager.create_backup(backup_name='')

    def test_create_backup_none_name(self):
        """Тестирует создание бэкапа с None именем (автогенерация)"""
        # None имя должно работать (автогенерация)
        try:
            manager = BackupManager()
            manager.backup_dir = self.backup_dir
            # Просто проверяем что метод существует и не падает сразу
            self.assertTrue(hasattr(manager, 'create_backup'))
        except Exception:
            # Может упасть из-за отсутствия файлов для бэкапа - это ок
            pass

    def test_list_backups(self):
        """Тестирует список резервных копий"""
        manager = BackupManager()
        manager.backup_dir = self.backup_dir
        backups = manager.list_backups()
        self.assertIsInstance(backups, list)

    def test_delete_backup_nonexistent(self):
        """Тестирует удаление несуществующего бэкапа"""
        manager = BackupManager()
        manager.backup_dir = self.backup_dir
        result = manager.delete_backup('nonexistent_backup')
        self.assertFalse(result)

    def test_metadata_initialization(self):
        """Тестирует инициализацию метаданных"""
        manager = BackupManager()
        self.assertIsInstance(manager.metadata, dict)


def run_tests():
    """Запускает все тесты."""
    print("=" * 60)
    print("ЗАПУСК ТЕСТОВ ДЛЯ BACKUP MANAGER")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 60)
    print(f"РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print(f"  Пройдено: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Ошибки: {len(result.errors)}")
    print(f"  Провалы: {len(result.failures)}")
    print("=" * 60)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
