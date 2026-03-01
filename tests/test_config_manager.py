#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тесты для модуля управления конфигурацией
"""

import unittest
import json
import tempfile
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

from config_manager import ConfigManager


class TestConfigManager(unittest.TestCase):
    """Тесты для класса ConfigManager"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "test_config.json"
        
        self.test_config = {
            "project": {
                "name": "Test Project",
                "version": "1.0.0"
            },
            "components": {
                "spm_simulator": {
                    "enabled": True,
                    "config": {"surface_size": [50, 50]}
                }
            },
            "paths": {
                "data_dir": "data",
                "log_dir": "logs"
            }
        }
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_config, f)

    def tearDown(self):
        """Очистка после теста"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Тестирует инициализацию ConfigManager"""
        config_mgr = ConfigManager(str(self.config_file))
        self.assertIsNotNone(config_mgr.config)
        self.assertEqual(config_mgr.config_file, self.config_file)

    def test_load_config(self):
        """Тестирует загрузку конфигурации"""
        config_mgr = ConfigManager(str(self.config_file))
        config = config_mgr.load_config()
        
        self.assertEqual(config["project"]["name"], "Test Project")
        self.assertEqual(config["project"]["version"], "1.0.0")

    def test_get_method(self):
        """Тестирует метод get для получения значений"""
        config_mgr = ConfigManager(str(self.config_file))
        
        name = config_mgr.get("project.name")
        self.assertEqual(name, "Test Project")
        
        version = config_mgr.get("project.version")
        self.assertEqual(version, "1.0.0")
        
        missing = config_mgr.get("project.missing", "default")
        self.assertEqual(missing, "default")

    def test_set_method(self):
        """Тестирует метод set для установки значений"""
        config_mgr = ConfigManager(str(self.config_file))
        
        success = config_mgr.set("project.version", "2.0.0")
        self.assertTrue(success)
        
        new_version = config_mgr.get("project.version")
        self.assertEqual(new_version, "2.0.0")

    def test_get_component_config(self):
        """Тестирует получение конфигурации компонента"""
        config_mgr = ConfigManager(str(self.config_file))
        
        spm_config = config_mgr.get_component_config("spm_simulator")
        self.assertIsNotNone(spm_config)
        self.assertIn("surface_size", spm_config)

    def test_update_component_config(self):
        """Тестирует обновление конфигурации компонента"""
        config_mgr = ConfigManager(str(self.config_file))
        
        success = config_mgr.update_component_config(
            "spm_simulator",
            {"surface_size": [100, 100]}
        )
        self.assertTrue(success)
        
        spm_config = config_mgr.get_component_config("spm_simulator")
        self.assertEqual(spm_config["surface_size"], [100, 100])

    def test_validate_config(self):
        """Тестирует валидацию конфигурации"""
        config_mgr = ConfigManager(str(self.config_file))
        
        is_valid = config_mgr.validate_config()
        self.assertIsInstance(is_valid, bool)

    def test_default_config(self):
        """Тестирует создание конфигурации по умолчанию"""
        temp_config = Path(self.temp_dir) / "default_config.json"
        config_mgr = ConfigManager(str(temp_config))
        
        self.assertTrue(temp_config.exists())
        default_config = config_mgr.get_default_config()
        self.assertIn("project", default_config)
        self.assertIn("components", default_config)


class TestConfigManagerMissingFile(unittest.TestCase):
    """Тесты для случая отсутствия файла конфигурации"""

    def test_missing_config_file(self):
        """Тестирует поведение при отсутствии файла"""
        config_mgr = ConfigManager("nonexistent_config.json")
        
        self.assertIsNotNone(config_mgr.config)
        self.assertIn("project", config_mgr.config)


if __name__ == '__main__':
    unittest.main()
