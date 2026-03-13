"""Unit-тесты для модуля управления конфигурацией."""

import unittest
import tempfile
import json
import sys
from pathlib import Path
from typing import Any, Dict

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_manager import ConfigManager


class TestConfigManager(unittest.TestCase):
    """Тесты для класса ConfigManager"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "config.json"
        
        # Создаём тестовую конфигурацию
        self.test_config = {
            "project": {
                "name": "Test Project",
                "version": "1.0.0"
            },
            "components": {
                "test_component": {
                    "enabled": True,
                    "config": {"param1": "value1"}
                }
            },
            "paths": {
                "data_dir": "data",
                "output_dir": "output"
            }
        }
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_config, f, ensure_ascii=False)

    def tearDown(self):
        """Очистка после теста"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization_with_existing_file(self):
        """Тестирует инициализацию с существующим файлом"""
        cm = ConfigManager(str(self.config_file))
        self.assertIsNotNone(cm.config)
        self.assertEqual(cm.config["project"]["name"], "Test Project")

    def test_get_existing_key(self):
        """Тестирует получение существующего ключа"""
        cm = ConfigManager(str(self.config_file))
        result = cm.get("project.name")
        self.assertEqual(result, "Test Project")

    def test_get_nested_key(self):
        """Тестирует получение вложенного ключа"""
        cm = ConfigManager(str(self.config_file))
        result = cm.get("components.test_component.enabled")
        self.assertEqual(result, True)

    def test_get_nonexistent_key_with_default(self):
        """Тестирует получение несуществующего ключа с значением по умолчанию"""
        cm = ConfigManager(str(self.config_file))
        result = cm.get("nonexistent.key", "default_value")
        self.assertEqual(result, "default_value")

    def test_set_new_value(self):
        """Тестирует установку нового значения"""
        cm = ConfigManager(str(self.config_file))
        success = cm.set("project.new_field", "new_value")
        self.assertTrue(success)
        
        # Перезагружаем и проверяем
        cm2 = ConfigManager(str(self.config_file))
        self.assertEqual(cm2.get("project.new_field"), "new_value")

    def test_validate_config_valid(self):
        """Тестирует валидацию валидной конфигурации"""
        # Создаём полную конфигурацию
        full_config = {
            "project": {"name": "Test", "version": "1.0"},
            "components": {
                "spm_simulator": {},
                "surface_analyzer": {},
                "sstv_groundstation": {}
            },
            "paths": {"data_dir": "data"}
        }
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(full_config, f)
        
        cm = ConfigManager(str(self.config_file))
        result = cm.validate_config()
        self.assertTrue(result)

    def test_validate_config_missing_project(self):
        """Тестирует валидацию с отсутствующим project"""
        invalid_config = {
            "components": {"spm_simulator": {}},
            "paths": {"data_dir": "data"}
        }
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(invalid_config, f)
        
        cm = ConfigManager(str(self.config_file))
        result = cm.validate_config()
        self.assertFalse(result)

    def test_get_default_config(self):
        """Тестирует получение конфигурации по умолчанию"""
        cm = ConfigManager(str(self.config_file))
        default_config = cm.get_default_config()
        
        self.assertIn("project", default_config)
        self.assertIn("components", default_config)
        self.assertIn("paths", default_config)

    def test_update_component_config(self):
        """Тестирует обновление конфигурации компонента"""
        cm = ConfigManager(str(self.config_file))
        
        new_settings = {"param2": "value2", "param3": "value3"}
        success = cm.update_component_config("test_component", new_settings)
        
        self.assertTrue(success)
        
        # Проверяем, что старые параметры сохранились и новые добавились
        cm2 = ConfigManager(str(self.config_file))
        component_config = cm2.get_component_config("test_component")
        self.assertEqual(component_config["param1"], "value1")
        self.assertEqual(component_config["param2"], "value2")

    def test_get_component_config_existing(self):
        """Тестирует получение конфигурации существующего компонента"""
        cm = ConfigManager(str(self.config_file))
        config = cm.get_component_config("test_component")
        
        self.assertIsNotNone(config)
        self.assertEqual(config["param1"], "value1")

    def test_get_component_config_nonexistent(self):
        """Тестирует получение конфигурации несуществующего компонента"""
        cm = ConfigManager(str(self.config_file))
        config = cm.get_component_config("nonexistent_component")
        
        self.assertIsNone(config)

    def test_save_config(self):
        """Тестирует сохранение конфигурации"""
        cm = ConfigManager(str(self.config_file))
        cm.config["test_key"] = "test_value"
        
        success = cm.save_config()
        self.assertTrue(success)
        
        # Проверяем, что файл обновился
        with open(self.config_file, 'r', encoding='utf-8') as f:
            saved_config = json.load(f)
        
        self.assertEqual(saved_config["test_key"], "test_value")


class TestConfigManagerEdgeCases(unittest.TestCase):
    """Тесты для граничных случаев ConfigManager"""

    def test_nonexistent_config_file(self):
        """Тестирует поведение с несуществующим файлом"""
        cm = ConfigManager("/nonexistent/path/config.json")
        # Должна создаться конфигурация по умолчанию
        self.assertIsNotNone(cm.config)

    def test_invalid_json_config(self):
        """Тестирует поведение с некорректным JSON"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        temp_file.write("{ invalid json }")
        temp_file.close()
        
        try:
            cm = ConfigManager(temp_file.name)
            # Должна вернуться конфигурация по умолчанию
            self.assertIsNotNone(cm.config)
        finally:
            Path(temp_file.name).unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main()
