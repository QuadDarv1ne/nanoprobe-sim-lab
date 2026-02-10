# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Модуль управления конфигурацией для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет централизованное управление конфигурацией
для всех компонентов проекта.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os

class ConfigManager:
    """
    Класс для управления конфигурацией проекта
    Обеспечивает централизованное хранение и доступ к параметрам конфигурации
    для всех компонентов проекта.
    """


    def __init__(self, config_file: str = "config.json"):
        """
        Инициализирует менеджер конфигурации

        Args:
            config_file: Путь к файлу конфигурации
        """
        # Определяем путь к конфигурационному файлу
        if Path(config_file).is_absolute():
            self.config_file = Path(config_file)
        else:
            # Ищем конфигурационный файл в стандартных местах
            possible_paths = [
                Path(config_file),
                Path("config") / config_file,
                Path(__file__).parent.parent / "config" / config_file
            ]

            for path in possible_paths:
                if path.exists():
                    self.config_file = path
                    break
            else:
                # Если файл не найден, используем первый вариант
                self.config_file = possible_paths[0]

        self.config = self.load_config()


    def load_config(self) -> Dict[str, Any]:
        """
        Загружает конфигурацию из файла

        Returns:
            Словарь с параметрами конфигурации
        """
        if not os.path.exists(self.config_file):
            print(f"Файл конфигурации {self.config_file} не найден. Создается стандартная конфигурация.")
            self.create_default_config()

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                if str(self.config_file).endswith('.yaml') or str(self.config_file).endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        except Exception as e:
            print(f"Ошибка при загрузке конфигурации: {e}")
            return self.get_default_config()


    def save_config(self) -> bool:
        """
        Сохраняет текущую конфигурацию в файл

        Returns:
            bool: True если успешно сохранено, иначе False
        """
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Ошибка при сохранении конфигурации: {e}")
            return False


    def get_default_config(self) -> Dict[str, Any]:
        """
        Возвращает стандартную конфигурацию проекта

        Returns:
            Словарь со стандартными параметрами конфигурации
        """
        return {
            "project": {
                "name": "Nanoprobe Simulation Lab",
                "version": "1.0.0",
                "description": "Комплекс инструментов для моделирования наноразмерных измерительных систем",
                "author": "Школа программирования Maestro7IT",
                "copyright": "все права защищены"
            },
            "components": {
                "spm_simulator": {
                    "name": "Симулятор СЗМ",
                    "path": "cpp-spm-hardware-sim",
                    "language": "C++/Python",
                    "enabled": True,
                    "config": {
                        "surface_size": [50, 50],
                        "probe_scan_speed": 0.1,
                        "output_format": "txt"
                    }
                },
                "surface_analyzer": {
                    "name": "Анализатор изображений",
                    "path": "py-surface-image-analyzer",
                    "language": "Python",
                    "enabled": True,
                    "config": {
                        "supported_formats": [".jpg", ".png", ".bmp"],
                        "default_filter": "gaussian",
                        "analysis_metrics": ["roughness", "defect_detection"]
                    }
                },
                "sstv_groundstation": {
                    "name": "Наземная станция SSTV",
                    "path": "py-sstv-groundstation",
                    "language": "Python/C++",
                    "enabled": True,
                    "config": {
                        "supported_modes": ["MartinM1", "ScottieS1"],
                        "sample_rate": 44100,
                        "frequency_range": [144000000, 148000000]
                    }
                }
            },
            "paths": {
                "data_dir": "data",
                "output_dir": "output",
                "temp_dir": "temp",
                "log_dir": "logs"
            },
            "simulation": {
                "default_duration": 3600,
                "real_time": True,
                "save_results": True
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "nanoprobe_simulation.log"
            },
            "license": {
                "type": "All Rights Reserved",
                "owner": "Школа программирования Maestro7IT",
                "reserved_rights": [
                    "Использование материалов проекта возможно только с разрешения владельца",
                    "Запрещено копирование, распространение и коммерческое использование без разрешения",
                    "Все права на исходный код принадлежат Школе программирования Maestro7IT"
                ]
            }
        }


    def create_default_config(self):
        """Создает стандартный файл конфигурации"""
        default_config = self.get_default_config()
        self.config = default_config
        self.save_config()
        print(f"Создан стандартный файл конфигурации: {self.config_file}")


    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Получает значение конфигурации по пути ключа

        Args:
            key_path: Путь к ключу в формате 'section.subsection.key'
            default: Значение по умолчанию, если ключ не найден

        Returns:
            Значение конфигурации или значение по умолчанию
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value


    def set(self, key_path: str, value: Any) -> bool:
        """
        Устанавливает значение конфигурации по пути ключа

        Args:
            key_path: Путь к ключу в формате 'section.subsection.key'
            value: Новое значение

        Returns:
            bool: True если успешно установлено, иначе False
        """
        keys = key_path.split('.')
        config_ref = self.config

        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]

        config_ref[keys[-1]] = value
        return self.save_config()


    def update_component_config(self, component_name: str, new_config: Dict[str, Any]) -> bool:
        """
        Обновляет конфигурацию компонента

        Args:
            component_name: Название компонента
            new_config: Новая конфигурация компонента

        Returns:
            bool: True если успешно обновлено, иначе False
        """
        if "components" in self.config and component_name in self.config["components"]:
            self.config["components"][component_name]["config"].update(new_config)
            return self.save_config()
        else:
            print(f"Компонент {component_name} не найден в конфигурации")
            return False


    def get_component_config(self, component_name: str) -> Optional[Dict[str, Any]]:
        """
        Получает конфигурацию компонента

        Args:
            component_name: Название компонента

        Returns:
            Конфигурация компонента или None если компонент не найден
        """
        if "components" in self.config and component_name in self.config["components"]:
            return self.config["components"][component_name]["config"]
        else:
            print(f"Компонент {component_name} не найден в конфигурации")
            return None


    def validate_config(self) -> bool:
        """
        Проверяет валидность конфигурации

        Returns:
            bool: True если конфигурация валидна, иначе False
        """
        required_keys = ["project", "components", "paths"]
        for key in required_keys:
            if key not in self.config:
                print(f"Отсутствует обязательный раздел конфигурации: {key}")
                return False

        # Проверяем наличие необходимых компонентов
        required_components = ["spm_simulator", "surface_analyzer", "sstv_groundstation"]
        for comp in required_components:
            if comp not in self.config["components"]:
                print(f"Отсутствует обязательный компонент: {comp}")
                return False

        return True

def main():
    """Главная функция для демонстрации работы менеджера конфигурации"""
    print("=== МЕНЕДЖЕР КОНФИГУРАЦИИ ПРОЕКТА ===")

    # Создаем менеджер конфигурации
    config_manager = ConfigManager()

    # Проверяем валидность конфигурации
    if config_manager.validate_config():
        print("✓ Конфигурация валидна")
    else:
        print("✗ Конфигурация не валидна")

    # Получаем информацию о проекте
    project_name = config_manager.get("project.name", "Неизвестный проект")
    print(f"Название проекта: {project_name}")

    # Получаем конфигурацию СЗМ симулятора
    spm_config = config_manager.get_component_config("spm_simulator")
    if spm_config:
        print(f"Размер поверхности СЗМ: {spm_config.get('surface_size')}")

    # Обновляем конфигурацию
    success = config_manager.update_component_config("spm_simulator", {
        "surface_size": [100, 100],
        "probe_scan_speed": 0.05
    })

    if success:
        print("✓ Конфигурация СЗМ симулятора обновлена")

    print("Менеджер конфигурации успешно инициализирован")

if __name__ == "__main__":
    main()

