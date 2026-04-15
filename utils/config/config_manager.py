"""
Модуль управления конфигурацией для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет централизованное управление конфигурацией
для всех компонентов проекта.
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Класс для управления конфигурацией проекта.

    Обеспечивает централизованное хранение и доступ к параметрам конфигурации
    для всех компонентов проекта.
    """

    _instance: Optional["ConfigManager"] = None
    _lock = threading.Lock()

    def __new__(cls, config_file: str = "config.json") -> "ConfigManager":
        """Singleton паттерн для ConfigManager"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_file: str = "config.json") -> None:
        """
        Инициализирует менеджер конфигурации.

        Args:
            config_file: Путь к файлу конфигурации.
        """
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.config_file: Path
        self.config: Dict[str, Any]
        self._last_modified: Optional[datetime] = None
        self._lock = threading.Lock()

        # Определяем путь к конфигурационному файлу
        if Path(config_file).is_absolute():
            self.config_file = Path(config_file)
        else:
            # Ищем конфигурационный файл в стандартных местах
            possible_paths: List[Path] = [
                Path(config_file),
                Path("config") / config_file,
                Path(__file__).parent.parent / "config" / config_file,
            ]

            for path in possible_paths:
                if path.exists():
                    self.config_file = path
                    break
            else:
                # Если файл не найден, используем первый вариант
                self.config_file = possible_paths[0]

        self.config = self.load_config()
        self._initialized = True

    def load_config(self) -> Dict[str, Any]:
        """
        Загружает конфигурацию из файла.

        Returns:
            Словарь с параметрами конфигурации.
        """
        if not os.path.exists(self.config_file):
            logger.info("Config file %s not found. Creating default config.", self.config_file)
            self.create_default_config()

        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._last_modified = datetime.fromtimestamp(
                    os.path.getmtime(self.config_file, tz=timezone.utc)
                )
                return data if isinstance(data, dict) else {}
        except json.JSONDecodeError as e:
            logger.error("Failed to load config: %s", e)
            return self.get_default_config()
        except Exception as e:
            logger.error("Unexpected error loading config: %s", e)
            return self.get_default_config()

    def save_config(self) -> bool:
        """
        Сохраняет текущую конфигурацию в файл (thread-safe).

        Returns:
            True если сохранение успешно, иначе False.
        """
        with self._lock:
            try:
                with open(self.config_file, "w", encoding="utf-8") as f:
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
                self._last_modified = datetime.now(timezone.utc)
                return True
            except Exception as e:
                logger.error("Failed to save config: %s", e)
                return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        Получает значение по ключу (thread-safe).

        Args:
            key: Ключ в формате "section.subsection.key"
            default: Значение по умолчанию

        Returns:
            Значение конфигурации или default
        """
        with self._lock:
            keys = key.split(".")
            value = self.config
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value

    def set(self, key: str, value: Any, save: bool = True) -> bool:
        """
        Устанавливает значение по ключу (thread-safe).

        Args:
            key: Ключ в формате "section.subsection.key"
            value: Значение для установки
            save: Сохранить ли в файл

        Returns:
            True если успешно
        """
        with self._lock:
            keys = key.split(".")
            config = self.config
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            config[keys[-1]] = value
        if save:
            return self.save_config()
        return True

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
                "description": (
                    "Комплекс инструментов для моделирования " "наноразмерных измерительных систем"
                ),
                "author": "Школа программирования Maestro7IT",
                "copyright": "все права защищены",
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
                        "output_format": "txt",
                    },
                },
                "surface_analyzer": {
                    "name": "Анализатор изображений",
                    "path": "py-surface-image-analyzer",
                    "language": "Python",
                    "enabled": True,
                    "config": {
                        "supported_formats": [".jpg", ".png", ".bmp"],
                        "default_filter": "gaussian",
                        "analysis_metrics": ["roughness", "defect_detection"],
                    },
                },
                "sstv_groundstation": {
                    "name": "Наземная станция SSTV",
                    "path": "py-sstv-groundstation",
                    "language": "Python/C++",
                    "enabled": True,
                    "config": {
                        "supported_modes": ["MartinM1", "ScottieS1"],
                        "sample_rate": 44100,
                        "frequency_range": [144000000, 148000000],
                    },
                },
            },
            "paths": {
                "data_dir": "data",
                "output_dir": "output",
                "temp_dir": "temp",
                "log_dir": "logs",
            },
            "simulation": {"default_duration": 3600, "real_time": True, "save_results": True},
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "nanoprobe_simulation.log",
            },
            "license": {
                "type": "All Rights Reserved",
                "owner": "Школа программирования Maestro7IT",
                "reserved_rights": [
                    ("Использование материалов проекта возможно " "только с разрешения владельца"),
                    (
                        "Запрещено копирование, распространение и "
                        "коммерческое использование без разрешения"
                    ),
                    ("Все права на исходный код принадлежат " "Школе программирования Maestro7IT"),
                ],
            },
        }

    def create_default_config(self):
        """Создает стандартный файл конфигурации"""
        default_config = self.get_default_config()
        self.config = default_config
        self.save_config()
        logger.info("Created default config file: %s", self.config_file)

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
            logger.warning("Component %s not found in config", component_name)
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
            logger.warning("Component %s not found in config", component_name)
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
                logger.warning("Missing required config section: %s", key)
                return False

        # Проверяем наличие необходимых компонентов
        required_components = ["spm_simulator", "surface_analyzer", "sstv_groundstation"]
        for comp in required_components:
            if comp not in self.config["components"]:
                logger.warning("Missing required component: %s", comp)
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
    success = config_manager.update_component_config(
        "spm_simulator", {"surface_size": [100, 100], "probe_scan_speed": 0.05}
    )

    if success:
        print("✓ Конфигурация СЗМ симулятора обновлена")

    print("Менеджер конфигурации успешно инициализирован")


if __name__ == "__main__":
    main()
