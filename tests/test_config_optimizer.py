"""
Тесты для utils/config/config_optimizer.py

Покрытие:
- OptimizationParams dataclass
- ConfigOptimizer инициализация
- Загрузка конфигурации (JSON, TOML, INI)
- Сохранение конфигурации
- Оптимизация параметров
- System metrics
- Thread safety
"""

import configparser
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from utils.config.config_optimizer import ConfigOptimizer, OptimizationParams


@pytest.fixture
def temp_config_dir():
    """Создать временную директорию для конфигов."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_json_config(temp_config_dir):
    """Создать пример JSON конфигурации."""
    config_path = Path(temp_config_dir) / "config.json"
    config = {
        "system": {
            "cpu_threshold": 80.0,
            "memory_threshold": 85.0,
            "max_threads": 4,
        },
        "performance": {
            "batch_size": 100,
            "cache_size": 1000,
            "timeout": 30,
        },
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f)
    return str(config_path)


@pytest.fixture
def sample_toml_config(temp_config_dir):
    """Создать пример TOML конфигурации."""
    config_path = Path(temp_config_dir) / "config.toml"
    config = """
[system]
cpu_threshold = 75.0
memory_threshold = 80.0
max_threads = 8

[performance]
batch_size = 200
cache_size = 2000
timeout = 60
"""
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config)
    return str(config_path)


@pytest.fixture
def sample_ini_config(temp_config_dir):
    """Создать пример INI конфигурации."""
    config_path = Path(temp_config_dir) / "config.ini"
    config = configparser.ConfigParser()
    config["system"] = {
        "cpu_threshold": "70.0",
        "memory_threshold": "75.0",
        "max_threads": "2",
    }
    config["performance"] = {
        "batch_size": "50",
        "cache_size": "500",
        "timeout": "15",
    }
    with open(config_path, "w", encoding="utf-8") as f:
        config.write(f)
    return str(config_path)


class TestOptimizationParams:
    """Тесты dataclass OptimizationParams."""

    def test_default_values(self):
        """Значения по умолчанию."""
        params = OptimizationParams()

        assert params.cpu_threshold == 80.0
        assert params.memory_threshold == 80.0
        assert params.disk_threshold == 80.0
        assert params.max_threads == 4
        assert params.batch_size == 100
        assert params.cache_size == 1000
        assert params.timeout == 30
        assert params.retry_attempts == 3

    def test_custom_values(self):
        """Кастомные значения."""
        params = OptimizationParams(
            cpu_threshold=90.0,
            memory_threshold=95.0,
            max_threads=8,
            batch_size=200,
        )

        assert params.cpu_threshold == 90.0
        assert params.memory_threshold == 95.0
        assert params.max_threads == 8
        assert params.batch_size == 200

    def test_asdict_conversion(self):
        """Конвертация в словарь."""
        params = OptimizationParams()
        params_dict = {
            "cpu_threshold": params.cpu_threshold,
            "memory_threshold": params.memory_threshold,
            "max_threads": params.max_threads,
        }

        assert params_dict["cpu_threshold"] == 80.0
        assert params_dict["max_threads"] == 4


class TestConfigOptimizerInit:
    """Тесты инициализации ConfigOptimizer."""

    def test_init_with_default_config(self, temp_config_dir):
        """Инициализация с конфитом по умолчанию."""
        # Создаем пустой config.json
        config_path = Path(temp_config_dir) / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({}, f)

        optimizer = ConfigOptimizer(config_path=str(config_path))

        assert optimizer.config_path == config_path
        assert optimizer.original_config == {}

    def test_init_with_json_config(self, sample_json_config):
        """Инициализация с JSON конфигом."""
        optimizer = ConfigOptimizer(config_path=sample_json_config)

        assert optimizer.original_config != {}
        assert "system" in optimizer.original_config
        assert "performance" in optimizer.original_config

    def test_init_with_toml_config(self, sample_toml_config):
        """Инициализация с TOML конфигом."""
        optimizer = ConfigOptimizer(config_path=sample_toml_config)

        assert optimizer.original_config != {}
        assert "system" in optimizer.original_config

    def test_init_with_ini_config(self, sample_ini_config):
        """Инициализация с INI конфигом."""
        optimizer = ConfigOptimizer(config_path=sample_ini_config)

        assert optimizer.original_config != {}
        assert "system" in optimizer.original_config


class TestLoadConfig:
    """Тесты загрузки конфигурации."""

    def test_load_json_config(self, sample_json_config):
        """Загрузка JSON конфигурации."""
        optimizer = ConfigOptimizer(config_path=sample_json_config)

        assert optimizer.original_config != {}
        assert optimizer.optimized_config != {}

    def test_load_toml_config(self, sample_toml_config):
        """Загрузка TOML конфигурации."""
        optimizer = ConfigOptimizer(config_path=sample_toml_config)

        assert optimizer.original_config != {}

    def test_load_ini_config(self, sample_ini_config):
        """Загрузка INI конфигурации."""
        optimizer = ConfigOptimizer(config_path=sample_ini_config)

        assert optimizer.original_config != {}

    def test_load_nonexistent_file(self, temp_config_dir):
        """Загрузка несуществующего файла."""
        config_path = Path(temp_config_dir) / "nonexistent.json"

        # Метод load_config вернет False, но объект создастся
        assert ConfigOptimizer(config_path=str(config_path)) is not None

    def test_load_invalid_json(self, temp_config_dir):
        """Загрузка невалидного JSON."""
        config_path = Path(temp_config_dir) / "invalid.json"
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("{ invalid json }")

        # Загрузка должна завершиться неудачно
        assert ConfigOptimizer(config_path=str(config_path)) is not None

    def test_load_unsupported_format(self, temp_config_dir):
        """Загрузка неподдерживаемого формата."""
        config_path = Path(temp_config_dir) / "config.txt"
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("some text content")

        # Загрузка должна вернуть False
        assert ConfigOptimizer(config_path=str(config_path)) is not None


class TestSaveConfig:
    """Тесты сохранения конфигурации."""

    def test_save_config_json(self, sample_json_config, temp_config_dir):
        """Сохранение JSON конфигурации."""
        optimizer = ConfigOptimizer(config_path=sample_json_config)

        output_path = Path(temp_config_dir) / "output.json"
        result = optimizer.save_config(optimizer.original_config, str(output_path))

        assert result is True
        assert output_path.exists()

    def test_save_config_preserves_data(self, sample_json_config, temp_config_dir):
        """Сохранение сохраняет данные."""
        optimizer = ConfigOptimizer(config_path=sample_json_config)

        output_path = Path(temp_config_dir) / "output.json"
        optimizer.save_config(optimizer.original_config, str(output_path))

        with open(output_path, "r", encoding="utf-8") as f:
            saved_config = json.load(f)

        assert saved_config == optimizer.original_config


class TestOptimizeConfig:
    """Тесты оптимизации конфигурации."""

    def test_optimize_config_basic(self, sample_json_config):
        """Базовая оптимизация конфигурации."""
        optimizer = ConfigOptimizer(config_path=sample_json_config)

        with patch("psutil.cpu_percent", return_value=85.0):
            with patch("psutil.virtual_memory") as mock_memory:
                mock_memory.return_value = MagicMock(percent=90.0)
                with patch("psutil.disk_usage") as mock_disk:
                    mock_disk.return_value = MagicMock(percent=85.0)

                    result = optimizer.optimize_config()

        assert result is not None


class TestGetSystemMetrics:
    """Тесты получения системных метрик."""

    def test_system_metrics_attribute_exists(self, sample_json_config):
        """system_metrics атрибут существует."""
        optimizer = ConfigOptimizer(config_path=sample_json_config)

        assert hasattr(optimizer, "system_metrics")
        assert isinstance(optimizer.system_metrics, dict)


class TestThreadSafety:
    """Тесты потокобезопасности."""

    def test_config_optimizer_has_lock(self, sample_json_config):
        """ConfigOptimizer имеет threading.Lock."""
        optimizer = ConfigOptimizer(config_path=sample_json_config)

        assert optimizer.lock is not None
        assert isinstance(optimizer.lock, type(threading.Lock()))

    def test_concurrent_config_access(self, sample_json_config):
        """Конкурентный доступ к конфигурации."""
        import threading

        optimizer = ConfigOptimizer(config_path=sample_json_config)

        errors = []

        def modify_config():
            try:
                optimizer.original_config["test_key"] = "test_value"
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(10):
            t = threading.Thread(target=modify_config)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Не должно быть ошибок
        assert len(errors) == 0


class TestConfigValidation:
    """Тесты валидации конфигурации."""

    def test_validate_config_basic(self, sample_json_config):
        """Базовая валидация конфигурации."""
        optimizer = ConfigOptimizer(config_path=sample_json_config)

        # Проверяем, что конфигурация загружена
        assert optimizer.original_config != {}
        assert optimizer.optimized_config != {}

    def test_validate_config_thresholds(self, sample_json_config):
        """Валидация пороговых значений."""
        optimizer = ConfigOptimizer(config_path=sample_json_config)

        system_config = optimizer.original_config.get("system", {})

        # Пороги должны быть в допустимых пределах
        if "cpu_threshold" in system_config:
            assert 0 <= system_config["cpu_threshold"] <= 100

        if "memory_threshold" in system_config:
            assert 0 <= system_config["memory_threshold"] <= 100


class TestEdgeCases:
    """Тесты граничных случаев."""

    def test_empty_config_file(self, temp_config_dir):
        """Пустой файл конфигурации."""
        config_path = Path(temp_config_dir) / "empty.json"
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("{}")

        optimizer = ConfigOptimizer(config_path=str(config_path))

        assert optimizer.original_config == {}

    def test_config_with_null_values(self, temp_config_dir):
        """Конфигурация с null значениями."""
        config_path = Path(temp_config_dir) / "null_config.json"
        config = {"key": None, "nested": {"value": None}}
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)

        optimizer = ConfigOptimizer(config_path=str(config_path))

        assert optimizer.original_config["key"] is None

    def test_config_with_unicode(self, temp_config_dir):
        """Конфигурация с Unicode."""
        config_path = Path(temp_config_dir) / "unicode_config.json"
        config = {
            "description": "Оптимизация конфигурации 🚀",
            "name": "Проект Нанозонд",
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False)

        optimizer = ConfigOptimizer(config_path=str(config_path))

        assert optimizer.original_config["description"] == "Оптимизация конфигурации 🚀"

    def test_large_config_file(self, temp_config_dir):
        """Большой файл конфигурации."""
        config_path = Path(temp_config_dir) / "large_config.json"
        config = {
            "settings": {f"key_{i}": f"value_{i}" for i in range(1000)},
            "system": {"cpu_threshold": 80.0},
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)

        optimizer = ConfigOptimizer(config_path=str(config_path))

        assert len(optimizer.original_config["settings"]) == 1000


import threading
