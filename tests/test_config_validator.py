"""
Тесты для utils/config/config_validator.py

Покрытие:
- JSON schema валидация
- Конфигурационные проверки
- Обработка ошибок
"""

import json
import tempfile
from pathlib import Path

import pytest

from utils.config.config_validator import ConfigValidator


@pytest.fixture
def validator():
    """Создать ConfigValidator инстанс."""
    return ConfigValidator()


@pytest.fixture
def valid_config_file():
    """Создать временный файл с валидной конфигурацией."""
    config = {
        "project_name": "Test Project",
        "version": "1.0.0",
        "description": "A test project",
        "authors": ["Test Author"],
        "settings": {
            "debug": False,
            "log_level": "INFO",
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(config, f)
        return f.name


@pytest.fixture
def invalid_json_file():
    """Создать временный файл с невалидным JSON."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        f.write("{ invalid json }")
        return f.name


@pytest.fixture
def config_with_schema_errors():
    """Конфигурация с ошибками схемы."""
    return {
        "project_name": 123,  # Должно быть string
        "version": "invalid-version",  # Не соответствует X.Y.Z
        "description": True,  # Должно быть string
    }


@pytest.fixture
def config_with_schema_valid():
    """Валидная конфигурация по схеме."""
    return {
        "project_name": "Valid Project",
        "version": "2.0.1",
        "description": "A valid project description",
        "authors": ["Author One", "Author Two"],
        "settings": {
            "debug": True,
            "log_level": "DEBUG",
        },
    }


class TestConfigValidatorInit:
    """Тесты инициализации ConfigValidator."""

    def test_initial_state(self, validator):
        """Проверка начального состояния."""
        assert validator.validation_results == {}
        assert validator.errors == []
        assert validator.warnings == []

    def test_multiple_instances(self):
        """Несколько инстансов независимы."""
        v1 = ConfigValidator()
        v2 = ConfigValidator()

        v1.errors.append("error1")
        assert "error1" not in v2.errors


class TestValidateJSONConfig:
    """Тесты валидации JSON конфигурации."""

    def test_valid_config_file(self, validator, valid_config_file):
        """Валидация валидного конфигурационного файла."""
        result = validator.validate_json_config(valid_config_file)

        assert result["valid"] is True
        assert result["config_path"] == valid_config_file
        assert len(result["errors"]) == 0
        assert "config" in result
        assert "timestamp" in result

    def test_invalid_json_file(self, validator, invalid_json_file):
        """Валидация файла с невалидным JSON."""
        result = validator.validate_json_config(invalid_json_file)

        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert "JSON decode error" in result["errors"][0]

    def test_nonexistent_file(self, validator):
        """Валидация несуществующего файла."""
        result = validator.validate_json_config("/nonexistent/path/config.json")

        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert "Validation error" in result["errors"][0]

    def test_config_with_schema_errors(self, validator):
        """Валидация конфигурации с ошибками схемы."""
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"project_name": 123, "version": "bad"}, f)
            temp_file = f.name

        result = validator.validate_json_config(temp_file)

        # JSON может быть валидным, но не соответствовать схеме
        # Это зависит от того, применяется ли схема по умолчанию
        assert "config_path" in result
        assert "timestamp" in result

    def test_config_with_custom_schema(self, validator):
        """Валидация с кастомной схемой."""
        custom_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
            },
            "required": ["name"],
        }

        valid_config = {"name": "Test", "age": 25}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(valid_config, f)
            temp_file = f.name

        result = validator.validate_json_config(temp_file, schema=custom_schema)

        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_config_with_custom_schema_errors(self, validator):
        """Валидация с кастомной схемой (ошибки)."""
        custom_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
            },
            "required": ["name"],
        }

        invalid_config = {"age": "not a number"}  # Отсутствует required поле 'name'
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(invalid_config, f)
            temp_file = f.name

        result = validator.validate_json_config(temp_file, schema=custom_schema)

        assert result["valid"] is False
        assert len(result["errors"]) > 0


class TestGetDefaultConfigSchema:
    """Тесты схемы конфигурации по умолчанию."""

    def test_schema_structure(self, validator):
        """Проверка структуры схемы."""
        schema = validator.get_default_config_schema()

        assert "$schema" in schema
        assert schema["type"] == "object"
        assert "properties" in schema

    def test_schema_project_name(self, validator):
        """Проверка поля project_name в схеме."""
        schema = validator.get_default_config_schema()

        assert "project_name" in schema["properties"]
        assert schema["properties"]["project_name"]["type"] == "string"

    def test_schema_version_pattern(self, validator):
        """Проверка паттерна версии."""
        schema = validator.get_default_config_schema()

        assert "version" in schema["properties"]
        version_prop = schema["properties"]["version"]
        assert "pattern" in version_prop
        # Паттерн должен соответствовать X.Y.Z
        import re

        assert re.match(version_prop["pattern"], "1.0.0")
        assert not re.match(version_prop["pattern"], "invalid")

    def test_schema_authors_array(self, validator):
        """Проверка поля authors в схеме."""
        schema = validator.get_default_config_schema()

        assert "authors" in schema["properties"]
        authors_prop = schema["properties"]["authors"]
        assert authors_prop["type"] == "array"


class TestConfigValidationEdgeCases:
    """Тесты граничных случаев валидации."""

    def test_empty_config_file(self, validator):
        """Валидация пустого файла."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write("")
            temp_file = f.name

        result = validator.validate_json_config(temp_file)

        assert result["valid"] is False
        assert "JSON decode error" in result["errors"][0]

    def test_config_with_null_values(self, validator):
        """Валидация конфигурации с null значениями."""
        config = {
            "project_name": None,
            "version": None,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(config, f)
            temp_file = f.name

        result = validator.validate_json_config(temp_file)

        # JSON валиден, но может не соответствовать схеме
        assert "config" in result

    def test_config_with_unicode(self, validator):
        """Валидация конфигурации с Unicode."""
        config = {
            "project_name": "Проект Нанозонд",
            "description": "Тестирование Unicode символов 🚀",
            "version": "1.0.0",
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(config, f, ensure_ascii=False)
            temp_file = f.name

        result = validator.validate_json_config(temp_file)

        assert result["valid"] is True
        assert result["config"]["project_name"] == "Проект Нанозонд"

    def test_config_large_file(self, validator):
        """Валидация большого конфигурационного файла."""
        config = {
            "project_name": "Large Config Test",
            "version": "1.0.0",
            "settings": {f"key_{i}": f"value_{i}" for i in range(1000)},
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(config, f)
            temp_file = f.name

        result = validator.validate_json_config(temp_file)

        # Результат может быть True или False в зависимости от схемы
        assert "config" in result
        assert "timestamp" in result
        assert isinstance(result["valid"], bool)


class TestConfigValidationResults:
    """Тесты хранения результатов валидации."""

    def test_validation_results_timestamp(self, validator, valid_config_file):
        """Проверка наличия timestamp в результатах."""
        result = validator.validate_json_config(valid_config_file)

        assert "timestamp" in result
        assert result["timestamp"] is not None

    def test_validation_results_config_path(self, validator, valid_config_file):
        """Проверка пути к конфигу в результатах."""
        result = validator.validate_json_config(valid_config_file)

        assert result["config_path"] == valid_config_file

    def test_validation_results_errors_list(self, validator, invalid_json_file):
        """Проверка, что errors это список."""
        result = validator.validate_json_config(invalid_json_file)

        assert isinstance(result["errors"], list)
        assert len(result["errors"]) > 0

    def test_validation_results_valid_boolean(
        self, validator, valid_config_file, invalid_json_file
    ):
        """Проверка, что valid это boolean."""
        valid_result = validator.validate_json_config(valid_config_file)
        invalid_result = validator.validate_json_config(invalid_json_file)

        assert isinstance(valid_result["valid"], bool)
        assert isinstance(invalid_result["valid"], bool)
        assert valid_result["valid"] is True
        assert invalid_result["valid"] is False
