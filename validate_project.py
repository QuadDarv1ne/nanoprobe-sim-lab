"""Скрипт валидации проекта Nanoprobe Simulation Lab."""

import os
import sys
import json
import subprocess
import io
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import importlib.util

# Исправление кодировки для Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class ProjectValidator:
    """
    Класс для валидации проекта
    """

    def __init__(self):
        """Инициализация валидатора проекта"""
        self.project_root = Path(__file__).parent
        self.log_messages = []
        self.validation_results = []
        self.errors_found = []
        self.warnings_found = []

    def log_message(self, message: str, level: str = "INFO"):
        """Логирование сообщений"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message
        }
        self.log_messages.append(log_entry)
        # Замена Unicode символов на ASCII аналоги для Windows
        safe_message = message.replace('✓', '[OK]').replace('✗', '[ERR]').replace(
            '⚠', '[WARN]').replace('🎉', '[DONE]').replace('❌', '[FAIL]')
        print(f"[{level}] {timestamp}: {safe_message}")

    def check_project_structure(self) -> Dict[str, Any]:
        """Проверка структуры проекта"""
        self.log_message("Проверка структуры проекта...")

        expected_dirs = [
            "src",
            "src/cli",
            "src/web",
            "src/core",
            "components",
            "utils",
            "config",
            "tests",
            "docs",
            "data",
            "logs",
            "output",
            "templates",
            "reports",
            "reports/logs",
            "profiles",
            "profiles/memory",
            "profiles/performance"
        ]

        expected_files = [
            "start.py",
            "README.md",
            "requirements.txt",
            "src/cli/main.py",
            "src/cli/project_manager.py",
            "src/web/web_dashboard.py"
        ]

        missing_dirs = []
        missing_files = []

        for dir_name in expected_dirs:
            if not (self.project_root / dir_name).exists():
                missing_dirs.append(dir_name)

        for file_name in expected_files:
            if not (self.project_root / file_name).exists():
                missing_files.append(file_name)

        result = {
            "missing_directories": missing_dirs,
            "missing_files": missing_files,
            "structure_valid": len(missing_dirs) == 0 and len(missing_files) == 0,
            "total_directories_checked": len(expected_dirs),
            "total_files_checked": len(expected_files)
        }

        if missing_dirs:
            self.log_message(f"Отсутствующие директории: {missing_dirs}", "WARNING")
            self.warnings_found.extend([f"Missing directory: {d}" for d in missing_dirs])
        if missing_files:
            self.log_message(f"Отсутствующие файлы: {missing_files}", "WARNING")
            self.warnings_found.extend([f"Missing file: {f}" for f in missing_files])

        if result["structure_valid"]:
            self.log_message("✓ Структура проекта в порядке", "INFO")
        else:
            self.log_message("⚠ Обнаружены проблемы со структурой проекта", "WARNING")

        return result

    def check_python_syntax(self) -> Dict[str, Any]:
        """Проверка синтаксиса Python файлов"""
        self.log_message("Проверка синтаксиса Python файлов...")

        # Исключаем виртуальное окружение и другие ненужные директории
        excluded_dirs = {'venv', '.venv', 'env', '__pycache__', '.git', '.pytest_cache', '.vscode', '.idea'}
        python_files = []

        for py_file in self.project_root.rglob("*.py"):
            # Проверяем, не находится ли файл в исключенной директории
            is_excluded = False
            for parent in py_file.parents:
                if parent.name in excluded_dirs:
                    is_excluded = True
                    break
            if not is_excluded:
                python_files.append(py_file)

        total_files = len(python_files)
        valid_files = 0
        invalid_files = []

        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Проверяем синтаксис
                compile(content, str(py_file), 'exec')
                valid_files += 1
            except SyntaxError as e:
                invalid_files.append({
                    "file": str(py_file),
                    "error": str(e),
                    "line": e.lineno,
                    "text": e.text
                })
                self.log_message(f"Синтаксическая ошибка в {py_file}:{e.lineno}: {e.msg}", "ERROR")
                self.errors_found.append(f"Syntax error in {py_file}: {e.msg}")
            except Exception as e:
                self.log_message(f"Ошибка чтения файла {py_file}: {str(e)}", "ERROR")
                self.errors_found.append(f"Read error in {py_file}: {str(e)}")

        result = {
            "total_files": total_files,
            "valid_files": valid_files,
            "invalid_files": invalid_files,
            "syntax_valid": len(invalid_files) == 0
        }

        self.log_message(f"Проверено файлов: {total_files}, валидных: {valid_files}, ошибок: {len(invalid_files)}")

        return result

    def check_imports(self) -> Dict[str, Any]:
        """Проверка импортов в Python файлах"""
        self.log_message("Проверка импортов...")

        # Исключаем виртуальное окружение и другие ненужные директории
        excluded_dirs = {'venv', '.venv', 'env', '__pycache__', '.git', '.pytest_cache', '.vscode', '.idea'}
        python_files = []

        for py_file in self.project_root.rglob("*.py"):
            # Проверяем, не находится ли файл в исключенной директории
            is_excluded = False
            for parent in py_file.parents:
                if parent.name in excluded_dirs:
                    is_excluded = True
                    break
            if not is_excluded:
                python_files.append(py_file)

        total_files = len(python_files)
        importable_files = 0
        unimportable_files = []

        for py_file in python_files:
            try:
                # Проверяем возможность импорта модуля
                spec = importlib.util.spec_from_file_location("temp_module", py_file)
                if spec and spec.loader:
                    # Добавляем директорию в sys.path для правильного импорта
                    file_dir = str(py_file.parent)
                    if file_dir not in sys.path:
                        sys.path.insert(0, file_dir)

                    # Пытаемся создать модуль (без выполнения)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    importable_files += 1
                else:
                    unimportable_files.append({
                        "file": str(py_file),
                        "error": "Could not create module spec"
                    })
                    self.log_message(f"Невозможно импортировать {py_file}", "WARNING")
                    self.warnings_found.append(f"Import error in {py_file}")
            except ImportError as e:
                unimportable_files.append({
                    "file": str(py_file),
                    "error": str(e)
                })
                self.log_message(f"Ошибка импорта в {py_file}: {str(e)}", "WARNING")
                self.warnings_found.append(f"Import error in {py_file}: {str(e)}")
            except Exception as e:
                unimportable_files.append({
                    "file": str(py_file),
                    "error": str(e)
                })
                self.log_message(f"Ошибка загрузки модуля {py_file}: {str(e)}", "WARNING")
                self.warnings_found.append(f"Module load error in {py_file}: {str(e)}")

        result = {
            "total_files": total_files,
            "importable_files": importable_files,
            "unimportable_files": unimportable_files,
            "imports_valid": len(unimportable_files) == 0
        }

        self.log_message(f"Файлов с успешными импортами: {importable_files}, проблемных: {len(unimportable_files)}")

        return result

    def check_requirements(self) -> Dict[str, Any]:
        """Проверка соответствия requirements.txt"""
        self.log_message("Проверка зависимостей...")

        req_file = self.project_root / "requirements.txt"
        if not req_file.exists():
            result = {
                "requirements_exists": False,
                "dependencies_valid": False,
                "message": "Файл requirements.txt не найден"
            }
            self.log_message("Файл requirements.txt не найден", "WARNING")
            self.warnings_found.append("requirements.txt not found")
            return result

        # Читаем зависимости
        with open(req_file, 'r', encoding='utf-8') as f:
            requirements = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]

        # Проверяем, установлены ли зависимости
        try:
            installed_packages = subprocess.check_output([sys.executable, '-m', 'pip', 'list'], text=True)
        except subprocess.CalledProcessError:
            installed_packages = ""

        missing_deps = []
        for req in requirements:
            # Извлекаем имя пакета (до знака >=, <=, == и т.д.)
            pkg_name = req.split('>=')[0].split('<=')[0].split('==')[0].split('>')[0].split('<')[0].strip()
            if pkg_name and pkg_name.lower() not in installed_packages.lower():
                missing_deps.append(pkg_name)

        result = {
            "requirements_exists": True,
            "dependencies_valid": len(missing_deps) == 0,
            "missing_dependencies": missing_deps,
            "total_dependencies": len(requirements)
        }

        if missing_deps:
            self.log_message(f"Отсутствующие зависимости: {missing_deps}", "WARNING")
            self.warnings_found.extend([f"Missing dependency: {dep}" for dep in missing_deps])
        else:
            self.log_message(f"Все зависимости ({len(requirements)}) установлены", "INFO")

        return result

    def check_main_components(self) -> Dict[str, Any]:
        """Проверка основных компонентов проекта"""
        self.log_message("Проверка основных компонентов...")

        components_to_check = [
            ("start.py", "Главный запуск"),
            ("src/cli/main.py", "Консольный интерфейс"),
            ("src/cli/project_manager.py", "Менеджер проекта"),
            ("src/web/web_dashboard.py", "Веб-панель"),
            ("utils/cache_manager.py", "Менеджер кэша"),
            ("utils/config_manager.py", "Менеджер конфигов")
        ]

        working_components = 0
        broken_components = []

        for component_path, description in components_to_check:
            full_path = self.project_root / component_path
            if full_path.exists():
                working_components += 1
                self.log_message(f"✓ {description} ({component_path}) - OK", "INFO")
            else:
                broken_components.append({
                    "path": component_path,
                    "description": description
                })
                self.log_message(f"✗ {description} ({component_path}) - НЕ НАЙДЕН", "ERROR")
                self.errors_found.append(f"Missing component: {component_path}")

        result = {
            "working_components": working_components,
            "broken_components": broken_components,
            "components_valid": len(broken_components) == 0,
            "total_components": len(components_to_check)
        }

        return result

    def run_all_validations(self) -> Dict[str, Any]:
        """Запуск всех проверок проекта"""
        self.log_message("="*60, "INFO")
        self.log_message("ЗАПУСК ВАЛИДАЦИИ ПРОЕКТА", "INFO")
        self.log_message("="*60, "INFO")

        # Выполняем все проверки
        structure_result = self.check_project_structure()
        syntax_result = self.check_python_syntax()
        import_result = self.check_imports()
        requirements_result = self.check_requirements()
        components_result = self.check_main_components()

        # Сводка
        self.log_message("="*60, "INFO")
        self.log_message("СВОДКА ВАЛИДАЦИИ", "INFO")
        self.log_message("="*60, "INFO")

        self.log_message(f"Структура проекта: {'✓' if structure_result['structure_valid'] else '✗'}")
        self.log_message(f"Синтаксис Python: {'✓' if syntax_result['syntax_valid'] else '✗'} (валидных: {syntax_result['valid_files']}/{syntax_result['total_files']})")
        self.log_message(f"Импорты: {'✓' if import_result['imports_valid'] else '✗'} (импортируемых: {import_result['importable_files']}/{import_result['total_files']})")
        self.log_message(f"Зависимости: {'✓' if requirements_result['dependencies_valid'] else '✗'} (отсутствует: {len(requirements_result['missing_dependencies'])})")
        self.log_message(f"Компоненты: {'✓' if components_result['components_valid'] else '✗'} (работающих: {components_result['working_components']}/{components_result['total_components']})")

        # Подсчет ошибок и предупреждений
        total_errors = len(self.errors_found)
        total_warnings = len(self.warnings_found)

        self.log_message(f"Всего ошибок: {total_errors}")
        self.log_message(f"Всего предупреждений: {total_warnings}")

        # Оценка общего состояния
        all_checks_passed = (
            structure_result['structure_valid'] and
            syntax_result['syntax_valid'] and
            import_result['imports_valid'] and
            requirements_result['dependencies_valid'] and
            components_result['components_valid']
        )

        if all_checks_passed and total_errors == 0:
            self.log_message("🎉 Проект полностью валиден!", "INFO")
        elif total_errors == 0:
            self.log_message("⚠ Проект валиден с предупреждениями", "INFO")
        else:
            self.log_message("❌ Проект имеет ошибки валидации", "ERROR")

        # Сохраняем результаты
        validation_data = {
            "timestamp": datetime.now().isoformat(),
            "project_name": "Nanoprobe Simulation Lab",
            "validation_results": {
                "structure": structure_result,
                "syntax": syntax_result,
                "imports": import_result,
                "requirements": requirements_result,
                "components": components_result
            },
            "summary": {
                "all_checks_passed": all_checks_passed,
                "total_errors": total_errors,
                "total_warnings": total_warnings,
                "errors": self.errors_found,
                "warnings": self.warnings_found
            },
            "validation_log": self.log_messages
        }

        self.save_validation_report(validation_data)

        return validation_data

    def save_validation_report(self, validation_data: Dict[str, Any]):
        """Сохранение отчета о валидации"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.project_root / "reports" / "logs" / f"validation_report_{timestamp}.json"

        # Создаем папку отчетов если не существует
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(validation_data, f, indent=2, ensure_ascii=False)

        self.log_message(f"Отчет о валидации сохранен: {report_path}", "INFO")


def main():
    """Основная функция запуска валидации"""
    print("Запуск валидации проекта Nanoprobe Simulation Lab...")

    validator = ProjectValidator()
    results = validator.run_all_validations()

    print("\nВалидация завершена!")
    print(f"Всего ошибок: {results['summary']['total_errors']}")
    print(f"Всего предупреждений: {results['summary']['total_warnings']}")

    if results['summary']['total_errors'] == 0:
        print("✅ Проект успешно прошел валидацию!")
    else:
        print("❌ Проект имеет ошибки, требующие внимания.")

    if results['summary']['total_warnings'] > 0:
        print(f"⚠ Найдено {results['summary']['total_warnings']} предупреждений")

    print(f"\nДетали валидации сохранены в: reports/logs/validation_report_*.json")


if __name__ == "__main__":
    main()
