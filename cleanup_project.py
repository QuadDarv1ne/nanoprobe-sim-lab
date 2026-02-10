#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт очистки и организации проекта Nanoprobe Simulation Lab
Этот скрипт выполняет комплексную очистку и организацию проекта.
"""

import os
import shutil
import json
from datetime import datetime
from pathlib import Path
import re
import tempfile
from typing import Dict, List, Tuple


class ProjectCleaner:
    """
    Класс для очистки и организации проекта
    """

    def __init__(self):
        """Инициализация очистки проекта"""
        self.project_root = Path(__file__).parent
        self.log_messages = []
        self.changes_made = []

    def log_message(self, message: str, level: str = "INFO"):
        """Логирование сообщений"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message
        }
        self.log_messages.append(log_entry)
        print(f"[{level}] {timestamp}: {message}")

    def remove_temporary_files(self) -> int:
        """Удаление временных файлов и директорий"""
        self.log_message("Удаление временных файлов...")

        # Паттерны для удаления
        temp_patterns = [
            "**/__pycache__",
            "**/*.pyc",
            "**/.pytest_cache",
            "**/__pycache__/",
            "**/*~",
            "**/Thumbs.db",
            "**/.DS_Store",
            "temp/",
            "cache/",
            ".cache/",
            "**/*.tmp",
            "**/*.temp",
            "build/temp.*",
            "**/.coverage",
            "**/htmlcov/"
        ]

        removed_count = 0

        for pattern in temp_patterns:
            for path in self.project_root.glob(pattern):
                try:
                    if path.is_file():
                        path.unlink()
                        removed_count += 1
                        self.changes_made.append({
                            "file": str(path),
                            "change": "Removed temporary file"
                        })
                    elif path.is_dir():
                        shutil.rmtree(path)
                        removed_count += 1
                        self.changes_made.append({
                            "file": str(path),
                            "change": "Removed temporary directory"
                        })
                except (OSError, PermissionError) as e:
                    self.log_message(f"Ошибка удаления {path}: {str(e)}", "WARNING")

        self.log_message(f"Удалено временных файлов/директорий: {removed_count}")
        return removed_count

    def organize_directories(self) -> Dict[str, int]:
        """Организация структуры директорий"""
        self.log_message("Организация структуры директорий...")

        # Создаем основные директории если они не существуют
        dirs_to_create = [
            "data/raw",
            "data/processed",
            "data/output",
            "reports/logs",
            "reports/analytics",
            "reports/health",
            "tests/unit",
            "tests/integration",
            "docs/api",
            "docs/user_guide",
            "backup/configs",
            "profiles/memory",
            "profiles/performance"
        ]

        created_dirs = 0
        for dir_path in dirs_to_create:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                full_path.mkdir(parents=True, exist_ok=True)
                created_dirs += 1
                self.changes_made.append({
                    "file": str(full_path),
                    "change": "Created directory"
                })

        # Перемещаем файлы в соответствующие директории
        moved_files = 0

        # Перемещаем профили в правильную директорию
        profiles_to_move = list((self.project_root / "profiles").glob("*"))
        for profile_file in profiles_to_move:
            if profile_file.is_file():  # Обрабатываем только файлы, не директории
                if "memory" in str(profile_file) or profile_file.suffix in ['.txt']:
                    target_dir = self.project_root / "profiles" / "memory"
                elif "profile" in str(profile_file) or "performance" in str(profile_file):
                    target_dir = self.project_root / "profiles" / "performance"
                else:
                    continue

                target_path = target_dir / profile_file.name
                if profile_file != target_path:
                    target_dir.mkdir(parents=True, exist_ok=True)  # Убедимся, что целевая директория существует
                    shutil.move(str(profile_file), str(target_path))
                    moved_files += 1
                    self.changes_made.append({
                        "file": str(profile_file),
                        "change": f"Moved to {target_path}"
                    })

        # Перемещаем отчеты
        reports_to_move = []
        for ext in ['.json', '.txt', '.log']:
            reports_to_move.extend(list(self.project_root.glob(f"*.{ext}")))

        for report_file in reports_to_move:
            if report_file.is_file():  # Обрабатываем только файлы, не директории
                if "monitoring" in str(report_file) or "health" in str(report_file):
                    target_dir = self.project_root / "reports" / "health"
                elif "analytics" in str(report_file):
                    target_dir = self.project_root / "reports" / "analytics"
                elif "log" in str(report_file) or report_file.name.startswith("log"):
                    target_dir = self.project_root / "reports" / "logs"
                else:
                    continue

                target_path = target_dir / report_file.name
                if report_file != target_path:
                    target_dir.mkdir(parents=True, exist_ok=True)  # Убедимся, что целевая директория существует
                    shutil.move(str(report_file), str(target_path))
                    moved_files += 1
                    self.changes_made.append({
                        "file": str(report_file),
                        "change": f"Moved to {target_path}"
                    })

        result = {
            "created_dirs": created_dirs,
            "moved_files": moved_files
        }

        self.log_message(f"Создано директорий: {created_dirs}, перемещено файлов: {moved_files}")
        return result

    def fix_file_permissions(self) -> int:
        """Исправление прав доступа к файлам"""
        self.log_message("Исправление прав доступа к файлам...")

        fixed_count = 0
        # На Windows просто убедимся, что файлы доступны для записи
        for py_file in self.project_root.rglob("*.py"):
            try:
                # Проверяем, можно ли открыть файл на чтение
                with open(py_file, 'r', encoding='utf-8'):
                    pass
                fixed_count += 1
            except (PermissionError, OSError):
                self.log_message(f"Проблема с доступом к файлу: {py_file}", "WARNING")

        self.log_message(f"Проверено файлов: {fixed_count}")
        return fixed_count

    def validate_project_structure(self) -> Dict:
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
            "templates"
        ]

        missing_dirs = []
        for dir_name in expected_dirs:
            if not (self.project_root / dir_name).exists():
                missing_dirs.append(dir_name)

        expected_files = [
            "start.py",
            "README.md",
            "requirements.txt",
            "src/cli/main.py",
            "src/cli/project_manager.py",
            "src/web/web_dashboard.py"
        ]

        missing_files = []
        for file_name in expected_files:
            if not (self.project_root / file_name).exists():
                missing_files.append(file_name)

        result = {
            "missing_directories": missing_dirs,
            "missing_files": missing_files,
            "structure_valid": len(missing_dirs) == 0 and len(missing_files) == 0
        }

        if missing_dirs:
            self.log_message(f"Отсутствующие директории: {missing_dirs}", "WARNING")
        if missing_files:
            self.log_message(f"Отсутствующие файлы: {missing_files}", "WARNING")

        if result["structure_valid"]:
            self.log_message("Структура проекта в порядке", "INFO")
        else:
            self.log_message("Обнаружены проблемы со структурой проекта", "WARNING")

        return result

    def optimize_config_files(self) -> int:
        """Оптимизация конфигурационных файлов"""
        self.log_message("Оптимизация конфигурационных файлов...")

        config_files = [
            self.project_root / "config" / "cache_config.json",
            self.project_root / "config" / "config.json",
            self.project_root / "config" / "optimization_config.json"
        ]

        optimized_configs = 0
        for config_file in config_files:
            if config_file.exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)

                    # Форматируем JSON красиво
                    formatted_config = json.dumps(config_data, indent=2, ensure_ascii=False)

                    with open(config_file, 'w', encoding='utf-8') as f:
                        f.write(formatted_config)

                    optimized_configs += 1
                    self.changes_made.append({
                        "file": str(config_file),
                        "change": "Optimized configuration file format"
                    })
                except (json.JSONDecodeError, OSError) as e:
                    self.log_message(f"Ошибка обработки конфига {config_file}: {str(e)}", "ERROR")

        self.log_message(f"Оптимизировано конфигурационных файлов: {optimized_configs}")
        return optimized_configs

    def clean_up_duplicate_scripts(self) -> int:
        """Очистка дублирующихся скриптов и улучшений"""
        self.log_message("Очистка дублирующихся скриптов...")

        # Найдем потенциальные дубликаты улучшений
        improvement_scripts = [
            "improve_project.py",
            "optimize_all.py",
            "test_optimizations.py",
            "run_monitoring_and_improvements.py"
        ]

        cleaned_count = 0
        for script_name in improvement_scripts:
            script_path = self.project_root / script_name
            if script_path.exists():
                # Проверим размер файла - если он больше 10KB, возможно это полноценный скрипт
                if script_path.stat().st_size > 10 * 1024:  # > 10KB
                    self.log_message(f"Найден большой скрипт улучшения: {script_name} ({script_path.stat().st_size} bytes)")

        # Найдем дубликаты в deployment
        deployment_dirs = list(self.project_root.glob("deployment/nanoprobe-lab-*"))
        if len(deployment_dirs) > 1:
            self.log_message(f"Найдено несколько директорий развертывания: {len(deployment_dirs)}", "INFO")
            # Оставим только самую новую, остальные пометим как устаревшие
            sorted_dirs = sorted(deployment_dirs, key=lambda x: x.name, reverse=True)
            for old_dir in sorted_dirs[1:]:
                self.log_message(f"Устаревшая директория развертывания: {old_dir.name}", "INFO")

        return cleaned_count

    def run_comprehensive_cleanup(self) -> Dict:
        """Запуск комплексной очистки проекта"""
        self.log_message("="*60, "INFO")
        self.log_message("ЗАПУСК КОМПЛЕКСНОЙ ОЧИСТКИ ПРОЕКТА", "INFO")
        self.log_message("="*60, "INFO")

        # Выполняем все этапы очистки
        temp_removed = self.remove_temporary_files()
        dir_result = self.organize_directories()
        perms_fixed = self.fix_file_permissions()
        structure_valid = self.validate_project_structure()
        configs_optimized = self.optimize_config_files()
        duplicates_cleaned = self.clean_up_duplicate_scripts()

        # Сводка
        self.log_message("="*60, "INFO")
        self.log_message("СВОДКА ОЧИСТКИ", "INFO")
        self.log_message("="*60, "INFO")

        self.log_message(f"Удалено временных файлов/директорий: {temp_removed}", "INFO")
        self.log_message(f"Создано директорий: {dir_result['created_dirs']}", "INFO")
        self.log_message(f"Перемещено файлов: {dir_result['moved_files']}", "INFO")
        self.log_message(f"Проверено файлов: {perms_fixed}", "INFO")
        self.log_message(f"Оптимизировано конфигураций: {configs_optimized}", "INFO")
        self.log_message(f"Обработано дубликатов: {duplicates_cleaned}", "INFO")
        self.log_message(f"Всего изменений: {len(self.changes_made)}", "INFO")

        # Сохраняем отчет
        self.save_cleanup_report()

        return {
            "temp_files_removed": temp_removed,
            "directories_created": dir_result['created_dirs'],
            "files_moved": dir_result['moved_files'],
            "permissions_fixed": perms_fixed,
            "structure_validation": structure_valid,
            "configs_optimized": configs_optimized,
            "duplicates_handled": duplicates_cleaned,
            "changes_made": self.changes_made
        }

    def save_cleanup_report(self):
        """Сохранение отчета об очистке"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.project_root / "reports" / "logs" / f"cleanup_report_{timestamp}.json"

        # Создаем папку отчетов если не существует
        report_path.parent.mkdir(parents=True, exist_ok=True)

        report_data = {
            "timestamp": datetime.now().isoformat(),
            "project_name": "Nanoprobe Simulation Lab",
            "cleanup_operation": "Comprehensive project cleanup and organization",
            "cleanup_log": self.log_messages,
            "changes_made": self.changes_made,
            "summary": {
                "total_changes": len(self.changes_made),
                "total_logs": len(self.log_messages),
                "errors": len([log for log in self.log_messages if log['level'] == 'ERROR']),
                "warnings": len([log for log in self.log_messages if log['level'] == 'WARNING']),
                "infos": len([log for log in self.log_messages if log['level'] == 'INFO'])
            }
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        self.log_message(f"Отчет об очистке сохранен: {report_path}", "INFO")


def main():
    """Основная функция запуска очистки"""
    print("Запуск комплексной очистки и организации проекта Nanoprobe Simulation Lab...")

    cleaner = ProjectCleaner()
    results = cleaner.run_comprehensive_cleanup()

    print("\nОчистка завершена!")
    print(f"Всего внесено изменений: {len(results['changes_made'])}")
    print(f"Удалено временных файлов: {results['temp_files_removed']}")
    print(f"Создано директорий: {results['directories_created']}")
    print(f"Перемещено файлов: {results['files_moved']}")

    print("\nПервые 5 изменений:")
    for i, change in enumerate(results['changes_made'][:5], 1):
        print(f"{i}. {change['file']}: {change['change']}")

    if len(results['changes_made']) > 5:
        print(f"... и еще {len(results['changes_made']) - 5} изменений")

    print("\nСтруктура проекта проверена.")
    if results['structure_validation']['structure_valid']:
        print("✓ Структура проекта корректна")
    else:
        print("⚠ Обнаружены проблемы со структурой проекта:")
        if results['structure_validation']['missing_directories']:
            print(f"  - Отсутствующие директории: {results['structure_validation']['missing_directories']}")
        if results['structure_validation']['missing_files']:
            print(f"  - Отсутствующие файлы: {results['structure_validation']['missing_files']}")


if __name__ == "__main__":
    main()
