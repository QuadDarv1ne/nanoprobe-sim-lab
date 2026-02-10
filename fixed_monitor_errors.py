# -*- coding: utf-8 -*-
#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3

"""
Исправленный скрипт мониторинга ошибок и улучшения проекта Nanoprobe Simulation Lab
Этот скрипт позволяет отслеживать ошибки, производительность и
предоставляет рекомендации по улучшению проекта.
Безопасная версия, не зависящая от поврежденных пакетов.
"""

import os
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
import subprocess
import time
from typing import Dict, List, Any

# Добавляем путь к проекту
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Попробовать импортировать зависимости, используя только встроенные модули
# если стандартные утилиты недоступны


class ProjectMonitor:
    """
    Класс для мониторинга проекта, отслеживания ошибок и улучшения системы
    """

    def __init__(self):
        """Инициализация мониторинга проекта"""
        self.monitoring_log = []

    def log_message(self, message: str, level: str = "INFO"):
        """Логирование сообщений"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message
        }
        self.monitoring_log.append(log_entry)
        print(f"[{level}] {timestamp}: {message}")

    def check_system_health(self):
        """Проверка общего состояния системы"""
        self.log_message("Проверка состояния системы...")

        try:
            # Проверка основных метрик системы без использования psutil
            import platform
            import multiprocessing

            system_info = {
                "os": platform.system(),
                "platform": platform.platform(),
                "processor": platform.processor(),
                "architecture": platform.architecture()[0],
                "cpu_count": multiprocessing.cpu_count(),
                "python_version": platform.python_version(),
            }

            self.log_message(f"ОС: {system_info['os']}", "INFO")
            self.log_message(f"Архитектура: {system_info['architecture']}", "INFO")
            self.log_message(f"Количество CPU: {system_info['cpu_count']}", "INFO")

            # Проверка наличия необходимых директорий
            essential_dirs = ['src', 'components', 'config', 'utils', 'templates']
            missing_dirs = []

            for dir_name in essential_dirs:
                dir_path = project_root / dir_name
                if not dir_path.exists():
                    missing_dirs.append(str(dir_path))

            if missing_dirs:
                for missing_dir in missing_dirs:
                    self.log_message(f"Отсутствует директория: {missing_dir}", "WARNING")
            else:
                self.log_message("Все основные директории на месте", "INFO")

            # Оценка состояния системы
            health_score = 100
            if missing_dirs:
                health_score -= len(missing_dirs) * 10  # 10 баллов за каждую отсутствующую директорию

            health_score = max(0, health_score)  # Минимум 0

            self.log_message(f"Состояние системы: {health_score}/100",
                           "INFO" if health_score >= 70 else "WARNING")

            return {"health_score": health_score, "system_info": system_info}
        except Exception as e:
            self.log_message(f"Ошибка проверки состояния системы: {str(e)}", "ERROR")
            return None

    def analyze_code_quality(self):
        """Анализ качества кода"""
        self.log_message("Анализ качества кода...")

        try:
            # Поиск Python файлов в проекте
            python_files = list(project_root.rglob("*.py"))
            # Исключаем виртуальное окружение и временные директории
            python_files = [f for f in python_files if 'venv' not in str(f) and '__pycache__' not in str(f)]

            self.log_message(f"Найдено Python файлов для анализа: {len(python_files)}", "INFO")

            issues = []
            total_lines = 0

            # Простой анализ - поиск потенциальных проблем
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        total_lines += len(lines)

                        for i, line in enumerate(lines, 1):
                            # Проверка на потенциальные проблемы
                            stripped_line = line.strip()

                            # Проверка на отступы и табуляции
                            if '\t' in line[:line.index(stripped_line[0]) if stripped_line else len(line)]:
                                issues.append({
                                    "file_path": str(file_path),
                                    "line_number": i,
                                    "message": "Обнаружен символ табуляции, используйте пробелы",
                                    "severity": "WARNING"
                                })

                            # Проверка на слишком длинные строки
                            if len(line) > 120:
                                issues.append({
                                    "file_path": str(file_path),
                                    "line_number": i,
                                    "message": f"Слишком длинная строка ({len(line)} символов)",
                                    "severity": "WARNING"
                                })

                            # Проверка на print() в продакшен коде
                            if 'print(' in stripped_line and 'print(' in stripped_line[:stripped_line.index('print(') + 7]:
                                if 'debug' not in str(file_path).lower() and 'test' not in str(file_path).lower():
                                    issues.append({
                                        "file_path": str(file_path),
                                        "line_number": i,
                                        "message": "Обнаружен print() в продакшен коде",
                                        "severity": "INFO"
                                    })

                            # Проверка на закомментированный код (длинные строки комментариев)
                            if stripped_line.startswith('#') and len(stripped_line) > 50:
                                parts = stripped_line[1:].split()
                                if len(parts) > 5:  # Если много слов в комментарии
                                    issues.append({
                                        "file_path": str(file_path),
                                        "line_number": i,
                                        "message": "Длинный комментарий, возможно, закомментированный код",
                                        "severity": "INFO"
                                    })

                except Exception as e:
                    self.log_message(f"Ошибка чтения файла {file_path}: {str(e)}", "WARNING")

            self.log_message(f"Найдено проблем: {len(issues)}",
                           "WARNING" if len(issues) > 0 else "INFO")

            # Показываем топ-5 проблем
            for issue in issues[:5]:
                self.log_message(f"Проблема: {issue['file_path']}:{issue['line_number']} - {issue['message']}",
                               issue['severity'])

            if len(issues) > 5:
                self.log_message(f"... и еще {len(issues) - 5} проблем", "WARNING")

            return {
                "issues": issues,
                "total_files": len(python_files),
                "total_lines": total_lines,
                "metrics": {"files_count": len(python_files), "lines_count": total_lines}
            }
        except Exception as e:
            self.log_message(f"Ошибка анализа кода: {str(e)}", "ERROR")
            return None

    def validate_configurations(self):
        """Проверка конфигураций проекта"""
        self.log_message("Проверка конфигураций...")

        try:
            config_issues = []

            # Проверяем наличие основных конфигурационных файлов
            config_files = [
                project_root / "config" / "config.json",
                project_root / "config.json",
                project_root / "requirements.txt",
                project_root / "setup.py",
                project_root / "pyproject.toml"
            ]

            found_configs = []
            missing_configs = []

            for config_path in config_files:
                if config_path.exists():
                    found_configs.append(str(config_path))
                    # Проверяем валидность JSON для .json файлов
                    if str(config_path).endswith('.json'):
                        try:
                            with open(config_path, 'r', encoding='utf-8') as f:
                                json.load(f)
                        except json.JSONDecodeError as e:
                            config_issues.append(f"{config_path}: {str(e)}")
                        except Exception as e:
                            config_issues.append(f"{config_path}: {str(e)}")
                else:
                    missing_configs.append(str(config_path))

            if config_issues:
                for issue in config_issues:
                    self.log_message(f"Ошибка в конфигурации: {issue}", "ERROR")
            else:
                self.log_message("Конфигурационные файлы валидны", "INFO")

            if missing_configs:
                for missing in missing_configs:
                    self.log_message(f"Отсутствует конфигурационный файл: {missing}", "WARNING")

            return {
                "valid": len(config_issues) == 0,
                "errors": config_issues,
                "found_configs": found_configs,
                "missing_configs": missing_configs
            }
        except Exception as e:
            self.log_message(f"Ошибка проверки конфигурации: {str(e)}", "ERROR")
            return None

    def run_tests(self):
        """Запуск тестов проекта"""
        self.log_message("Поиск тестов...")

        try:
            # Ищем тестовые файлы
            test_files = list(project_root.rglob("test_*.py")) + list(project_root.rglob("*_test.py"))
            # Исключаем виртуальное окружение
            test_files = [f for f in test_files if 'venv' not in str(f)]

            self.log_message(f"Найдено тестовых файлов: {len(test_files)}", "INFO")

            if test_files:
                self.log_message("Проверка возможности импорта тестов...", "INFO")

                import_successful = 0
                import_failed = 0

                for test_file in test_files:
                    try:
                        # Попробуем импортировать тест как модуль
                        import importlib.util
                        import sys

                        spec = importlib.util.spec_from_file_location(
                            test_file.stem,
                            test_file
                        )
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            import_successful += 1
                    except Exception as e:
                        import_failed += 1
                        self.log_message(f"Ошибка импорта теста {test_file}: {str(e)}", "ERROR")

                self.log_message(f"Тесты: {import_successful} успешно импортировано, {import_failed} ошибок",
                               "ERROR" if import_failed > 0 else "INFO")

                return {
                    "total_tests": len(test_files),
                    "import_successful": import_successful,
                    "import_failed": import_failed
                }
            else:
                self.log_message("Тестовые файлы не найдены", "WARNING")
                return {
                    "total_tests": 0,
                    "import_successful": 0,
                    "import_failed": 0
                }
        except Exception as e:
            self.log_message(f"Ошибка запуска тестов: {str(e)}", "ERROR")
            return None

    def check_performance(self):
        """Проверка производительности"""
        self.log_message("Проверка производительности...")

        try:
            # Проверяем время загрузки модулей
            import importlib.util
            import time

            modules_to_check = [
                ('src.cli.main', 'src/cli/main.py'),
                ('src.cli.project_manager', 'src/cli/project_manager.py'),
            ]

            load_times = {}

            for module_name, module_path in modules_to_check:
                full_path = project_root / module_path
                if full_path.exists():
                    start_time = time.time()
                    try:
                        spec = importlib.util.spec_from_file_location(module_name, full_path)
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)

                        load_time = time.time() - start_time
                        load_times[module_name] = load_time
                        self.log_message(f"Загрузка модуля {module_name}: {load_time:.3f} сек", "INFO")
                    except Exception as e:
                        self.log_message(f"Ошибка загрузки модуля {module_name}: {str(e)}", "ERROR")
                        load_times[module_name] = -1  # Ошибка загрузки
                else:
                    self.log_message(f"Модуль {module_name} не найден по пути {full_path}", "WARNING")
                    load_times[module_name] = -1  # Не найден

            # Проверка размера проекта
            total_size = 0
            total_files = 0

            for root, dirs, files in os.walk(project_root):
                # Пропускаем виртуальное окружение
                dirs[:] = [d for d in dirs if d != 'venv']

                for file in files:
                    file_path = Path(root) / file
                    try:
                        total_size += file_path.stat().st_size
                        total_files += 1
                    except:
                        pass  # Пропускаем файлы, к которым нет доступа

            self.log_message(f"Размер проекта: {total_size / (1024*1024):.2f} MB", "INFO")
            self.log_message(f"Всего файлов: {total_files}", "INFO")

            return {
                "load_times": load_times,
                "project_size_mb": total_size / (1024*1024),
                "total_files": total_files
            }
        except Exception as e:
            self.log_message(f"Ошибка проверки производительности: {str(e)}", "ERROR")
            return None

    def generate_improvement_recommendations(self):
        """Генерация рекомендаций по улучшению"""
        self.log_message("Генерация рекомендаций по улучшению...")

        recommendations = []

        # Проверяем наличие документации
        docs_exists = (project_root / "docs").exists()
        if not docs_exists:
            recommendations.append("Создать документацию проекта в папке docs/")

        # Проверяем наличие тестов
        test_files = list(project_root.glob("tests/**/*.py"))
        if len(test_files) < 5:
            recommendations.append("Добавить больше тестов в папку tests/")

        # Проверяем наличие CI/CD
        ci_exists = (project_root / ".github/workflows").exists() or (project_root / ".gitlab-ci.yml").exists()
        if not ci_exists:
            recommendations.append("Настроить CI/CD pipeline для автоматических тестов")

        # Проверяем зависимости
        req_file = project_root / "requirements.txt"
        if req_file.exists():
            with open(req_file, 'r', encoding='utf-8') as f:
                req_content = f.read()
                if '==' not in req_content:
                    recommendations.append("Зафиксировать версии зависимостей в requirements.txt")

        # Проверяем наличие .gitignore
        gitignore_exists = (project_root / ".gitignore").exists()
        if not gitignore_exists:
            recommendations.append("Создать файл .gitignore для игнорирования временных файлов")

        # Проверяем наличие README
        readme_exists = (project_root / "README.md").exists() or (project_root / "README.rst").exists()
        if not readme_exists:
            recommendations.append("Создать файл README с описанием проекта")

        for rec in recommendations:
            self.log_message(f"Рекомендация: {rec}", "INFO")

        return recommendations

    def run_full_monitoring(self):
        """Запуск полного мониторинга проекта"""
        self.log_message("="*60, "INFO")
        self.log_message("ЗАПУСК МОНИТОРИНГА ПРОЕКТА NANOPROBE SIMULATION LAB", "INFO")
        self.log_message("="*60, "INFO")

        # Запускаем все проверки
        health = self.check_system_health()
        code_analysis = self.analyze_code_quality()
        config_validation = self.validate_configurations()
        test_results = self.run_tests()
        performance = self.check_performance()
        recommendations = self.generate_improvement_recommendations()

        # Сводка
        self.log_message("="*60, "INFO")
        self.log_message("СВОДКА МОНИТОРИНГА", "INFO")
        self.log_message("="*60, "INFO")

        if health:
            self.log_message(f"Состояние системы: {health['health_score']}/100",
                           "INFO" if health['health_score'] >= 70 else "WARNING")

        if code_analysis:
            self.log_message(f"Проблем в коде: {len(code_analysis.get('issues', []))}",
                           "WARNING" if len(code_analysis.get('issues', [])) > 0 else "INFO")

        if test_results:
            total = test_results.get('total_tests', 0)
            failed = test_results.get('import_failed', 0)
            self.log_message(f"Тестов найдено: {total}, ошибок импорта: {failed}",
                           "ERROR" if failed > 0 else "INFO")

        self.log_message(f"Рекомендаций по улучшению: {len(recommendations)}", "INFO")

        # Сохраняем отчет
        self.save_monitoring_report()

        return {
            "health": health,
            "code_analysis": code_analysis,
            "config_validation": config_validation,
            "test_results": test_results,
            "performance": performance,
            "recommendations": recommendations
        }

    def save_monitoring_report(self):
        """Сохранение отчета мониторинга"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = project_root / "reports" / f"monitoring_report_{timestamp}.json"

        # Создаем папку отчетов если не существует
        report_path.parent.mkdir(exist_ok=True)

        report_data = {
            "timestamp": datetime.now().isoformat(),
            "project_name": "Nanoprobe Simulation Lab",
            "monitoring_log": self.monitoring_log,
            "summary": {
                "total_messages": len(self.monitoring_log),
                "errors": len([log for log in self.monitoring_log if log['level'] == 'ERROR']),
                "warnings": len([log for log in self.monitoring_log if log['level'] == 'WARNING']),
                "infos": len([log for log in self.monitoring_log if log['level'] == 'INFO'])
            }
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        self.log_message(f"Отчет мониторинга сохранен: {report_path}", "INFO")


def main():
    """Основная функция запуска мониторинга"""
    print("Запуск мониторинга проекта Nanoprobe Simulation Lab...")
    print(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Рабочая директория: {project_root}")

    monitor = ProjectMonitor()
    results = monitor.run_full_monitoring()

    print("\n" + "="*60)
    print("МОНИТОРИНГ ЗАВЕРШЕН!")
    print("="*60)
    print(f"Найдено рекомендаций по улучшению: {len(results['recommendations'])}")
    print(f"Обнаружено проблем в коде: {len(results['code_analysis']['issues']) if results['code_analysis'] else 0}")
    print(f"Состояние системы: {results['health']['health_score'] if results['health'] else 'N/A'}/100")

    # Показываем топ-5 рекомендаций
    print("\nТоп-5 рекомендаций:")
    for i, rec in enumerate(results['recommendations'][:5], 1):
        print(f"{i}. {rec}")

    if len(results['recommendations']) > 5:
        print(f"... и еще {len(results['recommendations']) - 5} рекомендаций")

    print(f"\nПодробный отчет сохранен в: {project_root / 'reports'}")


if __name__ == "__main__":
    main()
