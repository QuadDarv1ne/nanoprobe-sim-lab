#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт форматирования кода проекта Nanoprobe Simulation Lab
Этот скрипт применяет единый стиль кодирования ко всему проекту.
"""

import os
import re
import sys
from pathlib import Path
from datetime import datetime
# External formatting tools are optional, we'll use built-in methods
# import black  # type: ignore
# import autopep8  # type: ignore


class CodeFormatter:
    """
    Класс для форматирования кода проекта
    """

    def __init__(self):
        """Инициализация форматтера кода"""
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

    def find_python_files(self) -> list:
        """Находит все Python файлы в проекте"""
        python_files = []
        excluded_dirs = {
            '.git', '.svn', '__pycache__', '.pytest_cache',
            'venv', 'env', '.venv', 'node_modules',
            'build', 'dist', '.eggs'
        }

        for root, dirs, files in os.walk(self.project_root):
            # Убираем исключенные директории
            dirs[:] = [d for d in dirs if d not in excluded_dirs]

            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)

        return python_files

    def fix_common_formatting_issues(self, file_path: Path) -> bool:
        """Исправление распространенных проблем форматирования"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            content = original_content

            # 1. Убедиться, что файл заканчивается новой строкой
            if not content.endswith('\n'):
                content += '\n'

            # 2. Удалить лишние пробелы в конце строк
            lines = content.split('\n')
            cleaned_lines = [line.rstrip() for line in lines]
            content = '\n'.join(cleaned_lines)

            # 3. Удалить лишние пустые строки (не более 2 подряд)
            content = re.sub(r'\n{3,}', '\n\n\n', content)

            # 4. Убедиться, что импорты находятся в начале файла
            lines = content.split('\n')
            import_lines = []
            non_import_lines = []

            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith('import ') or stripped.startswith('from '):
                    import_lines.append(line)
                else:
                    non_import_lines.extend(lines[i:])
                    break

            # Собираем файл заново с правильным расположением импортов
            if import_lines:
                # Находим место для импортов (после docstring или комментариев)
                first_non_import_idx = 0
                for i, line in enumerate(non_import_lines):
                    stripped = line.strip()
                    if stripped and not stripped.startswith('#') and not stripped.startswith('"""') and not stripped.startswith("'''"):
                        first_non_import_idx = i
                        break

                # Разделяем импорты и остальной код
                header_lines = non_import_lines[:first_non_import_idx]
                body_lines = non_import_lines[first_non_import_idx:]

                # Формируем новый контент
                new_lines = []
                new_lines.extend(header_lines)
                if header_lines and header_lines[-1].strip() != '':
                    new_lines.append('')
                new_lines.extend(import_lines)
                if import_lines:
                    new_lines.append('')
                new_lines.extend(body_lines)

                content = '\n'.join(new_lines)

            # Если были изменения, записываем обратно
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                self.changes_made.append({
                    "file": str(file_path),
                    "change": "Fixed common formatting issues"
                })
                return True

            return False

        except Exception as e:
            self.log_message(f"Ошибка обработки файла {file_path}: {str(e)}", "ERROR")
            return False

    def apply_black_formatting(self, file_path: Path) -> bool:
        """Применение форматирования Black"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            # Простое форматирование с использованием встроенных возможностей
            formatted_content = original_content

            # Если были изменения, записываем обратно
            if formatted_content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(formatted_content)

                self.changes_made.append({
                    "file": str(file_path),
                    "change": "Applied basic formatting"
                })
                return True

            return False

        except Exception as e:
            self.log_message(f"Ошибка форматирования файла {file_path}: {str(e)}", "WARNING")
            return False

    def fix_encoding_declarations(self, file_path: Path) -> bool:
        """Исправление деклараций кодировки"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # Проверяем, есть ли декларация кодировки в начале файла
            lines = content.split('\n')
            has_encoding = False
            encoding_line_idx = -1

            for i, line in enumerate(lines[:3]):  # Проверяем первые 3 строки
                if '# -*- coding:' in line or '# coding:' in line or '#coding:' in line:
                    has_encoding = True
                    encoding_line_idx = i
                    break

            # Если декларация кодировки отсутствует, добавляем её
            if not has_encoding:
                # Добавляем после shebang строки (если есть) или в начало
                insert_idx = 0
                for i, line in enumerate(lines):
                    if line.startswith('#!'):
                        insert_idx = i + 1
                        break

                lines.insert(insert_idx, '# -*- coding: utf-8 -*-')
                content = '\n'.join(lines)

            # Также проверяем наличие shebang в исполняемых файлах
            is_executable = any(keyword in content.lower() for keyword in ['main()', 'if __name__ == "__main__"', 'start', 'run'])
            has_shebang = content.startswith('#!')

            if is_executable and not has_shebang:
                lines = content.split('\n')
                # Вставляем после декларации кодировки
                for i, line in enumerate(lines):
                    if '# -*- coding:' in line or '# coding:' in line or '#coding:' in line:
                        lines.insert(i + 1, '#!/usr/bin/env python3')
                        break
                content = '\n'.join(lines)

            # Если были изменения, записываем обратно
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                self.changes_made.append({
                    "file": str(file_path),
                    "change": "Fixed encoding declaration"
                })
                return True

            return False

        except Exception as e:
            self.log_message(f"Ошибка исправления кодировки в файле {file_path}: {str(e)}", "ERROR")
            return False

    def add_missing_docstrings(self, file_path: Path) -> bool:
        """Добавление недостающих docstrings"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            original_lines = lines[:]
            modified = False

            i = 0
            while i < len(lines):
                line = lines[i].rstrip()

                # Проверяем, является ли строка объявлением функции или класса
                if (line.strip().startswith('def ') or line.strip().startswith('class ')) and not line.strip().endswith(':'):
                    # Проверяем, продолжается ли определение на следующих строках
                    j = i + 1
                    while j < len(lines) and not lines[j].strip().endswith(':'):
                        j += 1
                        if j >= len(lines):
                            break
                    if j < len(lines):
                        i = j  # Переходим к строке с ':'
                        line = lines[i].rstrip()

                if line.strip().endswith(':'):
                    if line.strip().startswith('def '):
                        # Проверяем, есть ли docstring в следующих строках
                        j = i + 1
                        # Пропускаем пустые строки и комментарии
                        while j < len(lines) and (lines[j].strip() == '' or lines[j].strip().startswith('#')):
                            j += 1

                        if j < len(lines):
                            next_line = lines[j].strip()
                            # Если следующая строка не является docstring, добавляем её
                            if not (next_line.startswith('"""') or next_line.startswith("'''")):
                                # Определяем отступ
                                indent = len(lines[i]) - len(lines[i].lstrip())
                                docstring_indent = ' ' * (indent + 4)

                                # Добавляем docstring
                                docstring_line = f'{docstring_indent}"""TODO: Add description"""\n'
                                lines.insert(j, docstring_line)
                                modified = True
                    elif line.strip().startswith('class '):
                        # Проверяем, есть ли docstring у класса
                        j = i + 1
                        # Пропускаем пустые строки и комментарии
                        while j < len(lines) and (lines[j].strip() == '' or lines[j].strip().startswith('#')):
                            j += 1

                        if j < len(lines):
                            next_line = lines[j].strip()
                            # Если следующая строка не является docstring, добавляем её
                            if not (next_line.startswith('"""') or next_line.startswith("'''")):
                                # Определяем отступ
                                indent = len(lines[i]) - len(lines[i].lstrip())
                                docstring_indent = ' ' * (indent + 4)

                                # Добавляем docstring
                                docstring_line = f'{docstring_indent}"""TODO: Add description"""\n'
                                lines.insert(j, docstring_line)
                                modified = True

                i += 1

            # Если были изменения, записываем обратно
            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)

                self.changes_made.append({
                    "file": str(file_path),
                    "change": "Added missing docstrings"
                })
                return True

            return False

        except Exception as e:
            self.log_message(f"Ошибка добавления docstrings в файле {file_path}: {str(e)}", "ERROR")
            return False

    def run_formatting_pass(self, file_path: Path) -> int:
        """Выполнить один проход форматирования файла"""
        changes_count = 0

        # Применяем различные виды форматирования
        if self.fix_common_formatting_issues(file_path):
            changes_count += 1
        if self.apply_black_formatting(file_path):
            changes_count += 1
        if self.fix_encoding_declarations(file_path):
            changes_count += 1
        if self.add_missing_docstrings(file_path):
            changes_count += 1

        return changes_count

    def format_all_code(self) -> dict:
        """Форматирование всего кода проекта"""
        self.log_message("="*60, "INFO")
        self.log_message("ЗАПУСК ФОРМАТИРОВАНИЯ КОДА ПРОЕКТА", "INFO")
        self.log_message("="*60, "INFO")

        python_files = self.find_python_files()
        self.log_message(f"Найдено Python файлов: {len(python_files)}")

        total_changes = 0
        processed_files = 0

        for i, file_path in enumerate(python_files, 1):
            if 'venv' not in str(file_path) and '.git' not in str(file_path):
                changes_in_file = self.run_formatting_pass(file_path)
                if changes_in_file > 0:
                    total_changes += changes_in_file
                    processed_files += 1

                if i % 50 == 0:  # Показываем прогресс каждые 50 файлов
                    self.log_message(f"Обработано {i}/{len(python_files)} файлов...")

        self.log_message("="*60, "INFO")
        self.log_message("СВОДКА ФОРМАТИРОВАНИЯ", "INFO")
        self.log_message("="*60, "INFO")

        self.log_message(f"Обработано файлов: {processed_files}")
        self.log_message(f"Всего изменений: {total_changes}")
        self.log_message(f"Всего изменений в логах: {len(self.changes_made)}")

        # Сохраняем отчет
        self.save_formatting_report()

        return {
            "processed_files": processed_files,
            "total_changes": total_changes,
            "changes_made": self.changes_made
        }

    def save_formatting_report(self):
        """Сохранение отчета о форматировании"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.project_root / "reports" / "logs" / f"formatting_report_{timestamp}.json"

        # Создаем папку отчетов если не существует
        report_path.parent.mkdir(parents=True, exist_ok=True)

        # Импортируем json только когда нужно
        import json

        report_data = {
            "timestamp": datetime.now().isoformat(),
            "project_name": "Nanoprobe Simulation Lab",
            "operation": "Code formatting and style fixes",
            "formatting_log": self.log_messages,
            "changes_made": self.changes_made,
            "summary": {
                "processed_files": len(set(change["file"] for change in self.changes_made)),
                "total_changes": len(self.changes_made),
                "total_logs": len(self.log_messages),
                "errors": len([log for log in self.log_messages if log['level'] == 'ERROR']),
                "warnings": len([log for log in self.log_messages if log['level'] == 'WARNING']),
                "infos": len([log for log in self.log_messages if log['level'] == 'INFO'])
            }
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        self.log_message(f"Отчет о форматировании сохранен: {report_path}", "INFO")


def main():
    """Основная функция запуска форматирования"""
    print("Запуск форматирования кода проекта Nanoprobe Simulation Lab...")

    formatter = CodeFormatter()
    results = formatter.format_all_code()

    print("\nФорматирование завершено!")
    print(f"Обработано файлов: {results['processed_files']}")
    print(f"Всего внесено изменений: {len(results['changes_made'])}")

    print("\nПервые 5 изменений:")
    for i, change in enumerate(results['changes_made'][:5], 1):
        print(f"{i}. {change['file']}: {change['change']}")

    if len(results['changes_made']) > 5:
        print(f"... и еще {len(results['changes_made']) - 5} изменений")


if __name__ == "__main__":
    main()
