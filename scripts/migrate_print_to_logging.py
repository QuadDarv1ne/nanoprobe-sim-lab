#!/usr/bin/env python3
"""Скрипт для массовой замены print() на logging в utils/"""

import re
import shutil
from pathlib import Path
from typing import Callable, Tuple

MIGRATION_PATTERNS = [
    (
        r'print\("===\s*(.+?)\s*==="\)',
        'logger.info("=" * 40)\nlogger.info("\\1")\nlogger.info("=" * 40)',
    ),
    (r'print\("✓\s*(.+?)"\)', 'logger.info("✓ \\1")'),
    (r'print\("✗\s*(.+?)"\)', 'logger.error("✗ \\1")'),
    (r'print\(f"(.+?)"\)', 'logger.info(f"\\1")'),
    (r'print\("(.+?)"\)', 'logger.info("\\1")'),
]


def create_backup(filepath: Path) -> Path:
    backup_path = filepath.with_suffix(filepath.suffix + ".bak")
    shutil.copy2(filepath, backup_path)
    return backup_path


def add_logging_import(content: str) -> str:
    lines = content.split("\n")
    has_logging = any(line.strip().startswith("import logging") for line in lines)
    if has_logging:
        return content
    import_end = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("import ") or line.strip().startswith("from "):
            import_end = i + 1
        elif line.strip() and not line.strip().startswith("#"):
            if import_end == 0:
                import_end = i
            break
    lines.insert(import_end, "import logging")
    lines.insert(import_end + 1, "logger = logging.getLogger(__name__)")
    lines.insert(import_end + 2, "")
    return "\n".join(lines)


def _create_multi_replace(replacement: str) -> Callable[[re.Match], str]:
    """Создание функции замены для многострочных паттернов."""

    def multi_replace(match: re.Match) -> str:
        return replacement.replace("\\1", match.group(1))

    return multi_replace


def migrate_file(filepath: Path, dry_run: bool = False) -> Tuple[int, int]:
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception as e:
        print(f" Ошибка чтения {filepath.name}: {e}")
        return (0, 0)

    original_content = content
    total_replacements = 0
    print_count = len(re.findall(r"print\(", content))
    if print_count == 0:
        return (0, 0)

    print(f" {filepath.name}: найдено {print_count} print()")

    if "print(" in content and "import logging" not in content:
        content = add_logging_import(content)

    for pattern, replacement in MIGRATION_PATTERNS:
        matches = re.findall(pattern, content)
        if matches:
            if "\\n" in replacement:
                multi_replace = _create_multi_replace(replacement)
                content = re.sub(pattern, multi_replace, content)
            else:
                content = re.sub(pattern, replacement, content)
            total_replacements += len(matches)

    if content != original_content:
        if not dry_run:
            backup = create_backup(filepath)
            print(f" Резервная копия: {backup.name}")
            try:
                filepath.write_text(content, encoding="utf-8")
                print(f" Изменено {total_replacements} строк")
            except Exception as e:
                print(f" Ошибка записи: {e}")
                shutil.copy2(backup, filepath)
        else:
            print(f" Dry run: будет изменено {total_replacements} строк")

    return (total_replacements, 1)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Миграция print() на logging")
    parser.add_argument("--dry-run", action="store_true", help="Предпросмотр")
    parser.add_argument("--directory", type=str, default="utils", help="Директория")
    args = parser.parse_args()

    utils_dir = Path(args.directory)
    if not utils_dir.exists():
        print(f"Директория {utils_dir} не найдена")
        return 1

    print(f"Поиск Python файлов в {utils_dir}/...")
    py_files = [f for f in utils_dir.glob("**/*.py") if f.name != "__init__.py"]
    print(f"Найдено {len(py_files)} файлов\n")

    total_files = 0
    total_replacements = 0

    for filepath in py_files:
        replacements, _ = migrate_file(filepath, dry_run=args.dry_run)
        if replacements > 0:
            total_files += 1
            total_replacements += replacements

    print(f"\n{'=' * 50}")
    print(f"Итоги:")
    print(f" Изменено файлов: {total_files}")
    print(f" Выполнено замен: {total_replacements}")

    if args.dry_run:
        print("\nЭто был dry run. Запустите без --dry-run для применения.")
    else:
        print("\nМиграция завершена!")
        print("Резервные копии созданы с расширением .bak")

    return 0


if __name__ == "__main__":
    exit(main())
