#!/usr/bin/env python3
"""
Скрипт автоматической миграции datetime.now(timezone.utc) → datetime.now(timezone.utc)

Находит все .py файлы в проекте и заменяет:
- datetime.now(timezone.utc) → datetime.now(timezone.utc)
- Добавляет from datetime import timezone если нужно

Автоматически создаёт backup перед изменениями.
"""

import re
import os
from pathlib import Path
from datetime import datetime
import shutil
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Исключённые директории
EXCLUDE_DIRS = {
    '.git', '__pycache__', 'venv', '.venv', 'node_modules',
    'build', 'dist', '.pytest_cache', '.mypy_cache',
    'backups', 'output', 'logs'
}

# Паттерны для поиска
DATETIME_NOW_PATTERN = re.compile(r'datetime\.now\(\)')

# Паттерны для проверки импортов
DATETIME_IMPORT_PATTERN = re.compile(
    r'from\s+datetime\s+import\s+([^#\n]+)'
)


def find_python_files(root_dir: Path) -> list[Path]:
    """Рекурсивный поиск всех .py файлов"""
    py_files = []
    for root, dirs, files in os.walk(root_dir):
        # Исключаем ненужные директории
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for file in files:
            if file.endswith('.py'):
                py_files.append(Path(root) / file)
    
    return py_files


def needs_timezone_import(content: str) -> bool:
    """Проверяет, нужен ли import timezone"""
    # Проверяем, есть ли datetime.now(timezone.utc)
    if 'datetime.now(timezone.utc)' not in content:
        return False
    
    # Проверяем, импортирован ли уже timezone
    for match in DATETIME_IMPORT_PATTERN.finditer(content):
        imports = match.group(1)
        if 'timezone' in imports:
            return False
    
    return True


def add_timezone_import(content: str) -> str:
    """Добавляет timezone в существующий import datetime"""
    # Ищем from datetime import ...
    match = DATETIME_IMPORT_PATTERN.search(content)
    if match:
        imports_line = match.group(1)
        # Добавляем timezone если его нет
        if 'timezone' not in imports_line:
            new_imports = imports_line.rstrip() + ', timezone'
            content = content[:match.start(1)] + new_imports + content[match.end(1):]
            return content
    
    # Ищем import datetime
    if re.search(r'^import datetime$', content, re.MULTILINE):
        # Меняем на from datetime import datetime, timezone
        content = re.sub(
            r'^import datetime$',
            'from datetime import datetime, timezone',
            content,
            flags=re.MULTILINE
        )
        return content
    
    # Если ничего не нашли, добавляем новый import в начало
    # Ищем первый import или from
    first_import = re.search(r'^(import|from)\s', content, re.MULTILINE)
    if first_import:
        insert_pos = first_import.start()
        content = content[:insert_pos] + 'from datetime import datetime, timezone\n' + content[insert_pos:]
    else:
        # Добавляем после shebang/docstring
        content = 'from datetime import datetime, timezone\n' + content
    
    return content


def migrate_file(filepath: Path, dry_run: bool = False) -> dict:
    """Миграция одного файла"""
    stats = {
        'file': str(filepath),
        'changes': 0,
        'added_import': False,
        'backup': None
    }
    
    try:
        content = filepath.read_text(encoding='utf-8')
    except Exception as e:
        logger.error(f"Ошибка чтения {filepath}: {e}")
        return stats
    
    # Считаем количество замен
    matches = DATETIME_NOW_PATTERN.findall(content)
    if not matches:
        return stats
    
    stats['changes'] = len(matches)
    
    if dry_run:
        logger.info(f"[DRY RUN] {filepath}: {len(matches)} замен")
        return stats
    
    # Создаём backup
    backup_path = filepath.with_suffix(f'.py.bak.{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}')
    try:
        shutil.copy2(filepath, backup_path)
        stats['backup'] = str(backup_path)
    except Exception as e:
        logger.error(f"Ошибка создания backup {filepath}: {e}")
        return stats
    
    # Заменяем datetime.now(timezone.utc) → datetime.now(timezone.utc)
    new_content = DATETIME_NOW_PATTERN.sub('datetime.now(timezone.utc)', content)
    
    # Добавляем import timezone если нужно
    if needs_timezone_import(new_content):
        new_content = add_timezone_import(new_content)
        stats['added_import'] = True
    
    # Записываем изменения
    try:
        filepath.write_text(new_content, encoding='utf-8')
        logger.info(f"✓ {filepath}: {stats['changes']} замен, import добавлен: {stats['added_import']}")
    except Exception as e:
        logger.error(f"Ошибка записи {filepath}: {e}")
        # Восстанавливаем из backup
        shutil.copy2(backup_path, filepath)
        logger.info(f"↩ Восстановлено из backup")
    
    return stats


def main():
    """Основная функцияя"""
    project_root = Path(__file__).parent
    
    logger.info("=" * 80)
    logger.info("datetime.now(timezone.utc) → datetime.now(timezone.utc) Migration")
    logger.info("=" * 80)
    
    # Сначала dry run
    logger.info("\n📊 DRY RUN - анализ изменений...")
    py_files = find_python_files(project_root)
    logger.info(f"Найдено {len(py_files)} Python файлов")
    
    total_changes = 0
    files_to_migrate = []
    
    for filepath in py_files:
        stats = migrate_file(filepath, dry_run=True)
        if stats['changes'] > 0:
            total_changes += stats['changes']
            files_to_migrate.append(stats)
    
    logger.info(f"\n📈 Найдено {total_changes} замен в {len(files_to_migrate)} файлах")
    
    if total_changes == 0:
        logger.info("✅ Изменений не требуется")
        return
    
    # Спрашиваем подтверждение
    response = input("\n🔧 Продолжить миграцию? (y/n): ").strip().lower()
    if response not in ('y', 'yes', 'да'):
        logger.info("❌ Миграция отменена")
        return
    
    # Выполняем миграцию
    logger.info("\n🚀 Выполнение миграции...")
    success_count = 0
    error_count = 0
    total_replacements = 0
    
    for file_stat in files_to_migrate:
        filepath = Path(file_stat['file'])
        stats = migrate_file(filepath, dry_run=False)
        if stats['changes'] > 0:
            success_count += 1
            total_replacements += stats['changes']
        else:
            error_count += 1
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ МИГРАЦИЯ ЗАВЕРШЕНА")
    logger.info("=" * 80)
    logger.info(f"Файлов изменено: {success_count}")
    logger.info(f"Ошибок: {error_count}")
    logger.info(f"Всего замен: {total_replacements}")
    logger.info(f"\nBackup файлы: *.py.bak.*")
    logger.info("\nДля отмены изменений:")
    logger.info("  python migrate_datetime.py --rollback")
    logger.info("=" * 80)


def rollback():
    """Откат миграции из backup файлов"""
    project_root = Path(__file__).parent
    
    logger.info("🔄 Откат миграции...")
    
    backup_files = list(project_root.glob('**/*.py.bak.*'))
    if not backup_files:
        logger.info("❌ Backup файлы не найдены")
        return
    
    for backup in backup_files:
        original = backup.with_suffix('').with_suffix('.py')
        if original.exists():
            shutil.copy2(backup, original)
            backup.unlink()
            logger.info(f"↩ Восстановлено: {original}")
    
    logger.info(f"✅ Откат завершён, восстановлено {len(backup_files)} файлов")


if __name__ == '__main__':
    import sys
    if '--rollback' in sys.argv:
        rollback()
    else:
        main()
