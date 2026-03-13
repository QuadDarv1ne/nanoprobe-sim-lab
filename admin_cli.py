#!/usr/bin/env python
"""
CLI утилита для администратора Nanoprobe Sim Lab
Управление проектом из командной строки
"""

import argparse
import sys
import os
import json
from pathlib import Path
from datetime import datetime
import shutil


def print_header(text: str):
    """Вывод заголовка"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def print_success(text: str):
    """Вывод успешного сообщения"""
    print(f"✅ {text}")


def print_error(text: str):
    """Вывод ошибки"""
    print(f"❌ {text}")


def print_info(text: str):
    """Вывод информации"""
    print(f"ℹ️  {text}")


# ==================== Команды ====================

def cmd_status(args):
    """Проверка статуса системы"""
    print_header("Статус системы Nanoprobe Sim Lab")
    
    # Проверка директорий
    dirs = ["data", "logs", "output", "reports", "config"]
    print_info("Директории:")
    for dir_name in dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            size = sum(f.stat().st_size for f in dir_path.rglob("*") if f.is_file())
            print(f"  ✅ {dir_name}/ ({size:,} байт)")
        else:
            print(f"  ❌ {dir_name}/ (не существует)")
    
    # Проверка БД
    print_info("\nБаза данных:")
    db_path = Path("data/nanoprobe.db")
    if db_path.exists():
        print(f"  ✅ Файл: {db_path}")
        print(f"  ✅ Размер: {db_path.stat().st_size:,} байт")
        print(f"  ✅ Изменён: {datetime.fromtimestamp(db_path.stat().st_mtime)}")
    else:
        print(f"  ❌ {db_path} (не существует)")
    
    # Проверка логов
    print_info("\nЛоги:")
    logs_dir = Path("logs")
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.log"))
        print(f"  Найдено логов: {len(log_files)}")
        for log_file in log_files[:5]:
            print(f"    - {log_file.name} ({log_file.stat().st_size:,} байт)")
    else:
        print("  ❌ Директория logs не существует")
    
    # Проверка портов
    print_info("\nПорты:")
    try:
        import socket
        for port in [5000, 8000]:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', port))
            if result == 0:
                print(f"  🔴 Порт {port} занят")
            else:
                print(f"  🟢 Порт {port} свободен")
            sock.close()
    except Exception as e:
        print(f"  ⚠️  Не удалось проверить порты: {e}")


def cmd_cleanup(args):
    """Очистка временных файлов"""
    print_header("Очистка временных файлов")
    
    targets = []
    
    if args.all or args.pycache:
        targets.extend(list(Path(".").rglob("__pycache__")))
        targets.extend(list(Path(".").rglob("*.pyc")))
    
    if args.all or args.temp:
        targets.extend(list(Path("temp").rglob("*")))
        targets.extend(list(Path("cache").rglob("*")))
    
    if args.all or args.logs:
        targets.extend(list(Path("logs").glob("*.log")))
    
    if args.dry_run:
        print_info("Режим сухой проверки (файлы не будут удалены)")
        print_info(f"Найдено файлов для удаления: {len(targets)}")
        for target in targets[:20]:
            print(f"  - {target}")
        if len(targets) > 20:
            print(f"  ... и ещё {len(targets) - 20} файлов")
    else:
        deleted_count = 0
        for target in targets:
            try:
                if target.is_file():
                    target.unlink()
                    deleted_count += 1
                elif target.is_dir():
                    shutil.rmtree(target)
                    deleted_count += 1
            except Exception as e:
                print_error(f"Не удалось удалить {target}: {e}")
        
        print_success(f"Удалено файлов: {deleted_count}")


def cmd_backup(args):
    """Создание резервной копии"""
    print_header("Резервное копирование")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(args.output) if args.output else Path("backups")
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    backup_name = f"nanoprobe_backup_{timestamp}"
    backup_path = backup_dir / backup_name
    
    print_info(f"Создание резервной копии в {backup_path}")
    
    # Копирование БД
    db_path = Path("data/nanoprobe.db")
    if db_path.exists():
        db_backup = backup_path / "data" / "nanoprobe.db"
        db_backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(db_path, db_backup)
        print_success(f"База данных скопирована: {db_backup.stat().st_size:,} байт")
    
    # Копирование конфигов
    config_dir = backup_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    for config_file in Path("config").glob("*.json"):
        shutil.copy2(config_file, config_dir / config_file.name)
        print_success(f"Конфиг скопирован: {config_file.name}")
    
    # Копирование отчётов
    if args.include_reports:
        reports_dir = backup_path / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        for report_file in Path("reports").rglob("*.pdf"):
            shutil.copy2(report_file, reports_dir / report_file.name)
        print_success("Отчёты скопированы")
    
    # Создание метаданных
    metadata = {
        "backup_name": backup_name,
        "created_at": datetime.now().isoformat(),
        "source_path": str(Path(".").absolute()),
        "files": {
            "database": db_path.exists(),
            "configs": len(list(Path("config").glob("*.json"))),
        },
    }
    
    with open(backup_path / "backup_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print_success(f"Резервная копия создана: {backup_path}")


def cmd_restore(args):
    """Восстановление из резервной копии"""
    print_header("Восстановление из резервной копии")
    
    backup_path = Path(args.backup)
    
    if not backup_path.exists():
        print_error(f"Резервная копия не найдена: {backup_path}")
        sys.exit(1)
    
    print_info(f"Восстановление из {backup_path}")
    
    # Восстановление БД
    db_backup = backup_path / "data" / "nanoprobe.db"
    if db_backup.exists():
        db_path = Path("data/nanoprobe.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        if args.force or not db_path.exists():
            shutil.copy2(db_backup, db_path)
            print_success(f"База данных восстановлена: {db_path}")
        else:
            print_error("База данных уже существует. Используйте --force для перезаписи")
    
    # Восстановление конфигов
    config_backup = backup_path / "config"
    if config_backup.exists():
        for config_file in config_backup.glob("*.json"):
            config_dest = Path("config") / config_file.name
            config_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config_file, config_dest)
            print_success(f"Конфиг восстановлён: {config_file.name}")
    
    print_success("Восстановление завершено")


def cmd_users(args):
    """Управление пользователями"""
    print_header("Пользователи")
    
    if args.action == "list":
        from api.routes.auth import USERS_DB
        
        print_info(f"Всего пользователей: {len(USERS_DB)}")
        print()
        print(f"{'Username':<20} {'Role':<10} {'ID':<5} {'Created':<25}")
        print("-" * 60)
        
        for username, user_data in USERS_DB.items():
            print(f"{username:<20} {user_data['role']:<10} {user_data['id']:<5} {user_data['created_at']:<25}")
    
    elif args.action == "create":
        from api.routes.auth import USERS_DB, hash_password
        
        if args.username in USERS_DB:
            print_error(f"Пользователь '{args.username}' уже существует")
            sys.exit(1)
        
        new_id = max(u["id"] for u in USERS_DB.values()) + 1 if USERS_DB else 1
        
        USERS_DB[args.username] = {
            "id": new_id,
            "username": args.username,
            "password_hash": hash_password(args.password),
            "role": args.role or "user",
            "created_at": datetime.now().isoformat(),
        }
        
        print_success(f"Пользователь '{args.username}' создан (ID: {new_id})")
        print_info("Примечание: Пользователь добавлен только в памяти. Для постоянного хранения используйте БД.")
    
    elif args.action == "delete":
        from api.routes.auth import USERS_DB
        
        if args.username not in USERS_DB:
            print_error(f"Пользователь '{args.username}' не найден")
            sys.exit(1)
        
        del USERS_DB[args.username]
        print_success(f"Пользователь '{args.username}' удалён")


def cmd_migrate(args):
    """Миграция базы данных"""
    print_header("Миграция базы данных")
    
    from utils.database import DatabaseManager
    
    db = DatabaseManager("data/nanoprobe.db")
    
    print_info("Проверка структуры базы данных...")
    
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Проверка таблиц
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            print_success(f"Найдено таблиц: {len(tables)}")
            for table in tables:
                print(f"  - {table}")
            
            # Проверка индексов
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = [row[0] for row in cursor.fetchall() if not row[0].startswith('sqlite_')]
            
            print_info(f"Найдено индексов: {len(indexes)}")
            
    except Exception as e:
        print_error(f"Ошибка проверки БД: {e}")
        sys.exit(1)
    
    print_success("Миграция завершена успешно")


def cmd_info(args):
    """Информация о проекте"""
    print_header("Nanoprobe Sim Lab - Информация")
    
    info = {
        "name": "Nanoprobe Simulation Lab",
        "version": "1.0.0",
        "description": "Лаборатория моделирования нанозонда",
        "owner": "Школа программирования Maestro7IT",
        "email": "maksimqwe42@mail.ru",
        "website": "https://school-maestro7it.ru/",
        "license": "Proprietary",
        "python_version": sys.version,
        "platform": sys.platform,
    }
    
    for key, value in info.items():
        print(f"{key.replace('_', ' ').title():<20}: {value}")
    
    # Статистика проекта
    print_info("\nСтатистика проекта:")
    
    py_files = list(Path(".").rglob("*.py"))
    print(f"  Python файлов: {len(py_files)}")
    
    total_lines = sum(1 for f in py_files for _ in open(f, 'r', encoding='utf-8', errors='ignore'))
    print(f"  Строк кода: {total_lines:,}")


# ==================== Основная функция ====================

def main():
    parser = argparse.ArgumentParser(
        description="CLI утилита для администратора Nanoprobe Sim Lab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Доступные команды")
    
    # Команда: status
    parser_status = subparsers.add_parser("status", help="Проверка статуса системы")
    parser_status.set_defaults(func=cmd_status)
    
    # Команда: cleanup
    parser_cleanup = subparsers.add_parser("cleanup", help="Очистка временных файлов")
    parser_cleanup.add_argument("--all", action="store_true", help="Очистить всё")
    parser_cleanup.add_argument("--pycache", action="store_true", help="Очистить __pycache__")
    parser_cleanup.add_argument("--temp", action="store_true", help="Очистить temp/cache")
    parser_cleanup.add_argument("--logs", action="store_true", help="Очистить логи")
    parser_cleanup.add_argument("--dry-run", action="store_true", help="Сухая проверка")
    parser_cleanup.set_defaults(func=cmd_cleanup)
    
    # Команда: backup
    parser_backup = subparsers.add_parser("backup", help="Создание резервной копии")
    parser_backup.add_argument("-o", "--output", type=str, help="Директория для бэкапа")
    parser_backup.add_argument("--include-reports", action="store_true", help="Включить отчёты")
    parser_backup.set_defaults(func=cmd_backup)
    
    # Команда: restore
    parser_restore = subparsers.add_parser("restore", help="Восстановление из бэкапа")
    parser_restore.add_argument("backup", type=str, help="Путь к бэкапу")
    parser_restore.add_argument("--force", action="store_true", help="Перезаписать существующие файлы")
    parser_restore.set_defaults(func=cmd_restore)
    
    # Команда: users
    parser_users = subparsers.add_parser("users", help="Управление пользователями")
    parser_users.add_argument("action", choices=["list", "create", "delete"], help="Действие")
    parser_users.add_argument("--username", type=str, help="Имя пользователя")
    parser_users.add_argument("--password", type=str, help="Пароль")
    parser_users.add_argument("--role", type=str, choices=["user", "admin"], help="Роль")
    parser_users.set_defaults(func=cmd_users)
    
    # Команда: migrate
    parser_migrate = subparsers.add_parser("migrate", help="Миграция БД")
    parser_migrate.set_defaults(func=cmd_migrate)
    
    # Команда: info
    parser_info = subparsers.add_parser("info", help="Информация о проекте")
    parser_info.set_defaults(func=cmd_info)
    
    # Парсинг аргументов
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Выполнение команды
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print_error(f"Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
