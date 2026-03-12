#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI для управления миграциями Alembic
Использование: python migrate.py [command]
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd):
    """Выполнение команды"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode


def main():
    if len(sys.argv) < 2:
        print("Usage: python migrate.py [command]")
        print("Commands:")
        print("  init       - Initialize alembic (done)")
        print("  migrate    - Create new migration")
        print("  upgrade    - Apply all migrations")
        print("  downgrade  - Downgrade one version")
        print("  current    - Show current version")
        print("  history    - Show migration history")
        sys.exit(1)

    command = sys.argv[1]
    project_root = Path(__file__).parent

    if command == "migrate":
        # Создание новой миграции
        message = sys.argv[2] if len(sys.argv) > 2 else "auto_migration"
        cmd = ["alembic", "-c", str(project_root / "alembic.ini"), "revision", "--autogenerate", "-m", message]
        sys.exit(run_command(cmd))

    elif command == "upgrade":
        # Применение миграций
        cmd = ["alembic", "-c", str(project_root / "alembic.ini"), "upgrade", "head"]
        sys.exit(run_command(cmd))

    elif command == "downgrade":
        # Откат миграций
        cmd = ["alembic", "-c", str(project_root / "alembic.ini"), "downgrade", "-1"]
        sys.exit(run_command(cmd))

    elif command == "current":
        # Текущая версия
        cmd = ["alembic", "-c", str(project_root / "alembic.ini"), "current"]
        sys.exit(run_command(cmd))

    elif command == "history":
        # История миграций
        cmd = ["alembic", "-c", str(project_root / "alembic.ini"), "history"]
        sys.exit(run_command(cmd))

    elif command == "stamp":
        # Пометить текущую БД
        revision = sys.argv[2] if len(sys.argv) > 2 else "head"
        cmd = ["alembic", "-c", str(project_root / "alembic.ini"), "stamp", revision]
        sys.exit(run_command(cmd))

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
