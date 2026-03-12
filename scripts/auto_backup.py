#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для автоматических резервных копий
Использование: python scripts/auto_backup.py [daily|weekly|monthly]
"""

import sys
import os
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.backup_manager import BackupManager
from utils.config_manager import ConfigManager


def main():
    if len(sys.argv) < 2:
        strategy = "daily"
    else:
        strategy = sys.argv[1]

    print(f"Creating {strategy} backup...")

    try:
        config = ConfigManager()
        backup_mgr = BackupManager(config)

        backup_path = backup_mgr.create_backup_strategy(
            strategy=strategy,
            auto_cleanup=True
        )

        if backup_path:
            print(f"Backup created: {backup_path}")
            return 0
        else:
            print("Backup failed")
            return 1

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
