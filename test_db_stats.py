#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Тест статистики БД"""

from utils.database import DatabaseManager

db = DatabaseManager('data/nanoprobe.db')
stats = db.get_statistics()
print("Статистика БД:")
for key, value in stats.items():
    print(f"  {key}: {value}")
