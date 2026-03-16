#!/usr/bin/env python3
"""Script to fix database imports"""
import os
import re

root = r'C:\Users\maksi\OneDrive\Documents\GitHub\nanoprobe-sim-lab'

# Файлы для исправления
files_to_fix = [
    'api/database_init.py',
    'api/dependencies.py',
    'api/graphql_schema.py',
    'api/routes/simulations.py',
    'api/routes/scans.py',
    'api/routes/reports.py',
    'api/routes/dashboard_unified.py',
    'api/routes/comparison.py',
    'api/routes/analysis.py',
    'admin_cli.py',
]

for filepath in files_to_fix:
    full_path = os.path.join(root, filepath)
    if os.path.exists(full_path):
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Замена импорта
        new_content = content.replace('from utils.database.database import', 'from utils.database import')
        
        if content != new_content:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f'Fixed: {filepath}')
        else:
            print(f'No change: {filepath}')
    else:
        print(f'Not found: {filepath}')

print('Done!')
