#!/usr/bin/env python3
"""Script to fix remaining imports"""
import os

root = r'C:\Users\maksi\OneDrive\Documents\GitHub\nanoprobe-sim-lab'

# Файлы и замены
replacements = [
    ('api/main.py', 'from utils.circuit_breaker import', 'from utils.caching.circuit_breaker import'),
    ('api/routes/external_services.py', 'from utils.circuit_breaker import', 'from utils.caching.circuit_breaker import'),
]

for filepath, old, new in replacements:
    full_path = os.path.join(root, filepath)
    if os.path.exists(full_path):
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content = content.replace(old, new)
        
        if content != new_content:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f'Fixed: {filepath} ({old} -> {new})')
        else:
            print(f'No change: {filepath}')
    else:
        print(f'Not found: {filepath}')

print('Done!')
