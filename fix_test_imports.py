#!/usr/bin/env python3
"""Fix test imports"""
import os

root = r'C:\Users\maksi\OneDrive\Documents\GitHub\nanoprobe-sim-lab'

replacements = {
    'from utils.database.database import': 'from utils.database import',
    'from utils.circuit_breaker import': 'from utils.caching.circuit_breaker import',
    'from utils.nasa_api_client import': 'from utils.api.nasa_api_client import',
    'from utils.pretrained_defect_analyzer import': 'from utils.ai.pretrained_defect_analyzer import',
}

import glob
test_files = glob.glob(os.path.join(root, 'tests', '*.py'))

for full_path in test_files:
    filepath = os.path.relpath(full_path, root).replace('\\', '/')
    with open(full_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content
    for old, new in replacements.items():
        new_content = new_content.replace(old, new)
    
    if content != new_content:
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f'Fixed: {filepath}')
    else:
        print(f'No change: {filepath}')

print('Done!')
