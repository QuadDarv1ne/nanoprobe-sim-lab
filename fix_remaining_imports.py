#!/usr/bin/env python3
"""Fix remaining incorrect imports"""
import os

root = r'C:\Users\maksi\OneDrive\Documents\GitHub\nanoprobe-sim-lab'

replacements = {
    'from utils.config_manager import': 'from utils.config.config_manager import',
    'from utils.data_manager import': 'from utils.data.data_manager import',
    'from utils.two_factor_auth import': 'from utils.security.two_factor_auth import',
    'from utils.pretrained_defect_analyzer import': 'from utils.ai.pretrained_defect_analyzer import',
}

files_to_fix = [
    'api/websocket_server.py',
    'api/api_interface.py',
    'api/routes/auth.py',
    'api/routes/ml_analysis.py',
    'utils/simulator_orchestrator.py',
]

for filepath in files_to_fix:
    full_path = os.path.join(root, filepath)
    if os.path.exists(full_path):
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
    else:
        print(f'Not found: {filepath}')

print('Done!')
