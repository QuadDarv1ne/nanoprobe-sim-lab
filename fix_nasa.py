#!/usr/bin/env python3
"""Fix api_limit -> rate_limit in nasa.py"""
import os

filepath = r'C:\Users\maksi\OneDrive\Documents\GitHub\nanoprobe-sim-lab\api\routes\nasa.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

new_content = content.replace('@api_limit(', '@rate_limit(')
new_content = new_content.replace('window=', 'window_seconds=')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(new_content)

print('Fixed nasa.py')
