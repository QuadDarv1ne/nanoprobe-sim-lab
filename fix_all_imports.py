#!/usr/bin/env python3
"""Final comprehensive import fix script"""
import os
import re

root = r'C:\Users\maksi\OneDrive\Documents\GitHub\nanoprobe-sim-lab'

# Массовые замены
replacements = {
    # Database imports
    'from utils.database.database import': 'from utils.database import',
    
    # Batch imports  
    'from utils.batch.batch_processor import': 'from utils.batch_processor import',
    
    # Circuit breaker
    'from utils.circuit_breaker import': 'from utils.caching.circuit_breaker import',
    
    # NASA
    'from utils.nasa_api_client import': 'from utils.api.nasa_api_client import',
    
    # Rate limiter
    'from utils.security.rate_limiter import api_limit': 'from utils.security.rate_limiter import rate_limit',
    '@api_limit(': '@rate_limit(',
    'window=60)': 'window_seconds=60)',
    
    # Defect analyzer
    'from utils.defect_analyzer import': 'from utils.ai.defect_analyzer import',
    
    # PDF report
    'from utils.pdf_report_generator import': 'from utils.reporting.pdf_report_generator import',
    
    # Enhanced monitor
    'from utils.enhanced_monitor import': 'from utils.monitoring.enhanced_monitor import',
    
    # Query analyzer - remove import
    'from utils.database.query_analyzer import': '# REMOVED: from utils.database.query_analyzer import',
}

# Файлы для обработки
files_to_fix = [
    'api/main.py',
    'api/state.py',
    'api/database_init.py',
    'api/dependencies.py',
    'api/graphql_schema.py',
    'api/routes/simulations.py',
    'api/routes/scans.py',
    'api/routes/reports.py',
    'api/routes/dashboard_unified.py',
    'api/routes/comparison.py',
    'api/routes/analysis.py',
    'api/routes/batch.py',
    'api/routes/external_services.py',
    'api/routes/nasa.py',
    'api/routes/monitoring.py',
    'admin_cli.py',
]

# Удаляем database.py роут
database_route_path = os.path.join(root, 'api', 'routes', 'database.py')
if os.path.exists(database_route_path):
    os.remove(database_route_path)
    print(f'Deleted: api/routes/database.py')

for filepath in files_to_fix:
    full_path = os.path.join(root, filepath)
    if os.path.exists(full_path):
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content = content
        for old, new in replacements.items():
            new_content = new_content.replace(old, new)
        
        # Удаляем строки с REMOVED импортами и связанные
        if '# REMOVED:' in new_content:
            lines = new_content.split('\n')
            new_lines = []
            skip_next = 0
            for line in lines:
                if skip_next > 0:
                    skip_next -= 1
                    continue
                if '# REMOVED:' in line:
                    # Skip this line and any continuation
                    if line.strip().endswith('('):
                        skip_next = 5  # Skip up to 5 more lines
                    continue
                new_lines.append(line)
            new_content = '\n'.join(new_lines)
        
        if content != new_content:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f'Fixed: {filepath}')
        else:
            print(f'No change: {filepath}')
    else:
        print(f'Not found: {filepath}')

# Исправляем api/main.py - удаляем database из импортов и регистрации
main_path = os.path.join(root, 'api', 'main.py')
with open(main_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Удаляем database из импорта роутов
content = content.replace(
    'from api.routes import graphql, ml_analysis, external_services, nasa, database, monitoring',
    'from api.routes import graphql, ml_analysis, external_services, nasa, monitoring'
)

# Удаляем регистрацию database роутера
content = re.sub(r'\n# Database Query Analyzer\napp\.include_router\(database\.router.*?\nlogger\.info\("Database routes registered"\)\n', '\n', content, flags=re.DOTALL)

with open(main_path, 'w', encoding='utf-8') as f:
    f.write(content)
print('Fixed: api/main.py (database router removed)')

print('\nDone! All imports fixed.')
