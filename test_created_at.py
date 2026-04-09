"""Quick test: verify created_at fix"""
import sys
sys.path.insert(0, '.')
from utils.database import DatabaseManager
import tempfile
import os

db_path = tempfile.mktemp(suffix='.db')
db = DatabaseManager(db_path)

print('🧪 created_at test...', flush=True)

scan_id = db.add_scan_result(scan_type='spm', surface_type='test', width=100, height=100)
print(f'✅ Scan created: id={scan_id}', flush=True)

scan = db.get_scan_by_id(scan_id)

if scan and scan.get('created_at'):
    print(f'✅ created_at: {scan["created_at"]}', flush=True)
    print('🎉 created_at FIX works!', flush=True)
else:
    print('❌ created_at still NULL!', flush=True)

db.close_pool()
os.remove(db_path)
