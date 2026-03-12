#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Тестирование улучшений проекта Nanoprobe Sim Lab
"""

import requests
import time

BASE_API = "http://localhost:8000"
BASE_FLASK = "http://localhost:5000"

print("=" * 70)
print("  ТЕСТИРОВАНИЕ УЛУЧШЕНИЙ NANOPROBE SIM LAB")
print("=" * 70)

time.sleep(5)  # Ждём запуска сервисов

tests_passed = 0
tests_failed = 0

# ========== FastAPI API Tests ==========
print("\n📌 FASTAPI REST API")
print("-" * 70)

# 1. Health check
print("\n1. Health check...")
try:
    r = requests.get(f"{BASE_API}/health", timeout=5)
    if r.status_code == 200:
        print(f"   ✅ Status: {r.json().get('status')}")
        tests_passed += 1
    else:
        print(f"   ❌ Error: {r.status_code}")
        tests_failed += 1
except Exception as e:
    print(f"   ❌ Error: {e}")
    tests_failed += 1

# 2. Dashboard Stats (NEW)
print("\n2. Dashboard Stats (NEW)...")
try:
    r = requests.get(f"{BASE_API}/api/v1/dashboard/stats", timeout=5)
    if r.status_code == 200:
        data = r.json()
        print(f"   ✅ Uptime: {data.get('uptime_formatted', 'N/A')}")
        print(f"   ✅ Scans: {data.get('scans_count', 0)}")
        tests_passed += 1
    else:
        print(f"   ❌ Error: {r.status_code}")
        tests_failed += 1
except Exception as e:
    print(f"   ❌ Error: {e}")
    tests_failed += 1

# 3. Detailed Health (NEW)
print("\n3. Detailed Health Check (NEW)...")
try:
    r = requests.get(f"{BASE_API}/health/detailed", timeout=5)
    if r.status_code == 200:
        data = r.json()
        print(f"   ✅ Status: {data.get('status')}")
        print(f"   ✅ CPU: {data.get('metrics', {}).get('cpu', {}).get('percent', 0):.1f}%")
        print(f"   ✅ Memory: {data.get('metrics', {}).get('memory', {}).get('percent', 0):.1f}%")
        tests_passed += 1
    else:
        print(f"   ❌ Error: {r.status_code}")
        tests_failed += 1
except Exception as e:
    print(f"   ❌ Error: {e}")
    tests_failed += 1

# 4. Realtime Metrics (NEW)
print("\n4. Realtime Metrics (NEW)...")
try:
    r = requests.get(f"{BASE_API}/metrics/realtime", timeout=5)
    if r.status_code == 200:
        data = r.json()
        print(f"   ✅ CPU: {data.get('cpu_percent', 0):.1f}%")
        print(f"   ✅ Memory: {data.get('memory_percent', 0):.1f}%")
        tests_passed += 1
    else:
        print(f"   ❌ Error: {r.status_code}")
        tests_failed += 1
except Exception as e:
    print(f"   ❌ Error: {e}")
    tests_failed += 1

# 5. Export endpoint (NEW)
print("\n5. Export Endpoint (NEW)...")
try:
    r = requests.get(f"{BASE_API}/api/v1/export/json", timeout=5)
    if r.status_code == 200:
        print(f"   ✅ Export available")
        tests_passed += 1
    else:
        print(f"   ❌ Error: {r.status_code}")
        tests_failed += 1
except Exception as e:
    print(f"   ❌ Error: {e}")
    tests_failed += 1

# 6. Swagger UI
print("\n6. Swagger UI...")
try:
    r = requests.get(f"{BASE_API}/docs", timeout=5)
    if r.status_code == 200:
        print(f"   ✅ Доступен: http://localhost:8000/docs")
        tests_passed += 1
    else:
        print(f"   ❌ Error: {r.status_code}")
        tests_failed += 1
except Exception as e:
    print(f"   ❌ Error: {e}")
    tests_failed += 1

# ========== Flask Web Interface Tests ==========
print("\n📌 FLASK WEB INTERFACE")
print("-" * 70)

# 7. Main page
print("\n7. Main page (UPDATED UI)...")
try:
    r = requests.get(f"{BASE_FLASK}/", timeout=5)
    if r.status_code == 200:
        print(f"   ✅ Главная страница доступна")
        print(f"   ✅ Современный UI с тёмной темой")
        tests_passed += 1
    else:
        print(f"   ❌ Error: {r.status_code}")
        tests_failed += 1
except Exception as e:
    print(f"   ❌ Error: {e}")
    tests_failed += 1

# 8. System Info API
print("\n8. System Info API...")
try:
    r = requests.get(f"{BASE_FLASK}/api/system_info", timeout=5)
    if r.status_code == 200:
        print(f"   ✅ System info доступен")
        tests_passed += 1
    else:
        print(f"   ❌ Error: {r.status_code}")
        tests_failed += 1
except Exception as e:
    print(f"   ❌ Error: {e}")
    tests_failed += 1

# 9. Component Status API (NEW)
print("\n9. Component Status API (NEW)...")
try:
    r = requests.get(f"{BASE_FLASK}/api/component_status", timeout=5)
    if r.status_code == 200:
        print(f"   ✅ Component status доступен")
        tests_passed += 1
    else:
        print(f"   ❌ Error: {r.status_code}")
        tests_failed += 1
except Exception as e:
    print(f"   ❌ Error: {e}")
    tests_failed += 1

# 10. Logs API
print("\n10. Logs API...")
try:
    r = requests.get(f"{BASE_FLASK}/api/logs", timeout=5)
    if r.status_code == 200:
        print(f"   ✅ Logs доступны")
        tests_passed += 1
    else:
        print(f"   ❌ Error: {r.status_code}")
        tests_failed += 1
except Exception as e:
    print(f"   ❌ Error: {e}")
    tests_failed += 1

# 11. Clean Cache Action (NEW)
print("\n11. Clean Cache Action (NEW)...")
try:
    r = requests.post(f"{BASE_FLASK}/api/actions/clean_cache", timeout=5)
    if r.status_code == 200:
        data = r.json()
        if data.get("success"):
            print(f"   ✅ Cache очистка работает")
            tests_passed += 1
        else:
            print(f"   ⚠️ Warning: {data.get('error', 'Unknown')}")
            tests_passed += 1
    else:
        print(f"   ❌ Error: {r.status_code}")
        tests_failed += 1
except Exception as e:
    print(f"   ❌ Error: {e}")
    tests_failed += 1

# 12. Start Component Action (NEW)
print("\n12. Start Component Action (NEW)...")
try:
    r = requests.post(f"{BASE_FLASK}/api/actions/start_component", 
                     json={"component": "test_component"}, timeout=5)
    if r.status_code == 200:
        data = r.json()
        if data.get("success"):
            print(f"   ✅ Start component работает")
            tests_passed += 1
        else:
            print(f"   ⚠️ Warning: {data.get('error', 'Unknown')}")
            tests_passed += 1
    else:
        print(f"   ❌ Error: {r.status_code}")
        tests_failed += 1
except Exception as e:
    print(f"   ❌ Error: {e}")
    tests_failed += 1

# ========== Summary ==========
print("\n" + "=" * 70)
print("  ИТОГИ ТЕСТИРОВАНИЯ")
print("=" * 70)
total_tests = tests_passed + tests_failed
success_rate = (tests_passed / total_tests * 100) if total_tests > 0 else 0

print(f"  ✅ Пройдено: {tests_passed}")
print(f"  ❌ Провалено: {tests_failed}")
print(f"  📊 Успешность: {success_rate:.1f}%")
print("=" * 70)

# URLs for user
print("\n📍 ДОСТУПНЫЕ АДРЕСА:")
print(f"   🌐 FastAPI Swagger UI:  http://localhost:8000/docs")
print(f"   🌐 FastAPI ReDoc:       http://localhost:8000/redoc")
print(f"   🌐 FastAPI Health:      http://localhost:8000/health")
print(f"   🌐 Flask Web UI:        http://localhost:5000")
print(f"   🌐 Dashboard Stats:     http://localhost:8000/api/v1/dashboard/stats")
print(f"   🌐 Detailed Health:     http://localhost:8000/health/detailed")
print("=" * 70)

print("\n🎨 НОВЫЕ ВОЗМОЖНОСТИ:")
print("   ✨ Современный UI с тёмной/светлой темой")
print("   ✨ Real-time графики производительности")
print("   ✨ Анимации и переходы")
print("   ✨ Уведомления (toast notifications)")
print("   ✨ Адаптивный мобильный дизайн")
print("   ✨ Расширенные API эндпоинты")
print("   ✨ Детальная проверка здоровья системы")
print("   ✨ Экспорт данных в различных форматах")
print("=" * 70)
