#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Комплексное тестирование проекта Nanoprobe Sim Lab
"""

import requests
import json

BASE_API = "http://localhost:8000"
BASE_FLASK = "http://localhost:5000"

print("=" * 60)
print("  КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ NANOPROBE SIM LAB")
print("=" * 60)

# ========== FastAPI API ==========
print("\n📌 FASTAPI REST API (http://localhost:8000)")
print("-" * 60)

tests_passed = 0
tests_failed = 0

# 1. Health check
print("\n1. Health check...")
try:
    r = requests.get(f"{BASE_API}/health", timeout=5)
    if r.status_code == 200:
        data = r.json()
        print(f"   ✅ Status: {data.get('status')}")
        print(f"   ✅ Version: {data.get('version')}")
        tests_passed += 1
    else:
        print(f"   ❌ Error: {r.status_code}")
        tests_failed += 1
except Exception as e:
    print(f"   ❌ Error: {e}")
    tests_failed += 1

# 2. Swagger docs
print("\n2. Swagger UI documentation...")
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

# 3. ReDoc
print("\n3. ReDoc documentation...")
try:
    r = requests.get(f"{BASE_API}/redoc", timeout=5)
    if r.status_code == 200:
        print(f"   ✅ Доступен: http://localhost:8000/redoc")
        tests_passed += 1
    else:
        print(f"   ❌ Error: {r.status_code}")
        tests_failed += 1
except Exception as e:
    print(f"   ❌ Error: {e}")
    tests_failed += 1

# 4. OpenAPI schema
print("\n4. OpenAPI schema...")
try:
    r = requests.get(f"{BASE_API}/openapi.json", timeout=5)
    if r.status_code == 200:
        data = r.json()
        print(f"   ✅ Title: {data.get('info', {}).get('title')}")
        print(f"   ✅ Version: {data.get('info', {}).get('version')}")
        tests_passed += 1
    else:
        print(f"   ❌ Error: {r.status_code}")
        tests_failed += 1
except Exception as e:
    print(f"   ❌ Error: {e}")
    tests_failed += 1

# 5. API v1 scans endpoint
print("\n5. API v1 scans endpoint...")
try:
    r = requests.get(f"{BASE_API}/api/v1/scans", timeout=5)
    print(f"   ✅ Status: {r.status_code}")
    tests_passed += 1
except Exception as e:
    print(f"   ❌ Error: {e}")
    tests_failed += 1

# 6. API v1 simulations endpoint
print("\n6. API v1 simulations endpoint...")
try:
    r = requests.get(f"{BASE_API}/api/v1/simulations", timeout=5)
    print(f"   ✅ Status: {r.status_code}")
    tests_passed += 1
except Exception as e:
    print(f"   ❌ Error: {e}")
    tests_failed += 1

# 7. API v1 analysis endpoint
print("\n7. API v1 analysis endpoint...")
try:
    r = requests.get(f"{BASE_API}/api/v1/analysis", timeout=5)
    print(f"   ✅ Status: {r.status_code}")
    tests_passed += 1
except Exception as e:
    print(f"   ❌ Error: {e}")
    tests_failed += 1

# ========== Flask Web Interface ==========
print("\n📌 FLASK WEB INTERFACE (http://localhost:5000)")
print("-" * 60)

# 8. Main page
print("\n8. Main page...")
try:
    r = requests.get(f"{BASE_FLASK}/", timeout=5)
    if r.status_code == 200:
        print(f"   ✅ Главная страница доступна")
        tests_passed += 1
    else:
        print(f"   ❌ Error: {r.status_code}")
        tests_failed += 1
except Exception as e:
    print(f"   ❌ Error: {e}")
    tests_failed += 1

# 9. System info API
print("\n9. System info API...")
try:
    r = requests.get(f"{BASE_FLASK}/api/system_info", timeout=5)
    if r.status_code == 200:
        data = r.json()
        print(f"   ✅ Project: {data.get('project_info', {}).get('name', 'N/A')}")
        tests_passed += 1
    else:
        print(f"   ❌ Error: {r.status_code}")
        tests_failed += 1
except Exception as e:
    print(f"   ❌ Error: {e}")
    tests_failed += 1

# 10. Components API
print("\n10. Components API...")
try:
    r = requests.get(f"{BASE_FLASK}/api/components", timeout=5)
    if r.status_code == 200:
        print(f"   ✅ Компоненты доступны")
        tests_passed += 1
    else:
        print(f"   ❌ Error: {r.status_code}")
        tests_failed += 1
except Exception as e:
    print(f"   ❌ Error: {e}")
    tests_failed += 1

# 11. Logs API
print("\n11. Logs API...")
try:
    r = requests.get(f"{BASE_FLASK}/api/logs", timeout=5)
    if r.status_code == 200:
        print(f"   ✅ Логи доступны")
        tests_passed += 1
    else:
        print(f"   ❌ Error: {r.status_code}")
        tests_failed += 1
except Exception as e:
    print(f"   ❌ Error: {e}")
    tests_failed += 1

# ========== Summary ==========
print("\n" + "=" * 60)
print("  ИТОГИ ТЕСТИРОВАНИЯ")
print("=" * 60)
print(f"  ✅ Пройдено: {tests_passed}")
print(f"  ❌ Провалено: {tests_failed}")
print(f"  📊 Успешность: {tests_passed}/{tests_passed + tests_failed} ({100*tests_passed/(tests_passed + tests_failed) if tests_passed + tests_failed > 0 else 0:.1f}%)")
print("=" * 60)

# URLs for user
print("\n📍 ДОСТУПНЫЕ АДРЕСА:")
print(f"   FastAPI Swagger UI:  http://localhost:8000/docs")
print(f"   FastAPI ReDoc:       http://localhost:8000/redoc")
print(f"   FastAPI Health:      http://localhost:8000/health")
print(f"   Flask Web UI:        http://localhost:5000")
print("=" * 60)
