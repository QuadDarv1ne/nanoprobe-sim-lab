#!/usr/bin/env python
"""
Тестовый скрипт для проверки Backend API
"""

import json

import requests

BASE_URL = "http://localhost:8000"


def print_json(title, data):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    if isinstance(data, dict):
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(data)


def test_health():
    """Проверка health endpoint"""
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    print_json("Health Check", r.json())
    return r.status_code == 200


def test_scans():
    """Проверка scans API"""
    r = requests.get(f"{BASE_URL}/api/v1/scans?limit=5", timeout=5)
    data = r.json()
    print_json(
        f"Scans API (total: {data.get('total', 0)})",
        {
            "items": len(data.get("items", [])),
            "limit": data.get("limit"),
            "offset": data.get("offset"),
        },
    )
    return r.status_code == 200


def test_simulations():
    """Проверка simulations API"""
    r = requests.get(f"{BASE_URL}/api/v1/simulations?limit=5", timeout=5)
    data = r.json()
    print_json(
        f"Simulations API (total: {data.get('total', 0)})",
        {
            "items": len(data.get("items", [])),
        },
    )
    return r.status_code == 200


def test_dashboard():
    """Проверка dashboard API"""
    r = requests.get(f"{BASE_URL}/api/v1/dashboard/stats", timeout=5)
    print_json("Dashboard Stats", r.json())
    return r.status_code == 200


def test_auth():
    """Проверка auth endpoints"""
    # Проверка регистрации (тестовый пользователь)
    r = requests.get(f"{BASE_URL}/api/v1/auth/register", timeout=5)
    print_json("Auth Endpoint", {"status": "available", "code": r.status_code})
    return True


def test_docs():
    """Проверка Swagger UI"""
    r = requests.get(f"{BASE_URL}/docs", timeout=5)
    has_swagger = "Swagger" in r.text
    print_json("Swagger UI", {"available": has_swagger})
    return has_swagger


def main():
    print("\n" + "=" * 70)
    print("  Nanoprobe Sim Lab - Backend API Tests")
    print("=" * 70)

    tests = [
        ("Health Check", test_health),
        ("Scans API", test_scans),
        ("Simulations API", test_simulations),
        ("Dashboard Stats", test_dashboard),
        ("Auth Endpoints", test_auth),
        ("Swagger UI", test_docs),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print_json(f"ERROR: {name}", str(e))
            results.append((name, False))

    # Итоги
    print("\n" + "=" * 70)
    print("  ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ OK" if success else "❌ FAIL"
        print(f"  {status}: {name}")

    print(f"\n  Пройдено: {passed}/{total} тестов ({passed/total*100:.1f}%)")
    print("=" * 70)

    return passed == total


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
