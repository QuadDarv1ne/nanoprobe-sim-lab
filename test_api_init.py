#!/usr/bin/env python
"""Тест для проверки инициализации и работы API"""
import sys
import os
import traceback

print("Шаг 1: Импорт app")
try:
    from api.main import app
    print("✅ App импортирован успешно")
except Exception as e:
    print(f"❌ Ошибка импорта app: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\nШаг 2: Проверка routes")
try:
    routes = [route.path for route in app.routes]
    print(f"✅ Зарегистрировано {len(routes)} маршрутов")
    if '/health' in routes:
        print("✅ /health маршрут есть")
    else:
        print("❌ /health маршрут НЕ найден")
except Exception as e:
    print(f"❌ Ошибка проверки маршрутов: {e}")
    traceback.print_exc()

print("\nШаг 3: Тест /health endpoint")
try:
    from fastapi.testclient import TestClient
    client = TestClient(app)
    response = client.get("/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    if response.status_code == 200:
        print("✅ /health работает")
    else:
        print("❌ /health вернул ошибку")
except Exception as e:
    print(f"❌ Ошибка теста /health: {e}")
    traceback.print_exc()

print("\nГотово!")
