#!/usr/bin/env python
"""
Мониторинг Frontend + Backend для Nanoprobe Sim Lab
Проверка здоровья, логирование ошибок, метрики
"""

import requests
import json
import time
import sys
from datetime import datetime
from pathlib import Path


BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:5000"
LOG_FILE = Path("logs/monitor.log")
ERROR_LOG = Path("logs/monitor_errors.log")

# Создание директории логов
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log_message(message: str, level: str = "INFO"):
    """Логирование сообщения"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{level}] {message}"
    print(log_entry)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")


def log_error(error: str, details: str = ""):
    """Логирование ошибки"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_entry = {
        "timestamp": timestamp,
        "error": error,
        "details": details
    }

    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(error_entry, ensure_ascii=False, indent=2) + "\n")

    log_message(f"ERROR: {error}", "ERROR")


def check_service(url: str, endpoint: str = "", timeout: int = 5) -> dict:
    """Проверка доступности сервиса"""
    try:
        full_url = f"{url}{endpoint}" if endpoint else url
        response = requests.get(full_url, timeout=timeout)
        return {
            "status": "ok",
            "status_code": response.status_code,
            "response_time_ms": response.elapsed.total_seconds() * 1000,
            "data": response.json() if response.headers.get("content-type", "").startswith("application/json") else None
        }
    except requests.exceptions.RequestException as e:
        return {
            "status": "error",
            "error": str(e)
        }


def check_backend() -> dict:
    """Проверка Backend (FastAPI)"""
    health = check_service(BACKEND_URL, "/health")
    detailed = check_service(BACKEND_URL, "/health/detailed")

    return {
        "name": "Backend (FastAPI)",
        "url": f"{BACKEND_URL}",
        "health": health,
        "detailed": detailed,
        "endpoints": check_backend_endpoints()
    }


def check_backend_endpoints() -> dict:
    """Проверка основных endpoint'ов Backend"""
    endpoints = {
        "/api/v1/scans": "Scans API",
        "/api/v1/simulations": "Simulations API",
        "/api/v1/dashboard/stats": "Dashboard API",
        "/docs": "Swagger UI"
    }

    results = {}
    for endpoint, name in endpoints.items():
        result = check_service(BACKEND_URL, endpoint, timeout=3)
        results[name] = result["status"] == "ok"

    return results


def check_frontend() -> dict:
    """Проверка Frontend (Flask)"""
    main_page = check_service(FRONTEND_URL)
    api_status = check_service(FRONTEND_URL, "/api/status")

    return {
        "name": "Frontend (Flask)",
        "url": f"{FRONTEND_URL}",
        "main_page": main_page,
        "api_status": api_status,
        "endpoints": check_frontend_endpoints()
    }


def check_frontend_endpoints() -> dict:
    """Проверка основных endpoint'ов Frontend"""
    endpoints = {
        "/api/status": "Status API",
        "/api/components": "Components API",
        "/api/logs": "Logs API"
    }

    results = {}
    for endpoint, name in endpoints.items():
        result = check_service(FRONTEND_URL, endpoint, timeout=3)
        results[name] = result["status"] == "ok"

    return results


def print_status(backend: dict, frontend: dict):
    """Вывод статуса в консоль"""
    print("\n" + "=" * 70)
    print("📊 NANOPROBE SIM LAB - МОНИТОРИНГ СИСТЕМЫ")
    print("=" * 70)

    # Backend статус
    b_health = backend.get("health", {})
    if b_health.get("status") == "ok":
        data = b_health.get("data", {})
        print(f"✅ BACKEND (FastAPI) - {data.get('status', 'unknown').upper()}")
        print(f"   URL: {backend['url']}")
        print(f"   Response: {b_health.get('response_time_ms', 0):.1f}ms")
        print(f"   Version: {data.get('version', 'N/A')}")
    else:
        print(f"❌ BACKEND (FastAPI) - OFFLINE")
        print(f"   Error: {b_health.get('error', 'Unknown')}")

    # Frontend статус
    f_main = frontend.get("main_page", {})
    if f_main.get("status") == "ok":
        print(f"✅ FRONTEND (Flask) - ONLINE ({f_main.get('status_code', 0)})")
        print(f"   URL: {frontend['url']}")
        print(f"   Response: {f_main.get('response_time_ms', 0):.1f}ms")
    else:
        print(f"❌ FRONTEND (Flask) - OFFLINE")
        print(f"   Error: {f_main.get('error', 'Unknown')}")

    # Детальные метрики (если доступны)
    b_detailed = backend.get("detailed", {})
    if b_detailed.get("status") == "ok":
        metrics = b_detailed.get("data", {}).get("metrics", {})
        print("\n📈 МЕТРИКИ:")

        cpu = metrics.get("cpu", {})
        print(f"   CPU: {cpu.get('percent', 0):.1f}% [{cpu.get('status', 'unknown')}]")

        memory = metrics.get("memory", {})
        print(f"   Memory: {memory.get('percent', 0):.1f}% [{memory.get('status', 'unknown')}]")

        disk = metrics.get("disk", {})
        disk_status = disk.get('status', 'unknown')
        disk_icon = "⚠️" if disk_status == "warning" else "✅"
        print(f"   Disk: {disk.get('percent', 0):.1f}% [{disk_status}] {disk_icon}")

    # Проверка endpoint'ов
    print("\n🔌 ENDPOINTS:")
    b_endpoints = backend.get("endpoints", {})
    for name, ok in b_endpoints.items():
        icon = "✅" if ok else "❌"
        print(f"   {icon} {name}")

    f_endpoints = frontend.get("endpoints", {})
    for name, ok in f_endpoints.items():
        icon = "✅" if ok else "❌"
        print(f"   {icon} {name}")

    print("=" * 70 + "\n")


def monitor_loop(interval: int = 30):
    """Основный цикл мониторинга"""
    log_message("=" * 70)
    log_message("Запуск мониторинга Nanoprobe Sim Lab (Frontend + Backend)")
    log_message(f"Backend: {BACKEND_URL} | Frontend: {FRONTEND_URL}")
    log_message(f"Интервал проверки: {interval} сек")
    log_message("=" * 70)

    consecutive_errors = 0
    max_errors = 5

    while True:
        try:
            # Проверка сервисов
            backend = check_backend()
            frontend = check_frontend()

            # Вывод статуса
            print_status(backend, frontend)

            # Логирование
            b_ok = backend.get("health", {}).get("status") == "ok"
            f_ok = frontend.get("main_page", {}).get("status") == "ok"

            if b_ok and f_ok:
                log_message("Все сервисы работают нормально")
                consecutive_errors = 0
            else:
                if not b_ok:
                    log_error("Backend offline", backend.get("health", {}).get("error", ""))
                if not f_ok:
                    log_error("Frontend offline", frontend.get("main_page", {}).get("error", ""))
                consecutive_errors += 1

            # Проверка на критические ошибки
            if consecutive_errors >= max_errors:
                log_message(f"CRITICAL: {consecutive_errors} consecutive errors!", "CRITICAL")

            # Проверка метрик
            b_detailed = backend.get("detailed", {})
            if b_detailed.get("status") == "ok":
                metrics = b_detailed.get("data", {}).get("metrics", {})
                disk = metrics.get("disk", {}).get("percent", 0)
                if disk > 90:
                    log_message(f"WARNING: Disk usage {disk:.1f}% > 90%", "WARNING")

        except Exception as e:
            log_error(f"Unexpected error: {e}", str(type(e)))
            consecutive_errors += 1

        time.sleep(interval)


if __name__ == "__main__":
    interval = 30  # секунды

    if len(sys.argv) > 1:
        try:
            interval = int(sys.argv[1])
        except ValueError:
            print(f"Использование: {sys.argv[0]} [interval_seconds]")
            sys.exit(1)

    try:
        monitor_loop(interval)
    except KeyboardInterrupt:
        log_message("Мониторинг остановлен пользователем")
        sys.exit(0)
