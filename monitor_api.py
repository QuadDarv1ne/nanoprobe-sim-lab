#!/usr/bin/env python
"""
Мониторинг API для Nanoprobe Sim Lab
Проверка здоровья, логирование ошибок, метрики
"""

import requests
import json
import time
import sys
from datetime import datetime
from pathlib import Path


API_URL = "http://localhost:8000"
LOG_FILE = Path("logs/api_monitor.log")
ERROR_LOG = Path("logs/api_errors.log")

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
        f.write(json.dumps(error_entry, ensure_ascii=False) + "\n")

    log_message(f"ERROR: {error}", "ERROR")


def check_health() -> dict:
    """Проверка health endpoint"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return {
            "status": "ok",
            "data": response.json(),
            "response_time_ms": response.elapsed.total_seconds() * 1000
        }
    except requests.exceptions.RequestException as e:
        return {
            "status": "error",
            "error": str(e)
        }


def check_health_detailed() -> dict:
    """Проверка detailed health"""
    try:
        response = requests.get(f"{API_URL}/health/detailed", timeout=5)
        return {
            "status": "ok",
            "data": response.json(),
            "response_time_ms": response.elapsed.total_seconds() * 1000
        }
    except requests.exceptions.RequestException as e:
        return {
            "status": "error",
            "error": str(e)
        }


def check_api_endpoints() -> dict:
    """Проверка основных API эндпоинтов"""
    endpoints = [
        "/api/v1/scans",
        "/api/v1/simulations",
        "/api/v1/dashboard/stats",
        "/docs",
    ]

    results = {}
    for endpoint in endpoints:
        try:
            response = requests.get(f"{API_URL}{endpoint}", timeout=5)
            results[endpoint] = {
                "status": response.status_code,
                "ok": response.status_code < 400,
                "time_ms": response.elapsed.total_seconds() * 1000
            }
        except requests.exceptions.RequestException as e:
            results[endpoint] = {
                "status": "error",
                "ok": False,
                "error": str(e)
            }

    return results


def monitor_loop(interval: int = 60):
    """Основный цикл мониторинга"""
    log_message("=" * 60)
    log_message("Запуск мониторинга API Nanoprobe Sim Lab")
    log_message(f"Интервал проверки: {interval} сек")
    log_message("=" * 60)

    consecutive_errors = 0
    max_errors = 5

    while True:
        try:
            # Проверка health
            health = check_health()
            if health["status"] == "ok":
                log_message(f"Health OK | Response: {health['response_time_ms']:.1f}ms")
                consecutive_errors = 0
            else:
                log_error("Health check failed", health.get("error", ""))
                consecutive_errors += 1

            # Проверка detailed health
            detailed = check_health_detailed()
            if detailed["status"] == "ok":
                data = detailed["data"]
                metrics = data.get("metrics", {})

                # Проверка метрик
                cpu = metrics.get("cpu", {}).get("percent", 0)
                memory = metrics.get("memory", {}).get("percent", 0)
                disk = metrics.get("disk", {}).get("percent", 0)

                log_message(
                    f"Metrics | CPU: {cpu:.1f}% | Memory: {memory:.1f}% | Disk: {disk:.1f}%"
                )

                # Предупреждения
                if disk > 90:
                    log_message("WARNING: Disk usage > 90%!", "WARNING")
                if memory > 80:
                    log_message("WARNING: Memory usage > 80%!", "WARNING")
                if cpu > 90:
                    log_message("WARNING: CPU usage > 90%!", "WARNING")
            else:
                log_error("Detailed health check failed", detailed.get("error", ""))

            # Проверка API эндпоинтов (каждые 5 циклов)
            if consecutive_errors % 5 == 0:
                endpoints = check_api_endpoints()
                failed = [ep for ep, r in endpoints.items() if not r.get("ok", True)]
                if failed:
                    log_message(f"Failed endpoints: {failed}", "WARNING")

            # Проверка на критические ошибки
            if consecutive_errors >= max_errors:
                log_message(
                    f"CRITICAL: {consecutive_errors} consecutive errors! "
                    "API may be down.",
                    "CRITICAL"
                )

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
