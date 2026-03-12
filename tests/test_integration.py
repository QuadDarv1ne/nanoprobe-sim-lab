#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для тестирования интеграции Flask + FastAPI
Проверяет все компоненты интеграции и выводит отчёт
"""

import sys
import time
import requests
from datetime import datetime
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.integration import FlaskFastAPIIntegration, health_check, get_scans


class Colors:
    """Цвета для вывода в консоль"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Вывод заголовка"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")


def print_result(test_name, success, message=""):
    """Вывод результата теста"""
    status = f"{Colors.GREEN}✓ PASS{Colors.END}" if success else f"{Colors.RED}✗ FAIL{Colors.END}"
    print(f"  {status} | {test_name}")
    if message:
        color = Colors.GREEN if success else Colors.RED
        print(f"         {color}{message}{Colors.END}")


def test_fastapi_health(fastapi_url):
    """Тест 1: Проверка здоровья FastAPI"""
    print_header("Тест 1: Проверка здоровья FastAPI")
    
    try:
        start = time.time()
        response = requests.get(f"{fastapi_url}/health", timeout=5)
        elapsed = (time.time() - start) * 1000
        
        if response.status_code == 200:
            data = response.json()
            print_result("FastAPI health check", True, f"Status: {data.get('status', 'unknown')}, Time: {elapsed:.2f}ms")
            return True
        else:
            print_result("FastAPI health check", False, f"Status code: {response.status_code}")
            return False
    except requests.RequestException as e:
        print_result("FastAPI health check", False, str(e))
        return False


def test_flask_health(flask_url):
    """Тест 2: Проверка здоровья Flask"""
    print_header("Тест 2: Проверка здоровья Flask")
    
    try:
        start = time.time()
        response = requests.get(f"{flask_url}/api/system_info", timeout=5)
        elapsed = (time.time() - start) * 1000
        
        if response.status_code == 200:
            data = response.json()
            print_result("Flask health check", True, f"Time: {elapsed:.2f}ms")
            return True
        else:
            print_result("Flask health check", False, f"Status code: {response.status_code}")
            return False
    except requests.RequestException as e:
        print_result("Flask health check", False, str(e))
        return False


def test_integration_module():
    """Тест 3: Тестирование модуля интеграции"""
    print_header("Тест 3: Модуль интеграции")
    
    try:
        integration = FlaskFastAPIIntegration()
        
        # Проверка создания экземпляра
        print_result("Создание экземпляра", True, f"FastAPI URL: {integration.fastapi_url}")
        
        # Проверка health_check метода
        health = integration.health_check()
        fastapi_ok = health['fastapi']['status'] in ('healthy', 'unhealthy')
        flask_ok = health['flask']['status'] in ('healthy', 'unhealthy')
        
        print_result("Health check метод", fastapi_ok and flask_ok)
        
        if fastapi_ok:
            print(f"         FastAPI: {health['fastapi']['status']}")
        if flask_ok:
            print(f"         Flask: {health['flask']['status']}")
        
        return True
    except Exception as e:
        print_result("Модуль интеграции", False, str(e))
        return False


def test_api_endpoints(fastapi_url):
    """Тест 4: Тестирование API endpoints"""
    print_header("Тест 4: API Endpoints")
    
    results = []
    
    # Тест: GET /api/v1/scans
    try:
        response = requests.get(f"{fastapi_url}/api/v1/scans?limit=5", timeout=10)
        success = response.status_code in (200, 401)  # 401 OK если нет аутентификации
        print_result("GET /api/v1/scans", success, f"Status: {response.status_code}")
        results.append(success)
    except Exception as e:
        print_result("GET /api/v1/scans", False, str(e))
        results.append(False)
    
    # Тест: GET /api/v1/simulations
    try:
        response = requests.get(f"{fastapi_url}/api/v1/simulations?limit=5", timeout=10)
        success = response.status_code in (200, 401)
        print_result("GET /api/v1/simulations", success, f"Status: {response.status_code}")
        results.append(success)
    except Exception as e:
        print_result("GET /api/v1/simulations", False, str(e))
        results.append(False)
    
    # Тест: GET /docs (Swagger)
    try:
        response = requests.get(f"{fastapi_url}/docs", timeout=5)
        success = response.status_code == 200
        print_result("GET /docs (Swagger)", success, f"Status: {response.status_code}")
        results.append(success)
    except Exception as e:
        print_result("GET /docs (Swagger)", False, str(e))
        results.append(False)
    
    return all(results)


def test_reverse_proxy_import():
    """Тест 5: Проверка reverse proxy модуля"""
    print_header("Тест 5: Reverse Proxy модуль")
    
    try:
        from api.reverse_proxy import register_proxy, FASTAPI_URL, api_proxy
        print_result("Импорт reverse_proxy", True, f"FASTAPI_URL: {FASTAPI_URL}")
        return True
    except ImportError as e:
        print_result("Импорт reverse_proxy", False, str(e))
        return False


def test_database_connection():
    """Тест 6: Проверка подключения к БД"""
    print_header("Тест 6: Подключение к базе данных")
    
    try:
        from utils.database import DatabaseManager
        
        db = DatabaseManager("data/nanoprobe.db")
        stats = db.get_statistics()
        
        print_result("Подключение к SQLite", True)
        print(f"         Всего сканирований: {stats.get('total_scans', 0)}")
        print(f"         Всего симуляций: {stats.get('total_simulations', 0)}")
        print(f"         Всего сравнений: {stats.get('total_comparisons', 0)}")
        print(f"         Всего анализов: {stats.get('total_defect_analyses', 0)}")
        
        return True
    except Exception as e:
        print_result("Подключение к SQLite", False, str(e))
        return False


def test_utils_integration():
    """Тест 7: Тестирование utils модулей"""
    print_header("Тест 7: Utils модули")
    
    results = []
    
    # Defect Analyzer
    try:
        from utils.defect_analyzer import DefectAnalysisPipeline
        print_result("DefectAnalysisPipeline", True, "Импорт успешен")
        results.append(True)
    except Exception as e:
        print_result("DefectAnalysisPipeline", False, str(e))
        results.append(False)
    
    # Surface Comparator
    try:
        from utils.surface_comparator import SurfaceComparator
        print_result("SurfaceComparator", True, "Импорт успешен")
        results.append(True)
    except Exception as e:
        print_result("SurfaceComparator", False, str(e))
        results.append(False)
    
    # PDF Report Generator
    try:
        from utils.pdf_report_generator import ScientificPDFReport
        print_result("ScientificPDFReport", True, "Импорт успешен")
        results.append(True)
    except Exception as e:
        print_result("ScientificPDFReport", False, str(e))
        results.append(False)
    
    return all(results)


def run_all_tests(fastapi_url="http://localhost:8000", flask_url="http://localhost:5000"):
    """Запуск всех тестов"""
    print_header("🧪 ТЕСТИРОВАНИЕ ИНТЕГРАЦИИ FLASK + FASTAPI")
    
    print(f"  FastAPI URL: {fastapi_url}")
    print(f"  Flask URL: {flask_url}")
    print(f"  Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        "FastAPI Health": test_fastapi_health(fastapi_url),
        "Flask Health": test_flask_health(flask_url),
        "Integration Module": test_integration_module(),
        "API Endpoints": test_api_endpoints(fastapi_url),
        "Reverse Proxy": test_reverse_proxy_import(),
        "Database Connection": test_database_connection(),
        "Utils Modules": test_utils_integration(),
    }
    
    # Итоги
    print_header("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = f"{Colors.GREEN}✓{Colors.END}" if result else f"{Colors.RED}✗{Colors.END}"
        print(f"  {status} {test_name}")
    
    print(f"\n  {Colors.BOLD}Всего:{Colors.END} {passed}/{total} тестов пройдено")
    
    if passed == total:
        print(f"\n  {Colors.GREEN}{Colors.BOLD}🎉 Все тесты пройдены успешно!{Colors.END}\n")
        return 0
    else:
        failed = total - passed
        print(f"\n  {Colors.RED}{Colors.BOLD}⚠ {failed} тест(а) не пройдено{Colors.END}\n")
        return 1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Тестирование интеграции Flask + FastAPI')
    parser.add_argument('--fastapi-url', default='http://localhost:8000', help='URL FastAPI')
    parser.add_argument('--flask-url', default='http://localhost:5000', help='URL Flask')
    
    args = parser.parse_args()
    
    exit_code = run_all_tests(args.fastapi_url, args.flask_url)
    sys.exit(exit_code)
