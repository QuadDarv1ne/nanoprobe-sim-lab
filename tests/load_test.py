#!/usr/bin/env python
"""
Load Testing для Nanoprobe Sim Lab API

Тестирование производительности API под нагрузкой

Использование:
    python tests/load_test.py
    
    # Быстрый тест (30 сек)
    python tests/load_test.py --duration 30
    
    # Тест с большим количеством пользователей
    python tests/load_test.py --users 50 --duration 120

Требования:
    pip install locust
"""

import sys
import os
import time
import json
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent.parent))

# Попытка импорта requests
try:
    import requests
    from requests.exceptions import RequestException, Timeout
except ImportError:
    print("❌ Установите requests: pip install requests")
    sys.exit(1)


@dataclass
class LoadTestResult:
    """Результат нагрузочного теста"""
    endpoint: str
    method: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    min_response_time: float
    max_response_time: float
    avg_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    duration_seconds: float
    errors: List[str]


class LoadTester:
    """Нагрузочное тестирование API"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        users: int = 10,
        duration: int = 60,
        timeout: int = 10,
    ):
        self.base_url = base_url
        self.users = users
        self.duration = duration
        self.timeout = timeout
        self.session = requests.Session()
        self.results: Dict[str, LoadTestResult] = {}
        
        # Тестовые данные
        self.test_scan_data = {
            "surface_type": "graphene",
            "resolution": 128,
            "scan_size": 1.0,
            "noise_level": 0.1,
        }

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> tuple:
        """Выполнение одного запроса"""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            if method == "GET":
                response = self.session.get(url, timeout=self.timeout)
            elif method == "POST":
                response = self.session.post(url, json=data, timeout=self.timeout)
            elif method == "PUT":
                response = self.session.put(url, json=data, timeout=self.timeout)
            elif method == "DELETE":
                response = self.session.delete(url, timeout=self.timeout)
            else:
                raise ValueError(f"Неизвестный метод: {method}")
            
            response_time = (time.time() - start_time) * 1000  # ms
            return response.status_code, response_time, None
            
        except Timeout:
            return 0, 0, "Timeout"
        except RequestException as e:
            return 0, 0, str(e)
        except Exception as e:
            return 0, 0, str(e)

    def _worker(
        self,
        endpoint: str,
        method: str,
        data: Optional[Dict] = None,
        results_dict: Optional[Dict] = None,
        worker_id: int = 0,
    ):
        """Рабочий поток для нагрузочного теста"""
        start_time = time.time()
        request_results = []
        
        while time.time() - start_time < self.duration:
            status_code, response_time, error = self._make_request(method, endpoint, data)
            request_results.append({
                "status_code": status_code,
                "response_time": response_time,
                "error": error,
                "timestamp": time.time(),
            })
            
            # Небольшая задержка между запросами
            time.sleep(0.1)
        
        if results_dict is not None:
            results_dict[worker_id] = request_results

    def test_endpoint(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict] = None,
        description: str = "",
    ) -> LoadTestResult:
        """Тестирование одного эндпоинта"""
        desc = description or f"{method} {endpoint}"
        print(f"\n🔵 Тест: {desc}")
        print(f"   Пользователей: {self.users}, Длительность: {self.duration}с")
        
        all_results = []
        threads = []
        results_dict = {}
        
        # Запуск потоков
        with ThreadPoolExecutor(max_workers=self.users) as executor:
            for i in range(self.users):
                thread = executor.submit(
                    self._worker,
                    endpoint,
                    method,
                    data,
                    results_dict,
                    i,
                )
                threads.append(thread)
        
        # Сбор результатов
        for worker_id, worker_results in results_dict.items():
            all_results.extend(worker_results)
        
        # Анализ результатов
        total = len(all_results)
        successful = sum(1 for r in all_results if 200 <= r["status_code"] < 400)
        failed = total - successful
        
        response_times = [r["response_time"] for r in all_results if r["response_time"] > 0]
        errors = [r["error"] for r in all_results if r["error"]]
        
        # Статистика
        if response_times:
            sorted_times = sorted(response_times)
            min_time = min(response_times)
            max_time = max(response_times)
            avg_time = statistics.mean(response_times)
            median_time = statistics.median(response_times)
            p95_time = sorted_times[int(len(sorted_times) * 0.95)] if len(sorted_times) > 1 else max_time
            p99_time = sorted_times[int(len(sorted_times) * 0.99)] if len(sorted_times) > 1 else max_time
        else:
            min_time = max_time = avg_time = median_time = p95_time = p99_time = 0
        
        success_rate = (successful / total * 100) if total > 0 else 0
        rps = total / self.duration if self.duration > 0 else 0
        
        result = LoadTestResult(
            endpoint=endpoint,
            method=method,
            total_requests=total,
            successful_requests=successful,
            failed_requests=failed,
            success_rate=success_rate,
            min_response_time=min_time,
            max_response_time=max_time,
            avg_response_time=avg_time,
            median_response_time=median_time,
            p95_response_time=p95_time,
            p99_response_time=p99_time,
            requests_per_second=rps,
            duration_seconds=self.duration,
            errors=errors[:10],  # Первые 10 ошибок
        )
        
        self.results[f"{method} {endpoint}"] = result
        
        # Вывод результатов
        self._print_result(result)
        
        return result

    def _print_result(self, result: LoadTestResult):
        """Вывод результатов теста"""
        status = "✅" if result.success_rate >= 95 else "⚠️" if result.success_rate >= 80 else "❌"
        
        print(f"\n   {status} Результаты:")
        print(f"      Запросов: {result.total_requests} (успешно: {result.successful_requests}, ошибок: {result.failed_requests})")
        print(f"      Success Rate: {result.success_rate:.1f}%")
        print(f"      RPS: {result.requests_per_second:.2f} запросов/сек")
        print(f"      Response Time:")
        print(f"         Min: {result.min_response_time:.2f}ms")
        print(f"         Max: {result.max_response_time:.2f}ms")
        print(f"         Avg: {result.avg_response_time:.2f}ms")
        print(f"         Median: {result.median_response_time:.2f}ms")
        print(f"         P95: {result.p95_response_time:.2f}ms")
        print(f"         P99: {result.p99_response_time:.2f}ms")
        
        if result.errors:
            print(f"      Ошибки ({len(result.errors)}):")
            for error in result.errors[:3]:
                print(f"         - {error}")

    def run_full_test(self) -> Dict[str, LoadTestResult]:
        """Запуск полного набора тестов"""
        print("=" * 70)
        print("🚀 Load Testing: Nanoprobe Sim Lab API")
        print("=" * 70)
        print(f"Base URL: {self.base_url}")
        print(f"Users: {self.users}, Duration: {self.duration}с")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # Health check перед тестом
        print("\n📋 Проверка доступности API...")
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ API доступен: {response.json()}")
            else:
                print(f"⚠️ API вернул статус {response.status_code}")
        except Exception as e:
            print(f"❌ API недоступен: {e}")
            return {}
        
        # Тестирование эндпоинтов
        endpoints = [
            ("/health", "GET", "Health check"),
            ("/health/detailed", "GET", "Detailed health"),
            ("/api/v1/dashboard/stats", "GET", "Dashboard statistics"),
            ("/api/v1/scans/", "GET", "List scans"),
            ("/api/v1/simulations/", "GET", "List simulations"),
            ("/metrics/realtime", "GET", "Realtime metrics"),
        ]
        
        # POST тест (создание сканирования)
        # endpoints.append(("/api/v1/scans/", "POST", "Create scan", self.test_scan_data))
        
        for endpoint_info in endpoints:
            if len(endpoint_info) == 3:
                endpoint, method, description = endpoint_info
                self.test_endpoint(endpoint, method, description=description)
            else:
                endpoint, method, description, data = endpoint_info
                self.test_endpoint(endpoint, method, data, description)
        
        # Итоговый отчёт
        self._print_summary()
        
        return self.results

    def _print_summary(self):
        """Вывод итогового отчёта"""
        print("\n" + "=" * 70)
        print("📊 ИТОГОВЫЙ ОТЧЁТ")
        print("=" * 70)
        
        total_requests = sum(r.total_requests for r in self.results.values())
        total_successful = sum(r.successful_requests for r in self.results.values())
        total_failed = sum(r.failed_requests for r in self.results.values())
        overall_success_rate = (total_successful / total_requests * 100) if total_requests > 0 else 0
        
        all_rps = [r.requests_per_second for r in self.results.values()]
        avg_rps = statistics.mean(all_rps) if all_rps else 0
        
        all_avg_times = [r.avg_response_time for r in self.results.values()]
        overall_avg_time = statistics.mean(all_avg_times) if all_avg_times else 0
        
        print(f"\n📈 Общая статистика:")
        print(f"   Всего запросов: {total_requests}")
        print(f"   Успешно: {total_successful}")
        print(f"   Ошибок: {total_failed}")
        print(f"   Success Rate: {overall_success_rate:.1f}%")
        print(f"   Средний RPS: {avg_rps:.2f} запросов/сек")
        print(f"   Среднее время ответа: {overall_avg_time:.2f}ms")
        
        print(f"\n📋 Результаты по эндпоинтам:")
        print(f"   {'Endpoint':<40} {'RPS':>8} {'Avg (ms)':>10} {'P95 (ms)':>10} {'Success':>8}")
        print(f"   {'-'*40} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
        
        for key, result in self.results.items():
            endpoint_name = f"{result.method} {result.endpoint}"
            if len(endpoint_name) > 40:
                endpoint_name = endpoint_name[:37] + "..."
            
            success_status = "✅" if result.success_rate >= 95 else "⚠️" if result.success_rate >= 80 else "❌"
            
            print(f"   {endpoint_name:<40} {result.requests_per_second:>8.2f} {result.avg_response_time:>10.2f} {result.p95_response_time:>10.2f} {success_status} {result.success_rate:.0f}%")
        
        # Рекомендации
        print(f"\n💡 Рекомендации:")
        
        if overall_success_rate < 95:
            print("   ⚠️  Success Rate ниже 95% - проверьте логи ошибок")
        
        if overall_avg_time > 1000:
            print("   ⚠️  Среднее время ответа > 1с - рассмотрите кэширование")
        
        if overall_avg_time > 5000:
            print("   ❌  Критически высокое время ответа - требуется оптимизация")
        
        if avg_rps < 10:
            print("   ℹ️  Низкий RPS - возможно ограничение на стороне сервера")
        
        print("=" * 70)


def main():
    """Точка входа"""
    parser = argparse.ArgumentParser(description="Load Testing для Nanoprobe API")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--users",
        type=int,
        default=10,
        help="Количество одновременных пользователей (default: 10)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Длительность теста в секундах (default: 60)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Таймаут запроса в секундах (default: 10)",
    )
    
    args = parser.parse_args()
    
    tester = LoadTester(
        base_url=args.url,
        users=args.users,
        duration=args.duration,
        timeout=args.timeout,
    )
    
    results = tester.run_full_test()
    
    # Сохранение результатов в JSON
    if results:
        output_file = Path(__file__).parent / "load_test_results.json"
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "base_url": args.url,
                "users": args.users,
                "duration": args.duration,
            },
            "results": {
                key: {
                    "endpoint": r.endpoint,
                    "method": r.method,
                    "total_requests": r.total_requests,
                    "successful_requests": r.successful_requests,
                    "failed_requests": r.failed_requests,
                    "success_rate": r.success_rate,
                    "avg_response_time": r.avg_response_time,
                    "p95_response_time": r.p95_response_time,
                    "requests_per_second": r.requests_per_second,
                }
                for key, r in results.items()
            },
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Результаты сохранены в: {output_file}")
    
    return 0 if all(r.success_rate >= 95 for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
