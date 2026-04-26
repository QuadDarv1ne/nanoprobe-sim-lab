"""
API Performance Profiler для Nanoprobe Simulation Lab

Профилирование производительности API endpoints:
- Время ответа
- Использование ресурсов
- Статистика по endpoint'ам
- Генерация отчётов

Использование:
    python utils/testing/api_profiler.py
    python utils/testing/api_profiler.py --base-url http://localhost:8000 --requests 100
    python utils/testing/api_profiler.py --endpoint /health --iterations 50
"""

import argparse
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import httpx

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class EndpointStats:
    """Статистика для одного endpoint"""

    endpoint: str
    method: str
    iterations: int
    response_times: List[float] = field(default_factory=list)
    status_codes: Dict[int, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def min_time(self) -> float:
        return min(self.response_times) if self.response_times else 0.0

    @property
    def max_time(self) -> float:
        return max(self.response_times) if self.response_times else 0.0

    @property
    def avg_time(self) -> float:
        return statistics.mean(self.response_times) if self.response_times else 0.0

    @property
    def median_time(self) -> float:
        return statistics.median(self.response_times) if self.response_times else 0.0

    @property
    def p95_time(self) -> float:
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    @property
    def success_rate(self) -> float:
        if len(self.response_times) == 0:
            return 0.0
        # Count successful status codes (2xx and 3xx)
        success_count = sum(count for code, count in self.status_codes.items() if 200 <= code < 400)
        return (success_count / len(self.response_times)) * 100

    def add_result(self, response_time: float, status_code: int, error: Optional[str] = None):
        """Добавить результат запроса"""
        self.response_times.append(response_time)
        self.status_codes[status_code] = self.status_codes.get(status_code, 0) + 1
        if error:
            self.errors.append(error)


@dataclass
class ProfilerReport:
    """Отчёт о профилировании"""

    base_url: str
    total_requests: int = 0
    total_time: float = 0.0
    endpoint_stats: Dict[str, EndpointStats] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add_stats(self, stats: EndpointStats):
        """Добавить статистику endpoint"""
        key = f"{stats.method} {stats.endpoint}"
        self.endpoint_stats[key] = stats
        self.total_requests += stats.iterations

    def generate_summary(self) -> str:
        """Генерация текстового отчёта"""
        lines = [
            "\n" + "=" * 80,
            "API PERFORMANCE PROFILER REPORT",
            "=" * 80,
            f"Base URL: {self.base_url}",
            f"Timestamp: {self.timestamp}",
            f"Total Requests: {self.total_requests}",
            f"Total Time: {self.total_time:.2f}s",
            f"Average Rate: {self.total_requests / max(self.total_time, 0.001):.2f} req/s",
            "=" * 80,
        ]

        for key, stats in self.endpoint_stats.items():
            lines.extend(
                [
                    f"\n📊 {key}",
                    "-" * 80,
                    f"  Iterations: {stats.iterations}",
                    f"  Success Rate: {stats.success_rate:.1f}%",
                    f"  Min Time: {stats.min_time * 1000:.2f}ms",
                    f"  Max Time: {stats.max_time * 1000:.2f}ms",
                    f"  Avg Time: {stats.avg_time * 1000:.2f}ms",
                    f"  Median: {stats.median_time * 1000:.2f}ms",
                    f"  P95: {stats.p95_time * 1000:.2f}ms",
                ]
            )

            if stats.errors:
                lines.append(f"  Errors: {len(stats.errors)}")
                for error in stats.errors[:3]:  # Показать первые 3 ошибки
                    lines.append(f"    - {error}")

        lines.append("=" * 80)
        return "\n".join(lines)

    def save_json(self, filepath: str):
        """Сохранить отчёт в JSON формате"""
        data = {
            "base_url": self.base_url,
            "timestamp": self.timestamp,
            "total_requests": self.total_requests,
            "total_time": self.total_time,
            "average_rate": self.total_requests / max(self.total_time, 0.001),
            "endpoints": {
                key: {
                    "endpoint": stats.endpoint,
                    "method": stats.method,
                    "iterations": stats.iterations,
                    "min_time_ms": stats.min_time * 1000,
                    "max_time_ms": stats.max_time * 1000,
                    "avg_time_ms": stats.avg_time * 1000,
                    "median_time_ms": stats.median_time * 1000,
                    "p95_time_ms": stats.p95_time * 1000,
                    "success_rate": stats.success_rate,
                    "status_codes": stats.status_codes,
                    "error_count": len(stats.errors),
                }
                for key, stats in self.endpoint_stats.items()
            },
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Report saved to %s", filepath)


class APIProfiler:
    """Профилировщик производительности API"""

    def __init__(
        self,
        base_url: str,
        default_iterations: int = 10,
        timeout: float = 30.0,
        verbose: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.default_iterations = default_iterations
        self.timeout = timeout
        self.verbose = verbose
        self.report = ProfilerReport(base_url=base_url)
        self.client: Optional[httpx.Client] = None

    def start(self):
        """Начать профилирование"""
        logger.info("Starting API performance profiler for %s", self.base_url)
        logger.info("Default iterations per endpoint: %d", self.default_iterations)

        start_time = time.time()

        try:
            self.client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

            # Профилирование основных endpoints
            self._profile_health_endpoints()
            self._profile_public_endpoints()
            self._profile_auth_endpoints()

        except Exception as e:
            logger.exception("Profiler error: %s", e)
        finally:
            if self.client:
                self.client.close()

            self.report.total_time = time.time() - start_time
            self._print_report()

    def _profile_endpoint(
        self,
        endpoint: str,
        method: str = "GET",
        iterations: Optional[int] = None,
        headers: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> EndpointStats:
        """Профилировать один endpoint"""
        iterations = iterations or self.default_iterations
        stats = EndpointStats(endpoint=endpoint, method=method, iterations=iterations)

        logger.info("Profiling %s %s (%d iterations)...", method, endpoint, iterations)

        for i in range(iterations):
            try:
                start = time.time()

                if method.upper() == "GET":
                    response = self.client.get(endpoint, headers=headers, params=params)
                elif method.upper() == "POST":
                    response = self.client.post(endpoint, headers=headers, json=json_data)
                elif method.upper() == "PUT":
                    response = self.client.put(endpoint, headers=headers, json=json_data)
                elif method.upper() == "DELETE":
                    response = self.client.delete(endpoint, headers=headers)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                response_time = time.time() - start
                stats.add_result(response_time, response.status_code)

                if self.verbose and (i + 1) % 10 == 0:
                    logger.debug("  Iteration %d/%d: %dms", i + 1, iterations, response_time * 1000)

            except httpx.TimeoutException:
                stats.add_result(0, 0, "Timeout")
                logger.warning("  Timeout on iteration %d/%d", i + 1, iterations)
            except Exception as e:
                stats.add_result(0, 0, str(e))
                logger.warning("  Error on iteration %d/%d: %s", i + 1, iterations, e)

        self.report.add_stats(stats)
        return stats

    def _profile_health_endpoints(self):
        """Профилирование health endpoints"""
        logger.info("\n📊 Profiling Health Endpoints...")

        self._profile_endpoint("/health", "GET")
        self._profile_endpoint("/health/detailed", "GET")
        self._profile_endpoint("/", "GET")

    def _profile_public_endpoints(self):
        """Профилирование public endpoints"""
        logger.info("\n🌐 Profiling Public Endpoints...")

        endpoints = [
            ("/api/v1/scans", "GET"),
            ("/api/v1/simulations", "GET"),
            ("/api/v1/analysis/defects", "GET"),
            ("/api/v1/alerts", "GET"),
            ("/api/v1/sync/status", "GET"),
            ("/api/v1/monitoring/status", "GET"),
            ("/api/v1/dashboard/status", "GET"),
        ]

        for endpoint, method in endpoints:
            self._profile_endpoint(endpoint, method)

    def _profile_auth_endpoints(self):
        """Профилирование auth endpoints"""
        logger.info("\n🔐 Profiling Auth Endpoints...")

        # Login с невалидными данными (быстрый тест)
        self._profile_endpoint(
            "/api/v1/auth/login",
            "POST",
            iterations=5,
            json_data={"username": "test", "password": "test"},
        )

    def _print_report(self):
        """Вывести отчёт"""
        print(self.report.generate_summary())


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="API Performance Profiler for Nanoprobe Sim Lab")
    parser.add_argument(
        "--base-url",
        "-b",
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=10,
        help="Number of iterations per endpoint (default: 10)",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=float,
        default=30.0,
        help="Request timeout in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Save report to JSON file",
    )

    args = parser.parse_args()

    profiler = APIProfiler(
        base_url=args.base_url,
        default_iterations=args.iterations,
        timeout=args.timeout,
        verbose=args.verbose,
    )
    profiler.start()

    if args.output:
        profiler.report.save_json(args.output)


if __name__ == "__main__":
    main()
