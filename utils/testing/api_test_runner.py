"""
API Test Runner для Nanoprobe Simulation Lab

Автоматическое тестирование всех API endpoints с валидацией ответов.

Использование:
    python utils/testing/api_test_runner.py
    python utils/testing/api_test_runner.py --base-url http://localhost:8000
    python utils/testing/api_test_runner.py --verbose
"""

import argparse
import json
import logging
import sys
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
class TestResult:
    """Результат тестирования одного endpoint"""

    __test__ = False  # Prevent pytest collection

    endpoint: str
    method: str
    status_code: int
    expected_status: int
    response_time: float
    success: bool
    error: Optional[str] = None
    response_data: Optional[Dict] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class TestReport:
    """Отчёт о тестировании API"""

    __test__ = False  # Prevent pytest collection

    base_url: str
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    total_time: float = 0.0
    results: List[TestResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def success_rate(self) -> float:
        """Процент успешных тестов"""
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100

    def add_result(self, result: TestResult):
        """Добавить результат теста"""
        self.results.append(result)
        self.total_tests += 1
        if result.success:
            self.passed_tests += 1
        else:
            self.failed_tests += 1

    def generate_summary(self) -> str:
        """Генерация текстового отчёта"""
        lines = [
            "\n" + "=" * 70,
            "API TEST REPORT",
            "=" * 70,
            f"Base URL: {self.base_url}",
            f"Timestamp: {self.timestamp}",
            f"Total Tests: {self.total_tests}",
            f"Passed: {self.passed_tests} ✅",
            f"Failed: {self.failed_tests} ❌",
            f"Success Rate: {self.success_rate:.1f}%",
            f"Total Time: {self.total_time:.2f}s",
            "=" * 70,
        ]

        if self.failed_tests > 0:
            lines.append("\n❌ FAILED TESTS:")
            lines.append("-" * 70)
            for result in self.results:
                if not result.success:
                    lines.append(f"  {result.method} {result.endpoint}")
                    lines.append(
                        f"    Expected: {result.expected_status}, Got: {result.status_code}"
                    )
                    lines.append(f"    Error: {result.error}")
                    lines.append(f"    Time: {result.response_time:.3f}s")

        lines.append("=" * 70)
        return "\n".join(lines)

    def save_json(self, filepath: str):
        """Сохранить отчёт в JSON формате"""
        data = {
            "base_url": self.base_url,
            "timestamp": self.timestamp,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "success_rate": self.success_rate,
            "total_time": self.total_time,
            "results": [
                {
                    "endpoint": r.endpoint,
                    "method": r.method,
                    "status_code": r.status_code,
                    "expected_status": r.expected_status,
                    "response_time": r.response_time,
                    "success": r.success,
                    "error": r.error,
                    "timestamp": r.timestamp,
                }
                for r in self.results
            ],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Report saved to %s", filepath)


class APITestRunner:
    """Runner для тестирования API endpoints"""

    def __init__(self, base_url: str, timeout: float = 30.0, verbose: bool = False):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verbose = verbose
        self.report = TestReport(base_url=base_url)
        self.client: Optional[httpx.Client] = None

    def start(self):
        """Начать тестирование"""
        logger.info("Starting API tests for %s", self.base_url)
        start_time = time.time()

        try:
            self.client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

            # Тестирование endpoints
            self._test_health_endpoints()
            self._test_auth_endpoints()
            self._test_public_endpoints()

        except Exception as e:
            logger.exception("Test runner error: %s", e)
        finally:
            if self.client:
                self.client.close()

            self.report.total_time = time.time() - start_time
            self._print_report()

    def _log(self, message: str, level: str = "info"):
        """Логирование с поддержкой verbose режима"""
        if level == "info":
            logger.info(message)
        elif level == "debug" and self.verbose:
            logger.debug(message)

    def _run_test(
        self,
        endpoint: str,
        method: str = "GET",
        expected_status: int = 200,
        headers: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> TestResult:
        """Выполнить тест одного endpoint"""
        self._log(f"Testing: {method} {endpoint}", "debug")

        try:
            start_time = time.time()

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

            response_time = time.time() - start_time
            success = response.status_code == expected_status

            result = TestResult(
                endpoint=endpoint,
                method=method,
                status_code=response.status_code,
                expected_status=expected_status,
                response_time=response_time,
                success=success,
                error=(
                    None if success else f"Expected {expected_status}, got {response.status_code}"
                ),
                response_data=response.json() if response.content else None,
            )

            status_icon = "✅" if success else "❌"
            self._log(
                f"{status_icon} {method} {endpoint} - {response.status_code} ({response_time:.3f}s)"
            )

            self.report.add_result(result)
            return result

        except httpx.TimeoutException:
            response_time = time.time() - start_time
            result = TestResult(
                endpoint=endpoint,
                method=method,
                status_code=0,
                expected_status=expected_status,
                response_time=response_time,
                success=False,
                error="Request timeout",
            )
            self._log(f"❌ {method} {endpoint} - TIMEOUT", "info")
            self.report.add_result(result)
            return result
        except Exception as e:
            response_time = time.time() - start_time
            result = TestResult(
                endpoint=endpoint,
                method=method,
                status_code=0,
                expected_status=expected_status,
                response_time=response_time,
                success=False,
                error=str(e),
            )
            self._log(f"❌ {method} {endpoint} - ERROR: {e}", "info")
            self.report.add_result(result)
            return result

    def _test_health_endpoints(self):
        """Тестирование health check endpoints"""
        self._log("\n📊 Testing Health Endpoints...", "info")

        # Basic health check
        self._run_test("/health", "GET", 200)

        # Detailed health check
        self._run_test("/health/detailed", "GET", 200)

        # API root
        self._run_test("/", "GET", 200)

    def _test_auth_endpoints(self):
        """Тестирование auth endpoints"""
        self._log("\n🔐 Testing Auth Endpoints...", "info")

        # Login with invalid credentials (expect 401)
        self._run_test(
            "/api/v1/auth/login",
            "POST",
            401,
            json_data={"username": "invalid_user", "password": "invalid_password"},
        )

        # Auth endpoints require valid credentials, skip for now
        self._log("  ⚠️  Skipping authenticated endpoints (no valid credentials)", "info")

    def _test_public_endpoints(self):
        """Тестирование public endpoints"""
        self._log("\n🌐 Testing Public Endpoints...", "info")

        # Scans (empty list expected)
        self._run_test("/api/v1/scans", "GET", 200)

        # Simulations
        self._run_test("/api/v1/simulations", "GET", 200)

        # Analysis
        self._run_test("/api/v1/analysis/defects", "GET", 200)

        # Comparison
        self._run_test("/api/v1/comparison/surfaces", "POST", 200)

        # Batch status
        self._run_test("/api/v1/batch/status", "GET", 200)

        # Alerting
        self._run_test("/api/v1/alerts", "GET", 200)

        # Admin endpoints (may require admin token)
        self._run_test("/api/v1/admin/system-info", "GET", 200)

        # Dashboard endpoints
        self._run_test("/api/v1/dashboard/status", "GET", 200)

        # Sync manager
        self._run_test("/api/v1/sync/status", "GET", 200)

        # Monitoring
        self._run_test("/api/v1/monitoring/status", "GET", 200)

    def _print_report(self):
        """Вывести отчёт в консоль"""
        logger.info(self.report.generate_summary())

        if self.report.failed_tests > 0:
            logger.warning("%d tests failed", self.report.failed_tests)
            sys.exit(1)
        else:
            logger.info("All tests passed! ✅")
            sys.exit(0)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="API Test Runner for Nanoprobe Sim Lab")
    parser.add_argument(
        "--base-url",
        "-b",
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)",
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

    runner = APITestRunner(base_url=args.base_url, timeout=args.timeout, verbose=args.verbose)
    runner.start()

    if args.output:
        runner.report.save_json(args.output)


if __name__ == "__main__":
    main()
