#!/usr/bin/env python
"""
Security Testing для Nanoprobe Sim Lab API

Тестирование безопасности API на уязвимости

Виды тестов:
- SQL Injection
- XSS (Cross-Site Scripting)
- Authentication Bypass
- Rate Limiting
- CORS Misconfiguration
- Security Headers
- Sensitive Data Exposure

Использование:
    python tests/security_test.py

    # Быстрый тест
    python tests/security_test.py --quick

    # Полный тест с отчётом
    python tests/security_test.py --full --report

Требования:
    pip install requests
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import requests
    from requests.exceptions import ConnectionError, RequestException, Timeout
except ImportError:
    print("❌ Установите requests: pip install requests")
    sys.exit(1)


class Severity(Enum):
    """Уровень критичности уязвимости"""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


@dataclass
class SecurityFinding:
    """Найденная уязвимость"""

    test_name: str
    severity: Severity
    endpoint: str
    description: str
    evidence: str
    recommendation: str
    cwe_id: Optional[str] = None
    cvss_score: Optional[float] = None


@dataclass
class SecurityTestResult:
    """Результат теста безопасности"""

    test_name: str
    passed: bool
    findings: List[SecurityFinding] = field(default_factory=list)
    duration_seconds: float = 0.0
    requests_made: int = 0


class SecurityTester:
    """Тестирование безопасности API"""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.results: List[SecurityTestResult] = []
        self.findings: List[SecurityFinding] = []
        self.access_token: Optional[str] = None

        # SQL Injection payloads
        self.sql_injection_payloads = [
            "' OR '1'='1",
            "' OR '1'='1' --",
            "' OR '1'='1' /*",
            "'; DROP TABLE users; --",
            "' UNION SELECT NULL, NULL, NULL --",
            "1; SELECT * FROM users",
            "admin'--",
            "' OR 1=1 --",
            "1' OR '1'='1",
        ]

        # XSS payloads
        self.xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "javascript:alert('XSS')",
            "<body onload=alert('XSS')>",
            "'\"><script>alert('XSS')</script>",
        ]

        # Path traversal payloads
        self.path_traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        ]

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[requests.Response]:
        """Выполнение HTTP запроса"""
        url = f"{self.base_url}{endpoint}"

        # Добавляем токен если есть
        if self.access_token and "headers" not in kwargs:
            kwargs["headers"] = {}
        if self.access_token:
            kwargs["headers"]["Authorization"] = f"Bearer {self.access_token}"

        try:
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
            return response
        except Timeout:
            return None
        except (ConnectionError, RequestException) as e:
            print(f"  ⚠️  Ошибка запроса: {e}")
            return None

    def _login(self) -> bool:
        """Получение токена доступа"""
        print("\n🔐 Аутентификация...")

        # Пробуем стандартные учётные данные
        credentials = [
            {"username": "admin", "password": "Admin123!"},
            {"username": "admin", "password": "admin"},
            {"username": "test", "password": "Test123!"},
        ]

        for creds in credentials:
            response = self._make_request("POST", "/api/v1/auth/login", json=creds)
            if response and response.status_code == 200:
                data = response.json()
                self.access_token = data.get("access_token")
                print(f"  ✅ Успешный вход как {creds['username']}")
                return True

        print("  ℹ️  Аутентификация не удалась (тесты без токена)")
        return False

    def test_security_headers(self) -> SecurityTestResult:
        """Тест 1: Проверка Security Headers"""
        print("\n📋 Тест 1: Security Headers")

        result = SecurityTestResult(test_name="Security Headers", passed=True)
        start_time = time.time()

        response = self._make_request("GET", "/health")
        result.requests_made += 1

        if not response:
            result.passed = False
            result.findings.append(
                SecurityFinding(
                    test_name="Security Headers",
                    severity=Severity.HIGH,
                    endpoint="/health",
                    description="API недоступен",
                    evidence="No response",
                    recommendation="Проверьте доступность API",
                )
            )
            return result

        # Проверка headers
        required_headers = {
            "X-Frame-Options": ("Clickjacking защита", Severity.MEDIUM, "CWE-1021"),
            "X-Content-Type-Options": ("MIME Sniffing защита", Severity.MEDIUM, "CWE-693"),
            "X-XSS-Protection": ("XSS защита", Severity.LOW, "CWE-79"),
            "Referrer-Policy": ("Referrer контроль", Severity.LOW, "CWE-200"),
            "Content-Security-Policy": ("CSP защита", Severity.MEDIUM, "CWE-79"),
            "Strict-Transport-Security": ("HSTS (только HTTPS)", Severity.HIGH, "CWE-319"),
        }

        missing_headers = []
        for header, (desc, severity, cwe) in required_headers.items():
            if header not in response.headers:
                missing_headers.append((header, desc, severity, cwe))
                result.passed = False

        if missing_headers:
            for header, desc, severity, cwe in missing_headers:
                finding = SecurityFinding(
                    test_name="Security Headers",
                    severity=severity,
                    endpoint="/health",
                    description=f"Отсутствует заголовок {header}: {desc}",
                    evidence=f"Headers: {dict(response.headers)}",
                    recommendation=f"Добавьте заголовок {header}",
                    cwe_id=cwe,
                )
                result.findings.append(finding)
                self.findings.append(finding)
        else:
            print("  ✅ Все security headers присутствуют")

        # Проверка на утечку информации
        if "Server" in response.headers or "X-Powered-By" in response.headers:
            finding = SecurityFinding(
                test_name="Security Headers",
                severity=Severity.LOW,
                endpoint="/health",
                description="Утечка информации о сервере",
                evidence=f"Server: {response.headers.get('Server', 'N/A')}",
                recommendation="Удалите заголовки Server и X-Powered-By",
                cwe_id="CWE-200",
            )
            result.findings.append(finding)
            self.findings.append(finding)
            print("  ⚠️  Заголовки сервера не удалены")

        result.duration_seconds = time.time() - start_time
        self.results.append(result)

        return result

    def test_sql_injection(self) -> SecurityTestResult:
        """Тест 2: SQL Injection"""
        print("\n📋 Тест 2: SQL Injection")

        result = SecurityTestResult(test_name="SQL Injection", passed=True)
        start_time = time.time()

        # Тестовые эндпоинты с параметрами
        test_endpoints = [
            ("/api/v1/scans/", "POST", {"surface_type": "PAYLOAD", "resolution": 128}),
            ("/api/v1/simulations/", "POST", {"surface_type": "PAYLOAD", "resolution": 64}),
        ]

        sql_errors = [
            "SQL",
            "syntax",
            "database",
            "mysql",
            "postgresql",
            "sqlite",
            "ORA-",
            "Microsoft SQL Server",
            "Unclosed quotation mark",
        ]

        for endpoint, method, template in test_endpoints:
            for payload in self.sql_injection_payloads[:5]:  # Первые 5 payload
                test_data = template.copy()
                test_data["surface_type"] = payload

                response = self._make_request(method, endpoint, json=test_data)
                result.requests_made += 1

                if response and response.status_code >= 500:
                    response_text = response.text.lower()
                    if any(err.lower() in response_text for err in sql_errors):
                        result.passed = False
                        finding = SecurityFinding(
                            test_name="SQL Injection",
                            severity=Severity.CRITICAL,
                            endpoint=endpoint,
                            description=f"Возможна SQL Injection уязвимость",
                            evidence=f"Payload: {payload}\nResponse: {response.text[:200]}",
                            recommendation="Используйте параметризованные запросы",
                            cwe_id="CWE-89",
                            cvss_score=9.8,
                        )
                        result.findings.append(finding)
                        self.findings.append(finding)
                        print(f"  ❌ Возможна SQL Injection: {endpoint}")

        if result.passed:
            print("  ✅ SQL Injection уязвимостей не найдено")

        result.duration_seconds = time.time() - start_time
        self.results.append(result)

        return result

    def test_xss(self) -> SecurityTestResult:
        """Тест 3: XSS (Cross-Site Scripting)"""
        print("\n📋 Тест 3: XSS (Cross-Site Scripting)")

        result = SecurityTestResult(test_name="XSS", passed=True)
        start_time = time.time()

        # Тестовые эндпоинты
        test_endpoints = [
            ("/api/v1/scans/", "POST"),
            ("/api/v1/simulations/", "POST"),
        ]

        for endpoint, method in test_endpoints:
            for payload in self.xss_payloads[:3]:  # Первые 3 payload
                test_data = {
                    "surface_type": payload,
                    "resolution": 128,
                    "scan_size": 1.0,
                }

                response = self._make_request(method, endpoint, json=test_data)
                result.requests_made += 1

                if response and response.status_code == 200:
                    try:
                        data = response.json()
                        response_text = json.dumps(data)

                        # Проверка на отражение payload
                        if payload in response_text or payload.replace("'", "%27") in response_text:
                            # Проверка на отсутствие экранирования
                            if "<script>" in response_text or "alert(" in response_text:
                                result.passed = False
                                finding = SecurityFinding(
                                    test_name="XSS",
                                    severity=Severity.HIGH,
                                    endpoint=endpoint,
                                    description=f"Возможна XSS уязвимость",
                                    evidence=f"Payload: {payload}",
                                    recommendation="Экранируйте пользовательские данные",
                                    cwe_id="CWE-79",
                                    cvss_score=7.5,
                                )
                                result.findings.append(finding)
                                self.findings.append(finding)
                    except (json.JSONDecodeError, KeyError):
                        pass

        if result.passed:
            print("  ✅ XSS уязвимостей не найдено")

        result.duration_seconds = time.time() - start_time
        self.results.append(result)

        return result

    def test_authentication_bypass(self) -> SecurityTestResult:
        """Тест 4: Authentication Bypass"""
        print("\n📋 Тест 4: Authentication Bypass")

        result = SecurityTestResult(test_name="Authentication Bypass", passed=True)
        start_time = time.time()

        # Защищённые эндпоинты (требуют аутентификации)
        protected_endpoints = [
            "/api/v1/admin/users",
            "/api/v1/admin/settings",
            "/api/v1/scans/",
            "/api/v1/simulations/",
        ]

        for endpoint in protected_endpoints:
            # Попытка доступа без токена
            response = self._make_request("GET", endpoint)
            result.requests_made += 1

            if response and response.status_code == 200:
                result.passed = False
                finding = SecurityFinding(
                    test_name="Authentication Bypass",
                    severity=Severity.CRITICAL,
                    endpoint=endpoint,
                    description=f"Доступ к защищённому ресурсу без аутентификации",
                    evidence=f"Status: {response.status_code}",
                    recommendation="Требуйте аутентификацию для защищённых ресурсов",
                    cwe_id="CWE-287",
                    cvss_score=9.0,
                )
                result.findings.append(finding)
                self.findings.append(finding)
                print(f"  ❌ Доступ без аутентификации: {endpoint}")
            elif response and response.status_code in [401, 403]:
                print(f"  ✅ Защищено: {endpoint} ({response.status_code})")

        result.duration_seconds = time.time() - start_time
        self.results.append(result)

        return result

    def test_rate_limiting(self) -> SecurityTestResult:
        """Тест 5: Rate Limiting"""
        print("\n📋 Тест 5: Rate Limiting")

        result = SecurityTestResult(test_name="Rate Limiting", passed=True)
        start_time = time.time()

        endpoint = "/health"
        rate_limit_detected = False
        status_codes = []

        # Быстрые запросы (20 за 5 секунд)
        for i in range(20):
            response = self._make_request("GET", endpoint)
            result.requests_made += 1

            if response:
                status_codes.append(response.status_code)

                # Проверка на 429 Too Many Requests
                if response.status_code == 429:
                    rate_limit_detected = True
                    print(f"  ✅ Rate Limiting обнаружен после {i+1} запросов")
                    break

            time.sleep(0.25)

        if not rate_limit_detected:
            # Проверка на наличие заголовков rate limiting
            if status_codes:
                response = self._make_request("GET", endpoint)
                result.requests_made += 1

                rate_limit_headers = [
                    "X-RateLimit-Limit",
                    "X-RateLimit-Remaining",
                    "X-RateLimit-Reset",
                    "Retry-After",
                ]

                has_rate_limit_headers = any(h in response.headers for h in rate_limit_headers)

                if not has_rate_limit_headers:
                    result.passed = False
                    finding = SecurityFinding(
                        test_name="Rate Limiting",
                        severity=Severity.MEDIUM,
                        endpoint=endpoint,
                        description="Rate Limiting не обнаружен",
                        evidence=f"20 запросов без ограничений (статусы: {set(status_codes)})",
                        recommendation="Внедрите rate limiting для защиты от DDoS/bruteforce",
                        cwe_id="CWE-770",
                    )
                    result.findings.append(finding)
                    self.findings.append(finding)
                    print("  ⚠️  Rate Limiting не обнаружен")
                else:
                    print("  ✅ Rate Limiting заголовки присутствуют")
        else:
            print("  ✅ Rate Limiting работает")

        result.duration_seconds = time.time() - start_time
        self.results.append(result)

        return result

    def test_cors(self) -> SecurityTestResult:
        """Тест 6: CORS Configuration"""
        print("\n📋 Тест 6: CORS Configuration")

        result = SecurityTestResult(test_name="CORS", passed=True)
        start_time = time.time()

        endpoint = "/health"

        # Проверка CORS с malicious origin
        malicious_origins = [
            "https://evil.com",
            "https://attacker.com",
            "null",
        ]

        for origin in malicious_origins:
            headers = {"Origin": origin}
            response = self._make_request("GET", endpoint, headers=headers)
            result.requests_made += 1

            if response:
                acao = response.headers.get("Access-Control-Allow-Origin")

                # Проверка на wildcard или отражение malicious origin
                if acao == "*" or acao == origin:
                    acac = response.headers.get("Access-Control-Allow-Credentials")

                    if acac == "true":
                        result.passed = False
                        finding = SecurityFinding(
                            test_name="CORS",
                            severity=Severity.HIGH,
                            endpoint=endpoint,
                            description=f"Небезопасная CORS конфигурация: {origin}",
                            evidence=f"Access-Control-Allow-Origin: {acao}\nAccess-Control-Allow-Credentials: {acac}",
                            recommendation="Ограничьте CORS доверенными доменами",
                            cwe_id="CWE-942",
                        )
                        result.findings.append(finding)
                        self.findings.append(finding)
                        print(f"  ❌ Небезопасный CORS: {origin}")

        if result.passed:
            print("  ✅ CORS конфигурация безопасна")

        result.duration_seconds = time.time() - start_time
        self.results.append(result)

        return result

    def test_sensitive_data_exposure(self) -> SecurityTestResult:
        """Тест 7: Sensitive Data Exposure"""
        print("\n📋 Тест 7: Sensitive Data Exposure")

        result = SecurityTestResult(test_name="Sensitive Data Exposure", passed=True)
        start_time = time.time()

        endpoints_to_check = [
            "/health",
            "/health/detailed",
            "/api/v1/dashboard/stats",
        ]

        sensitive_patterns = [
            (r"password", "Пароли в ответе"),
            (r"secret", "Секреты в ответе"),
            (r"api[_-]?key", "API ключи в ответе"),
            (r"token", "Токены в ответе"),
            (r"private[_-]?key", "Приватные ключи"),
        ]

        for endpoint in endpoints_to_check:
            response = self._make_request("GET", endpoint)
            result.requests_made += 1

            if response and response.status_code == 200:
                try:
                    response_text = response.text.lower()

                    for pattern, description in sensitive_patterns:
                        if re.search(pattern, response_text, re.IGNORECASE):
                            # Проверяем, не является ли это просто названием поля
                            if f'"{pattern}"' in response_text or f"'{pattern}'" in response_text:
                                finding = SecurityFinding(
                                    test_name="Sensitive Data Exposure",
                                    severity=Severity.MEDIUM,
                                    endpoint=endpoint,
                                    description=f"Возможная утечка данных: {description}",
                                    evidence=f"Pattern: {pattern}",
                                    recommendation="Удалите чувствительные данные из ответов API",
                                    cwe_id="CWE-200",
                                )
                                result.findings.append(finding)
                                self.findings.append(finding)
                                print(f"  ⚠️  Возможная утечка: {description} в {endpoint}")
                except Exception:
                    pass

        if not result.findings:
            print("  ✅ Утечек чувствительных данных не найдено")

        result.duration_seconds = time.time() - start_time
        self.results.append(result)

        return result

    def run_full_test(self) -> List[SecurityTestResult]:
        """Запуск полного набора тестов безопасности"""
        print("=" * 70)
        print("🔒 Security Testing: Nanoprobe Sim Lab API")
        print("=" * 70)
        print(f"Base URL: {self.base_url}")
        print(f"Start Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        # Попытка аутентификации
        self._login()

        # Запуск тестов
        tests = [
            self.test_security_headers,
            self.test_sql_injection,
            self.test_xss,
            self.test_authentication_bypass,
            self.test_rate_limiting,
            self.test_cors,
            self.test_sensitive_data_exposure,
        ]

        for test in tests:
            try:
                test()
            except Exception as e:
                print(f"  ❌ Ошибка теста: {e}")
                result = SecurityTestResult(
                    test_name=test.__name__,
                    passed=False,
                    findings=[
                        SecurityFinding(
                            test_name=test.__name__,
                            severity=Severity.HIGH,
                            endpoint="N/A",
                            description=f"Ошибка выполнения теста: {e}",
                            evidence=str(e),
                            recommendation="Проверьте логи API",
                        )
                    ],
                )
                self.results.append(result)

        # Итоговый отчёт
        self._print_summary()

        return self.results

    def _print_summary(self):
        """Вывод итогового отчёта"""
        print("\n" + "=" * 70)
        print("📊 ИТОГОВЫЙ ОТЧЁТ ПО БЕЗОПАСНОСТИ")
        print("=" * 70)

        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        _ = total_tests - passed_tests  # failed_tests, можно использовать в отчёте

        total_findings = len(self.findings)
        critical = sum(1 for f in self.findings if f.severity == Severity.CRITICAL)
        high = sum(1 for f in self.findings if f.severity == Severity.HIGH)
        medium = sum(1 for f in self.findings if f.severity == Severity.MEDIUM)
        low = sum(1 for f in self.findings if f.severity == Severity.LOW)

        total_requests = sum(r.requests_made for r in self.results)
        total_duration = sum(r.duration_seconds for r in self.results)

        print(f"\n📈 Статистика:")
        print(f"   Тестов пройдено: {passed_tests}/{total_tests}")
        print(f"   Найдено уязвимостей: {total_findings}")
        print(f"      🔴 Critical: {critical}")
        print(f"      🟠 High: {high}")
        print(f"      🟡 Medium: {medium}")
        print(f"      🟢 Low: {low}")
        print(f"   Всего запросов: {total_requests}")
        print(f"   Длительность: {total_duration:.2f}с")

        # Оценка безопасности
        if critical > 0:
            security_score = "❌ CRITICAL"
            recommendation = "Немедленно устраните критические уязвимости!"
        elif high > 0:
            security_score = "🟠 HIGH RISK"
            recommendation = "Требуется срочное исправление уязвимостей"
        elif medium > 0:
            security_score = "🟡 MEDIUM RISK"
            recommendation = "Рекомендуется исправление уязвимостей"
        elif low > 0:
            security_score = "🟢 LOW RISK"
            recommendation = "Небольшие улучшения безопасности"
        else:
            security_score = "✅ SECURE"
            recommendation = "Отличный уровень безопасности!"

        print(f"\n🎯 Оценка безопасности: {security_score}")
        print(f"💡 Рекомендация: {recommendation}")

        # Детали по уязвимостям
        if self.findings:
            print(f"\n📋 Найденные уязвимости:")
            for i, finding in enumerate(self.findings, 1):
                severity_icon = {
                    Severity.CRITICAL: "🔴",
                    Severity.HIGH: "🟠",
                    Severity.MEDIUM: "🟡",
                    Severity.LOW: "🟢",
                    Severity.INFO: "ℹ️",
                }.get(finding.severity, "•")

                print(f"\n   {i}. {severity_icon} [{finding.severity.value}] {finding.test_name}")
                print(f"      Endpoint: {finding.endpoint}")
                print(f"      Описание: {finding.description}")
                print(f"      CWE: {finding.cwe_id or 'N/A'}")
                print(f"      Рекомендация: {finding.recommendation}")

        print("=" * 70)

    def save_report(self, filename: str = "security_report.json"):
        """Сохранение отчёта в JSON"""
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "base_url": self.base_url,
            "summary": {
                "total_tests": len(self.results),
                "passed_tests": sum(1 for r in self.results if r.passed),
                "total_findings": len(self.findings),
                "critical": sum(1 for f in self.findings if f.severity == Severity.CRITICAL),
                "high": sum(1 for f in self.findings if f.severity == Severity.HIGH),
                "medium": sum(1 for f in self.findings if f.severity == Severity.MEDIUM),
                "low": sum(1 for f in self.findings if f.severity == Severity.LOW),
            },
            "findings": [
                {
                    "test_name": f.test_name,
                    "severity": f.severity.value,
                    "endpoint": f.endpoint,
                    "description": f.description,
                    "evidence": f.evidence,
                    "recommendation": f.recommendation,
                    "cwe_id": f.cwe_id,
                    "cvss_score": f.cvss_score,
                }
                for f in self.findings
            ],
            "results": [
                {
                    "test_name": r.test_name,
                    "passed": r.passed,
                    "duration_seconds": r.duration_seconds,
                    "requests_made": r.requests_made,
                }
                for r in self.results
            ],
        }

        output_path = Path(__file__).parent / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Отчёт сохранён: {output_path}")


def main():
    """Точка входа"""
    parser = argparse.ArgumentParser(description="Security Testing для Nanoprobe API")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Таймаут запроса в секундах (default: 10)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Быстрый тест (только основные проверки)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Полный тест со всеми проверками",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Сохранить отчёт в JSON",
    )

    args = parser.parse_args()

    tester = SecurityTester(
        base_url=args.url,
        timeout=args.timeout,
    )

    _ = tester.run_full_test()  # results, можно использовать в отчёте

    # Сохранение отчёта
    if args.report:
        tester.save_report()

    # Возвращаем код выхода
    critical_or_high = sum(
        1 for f in tester.findings if f.severity in [Severity.CRITICAL, Severity.HIGH]
    )

    return 1 if critical_or_high > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
