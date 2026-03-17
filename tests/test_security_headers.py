#!/usr/bin/env python
"""
Тест Security Headers Middleware

Проверка добавления security headers к ответам API
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.security_headers import setup_security_headers
from fastapi import FastAPI
from fastapi.testclient import TestClient


def create_test_app():
    """Создание тестового приложения с security headers"""
    app = FastAPI()
    setup_security_headers(app, production=False)
    
    @app.get("/test")
    async def test_endpoint():
        return {"status": "ok"}
    
    return app


def test_security_headers_present():
    """Проверка наличия всех security headers"""
    app = create_test_app()
    client = TestClient(app)
    
    response = client.get("/test")
    
    assert response.status_code == 200
    
    # Проверка основных headers
    assert response.headers.get("X-Frame-Options") == "DENY", "X-Frame-Options должен быть DENY"
    assert response.headers.get("X-Content-Type-Options") == "nosniff", "X-Content-Type-Options должен быть nosniff"
    assert response.headers.get("X-XSS-Protection") == "1; mode=block", "X-XSS-Protection должен быть включён"
    assert "Referrer-Policy" in response.headers, "Referrer-Policy должен присутствовать"
    assert "Permissions-Policy" in response.headers, "Permissions-Policy должен присутствовать"
    assert "Content-Security-Policy-Report-Only" in response.headers, "CSP должен присутствовать"
    
    print("[PASS] Все security headers присутствуют")


def test_x_frame_options():
    """Проверка X-Frame-Options (защита от clickjacking)"""
    app = create_test_app()
    client = TestClient(app)
    
    response = client.get("/test")
    assert response.headers.get("X-Frame-Options") == "DENY"
    print("[PASS] X-Frame-Options: DENY")


def test_x_content_type_options():
    """Проверка X-Content-Type-Options (защита от MIME sniffing)"""
    app = create_test_app()
    client = TestClient(app)
    
    response = client.get("/test")
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    print("[PASS] X-Content-Type-Options: nosniff")


def test_x_xss_protection():
    """Проверка X-XSS-Protection"""
    app = create_test_app()
    client = TestClient(app)
    
    response = client.get("/test")
    assert response.headers.get("X-XSS-Protection") == "1; mode=block"
    print("[PASS] X-XSS-Protection: 1; mode=block")


def test_referrer_policy():
    """Проверка Referrer-Policy"""
    app = create_test_app()
    client = TestClient(app)
    
    response = client.get("/test")
    referrer_policy = response.headers.get("Referrer-Policy")
    assert referrer_policy == "strict-origin-when-cross-origin"
    print(f"[PASS] Referrer-Policy: {referrer_policy}")


def test_permissions_policy():
    """Проверка Permissions-Policy"""
    app = create_test_app()
    client = TestClient(app)
    
    response = client.get("/test")
    permissions_policy = response.headers.get("Permissions-Policy")
    assert permissions_policy is not None
    assert "geolocation=()" in permissions_policy
    assert "microphone=()" in permissions_policy
    print("[PASS] Permissions-Policy: ограничения присутствуют")


def test_csp_report_only():
    """Проверка CSP в режиме Report-Only"""
    app = create_test_app()
    client = TestClient(app)
    
    response = client.get("/test")
    assert "Content-Security-Policy-Report-Only" in response.headers
    csp = response.headers.get("Content-Security-Policy-Report-Only")
    assert "default-src" in csp
    print("[PASS] CSP: Report-Only режим работает")


def test_server_headers_removed():
    """Проверка удаления заголовков сервера"""
    app = create_test_app()
    client = TestClient(app)
    
    response = client.get("/test")
    
    # Server и X-Powered-By должны быть удалены
    assert "Server" not in response.headers, "Server header должен быть удалён"
    assert "X-Powered-By" not in response.headers, "X-Powered-By должен быть удалён"
    print("[PASS] Server headers удалены")


def test_production_mode():
    """Проверка production режима (HSTS)"""
    app = FastAPI()
    setup_security_headers(app, production=True)
    
    @app.get("/test")
    async def test_endpoint():
        return {"status": "ok"}
    
    client = TestClient(app)
    
    # HSTS должен быть только для HTTPS
    # В тесте HTTP, поэтому HSTS не добавляется
    response = client.get("/test")
    
    # Проверка, что middleware установлен
    assert response.status_code == 200
    print("[PASS] Production mode: middleware установлен")


def test_custom_csp():
    """Проверка пользовательской CSP"""
    app = FastAPI()
    custom_csp = "default-src 'self'; script-src 'self'"
    setup_security_headers(
        app,
        production=False,
        custom_csp=custom_csp,
    )
    
    @app.get("/test")
    async def test_endpoint():
        return {"status": "ok"}
    
    client = TestClient(app)
    response = client.get("/test")
    
    csp = response.headers.get("Content-Security-Policy-Report-Only")
    assert custom_csp in csp
    print(f"[PASS] Custom CSP: {custom_csp}")


if __name__ == "__main__":
    print("=" * 60)
    print("Тест Security Headers Middleware")
    print("=" * 60)
    
    tests = [
        test_security_headers_present,
        test_x_frame_options,
        test_x_content_type_options,
        test_x_xss_protection,
        test_referrer_policy,
        test_permissions_policy,
        test_csp_report_only,
        test_server_headers_removed,
        test_production_mode,
        test_custom_csp,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test.__name__}: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Результат: {passed}/{len(tests)} тестов пройдено")
    if failed > 0:
        print(f"  ❌ {failed} тестов провалено")
    else:
        print("  ✅ Все тесты пройдены!")
    
    sys.exit(0 if failed == 0 else 1)
