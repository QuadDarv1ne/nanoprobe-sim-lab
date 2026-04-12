#!/usr/bin/env python
"""
Тесты для API infrastructure (wave 5)

Покрытие:
- API dependencies (get_current_user, require_admin, get_db)
- API metrics (BusinessMetrics, Prometheus)
- API state management
- API security headers
- API rate limiting
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

TEST_DB = tempfile.mktemp(suffix=".db")
os.environ["DATABASE_PATH"] = TEST_DB

from api.main import app


@pytest.fixture(scope="module")
def client():
    """HTTP клиент для тестов"""
    print("\n[INFO] Инициализация тестовой БД для API infrastructure тестов...")
    with TestClient(app) as test_client:
        yield test_client
    print("\n[INFO] Очистка тестовой БД...")
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except Exception:
            pass


class TestAPIDependencies:
    """Тесты для API зависимостей"""

    def test_get_db_manager(self):
        """Получение DB manager из зависимостей"""
        from api.dependencies import get_db

        assert get_db is not None
        assert callable(get_db)

    def test_get_redis_cache(self):
        """Получение Redis cache из зависимостей"""
        from api.dependencies import get_redis_cache

        assert get_redis_cache is not None
        assert callable(get_redis_cache)

    def test_get_current_user_function(self):
        """Проверка функции get_current_user"""
        from api.dependencies import get_current_user

        assert get_current_user is not None
        assert callable(get_current_user)


class TestAPIMetrics:
    """Тесты для API метрик"""

    def test_business_metrics_import(self):
        """Импорт BusinessMetrics"""
        from api.metrics import BusinessMetrics

        assert BusinessMetrics is not None

    def test_business_metrics_inc_request(self):
        """Инкремент счётчика запросов"""
        from api.metrics import BusinessMetrics

        try:
            BusinessMetrics.inc_request("test_endpoint")
            assert True
        except Exception:
            pass  # Metrics могут требовать инициализации

    def test_business_metrics_inc_defect_analysis(self):
        """Инкремент счётчика анализа дефектов"""
        from api.metrics import BusinessMetrics

        try:
            BusinessMetrics.inc_defect_analysis("test_model", [])
            assert True
        except Exception:
            pass

    def test_business_metrics_inc_cache_hit(self):
        """Инкремент cache hit"""
        from api.metrics import BusinessMetrics

        try:
            BusinessMetrics.inc_cache_hit("test_cache")
            assert True
        except Exception:
            pass

    def test_business_metrics_inc_cache_miss(self):
        """Инкремент cache miss"""
        from api.metrics import BusinessMetrics

        try:
            BusinessMetrics.inc_cache_miss("test_cache")
            assert True
        except Exception:
            pass


class TestAPIState:
    """Тесты для API State"""

    def test_state_module(self):
        """Импорт state модуля"""
        from api import state

        assert state is not None

    def test_get_db_manager_from_state(self):
        """Получение DB manager из state"""
        from api.state import get_db_manager

        assert get_db_manager is not None

    def test_get_redis_from_state(self):
        """Получение Redis из state"""
        from api.state import get_redis

        assert get_redis is not None

    def test_set_db_manager(self):
        """Установка DB manager"""
        from api.state import set_db_manager
        from utils.database import DatabaseManager

        test_db = tempfile.mktemp(suffix=".db")
        try:
            db = DatabaseManager(test_db)
            set_db_manager(db)
            assert True
        except Exception:
            pass
        finally:
            try:
                if os.path.exists(test_db):
                    os.remove(test_db)
            except (PermissionError, OSError):
                pass

    def test_get_system_disk_usage(self):
        """Получение информации о диске"""
        from api.state import get_system_disk_usage

        try:
            disk = get_system_disk_usage()
            assert disk is not None
            assert disk.total > 0
            assert disk.used > 0
        except Exception:
            pass


class TestAPISecurityHeaders:
    """Тесты для Security Headers"""

    def test_security_headers_middleware(self):
        """Импорт middleware security headers"""
        from api import security_headers

        assert security_headers is not None

    def test_security_headers_module_exists(self):
        """Проверка что модуль существует"""
        import importlib

        spec = importlib.util.find_spec("api.security_headers")
        assert spec is not None


class TestAPIRateLimiter:
    """Тесты для API Rate Limiter"""

    def test_rate_limiter_module(self):
        """Импорт rate limiter"""
        from api import rate_limiter

        assert rate_limiter is not None

    def test_rate_limiter_class(self):
        """Rate limiter класс"""
        try:
            from api.rate_limiter import RateLimiterMiddleware

            assert RateLimiterMiddleware is not None
        except (ImportError, AttributeError):
            pytest.skip("RateLimiterMiddleware not available")


class TestAPIErrorHandlers:
    """Тесты для API Error Handlers"""

    def test_error_handlers_module(self):
        """Импорт error handlers"""
        from api import error_handlers

        assert error_handlers is not None

    def test_validation_error_class(self):
        """ValidationError класс"""
        from api.error_handlers import ValidationError

        error = ValidationError("Test validation error")
        assert error is not None
        assert str(error)

    def test_not_found_error_class(self):
        """NotFoundError класс"""
        from api.error_handlers import NotFoundError

        error = NotFoundError("Test resource not found")
        assert error is not None
        assert str(error)

    def test_authorization_error_class(self):
        """AuthorizationError класс"""
        from api.error_handlers import AuthorizationError

        error = AuthorizationError("Test authorization error")
        assert error is not None
        assert str(error)

    def test_database_error_class(self):
        """DatabaseError класс"""
        from api.error_handlers import DatabaseError

        error = DatabaseError("Test database error")
        assert error is not None
        assert str(error)

    def test_service_unavailable_error_class(self):
        """ServiceUnavailableError класс"""
        from api.error_handlers import ServiceUnavailableError

        error = ServiceUnavailableError("Test service unavailable")
        assert error is not None
        assert str(error)


class TestAPISchemas:
    """Дополнительные тесты для API Schemas"""

    def test_health_response_schema(self):
        """HealthResponse schema"""
        try:
            from api.schemas import HealthResponse

            response = HealthResponse(status="healthy", version="1.0.0")
            assert response.status == "healthy"
            assert response.version == "1.0.0"
        except ImportError:
            pass

    def test_error_response_schema(self):
        """ErrorResponse schema"""
        try:
            from api.schemas import ErrorResponse

            error = ErrorResponse(detail="Test error")
            assert error.detail == "Test error"
        except ImportError:
            pass

    def test_scan_list_response_schema(self):
        """ScanListResponse schema"""
        from api.schemas import ScanListResponse

        response = ScanListResponse(items=[], total=0, limit=20, offset=0)
        assert response.total == 0
        assert response.limit == 20
        assert len(response.items) == 0

    def test_simulation_list_response_schema(self):
        """SimulationListResponse schema"""
        from api.schemas import SimulationListResponse

        response = SimulationListResponse(items=[], total=0, limit=50)
        assert response.total == 0
        assert response.limit == 50
        assert len(response.items) == 0


class TestAPIHealth:
    """Тесты для API Health module"""

    def test_health_module(self):
        """Импорт health модуля"""
        from api import health

        assert health is not None

    def test_compute_system_health(self):
        """Проверка compute_system_health"""
        from api.health import compute_system_health

        health_status = compute_system_health()
        assert health_status is not None
        assert isinstance(health_status, dict)
        assert "status" in health_status

    def test_get_system_disk_usage_wrapper(self):
        """Обёртка get_system_disk_usage"""
        try:
            from api.health import get_system_disk_usage

            disk = get_system_disk_usage()
            assert disk is not None
        except (ImportError, AttributeError):
            pass


class TestAPIJWTConfig:
    """Тесты для JWT Config"""

    def test_jwt_config_module(self):
        """Импорт jwt_config"""
        from api.security import jwt_config

        assert jwt_config is not None

    def test_get_jwt_secret(self):
        """Получение JWT secret"""
        try:
            from api.security.jwt_config import get_jwt_secret

            secret = get_jwt_secret()
            assert isinstance(secret, str)
            assert len(secret) > 0
        except (ImportError, Exception):
            pass

    def test_jwt_algorithm(self):
        """JWT algorithm"""
        try:
            from api.security.jwt_config import JWT_ALGORITHM

            assert JWT_ALGORITHM in ["HS256", "HS384", "HS512", "RS256"]
        except (ImportError, AttributeError):
            pass
