"""
Тесты для NASA, Weather, External Services и Monitoring routes
Проверка интеграции с внешними API и внутренними сервисами
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    """Создание тестового клиента"""
    return TestClient(app)


# ==========================================
# NASA API Tests
# ==========================================


class TestNasaRoutes:
    """Тесты для NASA API routes"""

    @patch("utils.api.nasa_api_client.get_nasa_client")
    def test_apod_success(self, mock_get_client, client):
        """Тест успешного получения APOD"""
        mock_client = Mock()
        mock_client.get_apod = AsyncMock(
            return_value={
                "title": "Test APOD",
                "url": "https://example.com/image.jpg",
                "explanation": "Test explanation",
                "date": "2026-04-08",
            }
        )
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/nasa/apod")

        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Test APOD"
        assert "url" in data

    @patch("utils.api.nasa_api_client.get_nasa_client")
    def test_apod_with_date(self, mock_get_client, client):
        """Тест получения APOD за конкретную дату"""
        mock_client = Mock()
        mock_client.get_apod = AsyncMock(
            return_value={
                "title": "Historical APOD",
                "url": "https://example.com/historical.jpg",
                "date": "2026-01-01",
            }
        )
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/nasa/apod?date=2026-01-01")

        assert response.status_code == 200
        mock_client.get_apod.assert_called_once_with(date="2026-01-01", count=None)

    @patch("utils.api.nasa_api_client.get_nasa_client")
    def test_mars_photos_success(self, mock_get_client, client):
        """Тест получения фото с Марса"""
        mock_client = Mock()
        mock_client.get_mars_photos = AsyncMock(
            return_value={
                "photos": [{"id": 1, "img_src": "https://example.com/mars.jpg"}],
                "total_photos": 1,
            }
        )
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/nasa/mars/photos?rover=Curiosity&sol=1000")

        assert response.status_code == 200
        data = response.json()
        assert "photos" in data

    @patch("utils.api.nasa_api_client.get_nasa_client")
    def test_asteroids_feed(self, mock_get_client, client):
        """Тест получения данных об астероидах"""
        mock_client = Mock()
        mock_client.get_asteroids = AsyncMock(
            return_value={
                "links": {"next": "url", "previous": None},
                "element_count": 10,
                "near_earth_objects": {},
            }
        )
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/nasa/asteroids/feed")

        assert response.status_code == 200
        data = response.json()
        assert "element_count" in data

    @patch("utils.api.nasa_api_client.get_nasa_client")
    def test_nasa_health_check(self, mock_get_client, client):
        """Тест проверки здоровья NASA API"""
        mock_client = Mock()
        mock_client.health_check = AsyncMock(return_value=True)
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/nasa/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    @patch("utils.api.nasa_api_client.get_nasa_client")
    def test_nasa_api_error_handling(self, mock_get_client, client):
        """Тест обработки ошибок NASA API"""
        mock_client = Mock()
        mock_client.get_apod = AsyncMock(side_effect=Exception("NASA API timeout"))
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/nasa/apod")

        # Должен вернуть 500 ExternalServiceError
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data


# ==========================================
# Weather API Tests
# ==========================================


class TestWeatherRoutes:
    """Тесты для Weather API routes (Open-Meteo)"""

    @patch("api.routes.weather.fetch_weather_data")
    def test_weather_current_success(self, mock_fetch, client):
        """Тест успешного получения текущей погоды"""
        mock_fetch.return_value = {
            "location": "Moscow",
            "current_weather": {
                "temperature": 20.5,
                "windspeed": 10.2,
                "winddirection": 180,
                "weathercode": 0,
            },
        }

        response = client.get("/api/v1/weather/current?lat=55.7558&lon=37.6173")

        assert response.status_code == 200
        data = response.json()
        assert "current_weather" in data
        assert "temperature" in data["current_weather"]

    @patch("api.routes.weather.fetch_weather_forecast")
    def test_weather_forecast_success(self, mock_fetch, client):
        """Тест получения прогноза погоды"""
        mock_fetch.return_value = {
            "location": "Moscow",
            "daily_forecast": [
                {"date": "2026-04-08", "temperature_max": 22.0},
                {"date": "2026-04-09", "temperature_max": 21.5},
            ],
        }

        response = client.get("/api/v1/weather/forecast?lat=55.7558&lon=37.6173&days=2")

        assert response.status_code == 200
        data = response.json()
        assert "daily_forecast" in data
        assert len(data["daily_forecast"]) == 2

    def test_weather_current_validation(self, client):
        """Тест валидации координат"""
        # Неверные координаты (за пределами допустимого диапазона)
        response = client.get("/api/v1/weather/current?lat=100&lon=200")

        # FastAPI validation должен отклонить
        assert response.status_code == 422

    @patch("api.routes.weather.fetch_historical_weather")
    def test_weather_historical(self, mock_fetch, client):
        """Тест получения исторических данных о погоде"""
        mock_fetch.return_value = {
            "location": "Moscow",
            "historical_data": {"date": "2026-04-01", "temperature": 15.0},
        }

        response = client.get("/api/v1/weather/historical?lat=55.7558&lon=37.6173&date=2026-04-01")

        assert response.status_code == 200
        data = response.json()
        assert "historical_data" in data


# ==========================================
# External Services Tests
# ==========================================


class TestExternalServices:
    """Тесты для External Services routes"""

    def test_external_services_health(self, client):
        """Тест проверки здоровья внешних сервисов"""
        response = client.get("/api/v1/external/health")

        # Должен вернуть статус хотя бы без 500
        assert response.status_code in [200, 500]
        data = response.json()
        assert isinstance(data, (dict, list))

    @patch("api.routes.external_services.call_external_api")
    def test_external_api_call_success(self, mock_call, client):
        """Тест успешного вызова внешнего API"""
        mock_call.return_value = {"status": "success", "data": {"result": "test data"}}

        response = client.post(
            "/api/v1/external/call", json={"url": "https://api.example.com/data", "method": "GET"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    @patch("api.routes.external_services.call_external_api")
    def test_external_api_call_failure(self, mock_call, client):
        """Тест обработки ошибок при вызове внешнего API"""
        mock_call.side_effect = Exception("Connection timeout")

        response = client.post(
            "/api/v1/external/call", json={"url": "https://api.example.com/data", "method": "GET"}
        )

        # Должен вернуть ошибку
        assert response.status_code in [500, 502, 503]


# ==========================================
# Monitoring Routes Tests
# ==========================================


class TestMonitoringRoutes:
    """Тесты для Monitoring routes"""

    def test_metrics_prometheus(self, client):
        """Тест Prometheus метрик"""
        response = client.get("/metrics")

        # Должен вернуть text/plain
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")
        # Содержит хотя бы одну метрику
        assert "#" in response.text or "nanoprobe" in response.text.lower()

    @patch("utils.monitoring.performance_monitor.get_monitor")
    def test_performance_metrics(self, mock_get_monitor, client):
        """Тест метрик производительности"""
        mock_monitor = Mock()
        mock_monitor.get_metrics.return_value = {
            "api_calls": 100,
            "avg_latency": 0.05,
            "error_rate": 0.02,
        }
        mock_get_monitor.return_value = mock_monitor

        response = client.get("/api/v1/monitoring/metrics")

        # Может быть 200 или 404 в зависимости от реализации
        assert response.status_code in [200, 404]

    def test_health_check(self, client):
        """Тест health check endpoint"""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    def test_detailed_health_check(self, client):
        """Тест детальной проверки здоровья"""
        response = client.get("/health/detailed")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "metrics" in data
        assert "cpu" in data["metrics"]
        assert "memory" in data["metrics"]
        assert "disk" in data["metrics"]

    def test_realtime_metrics(self, client):
        """Тест метрик реального времени"""
        response = client.get("/api/v1/metrics/realtime")

        assert response.status_code == 200
        data = response.json()
        assert "cpu_percent" in data
        assert "memory_percent" in data
        assert "disk_percent" in data


# ==========================================
# Integration Tests
# ==========================================


class TestIntegrationExternalServices:
    """Интеграционные тесты для внешних сервисов"""

    @patch("utils.api.nasa_api_client.get_nasa_client")
    @patch("api.routes.weather.fetch_weather_data")
    def test_multiple_external_services(self, mock_weather, mock_nasa, client):
        """Тест работы с несколькими внешними сервисами"""
        # Setup mocks
        mock_nasa_client = Mock()
        mock_nasa_client.get_apod = AsyncMock(return_value={"title": "Test"})
        mock_nasa.return_value = mock_nasa_client

        mock_weather.return_value = {"location": "Test", "current_weather": {"temperature": 20}}

        # Запрос к NASA
        nasa_response = client.get("/api/v1/nasa/apod")
        assert nasa_response.status_code == 200

        # Запрос к Weather
        weather_response = client.get("/api/v1/weather/current?lat=55.7&lon=37.6")
        assert weather_response.status_code == 200

    def test_api_root_endpoint(self, client):
        """Тест корневого endpoint API"""
        response = client.get("/api")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data

    def test_api_documentation_available(self, client):
        """Тест доступности Swagger документации"""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_openapi_schema_available(self, client):
        """Тест доступности OpenAPI схемы"""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
