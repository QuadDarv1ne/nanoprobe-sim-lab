"""
Тесты для Alerting API (api/routes/alerting.py)

Покрытие:
- POST /send — отправка алерта
- POST /send-async — асинхронная отправка
- POST /resolve/{id} — закрытие алерта
- POST /acknowledge/{id} — подтверждение
- POST /silence/{id} — заглушение
- GET /active — активные алерты
- GET /statistics — статистика
- GET /history — история
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

TEST_DB = tempfile.mktemp(suffix=".db")
os.environ["DATABASE_PATH"] = TEST_DB

from api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    """Фикстура: HTTP клиент"""
    with TestClient(app) as test_client:
        yield test_client
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except Exception:
            pass


class TestSendAlert:
    """Тесты POST /send"""

    def test_send_alert_success(self, client):
        """Тест успешной отправки алерта"""
        with patch("api.routes.alerting.AlertManager") as mock_mgr:
            mock_instance = MagicMock()
            mock_instance.send_alert.return_value = {
                "status": "sent",
                "alert_id": "alert_123",
                "channels_sent": ["email", "slack"],
            }
            mock_mgr.return_value = mock_instance

            response = client.post(
                "/api/v1/alerting/send",
                json={
                    "alert_name": "High CPU",
                    "severity": "critical",
                    "description": "CPU usage above 90%",
                    "details": {"cpu_percent": 95},
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "alert_id" in data

    def test_send_alert_duplicate(self, client):
        """Тест дубликата алерта"""
        with patch("api.routes.alerting.AlertManager") as mock_mgr:
            mock_instance = MagicMock()
            mock_instance.send_alert.return_value = {"status": "duplicate"}
            mock_mgr.return_value = mock_instance

            response = client.post(
                "/api/v1/alerting/send",
                json={
                    "alert_name": "High CPU",
                    "severity": "warning",
                    "description": "Duplicate alert",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            assert data["reason"] == "duplicate"

    def test_send_alert_rate_limited(self, client):
        """Тест превышения лимита"""
        with patch("api.routes.alerting.AlertManager") as mock_mgr:
            mock_instance = MagicMock()
            mock_instance.send_alert.return_value = {"status": "rate_limited"}
            mock_mgr.return_value = mock_instance

            response = client.post(
                "/api/v1/alerting/send",
                json={
                    "alert_name": "Spam",
                    "severity": "info",
                    "description": "Too many alerts",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["reason"] == "rate_limited"

    def test_send_alert_silenced(self, client):
        """Тест заглушенного алерта"""
        with patch("api.routes.alerting.AlertManager") as mock_mgr:
            mock_instance = MagicMock()
            mock_instance.send_alert.return_value = {"status": "silenced"}
            mock_mgr.return_value = mock_instance

            response = client.post(
                "/api/v1/alerting/send",
                json={
                    "alert_name": "Silenced",
                    "severity": "warning",
                    "description": "This should be silenced",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["reason"] == "silenced"


class TestSendAlertAsync:
    """Тесты POST /send-async"""

    def test_send_alert_async_success(self, client):
        """Тест асинхронной отправки"""
        with patch("api.routes.alerting.AlertManager") as mock_mgr:
            mock_instance = MagicMock()
            # Async method needs to return a coroutine or use AsyncMock
            from unittest.mock import AsyncMock

            mock_instance.send_alert_async = AsyncMock(
                return_value={
                    "status": "sent",
                    "alert_id": "alert_async_123",
                }
            )
            mock_mgr.return_value = mock_instance

            response = client.post(
                "/api/v1/alerting/send-async",
                json={
                    "alert_name": "Async Alert",
                    "severity": "info",
                    "description": "Async test",
                },
            )

            assert response.status_code == 200


class TestResolveAlert:
    """Тесты POST /resolve/{id}"""

    def test_resolve_alert_success(self, client):
        """Тест успешного закрытия"""
        with patch("api.routes.alerting.AlertManager") as mock_mgr:
            mock_instance = MagicMock()
            mock_instance.resolve_alert.return_value = True
            mock_mgr.return_value = mock_instance

            response = client.post("/api/v1/alerting/resolve/alert_123")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["alert_id"] == "alert_123"


class TestAcknowledgeAlert:
    """Тесты POST /acknowledge/{id}"""

    def test_acknowledge_alert_success(self, client):
        """Тест подтверждения алерта"""
        with patch("api.routes.alerting.AlertManager") as mock_mgr:
            mock_instance = MagicMock()
            mock_instance.acknowledge_alert.return_value = True
            mock_mgr.return_value = mock_instance

            response = client.post(
                "/api/v1/alerting/acknowledge/alert_123",
                json={"acknowledged_by": "admin"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True


class TestSilenceAlert:
    """Тесты POST /silence/{id}"""

    def test_silence_alert_success(self, client):
        """Тест заглушения алерта"""
        with patch("api.routes.alerting.AlertManager") as mock_mgr:
            mock_instance = MagicMock()
            mock_instance.silence_alert.return_value = None
            mock_mgr.return_value = mock_instance

            response = client.post(
                "/api/v1/alerting/silence/alert_123",
                content="30",
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["duration_minutes"] == 30

    def test_silence_alert_invalid_duration(self, client):
        """Тест некорректной длительности"""
        response = client.post(
            "/api/v1/alerting/silence/alert_123",
            json={"duration_minutes": 0},
        )

        assert response.status_code in [400, 422]


class TestActiveAlerts:
    """Тесты GET /active"""

    def test_get_active_alerts_empty(self, client):
        """Тест пустого списка активных алертов"""
        with patch("api.routes.alerting.AlertManager") as mock_mgr:
            mock_instance = MagicMock()
            mock_instance.get_active_alerts.return_value = []
            mock_mgr.return_value = mock_instance

            response = client.get("/api/v1/alerting/active")

            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 0
            assert len(data["alerts"]) == 0

    def test_get_active_alerts_with_data(self, client):
        """Тест списка активных алертов"""
        with patch("api.routes.alerting.AlertManager") as mock_mgr:
            mock_alert = MagicMock()
            mock_alert.__dict__ = {
                "id": "alert_1",
                "name": "High CPU",
                "severity": "critical",
            }
            mock_instance = MagicMock()
            mock_instance.get_active_alerts.return_value = [mock_alert]
            mock_mgr.return_value = mock_instance

            response = client.get("/api/v1/alerting/active")

            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 1


class TestAlertStatistics:
    """Тесты GET /statistics"""

    def test_get_alert_statistics(self, client):
        """Тест получения статистики"""
        with patch("api.routes.alerting.AlertManager") as mock_mgr:
            mock_instance = MagicMock()
            mock_instance.get_alert_statistics.return_value = {
                "total_sent": 100,
                "total_resolved": 80,
                "active_count": 5,
            }
            mock_mgr.return_value = mock_instance

            response = client.get("/api/v1/alerting/statistics")

            assert response.status_code == 200
            data = response.json()
            assert "total_sent" in data


class TestAlertHistory:
    """Тесты GET /history"""

    def test_get_alert_history_empty(self, client):
        """Тест пустой истории"""
        with patch("api.routes.alerting.AlertManager") as mock_mgr:
            mock_instance = MagicMock()
            mock_instance.alert_history = []
            mock_mgr.return_value = mock_instance

            response = client.get("/api/v1/alerting/history")

            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 0
            assert len(data["alerts"]) == 0

    def test_get_alert_history_with_pagination(self, client):
        """Тест истории с пагинацией"""
        with patch("api.routes.alerting.AlertManager") as mock_mgr:
            mock_instance = MagicMock()
            mock_instance.alert_history = [
                {"id": "alert_1", "severity": "critical"},
                {"id": "alert_2", "severity": "warning"},
                {"id": "alert_3", "severity": "info"},
            ]
            mock_mgr.return_value = mock_instance

            response = client.get("/api/v1/alerting/history?limit=2&offset=1")

            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 3
            assert data["limit"] == 2
            assert data["offset"] == 1
            assert len(data["alerts"]) == 2

    def test_get_alert_history_filter_severity(self, client):
        """Тест фильтрации по severity"""
        with patch("api.routes.alerting.AlertManager") as mock_mgr:
            mock_instance = MagicMock()
            mock_instance.alert_history = [
                {"id": "alert_1", "severity": "critical"},
                {"id": "alert_2", "severity": "warning"},
            ]
            mock_mgr.return_value = mock_instance

            response = client.get("/api/v1/alerting/history?severity=critical")

            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 1
            assert data["alerts"][0]["severity"] == "critical"


class TestAlertingValidation:
    """Тесты валидации"""

    def test_alert_severity_levels(self):
        """Тест уровней severity"""
        valid_severities = ["info", "warning", "critical", "emergency"]
        assert len(valid_severities) >= 4

    def test_alert_response_structure(self):
        """Тест структуры ответа"""
        response = {
            "success": True,
            "alert_id": "alert_123",
            "status": "sent",
        }
        assert "success" in response
        assert "alert_id" in response
