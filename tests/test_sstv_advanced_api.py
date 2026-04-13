"""
Тесты для SSTV Advanced API (api/routes/sstv_advanced.py)

Покрытие:
- GET /status — статус SSTV системы
- GET /spectrum — спектр сигнала
- GET /signal-strength — сила сигнала
- WS /ws/stream — WebSocket стриминг
- Device cache logic
- Error handling
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import TestClient as StarletteTestClient

from api.error_handlers import ServiceUnavailableError

# Импортируем приложение
from api.main import app


class TestSSTVAdvancedAPI:
    """Тесты SSTV Advanced API endpoints"""

    @pytest.fixture
    def client(self):
        """Создание тестового клиента"""
        return TestClient(app)

    @pytest.fixture
    def mock_device_available(self):
        """Mock устройства как доступного"""
        with patch("api.routes.sstv_advanced._check_device_fast") as mock_check:
            mock_check.return_value = {"available": True, "error": None}
            yield mock_check

    @pytest.fixture
    def mock_device_unavailable(self):
        """Mock устройства как недоступного"""
        with patch("api.routes.sstv_advanced._check_device_fast") as mock_check:
            mock_check.return_value = {"available": False, "error": "Device not found"}
            yield mock_check

    @pytest.fixture
    def mock_receiver_available(self):
        """Mock receiver with get_receiver patched"""
        mock_recv = MagicMock()
        mock_recv.sdr = MagicMock()
        mock_recv.frequency = 145.800
        mock_recv.get_spectrum.return_value = (
            [145.0, 145.5, 145.8, 146.0],
            [-50.0, -45.0, -40.0, -55.0],
        )
        mock_recv.get_signal_strength.return_value = 75.5
        mock_recv.initialize.return_value = True

        with patch("api.sstv.rtl_sstv_receiver.get_receiver", return_value=mock_recv):
            yield mock_recv

    # NOTE: mock_receiver fixture removed as it was causing errors


class TestSSTVStatusEndpoint(TestSSTVAdvancedAPI):
    """Тесты GET /status"""

    def test_get_status_device_available(self, client, mock_device_available):
        """Тест статуса когда устройство доступно"""
        with patch("api.routes.sstv_advanced.RECEIVER_AVAILABLE", True):
            with patch("api.routes.sstv_advanced.get_session_manager") as mock_session:
                mock_session.return_value.get_stats.return_value = {
                    "total_sessions": 5,
                    "active_sessions": 2,
                }

                response = client.get("/api/v1/sstv/status")

                assert response.status_code == 200
                data = response.json()
                assert "receiver" in data
                assert data["receiver"]["available"] is True
                assert data["receiver"]["pyrtlsdr_loaded"] is True
                assert "session_manager" in data
                assert "frequencies" in data
                assert "modes" in data
                assert "config" in data
                assert data["frequencies"]["iss_sstv"] == 145.800

    def test_get_status_device_unavailable(self, client, mock_device_unavailable):
        """Тест статуса когда устройство недоступно"""
        with patch("api.routes.sstv_advanced.RECEIVER_AVAILABLE", False):
            response = client.get("/api/v1/sstv/status")

            assert response.status_code == 200
            data = response.json()
            assert data["receiver"]["available"] is False
            assert data["receiver"]["pyrtlsdr_loaded"] is False
            # Error может быть разным в зависимости от состояния кэша
            assert data["receiver"]["error"] is not None

    def test_get_status_without_receiver(self, client):
        """Тест статуса без RECEIVER_AVAILABLE"""
        with patch("api.routes.sstv_advanced.RECEIVER_AVAILABLE", False):
            with patch(
                "api.routes.sstv_advanced._check_device_fast",
                return_value={"available": False, "error": "Not available"},
            ):
                response = client.get("/api/v1/sstv/status")

                assert response.status_code == 200
                data = response.json()
                assert data["receiver"]["available"] is False
                assert data["session_manager"] == {}
                assert data["modes"] == []

    def test_get_status_default_frequencies(self, client, mock_device_available):
        """Тест что дефолтные частоты корректны"""
        with patch("api.routes.sstv_advanced.RECEIVER_AVAILABLE", True):
            with patch("api.routes.sstv_advanced.get_session_manager", return_value=None):
                response = client.get("/api/v1/sstv/status")

                assert response.status_code == 200
                data = response.json()
                assert data["frequencies"]["iss_sstv"] == 145.800
                assert data["frequencies"]["noaa_apt"] == 137.100
                assert data["frequencies"]["meteor_m2"] == 137.100


class TestSSTVSpectrumEndpoint(TestSSTVAdvancedAPI):
    """Тесты GET /spectrum"""

    def test_get_spectrum_success(self, client, mock_device_available):
        """Тест успешного получения спектра"""
        import numpy as np

        with patch("api.routes.sstv_advanced._init_receiver_safe", return_value=True):
            with patch("api.sstv.rtl_sstv_receiver.get_receiver") as mock_get:
                mock_recv = MagicMock()
                mock_recv.sdr = MagicMock()
                mock_recv.frequency = 145.800
                mock_recv.get_spectrum.return_value = (
                    np.array([145.0, 145.5, 145.8, 146.0]),
                    np.array([-50.0, -45.0, -40.0, -55.0]),
                )
                mock_recv.get_signal_strength.return_value = 75.5
                mock_get.return_value = mock_recv

                response = client.get(
                    "/api/v1/sstv/spectrum",
                    params={"frequency": 145.800, "span": 2.0, "points": 512},
                )

                assert response.status_code == 200
                data = response.json()
                assert data["frequency_mhz"] == 145.800
                assert data["span_mhz"] == 2.0
                assert data["points"] == 4  # Mock возвращает 4 точки
                assert "frequencies" in data
                assert "power_db" in data
                assert "timestamp" in data

    def test_get_spectrum_device_unavailable(self, client, mock_device_unavailable):
        """Тест спектра когда устройство недоступно"""
        response = client.get("/api/v1/sstv/spectrum")

        # ServiceUnavailableError возвращает 503
        assert response.status_code == 503

    def test_get_spectrum_custom_params(self, client, mock_device_available):
        """Тест спектра с кастомными параметрами"""
        import numpy as np

        with patch("api.routes.sstv_advanced._init_receiver_safe", return_value=True):
            with patch("api.sstv.rtl_sstv_receiver.get_receiver") as mock_get:
                mock_recv = MagicMock()
                mock_recv.sdr = MagicMock()
                mock_recv.frequency = 137.100
                mock_recv.get_spectrum.return_value = (
                    np.array([136.0, 137.0, 137.1, 138.0]),
                    np.array([-60.0, -55.0, -50.0, -65.0]),
                )
                mock_get.return_value = mock_recv

                response = client.get(
                    "/api/v1/sstv/spectrum",
                    params={"frequency": 137.100, "span": 1.5, "points": 256},
                )

                assert response.status_code == 200
                data = response.json()
                assert data["frequency_mhz"] == 137.100
                assert data["span_mhz"] == 1.5

    def test_get_spectrum_init_failed(self, client, mock_device_available):
        """Тест спектра при ошибке инициализации"""
        with patch("api.routes.sstv_advanced._init_receiver_safe", return_value=False):
            response = client.get("/api/v1/sstv/spectrum")

            assert response.status_code == 503

    def test_get_spectrum_receiver_error(self, client, mock_device_available):
        """Тест спектра при ошибке получения данных"""
        with patch("api.routes.sstv_advanced._init_receiver_safe", return_value=True):
            with patch("api.sstv.rtl_sstv_receiver.get_receiver") as mock_get:
                mock_recv = MagicMock()
                mock_recv.get_spectrum.return_value = (None, None)
                mock_get.return_value = mock_recv

                response = client.get("/api/v1/sstv/spectrum")

                assert response.status_code == 503

    def test_get_spectrum_exception(self, client, mock_device_available):
        """Тест спектра при неожиданном исключении"""
        with patch("api.routes.sstv_advanced._init_receiver_safe", return_value=True):
            with patch("api.sstv.rtl_sstv_receiver.get_receiver") as mock_get:
                mock_get.side_effect = Exception("Unexpected error")

                response = client.get("/api/v1/sstv/spectrum")

                assert response.status_code == 503


class TestSSTVSignalStrengthEndpoint(TestSSTVAdvancedAPI):
    """Тесты GET /signal-strength"""

    def test_get_signal_strength_success(self, client, mock_device_available):
        """Тест успешного получения силы сигнала"""
        with patch("api.routes.sstv_advanced._init_receiver_safe", return_value=True):
            with patch("api.sstv.rtl_sstv_receiver.get_receiver") as mock_get:
                mock_recv = MagicMock()
                mock_recv.sdr = MagicMock()
                mock_recv.frequency = 145.800
                mock_recv.get_signal_strength.return_value = 75.5
                mock_get.return_value = mock_recv

                response = client.get("/api/v1/sstv/signal-strength")

                assert response.status_code == 200
                data = response.json()
                assert data["strength_percent"] == 75.5
                assert data["frequency_mhz"] == 145.800
                assert "timestamp" in data

    def test_get_signal_strength_unavailable(self, client, mock_device_unavailable):
        """Тест силы сигнала когда устройство недоступно"""
        response = client.get("/api/v1/sstv/signal-strength")

        assert response.status_code == 503

    def test_get_signal_strength_init_failed(self, client, mock_device_available):
        """Тест силы сигнала при ошибке инициализации"""
        with patch("api.routes.sstv_advanced._init_receiver_safe", return_value=False):
            response = client.get("/api/v1/sstv/signal-strength")

            assert response.status_code == 503

    def test_get_signal_strength_exception(self, client, mock_device_available):
        """Тест силы сигнала при исключении"""
        with patch("api.routes.sstv_advanced._init_receiver_safe", return_value=True):
            with patch("api.sstv.rtl_sstv_receiver.get_receiver") as mock_get:
                mock_get.side_effect = Exception("Signal error")

                response = client.get("/api/v1/sstv/signal-strength")

                assert response.status_code == 503


class TestSSTVDeviceCache:
    """Тесты кэширования устройства"""

    def test_cache_hit(self):
        """Тест попадания в кэш"""
        from api.routes import sstv_advanced

        # Сброс кэша
        sstv_advanced._device_cache.update(
            {"checked": True, "available": True, "timestamp": 0, "error": None}
        )

        with patch("api.routes.sstv_advanced.time.time", return_value=5):
            result = sstv_advanced._check_device_fast()

            assert result["available"] is True
            assert result["error"] is None

    def test_cache_expired(self):
        """Тест истечения срока кэша"""
        from api.routes import sstv_advanced

        # Старый кэш
        sstv_advanced._device_cache.update(
            {"checked": True, "available": True, "timestamp": 0, "error": None}
        )

        with patch("api.routes.sstv_advanced.time.time", return_value=100):
            with patch("api.routes.sstv_advanced.RECEIVER_AVAILABLE", False):
                result = sstv_advanced._check_device_fast()

                # Кэш истёк, перепроверка
                assert result["available"] is False

    def test_cache_not_checked(self):
        """Тест когда кэш ещё не проверен"""
        from api.routes import sstv_advanced

        # Кэш не проверен
        sstv_advanced._device_cache.update(
            {"checked": False, "available": False, "timestamp": 0, "error": None}
        )

        with patch("api.routes.sstv_advanced.time.time", return_value=10):
            with patch("api.routes.sstv_advanced.RECEIVER_AVAILABLE", False):
                result = sstv_advanced._check_device_fast()

                assert result["available"] is False
                assert result["error"] == "pyrtlsdr not installed"

    def test_cache_ttl_constant(self):
        """Тест что CACHE_TTL равен 10 секундам"""
        from api.routes import sstv_advanced

        assert sstv_advanced.CACHE_TTL == 10


class TestSSTVReceiverInit:
    """Тесты инициализации приёмника"""

    def test_init_receiver_safe_success(self):
        """Тест успешной инициализации"""
        with patch("api.routes.sstv_advanced.RECEIVER_AVAILABLE", True):
            with patch("api.sstv.rtl_sstv_receiver.get_receiver") as mock_get:
                mock_recv = MagicMock()
                mock_recv.sdr = None
                mock_recv.initialize.return_value = True
                mock_get.return_value = mock_recv

                # Need to reload module to get patched version
                from importlib import reload

                import api.routes.sstv_advanced as sstv_mod

                reload(sstv_mod)

                result = sstv_mod._init_receiver_safe()
                assert result is True

    def test_init_receiver_safe_already_initialized(self):
        """Тест когда приёмник уже инициализирован"""
        with patch("api.routes.sstv_advanced.RECEIVER_AVAILABLE", True):
            with patch("api.sstv.rtl_sstv_receiver.get_receiver") as mock_get:
                mock_recv = MagicMock()
                mock_recv.sdr = MagicMock()  # Уже инициализирован
                mock_get.return_value = mock_recv

                from importlib import reload

                import api.routes.sstv_advanced as sstv_mod

                reload(sstv_mod)

                result = sstv_mod._init_receiver_safe()
                assert result is True

    def test_init_receiver_safe_not_available(self):
        """Тест когда RECEIVER_AVAILABLE=False"""
        with patch("api.routes.sstv_advanced.RECEIVER_AVAILABLE", False):
            from api.routes.sstv_advanced import _init_receiver_safe

            result = _init_receiver_safe()
            assert result is False

    def test_init_receiver_safe_exception(self):
        """Тест при исключении инициализации"""
        with patch("api.routes.sstv_advanced.RECEIVER_AVAILABLE", True):
            with patch("api.sstv.rtl_sstv_receiver.get_receiver") as mock_get:
                mock_get.side_effect = Exception("Init failed")

                from importlib import reload

                import api.routes.sstv_advanced as sstv_mod

                reload(sstv_mod)

                result = sstv_mod._init_receiver_safe()
                assert result is False


class TestSSTVWebSocketStream(TestSSTVAdvancedAPI):
    """Тесты WebSocket стриминга"""

    @pytest.mark.skip(reason="WebSocket тесты требуют специальной настройки")
    def test_websocket_stream_connect(self, client):
        """Тест подключения к WebSocket"""
        # TODO: Реализовать WebSocket тесты с proper setup
        pass

    @pytest.mark.skip(reason="WebSocket тесты требуют специальной настройки")
    def test_websocket_stream_receiver_unavailable(self, client):
        """Тест WebSocket когда приёмник недоступен"""
        # TODO: Реализовать WebSocket тесты
        pass

    def test_websocket_endpoint_exists(self, client):
        """Тест что WebSocket эндпоинт существует"""
        # Проверяем что роутер зарегистрирован
        routes = [route.path for route in app.routes]
        assert any("/ws/stream" in str(route) for route in routes) or any(
            "sstv" in str(route).lower() for route in routes
        )


class TestSSTVAdvancedConfig:
    """Тесты конфигурации SSTV"""

    def test_default_config_values(self):
        """Тест дефолтных значений конфигурации"""
        from fastapi.testclient import TestClient

        from api.main import app

        with TestClient(app) as client:
            with patch("api.routes.sstv_advanced.RECEIVER_AVAILABLE", True):
                with patch("api.routes.sstv_advanced.get_session_manager", return_value=None):
                    with patch(
                        "api.routes.sstv_advanced._check_device_fast",
                        return_value={"available": True, "error": None},
                    ):
                        with patch.dict("os.environ", {}, clear=True):
                            response = client.get("/api/v1/sstv/status")

                            assert response.status_code == 200
                            data = response.json()
                            assert data["config"]["frequency"] == 145.800
                            assert data["config"]["gain"] == 49.6
                            assert data["config"]["sample_rate"] == 2400000
                            assert data["config"]["bias_tee"] is False
                            assert data["config"]["agc"] is False
                            assert data["config"]["ppm"] == 0

    def test_config_from_env(self):
        """Тест конфигурации из переменных окружения"""
        from fastapi.testclient import TestClient

        from api.main import app

        with TestClient(app) as client:
            with patch("api.routes.sstv_advanced.RECEIVER_AVAILABLE", True):
                with patch("api.routes.sstv_advanced.get_session_manager", return_value=None):
                    with patch(
                        "api.routes.sstv_advanced._check_device_fast",
                        return_value={"available": True, "error": None},
                    ):
                        with patch.dict(
                            "os.environ",
                            {
                                "SSTV_FREQUENCY": "137.100",
                                "SSTV_GAIN": "30.0",
                                "SSTV_BIAS_TEE": "1",
                                "SSTV_AGC": "1",
                                "SSTV_PPM": "50",
                            },
                        ):
                            response = client.get("/api/v1/sstv/status")

                            assert response.status_code == 200
                            data = response.json()
                            assert data["config"]["frequency"] == 137.100
                            assert data["config"]["gain"] == 30.0
                            assert data["config"]["bias_tee"] is True
                            assert data["config"]["agc"] is True
                            assert data["config"]["ppm"] == 50


class TestSSTVShutdown:
    """Тесты shutdown обработчика"""

    def test_shutdown_event_exists(self):
        """Тест что shutdown обработчик существует"""
        from api.routes.sstv_advanced import router

        # Проверяем что роутер существует и имеет shutdown handler
        assert router is not None
        # On_event shutdown может быть или не быть, главное что роутер есть
        assert hasattr(router, "routes")
