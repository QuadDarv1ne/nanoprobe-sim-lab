#!/usr/bin/env python3
"""
Тесты для NASA API Client (utils/api/nasa_api_client.py)

Покрытие:
- Инициализация клиента
- Управление сессией
- APOD endpoint
- Mars Rover Photos endpoint
- Asteroids (NEO) endpoint
- Earth Imagery endpoint
- Edge cases
"""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# Добавляем корень проекта в path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.api.nasa_api_client import NASAAPIClient


class TestNASAAPIClientInit:
    """Тесты инициализации NASAAPIClient"""

    def test_init_with_api_key(self):
        """Инициализация с явным API ключом"""
        client = NASAAPIClient(api_key="my-test-key")
        assert client.api_key == "my-test-key"
        assert client.demo_key_fallback is True
        assert client.timeout == 30
        assert client._session is None

    def test_init_without_api_key_uses_demo(self):
        """Инициализация без ключа использует DEMO_KEY"""
        with patch.dict(os.environ, {}, clear=True):
            client = NASAAPIClient()
            assert client.api_key == "DEMO_KEY"

    def test_init_with_env_variable(self):
        """Инициализация с переменной окружения NASA_API_KEY"""
        with patch.dict(os.environ, {"NASA_API_KEY": "env-test-key"}):
            client = NASAAPIClient()
            assert client.api_key == "env-test-key"

    def test_init_demo_key_fallback_default(self):
        """demo_key_fallback по умолчанию True"""
        client = NASAAPIClient()
        assert client.demo_key_fallback is True

    def test_init_demo_key_fallback_disabled(self):
        """Отключение demo_key_fallback"""
        client = NASAAPIClient(demo_key_fallback=False)
        assert client.demo_key_fallback is False

    def test_init_timeout_default(self):
        """Таймаут по умолчанию 30 секунд"""
        client = NASAAPIClient()
        assert client.timeout == 30

    def test_init_custom_timeout(self):
        """Кастомный таймаут"""
        client = NASAAPIClient(timeout=60)
        assert client.timeout == 60

    def test_init_base_url(self):
        """Базовый URL клиента"""
        client = NASAAPIClient()
        assert client.BASE_URL == "https://api.nasa.gov"

    def test_init_session_is_none(self):
        """Сессия при инициализации равна None"""
        client = NASAAPIClient()
        assert client._session is None


class TestGetSession:
    """Тесты метода _get_session"""

    @pytest.mark.asyncio
    async def test_get_session_creates_new(self):
        """_get_session создаёт новую сессию"""
        client = NASAAPIClient()
        session = await client._get_session()
        assert session is not None
        assert client._session is session
        await client.close()

    @pytest.mark.asyncio
    async def test_get_session_reuses_existing(self):
        """_get_session переиспользует существующую сессию"""
        client = NASAAPIClient()
        session1 = await client._get_session()
        session2 = await client._get_session()
        assert session1 is session2
        await client.close()

    @pytest.mark.asyncio
    async def test_get_session_recreates_if_closed(self):
        """_get_session пересоздаёт закрытую сессию"""
        client = NASAAPIClient()
        session1 = await client._get_session()
        await client.close()
        assert session1.closed is True
        session2 = await client._get_session()
        assert session2 is not session1
        assert session2.closed is False
        await client.close()


class TestClose:
    """Тесты метода close"""

    @pytest.mark.asyncio
    async def test_close_session(self):
        """Закрытие активной сессии"""
        client = NASAAPIClient()
        await client._get_session()
        await client.close()
        assert client._session.closed is True

    @pytest.mark.asyncio
    async def test_close_no_session(self):
        """Закрытие без сессии (ничего не делает)"""
        client = NASAAPIClient()
        response = await client.close()
        assert response is None
        assert client._session is None

    @pytest.mark.asyncio
    async def test_close_session_already_closed(self):
        """Повторное закрытие уже закрытой сессии"""
        client = NASAAPIClient()
        await client._get_session()
        await client.close()
        await client.close()
        assert client._session.closed is True


class TestAPOD:
    """Тесты APOD endpoint"""

    @pytest.mark.asyncio
    async def test_get_apod_basic(self):
        """Базовый запрос APOD"""
        client = NASAAPIClient(api_key="test-key")
        mock_response = {"date": "2024-01-01", "title": "Test APOD", "url": "https://example.com"}
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            await client.get_apod()
            mock_request.assert_called_once_with("/planetary/apod", {})

    @pytest.mark.asyncio
    async def test_get_apod_with_date(self):
        """APOD с конкретной датой"""
        client = NASAAPIClient(api_key="test-key")
        mock_response = {"date": "2024-03-15", "title": "Test"}
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            await client.get_apod(date="2024-03-15")
            mock_request.assert_called_once_with("/planetary/apod", {"date": "2024-03-15"})

    @pytest.mark.asyncio
    async def test_get_apod_with_count(self):
        """APOD с количеством"""
        client = NASAAPIClient(api_key="test-key")
        mock_response = [{"date": "2024-01-01"}, {"date": "2024-01-02"}]
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            await client.get_apod(count=5)
            mock_request.assert_called_once_with("/planetary/apod", {"count": 5})

    @pytest.mark.asyncio
    async def test_get_apod_with_thumbs(self):
        """APOD с thumbnails"""
        client = NASAAPIClient(api_key="test-key")
        mock_response = {"date": "2024-01-01", "thumbnail_url": "https://example.com/thumb"}
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            await client.get_apod(thumbs=True)
            mock_request.assert_called_once_with("/planetary/apod", {"thumbs": "True"})


class TestMarsRoverPhotos:
    """Тесты Mars Rover Photos endpoint"""

    @pytest.mark.asyncio
    async def test_get_mars_photos_basic(self):
        """Базовый запрос фотографий марсохода"""
        client = NASAAPIClient(api_key="test-key")
        mock_response = {"photos": [], "page": 0, "per_page": 25}
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            await client.get_mars_photos()
            mock_request.assert_called_once_with(
                "/mars-photos/api/v1/rovers", {"page": 0, "per_page": 25}
            )

    @pytest.mark.asyncio
    async def test_get_mars_photos_with_date(self):
        """Фотографии с земной датой"""
        client = NASAAPIClient(api_key="test-key")
        mock_response = {"photos": [], "page": 0, "per_page": 25}
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            await client.get_mars_photos(earth_date="2024-01-01")
            mock_request.assert_called_once_with(
                "/mars-photos/api/v1/rovers",
                {"page": 0, "per_page": 25, "earth_date": "2024-01-01"},
            )

    @pytest.mark.asyncio
    async def test_get_mars_photos_with_sol(self):
        """Фотографии с sol (марсианский день)"""
        client = NASAAPIClient(api_key="test-key")
        mock_response = {"photos": [], "page": 0, "per_page": 25}
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            await client.get_mars_photos(sol=1000)
            mock_request.assert_called_once_with(
                "/mars-photos/api/v1/rovers", {"page": 0, "per_page": 25, "sol": 1000}
            )

    @pytest.mark.asyncio
    async def test_get_mars_photos_with_camera(self):
        """Фотографии с фильтром по камере"""
        client = NASAAPIClient(api_key="test-key")
        mock_response = {"photos": [], "page": 0, "per_page": 25}
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            await client.get_mars_photos(camera="FHAZ")
            mock_request.assert_called_once_with(
                "/mars-photos/api/v1/rovers", {"page": 0, "per_page": 25, "camera": "FHAZ"}
            )

    @pytest.mark.asyncio
    async def test_get_mars_photos_with_pagination(self):
        """Фотографии с пагинацией"""
        client = NASAAPIClient(api_key="test-key")
        mock_response = {"photos": [], "page": 2, "per_page": 50}
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            await client.get_mars_photos(page=2, per_page=50)
            mock_request.assert_called_once_with(
                "/mars-photos/api/v1/rovers", {"page": 2, "per_page": 50}
            )

    @pytest.mark.asyncio
    async def test_get_mars_rover_manifest(self):
        """Получение информации о марсоходах"""
        client = NASAAPIClient(api_key="test-key")
        mock_response = {"rovers": []}
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            await client.get_mars_rover_manifest()
            mock_request.assert_called_once_with("/mars-photos/api/v1/rovers")


class TestAsteroids:
    """Тесты Asteroids (NEO) endpoint"""

    @pytest.mark.asyncio
    async def test_get_asteroids_basic(self):
        """Базовый запрос астероидов"""
        client = NASAAPIClient(api_key="test-key")
        mock_response = {"near_earth_objects": [], "page": {}}
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            await client.get_asteroids(start_date="2024-01-01", end_date="2024-01-08")
            mock_request.assert_called_once_with(
                "/neo/rest/v1/feed",
                {"start_date": "2024-01-01", "end_date": "2024-01-08", "page": 0, "per_page": 25},
            )

    @pytest.mark.asyncio
    async def test_get_asteroids_with_dates(self):
        """Астероиды с кастомными датами"""
        client = NASAAPIClient(api_key="test-key")
        mock_response = {"near_earth_objects": [], "page": {}}
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            await client.get_asteroids(start_date="2024-06-01", end_date="2024-06-07")
            mock_request.assert_called_once_with(
                "/neo/rest/v1/feed",
                {"start_date": "2024-06-01", "end_date": "2024-06-07", "page": 0, "per_page": 25},
            )

    @pytest.mark.asyncio
    async def test_get_asteroids_with_pagination(self):
        """Астероиды с пагинацией"""
        client = NASAAPIClient(api_key="test-key")
        mock_response = {"near_earth_objects": [], "page": {}}
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            await client.get_asteroids(
                start_date="2024-01-01", end_date="2024-01-08", page=1, per_page=50
            )
            mock_request.assert_called_once_with(
                "/neo/rest/v1/feed",
                {
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-08",
                    "page": 1,
                    "per_page": 50,
                },
            )

    @pytest.mark.asyncio
    async def test_get_asteroids_by_id(self):
        """Получение астероида по ID"""
        client = NASAAPIClient(api_key="test-key")
        mock_response = {"id": "1234", "name": "Test Asteroid"}
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            await client.get_asteroid_by_id(1234)
            mock_request.assert_called_once_with("/neo/rest/v1/asteroid/1234")


class TestEarthImagery:
    """Тесты Earth Imagery endpoint"""

    @pytest.mark.asyncio
    async def test_get_earth_imagery_basic(self):
        """Базовый запрос изображений Земли"""
        client = NASAAPIClient(api_key="test-key")
        mock_response = [{"id": "epic_1", "date": "2024-01-01"}]
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            await client.get_earth_imagery()
            mock_request.assert_called_once_with("/EPIC/api/natural", {})

    @pytest.mark.asyncio
    async def test_get_earth_imagery_with_date(self):
        """Изображения Земли с датой"""
        client = NASAAPIClient(api_key="test-key")
        mock_response = [{"id": "epic_1", "date": "2024-03-15"}]
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            await client.get_earth_imagery(date="2024-03-15")
            mock_request.assert_called_once_with("/EPIC/api/natural/date", {"date": "2024-03-15"})


class TestEdgeCases:
    """Тесты edge cases"""

    def test_client_with_none_api_key_uses_env(self):
        """Клиент с None api_key использует переменную окружения"""
        with patch.dict(os.environ, {"NASA_API_KEY": "from-env-key"}, clear=True):
            client = NASAAPIClient(api_key=None)
            assert client.api_key == "from-env-key"

    @pytest.mark.asyncio
    async def test_user_agent_will_be_set_on_session(self):
        """User-Agent устанавливается при создании сессии"""
        client = NASAAPIClient()
        session = await client._get_session()
        assert "User-Agent" in session.headers
        assert session.headers["User-Agent"] == "Nanoprobe-Sim-Lab/2.0"
        await client.close()
