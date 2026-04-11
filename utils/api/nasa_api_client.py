"""
NASA API Client

Полный клиент для NASA API с поддержкой всех основных endpoints:
- APOD (Astronomy Picture of the Day)
- Mars Rover Photos
- Near Earth Objects (Asteroids)
- Earth Imagery
- NASA Image Library
- EONET (Natural Events)

Требуется API ключ: https://api.nasa.gov/
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class NASAAPIClient:
    """
    Асинхронный клиент для NASA API

    Endpoints:
    - /planetary/apod - Astronomy Picture of the Day
    - /mars-photos/api/v1/rovers - Mars Rover Photos
    - /neo/rest/v1/feed - Near Earth Objects
    - /EPIC/api/natural - Earth Imagery
    - /search - NASA Image Library
    - /api/v1/events - EONET Natural Events
    """

    BASE_URL = "https://api.nasa.gov"

    def __init__(
        self,
        api_key: Optional[str] = None,
        demo_key_fallback: bool = True,
        timeout: int = 30,
    ):
        """
        Инициализация клиента.

        Args:
            api_key: NASA API ключ (если None, используется DEMO_KEY)
            demo_key_fallback: Использовать DEMO_KEY при rate limit
            timeout: Timeout для запросов в секундах
        """
        self.api_key = api_key or os.getenv("NASA_API_KEY", "DEMO_KEY")
        self.demo_key_fallback = demo_key_fallback
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Получение или создание HTTP сессии"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={"User-Agent": "Nanoprobe-Sim-Lab/2.0"},
            )
        return self._session

    async def close(self):
        """Закрытие сессии"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        use_api_key: bool = True,
    ) -> Dict[str, Any]:
        """
        Базовый запрос к NASA API.

        Args:
            endpoint: API endpoint (например, '/planetary/apod')
            params: Query параметры
            use_api_key: Использовать ли API ключ

        Returns:
            JSON ответ
        """
        session = await self._get_session()

        # Подготовка параметров
        request_params = params or {}
        if use_api_key:
            request_params["api_key"] = self.api_key

        url = f"{self.BASE_URL}{endpoint}"

        try:
            async with session.get(url, params=request_params) as response:
                if response.status == 429 and self.demo_key_fallback:
                    # Rate limit - пробуем DEMO_KEY
                    logger.warning("Rate limit exceeded, switching to DEMO_KEY")
                    request_params["api_key"] = "DEMO_KEY"
                    async with session.get(url, params=request_params) as retry_response:
                        return await retry_response.json()

                response.raise_for_status()
                return await response.json()

        except aiohttp.ClientError as e:
            logger.error(f"NASA API request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

    # ==========================================
    # APOD - Astronomy Picture of the Day
    # ==========================================

    async def get_apod(
        self,
        date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        count: Optional[int] = None,
        thumbs: bool = False,
    ) -> Dict[str, Any]:
        """
        Получение Astronomy Picture of the Day.

        Args:
            date: Конкретная дата (YYYY-MM-DD)
            start_date: Начальная дата для диапазона
            end_date: Конечная дата для диапазона
            count: Количество случайных изображений
            thumbs: Включать ли thumbnails для видео

        Returns:
            APOD данные
        """
        params = {}

        if date:
            params["date"] = date
        elif start_date:
            params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date
        elif count:
            params["count"] = count

        if thumbs:
            params["thumbs"] = "True"

        return await self._request("/planetary/apod", params)

    # ==========================================
    # Mars Rover Photos
    # ==========================================

    async def get_mars_photos(
        self,
        sol: Optional[int] = None,
        earth_date: Optional[str] = None,
        camera: Optional[str] = None,
        rover: Optional[str] = None,
        page: int = 0,
        per_page: int = 25,
    ) -> Dict[str, Any]:
        """
        Получение фотографий с марсоходов NASA.

        Args:
            sol: Сол (марсианский день) миссии
            earth_date: Земная дата съёмки
            camera: Название камеры (FHAZ, RHAZ, MAHI, CHEMCAM, etc.)
            rover: Название ровера (Curiosity, Opportunity, Spirit, Perseverance)
            page: Номер страницы
            per_page: Количество на странице

        Returns:
            Список фотографий с метаданными
        """
        params = {
            "page": page,
            "per_page": per_page,
        }

        if sol:
            params["sol"] = sol
        elif earth_date:
            params["earth_date"] = earth_date

        if camera:
            params["camera"] = camera

        if rover:
            params["rover"] = rover

        return await self._request("/mars-photos/api/v1/rovers", params)

    async def get_mars_rover_manifest(self) -> Dict[str, Any]:
        """
        Получение информации о марсоходах.

        Returns:
            Информация о роверах (статус, сол запуска, дата посадки, etc.)
        """
        return await self._request("/mars-photos/api/v1/rovers")

    # ==========================================
    # Near Earth Objects (Asteroids)
    # ==========================================

    async def get_asteroids(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        page: int = 0,
        per_page: int = 25,
        feed_type: str = "browse",  # 'browse' или 'query'
    ) -> Dict[str, Any]:
        """
        Получение данных о околоземных объектах (астероидах).

        Args:
            start_date: Начальная дата (YYYY-MM-DD), по умолчанию сегодня
            end_date: Конечная дата, по умолчанию +7 дней
            page: Номер страницы
            per_page: Количество на странице
            feed_type: Тип данных ('browse' или 'query')

        Returns:
            Список NEO с орбитальными данными
        """
        if not start_date:
            start_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if not end_date:
            end_date = (datetime.now(timezone.utc) + timedelta(days=7)).strftime("%Y-%m-%d")

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "page": page,
            "per_page": per_page,
        }

        endpoint = "/neo/rest/v1/feed" if feed_type == "browse" else "/neo/rest/v1/query"
        return await self._request(endpoint, params)

    async def get_asteroid_by_id(self, asteroid_id: int) -> Dict[str, Any]:
        """
        Получение данных об астероиде по ID.

        Args:
            asteroid_id: NASA Asteroid ID

        Returns:
            Детальные данные об астероиде
        """
        return await self._request(f"/neo/rest/v1/asteroid/{asteroid_id}")

    # ==========================================
    # Earth Imagery (EPIC)
    # ==========================================

    async def get_earth_imagery(
        self,
        date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        coordinates: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Получение изображений Земли со спутника DSCOVR EPIC.

        Args:
            date: Конкретная дата
            start_date: Начальная дата диапазона
            end_date: Конечная дата диапазона
            coordinates: Координаты для поиска (lat, lon)

        Returns:
            Список изображений с метаданными
        """
        params = {}

        if coordinates:
            # Поиск по координатам (широта, долгота)
            params["lat"] = coordinates.get("lat", 0)
            params["lon"] = coordinates.get("lon", 0)
            if date:
                params["date"] = date
            endpoint = "/EPIC/api/natural/date"
        elif date:
            params["date"] = date
            endpoint = "/EPIC/api/natural/date"
        elif start_date and end_date:
            endpoint = f"/EPIC/api/natural/{start_date}/{end_date}"
        else:
            endpoint = "/EPIC/api/natural"

        return await self._request(endpoint, params)

    # ==========================================
    # NASA Image Library
    # ==========================================

    async def search_images(
        self,
        query: str,
        media_type: Optional[str] = None,  # 'image', 'video', 'audio'
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        page: int = 1,
        page_size: int = 25,
    ) -> Dict[str, Any]:
        """
        Поиск в библиотеке изображений NASA.

        Args:
            query: Поисковый запрос
            media_type: Тип медиа
            year_start: Начальный год
            year_end: Конечный год
            page: Номер страницы
            page_size: Размер страницы

        Returns:
            Результаты поиска
        """
        params = {
            "q": query,
            "page": page,
            "page_size": page_size,
        }

        if media_type:
            params["media_type"] = media_type
        if year_start:
            params["year_start"] = year_start
        if year_end:
            params["year_end"] = year_end

        return await self._request("/search", params, use_api_key=False)

    async def get_image_metadata(self, nasa_id: str) -> Dict[str, Any]:
        """
        Получение метаданных изображения по ID.

        Args:
            nasa_id: NASA ID изображения

        Returns:
            Метаданные изображения
        """
        return await self._request(f"/metadata/{nasa_id}", use_api_key=False)

    # ==========================================
    # EONET - Natural Events
    # ==========================================

    async def get_natural_events(
        self,
        status: Optional[str] = None,  # 'open', 'closed'
        days: Optional[int] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Получение данных о природных событиях (пожары, ураганы, etc.).

        Args:
            status: Статус событий ('open', 'closed')
            days: Количество дней для поиска
            start: Начальная дата
            end: Конечная дата
            limit: Лимит результатов

        Returns:
            Список природных событий
        """
        params = {"limit": limit}

        if status:
            params["status"] = status
        if days:
            params["days"] = days
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        return await self._request("/api/v1/events", params, use_api_key=False)

    async def get_event_by_id(self, event_id: str) -> Dict[str, Any]:
        """
        Получение данных о событии по ID.

        Args:
            event_id: EONET event ID

        Returns:
            Детальные данные о событии
        """
        return await self._request(f"/api/v1/events/{event_id}", use_api_key=False)

    # ==========================================
    # Utility методы
    # ==========================================

    async def get_api_info(self) -> Dict[str, Any]:
        """
        Получение информации об API.

        Returns:
            Информация об API (лимиты, статус, etc.)
        """
        return await self._request("", use_api_key=False)

    async def health_check(self) -> bool:
        """
        Проверка доступности NASA API.

        Returns:
            True если API доступен
        """
        try:
            await self.get_api_info()
            return True
        except Exception:
            return False


# Singleton для использования в приложении
_nasa_client: Optional[NASAAPIClient] = None


def get_nasa_client(api_key: Optional[str] = None) -> NASAAPIClient:
    """
    Получение экземпляра NASA API клиента.

    Args:
        api_key: Опциональный API ключ

    Returns:
        NASAAPIClient экземпляр
    """
    global _nasa_client
    if _nasa_client is None or _nasa_client.api_key != (api_key or os.getenv("NASA_API_KEY")):
        _nasa_client = NASAAPIClient(api_key=api_key)
    return _nasa_client


async def close_nasa_client():
    """Закрытие NASA API клиента"""
    global _nasa_client
    if _nasa_client:
        await _nasa_client.close()
        _nasa_client = None
