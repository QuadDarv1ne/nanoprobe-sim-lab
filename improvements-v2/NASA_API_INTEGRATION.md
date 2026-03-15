# NASA API Integration Guide

## Обзор

NASA предоставляет бесплатный API для доступа к различным данным: изображения космоса, данные о астероидах, марсоходы, Earth imagery и многое другое.

## 1. Получение NASA API Key

### Demo Key (для разработки)
NASA предоставляет demo key для тестирования:
```
DEMO_KEY
```
- Лимит: 30 запросов в час
- Лимит: 50 запросов в день

### Production Key

#### Шаг 1: Регистрация
1. Перейдите на https://api.nasa.gov/
2. Нажмите "Generate API Key"
3. Заполните форму:
   - First Name
   - Last Name
   - Email
   - Application Name: `Nanoprobe Sim Lab`
   - Application Description: `Научно-образовательный проект для моделирования нанозонда с интеграцией данных NASA для SSTV наземной станции`

#### Шаг 2: Получение ключа
После отправки формы вы получите email с API ключом.

#### Шаг 3: Rate Limits для Production Key
| Метрика | Значение |
|---------|----------|
| Запросов в час | 1,000 |
| Запросов в день | Нет лимита |
| IP-based rate limit | Да |

## 2. Структура конфигурации

### .env файл
```env
# NASA API
NASA_API_KEY=your_production_key_here
NASA_API_DEMO_KEY=DEMO_KEY

# Rate Limiting
NASA_API_RATE_LIMIT_PER_HOUR=1000
NASA_API_CACHE_TTL=3600  # 1 час

# VAPID для Push уведомлений
VAPID_PUBLIC_KEY=your_vapid_public_key
VAPID_PRIVATE_KEY=your_vapid_private_key
VAPID_EMAIL=your-email@example.com
```

### config/nasa_config.json
```json
{
  "api_key": "${NASA_API_KEY}",
  "base_urls": {
    "apod": "https://api.nasa.gov/planetary/apod",
    "mars_rover": "https://api.nasa.gov/mars-photos/api/v1/rovers",
    "asteroids": "https://api.nasa.gov/neo/rest/v1",
    "earth": "https://api.nasa.gov/planetary/earth",
    "eonet": "https://eonet.gsfc.nasa.gov/api/v3",
    "images": "https://images-api.nasa.gov"
  },
  "rate_limits": {
    "requests_per_hour": 1000,
    "requests_per_day": null,
    "burst_limit": 50
  },
  "cache": {
    "enabled": true,
    "default_ttl": 3600,
    "image_ttl": 86400
  },
  "fallback_to_demo": true
}
```

## 3. NASA API Client Implementation

### Backend (Python)

```python
# utils/nasa_api_client.py
import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging
from dataclasses import dataclass
from functools import wraps
import time
import os

logger = logging.getLogger(__name__)

@dataclass
class RateLimitInfo:
    remaining: int
    limit: int
    reset_time: datetime

class NASAAPIClient:
    """
    NASA API Client с автоматическим rate limiting и кэшированием.
    
    Features:
    - Rate limiting: 1000 req/hour (production)
    - Automatic fallback to demo key
    - Redis caching
    - Retry with exponential backoff
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        use_cache: bool = True,
        redis_client = None
    ):
        self.api_key = api_key or os.getenv("NASA_API_KEY", "DEMO_KEY")
        self.demo_key = "DEMO_KEY"
        self.use_cache = use_cache
        self.redis = redis_client
        
        # Rate limiting
        self.rate_limit = 1000 if self.api_key != self.demo_key else 30
        self.requests_made = 0
        self.window_start = time.time()
        self.window_duration = 3600  # 1 hour
        
        # Session
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Base URLs
        self.base_urls = {
            "apod": "https://api.nasa.gov/planetary/apod",
            "mars_rover": "https://api.nasa.gov/mars-photos/api/v1/rovers",
            "asteroids": "https://api.nasa.gov/neo/rest/v1",
            "earth": "https://api.nasa.gov/planetary/earth",
            "eonet": "https://eonet.gsfc.nasa.gov/api/v3",
            "images": "https://images-api.nasa.gov"
        }
    
    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
    
    async def _check_rate_limit(self) -> bool:
        """Проверка и обновление rate limit"""
        current_time = time.time()
        
        # Reset window if expired
        if current_time - self.window_start >= self.window_duration:
            self.requests_made = 0
            self.window_start = current_time
        
        # Check limit
        if self.requests_made >= self.rate_limit:
            wait_time = self.window_duration - (current_time - self.window_start)
            logger.warning(f"Rate limit reached. Waiting {wait_time:.0f} seconds")
            return False
        
        self.requests_made += 1
        return True
    
    async def _get_cached(self, cache_key: str) -> Optional[Dict]:
        """Получение данных из кэша"""
        if not self.use_cache or not self.redis:
            return None
        
        cached = await self.redis.get(cache_key)
        if cached:
            logger.debug(f"Cache hit: {cache_key}")
            return cached
        return None
    
    async def _set_cached(self, cache_key: str, data: Dict, ttl: int = 3600):
        """Сохранение данных в кэш"""
        if not self.use_cache or not self.redis:
            return
        
        await self.redis.set(cache_key, data, ex=ttl)
    
    async def _request(
        self,
        url: str,
        params: Dict,
        cache_ttl: int = 3600,
        retry_count: int = 3
    ) -> Dict:
        """Выполнение запроса с rate limiting и retry"""
        
        # Check cache first
        cache_key = f"nasa_api:{url}:{hash(frozenset(params.items()))}"
        cached = await self._get_cached(cache_key)
        if cached:
            return cached
        
        # Rate limiting
        for attempt in range(retry_count):
            if not await self._check_rate_limit():
                await asyncio.sleep(60)  # Wait before retry
                continue
            
            try:
                params["api_key"] = self.api_key
                
                async with self._session.get(url, params=params) as response:
                    if response.status == 429:
                        # Rate limited - fallback to demo key
                        if self.api_key != self.demo_key:
                            logger.warning("Rate limited, falling back to demo key")
                            params["api_key"] = self.demo_key
                            continue
                        else:
                            raise Exception("Rate limit exceeded")
                    
                    if response.status == 200:
                        data = await response.json()
                        await self._set_cached(cache_key, data, cache_ttl)
                        return data
                    
                    if response.status >= 500 and attempt < retry_count - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    
                    raise Exception(f"API error: {response.status}")
                    
            except aiohttp.ClientError as e:
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
        
        raise Exception("Max retries exceeded")
    
    # ===== APOD (Astronomy Picture of the Day) =====
    
    async def get_apod(
        self,
        date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        count: Optional[int] = None,
        thumbs: bool = True
    ) -> Dict:
        """
        Astronomy Picture of the Day
        
        Args:
            date: YYYY-MM-DD (default: today)
            start_date, end_date: Range for multiple APODs
            count: Random number of images
            thumbs: Return thumbnail URLs
        """
        params = {"thumbs": thumbs}
        if date:
            params["date"] = date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if count:
            params["count"] = count
        
        return await self._request(self.base_urls["apod"], params, cache_ttl=86400)
    
    # ===== Mars Rover Photos =====
    
    async def get_mars_photos(
        self,
        rover: str = "curiosity",
        sol: Optional[int] = None,
        earth_date: Optional[str] = None,
        camera: Optional[str] = None,
        page: int = 1
    ) -> Dict:
        """
        Mars Rover Photos
        
        Args:
            rover: curiosity, opportunity, spirit, perseverance
            sol: Martian sol (mission day)
            earth_date: YYYY-MM-DD
            camera: FHAZ, RHAZ, MAST, CHEMCAM, MAHLI, MARDI, NAVCAM, PANCAM, MINITES
        """
        url = f"{self.base_urls['mars_rover']}/{rover}/photos"
        params = {"page": page}
        
        if sol:
            params["sol"] = sol
        elif earth_date:
            params["earth_date"] = earth_date
        else:
            # Default to latest
            params["sol"] = 1000
        
        if camera:
            params["camera"] = camera
        
        return await self._request(url, params, cache_ttl=3600)
    
    # ===== Near Earth Objects (Asteroids) =====
    
    async def get_asteroids(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """
        Near Earth Objects
        
        Args:
            start_date: YYYY-MM-DD (default: today)
            end_date: YYYY-MM-DD (max 7 days from start)
        """
        today = datetime.now().strftime("%Y-%m-%d")
        params = {
            "start_date": start_date or today,
            "end_date": end_date or (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        }
        
        return await self._request(
            f"{self.base_urls['asteroids']}/feed",
            params,
            cache_ttl=3600
        )
    
    async def get_asteroid_by_id(self, asteroid_id: int) -> Dict:
        """Get specific asteroid by NASA ID"""
        return await self._request(
            f"{self.base_urls['asteroids']}/neo/{asteroid_id}",
            {},
            cache_ttl=86400
        )
    
    # ===== Earth Imagery =====
    
    async def get_earth_image(
        self,
        lat: float,
        lon: float,
        date: Optional[str] = None,
        dim: float = 0.15
    ) -> str:
        """
        Earth Imagery (Landsat 8)
        
        Args:
            lat: Latitude (-90 to 90)
            lon: Longitude (-180 to 180)
            date: YYYY-MM-DD
            dim: Width and height of image in degrees
        
        Returns:
            Image URL
        """
        params = {
            "lat": lat,
            "lon": lon,
            "dim": dim
        }
        if date:
            params["date"] = date
        
        # This endpoint returns image, not JSON
        return f"{self.base_urls['earth']}/imagery?lat={lat}&lon={lon}&dim={dim}&api_key={self.api_key}"
    
    # ===== NASA Image and Video Library =====
    
    async def search_nasa_images(
        self,
        query: str,
        media_type: str = "image",
        year_start: Optional[str] = None,
        year_end: Optional[str] = None,
        page: int = 1
    ) -> Dict:
        """
        Search NASA Image and Video Library
        
        Args:
            query: Search terms
            media_type: image, video, audio
            year_start, year_end: Date range
        """
        params = {
            "q": query,
            "media_type": media_type,
            "page": page
        }
        if year_start:
            params["year_start"] = year_start
        if year_end:
            params["year_end"] = year_end
        
        # Note: This API doesn't require api_key
        return await self._request(
            f"{self.base_urls['images']}/search",
            params,
            cache_ttl=86400
        )
    
    # ===== EONET (Earth Observatory Natural Event Tracker) =====
    
    async def get_natural_events(
        self,
        status: str = "open",
        days: Optional[int] = None,
        category: Optional[str] = None
    ) -> Dict:
        """
        Natural Events (wildfires, volcanoes, storms, etc.)
        
        Args:
            status: open, closed, all
            days: Number of days to look back
            category: wildfires, volcanoes, severeStorms, etc.
        """
        params = {"status": status}
        if days:
            params["days"] = days
        if category:
            params["category"] = category
        
        # Note: This API doesn't require api_key
        return await self._request(
            f"{self.base_urls['eonet']}/events",
            params,
            cache_ttl=1800  # 30 min
        )


# ===== Singleton Instance =====
_nasa_client: Optional[NASAAPIClient] = None

async def get_nasa_client() -> NASAAPIClient:
    """Get NASA API client singleton"""
    global _nasa_client
    if _nasa_client is None:
        from utils.redis_client import get_redis
        redis = await get_redis()
        _nasa_client = NASAAPIClient(redis_client=redis)
        await _nasa_client.__aenter__()
    return _nasa_client
```

### API Routes

```python
# api/routes/nasa.py
from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional, List
from utils.nasa_api_client import get_nasa_client, NASAAPIClient

router = APIRouter(prefix="/nasa", tags=["NASA API"])

@router.get("/apod")
async def get_apod(
    date: Optional[str] = Query(None, regex=r"^\d{4}-\d{2}-\d{2}$"),
    count: Optional[int] = Query(None, ge=1, le=100),
    client: NASAAPIClient = Depends(get_nasa_client)
):
    """
    Astronomy Picture of the Day
    
    - **date**: Specific date (YYYY-MM-DD)
    - **count**: Random selection of N images
    """
    try:
        result = await client.get_apod(date=date, count=count)
        return result
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"NASA API error: {str(e)}")

@router.get("/mars/{rover}")
async def get_mars_photos(
    rover: str,
    sol: Optional[int] = None,
    earth_date: Optional[str] = None,
    camera: Optional[str] = None,
    page: int = Query(1, ge=1),
    client: NASAAPIClient = Depends(get_nasa_client)
):
    """
    Mars Rover Photos
    
    - **rover**: curiosity, opportunity, spirit, perseverance
    - **sol**: Martian day
    - **earth_date**: Earth date (YYYY-MM-DD)
    - **camera**: Camera type
    """
    if rover not in ["curiosity", "opportunity", "spirit", "perseverance"]:
        raise HTTPException(status_code=400, detail="Invalid rover name")
    
    try:
        result = await client.get_mars_photos(
            rover=rover,
            sol=sol,
            earth_date=earth_date,
            camera=camera,
            page=page
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"NASA API error: {str(e)}")

@router.get("/asteroids")
async def get_asteroids(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    client: NASAAPIClient = Depends(get_nasa_client)
):
    """Near Earth Objects (Asteroids) for the next 7 days"""
    try:
        result = await client.get_asteroids(start_date, end_date)
        return result
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"NASA API error: {str(e)}")

@router.get("/earth/imagery")
async def get_earth_imagery(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    date: Optional[str] = None,
    client: NASAAPIClient = Depends(get_nasa_client)
):
    """Earth satellite imagery for coordinates"""
    try:
        image_url = await client.get_earth_image(lat, lon, date)
        return {"url": image_url}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"NASA API error: {str(e)}")

@router.get("/search")
async def search_nasa_media(
    q: str = Query(..., min_length=1),
    media_type: str = Query("image"),
    year_start: Optional[str] = None,
    page: int = Query(1, ge=1),
    client: NASAAPIClient = Depends(get_nasa_client)
):
    """Search NASA Image and Video Library"""
    try:
        result = await client.search_nasa_images(
            query=q,
            media_type=media_type,
            year_start=year_start,
            page=page
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"NASA API error: {str(e)}")

@router.get("/events")
async def get_natural_events(
    status: str = Query("open"),
    days: Optional[int] = Query(None, ge=1, le=365),
    category: Optional[str] = None,
    client: NASAAPIClient = Depends(get_nasa_client)
):
    """
    Natural Events (wildfires, volcanoes, storms)
    
    - **status**: open, closed, all
    - **days**: Days to look back
    - **category**: wildfires, volcanoes, severeStorms, etc.
    """
    try:
        result = await client.get_natural_events(status, days, category)
        return result
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"NASA API error: {str(e)}")
```

## 4. Frontend Integration (React/Next.js)

```typescript
// frontend/src/lib/nasaApi.ts
import axios from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface APOD {
  date: string;
  title: string;
  explanation: string;
  url: string;
  hdurl?: string;
  media_type: 'image' | 'video';
  copyright?: string;
  thumbnail_url?: string;
}

export interface MarsPhoto {
  id: number;
  sol: number;
  earth_date: string;
  img_src: string;
  camera: {
    name: string;
    full_name: string;
  };
  rover: {
    name: string;
    status: string;
  };
}

export interface Asteroid {
  id: string;
  name: string;
  estimated_diameter: {
    meters: {
      estimated_diameter_min: number;
      estimated_diameter_max: number;
    };
  };
  close_approach_data: Array<{
    close_approach_date: string;
    relative_velocity: {
      kilometers_per_hour: string;
    };
    miss_distance: {
      kilometers: string;
    };
  }>;
  is_potentially_hazardous_asteroid: boolean;
}

export const nasaApi = {
  // Astronomy Picture of the Day
  async getAPOD(date?: string, count?: number): Promise<APOD | APOD[]> {
    const params = new URLSearchParams();
    if (date) params.append('date', date);
    if (count) params.append('count', count.toString());
    
    const { data } = await axios.get(`${API_BASE}/api/v1/nasa/apod?${params}`);
    return data;
  },

  // Mars Rover Photos
  async getMarsPhotos(
    rover: string,
    sol?: number,
    earthDate?: string,
    page = 1
  ): Promise<{ photos: MarsPhoto[] }> {
    const params = new URLSearchParams();
    params.append('page', page.toString());
    if (sol) params.append('sol', sol.toString());
    if (earthDate) params.append('earth_date', earthDate);
    
    const { data } = await axios.get(`${API_BASE}/api/v1/nasa/mars/${rover}?${params}`);
    return data;
  },

  // Near Earth Asteroids
  async getAsteroids(startDate?: string, endDate?: string): Promise<{
    element_count: number;
    near_earth_objects: Record<string, Asteroid[]>;
  }> {
    const params = new URLSearchParams();
    if (startDate) params.append('start_date', startDate);
    if (endDate) params.append('end_date', endDate);
    
    const { data } = await axios.get(`${API_BASE}/api/v1/nasa/asteroids?${params}`);
    return data;
  },

  // Search NASA Media
  async searchMedia(query: string, mediaType = 'image'): Promise<{
    collection: {
      items: Array<{
        data: Array<{
          title: string;
          description: string;
          date_created: string;
        }>;
        links: Array<{
          href: string;
        }>;
      }>;
    };
  }> {
    const { data } = await axios.get(
      `${API_BASE}/api/v1/nasa/search?q=${encodeURIComponent(query)}&media_type=${mediaType}`
    );
    return data;
  },

  // Natural Events
  async getNaturalEvents(status = 'open'): Promise<{
    events: Array<{
      id: string;
      title: string;
      description: string;
      categories: Array<{ title: string }>;
      geometry: Array<{
        date: string;
        coordinates: [number, number];
      }>;
    }>;
  }> {
    const { data } = await axios.get(`${API_BASE}/api/v1/nasa/events?status=${status}`);
    return data;
  },
};
```

## 5. React Hook для NASA Data

```typescript
// frontend/src/hooks/useNASA.ts
import { useQuery } from '@tanstack/react-query';
import { nasaApi } from '@/lib/nasaApi';

export function useAPOD(date?: string) {
  return useQuery({
    queryKey: ['apod', date],
    queryFn: () => nasaApi.getAPOD(date),
    staleTime: 1000 * 60 * 60 * 24, // 24 hours
  });
}

export function useMarsPhotos(rover: string, sol?: number) {
  return useQuery({
    queryKey: ['mars', rover, sol],
    queryFn: () => nasaApi.getMarsPhotos(rover, sol),
    staleTime: 1000 * 60 * 60, // 1 hour
    enabled: !!rover,
  });
}

export function useAsteroids() {
  return useQuery({
    queryKey: ['asteroids'],
    queryFn: () => nasaApi.getAsteroids(),
    staleTime: 1000 * 60 * 30, // 30 minutes
    refetchInterval: 1000 * 60 * 30, // Auto-refresh every 30 min
  });
}

export function useNaturalEvents(status: 'open' | 'closed' | 'all' = 'open') {
  return useQuery({
    queryKey: ['events', status],
    queryFn: () => nasaApi.getNaturalEvents(status),
    staleTime: 1000 * 60 * 15, // 15 minutes
    refetchInterval: 1000 * 60 * 15,
  });
}
```

## 6. Dashboard Widget

```typescript
// frontend/src/components/dashboard/NASAWidget.tsx
'use client';

import { useAPOD, useAsteroids } from '@/hooks/useNASA';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Loader2, ExternalLink, Asterisk } from 'lucide-react';
import Image from 'next/image';

export function NASAWidget() {
  const { data: apod, isLoading: apodLoading } = useAPOD();
  const { data: asteroids, isLoading: asteroidsLoading } = useAsteroids();

  const hazardousCount = asteroids?.near_earth_objects
    ? Object.values(asteroids.near_earth_objects)
        .flat()
        .filter(a => a.is_potentially_hazardous_asteroid).length
    : 0;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {/* APOD Card */}
      <Card className="overflow-hidden">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2">
            🌌 Astronomy Picture of the Day
          </CardTitle>
        </CardHeader>
        <CardContent>
          {apodLoading ? (
            <div className="flex items-center justify-center h-48">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : apod && !Array.isArray(apod) ? (
            <div className="space-y-3">
              <div className="relative aspect-video rounded-lg overflow-hidden">
                <Image
                  src={apod.media_type === 'image' ? apod.url : apod.thumbnail_url || ''}
                  alt={apod.title}
                  fill
                  className="object-cover"
                />
              </div>
              <div>
                <h3 className="font-semibold">{apod.title}</h3>
                <p className="text-sm text-muted-foreground line-clamp-2">
                  {apod.explanation}
                </p>
              </div>
              {apod.hdurl && (
                <a
                  href={apod.hdurl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-sm text-blue-500 hover:underline"
                >
                  HD Version <ExternalLink className="h-3 w-3" />
                </a>
              )}
            </div>
          ) : null}
        </CardContent>
      </Card>

      {/* Asteroids Card */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2">
            <Asterisk className="h-5 w-5" />
            Near Earth Asteroids
          </CardTitle>
        </CardHeader>
        <CardContent>
          {asteroidsLoading ? (
            <div className="flex items-center justify-center h-48">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : asteroids ? (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-4 bg-blue-500/10 rounded-lg">
                  <div className="text-3xl font-bold text-blue-500">
                    {asteroids.element_count}
                  </div>
                  <div className="text-sm text-muted-foreground">Total (7 days)</div>
                </div>
                <div className="text-center p-4 bg-orange-500/10 rounded-lg">
                  <div className="text-3xl font-bold text-orange-500">
                    {hazardousCount}
                  </div>
                  <div className="text-sm text-muted-foreground">Hazardous</div>
                </div>
              </div>
              
              <div className="space-y-2 max-h-40 overflow-y-auto">
                {Object.entries(asteroids.near_earth_objects)
                  .slice(0, 5)
                  .map(([date, asts]) => (
                    <div key={date} className="text-sm">
                      <span className="font-medium">{date}:</span>{' '}
                      {asts.length} asteroid{asts.length !== 1 ? 's' : ''}
                    </div>
                  ))}
              </div>
            </div>
          ) : null}
        </CardContent>
      </Card>
    </div>
  );
}
```

## 7. Environment Setup

```bash
# Добавить в requirements.txt
aiohttp>=3.9.0
aioredis>=2.0.0
webpush>=1.0.0

# Установка
pip install aiohttp aioredis pywebpush
```

## 8. Testing NASA API

```python
# tests/test_nasa_api.py
import pytest
from utils.nasa_api_client import NASAAPIClient

@pytest.fixture
async def nasa_client():
    async with NASAAPIClient(api_key="DEMO_KEY", use_cache=False) as client:
        yield client

@pytest.mark.asyncio
async def test_get_apod(nasa_client):
    result = await nasa_client.get_apod()
    assert "title" in result
    assert "url" in result
    assert "explanation" in result

@pytest.mark.asyncio
async def test_get_mars_photos(nasa_client):
    result = await nasa_client.get_mars_photos(rover="curiosity", sol=1000)
    assert "photos" in result

@pytest.mark.asyncio
async def test_get_asteroids(nasa_client):
    result = await nasa_client.get_asteroids()
    assert "near_earth_objects" in result
```

## 9. Rate Limit Monitoring

```python
# utils/nasa_rate_monitor.py
import asyncio
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class NASARateMonitor:
    """Мониторинг использования NASA API"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.key_prefix = "nasa_rate:"
    
    async def record_request(self, endpoint: str):
        """Записать запрос"""
        today = datetime.now().strftime("%Y-%m-%d")
        hour = datetime.now().hour
        
        # Daily counter
        await self.redis.incr(f"{self.key_prefix}daily:{today}")
        
        # Hourly counter
        await self.redis.incr(f"{self.key_prefix}hourly:{today}:{hour}")
        
        # Endpoint counter
        await self.redis.incr(f"{self.key_prefix}endpoint:{endpoint}:{today}")
    
    async def get_usage_stats(self) -> dict:
        """Получить статистику использования"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        daily = await self.redis.get(f"{self.key_prefix}daily:{today}") or 0
        
        return {
            "today": int(daily),
            "limit": 1000,
            "remaining": max(0, 1000 - int(daily)),
            "percentage": round(int(daily) / 1000 * 100, 1)
        }
```
