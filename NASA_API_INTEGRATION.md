# NASA API Integration Guide

## Полная интеграция с NASA API

**Статус:** ✅ Реализовано (2026-03-15)

---

## 📦 Возможности

### Backend (FastAPI)

**6 основных endpoints:**

1. **APOD** - Astronomy Picture of the Day
2. **Mars Photos** - Фото с марсоходов
3. **Asteroids** - Околоземные объекты
4. **Earth Imagery** - Снимки Земли EPIC
5. **Image Library** - 100,000+ изображений NASA
6. **Natural Events** - Природные катаклизмы

### Frontend (React)

**8 React hooks:**
- `useAPOD()`
- `useMarsPhotos()`
- `useAsteroids()`
- `useEarthImagery()`
- `useNASAImageLibrary()`
- `useNaturalEvents()`
- `useMarsRovers()`
- `useNASAHealth()`

---

## 🔑 API Ключ

**Получить ключ:** https://api.nasa.gov/

**Установка:**
```bash
# .env
NASA_API_KEY=your_api_key_here
```

**Без ключа:** Используется DEMO_KEY (лимит 30 запросов/час)

---

## 🚀 Backend Usage

### Python Client

```python
from utils.nasa_api_client import get_nasa_client

client = get_nasa_client()

# APOD
apod = await client.get_apod()
print(apod['title'], apod['url'])

# APOD за диапазон
apod_range = await client.get_apod(
    start_date="2026-03-01",
    end_date="2026-03-15"
)

# Mars Photos
mars = await client.get_mars_photos(
    rover="Curiosity",
    sol=1000,
    camera="FHAZ"
)

# Asteroids
asteroids = await client.get_asteroids(
    start_date="2026-03-15",
    end_date="2026-03-22"
)

# Earth Imagery
earth = await client.get_earth_imagery(
    date="2026-03-15"
)

# Image Library
images = await client.search_images(
    query="nebula",
    media_type="image"
)

# Natural Events
events = await client.get_natural_events(
    status="open",
    days=7
)
```

### API Endpoints

```bash
# APOD
GET /api/v1/nasa/apod
GET /api/v1/nasa/apod/date-range?start_date=2026-03-01&end_date=2026-03-15

# Mars
GET /api/v1/nasa/mars/photos?rover=Curiosity&sol=1000
GET /api/v1/nasa/mars/rovers

# Asteroids
GET /api/v1/nasa/asteroids/feed?start_date=2026-03-15&end_date=2026-03-22
GET /api/v1/nasa/asteroids/12345

# Earth
GET /api/v1/nasa/earth/imagery?date=2026-03-15
GET /api/v1/nasa/earth/imagery?lat=51.5074&lon=-0.1278

# Image Library
GET /api/v1/nasa/image-library/search?query=nebula&page=1

# Natural Events
GET /api/v1/nasa/events/natural?status=open&days=7
GET /api/v1/nasa/events/{event_id}

# Health
GET /api/v1/nasa/health
```

---

## ⚛️ Frontend Usage

### APOD Component

```tsx
'use client';

import { useAPOD } from '@/hooks/useNASA';

export function APODWidget() {
  const { apod, loading, error, cached } = useAPOD();

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;
  if (!apod) return null;

  return (
    <div className="apod-widget">
      <h2>{apod.title}</h2>
      {cached && <span className="badge">Cached</span>}
      
      {apod.media_type === 'image' ? (
        <img src={apod.url} alt={apod.title} />
      ) : (
        <iframe src={apod.url} title="APOD Video" />
      )}
      
      <p>{apod.explanation}</p>
      {apod.copyright && <small>© {apod.copyright}</small>}
    </div>
  );
}
```

### Mars Photos Component

```tsx
import { useMarsPhotos } from '@/hooks/useNASA';

export function MarsGallery() {
  const { photos, loading, refetch } = useMarsPhotos({
    rover: 'Curiosity',
    per_page: 25,
  });

  return (
    <div>
      <button onClick={refetch}>Refresh</button>
      
      <div className="grid grid-cols-3 gap-4">
        {photos.map(photo => (
          <div key={photo.id}>
            <img src={photo.img_src} alt="Mars" />
            <p>{photo.rover.name} - {photo.camera.full_name}</p>
            <p>{photo.earth_date}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
```

### Asteroids Dashboard

```tsx
import { useAsteroids } from '@/hooks/useNASA';

export function AsteroidTracker() {
  const { asteroids, loading } = useAsteroids({
    start_date: new Date().toISOString().split('T')[0],
    end_date: new Date(Date.now() + 7*24*60*60*1000).toISOString().split('T')[0],
  });

  const allAsteroids = Object.values(asteroids).flat();
  const hazardous = allAsteroids.filter(a => a.is_potentially_hazardous_asteroid);

  return (
    <div>
      <h3>Near Earth Objects</h3>
      <p>Total: {allAsteroids.length}</p>
      <p className="text-red-500">Hazardous: {hazardous.length}</p>
      
      <ul>
        {allAsteroids.slice(0, 10).map(asteroid => (
          <li key={asteroid.id}>
            {asteroid.name} - 
            {asteroid.diameter.estimated_diameter_max.toFixed(0)}m
          </li>
        ))}
      </ul>
    </div>
  );
}
```

### Natural Events Map

```tsx
import { useNaturalEvents } from '@/hooks/useNASA';

export function NaturalEventsMap() {
  const { events } = useNaturalEvents({ status: 'open', days: 30 });

  return (
    <div>
      <h3>Active Natural Events</h3>
      {events.map(event => (
        <div key={event.id}>
          <h4>{event.title}</h4>
          <p>{event.category}</p>
          <p>{event.description}</p>
          <p>
            Coordinates: {event.geometry[0]?.coordinates.join(', ')}
          </p>
        </div>
      ))}
    </div>
  );
}
```

---

## 💾 Кэширование

**Redis кэш:**
- APOD: 1 час
- APOD Range: 2 часа
- Mars Photos: 1 час
- Mars Rovers: 24 часа
- Asteroids: 1 час
- Earth Imagery: 1 час
- Image Library: 1 час
- Natural Events: 30 минут
- Health: 5 минут

**Ключи:**
```
nasa:apod:2026-03-15
nasa:apod:range:2026-03-01:2026-03-15
nasa:mars:Curiosity:1000:FHAZ:0:25
nasa:asteroids:2026-03-15:2026-03-22:0:25
nasa:earth:2026-03-15:51.5074:-0.1278
nasa:events:open:7:50
```

---

## 🛡️ Rate Limiting

**NASA API лимиты:**
- DEMO_KEY: 30 запросов/час
- Personal Key: 1000 запросов/час

**Наши лимиты (FastAPI):**
- APOD: 30 запросов/мин
- Mars: 20 запросов/мин
- Asteroids: 20 запросов/мин
- Earth: 20 запросов/мин
- Image Library: 30 запросов/мин
- Events: 30 запросов/мин

**Fallback:** При rate limit автоматически используется DEMO_KEY

---

## 📊 Примеры использования

### NASA Dashboard Widget

```tsx
import { useAPOD, useMarsPhotos, useAsteroids } from '@/hooks';

export function NASADashboard() {
  const { apod } = useAPOD();
  const { photos } = useMarsPhotos({ per_page: 5 });
  const { asteroids } = useAsteroids();

  return (
    <div className="grid grid-cols-3 gap-4">
      {/* APOD Card */}
      <div className="card">
        <h3>Astronomy Picture of the Day</h3>
        <img src={apod?.url} alt={apod?.title} />
        <p>{apod?.title}</p>
      </div>

      {/* Mars Photos */}
      <div className="card">
        <h3>Latest from Mars</h3>
        {photos.slice(0, 3).map(photo => (
          <img key={photo.id} src={photo.img_src} />
        ))}
      </div>

      {/* Asteroids */}
      <div className="card">
        <h3>Near Earth Objects</h3>
        <p>{Object.values(asteroids).flat().length} this week</p>
      </div>
    </div>
  );
}
```

---

## 🧪 Тестирование

### Backend Tests

```bash
pytest tests/test_nasa_api.py -v
```

### Frontend Testing

```tsx
import { renderHook, waitFor } from '@testing-library/react';
import { useAPOD } from '@/hooks';

test('loads APOD data', async () => {
  const { result } = renderHook(() => useAPOD());
  
  expect(result.current.loading).toBe(true);
  
  await waitFor(() => expect(result.current.loading).toBe(false));
  
  expect(result.current.apod).toBeDefined();
  expect(result.current.error).toBeNull();
});
```

---

## 📈 Monitoring

**Health Check:**
```bash
GET /api/v1/nasa/health
```

**Response:**
```json
{
  "status": "healthy",
  "api": "NASA API",
  "timestamp": "2026-03-15T10:30:00Z"
}
```

---

## 🔗 Ссылки

- [NASA API Documentation](https://api.nasa.gov/)
- [APOD API](https://api.nasa.gov/planetary/apod/)
- [Mars Photos API](https://api.nasa.gov/mars-photos/)
- [NeoWs API](https://api.nasa.gov/neo/)
- [EPIC API](https://api.nasa.gov/EPIC/)
- [EONET API](https://eonet.gsfc.nasa.gov/api/v2.1)

---

*Обновлено: 2026-03-15*
