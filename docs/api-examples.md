# API Examples — curl команды для всех эндпоинтов

## Оглавление

- [Health & Monitoring](#health--monitoring)
- [Authentication](#authentication)
- [Scans](#scans)
- [Simulations](#simulations)
- [SSTV/RTL-SDR](#sstvrtl-sdr)
- [External Services](#external-services)
- [Monitoring](#monitoring)
- [Admin](#admin)

---

## Health & Monitoring

### Базовый health check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-04-10T12:00:00+00:00",
  "version": "1.0.0"
}
```

### Расширенный health check

```bash
curl http://localhost:8000/health/detailed
```

### Extended health check (все компоненты)

```bash
curl http://localhost:8000/monitoring/health/extended
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-04-10T12:00:00+00:00",
  "components": {
    "database": {"status": "healthy", "path": "data/nanoprobe.db"},
    "redis": {"status": "healthy", "url": "redis://localhost:6379"},
    "websocket": {"status": "healthy", "active_connections": 5},
    "tle_updates": {"status": "fresh", "age_seconds": 300},
    "rtl_sdr": {"status": "healthy", "message": "RTL-SDR device detected"},
    "background_tasks": {"status": "healthy", "tasks": {...}}
  }
}
```

### Prometheus метрики

```bash
curl http://localhost:8000/monitoring/metrics
```

---

## Authentication

### Регистрация

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "securepassword123"
  }'
```

### Логин

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "securepassword123"
  }'
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer"
}
```

### Refresh токена

```bash
curl -X POST http://localhost:8000/api/v1/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "eyJhbGciOiJIUzI1NiIs..."
  }'
```

### 2FA Setup

```bash
curl -X POST http://localhost:8000/api/v1/auth/2fa/setup \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### 2FA Verify

```bash
curl -X POST http://localhost:8000/api/v1/auth/2fa/verify \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "123456"
  }'
```

---

## Scans

### Получить все сканы

```bash
curl http://localhost:8000/api/v1/scans \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### Создать скан

```bash
curl -X POST http://localhost:8000/api/v1/scans \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "scan_type": "afm",
    "resolution": "512x512",
    "parameters": {
      "scan_rate": 1.0,
      "gain": 30
    }
  }'
```

### Получить скан по ID

```bash
curl http://localhost:8000/api/v1/scans/{scan_id} \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### Удалить скан

```bash
curl -X DELETE http://localhost:8000/api/v1/scans/{scan_id} \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

---

## Simulations

### Запустить симуляцию

```bash
curl -X POST http://localhost:8000/api/v1/simulations/run \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "simulation_type": "spm",
    "surface": "silicon",
    "parameters": {
      "probe_radius": 10e-9,
      "scan_area": "1x1"
    }
  }'
```

### Получить статус симуляции

```bash
curl http://localhost:8000/api/v1/simulations/{sim_id}/status \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

---

## SSTV/RTL-SDR

### Health check SSTV

```bash
curl http://localhost:8000/api/v1/sstv/health
```

### Extended health check SSTV

```bash
curl http://localhost:8000/api/v1/sstv/health/extended
```

**Response:**
```json
{
  "status": "healthy",
  "device": {
    "status": "connected",
    "info": {
      "name": "RTL-SDR Blog V4",
      "serial": "00000001",
      "sample_rate": 2400000,
      "center_freq": 145800000
    }
  },
  "capabilities": ["realtime_recording", "sstv_decoding", "satellite_tracking"],
  "degradation_level": "full",
  "recommendations": []
}
```

### Проверка RTL-SDR устройства

```bash
curl http://localhost:8000/api/v1/sstv/device/check
```

**Response:**
```json
{
  "status": "ok",
  "count": 1,
  "devices": [
    {
      "index": 0,
      "name": "RTL-SDR Blog V4",
      "serial": "00000001",
      "is_v4": true,
      "recommended_sample_rate": 2400000
    }
  ]
}
```

### Расписание спутников

```bash
curl "http://localhost:8000/api/v1/sstv/schedule?lat=55.75&lon=37.61&hours=24"
```

### Начать запись SSTV

```bash
curl -X POST http://localhost:8000/api/v1/sstv/record/start \
  -H "Content-Type: application/json" \
  -d '{
    "frequency": 145.800,
    "sample_rate": 2400000,
    "gain": 30,
    "duration": 600,
    "ppm": 0
  }'
```

### Остановить запись

```bash
curl -X POST http://localhost:8000/api/v1/sstv/record/stop
```

### Получить список записей

```bash
curl http://localhost:8000/api/v1/sstv/recordings
```

### Получить TLE данные

```bash
curl http://localhost:8000/api/v1/sstv/tle
```

### Обновить TLE данные

```bash
curl -X POST http://localhost:8000/api/v1/sstv/tle/update
```

---

## External Services

### NASA APOD

```bash
curl http://localhost:8000/api/v1/external/nasa/apod
```

### NASA Image Library

```bash
curl "http://localhost:8000/api/v1/external/nasa/search?q=iss"
```

### Weather Data

```bash
curl "http://localhost:8000/api/v1/external/weather?lat=55.75&lon=37.61"
```

### Health check внешних сервисов

```bash
curl http://localhost:8000/api/v1/external/health
```

---

## Monitoring

### Статистика производительности

```bash
curl http://localhost:8000/monitoring/stats
```

### Профиль БД

```bash
curl http://localhost:8000/monitoring/db/profile
```

### Индексы БД

```bash
curl http://localhost:8000/monitoring/db/indexes
```

---

## Admin

### Пользователи

```bash
curl http://localhost:8000/api/v1/admin/users \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

### Статистика системы

```bash
curl http://localhost:8000/api/v1/admin/system/stats \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

### Очистка кэша

```bash
curl -X POST http://localhost:8000/api/v1/admin/cache/clear \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

### Бэкап БД

```bash
curl -X POST http://localhost:8000/api/v1/admin/backup \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

---

## WebSocket

### Подключение к WebSocket

```bash
# Используя wscat
wscat -c ws://localhost:8000/ws/realtime

# Получение событий
{
  "type": "stats_update",
  "data": {
    "cpu": 45.2,
    "memory": 62.1,
    "disk": 38.5
  }
}
```

---

## Советы

### Аутентификация

Сохраните токен в переменную:

```bash
# Bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"password"}' \
  | jq -r '.access_token')

# Использование
curl http://localhost:8000/api/v1/scans \
  -H "Authorization: Bearer $TOKEN"
```

### Debug mode

Добавьте `-v` для подробного вывода:

```bash
curl -v http://localhost:8000/health
```

### Pretty print JSON

```bash
curl http://localhost:8000/health | jq .
```

### Export OpenAPI spec

```bash
curl http://localhost:8000/openapi.json > openapi.json
```
