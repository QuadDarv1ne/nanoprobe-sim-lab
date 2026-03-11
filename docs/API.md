# API Документация Nanoprobe Sim Lab

## Быстрый старт

### Запуск API

```bash
# Установка зависимостей
pip install -r requirements-api.txt

# Запуск сервера
python run_api.py --reload

# Или через uvicorn напрямую
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Документация

После запуска API документация доступна по адресам:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

## Аутентификация

### Вход в систему

```bash
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "admin",
  "password": "admin123"
}
```

**Ответ:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "id": 1,
    "username": "admin",
    "role": "admin"
  }
}
```

### Использование токена

Добавьте заголовок `Authorization` к запросам:

```bash
Authorization: Bearer <access_token>
```

---

## Сканирования

### Получить список сканирований

```bash
GET /api/v1/scans?limit=100&offset=0&scan_type=spm
```

**Ответ:**
```json
{
  "items": [
    {
      "id": 1,
      "timestamp": "2026-03-11T10:00:00",
      "scan_type": "spm",
      "surface_type": "graphite",
      "width": 256,
      "height": 256,
      "file_path": "output/scan_001.png",
      "metadata": {},
      "created_at": "2026-03-11T10:00:00"
    }
  ],
  "total": 1,
  "limit": 100,
  "offset": 0
}
```

### Создать сканирование

```bash
POST /api/v1/scans
Content-Type: application/json

{
  "scan_type": "spm",
  "surface_type": "graphite",
  "width": 256,
  "height": 256,
  "metadata": {"note": "Test scan"}
}
```

### Удалить сканирование

```bash
DELETE /api/v1/scans/{scan_id}
```

### Поиск сканирований

```bash
GET /api/v1/scans/search/{query}
```

---

## Симуляции

### Получить список симуляций

```bash
GET /api/v1/simulations?status=running&limit=50
```

### Создать симуляцию

```bash
POST /api/v1/simulations
Content-Type: application/json

{
  "simulation_type": "spm_scan",
  "parameters": {
    "resolution": 256,
    "scan_size": 100,
    "noise_level": 0.01
  }
}
```

### Обновить статус симуляции

```bash
PATCH /api/v1/simulations/{simulation_id}?status=completed
```

---

## Анализ дефектов

### Анализ изображения

```bash
POST /api/v1/analysis/defects
Content-Type: application/json

{
  "image_path": "output/surface_001.png",
  "model_name": "isolation_forest"
}
```

**Ответ:**
```json
{
  "analysis_id": "defect_abc123",
  "image_path": "output/surface_001.png",
  "model_name": "isolation_forest",
  "defects_count": 5,
  "defects": [
    {
      "type": "pit",
      "x": 123.5,
      "y": 45.2,
      "width": 10,
      "height": 8,
      "area": 80,
      "confidence": 0.92
    }
  ],
  "confidence_score": 0.89,
  "processing_time_ms": 234.5,
  "timestamp": "2026-03-11T10:00:00"
}
```

### История анализов

```bash
GET /api/v1/analysis/defects/history?limit=50
```

---

## Сравнение поверхностей

### Сравнить две поверхности

```bash
POST /api/v1/comparison
Content-Type: application/json

{
  "image1_path": "output/surface_001.png",
  "image2_path": "output/surface_002.png"
}
```

**Ответ:**
```json
{
  "comparison_id": "comp_xyz789",
  "image1_path": "output/surface_001.png",
  "image2_path": "output/surface_002.png",
  "similarity_score": 0.95,
  "metrics": {
    "ssim": 0.95,
    "psnr": 35.2,
    "mse": 0.001,
    "similarity": 0.96,
    "pearson": 0.94
  },
  "difference_map_path": "output/diff_xyz789.png",
  "created_at": "2026-03-11T10:00:00"
}
```

### История сравнений

```bash
GET /api/v1/comparison/history?limit=50
```

---

## PDF Отчёты

### Сгенерировать отчёт

```bash
POST /api/v1/reports
Content-Type: application/json

{
  "report_type": "surface_analysis",
  "title": "Анализ поверхности графита",
  "author": "Иванов И.И.",
  "data": {
    "surface_type": "graphite",
    "scan_size": 100,
    "mean_height": 10.5,
    "std_deviation": 2.3,
    "rms": 11.2
  },
  "images": [
    "output/surface_001.png",
    "output/profile_001.png"
  ]
}
```

**Ответ:**
```json
{
  "report_id": "report_abc123",
  "report_path": "reports/pdf/surface_analysis_20260311_100000.pdf",
  "report_type": "surface_analysis",
  "title": "Анализ поверхности графита",
  "file_size_bytes": 123456,
  "pages_count": 5,
  "created_at": "2026-03-11T10:00:00"
}
```

### Скачать отчёт

```bash
GET /api/v1/reports/{report_id}/download
```

---

## WebSocket

### Подключение к WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/realtime');

ws.onopen = () => {
  console.log('Connected');
  
  // Подписка на канал
  ws.send(JSON.stringify({
    type: 'subscribe',
    channel: 'scans'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

### Типы сообщений

**Ping/Pong:**
```json
{"type": "ping"}
{"type": "pong", "timestamp": "2026-03-11T10:00:00"}
```

**Подписка:**
```json
{"type": "subscribe", "channel": "scans"}
{"type": "subscribed", "channel": "scans", "timestamp": "..."}
```

---

## Коды ошибок

| Код | Описание |
|-----|----------|
| 200 | Успех |
| 201 | Создано |
| 204 | Успешно удалено |
| 400 | Ошибка валидации |
| 401 | Неавторизован |
| 404 | Не найдено |
| 500 | Внутренняя ошибка сервера |

---

## Примеры использования

### Python клиент

```python
import requests

# Логин
response = requests.post(
    'http://localhost:8000/api/v1/auth/login',
    json={'username': 'admin', 'password': 'admin123'}
)
token = response.json()['access_token']

# Заголовки с токеном
headers = {'Authorization': f'Bearer {token}'}

# Создание сканирования
scan_data = {
    'scan_type': 'spm',
    'surface_type': 'graphite',
    'width': 256,
    'height': 256
}
response = requests.post(
    'http://localhost:8000/api/v1/scans',
    json=scan_data,
    headers=headers
)
scan = response.json()
print(f"Создано сканирование: {scan['id']}")

# Анализ дефектов
analysis_data = {
    'image_path': 'output/surface.png',
    'model_name': 'isolation_forest'
}
response = requests.post(
    'http://localhost:8000/api/v1/analysis/defects',
    json=analysis_data,
    headers=headers
)
defects = response.json()
print(f"Обнаружено дефектов: {defects['defects_count']}")
```

### cURL примеры

```bash
# Логин
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'

# Получить сканирования
curl -X GET "http://localhost:8000/api/v1/scans?limit=10"

# Создать сканирование
curl -X POST http://localhost:8000/api/v1/scans \
  -H "Content-Type: application/json" \
  -d '{"scan_type":"spm","surface_type":"graphite","width":256,"height":256}'
```

---

## Rate Limiting

В текущей версии rate limiting не реализован. Для production рекомендуется добавить:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/api/v1/scans")
@limiter.limit("100/minute")
async def get_scans(request: Request):
    ...
```

---

## Production развертывание

### Docker Compose

```bash
docker-compose -f docker-compose.api.yml up -d
```

### Gunicorn + Uvicorn

```bash
gunicorn api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log
```

---

*Документация создана: 2026-03-11*
