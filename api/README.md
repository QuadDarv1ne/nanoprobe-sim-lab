# FastAPI REST API для Nanoprobe Sim Lab

## 📖 Обзор

REST API для Лаборатории моделирования нанозонда, реализованное на **FastAPI**.

### Возможности

- ✅ **CRUD операции** для сканирований, симуляций
- ✅ **AI/ML анализ** дефектов через API
- ✅ **Сравнение поверхностей** с метриками (SSIM, PSNR, MSE)
- ✅ **PDF отчёты** для научных публикаций
- ✅ **JWT аутентификация** с refresh токенами
- ✅ **WebSocket** для real-time обновлений
- ✅ **Автодокументация** (Swagger UI, ReDoc)
- ✅ **Интеграция** с существующими utils

---

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
# Основные зависимости
pip install -r requirements.txt

# API зависимости
pip install -r requirements-api.txt
```

### 2. Запуск API

```bash
# Через скрипт запуска
python run_api.py --reload

# Или напрямую через uvicorn
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Проверка работы

Откройте в браузере:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

---

## 📁 Структура API

```
api/
├── main.py                 # Главное приложение FastAPI
├── schemas.py              # Pydantic схемы для валидации
└── routes/
    ├── __init__.py
    ├── auth.py             # Аутентификация (JWT)
    ├── scans.py            # CRUD сканирований
    ├── simulations.py      # CRUD симуляций
    ├── analysis.py         # AI/ML анализ дефектов
    ├── comparison.py       # Сравнение поверхностей
    └── reports.py          # Генерация PDF отчётов
```

---

## 🔐 Аутентификация

### Тестовые учетные данные

```
Username: admin
Password: admin123

Username: user
Password: user123
```

### Пример входа

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'
```

---

## 📊 Основные эндпоинты

### Сканирования

| Метод | Эндпоинт | Описание |
|-------|----------|----------|
| GET | `/api/v1/scans` | Список сканирований |
| GET | `/api/v1/scans/{id}` | Детали сканирования |
| POST | `/api/v1/scans` | Создать сканирование |
| DELETE | `/api/v1/scans/{id}` | Удалить сканирование |
| GET | `/api/v1/scans/search/{query}` | Поиск |

### Симуляции

| Метод | Эндпоинт | Описание |
|-------|----------|----------|
| GET | `/api/v1/simulations` | Список симуляций |
| POST | `/api/v1/simulations` | Создать симуляцию |
| PATCH | `/api/v1/simulations/{id}` | Обновить статус |

### Анализ дефектов

| Метод | Эндпоинт | Описание |
|-------|----------|----------|
| POST | `/api/v1/analysis/defects` | Анализ изображения |
| GET | `/api/v1/analysis/defects/history` | История анализов |

### Сравнение поверхностей

| Метод | Эндпоинт | Описание |
|-------|----------|----------|
| POST | `/api/v1/comparison` | Сравнить две поверхности |
| GET | `/api/v1/comparison/history` | История сравнений |

### PDF Отчёты

| Метод | Эндпоинт | Описание |
|-------|----------|----------|
| POST | `/api/v1/reports` | Сгенерировать отчёт |
| GET | `/api/v1/reports` | Список отчётов |
| GET | `/api/v1/reports/{id}/download` | Скачать отчёт |

---

## 🔌 WebSocket

### Подключение

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/realtime');

ws.onopen = () => {
  // Подписка на канал
  ws.send(JSON.stringify({
    type: 'subscribe',
    channel: 'scans'
  }));
};

ws.onmessage = (event) => {
  console.log('Received:', JSON.parse(event.data));
};
```

---

## 🧪 Тестирование

```bash
# Запуск тестов
pytest tests/test_api.py -v

# С покрытием
pytest tests/test_api.py -v --cov=api --cov-report=html
```

---

## 🐳 Docker

### Запуск через Docker Compose

```bash
docker-compose -f docker-compose.api.yml up -d
```

### Проверка логов

```bash
docker logs -f nanoprobe-api
```

---

## 📈 Production развертывание

### Gunicorn + Uvicorn

```bash
gunicorn api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

### Переменные окружения

Создайте файл `.env`:

```bash
# .env
JWT_SECRET=your-super-secret-key-min-32-characters
DATABASE_PATH=data/nanoprobe.db
LOG_LEVEL=info
CORS_ORIGINS=["https://your-domain.com"]
```

---

## 📝 Примеры использования

### Python клиент

```python
import requests

# Логин
response = requests.post(
    'http://localhost:8000/api/v1/auth/login',
    json={'username': 'admin', 'password': 'admin123'}
)
token = response.json()['access_token']
headers = {'Authorization': f'Bearer {token}'}

# Создание сканирования
scan = requests.post(
    'http://localhost:8000/api/v1/scans',
    json={'scan_type': 'spm', 'width': 256, 'height': 256},
    headers=headers
).json()

# Анализ дефектов
analysis = requests.post(
    'http://localhost:8000/api/v1/analysis/defects',
    json={'image_path': 'output/surface.png'},
    headers=headers
).json()

print(f"Дефектов: {analysis['defects_count']}")
```

---

## 🛡️ Безопасность

### Текущая реализация

- ✅ JWT токены с expiration
- ✅ Refresh токены
- ✅ HTTPS рекомендуется для production
- ✅ CORS настройки

### Рекомендуется добавить

- [ ] Rate limiting (slowapi)
- [ ] API keys для сервисов
- [ ] OAuth2 провайдеры
- [ ] Audit логирование

---

## 🔧 Конфигурация

### uvicorn настройки

```python
uvicorn.run(
    "api.main:app",
    host="0.0.0.0",
    port=8000,
    reload=True,        # Auto-reload для разработки
    log_level="info",   # debug, info, warning, error
    workers=1,          # Количество workers для production
)
```

---

## 📊 Метрики

### Health Check

```bash
curl http://localhost:8000/health
```

**Ответ:**
```json
{
  "status": "healthy",
  "timestamp": "2026-03-11T10:00:00",
  "version": "1.0.0"
}
```

---

## 🤝 Интеграция с Flask

API работает параллельно с Flask веб-интерфейсом:

- **FastAPI**: порт 8000 (REST API)
- **Flask**: порт 5000 (Веб-интерфейс)

Flask может использовать FastAPI как backend:

```python
# Во Flask
import requests

API_URL = "http://localhost:8000/api/v1"

def get_scans():
    response = requests.get(f"{API_URL}/scans")
    return response.json()
```

---

## 📚 Документация

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Spec**: http://localhost:8000/openapi.json
- **API MD**: `docs/API.md`

---

## ⚙️ Зависимости

### Основные

- `fastapi>=0.109.0`
- `uvicorn[standard]>=0.27.0`
- `pydantic>=2.5.0`
- `PyJWT>=2.8.0`

### Опционально

- `gunicorn>=21.2.0` (production)
- `prometheus-client` (метрики)
- `sentry-sdk` (мониторинг ошибок)

---

## 🎯 Roadmap

- [ ] Rate limiting
- [ ] GraphQL поддержка
- [ ] gRPC для микросервисов
- [ ] Кэширование (Redis)
- [ ] Фоновые задачи (Celery)
- [ ] Версионирование API (v2)

---

## 📞 Контакты

**Школа программирования Maestro7IT**
- Email: maksimqwe42@mail.ru
- Сайт: https://school-maestro7it.ru/

---

*Последнее обновление: 2026-03-11*
