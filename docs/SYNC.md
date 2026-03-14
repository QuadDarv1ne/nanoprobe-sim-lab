# Синхронизация Backend ↔ Frontend

**Последнее обновление:** 2026-03-14  
**Статус:** ✅ Реализовано

---

## 📋 Обзор архитектуры

```
┌─────────────────────────────────────────────────────────────────┐
│                     NANOPROBE SIM LAB                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐         ┌─────────────────┐               │
│  │   Backend       │         │   Frontend      │               │
│  │   (FastAPI)     │◄───────►│   (Flask)       │               │
│  │   Порт: 8000    │  HTTP   │   Порт: 5000    │               │
│  │                 │  WS     │                 │               │
│  └────────┬────────┘         └────────┬────────┘               │
│           │                           │                         │
│           │  ┌────────────────────────┤                         │
│           │  │  Sync Manager          │                         │
│           │  │  - API Integration     │                         │
│           │  │  - WebSocket Bridge    │                         │
│           │  │  - Health Monitoring   │                         │
│           │  └────────────────────────┘                         │
│                                                                 │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              Общие компоненты                       │       │
│  │  - SQLite БД (data/nanoprobe.db)                   │       │
│  │  - Redis кэш (localhost:6379)                      │       │
│  │  - Логирование (logs/)                             │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Компоненты синхронизации

### 1. Sync Manager (`api/sync_manager.py`)

Централизованный менеджер синхронизации:

```python
from api.sync_manager import BackendFrontendSync

sync = BackendFrontendSync(
    backend_url="http://localhost:8000",
    frontend_url="http://localhost:5000"
)

# Синхронизация статистики
stats = await sync.sync_dashboard_stats()

# Синхронизация метрик
metrics = await sync.sync_realtime_metrics()

# Отправка события во Frontend
await sync.broadcast_to_frontend("update", {"data": value})
```

**Функции:**
- ✅ Health monitoring Backend/Frontend
- ✅ Синхронизация статистики дашборда
- ✅ Трансляция метрик реального времени
- ✅ WebSocket bridge между сервисами

---

### 2. Reverse Proxy (`api/reverse_proxy.py`)

Flask Blueprint для проксирования запросов к FastAPI:

```python
from api.reverse_proxy import register_proxy

# Регистрация во Flask приложении
register_proxy(flask_app)
```

**Проксированные эндпоинты:**

| Flask Route | FastAPI Endpoint | Метод |
|-------------|------------------|-------|
| `/api/v1/auth/login` | `/api/v1/auth/login` | POST |
| `/api/v1/auth/refresh` | `/api/v1/auth/refresh` | POST |
| `/api/v1/scans` | `/api/v1/scans` | GET, POST |
| `/api/v1/scans/<id>` | `/api/v1/scans/<id>` | GET, DELETE |
| `/api/v1/simulations` | `/api/v1/simulations` | GET, POST, PUT, DELETE |
| `/api/v1/analysis/defects` | `/api/v1/analysis/defects` | POST |
| `/api/v1/comparison/surfaces` | `/api/v1/comparison/surfaces` | POST |
| `/api/v1/reports` | `/api/v1/reports` | GET, POST |
| `/api/v1/admin/stats` | `/api/v1/admin/stats` | GET |
| `/api/v1/admin/health` | `/api/v1/admin/health` | GET |

---

### 3. WebSocket Интеграция

#### Backend WebSocket (FastAPI)
```
ws://localhost:8000/ws/realtime
```

**Команды:**
- `subscribe` - подписка на канал
- `unsubscribe` - отписка от канала
- `get_metrics` - запрос метрик

**Каналы:**
- `system` - системные метрики
- `scans` - обновления сканирований
- `simulations` - обновления симуляций
- `alerts` - алерты системы

#### Frontend WebSocket (Flask-Socket.IO)
```
http://localhost:5000/socket.io
```

**События:**
- `connect` - подключение клиента
- `disconnect` - отключение клиента
- `request_stats` - запрос статистики
- `request_metrics` - запрос метрик
- `component_status` - статус компонентов

---

## 🚀 Запуск синхронизации

### Вариант 1: Через start_all.py

```bash
python start_all.py
```

Автоматический запуск:
1. ✅ Проверка зависимостей
2. ✅ Запуск Backend (FastAPI)
3. ✅ Проверка здоровья Backend
4. ✅ Запуск Frontend (Flask)
5. ✅ Проверка здоровья Frontend
6. ✅ Автоматическая синхронизация данных
7. ✅ Открытие браузера с дашбордом

### Вариант 2: Раздельный запуск

**Backend:**
```bash
python run_api.py
```

**Frontend:**
```bash
python src/web/web_dashboard.py
```

---

## 📊 API Endpoints

### Backend (FastAPI) - Порт 8000

#### Dashboard
| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/api/v1/dashboard/stats` | GET | Сводная статистика |
| `/api/v1/dashboard/health/detailed` | GET | Детальный health check |
| `/api/v1/dashboard/metrics/realtime` | GET | Метрики реального времени |
| `/api/v1/dashboard/metrics/history` | GET | История метрик |
| `/api/v1/dashboard/alerts` | GET | Список алертов |
| `/api/v1/dashboard/processes` | GET | Топ процессов |
| `/api/v1/dashboard/storage` | GET | Статистика хранилища |
| `/api/v1/dashboard/export/{format}` | POST | Экспорт данных |
| `/api/v1/dashboard/actions/clean_cache` | POST | Очистка кэша |
| `/api/v1/dashboard/actions/start_component` | POST | Запуск компонента |
| `/api/v1/dashboard/actions/stop_component` | POST | Остановка компонента |

#### Scans
| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/api/v1/scans` | GET, POST | Список сканирований / Создание |
| `/api/v1/scans/{id}` | GET, DELETE | Получение / Удаление |
| `/api/v1/scans/search/{query}` | GET | Поиск сканирований |

#### Simulations
| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/api/v1/simulations` | GET, POST | Список симуляций / Создание |
| `/api/v1/simulations/{id}` | GET, PUT, DELETE | Получение / Обновление / Удаление |

#### Analysis
| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/api/v1/analysis/defects` | POST | Анализ дефектов |
| `/api/v1/analysis/history` | GET | История анализов |

#### Comparison
| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/api/v1/comparison/surfaces` | POST | Сравнение поверхностей |
| `/api/v1/comparison/history` | GET | История сравнений |

#### Reports
| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/api/v1/reports` | GET | Список отчётов |
| `/api/v1/reports/{id}` | GET | Получение отчёта |
| `/api/v1/reports/generate` | POST | Генерация отчёта |
| `/api/v1/reports/{id}/download` | GET | Скачивание отчёта |

#### Auth
| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/api/v1/auth/login` | POST | Вход |
| `/api/v1/auth/refresh` | POST | Обновление токена |
| `/api/v1/auth/logout` | POST | Выход |
| `/api/v1/auth/me` | GET | Текущий пользователь |

#### Admin
| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/api/v1/admin/stats` | GET | Общая статистика |
| `/api/v1/admin/health` | GET | Проверка здоровья |

#### GraphQL
| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/api/v1/graphql` | POST | GraphQL запросы |
| `/api/v1/graphql/schema` | GET | GraphQL схема |

#### External Services
| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/api/v1/external/nasa/apod` | GET | NASA APOD |
| `/api/v1/external/zenodo` | GET | Zenodo |
| `/api/v1/external/figshare` | GET | Figshare |

---

### Frontend (Flask) - Порт 5000

#### HTTP Endpoints
| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/` | GET | Главная страница |
| `/api/system_info` | GET | Информация о системе |
| `/api/health` | GET | Health check |
| `/api/performance_data` | GET | Данные производительности |
| `/api/component_status` | GET | Статус компонентов |
| `/api/components` | GET | Список компонентов |
| `/api/logs` | GET | Логи системы |
| `/api/processes` | GET | Статус процессов |
| `/api/logs/component/{name}` | GET | Логи компонента |
| `/api/stats` | GET | Сводная статистика |
| `/api/actions/quick` | POST | Быстрые действия |
| `/api/config` | GET, POST | Конфигурация |
| `/api/cache_stats` | GET | Статистика кэша |
| `/api/actions/clean_cache` | POST | Очистка кэша |
| `/api/actions/start_component` | POST | Запуск компонента |
| `/api/actions/stop_component` | POST | Остановка компонента |
| `/api/actions/restart_component` | POST | Перезапуск компонента |

#### Socket.IO Events
| Событие | Направление | Описание |
|---------|-------------|----------|
| `connect` | Client → Server | Подключение клиента |
| `connected` | Server → Client | Подтверждение подключения |
| `disconnect` | Client → Server | Отключение клиента |
| `request_stats` | Client → Server | Запрос статистики |
| `request_metrics` | Client → Server | Запрос метрик |
| `stats` | Server → Client | Статистика |
| `metrics` | Server → Client | Метрики |
| `component_status` | Server → Client | Статус компонента |

---

## 🔍 Health Monitoring

### Проверка статуса

```bash
# Backend health
curl http://localhost:8000/health

# Frontend health
curl http://localhost:5000/api/health

# Backend detailed health
curl http://localhost:8000/api/v1/dashboard/health/detailed

# Backend metrics
curl http://localhost:8000/api/v1/dashboard/metrics/realtime
```

### Sync Manager Status

```python
sync_status = sync.get_sync_status()
# Возвращает:
# {
#     "running": True,
#     "backend_url": "http://localhost:8000",
#     "frontend_url": "http://localhost:5000",
#     "last_sync_time": "2026-03-14T12:00:00",
#     "backend_connections": 5,
#     "frontend_connections": 10
# }
```

---

## 🛠️ Устранение неполадок

### Backend не запускается

```bash
# Проверка логов
cat logs/backend.log

# Проверка порта
netstat -ano | findstr :8000

# Перезапуск
python run_api.py --reload
```

### Frontend не подключается к Backend

```bash
# Проверка доступности Backend
curl http://localhost:8000/health

# Проверка CORS
# Убедитесь что .env содержит:
# CORS_ORIGINS=["http://localhost:5000","http://127.0.0.1:5000"]
```

### WebSocket не работает

```bash
# Проверка WebSocket
wscat -c ws://localhost:8000/ws/realtime

# Проверка Socket.IO
# Откройте браузер и консоль разработчика
# Должны быть видны события подключения
```

---

## 📈 Метрики производительности

| Метрика | Значение | Единицы |
|---------|----------|---------|
| API Response Time | < 100 | мс |
| WebSocket Latency | < 50 | мс |
| Sync Interval | 5 | сек |
| Max Concurrent WS | 100 | клиентов |

---

## 📚 Связанные документы

- [README.md](../README.md) - Общая документация проекта
- [IMPROVEMENTS.md](../IMPROVEMENTS.md) - Улучшения проекта
- [TODO.md](../TODO.md) - План развития

---

**Статус синхронизации:** ✅ Полностью реализовано и протестировано
