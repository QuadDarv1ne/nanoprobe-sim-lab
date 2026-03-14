# 🚀 Запуск Nanoprobe Sim Lab

**Последнее обновление:** 2026-03-14

---

## 📋 Быстрый старт

### Вариант 1: Синхронизированный запуск (Рекомендуется)

```bash
python start_all.py
```

**Что происходит:**
1. ✅ Проверка зависимостей
2. ✅ Запуск Backend (FastAPI, порт 8000)
3. ✅ Проверка здоровья Backend
4. ✅ Запуск Frontend (Flask, порт 5000)
5. ✅ Проверка здоровья Frontend
6. ✅ Автоматическая синхронизация данных
7. ✅ Открытие браузера с дашбордом

**Доступ к сервисам:**
- 🖥️ **Frontend:** http://localhost:5000
- 📊 **Backend API:** http://localhost:8000
- 📚 **Swagger UI:** http://localhost:8000/docs

---

### Вариант 2: Раздельный запуск

#### Шаг 1: Запуск Backend

```bash
python run_api.py
```

Ожидание сообщения:
```
[OK] Database initialized
[OK] Redis cache connected
Uvicorn running on http://0.0.0.0:8000
```

#### Шаг 2: Запуск Frontend

```bash
python src\web\web_dashboard.py
```

Ожидание сообщения:
```
* Running on http://127.0.0.1:5000
Веб-панель инициализирована
```

---

## 🔧 Требования

### Python
- Python 3.10+ (рекомендуется 3.13)

### Зависимости

**Backend (FastAPI):**
```bash
pip install -r requirements-api.txt
```

**Frontend (Flask):**
```bash
pip install -r requirements.txt
```

**Все зависимости:**
```bash
pip install -r requirements.txt -r requirements-api.txt
```

### Опционально: Redis

Для кэширования и refresh токенов:

```bash
# Windows (через Docker)
docker run -d -p 6379:6379 redis:latest
```

---

## 📁 Структура проекта

```
nanoprobe-sim-lab/
├── api/                        # Backend (FastAPI)
│   ├── main.py                 # Точка входа
│   ├── sync_manager.py         # Менеджер синхронизации
│   ├── reverse_proxy.py        # Reverse proxy для Flask
│   └── routes/                 # API маршруты
│       ├── dashboard.py        # Dashboard API
│       ├── scans.py            # Сканирования
│       ├── simulations.py      # Симуляции
│       ├── analysis.py         # Анализ
│       ├── comparison.py       # Сравнение
│       ├── reports.py          # Отчёты
│       ├── auth.py             # Аутентификация
│       └── ...
│
├── src/web/                    # Frontend (Flask)
│   └── web_dashboard.py        # Веб-интерфейс
│
├── templates/                  # HTML шаблоны
│   └── dashboard.html          # Главная страница
│
├── components/                 # Компоненты проекта
│   ├── cpp-spm-hardware-sim/   # СЗМ симулятор
│   ├── py-surface-image-analyzer/  # Анализатор изображений
│   └── py-sstv-groundstation/  # SSTV станция
│
├── utils/                      # Утилиты
│   ├── database.py             # БД менеджер
│   ├── cache_manager.py        # Кэш менеджер
│   └── ...
│
├── data/                       # Данные
│   └── nanoprobe.db            # SQLite БД
│
├── logs/                       # Логи
│   ├── backend.log
│   ├── frontend.log
│   └── components/
│
├── start_all.py                # Синхронизированный запуск
└── docs/
    └── SYNC.md                 # Документация по синхронизации
```

---

## 🔐 Конфигурация

### .env файл

Скопируйте `.env.example` в `.env` и настройте:

```bash
# Безопасность
JWT_SECRET=your-super-secret-key-min-32-chars

# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# CORS (важно для синхронизации!)
CORS_ORIGINS=["http://localhost:3000","http://localhost:5000","http://127.0.0.1:5000"]

# Redis (опционально)
REDIS_HOST=localhost
REDIS_PORT=6379

# База данных
DATABASE_PATH=data/nanoprobe.db
```

---

## 📊 API Endpoints

### Backend (FastAPI) - Порт 8000

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/health` | GET | Health check |
| `/docs` | GET | Swagger UI |
| `/api/v1/dashboard/stats` | GET | Статистика дашборда |
| `/api/v1/dashboard/health/detailed` | GET | Детальный health |
| `/api/v1/dashboard/metrics/realtime` | GET | Метрики realtime |
| `/api/v1/scans` | GET, POST | Сканирования |
| `/api/v1/simulations` | GET, POST, PUT, DELETE | Симуляции |
| `/api/v1/auth/login` | POST | Вход |
| `/ws/realtime` | WebSocket | Real-time обновления |

### Frontend (Flask) - Порт 5000

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/` | GET | Главная страница |
| `/api/health` | GET | Health check |
| `/api/component_status` | GET | Статус компонентов |
| `/api/v1/*` | Все | Reverse proxy к Backend |
| `/socket.io` | WebSocket | Socket.IO события |

---

## 🔄 Синхронизация

### Sync Manager

Автоматическая синхронизация между Backend и Frontend:

```python
# В start_all.py
sync_interval = 5  # секунд

# Автоматическая синхронизация:
# 1. Получение статистики из Backend
# 2. Проверка здоровья сервисов
# 3. Health check каждые 30 секунд
```

### WebSocket

**Backend WebSocket:**
```
ws://localhost:8000/ws/realtime
```

**Frontend Socket.IO:**
```
http://localhost:5000/socket.io
```

---

## 🛠️ Устранение неполадок

### Порт 8000 занят

```bash
# Windows: найти процесс
netstat -ano | findstr :8000

# Убить процесс
taskkill /PID <PID> /F
```

### Порт 5000 занят

```bash
# Windows: найти процесс
netstat -ano | findstr :5000

# Убить процесс
taskkill /PID <PID> /F
```

### Ошибка CORS

Убедитесь что `.env` содержит:
```
CORS_ORIGINS=["http://localhost:5000","http://127.0.0.1:5000"]
```

### Backend не запускается

```bash
# Проверка логов
cat logs/backend.log

# Проверка зависимостей
pip install -r requirements-api.txt

# Запуск в режиме отладки
python run_api.py --reload
```

### Frontend не подключается к Backend

```bash
# Проверка доступности Backend
curl http://localhost:8000/health

# Проверка reverse proxy
curl http://localhost:5000/api/v1/admin/health
```

---

## 📚 Документация

- [README.md](../README.md) - Основная документация
- [TODO.md](../TODO.md) - План развития
- [IMPROVEMENTS.md](../IMPROVEMENTS.md) - Улучшения
- [SYNC.md](SYNC.md) - Синхронизация (подробно)

---

## 🎯 Примеры использования

### Python клиент

```python
import requests

# Получение статистики
response = requests.get("http://localhost:8000/api/v1/dashboard/stats")
stats = response.json()
print(f"Total scans: {stats['total_scans']}")

# Получение метрик
response = requests.get("http://localhost:8000/api/v1/dashboard/metrics/realtime")
metrics = response.json()
print(f"CPU: {metrics['cpu_percent']}%")
```

### JavaScript клиент

```javascript
// Подключение к Socket.IO
const socket = io('http://localhost:5000');

socket.on('connect', () => {
    console.log('Connected!');
    
    // Запрос статистики
    socket.emit('request_stats');
});

socket.on('stats', (data) => {
    console.log('Stats:', data);
    updateDashboard(data);
});
```

---

## 📈 Мониторинг

### Логи

```bash
# Backend логи
tail -f logs/backend.log

# Frontend логи
tail -f logs/frontend.log

# Логи компонентов
tail -f logs/components/spm_simulator_stdout.log
```

### Health Check

```bash
# Backend
curl http://localhost:8000/health

# Frontend
curl http://localhost:5000/api/health

# Детальный health
curl http://localhost:8000/api/v1/dashboard/health/detailed
```

---

## ✅ Чеклист перед запуском

- [ ] Python 3.10+ установлен
- [ ] Зависимости установлены (`pip install -r requirements*.txt`)
- [ ] `.env` файл настроен
- [ ] Порты 8000 и 5000 свободны
- [ ] Redis запущен (опционально)
- [ ] База данных существует (`data/nanoprobe.db`)

---

**Готово к запуску!** 🚀

```bash
python start_all.py
```
