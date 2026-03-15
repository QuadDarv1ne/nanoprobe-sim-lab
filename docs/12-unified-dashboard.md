# Унифицированная веб-панель (Unified Dashboard)

**Версия:** 2.0  
**Дата:** 2026-03-15  
**Статус:** ✅ Готово к использованию

---

## 📋 Обзор

Унифицированная веб-панель управления Nanoprobe Sim Lab с полной интеграцией FastAPI.

### Основные возможности:

- ✅ **Reverse proxy** к FastAPI API (http://localhost:8000)
- ✅ **Аутентификация** через FastAPI (JWT токены)
- ✅ **WebSocket** для real-time обновлений
- ✅ **Кэширование** Redis (для API endpoints)
- ✅ **SSTV Ground Station** (приём с МКС)
- ✅ **СЗМ симулятор** (сканирование поверхностей)
- ✅ **Анализ изображений** (дефекты, сравнение)
- ✅ **Экспорт данных** (JSON, CSV)

---

## 🚀 Быстрый старт

### Вариант 1: Через start_all.py (рекомендуется)

```bash
python start_all.py
```

Автоматически запускает:
- Backend (FastAPI:8000)
- Frontend (Flask:5000)
- Sync Manager (синхронизация)

### Вариант 2: Только веб-панель

```bash
python src/web/web_dashboard_unified.py
```

Параметры:
```bash
python src/web/web_dashboard_unified.py --host 127.0.0.1 --port 5000
python src/web/web_dashboard_unified.py --no-browser  # Не открывать браузер
```

### Вариант 3: Через start.py

```bash
python start.py flask  # Запуск Flask dashboard
```

---

## 📁 Файлы

| Файл | Описание |
|------|----------|
| `src/web/web_dashboard_unified.py` | Основной Python модуль (631 строка) |
| `templates/dashboard_unified.html` | HTML шаблон (2540 строк) |
| `templates/sstv_station.html` | SSTV Ground Station шаблон |
| `templates/login.html` | Страница входа |

---

## 🔌 API Endpoints

### Аутентификация

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/api/auth/login` | POST | Вход в систему |
| `/api/auth/logout` | POST | Выход из системы |
| `/api/auth/refresh` | POST | Обновление токена |

### Система

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/api/system_info` | GET | Информация о системе |
| `/api/stats` | GET | Статистика дашборда |
| `/api/health` | GET | Проверка здоровья сервисов |

### СЗМ операции

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/api/spm/simulate` | POST | Запуск симуляции СЗМ |
| `/api/spm/scan` | POST | Сканирование поверхности |

### Анализ

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/api/analysis/compare` | POST | Сравнение поверхностей |
| `/api/analysis/defects` | POST | Анализ дефектов |

### Экспорт

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/api/export/json` | GET | Экспорт в JSON |
| `/api/export/csv` | GET | Экспорт в CSV |

---

## 🔐 Аутентификация

### Вход через форму

```html
<form action="/api/auth/login" method="POST">
    <input type="text" name="username" placeholder="Имя пользователя">
    <input type="password" name="password" placeholder="Пароль">
    <button type="submit">Войти</button>
</form>
```

### Пример запроса (curl)

```bash
curl -X POST "http://localhost:5000/api/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123"
```

### Ответ

```json
{
  "success": true,
  "username": "admin",
  "message": "Вход выполнен успешно"
}
```

---

## 📡 WebSocket события

### Подключение

```javascript
const socket = io('http://localhost:5000');

socket.on('connect', () => {
    console.log('Подключено к WebSocket');
});
```

### Подписка на метрики

```javascript
socket.emit('subscribe_metrics');

socket.on('metrics_update', (metrics) => {
    console.log('CPU:', metrics.cpu_percent);
    console.log('Memory:', metrics.memory_percent);
});
```

### Подписка на статистику

```javascript
socket.emit('subscribe_stats');

socket.on('stats_update', (stats) => {
    console.log('Scans:', stats.scans_count);
    console.log('Simulations:', stats.simulations_count);
});
```

---

## 🎨 Темы оформления

Поддерживаются две темы:

### Тёмная тема (по умолчанию)

```
--bg-primary: #0f172a
--bg-secondary: #1e293b
--text-primary: #f1f5f9
--primary: #3b82f6
```

### Светлая тема

```
--bg-primary: #f8fafc
--bg-secondary: #ffffff
--text-primary: #1e293b
--primary: #3b82f6
```

Переключение темы: кнопка в верхнем правом углу.

---

## 🔧 Конфигурация

### Переменные окружения

```bash
# Flask
FLASK_SECRET_KEY=your-secret-key-change-in-production

# FastAPI интеграция
FASTAPI_URL=http://localhost:8000
JWT_SECRET=your-jwt-secret-key

# Redis (опционально)
REDIS_HOST=localhost
REDIS_PORT=6379
```

### .env файл

```bash
cp .env.example .env
# Отредактируйте .env
```

---

## 🛠️ Интеграция с FastAPI

### Reverse proxy

Автоматически регистрируется при наличии модуля `api.reverse_proxy`:

```python
from api.reverse_proxy import register_proxy
register_proxy(app)
```

### Маршруты прокси

Все запросы к `/api/v1/*` автоматически перенаправляются к FastAPI:

```
Flask (5000) → Reverse Proxy → FastAPI (8000)
```

### JWT токены

Токены сохраняются в сессии Flask:

```python
session['access_token'] = tokens.get('access_token')
session['refresh_token'] = tokens.get('refresh_token')
```

---

## 📊 Мониторинг

### Фоновое обновление метрик

Каждые 5 секунд:
- CPU usage
- Memory usage
- Disk usage
- Network stats

### WebSocket рассылка

Метрики отправляются всем подключённым клиентам.

---

## 🐛 Troubleshooting

### Reverse proxy недоступен

**Проблема:** `Reverse proxy недоступен (FastAPI интеграция отключена)`

**Решение:**
1. Убедитесь, что FastAPI запущен
2. Проверьте `FASTAPI_URL` в .env
3. Установите зависимости: `pip install fastapi uvicorn requests`

### WebSocket не подключается

**Проблема:** `WebSocket connection failed`

**Решение:**
1. Проверьте CORS настройки
2. Убедитесь, что порт 5000 не занят
3. Проверьте firewall

### Аутентификация не работает

**Проблема:** `Сервер аутентификации недоступен`

**Решение:**
1. Запустите FastAPI: `python run_api.py`
2. Проверьте `/health` endpoint
3. Убедитесь, что JWT_SECRET совпадает

---

## 📝 Сравнение версий

| Функция | Legacy (web_dashboard.py) | Unified (web_dashboard_unified.py) |
|---------|---------------------------|-------------------------------------|
| Reverse proxy | ✅ | ✅ |
| Аутентификация | ❌ | ✅ |
| WebSocket | ✅ | ✅ (улучшенный) |
| SSTV Station | ✅ | ✅ |
| СЗМ симулятор | ✅ | ✅ |
| Анализ изображений | ✅ | ✅ |
| Экспорт данных | ✅ | ✅ (улучшенный) |
| Real-time метрики | ✅ | ✅ (5 сек интервал) |
| Кэширование Redis | ❌ | ✅ (через FastAPI) |

---

## 🚀 Следующие шаги

### После запуска:

1. Откройте http://localhost:5000
2. Войдите в систему (admin/admin123 или создайте пользователя)
3. Проверьте статус сервисов в разделе Health
4. Запустите симуляцию СЗМ
5. Подключитесь к SSTV Ground Station

### Для production:

1. Измените `FLASK_SECRET_KEY`
2. Измените `JWT_SECRET`
3. Настройте HTTPS
4. Включите Redis кэширование
5. Настройте логирование

---

## 📚 Дополнительные ресурсы

- [API Reference](docs/api_reference.md)
- [SYNC.md](docs/SYNC.md) - Синхронизация Backend ↔ Frontend
- [STARTUP.md](docs/STARTUP.md) - Руководство по запуску

---

**Nanoprobe Sim Lab - Unified Dashboard** 🛰️
