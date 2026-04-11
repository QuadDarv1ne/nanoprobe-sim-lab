# Руководство по запуску Nanoprobe Sim Lab

**Версия:** 2.0
**Дата:** 2026-03-15
**Статус:** ✅ Готово

---

## 🚀 Быстрый старт

### Windows

```bash
# Интерактивный выбор
start.bat

# Конкретный режим
start.bat flask    # Flask + FastAPI
start.bat nextjs   # Next.js + FastAPI
start.bat api      # Только Backend API
start.bat full     # Full Stack (Flask + FastAPI + Sync)
```

### Linux/macOS

```bash
# Интерактивный выбор
python start_universal.py

# Конкретный режим
python start_universal.py flask    # Flask + FastAPI
python start_universal.py nextjs   # Next.js + FastAPI
python start_universal.py api-only # Только Backend API
python start_universal.py full     # Full Stack
```

---

## 📋 Доступные режимы

### 1. Flask + FastAPI (Unified)

**Команда:** `python start_universal.py flask`

**Порты:**
- Backend (FastAPI): http://localhost:8000
- Frontend (Flask): http://localhost:5000
- Swagger UI: http://localhost:8000/docs

**Компоненты:**
- FastAPI Backend
- Flask Dashboard (Unified)
- Reverse Proxy (автоматическая интеграция)

**Использование:**
- Основной режим для разработки
- Полная интеграция с FastAPI
- WebSocket real-time обновления
- Аутентификация через FastAPI

---

### 2. Next.js + FastAPI

**Команда:** `python start_universal.py nextjs`

**Порты:**
- Backend (FastAPI): http://localhost:8000
- Frontend (Next.js): http://localhost:3000
- Swagger UI: http://localhost:8000/docs

**Компоненты:**
- FastAPI Backend
- Next.js 14 Dashboard (TypeScript)
- Modern UI/UX

**Использование:**
- Современный frontend
- TypeScript поддержка
- PWA возможности
- Лучшая производительность

---

### 3. Backend API Only

**Команда:** `python start_universal.py api-only`

**Порты:**
- Backend (FastAPI): http://localhost:8000
- Swagger UI: http://localhost:8000/docs

**Компоненты:**
- FastAPI Backend (33+ endpoints)

**Использование:**
- Тестирование API
- Разработка мобильных приложений
- Интеграция с другими сервисами

---

### 4. Full Stack (Flask + FastAPI + Sync Manager)

**Команда:** `python start_universal.py full`

**Порты:**
- Backend (FastAPI): http://localhost:8000
- Frontend (Flask): http://localhost:5000
- Sync Manager: автоматическая синхронизация

**Компоненты:**
- FastAPI Backend
- Flask Dashboard (Unified)
- Sync Manager (WebSocket bridge)

**Использование:**
- Production режим
- Расширенная синхронизация
- Health monitoring
- Автоматическое переподключение

---

## 🔧 Требования

### Python

- Python 3.11, 3.12, 3.13, or 3.14
- pip install -r requirements.txt

### Для Flask

```bash
pip install flask flask-socketio requests
```

### Для Next.js

```bash
# Node.js 18+ required
cd frontend
npm install
npm run dev
```

### Для Backend

```bash
pip install fastapi uvicorn slowapi redis
```

---

## 📁 Скрипты запуска

| Скрипт | Описание |
|--------|----------|
| `start_universal.py` | Универсальный лаунчер (рекомендуется) |
| `start.py` | Базовый лаунчер (поддерживается) |
| `start_all.py` | Запуск с Sync Manager |
| `start.bat` | Windows batch файл |

---

## 🏗️ Архитектура

```
┌─────────────────────────────────────────────────────────┐
│                    User Browser                         │
│              http://localhost:5000/3000                 │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   Frontend             │
        │   - Flask (5000)       │
        │   - Next.js (3000)     │
        └───────────┬────────────┘
                    │
                    │ Reverse Proxy
                    │ WebSocket
                    ▼
        ┌────────────────────────┐
        │   Backend (FastAPI)    │
        │   Port: 8000           │
        │   - 33+ endpoints      │
        │   - JWT Auth           │
        │   - Redis Cache        │
        └───────────┬────────────┘
                    │
                    │ Database
                    ▼
        ┌────────────────────────┐
        │   SQLite Database      │
        │   data/nanoprobe.db    │
        └────────────────────────┘
```

---

## 🔍 Проверка работоспособности

### Backend Health Check

```bash
curl http://localhost:8000/health
```

**Ответ:**
```json
{
  "status": "healthy",
  "components": {
    "database": "ok",
    "redis": "ok",
    "timestamp": "2026-03-15T12:00:00"
  }
}
```

### Frontend Check

**Flask:**
```bash
curl http://localhost:5000/api/health
```

**Next.js:**
```bash
curl http://localhost:3000/api/health
```

---

## 🐛 Troubleshooting

### Порт занят

**Проблема:** `Port 8000 is already in use`

**Решение:**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/macOS
lsof -i :8000
kill -9 <PID>
```

### ModuleNotFoundError

**Проблема:** `No module named 'flask'`

**Решение:**
```bash
pip install flask flask-socketio
```

### npm install fails

**Проблема:** `npm install` завершается с ошибкой

**Решение:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Sync Manager не запускается

**Проблема:** `Sync Manager failed to start`

**Решение:**
1. Убедитесь, что Backend запущен
2. Проверьте `api/sync_manager.py`
3. Установите зависимости: `pip install aiohttp`

---

## 📊 Сравнение режимов

| Режим | Backend | Frontend | Sync | Использование |
|-------|---------|----------|------|-------------|
| Flask | ✅ | ✅ (5000) | ❌ | Разработка |
| Next.js | ✅ | ✅ (3000) | ❌ | Production |
| API Only | ✅ | ❌ | ❌ | Testing |
| Full | ✅ | ✅ (5000) | ✅ | Production |

---

## 🎯 Рекомендации

### Для разработки

```bash
python start_universal.py flask
```

### Для production

```bash
python start_universal.py full
```

### Для тестирования API

```bash
python start_universal.py api-only
```

### Для современного UI

```bash
python start_universal.py nextjs
```

---

## 📚 Дополнительная документация

- [UNIFIED_DASHBOARD.md](docs/UNIFIED_DASHBOARD.md) - Flask Dashboard
- [SYNC.md](docs/SYNC.md) - Синхронизация Backend ↔ Frontend
- [API Reference](docs/api_reference.md) - API документация

---

**Nanoprobe Sim Lab - Universal Launcher** 🛰️
