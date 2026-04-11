# 🚀 Nanoprobe Sim Lab - Quick Start Guide

**Last Updated:** 2026-03-15
**Version:** 3.0 (Unified Launcher)

---

## ⚡ Быстрый старт

### 1. Установка зависимостей

```bash
# Python 3.11, 3.12, 3.13, or 3.14
pip install -r requirements.txt
pip install -r requirements-api.txt

# Для Next.js frontend (опционально)
cd frontend
npm install
```

### 2. Настройка окружения

```bash
# Скопируйте .env.example в .env
cp .env.example .env

# Отредактируйте .env (опционально)
# - JWT_SECRET
# - NASA_API_KEY (получите на https://api.nasa.gov/)
# - REDIS_HOST (если используете Redis)
```

### 3. Запуск проекта

#### 🎯 Интерактивный режим (рекомендуется)

```bash
python main.py
```

Откроется меню выбора режима.

#### 🔧 Конкретный режим

```bash
# Flask + FastAPI + Auto Sync
python main.py flask

# Next.js + FastAPI + Auto Sync
python main.py nextjs

# Только Backend API
python main.py api-only

# Full Stack (Flask + FastAPI + Sync Manager)
python main.py full

# Development mode (Flask + reload)
python main.py dev
```

#### 🪟 Windows (one-click)

```bash
start.bat
start.bat flask
```

---

## 📋 Доступные режимы

| Режим | Backend | Frontend | Sync | Порт | Описание |
|-------|---------|----------|------|------|----------|
| **flask** | ✅ FastAPI | ✅ Flask | ✅ | 5000 | Основной режим |
| **nextjs** | ✅ FastAPI | ✅ Next.js | ✅ | 3000 | Modern UI |
| **api-only** | ✅ FastAPI | ❌ | ❌ | 8000 | Только API |
| **full** | ✅ FastAPI | ✅ Flask | ✅ | 5000 | Full Stack |
| **dev** | ✅ (reload) | ✅ Flask | ✅ | 5000 | Development |

---

## 🔗 Полезные ссылки

### Backend API

| URL | Описание |
|-----|----------|
| http://localhost:8000/docs | Swagger UI (API документация) |
| http://localhost:8000/health | Health check |
| http://localhost:8000/api/v1/sync/status | Sync Manager статус |

### Frontend

| Режим | URL |
|-------|-----|
| Flask | http://localhost:5000 |
| Next.js | http://localhost:3000 |

---

## 🛠️ Управление процессами

### Остановка

Нажмите `Ctrl+C` в терминале для остановки всех сервисов.

### Проверка статуса

```bash
# Backend health
curl http://localhost:8000/health

# Sync Manager статус
curl http://localhost:8000/api/v1/sync/status

# Frontend (Flask)
curl http://localhost:5000/api/health
```

---

## 🐛 Troubleshooting

### Порт занят

```bash
# Windows: найти процесс
netstat -ano | findstr :8000

# Windows: убить процесс
taskkill /PID <PID> /F

# Linux/macOS
lsof -i :8000
kill -9 <PID>
```

### ModuleNotFoundError

```bash
# Установите зависимости
pip install -r requirements.txt
pip install -r requirements-api.txt
```

### Next.js не запускается

```bash
cd frontend
npm install
npm run dev
```

---

## 📚 Дополнительная документация

- [STARTUP_GUIDE.md](docs/STARTUP_GUIDE.md) - Полное руководство по запуску
- [UNIFIED_DASHBOARD.md](docs/UNIFIED_DASHBOARD.md) - Flask Dashboard документация
- [AUTO_SYNC.md](docs/AUTO_SYNC.md) - Автоматическая синхронизация
- [NASA_API_KEY_GUIDE.md](docs/NASA_API_KEY_GUIDE.md) - Получение NASA API ключа

---

## ✅ Чек-лист успешного запуска

- [ ] Python 3.11-3.14 установлен
- [ ] Зависимости установлены (`pip install -r requirements.txt`)
- [ ] `.env` файл создан и настроен
- [ ] Backend запускается (`python main.py api-only`)
- [ ] Frontend запускается (`python main.py flask`)
- [ ] Sync Manager работает (статус через `/api/v1/sync/status`)

---

**Проект готов к использованию!** 🎉
