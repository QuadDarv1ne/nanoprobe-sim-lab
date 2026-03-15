# Автоматическая синхронизация Nanoprobe Sim Lab

**Версия:** 2.0  
**Дата:** 2026-03-15  
**Статус:** ✅ Автоматически включена

---

## 🚀 Быстрый старт

### Запуск с автоматической синхронизацией

```bash
# Windows
start.bat flask    # Sync Manager запускается автоматически!

# Linux/macOS
python start_universal.py flask    # Sync Manager запускается автоматически!
```

**Всё!** Синхронизация работает автоматически каждые 5 секунд.

---

## 📊 Архитектура синхронизации

```
┌─────────────────────────────────────────────────────────┐
│                   Sync Manager                          │
│              (автоматический запуск)                    │
│                                                         │
│  ┌──────────────────┐    ┌──────────────────┐         │
│  │  Backend         │◄──►│  Frontend        │         │
│  │  FastAPI:8000    │    │  Flask:5000      │         │
│  │  - API endpoints │    │  - Dashboard     │         │
│  │  - Database      │    │  - WebSocket     │         │
│  │  - Redis Cache   │    │  - Socket.IO     │         │
│  └──────────────────┘    └──────────────────┘         │
│                                                         │
│  Интервал: 5 секунд                                     │
└─────────────────────────────────────────────────────────┘
```

---

## 🔧 Как это работает

### 1. Автоматический запуск

При запуске через `start_universal.py`:

```python
# Запуск Backend
backend_process = start_backend()

# АВТОМАТИЧЕСКИЙ запуск Sync Manager (кроме api-only)
if mode != "api-only" and SYNC_ENABLED_BY_DEFAULT:
    sync_process = start_sync_manager()  # ✅ Запускается!

# Запуск Frontend
frontend_process = start_flask_frontend()
```

### 2. Функции Sync Manager

**Health Monitoring:**
- Проверка Backend (каждые 5 сек)
- Проверка Frontend (каждые 5 сек)
- Автоматическое переподключение при сбоях

**Синхронизация данных:**
- Статистика дашборда
- Real-time метрики (CPU, Memory, Disk)
- WebSocket подключения

### 3. Интервал синхронизации

```python
SYNC_INTERVAL = 5  # секунд (по умолчанию)
```

---

## 📡 API Endpoints

### Проверка статуса синхронизации

**Request:**
```bash
GET http://localhost:8000/api/v1/sync/status
```

**Response:**
```json
{
  "running": true,
  "backend_url": "http://localhost:8000",
  "frontend_url": "http://localhost:5000",
  "last_sync_time": "2026-03-15T14:30:22.123456",
  "backend_connections": 1,
  "frontend_connections": 3
}
```

### Health check с Sync Manager

**Request:**
```bash
GET http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "components": {
    "flask": "ok",
    "fastapi": "ok",
    "database": "ok",
    "sync_manager": "ok",
    "sync_last_update": "2026-03-15T14:30:22"
  }
}
```

---

## 🎯 Режимы запуска

| Режим | Sync Manager | Интервал |
|-------|--------------|----------|
| `flask` | ✅ Автоматически | 5 сек |
| `nextjs` | ✅ Автоматически | 5 сек |
| `full` | ✅ Автоматически | 5 сек |
| `api-only` | ❌ Отключен | - |

---

## 🔍 Мониторинг

### Через веб-интерфейс

1. Откройте http://localhost:5000
2. Проверьте раздел "Health" или "System Status"
3. Статус Sync Manager отображается зелёным ✅

### Через API

```bash
# Статус синхронизации
curl http://localhost:8000/api/v1/sync/status

# Health check
curl http://localhost:8000/health
```

### Через логи

Sync Manager логирует события:

```
[SYNC] Инициализирован: Backend=http://localhost:8000, Frontend=http://localhost:5000
[SYNC] Статистика обновлена: 15 полей
[SYNC] Real-time метрики синхронизированы
[SYNC] Backend доступен
[SYNC] Frontend доступен
```

---

## ⚙️ Настройка

### Изменение интервала

В `start_universal.py`:

```python
SYNC_INTERVAL = 5  # секунд
```

### Отключение синхронизации

```bash
# Режим api-only автоматически отключает Sync Manager
python start_universal.py api-only
```

Или в коде:

```python
SYNC_ENABLED_BY_DEFAULT = False  # Отключить по умолчанию
```

---

## 🐛 Troubleshooting

### Sync Manager не запускается

**Проблема:** `[WARN] Sync Manager не найден`

**Решение:**
1. Убедитесь, что `api/sync_manager.py` существует
2. Проверьте права доступа к файлу
3. Установите зависимости: `pip install aiohttp`

### Синхронизация не работает

**Проблема:** Статус показывает `sync_manager: not_available`

**Решение:**
1. Проверьте, запущен ли Sync Manager
2. Убедитесь, что Backend доступен
3. Проверьте логи Sync Manager

### Частые переподключения

**Проблема:** Sync Manager постоянно переподключается

**Решение:**
1. Увеличьте `SYNC_INTERVAL` до 10 секунд
2. Проверьте стабильность сети
3. Убедитесь, что порты не заблокированы

---

## 📊 Производительность

### Нагрузка на систему

| Компонент | CPU | Memory |
|-----------|-----|--------|
| Sync Manager | <1% | ~20 MB |
| Синхронизация (5 сек) | <2% | ~5 MB |
| WebSocket | <1% | ~10 MB |

**Итого:** ~3% CPU, ~35 MB RAM

### Оптимизация

Для production рекомендуется:
- Увеличить интервал до 10-30 секунд
- Использовать Redis для кэширования
- Включить компрессию данных

---

## 📚 Дополнительная документация

- [SYNC.md](docs/SYNC.md) - Полная документация по синхронизации
- [STARTUP_GUIDE.md](docs/STARTUP_GUIDE.md) - Руководство по запуску
- [UNIFIED_DASHBOARD.md](docs/UNIFIED_DASHBOARD.md) - Flask Dashboard

---

## ✅ Чек-лист проверки

- [x] Sync Manager запускается автоматически
- [x] Интервал синхронизации 5 секунд
- [x] Health monitoring работает
- [x] WebSocket синхронизация активна
- [x] Статус отображается в API
- [x] Логи записываются корректно

---

**Синхронизация работает автоматически! 🎉**
