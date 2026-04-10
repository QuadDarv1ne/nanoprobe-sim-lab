# Architecture Decision Records (ADR)

Документация архитектурных решений проекта Nanoprobe Sim Lab.

## Индекс

| # | Решение | Статус | Дата |
|---|---------|--------|------|
| 001 | FastAPI вместо Flask для Backend | Принято | 2026-03-01 |
| 002 | SQLite для разработки, PostgreSQL для прода | Принято | 2026-03-05 |
| 003 | RTL-SDR интеграция: процесс-ориентированный подход | Принято | 2026-04-07 |
| 004 | Redis кэширование с in-memory fallback | Принято | 2026-03-15 |
| 005 | datetime.now(timezone.utc) везде | Принято | 2026-04-10 |

---

## ADR-001: FastAPI вместо Flask для Backend

**Статус:** Принято ✅
**Дата:** 2026-03-01
**Контекст:** Миграция с Flask на FastAPI для Backend API

### Проблема

Flask используется для Backend API, но:
- Нет встроенной асинхронности
- Нет автоматической генерации OpenAPI схем
- Сложнее масштабировать WebSocket подключения
- Нет встроенной валидации данных (Pydantic)

### Решение

Мигрировать Backend на FastAPI, оставив Flask только для Frontend Dashboard.

### Аргументы "За"

1. **Асинхронность**: Нативная поддержка async/await
2. **Type Safety**: Pydantic для валидации
3. **Auto Docs**: Swagger/ReDoc из коробки
4. **Производительность**: 2-5x быстрее Flask
5. **WebSocket**: Нативная поддержка для real-time данных

### Аргументы "Против"

1. Миграция потребует времени
2. Нужно переписать middleware
3. Команда знакома с Flask

### Компромиссы

- Flask остаётся для Frontend (обратная совместимость)
- Миграция поэтапная (роут за роутом)
- Оба фреймворка используют один WSGI/ASGI сервер

### Результат

✅ Backend полностью мигрирован на FastAPI
✅ Frontend остаётся на Flask + Socket.IO
✅ Синхронизация через Sync Manager

### Ссылки
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Flask vs FastAPI Benchmark](docs/36-performance-optimization.md)

---

## ADR-002: SQLite для разработки, PostgreSQL для прода

**Статус:** Принято ✅
**Дата:** 2026-03-05
**Контекст:** Выбор стратегии управления базами данных

### Проблема

Нужно поддерживать:
- Лёгкую разработку без внешних зависимостей
- Продакшен с высокой производительностью
- Миграции схемы без потери данных

### Решение

Использовать **SQLite для разработки** и **PostgreSQL для продакшена**.

### Аргументы "За"

1. **Разработка:**
   - Zero config — файл `.db` готов сразу
   - Не нужен Docker/установка БД
   - Быстрый старт для новых разработчиков

2. **Продакшен:**
   - Connection pooling (asyncpg)
   - Лучше конкурентная обработка
   - Point-in-time recovery
   - Репликация для HA

3. **Миграции:**
   - Alembic работает с обеими СУБД
   - Скрипт автоматической миграции

### Аргументы "Против"

1. Разное поведение в dev/prod
2. Нужно тестировать на обеих СУБД
3. Миграции требуют осторожности

### Компромиссы

- SQLAlchemy ORM для абстракции
- CI тестирует на SQLite
- Отдельный staging с PostgreSQL
- Скрипт `migrate_sqlite_to_postgres.py`

### Миграция

```bash
# Dry run
python scripts/migrate_sqlite_to_postgres.py \
  --dry-run \
  --pg-url postgresql://user:pass@localhost:5432/nanoprobe

# Выполнение
python scripts/migrate_sqlite_to_postgres.py \
  --execute \
  --pg-url postgresql://user:pass@localhost:5432/nanoprobe
```

### Ссылки
- [Скрипт миграции](../scripts/migrate_sqlite_to_postgres.py)
- [Alembic конфигурация](../alembic.ini)

---

## ADR-003: RTL-SDR интеграция — процесс-ориентированный подход

**Статус:** Принято ✅
**Дата:** 2026-04-07
**Контекст:** Интеграция RTL-SDR V4 устройства

### Проблема

RTL-SDR требует:
- Прямого доступа к USB устройству
- Реального времени обработки I/Q потока
- Graceful degradation при отключении
- Ограничения памяти при длительной записи

### Решение

Использовать **процесс-ориентированный подход**:
- RTL-SDR утилиты (rtl_fm, rtl_sdr) как subprocess
- Python обёртка для управления
- Circular buffers для waterfall
- Потоковая запись через ffmpeg pipe

### Аргументы "За"

1. **Надёжность:** Процесс можно перезапустить при сбое
2. **Изоляция:** Ошибки RTL-SDR не влияют на основной процесс
3. **Производительность:** Нативные утилиты оптимизированы
4. **Гибкость:** Легко заменить на HackRF/Airspy

### Аргументы "Против"

1. Overhead на IPC
2. Сложнее отладка
3. Зависимость от внешних утилит

### Компромиссы

- pyrtlsdr для простого контроля
- rtl_sdr для записи больших объёмов
- Graceful degradation при недоступности

### Архитектура

```
[Python API] ←subprocess→ [rtl_fm/rtl_sdr] ←USB→ [RTL-SDR V4]
     ↓
[Circular Buffer] → [FFT] → [Waterfall Display]
     ↓
[ffmpeg pipe] → [Video File]
```

### Ссылки
- [RTL-SDR документация](03-rtl-sdr-sstv-recording.md)
- [Waterfall оптимизация](../components/py-sstv-groundstation/src/waterfall_display_optimized.py)

---

## ADR-004: Redis кэширование с in-memory fallback

**Статус:** Принято ✅
**Дата:** 2026-03-15
**Контекст:** Кэширование API ответов

### Проблема

Нужно:
- Ускорить ответы API (stats, metrics)
- Работать без Redis в разработке
- Graceful degradation

### Решение

Использовать **Redis для продакшена** с **in-memory fallback** для разработки.

### Реализация

```python
# api/state.py
def get_redis():
    if os.getenv("REDIS_DISABLED", "false").lower() == "true":
        return InMemoryCache()
    return RedisCache()
```

### TTL настройки

| Тип данных | TTL | Пример |
|------------|-----|--------|
| Статистика дашборда | 5с | CPU, RAM, disk |
| Метрики реального времени | 1с | Sensor data |
| TLE данные | 1 час | Satellite positions |
| SSTV recordings | 24ч | File listings |

### Ссылки
- [Redis документация](../utils/caching/)
- [Sync Manager](../api/sync_manager.py)

---

## ADR-005: datetime.now(timezone.utc) везде

**Статус:** Принято ✅
**Дата:** 2026-04-10
**Контекст:** Унификация обработки дат

### Проблема

В проекте использовались:
- `datetime.now()` (naive, без timezone)
- `datetime.utcnow()` (deprecated в Python 3.12+)
- Смешение timezone в разных модулях

### Решение

Заменить **все** использования на `datetime.now(timezone.utc)`.

### Аргументы "За"

1. **Консистентность**: Все даты в UTC
2. **Future-proof**: Совместимо с Python 3.12+
3. **Научная точность**: Важно для SSTV/SDR данных
4. **ISO 8601**: Легко парсить и сортировать

### Миграция

```bash
# Автоматическая миграия
python migrate_datetime.py --execute

# Результат:
# - 542 замены в 116 файлах
# - 0 ошибок
```

### Примеры

```python
# ✅ Правильно
from datetime import datetime, timezone

timestamp = datetime.now(timezone.utc)
# 2026-04-10T12:34:56.789012+00:00

# ❌ Неправильно
timestamp = datetime.now()  # Naive
timestamp = datetime.utcnow()  # Deprecated
```

### Ссылки
- [Скрипт миграции](../migrate_datetime.py)
- [PEP 615](https://peps.python.org/pep-0615/)
