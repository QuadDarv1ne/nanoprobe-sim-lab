# Nanoprobe Sim Lab — TODO

**Последнее обновление:** 2026-04-14
**Ветка:** `dev` (текущая), `main` (стабильная) — **синхронизированы** ✅
**Python:** 3.11 - 3.14
**Последний коммит:** 7bbf5a7
**Всего тестов:** 1227 collected, ~20% coverage

---

## 🎯 Текущие приоритеты

### CRITICAL (исправить в первую очередь)

1. [ ] **Убрать legacy `security/auth_manager.py`** (540 строк)
   - Flask-based auth, не используется FastAPI
   - Импортирует flask, имеет свою отдельную auth.db
   - Конфликтует с `api/routes/auth_routes/` (JWT-based)
   - **Решение:** удалить файл, обновить импорты если есть

2. [ ] **Исправить 118 bare `except Exception:` без логирования**
   - `utils/database.py` — 4 шт (lines 60, 129, 315, 1122)
   - `api/routes/sstv_advanced.py` — 3 шт
   - `api/main.py` — 1 шт (line 621)
   - `utils/caching/`, `utils/data/` — по 1 шт
   - **Решение:** заменить на `except Exception as e: logger.exception("...")`

3. [ ] **Включить lint проверки в CI (`|| true` маскирует ошибки)**
   - `.github/workflows/ci-cd.yml` — Black и MyPy используют `|| true`
   - `.github/workflows/lint.yml` — MyPy использует `|| true`
   - CI всегда проходит даже при сломанном коде
   - **Решение:** убрать `|| true`, починить ошибки

### HIGH

4. [ ] **Разбить `utils/database.py`** (1947 строк → модули)
   - models.py — SQLAlchemy модели
   - connection.py — connection pool, engine init
   - operations.py — CRUD операции
   - users.py — user management
   - sessions.py — session management
   - **Цель:** файлы <400 строк каждый

5. [ ] **Заменить print() на logging в `utils/`** (940 вызовов)
   - `utils/monitoring/` — много print вместо логов
   - `utils/backup_manager.py`, `config_validator.py`
   - **Приоритет:** сначала core модули, потом остальные
   - **Исключение:** CLI инструменты (components/)

6. [ ] **Увеличить test coverage до 80%+**
   - 1227 тестов есть, но покрытие ~20% (много mock/stub)
   - Нужны реальные integration тесты для бизнес-логики
   - **Фокус:** api/routes/, utils/database.py, core modules

### MEDIUM

7. [ ] **Удалить `src/web/archived/`** (2506 строк мёртвого кода)
   - `web_dashboard.py` (1422 строки)
   - `web_dashboard_integrated.py` (1084 строки)
   - Блоатит репозиторий, замедляет CI

8. [ ] **Исправить CI: lint не проверяет `api/` директорию**
   - `.github/workflows/lint.yml` проверяет только `src/`
   - Основной код (FastAPI routes) не линтится в CI
   - **Решение:** добавить `api/` в flake8/black/mypy пути

9. [ ] **Убрать дублирование test jobs в CI**
   - `ci-cd.yml` и `tests.yml` оба запускают тесты
   - **Решение:** оставить один, другой удалить или изменить

10. [ ] **Исправить `docker-compose.api.yml` — `--reload` в production**
    - uvicorn reload mode не должен быть в production config
    - **Решение:** вынести в docker-compose.dev.yml

### LOW

11. [ ] Оптимизировать время тестов (>3min для 1227 тестов)
    - Добавить pytest markers (slow/fast)
    - Настроить pytest-xdist для параллельного запуска
12. [ ] Решить SQLite vs PostgreSQL (есть guide в docs/)
13. [ ] Мигрировать frontend на Next.js (убрать Flask legacy)
14. [ ] Откалибровать TCXO (--freq-correction для RTL-SDR)
15. [ ] Исправить E501 строки (HTML/CSS/SQL/config)

---

## 📊 Статистика проекта (актуально на 2026-04-14)

### Код
- **API роуты:** 41 файл в `api/routes/` (26 top-level + subdirs)
- **Utils:** 72 файла в `utils/` (15 поддиректорий)
- **Тесты:** 82 test файла, 1227 collected, ~20% coverage
- **Файлов >500 строк:** 30 (самый большой: database.py 1947)

### Качество кода
- **flake8:** 0 критических ошибок (F/E9) ✅
- **bare except Exception:** 118 occurrences ⚠️
- **print() в utils/:** 940 вызовов ⚠️
- **print() всего в проекте:** 3614 вызовов ⚠️
- **Pre-commit hooks:** black, isort, flake8 ✅
- **CI lint:** сломан (`|| true` маскирует ошибки) ⚠️

### Архитектура
- **Backend:** FastAPI + JWT + 2FA TOTP + WebSocket + GraphQL
- **Frontend:** Next.js v2.0 (production) + Flask v1.0 (legacy)
- **Database:** SQLAlchemy + Alembic (SQLite, есть PostgreSQL guide)
- **Cache:** Redis integration
- **CI/CD:** 12 GitHub Actions workflows (2 дублируются)

### RTL-SDR V4
- ✅ FM-радиовещание, ADS-B, NOAA, SSTV, RTL_433, POCSAG
- ✅ Real-time visualizer (spectrum + waterfall)
- ✅ ISS трекинг (SGP4 + CelesTrak)
- ⏳ RTL-SDR V4 hardware не получен (ожидается)

---

## 🔍 Детальные пометки по проблемам

### 1. Legacy Auth System
**Файл:** `security/auth_manager.py` (540 строк)
**Проблема:** Flask-based auth, полностью отдельная система
- Использует `from flask import g, jsonify, request`
- Имеет свою `auth.db` SQLite базу
- Реализует `@require_auth` Flask декоратор
- **Не импортируется** ни одним FastAPI роутом
- **Действие:** Удалить файл, проверить что ничего не ломается

### 2. Silent Exception Swallowing
**Критичные файлы:**
- `utils/database.py:60,129,315,1122` — core database operations
- `api/routes/sstv_advanced.py:196,280,343` — SSTV routes
- `api/main.py:621` — FastAPI app startup
- `api/routes/sstv/health.py:271` — health check

**Паттерн:**
```python
except Exception:  # BAD — ошибка молча игнорируется
    pass
```

**Решение:**
```python
except Exception as e:
    logger.exception("Failed to ...")
    raise  # или return error response
```

### 3. CI/CD Problems
**Файлы:** `.github/workflows/ci-cd.yml`, `lint.yml`, `tests.yml`

**Проблемы:**
```yaml
# ci-cd.yml — lint step
- run: black . || true  # BAD — ошибки игнорируются
- run: mypy . || true   # BAD — type check игнорируется
```

- Lint проверяет только `src/`, но не `api/` (основной код)
- `tests.yml` дублирует test job из `ci-cd.yml`
- **Действие:** убрать `|| true`, добавить `api/` в lint paths

### 4. utils/database.py (1947 строк)
**Что содержит:**
- ConnectionPool class
- Async wrappers
- User management (create, get, update, delete, verify)
- Scan storage
- Session management
- Multiple query methods
- Migration helpers

**План разбиения:**
```
utils/database.py (1947) →
  utils/db/connection.py (~150) — engine, pool, init
  utils/db/models.py (~400) — SQLAlchemy models
  utils/db/users.py (~300) — user CRUD
  utils/db/scans.py (~300) — scan CRUD
  utils/db/sessions.py (~200) — session management
  utils/db/queries.py (~400) — generic query helpers
  utils/db/__init__.py — exports
```

### 5. print() → logging
**Приоритетные файлы:**
1. `utils/monitoring/*.py` — production monitoring code
2. `utils/backup_manager.py` — production backup
3. `utils/config_validator.py` — production config
4. `utils/data/data_validator.py` — production validation

**Исключения (оставить print):**
- `components/py-sstv-groundstation/` — CLI инструменты
- `scripts/` — скрипты
- `rtl_sdr_tools/` — CLI tools

---

## ✅ Реализовано (recent)

- [x] Argon2 password hashing (мигрировано с bcrypt/passlib)
- [x] GraphQL API (10 тестов)
- [x] RTL-433 API (15 тестов)
- [x] Weather API (12 тестов)
- [x] FM Radio API (12 тестов)
- [x] Alerting API (17 тестов)
- [x] Структурированный TODO.md с реальными проблемами

---

## 📝 Правила работы с проектом

1. **НЕ создавать документацию без запроса** — только код и исправления
2. **Качество важнее количества** — лучше меньше, но лучше
3. **Работать в dev**, потом проверить и отправить в main
4. **Синхронизировать изменения** — не забывать push и merge
5. **Исправлять реальные проблемы**, не косметические

---

## 🔗 Ресурсы

- **RTL-SDR Blog:** https://www.rtl-sdr.com/
- **Celestrak TLE:** https://celestrak.org/
- **Satnobs:** https://satnobs.io/
- **ISS SSTV:** https://www.ariss.org/
