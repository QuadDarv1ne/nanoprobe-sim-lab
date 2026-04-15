# Nanoprobe Sim Lab — TODO

**Последнее обновление:** 2026-04-15
**Ветка:** `dev` (текущая), `main` (стабильная)
**Python:** 3.9 - 3.12 (CI матрица)
**Последний коммит:** fadfbe4
**Всего тестов:** 82 файла

---

## 🎯 Текущие приоритеты

### CRITICAL (исправить в первую очередь)

1. [x] **Убрать legacy `security/auth_manager.py`** (540 строк) — **УДАЛЁН** ✅
   - Flask-based auth, не используется FastAPI
   - Импортирует flask, имеет свою отдельную auth.db
   - Конфликтует с `api/routes/auth_routes/` (JWT-based)
   - **Решение:** удалить файл, обновить импорты если есть

2. [ ] **Исправить bare `except Exception:` без логирования**
   - ~60 в `tests/` (test scaffolding, низкий приоритет)
   - **Исправлено:** database.py:1124, redis_cache.py:304, data_validator.py:148,
     system_health_monitor.py:499, optimization_orchestrator.py:423,
     web_dashboard_unified.py:352/386/403, cli/main.py:202/206/242
   - **Осталось:** ~15 в production коде (низкий приоритет — cleanup при рефакторинге)
   - **Оставшиеся паттерны:** intentional fallbacks (pass после except — допустимо для cleanup)

3. [x] **Включить lint проверки в CI (`|| true` маскирует ошибки)** — **ИСПРАВЛЕНО** ✅
   - Убраны все `|| true` из `ci-cd.yml` и `lint.yml`
   - Добавлен `api/` во все lint пути (black, flake8, mypy)
   - CI теперь будет падать при ошибках линтинга

### HIGH

4. [ ] **Разбить `utils/database.py`** (2241 строка → модули)
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

7. [x] **Удалить `src/web/archived/`** — директория не существует (уже удалено)

8. [x] **Исправить CI: lint не проверяет `api/` директорию** — объединено с задачей #3

9. [x] **Убрать дублирование CI workflows** — **УДАЛЕНО** ✅
   - Удалены `tests.yml` + `lint.yml` (дублируют `ci-cd.yml`)
   - Удалены `build.yml` + `release.yml` + `publish-release.yml` (дублируют `auto-release.yml`)
   - Осталось 7 workflow файлов вместо 12
   - **Остался:** `auto-release.yml` (release + Docker), `deploy.yml`, `release-drafter.yml`

10. [x] **Исправить `.env` — inline комментарии ломают dotenv** — **ИСПРАВЛЕНО** ✅
    - `ADMIN_PASSWORD= # comment` — dotenv читает комментарий как значение
    - Убраны inline комментарии из ADMIN_PASSWORD и USER_PASSWORD
    - Тест `test_login_success` теперь проходит

11. [x] **Исправить `docker-compose.api.yml` — `--reload` в production** — **ИСПРАВЛЕНО** ✅
    - Убран `--reload` из uvicorn command

12. [x] **Убрать `|| true` из всех CI workflows** — **ИСПРАВЛЕНО** ✅
    - Убраны из `security.yml`, `benchmark.yml`, `docs-generator.yml`
    - CI теперь будет падать при реальных проблемах

### LOW

13. [ ] Оптимизировать время тестов (>3min для 1227 тестов)
    - Добавить pytest markers (slow/fast)
    - Настроить pytest-xdist для параллельного запуска
14. [ ] Решить SQLite vs PostgreSQL (есть guide в docs/)
15. [ ] Мигрировать frontend на Next.js (убрать Flask legacy)
16. [ ] Откалибровать TCXO (--freq-correction для RTL-SDR)
17. [ ] Исправить E501 строки (HTML/CSS/SQL/config)

---

## 📊 Статистика проекта (актуально на 2026-04-14)

### Код
- **API роуты:** 41 файл в `api/routes/` (26 top-level + subdirs)
- **Utils:** 72 файла в `utils/` (15 поддиректорий)
- **Тесты:** 82 test файла
- **Файлов >500 строк:** 30 (самый большой: database.py 2241)

### Качество кода
- **flake8:** 0 критических ошибок (F/E9) ✅
- **bare except Exception:** ~15 осталось в production ⚠️ (исправлено 10+)
- **print() в utils/:** 946 вызовов ⚠️
- **print() всего в проекте:** 3622 вызовов ⚠️
- **Pre-commit hooks:** black, isort, flake8 ✅
- **CI lint:** исправлен ✅ (убраны `|| true`, добавлен `api/`)
- **UTF-8 BOM:** 10 файлов в `api/routes/` начинаются с BOM ⚠️
- **CI дубликаты:** удалены 5 дублирующих workflow ✅ (осталось 7)
- **Dead code:** удалён `src/web/archived/` (133KB unused files) ✅
- **docker-compose:** убран `--reload` из production ✅

### Архитектура
- **Backend:** FastAPI + JWT + 2FA TOTP + WebSocket + GraphQL
- **Frontend:** Next.js v2.0 (production) + Flask v1.0 (legacy)
- **Database:** SQLAlchemy + Alembic (SQLite, есть PostgreSQL guide)
- **Cache:** Redis integration
- **CI/CD:** 7 GitHub Actions workflows (без дубликатов)

### RTL-SDR V4
- ✅ FM-радиовещание, ADS-B, NOAA, SSTV, RTL_433, POCSAG
- ✅ Real-time visualizer (spectrum + waterfall)
- ✅ ISS трекинг (SGP4 + CelesTrak)
- ⏳ RTL-SDR V4 hardware не получен (ожидается)

---

## 🔍 Детальные пометки по проблемам

### 1. Legacy Auth System — УДАЛЁН ✅
**Файл:** `security/auth_manager.py` (540 строк) — удалён 2026-04-15

### 2. Silent Exception Swallowing
**Критичные файлы (16 total):**

**utils/ (9 шт):**
- `utils/database.py:60,129,315,1123` — core database operations
- `utils/caching/cache_manager.py:104` — cache operations
- `utils/caching/redis_cache.py:304` — redis cache
- `utils/data/data_validator.py:148` — data validation
- `utils/monitoring/system_health_monitor.py:499` — health checks
- `utils/performance/optimization_orchestrator.py:423` — optimization

**api/ (7 шт):**
- `api/main.py:621` — FastAPI app startup
- `api/sstv/session_manager.py:43` — session management
- `api/sstv/rtl_sstv_receiver.py:496` — RTL-SDR receiver
- `api/routes/sstv_advanced.py:196,280,343` — SSTV routes
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
**Файлы:** `.github/workflows/ci-cd.yml`, `lint.yml`

**Проблемы:**
```yaml
# ci-cd.yml line 67
- run: black --check --diff src/ utils/ tests/ 2>&1 || true  # BAD

# ci-cd.yml line 75
- run: mypy src/ utils/ --ignore-missing-imports || true  # BAD

# lint.yml line 36
- run: mypy src/ utils/ --ignore-missing-imports || true  # BAD

# lint.yml line 45
- run: mypy --strict ... || true  # BAD
```

- Lint проверяет только `src/ utils/ tests/`, но не `api/` (основной код)
- **Действие:** убрать все `|| true`, добавить `api/` в lint paths

### 4. utils/database.py (2241 строка)
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
- [x] Удалены дублирующие CI workflows (tests.yml, lint.yml)
- [x] Исправлен .env — inline комментарии ломали dotenv
- [x] Исправлен data/.admin_password — trailing spaces ломали auth тест
- [x] Добавлено логирование в bare except (10+ файлов)
- [x] Убраны `|| true` из security.yml, benchmark.yml, docs-generator.yml
- [x] Удалены дублирующие release workflows (build.yml, release.yml, publish-release.yml)
- [x] Удалён dead code `src/web/archived/` (133KB)
- [x] Убран `--reload` из docker-compose.api.yml production

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
