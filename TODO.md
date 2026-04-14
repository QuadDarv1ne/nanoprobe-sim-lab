# Nanoprobe Sim Lab — TODO

**Последнее обновление:** 2026-04-14
**Ветка:** `dev` (текущая), `main` (стабильная) — **синхронизированы** ✅
**Python:** 3.11 - 3.14
**Последний коммит:** a3c1e2b
**Всего тестов:** 1227 collected

---

## 🎯 Текущие приоритеты

### HIGH
1. [x] Работать в ветке `dev`, проверять тесты, merge в `main`
2. [ ] Увеличить test coverage до 80%+ (сейчас ~20%)
   - 1227 тестов собрано, но покрытие низкое
   - Нужно добавить integration и unit тесты для core бизнес-логики
3. [ ] Убрать дублирование паттернов в тестах (DRY)
   - Много похожих тестов в test_api*.py файлах
   - Создать общие фикстуры и хелперы

### MEDIUM
4. [ ] Решить SQLite vs PostgreSQL (есть guide в docs/)
   - В docker-compose.prod.yml есть PostgreSQL конфигурация
   - API по умолчанию использует SQLite
5. [ ] Мигрировать frontend на Next.js (убрать Flask legacy)
   - Flask dashboard v1.0 всё ещё активен
   - Next.js v2.0 готов для production
6. [ ] Откалибровать TCXO (--freq-correction для RTL-SDR)
7. [ ] Проверить bias_tee для активной антенны
8. [ ] Оптимизировать время запуска тестов (сейчас >3min timeout)
   - 1227 тестов выполняются очень долго
   - Добавить маркировку slow/fast тестов
   - Настроить pytest-xdist для параллельного запуска

### LOW
9. [ ] Исправить E501 строки (HTML/CSS/SQL/config — low priority)
10. [ ] Создать mobile application
11. [ ] Разбить utils/database.py на модули (2242 строки)
12. [ ] Убрать print() statements (~3600+ в production code)

---

## 📊 Статистика проекта

### Код (актуально на 2026-04-14)
- **API роуты:** 24+ endpoint'ов в `api/routes/`
- **API модули:** 26 файлов в `api/` (routes, security, sstv, websocket, graphql)
- **Utils:** 36+ модулей в `utils/` (15 поддиректорий, 22 файла верхнего уровня)
- **Тесты:** 1227 collected, ~20% coverage
- **flake8:** 0 критических ошибок (F/E9) ✅
- **Пре-коммит хуки:** black, isort, flake8 ✅
- **Коммитов сегодня:** 0 (последний: a3c1e2b)

### Архитектура
- **Backend:** FastAPI + JWT + 2FA TOTP + WebSocket + GraphQL
- **Frontend:** Next.js v2.0 (TypeScript, Tailwind, PWA) + Flask v1.0 (legacy)
- **Database:** SQLAlchemy + Alembic (SQLite, есть PostgreSQL guide)
- **Cache:** Redis integration
- **Monitoring:** Prometheus + Grafana
- **CI/CD:** 12 GitHub Actions workflows

### RTL-SDR V4
- ✅ FM-радиовещание, ADS-B, NOAA, SSTV, RTL_433, POCSAG
- ✅ Real-time visualizer (spectrum + waterfall)
- ✅ ISS трекинг (SGP4 + CelesTrak)
- ⏳ Любительские радиостанции 2м/70см, AIS, HF через апконвертер

---

## 📝 Пометки по архитектуре

### Положительные моменты
- ✅ 0 критических flake8 ошибок (F/B/E9)
- ✅ Pre-commit hooks настроены и работают
- ✅ Argon2 password hashing (мигрировано с bcrypt/passlib)
- ✅ Модульная структура API и utils
- ✅ Хорошая документация
- ✅ 1227 тестов собрано
- ✅ GraphQL API добавлен (10 тестов)
- ✅ RTL-433 API добавлен (15 тестов)

### Проблемные места
- ⚠️ **Coverage ~20%** — цель 80%+, осталось много работы
  - 1227 тестов есть, но покрытие низкое (много mock/stub тестов)
  - Нужно добавить реальных integration тестов
- ⚠️ **utils/database.py** — 2242 строки, нужно разбить
- ⚠️ **print() statements** — ~3600+ в production code (низкий приоритет)
- ⚠️ **global statement** — 37 использований (оправдано для singleton, но следить)
- ⚠️ **except Exception:** — 127 bare except (api/main.py, sstv routes и др.)
- ⚠️ **SQLite vs PostgreSQL** — API использует SQLite, docker-compose.prod.yml имеет PostgreSQL
- ⚠️ **legacy code** — `security/auth_manager.py` (Flask, не используется FastAPI)
- ⚠️ **E501** — ~197 длинных строк (HTML/CSS/SQL/config — low priority)
- ⚠️ **pysstv** — только encoder, нет декодера SSTV
- ⚠️ **Время тестов** — >180 секунд timeout, 1227 тестов выполняются очень долго
  - Нужно: pytest-xdist, маркировка slow/fast, параллелизация

### Технические долги
- 🔧 **Тесты дублируются** — много test_api*.py с похожей структурой
  - Решение: создать общие фикстуры, параметризированные тесты
- 🔧 **Нет CI pipeline для длительных тестов**
  - Быстрые тесты (<30s) должны跑 в CI
  - Медленные тесты — отдельно, nightly
- 🔧 **utils/database.py 2242 строки** — слишком большой файл
  - Разбить на: models.py, connections.py, migrations.py, queries.py

### Deprecated (архивированы)
- FM Radio: 5 файлов → `fm_radio_unified.py`
- Scripts: 4 файла → `scripts/project.py`
- Utils: 7 файлов → `utils/archived/`

---

## 📊 Проделанная работа (2026-04-14 Afternoon - GraphQL API + RTL-433 API)

### Test Coverage - GraphQL API
✅ Добавлено 10 тестов:
  - `test_graphql_api.py`: 10 тестов (все прошли)
    - GET /graphql/schema — получение схемы (4 теста)
    - POST /graphql — endpoint (2 теста)
    - Validation (4 теста)

✅ Все 10 тестов прошли успешно (0 failed)
✅ Commit: 8a12326
✅ Push в origin/dev

---

## 📊 Проделанная работа (2026-04-14 Afternoon - RTL-433 API)

### Test Coverage - RTL-433 API
✅ Добавлено 15 тестов:
  - `test_rtl433_api.py`: 15 тестов (все прошли)
    - GET /readings — список показаний (5 тестов)
    - GET /devices — список устройств (2 теста)
    - GET /stats — статистика (1 тест)
    - POST /clear — очистка данных (1 тест)
    - Model validation (6 тестов)

✅ Все 15 тестов прошли успешно (0 failed)
✅ Commit: b5f353b
✅ Push в origin/dev

---

## 📊 Проделанная работа (2026-04-14 Morning - Weather API)

### Test Coverage - Weather API
✅ Добавлено 12 тестов:
  - `test_weather_api.py`: 12 тестов (все прошли)
    - GET /weather/{location} — прогноз погоды (7 тестов)
    - Validation (5 тестов)

✅ Все 12 тестов прошли успешно (0 failed)
✅ Commit: 27e3d98
✅ Push в origin/dev

---

## 📊 Проделанная работа (2026-04-14 Morning - FM Radio API)

### Test Coverage - FM Radio API
✅ Добавлено 12 тестов:
  - `test_fm_radio_api.py`: 12 тестов (все прошли)

✅ Все 12 тестов прошли успешно (0 failed)
✅ Commit: 735fa50
✅ Push в origin/dev

---

## 📊 Проделанная работа (2026-04-14 Morning - Alerting API)

### Test Coverage - Alerting API
✅ Добавлено 17 тестов:
  - `test_alerting_api.py`: 17 тестов (все прошли)

✅ Все 17 тестов прошли успешно (0 failed)
✅ Commit: aa610f2
✅ Push в origin/dev

---

## 📝 Правила работы с проектом

1. **НЕ создавать документацию без запроса** — только код и исправления
2. **Качество важнее количества** — лучше меньше, но лучше
3. **Работать в dev**, потом проверить и отправить в main
4. **Синхронизировать изменения** — не забывать push и merge

---

## 🔗 Ресурсы

- **RTL-SDR Blog:** https://www.rtl-sdr.com/
- **Celestrak TLE:** https://celestrak.org/
- **Satnobs:** https://satnobs.io/
- **ISS SSTV:** https://www.ariss.org/
