# Nanoprobe Sim Lab — TODO

**Последнее обновление:** 2026-04-16
**Ветка:** `dev` (текущая), `main` (стабильная)
**Python:** 3.11 - 3.14 (CI матрица)
**Последний коммит:** 5793512
**Всего тестов:** 1236 тестов (100% pass)

---

## 🎯 Текущие приоритеты

### CRITICAL (исправить в первую очередь)

1. [x] **Убрать legacy `security/auth_manager.py`** (540 строк) — **УДАЛЁН** ✅
   - Flask-based auth, не используется FastAPI
   - Импортирует flask, имеет свою отдельную auth.db
   - Конфликтует с `api/routes/auth_routes/` (JWT-based)

2. [x] **Исправить bare `except Exception:` без логирования** — **ИСПРАВЛЕНО** ✅
   - ~60 в `tests/` (test scaffolding, низкий приоритет, оставлено)
   - ~22 в CLI/tools (rtl_sdr_tools/, components/) — низкий приоритет, оставлено
   - **Исправлены все 16 в production коде**

3. [x] **Включить lint проверки в CI (`|| true` маскирует ошибки)** — **ИСПРАВЛЕНО** ✅
   - Убраны все `|| true` из `ci-cd.yml` и `lint.yml`
   - Добавлен `api/` во все lint пути (black, flake8, mypy)

4. [x] **Коммит всех staged изменений** — **ЗАВЕРШЁНО** ✅
   - 16 новых файлов в utils/sdr/, utils/ml/, utils/i18n/
   - Обновлённые API routes для SSTV
   - Интеграционные тесты для RTL-SDR
   - **Последний коммит:** 5793512

### HIGH

5. [x] **Разбить `utils/database.py`** (2241 строка → модули) — **РЕФАКТОРИНГ ЗАВЕРШЁН** ✅
   - `utils/db/connection.py` — ConnectionPool, AsyncConnectionPool
   - `utils/db/schema.py` — init_database_schema, get_database_stats
   - `utils/db/operations.py` — все CRUD операции, кэширование, пользователи
   - `utils/db/__init__.py` — DatabaseManager (объединяет всё)
   - `utils/database.py` — ре-экспорт для обратной совместимости

6. [x] **Ring Buffer в Shared Memory** — **РЕАЛИЗОВАНО** ✅
   - `utils/sdr/ring_buffer.py` — кроссплатформенный ring buffer
   - POSIX SHM на Linux, mmap на Windows
   - Ёмкость: 2M complex64 сэмплов (~16MB), потокобезопасный

7. [x] **SDR Resource Manager** — **РЕАЛИЗОВАНО** ✅
   - `utils/sdr/sdr_resource_manager.py` — приоритизация задач
   - Приоритеты: ISS пролет (100) > метеоспутники (80) > SSTV (60) > сканирование (20)
   - Вытеснение низкоприоритетных задач высокоприоритетными
   - Singleton паттерн, потокобезопасный

8. [x] **Hardware health-чек для RTL-SDR v4** — **РЕАЛИЗОВАНО** ✅
   - `utils/sdr/hardware_health.py` — диагностика оборудования
   - check_temperature() — через rtl_test -t или sysfs fallback
   - check_eeprom() — проверка EEPROM через rtl_eeprom
   - check_dropped_samples() — детекция потерь сэмплов
   - run_full_diagnostic() — полная диагностика

9. [x] **Trigger Recorder** — **РЕАЛИЗОВАНО** ✅
   - `utils/sdr/trigger_recorder.py` — триггерная запись
   - Pre-trigger buffer: 2 секунды до срабатывания
   - Триггеры: squelch (dBFS), VIS-код SSTV, manual
   - API endpoints: /trigger/start, /trigger/stop, /trigger/status

10. [ ] **Автоматическая коррекция PPM для RTL-SDR v4**
    - Реализовать `_calculate_ppm_from_signal()` в rtl_sdr_calibration.py
    - Метод: `rtl_test -p` для оценки PPM по известной частоте
    - Сохранение в `config/device_calibration.json`
    - API: `POST /api/v1/sstv/calibration/automated`

11. [ ] **Заменить print() на logging в `utils/`** (~900 вызовов)
    - **Исправлено:** 64 print() → logger (19 + 37 + 8)
    - **Осталось:** performance_*.py, caching/*.py, data/*.py и др.

12. [ ] **Увеличить test coverage до 80%+**
    - 1236 тестов есть, но покрытие ~20%
    - Фокус: api/routes/, utils/db/, core modules

### MEDIUM

13. [x] **TensorFlow Lite классификация сигналов** — **РЕАЛИЗОВАНО** ✅
    - `utils/ml/signal_classifier.py` — обёртка TFLite с graceful degradation
    - Классы: sstv, cw, rtty, fm, noise, unknown
    - Fallback: heuristic-классификация по спектральным характеристикам

14. [x] **REST endpoint `/api/v1/sstv/iq/raw`** — **РЕАЛИЗОВАНО** ✅
    - Форматы: CSV, binary (.bin), JSON
    - Параметры: count (1-65536), offset, format

15. [x] **WebSocket метрика dBFS** — **РЕАЛИЗОВАНО** ✅
    - `api/routes/sstv_advanced.py` — strength_dbfs и strength_percent в WS
    - Helper: `_strength_to_percent(dbfs)` для UI

16. [x] **Предиктивное кэширование TLE** — **РЕАЛИЗОВАНО** ✅
    - `api/routes/sstv/satellites.py` — проверка возраста TLE, авто-refresh
    - Интервал: 3 дня (Celestrak API)
    - Background task: _auto_refresh_tle_background()

17. [ ] **Автоматический захват NOAA APT / Meteor LRPT**
    - `utils/sdr/noaa_capture.py` — NOAA APT capture + decode
    - `utils/sdr/meteor_capture.py` — Meteor LRPT capture + decode
    - Фоновый планировщик: предсказание → автозапись → декодирование

18. [x] **Удалить `src/web/archived/`** — уже удалено ✅

19. [x] **Исправить `.env` — inline комментарии** — **ИСПРАВЛЕНО** ✅

20. [x] **Исправить `docker-compose.api.yml` — `--reload` в production** — **ИСПРАВЛЕНО** ✅

21. [x] **Убрать `|| true` из всех CI workflows** — **ИСПРАВЛЕНО** ✅

### LOW

22. [ ] **WebGL/Canvas водопад спектра на Next.js**
    - `frontend/src/components/sstv/WaterfallDisplay.tsx`

23. [x] **Bash-скрипт установки RTL-SDR v4** — **РЕАЛИЗОВАНО** ✅
    - `scripts/setup_rtlsdr_v4.sh` — автоопределение ОС, blacklist DVB-T, udev правила

24. [x] **Интеграционный тест `test_rtlsdr_roundtrip.py`** — **РЕАЛИЗОВАНО** ✅
    - Ring buffer write/read tests
    - PPM calibration file tests
    - Resource manager priority tests
    - Trigger recording tests

25. [x] **Локализация ошибок RTL-SDR** — **РЕАЛИЗОВАНО** ✅
    - `utils/i18n/sdr_errors.py` — русские сообщения об ошибках (RU/EN маппинг)

26. [ ] **README: Troubleshooting RTL-SDR v4**
    - DVB-T blacklist, udev правила, PPM drift, перегрев

---

## 📊 Статистика проекта

### Код

- **API роуты:** 41 файл в `api/routes/`
- **Utils:** 72 файла в `utils/`
- **Тесты:** 1236 тестов (82 test файла)
- **Новых файлов:** 16 (SDR utilities, ML classifier, i18n)

### Качество кода

- **flake8:** 0 критических ошибок (F/E9) ✅
- **bare except Exception:** 0 в production ✅
- **print() в utils/:** ~900 вызовов ⚠️
- **CI lint:** исправлен ✅
- **Test coverage:** ~20% (1236 тестов)

### Архитектура

- **Backend:** FastAPI + JWT + 2FA TOTP + WebSocket + GraphQL
- **Frontend:** Next.js v2.0 (production) + Flask v1.0 (legacy)
- **Database:** SQLAlchemy + Alembic (SQLite)
- **Cache:** Redis integration
- **SDR:** RTL-SDR v4 support with ring buffer, resource management

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
