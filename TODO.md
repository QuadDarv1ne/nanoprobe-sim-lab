# Nanoprobe Sim Lab — TODO

**Последнее обновление:** 2026-04-09 22:05

## Последние улучшения (2026-04-09)

- ✅ RTL-SDR V4: исправлены импорты pysstv, создана документация и тесты
- ✅ SSTV UI: подключены кнопки Eye/Download/Delete
- ✅ SSTV API: добавлены эндпоинты GET/DELETE /recordings/{filename}
- ✅ Тесты: исправлен test_login_success (чтение пароля из файла/ENV)
- ✅ Тесты: исправлены 24 теста auth.py (JWT, refresh tokens)
- ✅ Тесты: исправлены 15 тестов api.py (инициализация БД)
- ✅ Настройка окружения: создан .env для разработки
- ✅ **created_at FIX**: исправлены все 10 INSERT методов в database.py (14/14 тестов passing)
- ✅ **SSTV Receiver FIX**: исправлены 5 критических проблем (pysstv API, кэширование инициализации)

---

## Known Issues

- ~~⚠️ `created_at` возвращает NULL из БД (4 теста пропущены)~~ **ИСПРАВЛЕНО 2026-04-09**
  - **Файл:** `api/routes/scans.py`, `api/routes/simulations.py`
  - **Причина:** DatabaseManager не устанавливает created_at при создании записи
  - **Влияние:** Эндпоинты создания сканов/симуляций возвращают ValidationError
  - **Статус:** ✅ ИСПРАВЛЕНО - created_at устанавливается явно во всех INSERT методах
  - **Отчёт:** CREATED_AT_FIX_REPORT_2026-04-09.md

---

## Критично

- [x] `security/auth_manager.py` — устаревший Flask AuthManager, не используется FastAPI стеком (оставлен как legacy, не мешает)
- [x] `api/routes/admin.py` → `/admin/tasks/list` — реализован через `asyncio.all_tasks()`
- [x] `api/routes/auth.py` — `USERS_DB` перенесён в lazy-инициализацию `_get_users_db()`, нет блокировки при импорте
- [x] `from utils.cache_manager import CacheManager` — исправлен на `utils.caching.cache_manager` в 4 файлах
- [x] `api/sstv/__init__.py` — создан

## Функциональность

- [x] `push_realtime_updates()` — авто-подписка на `"metrics"` при connect добавлена в `ConnectionManager`
- [x] `api/routes/sstv.py` — кнопки Eye/Download/Delete в UI (`sstv/page.tsx`) подключены к обработчикам (2026-04-09)
- [x] `api/routes/reports.py` — PDF генерация реализована полностью через `ScientificPDFReport`
- [x] `api/routes/admin.py` → `/admin/cache/clear` — импорт исправлен

## Качество кода

- [x] `api/rate_limiter.py` — декораторы используют глобальный `limiter`, обновляемый в `setup_rate_limiter` до регистрации роутов — порядок корректен
- [x] `utils/database.py` — `count_reports()` fallback на `exports` — приемлемо, таблица `reports` не нужна
- [x] `api/routes/auth.py` — `_revoke_all_user_tokens` in-memory fallback задокументирован как known limitation

## RTL-SDR (железо подключено)

- [x] `satellite_tracker.py` — elevation расчёт заменён на точный ECI→ECEF через GMST
- [x] `waterfall_display.py` — добавлено Hann-окно, скользящий динамический диапазон
- [x] `sstv_decoder.py` — FM демодуляция с anti-aliasing FIR фильтром перед ресемплингом
- [x] `auto_recorder.py` — добавлен `import subprocess`
- [x] `sstv.py` — исправлен баг `elevation: position['latitude']`, `time_until_aos` из трекера
- [ ] Протестировать `--check` с реальным RTL-SDR V4
- [ ] Проверить `bias_tee=True` для активной антенны
- [ ] Откалибровать TCXO (`--freq-correction`)
- [ ] Записать первый пролёт NOAA/ISS

## Инфраструктура

- [ ] `deployment/docker-compose.prod.yml` — PostgreSQL сервис есть, но API использует SQLite. Либо добавить миграцию на PostgreSQL, либо убрать PostgreSQL из prod compose.
- [x] `.env` — `ENVIRONMENT=development` для локальной разработки (2026-04-09)
