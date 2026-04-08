# Nanoprobe Sim Lab — TODO

## Критично

- [x] `security/auth_manager.py` — устаревший Flask AuthManager, не используется FastAPI стеком (оставлен как legacy, не мешает)
- [x] `api/routes/admin.py` → `/admin/tasks/list` — реализован через `asyncio.all_tasks()`
- [x] `api/routes/auth.py` — `USERS_DB` перенесён в lazy-инициализацию `_get_users_db()`, нет блокировки при импорте
- [x] `from utils.cache_manager import CacheManager` — исправлен на `utils.caching.cache_manager` в 4 файлах
- [x] `api/sstv/__init__.py` — создан

## Функциональность

- [x] `push_realtime_updates()` — авто-подписка на `"metrics"` при connect добавлена в `ConnectionManager`
- [ ] `api/routes/sstv.py` — кнопки Eye/Download/Delete в UI (`sstv/page.tsx`) не подключены к обработчикам
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
- [ ] `.env` — `ENVIRONMENT=production` при локальной разработке. Поменять на `development`.
