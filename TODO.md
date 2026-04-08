# Nanoprobe Sim Lab — TODO

## Критично

- [x] `security/auth_manager.py` — устаревший Flask AuthManager, не используется FastAPI стеком (оставлен как legacy, не мешает)
- [x] `api/routes/admin.py` → `/admin/tasks/list` — реализован через `asyncio.all_tasks()`
- [x] `api/routes/auth.py` — `USERS_DB` перенесён в lazy-инициализацию `_get_users_db()`, нет блокировки при импорте
- [x] `from utils.cache_manager import CacheManager` — исправлен на `utils.caching.cache_manager` в 4 файлах
- [x] `api/sstv/__init__.py` — создан

## Функциональность

- [ ] `push_realtime_updates()` — рассылает метрики только в канал `"metrics"`, но клиенты подписываются через `/ws/realtime` вручную. Нет авто-подписки при коннекте. Добавить авто-подписку на `"metrics"` при connect.
- [ ] `api/routes/sstv.py` — кнопки Eye/Download/Delete в UI (`sstv/page.tsx`) не подключены к API. Нужны обработчики.
- [ ] `api/routes/reports.py` — проверить реализацию PDF генерации, `utils/reporting/pdf_report_generator.py` импортируется лениво.
- [ ] `api/routes/admin.py` → `/admin/cache/clear` — использует `utils/cache_manager.CacheManager`, проверить существование модуля.

## Качество кода

- [ ] `api/rate_limiter.py` — декораторы `auth_limit`, `api_limit` и т.д. используют модульный `limiter` (placeholder), а не пересозданный в `setup_rate_limiter`. Нужен механизм передачи актуального limiter в декораторы.
- [ ] `utils/database.py` — `count_reports()` делает `try/except` на несуществующую таблицу `reports`, fallback на `exports WHERE format='PDF'`. Создать таблицу `reports` или убрать fallback.
- [ ] `api/routes/auth.py` — `_revoke_all_user_tokens` при Redis fallback очищает весь `_in_memory_tokens` (все пользователи), а не только конкретного.

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
