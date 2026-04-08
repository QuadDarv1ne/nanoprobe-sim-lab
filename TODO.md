# Nanoprobe Sim Lab — TODO

## Критично

- [ ] `security/auth_manager.py` — устаревший Flask-based AuthManager, не используется FastAPI стеком. Либо удалить, либо мигрировать на FastAPI зависимости. Сейчас создаёт отдельную `auth.db` параллельно с `nanoprobe.db`.
- [ ] `api/routes/admin.py` → `/admin/tasks/list` — заглушка "Celery в разработке". Либо реализовать через asyncio task registry, либо убрать endpoint.
- [ ] `api/routes/auth.py` — `USERS_DB` инициализируется при импорте модуля (блокирующий Argon2 хеш). При холодном старте задержка ~1-2с. Перенести в lifespan.

## Функциональность

- [ ] `push_realtime_updates()` — рассылает метрики только в канал `"metrics"`, но клиенты подписываются через `/ws/realtime` вручную. Нет авто-подписки при коннекте. Добавить авто-подписку на `"metrics"` при connect.
- [ ] `api/routes/sstv.py` — кнопки Eye/Download/Delete в UI (`sstv/page.tsx`) не подключены к API. Нужны обработчики.
- [ ] `api/routes/reports.py` — проверить реализацию PDF генерации, `utils/reporting/pdf_report_generator.py` импортируется лениво.
- [ ] `api/routes/admin.py` → `/admin/cache/clear` — использует `utils/cache_manager.CacheManager`, проверить существование модуля.

## Качество кода

- [ ] `api/rate_limiter.py` — декораторы `auth_limit`, `api_limit` и т.д. используют модульный `limiter` (placeholder), а не пересозданный в `setup_rate_limiter`. Нужен механизм передачи актуального limiter в декораторы.
- [ ] `utils/database.py` — `count_reports()` делает `try/except` на несуществующую таблицу `reports`, fallback на `exports WHERE format='PDF'`. Создать таблицу `reports` или убрать fallback.
- [ ] `api/routes/auth.py` — `_revoke_all_user_tokens` при Redis fallback очищает весь `_in_memory_tokens` (все пользователи), а не только конкретного.

## RTL-SDR (ждём железо)

- [ ] Протестировать `components/py-sstv-groundstation` с реальным RTL-SDR V4
- [ ] Проверить `bias_tee=True` для активной антенны
- [ ] Откалибровать TCXO

## Инфраструктура

- [ ] `deployment/docker-compose.prod.yml` — PostgreSQL сервис есть, но API использует SQLite. Либо добавить миграцию на PostgreSQL, либо убрать PostgreSQL из prod compose.
- [ ] `.env` — `ENVIRONMENT=production` при локальной разработке. Поменять на `development`.
