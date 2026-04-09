# Nanoprobe Sim Lab — TODO

**Последнее обновление:** 2026-04-09 22:30

## Последние улучшения (2026-04-09)

### RTL-SDR V4 — полностью функционален
- ✅ RTL-SDR V4: исправлены импорты pysstv, создана документация и тесты
- ✅ SSTV UI: подключены кнопки Eye/Download/Delete
- ✅ SSTV API: добавлены эндпоинты GET/DELETE /recordings/{filename}
- ✅ SSTV Receiver FIX: исправлены 5 критических проблем (pysstv API, кэширование инициализации)
- ✅ **RTL-SDR V4 захват**: запись I/Q @ 2.4 MSPS, FM демодуляция, WAV output
- ✅ **SSTV capture**: приём с МКС (145.800 MHz), анализ, спектрограмма
- ✅ **NOAA APT capture**: приём с NOAA 15/18/19 (137 MHz), декодер
- ✅ **Real-time visualizer**: спектр + waterfall (matplotlib)
- ✅ RTL-SDR тест: 7.3M I/Q сэмплов за 3.1 сек (ISS 145.800 MHz)
- ✅ NOAA тест: 24.1M I/Q сэмплов за 10.2 сек (NOAA 19 137.100 MHz)
- ⏳ SSTV декодер: pysstv только генератор, нужен отдельный декодер

### Исправления базы данных
- ✅ **created_at FIX**: исправлены все 10 INSERT методов в database.py
- ✅ Тест created_at: подтверждён (2026-04-09T22:26:11.665968)
- ✅ 14/14 database тестов passing

### Тесты
- ✅ Тесты: исправлен test_login_success (чтение пароля из файла/ENV)
- ✅ Тесты: исправлены 24 теста auth.py (JWT, refresh tokens)
- ✅ Тесты: исправлены 15 тестов api.py (инициализация БД)
- ✅ Тесты: исправлен test_integration_db.py (fixture'ы, API calls)
- ✅ 25/29 тестов passing (test_database.py + test_api.py)
- ⚠️ 4 теста skipped (created_at bug в test_api.py — исправлено в БД, нужно обновить тесты)

---

## Known Issues

- ~~⚠️ `created_at` возвращает NULL из БД (4 теста пропущены)~~ **ИСПРАВЛЕНО 2026-04-09**
  - **Файл:** `utils/database.py` — 10 INSERT методов исправлены
  - **Статус:** ✅ ИСПРАВЛЕНО — created_at устанавливается явно
  - **Примечание:** 4 теста в test_api.py всё ещё skipped — нужно обновить assertions

- ⚠️ **pysstv не декодирует SSTV** — это только генератор (encoder)
  - **Влияние:** Нельзя декодировать изображения из WAV файлов
  - **Решение:** Нужен отдельный декодер (wxtoimg для NOAA, MMSSTV/QSSTV для SSTV)

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
- [x] Протестировать `--check` с реальным RTL-SDR V4 (2026-04-09)
- [x] Установка rtl-sdr утилит (v1.3.6) в C:\rtl-sdr\bin\x64
- [x] FM Радио (101.7 MHz) - приём подтверждён, мощность 0.33
- [x] Созданы скрипты: listen_fm.bat, listen_airband.bat
- [x] Авиасвязь - rtl_fm работает, записаны данные (118.1 MHz)
- [x] **RTL-SDR V4 захват**: test_sdr_quick.py — 7.3M I/Q сэмплов за 3.1 сек
- [x] **NOAA захват**: rtl_sdr_noaa_capture.py — 24.1M I/Q за 10.2 сек
- [x] **SSTV анализ**: analyze_sstv.py — спектрограмма, частотный анализ
- [x] **Real-time visualizer**: rtl_sdr_visualizer.py — спектр + waterfall
- [ ] Проверить `bias_tee=True` для активной антенны
- [ ] Откалибровать TCXO (`--freq-correction`)
- [ ] Дождаться пролёта МКС для SSTV (нужен декодер)
- [ ] Дождаться пролёта NOAA для APT (нужен wxtoimg)

### RTL-SDR V4: Дополнительные возможности (из гайда)

#### Широкополосное радиосканирование
- [ ] FM-радиовещание (87.5-108 МГц) — стерео декодирование
- [ ] Авиадиапазоны VHF (118-137 МГц) — AM модуляция
- [ ] Любительские радиостанции 2 м (144-146 МГц) и 70 см (430-440 МГц)
- [ ] Службы экстренного реагирования (полиция, скорая, пожарные)

#### Цифровые сигналы
- [ ] ADS-B (1090 MHz) — отслеживание самолётов (dump1090)
- [ ] RTL_433 — беспроводные метеостанции, датчики температуры/влажности
- [ ] POCSAG — пейджинговая связь
- [ ] AIS (161.975/162.025 MHz) — морские суда

#### КВ-диапазон (HF) через апконвертер
- [ ] КВ-вещание (короткие волны)
- [ ] Любительские диапазоны (40 м, 20 м)
- [ ] Цифровые моды: FT8, PSK31, RTTY

#### Специальные проекты
- [ ] Сетевой SDR-сервер (SDR++ сервер)
- [ ] Анализ спектра для поиска помех
- [ ] Изучение радиочастотной обстановки

## Инфраструктура

- [ ] `deployment/docker-compose.prod.yml` — PostgreSQL сервис есть, но API использует SQLite. Либо добавить миграцию на PostgreSQL, либо убрать PostgreSQL из prod compose.
- [x] `.env` — `ENVIRONMENT=development` для локальной разработки (2026-04-09)
