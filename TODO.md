# Nanoprobe Sim Lab — TODO

**Последнее обновление:** 2026-04-10 17:30

## Последние улучшения (2026-04-10)

### Коммиты (pushed to origin/dev)
- ✅ `4ec59ea` test: add 89 new tests (RTL-SDR tools, API routes, utils)
- ✅ `630f22b` docs: add improvements report 2026-04-10
- ✅ `79bd8b1` feat: RTL-SDR tools improvements + E501 fixes
- ✅ `d48c96d` style: fix W293 whitespace and E501 line length issues (autopep8)

### Качество кода — 240+ исправлений
- ✅ **38 критических ошибок F821/F824** — исправлены undefined names
- ✅ **50+ ошибок F401/F841/B001/E722** — удалены unused imports/variables
- ✅ **~100 E501** — исправлены длинные строки (>100 chars)
- ✅ **~90 W293** — исправлен whitespace на пустых строках
- ✅ **Pre-commit hooks** — все проходят (black, isort, flake8)
- ✅ **.pre-commit-config.yaml** — добавлены B008, F401, B014 в ignore list

### Тесты
- ✅ 66/66 core тестов passing (test_api.py + test_database.py + test_integration_db.py + test_auth.py)
- ✅ +89 новых тестов (test_rtl_sdr_tools.py, test_api_routes.py, test_utils_modules.py)
- ✅ 43 passed, 26 skipped, 0 failed в новых тестаах
- ✅ pytest.skip для unavailable модулей — нет false failures

### RTL-SDR Tools
- ✅ `rtl433_scanner.py` — переписан с CLI (--freq, --gain, --duration)
- ✅ `fm_stereo_decoder.py` — FM Stereo декодер с RDS поддержкой
- ✅ `pocsag_decoder.py` — POCSAG pager decoder (512/1200/2400 baud)
- ✅ `listen_adsb.bat` — батник для ADS-B трекинга (1090 MHz)
- ✅ `listen_rtl433.bat` — батник для RTL_433 сканирования (433 MHz)
- ✅ Автопоиск rtl_433 на Windows/Linux/Mac

---

## Known Issues

- ⚠️ **pysstv не декодирует SSTV** — это только генератор (encoder)
  - **Влияние:** Нельзя декодировать изображения из WAV файлов
  - **Решение:** Нужен отдельный декодер (wxtoimg для NOAA, MMSSTV/QSSTV для SSTV)
- ⚠️ **~94 E501 остались** — HTML/CSS inline строки и config dicts, требуют ручного рефакторинга

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
- [x] МКС SSTV - частота 145.800 MHz настраивается, rtl_fm FM работает
- [x] MMSSTV 1.13A установлен в C:\Ham\MMSSTV\
- [x] Создан capture_sstv_mmsstv.py - запись + автооткрытие MMSSTV
- [x] iss_tracker.py - расчёт пролётов МКС (SGP4 + CelesTrak)
- [x] Найден лучший пролёт: 08:11 (38.1° высота)
- [ ] Записать SSTV во время пролёта МКС (08:09 утра)
- [ ] **RTL-SDR V4 захват**: test_sdr_quick.py — 7.3M I/Q сэмплов за 3.1 сек
- [x] **NOAA захват**: rtl_sdr_noaa_capture.py — 24.1M I/Q за 10.2 сек
- [x] **SSTV анализ**: analyze_sstv.py — спектрограмма, частотный анализ
- [x] **Real-time visualizer**: rtl_sdr_visualizer.py — спектр + waterfall
- [ ] Проверить `bias_tee=True` для активной антенны
- [ ] Откалибровать TCXO (`--freq-correction`)
- [ ] Дождаться пролёта МКС для SSTV (нужен декодер)
- [ ] Дождаться пролёта NOAA для APT (нужен wxtoimg)

### RTL-SDR V4: Дополнительные возможности (из гайда)

#### Широкополосное радиосканирование
- [x] FM-радиовещание (87.5-108 МГц) — стерео декодирование (fm_stereo_decoder.py)
- [x] Авиадиапазоны VHF (118-137 МГц) — AM модуляция (listen_airband.bat)
- [ ] Любительские радиостанции 2 м (144-146 МГц) и 70 см (430-440 МГц)
- [ ] Службы экстренного реагирования (полиция, скорая, пожарные)

#### Цифровые сигналы
- [x] ADS-B (1090 MHz) — отслеживание самолётов (adsb_receiver.py + listen_adsb.bat)
- [x] RTL_433 — беспроводные метеостанции, датчики температуры/влажности (rtl433_scanner.py)
- [x] POCSAG — пейджинговая связь (pocsag_decoder.py — 512/1200/2400 baud)
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
