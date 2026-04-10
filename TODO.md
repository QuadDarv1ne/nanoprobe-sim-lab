# Nanoprobe Sim Lab — TODO

**Последнее обновление:** 2026-04-10 00:30

## Последние улучшения (2026-04-10)

### Качество кода — 140+ исправлений
- ✅ **38 критических ошибок F821/F824** — исправлены undefined names (jwt, time, BytesIO, get_nasa_client, cache)
- ✅ **50+ ошибок F401/F841/B001/E722** — удалены unused imports/variables, исправлены bare except
- ✅ **4 ошибки E265/B028** — исправлены дубликаты shebang, добавлен stacklevel
- ✅ **.flake8 конфигурация** — добавлен B008 в ignore list (намеренно для FastAPI patterns)
- ✅ **Pre-commit hooks** — все проходят успешно
- ✅ **0 критических ошибок flake8** (было 88+)

### Тесты
- ✅ 66/66 тестов passing (test_api.py + test_database.py + test_integration_db.py + test_auth.py)
- ✅ 0 skipped тестов
- ✅ 0 регрессий после всех исправлений

---

## Known Issues

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
