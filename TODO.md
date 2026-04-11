# Nanoprobe Sim Lab — TODO

**Последнее обновление:** 2026-04-11 18:30

## Статус проекта

- **Ветка:** `dev` == `main` ✅ (синхронизированы)
- **Рабочее дерево:** чистое ✅
- **Python:** 3.11 - 3.14

## 🎯 Текущие приоритеты

### HIGH (делать в первую очередь)
1. [x] Улучшать код в ветке `dev`
2. [x] Проверять тесты после изменений
3. [x] Merge в `main` после стабилизации
4. [x] Синхронизировать изменения с origin

### MEDIUM
5. [ ] Увеличить test coverage до 80%+
6. [x] Разбить api/routes/dashboard.py на модули (41 КБ)
7. [x] Разбить api/routes/auth.py на модули (30 КБ)
8. [ ] Настроить bias_tee для активной антенны

### LOW
9. [ ] Исправить оставшиеся ~150 E501 строк (HTML/CSS, SQL, config dicts)
10. [ ] Мигрировать frontend на Next.js (убрать Flask)
11. [ ] Решить SQLite vs PostgreSQL (есть guide в docs/)
12. [ ] Откалибровать TCXO (--freq-correction)
13. [ ] Создать mobile application

---

## 📝 Состояние проекта

### ✅ Реализовано
- FastAPI REST API с JWT + 2FA TOTP аутентификацией
- WebSocket real-time обновления
- GraphQL API
- Redis integration (кэширование)
- Next.js Frontend v2.0 (TypeScript, Tailwind, PWA) + legacy Flask
- RTL-SDR V4 полная поддержка (SSTV, NOAA, ADS-B, RTL_433)
- AI/ML анализ дефектов
- CI/CD pipeline (12 workflows)
- Prometheus + Grafana мониторинг
- Alembic migrations для БД
- Автоопределение координат по IP (МСК)
- FM Radio Unified (4 режима: listen, capture, scan, multi)
- Project CLI (validate, improve, cleanup, info)
- 24 API роута
- 66+ core тестов + 89 integration тестов

### ⚠️ Известные проблемы
- **pysstv** — только encoder, нет декодера SSTV
- **E501** — ~197 длинных строк (low priority, HTML/CSS/SQL/config)
- **SQLite vs PostgreSQL** — API использует SQLite, но docker-compose.prod.yml имеет PostgreSQL
- **legacy code** — `security/auth_manager.py` (Flask, не используется FastAPI стеком)

### 📦 Deprecated (архивированы)
- FM Radio: 5 файлов → `fm_radio_unified.py`
- Scripts: 4 файла → `scripts/project.py`
- Utils: 7 файлов → `utils/archived/`

---

## 🔧 RTL-SDR V4 Статус

### Реализовано
- [x] FM-радиовещание (87.5-108 МГц) — стерео декодирование
- [x] Авиадиапазоны VHF (118-137 МГц) — AM модуляция
- [x] ADS-B (1090 MHz) — трекинг самолётов
- [x] RTL_433 — беспроводные метеостанции, датчики
- [x] POCSAG — пейджинговая связь (512/1200/2400 baud)
- [x] NOAA захват — rtl_sdr_noaa_capture.py
- [x] SSTV анализ — спектрограмма, частотный анализ
- [x] Real-time visualizer — спектр + waterfall
- [x] ISS трекинг — расчёт пролётов МКС (SGP4 + CelesTrak)

### Ожидает
- [ ] Любительские радиостанции 2 м (144-146 МГц) и 70 см (430-440 МГц)
- [ ] AIS (161.975/162.025 MHz) — морские суда
- [ ] КВ-диапазон (HF) через апконвертер
- [ ] Проверить `bias_tee=True` для активной антенны
- [ ] Откалибровать TCXO (`--freq-correction`)
- [ ] Записать SSTV во время пролёта МКС

---

## 🏗️ Архитектура

### API Routes (24 endpoint'а)
`admin`, `adsb`, `alerting`, `analysis`, `auth`, `batch`, `comparison`, `dashboard`, `external_services`, `fm_radio`, `graphql`, `ml_analysis`, `monitoring`, `nasa`, `reports`, `rtl433`, `scans`, `simulations`, `sstv`, `sstv_advanced`, `sync_manager`, `system_export`, `weather`

### Utils (36 модулей)
`ai`, `analytics`, `api`, `backup_manager`, `batch_processor`, `caching`, `config`, `core`, `data`, `database`, `deployment`, `dev`, `logger`, `location_manager`, `monitoring`, `performance`, `reporting`, `security`, `simulator`, `spm_realtime_visualizer`, `structured_logger`, `surface_comparator`, `testing`, `test_framework`, `visualization`, `visualizer` + другие

### Тесты
- Core: test_api.py, test_database.py, test_integration_db.py, test_auth.py
- RTL-SDR: test_rtl_sdr_tools.py, test_rtl_sdr_recording.py, test_integration_rtlsdr.py
- API Routes: test_api_routes.py, test_sstv_api.py, test_external_routes.py
- Security: test_auth.py, test_security_headers.py, test_security_improvements.py, test_two_factor_auth.py
- Performance: test_cache_manager.py, test_redis_cache.py, test_rate_limiter.py, test_rate_limiting.py, test_circuit_breaker.py
- Utils: test_utils_modules.py, test_logger.py, test_error_handler.py

---

## 📏 Метрики качества

- **Тесты:** 66+ core passing (100%)
- **Integration:** 43 passed, 26 skipped, 0 failed
- **Pre-commit:** black, isort, flake8 проходят
- **Код:** 400+ исправлений качества (F821, F824, F401, F841, B001, E722, E501, W293)

---

## 🔄 Рабочий процесс

1. Разработка в ветке `dev`
2. Тестирование всех изменений
3. Code review (pre-commit hooks проходят)
4. Merge в `main` только стабильный код
5. Push в origin/main и origin/dev

---

## 📚 Ресурсы

- **RTL-SDR Blog:** https://www.rtl-sdr.com/
- **Celestrak TLE:** https://celestrak.org/
- **Satnobs:** https://satnobs.io/
- **ISS SSTV:** https://www.ariss.org/
