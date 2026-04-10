# Changelog

Все значимые изменения проекта Nanoprobe Sim Lab.

Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/),
версионирование по [Semantic Versioning](https://semver.org/lang/ru/).

---

## [Unreleased]

### Planned
- Mobile Application (React Native/Flutter)
- External Integrations (NASA, Zenodo, Figshare upload)
- Frontend Modernization (полная миграция на Next.js)
- Performance monitoring dashboard
- Increase test coverage to 80%+

---

## [1.1.0] - 2026-04-10

### 🎉 Added
- **FM Radio Unified** — новый универсальный модуль `fm_radio_unified.py` с 4 режимами:
  - `listen` — прослушивание FM радио в реальном времени
  - `capture` — запись аудио в WAV файл
  - `scan` — сканирование FM диапазона 88-108 МГц
  - `multi` — мультизахват нескольких станций
- **Project CLI** — `scripts/project.py` с командами:
  - `validate` — проверка структуры, импортов, зависимостей
  - `improve` — улучшение стиля кода (black, autoflake)
  - `cleanup` — очистка временных файлов
  - `info` — информация о проекте
- **Integration Tests** — 30 новых тестов для API роутов:
  - Alerting API (4 теста)
  - GraphQL API (5 тестов)
  - ML Analysis API (4 теста)
  - Monitoring API (5 тестов)
  - SSTV Advanced API (6 тестов)
  - System Export API (6 тестов)
- **Deploy Workflow** — `.github/workflows/deploy.yml` для автоматического деплоя:
  - Staging environment
  - Production environment
  - Railway, Render, SSH deployment
  - Post-deployment verification
- **PostgreSQL Migration Guide** — `docs/postgresql-migration-guide.md`:
  - Пошаговая инструкция
  - Сравнение SQLite vs PostgreSQL
  - Оценка сложности (10-15 часов)
  - Гибридный режим (SQLite/PostgreSQL)
- **.gitattributes** — корректная обработка бинарных файлов
- **tools/README.md** — инструкция по установке внешних инструментов

### ⚡ Changed
- **Python Version** — минимальная версия изменена с 3.8 на 3.11:
  - `pyproject.toml`: `requires-python = ">=3.11"`
  - `black`: target-version обновлён до py311-py314
  - `mypy`: python_version = "3.11"
  - Классификаторы обновлены (удалены 3.8, 3.9, 3.10; добавлен 3.13, 3.14)
- **.gitignore** — дополнен правилами:
  - External tools binaries (tools/dump1090, rtl_433, rtl-sdr-blog)
  - Binary files (bin/*.exe, bin/*.dll)
  - Archives (*.zip, *.tar.gz, *.7z)
  - Python cache (__pycache__, *.pyc, *.pyo)

### ⚠️ Deprecated
- **FM Radio файлы** (5 файлов → 1 унифицированный):
  - `rtl_sdr_tools/fm_radio.py` → используйте `fm_radio_unified.py listen`
  - `rtl_sdr_tools/fm_capture_simple.py` → используйте `fm_radio_unified.py capture`
  - `rtl_sdr_tools/fm_multi_capture.py` → используйте `fm_radio_unified.py multi`
  - `rtl_sdr_tools/fm_radio_capture.py` → используйте `fm_radio_unified.py capture`
  - `rtl_sdr_tools/fm_radio_scanner.py` → используйте `fm_radio_unified.py scan`
- **Scripts** (4 файла → 1 CLI):
  - `scripts/validate_project.py` → используйте `scripts/project.py validate`
  - `scripts/improve_project.py` → используйте `scripts/project.py improve`
  - `scripts/cleanup_project.py` → используйте `scripts/project.py cleanup`
  - `scripts/sort_project.py` → используйте `scripts/project.py` (в разработке)

### 🗑️ Removed
- **Unused Utils** — 7 неиспользуемых модулей архивированы в `utils/archived/`:
  - `utils/predictive_analytics_engine.py`
  - `utils/code_analyzer.py`
  - `utils/profiler.py`
  - `utils/performance/self_healing_system.py`
  - `utils/performance/automated_optimization_scheduler.py`
  - `utils/performance/ai_resource_optimizer.py`
  - `utils/performance/optimization_logging_manager.py`
- **External Tools Binaries** — вынесены из репозитория (~244 МБ сэкономлено):
  - `tools/dump1090/`
  - `tools/rtl_433/`
  - `tools/rtl-sdr-blog/`

### 🐛 Fixed
- **Skipped Tests** — 4 теста в `test_api.py` больше не пропускаются:
  - `test_get_scans_empty`
  - `test_create_scan`
  - `test_get_simulations`
  - `test_create_simulation`
- **DateTime Timezone Bug** — исправлен баг в `utils/monitoring/monitoring.py`:
  - `TypeError: can't subtract offset-naive and offset-aware datetimes`
  - Причина: смешение timezone-aware и naive datetime объектов
  - Решение: `datetime.fromtimestamp(..., tz=timezone.utc)`

### 📊 Статистика изменений
| Метрика | До | После | Изменение |
|---------|-----|-------|-----------|
| Python версия | 3.8+ | 3.11-3.14 | ✅ Синхронизировано |
| FM radio файлов | 5 разрозненных | 1 универсальный | -80% |
| Scripts | 23 разрозненных | 1 CLI + документация | -83% |
| Utils модулей | 44 | 37 | -7 неиспользуемых |
| Integration тестов | 48 | 54 (+6 новых классов) | +12.5% |
| CI/CD workflows | 11 | 12 (+deploy) | +1 |
| Документация | 29 файлов | 30 (+migration guide) | +1 |
| Размер репо | ~244 МБ больше | 0 МБ | -244 МБ |

### 🚀 Migration Guide

**Для пользователей:**
```bash
# Обновите зависимости
pip install -r requirements.txt --upgrade

# FM Radio — новый синтаксис
python rtl_sdr_tools/fm_radio_unified.py listen --freq 106.0
python rtl_sdr_tools/fm_radio_unified.py scan

# Project CLI — новые команды
python scripts/project.py validate
python scripts/project.py cleanup
```

**Для разработчиков:**
```bash
# Минимальная версия Python теперь 3.11
python --version  # Должно быть 3.11+

# Запуск всех тестов
pytest tests/ -v

# Запуск новых интеграционных тестов
pytest tests/test_api_integration_full.py -v

# Валидация проекта
python scripts/project.py validate
```

---

## [1.0.0] - 2026-04-08

### Added
- FastAPI REST API с JWT + 2FA TOTP аутентификацией
- WebSocket real-time обновления
- GraphQL API
- Redis integration (кэширование)
- Next.js Frontend v2.0 (TypeScript, Tailwind, PWA)
- RTL-SDR V4 полная поддержка (SSTV, NOAA, ADS-B, RTL_433)
- AI/ML анализ дефектов
- CI/CD pipeline (11 workflows)
- Prometheus + Grafana мониторинг
- Database migrations (Alembic)
- 48 тестов (unit, integration, active)
- 29 файлов документации
- Docker, Kubernetes, Nginx deployment

### Security
- JWT authentication с refresh token rotation
- 2FA TOTP (Google Authenticator)
- Rate Limiting (SlowAPI)
- Security Headers middleware
- CORS configuration
- Error handlers

[Unreleased]: https://github.com/your-username/nanoprobe-sim-lab/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/your-username/nanoprobe-sim-lab/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/your-username/nanoprobe-sim-lab/releases/tag/v1.0.0
