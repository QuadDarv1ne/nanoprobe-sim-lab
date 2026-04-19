# Nanoprobe Sim Lab — TODO
**Последнее обновление:** 2026-04-19
**Текущая ветка:** `dev`
**Целевая ветка:** `main`
**Python:** 3.11 - 3.14

---

## 📊 Статус синхронизации

| Ветка | Статус | Отставание |
|-------|--------|------------|
| `dev` | ✅ Синхронизирована | - |
| `main` | ✅ Синхронизирована | - |

**Last commit:** ``96e8b4b`` — docs: обновить TODO.md с WebGL водопадом

---

## 📊 Статистика тестов

| Модуль | Тесты | Статус |
|--------|-------|--------|
| ring_buffer.py | 20+ | ✅ |
| sdr_resource_manager.py | 17 | ✅ |
| DatabaseOperations | 24 | ✅ **FIXED** |
| sstv_advanced_api.py | 50+ | ✅ |
| hardware_health.py | 25+ | ✅ |
| signal_classifier.py | 20 | ✅ **FIXED** |
| defect_analyzer.py | 14 | ✅ |
| satellite_auto_capture.py | 23 | ✅ |
| rtl_sdr_auto_calibration.py | 15 | ✅ |
| **Всего** | **1405+** | 📈 |

---

## 🎯 Приоритетные задачи (CRITICAL → LOW)

### 🔴 CRITICAL

#### 1. Миграция print() → logging
- **Статус:** Исправлено ~112 из ~900 вызовов (~12.4%)
- **Выполнено:**
  - ✅ `api/main.py` — health check endpoint
  - ✅ `utils/api/space_image_downloader.py` — полный модуль
  - ✅ `utils/performance_profiler.py` — все print() заменены
  - ✅ `utils/test_framework.py` — замена print() на logger.info()
  - ✅ `utils/analytics.py` — замена print() на logger.info()
- **Осталось:**
  - Тестовые блоки `if __name__ == "__main__"` — низкий приоритет (CLI утилиты)

#### 2. Test coverage ~20% → 40%
- **Текущий статус:** ~20% (1405+ тестов)
- **Выполнено:**
  - ✅ Тесты для `ring_buffer.py` (20+ тестов)
  - ✅ Тесты для `sdr_resource_manager.py` (17 тестов)
  - ✅ Тесты для `DatabaseOperations` (24 теста) — **FIXED**
  - ✅ Тесты для `sstv_advanced_api.py` (50+ тестов)
  - ✅ Тесты для `hardware_health.py` (25+ тестов)
  - ✅ **Тесты для `signal_classifier.py` (20 тестов) — FIXED Python 3.14 compatibility**
  - ✅ Тесты для `defect_analyzer.py` (14 тестов)
  - ✅ Тесты для `satellite_auto_capture.py` (23 теста)
  - ✅ Тесты для `rtl_sdr_auto_calibration.py` (15 тестов)

---

### 🟡 HIGH

#### 3. RTL-SDR V4 Production Ready
**Реализовано:**
- [x] Ring buffer
- [x] Resource manager
- [x] Hardware health check
- [x] Trigger recorder
- [x] PPM калибровка
- [x] Автоматическая калибровка PPM — `rtl_sdr_auto_calibration.py`
- [x] Автозахват спутников NOAA/METEOR — `satellite_auto_capture.py`
- [x] **Troubleshooting guide** — `docs/rtl-sdr-troubleshooting.md`
- [x] **Улучшенный детектор устройств** — `utils/sdr/device_detector.py`
  - Автоматическое обнаружение всех устройств
  - Определение модели (V4/V3/V2/V1)
  - Температурный мониторинг для V4
  - Multi-device support
  - CLI интерфейс для диагностики

**Осталось:**
- [ ] **End-to-end тесты** с реальным устройством (ожидается)

#### 4. Next.js Frontend v2.0
**Реализовано:**
- [x] Базовая реализация
- [x] TypeScript + Tailwind CSS
- [x] Zustand state management
- [x] **WebGL водопад спектра** — `frontend/src/components/sstv/WaterfallDisplayWebGL.tsx`
- [x] **RealtimeWaterfall компонент** — `frontend/src/components/sstv/RealtimeWaterfall.tsx`
- [x] **WebSocket streaming** — real-time спектр через `/api/v1/sstv/ws/stream`
- [x] **Zoom/Pan функциональность** — колесо мыши, drag & drop, кнопки +/-
- [x] **GPU оптимизация** — WebGL shaders для color mapping

**Осталось:**
- [ ] **Миграция всех фич** из Flask dashboard v1.0
- [ ] **PWA оптимизация** — service worker, offline mode

#### 5. PostgreSQL Migration
**Статус:** Гайд создан, но миграция не выполнена
**Сложность:** 10-15 часов
**Действия:**
- [ ] Установить PostgreSQL
- [ ] Настроить alembic для PostgreSQL
- [ ] Мигрировать данные из SQLite
- [ ] Обновить `.env` и конфиги
- [ ] Протестировать под нагрузкой

---

### 🟢 MEDIUM

#### 6. AI/ML Features
**Реализовано:**
- [x] TensorFlow Lite классификация сигналов
- [x] AI анализ дефектов

**Осталось:**
- [ ] **Обучение моделей** — собрать датасет сигналов
- [ ] **Интеграция с внешними API** — NASA, Zenodo, Figshare
- [ ] **Model versioning** — MLflow или аналог

#### 7. Mobile Application
**Технологии:** React Native или Flutter
**Фичи:**
- Просмотр захваченных изображений
- Расписание пролётов спутников
- Уведомления о SSTV передачах с МКС
- Управление захватом (старт/стоп)

#### 8. Performance Optimization
- [ ] **Redis кэширование** — полностью реализовать
- [ ] **Database connection pooling** — оптимизация
- [ ] **WebSocket scaling** — Redis pub/sub для horizontal scaling
- [ ] **CDN для статики** — Next.js frontend

#### 9. Security Hardening
**Реализовано:**
- [x] JWT + 2FA TOTP
- [x] Rate limiting
- [x] Security headers

**Осталось:**
- [ ] **Penetration testing** — внешняя аудит безопасности
- [ ] **Secrets management** — HashiCorp Vault или аналог
- [ ] **Audit logging** — логирование всех действий пользователей

---

### 🔵 LOW

#### 10. Documentation
- [ ] **API Reference** — автогенерация из OpenAPI spec
- [ ] **User guides** — пошаговые инструкции для новичков
- [ ] **Video tutorials** — демонстрация возможностей
- [ ] **Changelog automation** — auto-generate from commit messages

#### 11. DevOps Improvements
**Реализовано:**
- [x] Docker Compose
- [x] Kubernetes manifests

**Осталось:**
- [ ] **Monitoring dashboard** — Prometheus + Grafana
- [ ] **Log aggregation** — ELK stack или Loki
- [ ] **Backup automation** — автоматические бэкапы БД

#### 12. Code Quality
- [x] **Pre-commit hooks** — полностью настроены
- [ ] **Code coverage badges** — в README
- [ ] **Dependency updates** — Dependabot или Renovate
- [ ] **Static analysis** — SonarQube интеграция

---

## 📊 Статистика проекта

### Код

| Метрика | Значение |
|---------|----------|
| API роуты | 43 файла (+2 новых) |
| Utils модули | 74 файла (+2 новых) |
| Тесты | 1405 теста |
| Строки кода | ~51K+ |

### Качество

| Метрика | Статус |
|---------|--------|
| flake8 критические | 0 ✅ |
| bare except | 0 в production ✅ |
| print() в utils/ | ~900 ⚠️ |
| CI lint | исправлен ✅ |
| Test coverage | ~20% 📈 |

### Архитектура

- **Backend:** FastAPI + JWT + 2FA TOTP + WebSocket + GraphQL
- **Frontend:** Next.js v2.0 (production) + Flask v1.0 (legacy)
- **Database:** SQLAlchemy + Alembic (SQLite → PostgreSQL migration planned)
- **Cache:** Redis integration
- **SDR:** RTL-SDR v4 support with ring buffer, resource management, auto-calibration, satellite auto-capture

---

## 🔄 Workflow

### Правила работы

1. **Работать в `dev`** — все изменения сначала в dev
2. **Проверки перед merge** — tests + lint + mypy
3. **Merge в `main`** — только после успешных проверок
4. **Синхронизация** — всегда делать push и merge
5. **Качество важнее количества** — фокус на реальные проблемы

### Git workflow

```bash
# Создать ветку для фичи
git checkout dev
git pull origin dev
git checkout -b feature/new-feature

# Разработать и тестировать
pytest tests/ -v
flake8 src/ utils/ api/
mypy src/ utils/ api/

# Коммит и push
git add .
git commit -m "feat: описание фичи"
git push origin feature/new-feature

# Create PR и merge в dev
# После проверки — merge dev в main
```

---

## 📝 Заметки

### Текущий спринт (2026-04-19)

- ✅ **Добавлена автоматическая калибровка PPM** — `rtl_sdr_auto_calibration.py`
- ✅ **Добавлен автозахват спутников NOAA/METEOR** — `satellite_auto_capture.py`
- ✅ **Создан скрипт миграции print() → logging** — `migrate_print_to_logging.py`
- ✅ **Добавлены API роуты** — `/api/v1/sstv/calibration/*`, `/api/v1/sstv/satellites/*`
- ✅ **Добавлены тесты** — 23 новых теста для новых модулей
- ✅ **Синхронизация dev и main веток** — выполнено
- ✅ **FIXED: signal_classifier.py совместимость с Python 3.14** — добавлены методы `_calculate_energy` и `_calculate_spectral_width`, исправлена обработка комплексных данных в FFT
- ✅ **FIXED: DatabaseOperations** — добавлен `__init__` с `db_path` и `enable_cache`, добавлен `get_connection()` метод
- ✅ **Добавлен WebGL водопадный дисплей** — `WaterfallDisplayWebGL.tsx`, `RealtimeWaterfall.tsx`
- ✅ **Создан troubleshooting guide** — `docs/rtl-sdr-troubleshooting.md`
- ✅ **Исправлено предупреждение pytest** — переименован `TestFramework` → `TestRunnerFramework`
- ✅ **Добавлен улучшенный детектор RTL-SDR устройств** — `utils/sdr/device_detector.py`
  - Автоматическое обнаружение всех устройств
  - Определение модели (V4/V3/V2/V1)
  - Температурный мониторинг для V4
  - Multi-device support
  - CLI интерфейс для диагностики
- ✅ **Завершён WebGL водопадный дисплей** — `frontend/src/components/sstv/WaterfallDisplayWebGL.tsx`
  - WebSocket streaming для real-time данных
  - Zoom/Pan (колесо мыши, drag & drop)
  - GPU оптимизация через WebGL shaders
  - Кнопки управления (+/-, Reset)
  - Индикатор уровня zoom

### Новые API endpoints

```
POST /api/v1/sstv/calibration/automated
GET /api/v1/sstv/calibration/current
GET /api/v1/sstv/calibration/status
POST /api/v1/sstv/calibration/reset
GET /api/v1/sstv/calibration/devices
GET /api/v1/sstv/satellites/passes
POST /api/v1/sstv/satellites/scheduler/start
POST /api/v1/sstv/satellites/scheduler/stop
GET /api/v1/sstv/satellites/status
GET /api/v1/sstv/satellites/supported
GET /api/v1/sstv/satellites/config
GET /api/v1/sstv/satellites/captures
DELETE /api/v1/sstv/satellites/captures/{filename}
```

### Следующие спринты

1. PostgreSQL migration
2. Next.js Frontend фичи миграция
3. Mobile app MVP
4. AI/ML model training

---

## 🔗 Ресурсы

- **RTL-SDR Blog:** https://www.rtl-sdr.com/
- **Celestrak TLE:** https://celestrak.org/
- **Satnobs:** https://satnobs.io/
- **ISS SSTV:** https://www.ariss.org/

---

**Владелец проекта:** Дуплей Максим Игоревич
**Лицензия:** Проприетарная (ограниченные права)
**Последняя синхронизация:** 2026-04-19
