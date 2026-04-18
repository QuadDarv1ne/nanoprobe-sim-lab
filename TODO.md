# Nanoprobe Sim Lab — TODO
**Последнее обновление:** 2026-04-18
**Текущая ветка:** `dev`
**Целевая ветка:** `main`
**Python:** 3.11 - 3.14

---

## 🎯 Приоритетные задачи (CRITICAL → LOW)

### 🔴 CRITICAL

#### 1. Синхронизация веток dev → main
- [x] Проверить тесты: `pytest tests/ -v`
- [x] Проверить lint: `flake8 src/ utils/ api/ --max-line-length=100`
- [x] Проверить type hints: `mypy src/ utils/ api/ --ignore-missing-imports`
- [x] Merge dev в main после успешных проверок

#### 2. Завершить миграцию print() → logging
- **Статус:** Исправлено ~104 из ~900 вызовов (~11.5%)
- **Осталось:**
  - `utils/performance_profiler.py` — критичные 3 print() в техническом коде
  - Тестовые блоки `if __name__ == "__main__"` — низкий приоритет
- **Действие:** Исправить production код вне тестовых блоков

#### 3. Увеличить test coverage до 40%
- **Текущий статус:** ~20% (1257 тестов)
- **Приоритетные модули:**
  - `api/routes/sstv_advanced.py` — SDR advanced endpoints
  - `utils/sdr/` — ring_buffer, resource_manager, hardware_health
  - `utils/db/operations.py` — CRUD операции
  - `utils/ml/` — signal_classifier, defect_analyzer

---

### 🟡 HIGH

#### 4. RTL-SDR V4 Production Ready
- [x] Ring buffer реализован
- [x] Resource manager реализован
- [x] Hardware health check реализован
- [x] Trigger recorder реализован
- [x] PPM калибровка реализована
- [x] **Автоматическая калибровка PPM** — `rtl_sdr_auto_calibration.py` с методами rtl_test, signal, auto
- [x] **Автозахват спутников NOAA/METEOR** — `satellite_auto_capture.py` с планировщиком
- [ ] **README: Troubleshooting RTL-SDR v4** — DVB-T blacklist, udev правила, PPM drift, перегрев
- [ ] **End-to-end тесты** с реальным устройством (ожидается)

#### 5. Next.js Frontend v2.0
- [x] Базовая реализация
- [x] TypeScript + Tailwind CSS
- [x] Zustand state management
- [ ] **WebGL/Canvas водопад спектра** — `frontend/src/components/sstv/WaterfallDisplay.tsx`
- [ ] **Миграция всех фич** из Flask dashboard v1.0
- [ ] **PWA оптимизация** — service worker, offline mode

#### 6. PostgreSQL Migration
- **Статус:** Гайд создан, но миграция не выполнена
- **Сложность:** 10-15 часов
- **Действия:**
  - [ ] Установить PostgreSQL
  - [ ] Настроить alembic для PostgreSQL
  - [ ] Мигрировать данные из SQLite
  - [ ] Обновить `.env` и конфиги
  - [ ] Протестировать под нагрузкой

---

### 🟢 MEDIUM

#### 7. AI/ML Features
- [x] TensorFlow Lite классификация сигналов
- [x] AI анализ дефектов
- [ ] **Обучение моделей** — собрать датасет сигналов
- [ ] **Интеграция с внешними API** — NASA, Zenodo, Figshare
- [ ] **Model versioning** — MLflow или аналог

#### 8. Mobile Application
- **Технологии:** React Native или Flutter
- **Фичи:**
  - Просмотр захваченных изображений
  - Расписание пролётов спутников
  - Уведомления о SSTV передачах с МКС
  - Управление захватом (старт/стоп)

#### 9. Performance Optimization
- [ ] **Redis кэширование** — полностью реализовать
- [ ] **Database connection pooling** — оптимизация
- [ ] **WebSocket scaling** — Redis pub/sub для horizontal scaling
- [ ] **CDN для статики** — Next.js frontend

#### 10. Security Hardening
- [x] JWT + 2FA TOTP
- [x] Rate limiting
- [x] Security headers
- [ ] **Penetration testing** — внешняя аудит безопасности
- [ ] **Secrets management** — HashiCorp Vault или аналог
- [ ] **Audit logging** — логирование всех действий пользователей

---

### 🔵 LOW

#### 11. Documentation
- [ ] **API Reference** — автогенерация из OpenAPI spec
- [ ] **User guides** — пошаговые инструкции для новичков
- [ ] **Video tutorials** — демонстрация возможностей
- [ ] **Changelog automation** — auto-generate from commit messages

#### 12. DevOps Improvements
- [x] Docker Compose
- [x] Kubernetes manifests
- [ ] **Monitoring dashboard** — Prometheus + Grafana
- [ ] **Log aggregation** — ELK stack или Loki
- [ ] **Backup automation** — автоматические бэкапы БД

#### 13. Code Quality
- [ ] **Pre-commit hooks** — полностью настроить
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
| Тесты | 1276 теста (+23 новых) |
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
- **SDR:** RTL-SDR v4 support with ring buffer, resource management, **auto-calibration**, **satellite auto-capture**

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

### Текущий спринт (2026-04-18)
- ✅ **Добавлена автоматическая калибровка PPM** — `rtl_sdr_auto_calibration.py`
- ✅ **Добавлен автозахват спутников NOAA/METEOR** — `satellite_auto_capture.py`
- ✅ **Создан скрипт миграции print() → logging** — `migrate_print_to_logging.py`
- ✅ **Добавлены API роуты** — `/api/v1/sstv/calibration/*`, `/api/v1/sstv/satellites/*`
- ✅ **Добавлены тесты** — 23 новых теста для новых модулей
- Синхронизация dev и main веток
- Подготовка к тестированию с реальным RTL-SDR V4

### Новые API endpoints
```
POST   /api/v1/sstv/calibration/automated
GET    /api/v1/sstv/calibration/current
GET    /api/v1/sstv/calibration/status
POST   /api/v1/sstv/calibration/reset
GET    /api/v1/sstv/calibration/devices

GET    /api/v1/sstv/satellites/passes
POST   /api/v1/sstv/satellites/scheduler/start
POST   /api/v1/sstv/satellites/scheduler/stop
GET    /api/v1/sstv/satellites/status
GET    /api/v1/sstv/satellites/supported
GET    /api/v1/sstv/satellites/config
GET    /api/v1/sstv/satellites/captures
DELETE /api/v1/sstv/satellites/captures/{filename}
```

### Следующие спринты
1. PostgreSQL migration
2. Next.js WebGL waterfall
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
**Последняя синхронизация:** 2026-04-18
