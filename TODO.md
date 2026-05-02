# TODO.md — Nanoprobe Sim Lab

**Ветка:** `dev` → `main`
**Последнее обновление:** 2026-05-02
**Python:** 3.14.4
**Тесты:** 1494 collected
**Статус:** ✅ Dev и main синхронизированы

---

## 🎯 Приоритетные задачи

### 🔴 CRITICAL

#### 1. PytestCollectionWarning — классы с `__init__`
- **Статус:** ✅ **Исправлено**
- **Проблема:** 4 warning о классах с `__init__`, которые pytest пытается собрать как тесты
- **Файлы:**
  - `utils/testing/api_test_runner.py` — `TestResult`, `TestReport`
  - `utils/test_framework.py` — `TestFramework`
- **Решение:** Добавлен `__test__ = False` во все классы
- **Влияние:** Загрязняет вывод тестов — **исправлено**

#### 2. DeprecationWarning — FastAPI on_event
- **Статус:** ✅ **Исправлено**
- **Проблема:** `@router.on_event("shutdown")` deprecated в FastAPI 0.118+
- **Файл:** `api/routes/sstv_advanced.py`
- **Решение:** Заменено на `@asynccontextmanager` lifespan
- **Влияние:** Warning при pytest collection — **исправлено**

#### 3. Mypy — дублирующийся модуль api.state
- **Статус:** ✅ **Исправлено**
- **Проблема:** `api/state.py` обнаруживается дважды как "state" и "api.state"
- **Влияние:** Mypy не может проверить проект корректно
- **Решение:** Добавлены `__init__.py` файлы в `api/` и `src/` директории

---

### 🟡 HIGH

#### 4. Test coverage ~20% → 40%
- **Статус:** 📈 Прогресс есть
- **Текущее:** ~1494 теста
- **Цель:** Увеличить покрытие критических модулей
- **Приоритетные модули:**
  - `utils/sdr/` — RTL-SDR менеджмент
  - `utils/ai/` — ML модели
  - `api/` — все роуты кроме auth
  - `src/` — CLI и веб-компоненты

#### 4. RTL-SDR V4 Production Ready
- **Статус:** ✅ Базовая функциональность готова
- **Реализовано:**
  - Ring buffer
  - Resource manager
  - Hardware health check
  - PPM калибровка
  - Автозахват спутников
- **Осталось:**
  - End-to-end тесты с реальным устройством
  - Документация по настройке

#### 5. Next.js Frontend v2.0 → Production
- **Статус:** ⚠️ Частично готово
- **Реализовано:**
  - WebGL водопад спектра
  - WebSocket streaming
  - Zoom/Pan
- **Осталось:**
  - Миграция всех фич из Flask v1.0
  - PWA оптимизация (service worker)
  - Offline mode

---

### 🟢 MEDIUM

#### 6. PostgreSQL Migration
- **Статус:** 📄 Гайд создан, не выполнено
- **Сложность:** 10-15 часов
- **Шаги:**
  1. Установить PostgreSQL
  2. Настроить alembic для PostgreSQL
  3. Мигрировать данные из SQLite
  4. Обновить `.env` и конфиги
  5. Протестировать под нагрузкой

#### 7. AI/ML Features Enhancement
- **Статус:** ⚠️ Базовая функциональность
- **Осталось:**
  - Сбор датасета сигналов для обучения
  - Интеграция с внешними API (NASA, Zenodo)
  - Model versioning (MLflow)

#### 8. Redis Caching — Full Implementation
- **Статус:** ⚠️ Частично реализовано
- **Осталось:**
  - Полная интеграция во все API endpoints
  - Cache invalidation strategies
  - Redis pub/sub для WebSocket scaling

---

### 🔵 LOW

#### 9. Code Quality Improvements
- **Статус:** ✅ Pre-commit hooks настроены
- **Осталось:**
  - Code coverage badges в README
  - Dependabot/Renovate для зависимостей
  - SonarQube интеграция (опционально)

#### 10. Documentation
- **Статус:** ⚠️ Частично
- **Осталось:**
  - API Reference автогенерация
  - Video tutorials
  - Changelog automation из commit messages

#### 11. DevOps Improvements
- **Статус:** ✅ Docker Compose готов
- **Осталось:**
  - Monitoring (Prometheus + Grafana)
  - Log aggregation (ELK/Loki)
  - Backup automation

#### 12. Удаление временных файлов
- **Статус:** ⚠️ Требует очистки
- **Файлы для удаления:**
  - `0` (неизвестный файл/директория)
  - `apply_fix.py`
  - `fix_pytest_warnings.py`
  - `fix_pytest_warnings2.py`
  - `utils/testing/api_test_runner.py.tmp`

---

## 📊 Метрики проекта

| Метрика | Значение | Статус |
|---------|----------|--------|
| API роуты | 43 файла | ✅ |
| Utils модули | 74 файла | ✅ |
| Тесты | 1494 | 📈 |
| Строки кода | ~51K+ | - |
| Flake8 критические | 0 | ✅ |
| bare except | 0 | ✅ |
| print() в production | ~210 | ⚠️ |
| Test coverage | ~20% | 📈 |
| GitHub Workflows | 7 | ✅ |
| PytestCollectionWarning | 0 | ✅ **Исправлено** |
| Mypy errors | 0 | ✅ **Исправлено** |

---

## 🔄 Workflow

### Правила
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

## 📝 Заметки по текущему спринту

### Выявленные проблемы
- ✅ **Исправлено:** PytestReturnNotNoneWarning в 7 файлах
- ✅ **Исправлено:** alerting API validation issue
- ✅ **Исправлено:** PytestCollectionWarning — добавлен `__test__ = False`
- ⚠️ **Mypy error** — дублирующийся модуль api/state.py
- ⚠️ **Интеграционные тесты** — требуют запущенного API сервера
- ⚠️ **print() в production** — ~210 случаев, требуется замена на logging

### Следующие шаги
1. Увеличить test coverage до 30%
2. RTL-SDR V4 end-to-end тесты
3. Миграция print() → logging
4. Исправить mypy ошибку (api/state.py)
5. PostgreSQL migration (опционально)
6. Очистить временные файлы

---

**Владелец проекта:** Дуплей Максим Игоревич
**Лицензия:** Проприетарная (ограниченные права)
