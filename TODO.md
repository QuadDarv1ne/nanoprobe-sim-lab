# TODO.md — Nanoprobe Sim Lab

**Ветка:** `dev` → `main`
**Последнее обновление:** 2026-05-01
**Python:** 3.14.4
**Тесты:** 1494 collected

---

## 🎯 Приоритетные задачи

### 🔴 CRITICAL

#### 1. Исправить PytestCollectionWarning
- **Статус:** ⚠️ В работе
- **Проблема:** 4 warning о классах с `__init__`, которые pytest пытается собрать как тесты
- **Файлы:**
  - `utils/testing/api_test_runner.py` — `TestResult`, `TestReport`
  - `utils/test_framework.py` — `TestFramework`
- **Решение:** Переименовать классы или добавить `__test__ = False`
- **Влияние:** Загрязняет вывод тестов, может мешать сборке

#### 2. print() → logging в src/
- **Статус:** ⚠️ ~190 вызовов осталось
- **Файлы:**
  - `src/cli/main.py` — 50+ print() в основном блоке
  - `src/cli/dashboard.py` — 5+ print()
  - `src/web/web_dashboard_unified.py` — 15+ print()
- **Приоритет:** LOW (CLI output, не критично)
- **Влияние:** Непоследовательность логирования

#### 3. print() → logging в api/
- **Статус:** ⚠️ ~20 вызовов осталось
- **Файлы:**
  - `api/alerting.py` — 4 print() в тестовом блоке
  - `api/integration.py` — 16 print() в тестовом блоке
- **Приоритет:** LOW (только тестовые блоки `if __name__ == "__main__"`)

---

### 🟡 HIGH

#### 4. Test coverage ~20% → 40%
- **Статус:** 📈 Прогресс есть
- **Текущее:** ~1494 теста
- **Цель:** Увеличить покрытие критических модулей
- **Приоритетные модули для тестирования:**
  - `utils/sdr/` — RTL-SDR менеджмент
  - `utils/ai/` — ML модели
  - `api/` — все роуты кроме auth
  - `src/` — CLI и веб-компоненты

#### 5. RTL-SDR V4 Production Ready
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

#### 6. Next.js Frontend v2.0 → Production
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

#### 7. PostgreSQL Migration
- **Статус:** 📄 Гайд создан, не выполнено
- **Сложность:** 10-15 часов
- **Шаги:**
  1. Установить PostgreSQL
  2. Настроить alembic для PostgreSQL
  3. Мигрировать данные из SQLite
  4. Обновить `.env` и конфиги
  5. Протестировать под нагрузкой

#### 8. AI/ML Features Enhancement
- **Статус:** ⚠️ Базовая функциональность
- **Осталось:**
  - Сбор датасета сигналов для обучения
  - Интеграция с внешними API (NASA, Zenodo)
  - Model versioning (MLflow)

#### 9. Redis Caching — Full Implementation
- **Статус:** ⚠️ Частично реализовано
- **Осталось:**
  - Полная интеграция во все API endpoints
  - Cache invalidation strategies
  - Redis pub/sub для WebSocket scaling

---

### 🔵 LOW

#### 10. Code Quality Improvements
- **Статус:** ✅ Pre-commit hooks настроены
- **Осталось:**
  - Code coverage badges в README
  - Dependabot/Renovate для зависимостей
  - SonarQube интеграция (опционально)

#### 11. Documentation
- **Статус:** ⚠️ Частично
- **Осталось:**
  - API Reference автогенерация
  - Video tutorials
  - Changelog automation из commit messages

#### 12. DevOps Improvements
- **Статус:** ✅ Docker Compose готов
- **Осталось:**
  - Monitoring (Prometheus + Grafana)
  - Log aggregation (ELK/Loki)
  - Backup automation

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
- ⚠️ **Интеграционные тесты** — требуют запущенного API сервера
- ⚠️ **PytestCollectionWarning** — 4 warning о классах с `__init__`

### Следующие шаги
1. Исправить PytestCollectionWarning (CRITICAL)
2. Продолжить миграцию print() → logging
3. Увеличить test coverage до 30%
4. RTL-SDR V4 end-to-end тесты

---

**Владелец проекта:** Дуплей Максим Игоревич
**Лицензия:** Проприетарная (ограниченные права)
