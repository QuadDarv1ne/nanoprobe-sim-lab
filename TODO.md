# TODO.md — Nanoprobe Sim Lab

**Ветка:** `dev` -> `main`
**Последнее обновление:** 2026-05-04
**Python:** 3.14.4
**Тесты:** 1494 collected
**Статус:** ✅ Dev и main синхронизированы

---

## 🎯 Приоритетные задачи

### 🔴 CRITICAL

#### 1. Mypy errors — 771 ошибка
- **Статус:** ⚠️ **Требует внимания**
- **Проблема:** Высокое количество mypy ошибок
- **Влияние:** Статическая типизация не работает корректно
- **Приоритет:** Исправить критические type errors

---

### 🟡 HIGH

#### 2. Test coverage ~20% → 40%
- **Статус:** 📈 Прогресс есть
- **Текущее:** 1494 теста
- **Цель:** Увеличить покрытие критических модулей
- **Приоритетные модули:**
  - `utils/sdr/` — RTL-SDR менеджмент
  - `utils/ai/` — ML модели
  - `api/` — все роуты кроме auth
  - `src/` — CLI и веб-компоненты

#### 3. RTL-SDR V4 Production Ready
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

#### 4. Next.js Frontend v2.0 → Production
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

#### 5. PostgreSQL Migration
- **Статус:** 📄 Гайд создан, не выполнено
- **Сложность:** 10-15 часов
- **Шаги:**
  1. Установить PostgreSQL
  2. Настроить alembic для PostgreSQL
  3. Мигрировать данные из SQLite
  4. Обновить `.env` и конфиги
  5. Протестировать под нагрузкой

#### 6. AI/ML Features Enhancement
- **Статус:** ⚠️ Базовая функциональность
- **Осталось:**
  - Сбор датасета сигналов для обучения
  - Интеграция с внешними API (NASA, Zenodo)
  - Model versioning (MLflow)

#### 7. Redis Caching — Full Implementation
- **Статус:** ⚠️ Частично реализовано
- **Осталось:**
  - Полная интеграция во все API endpoints
  - Cache invalidation strategies
  - Redis pub/sub для WebSocket scaling

#### 8. Миграция print() → logging
- **Статус:** 📈 **Прогресс**
- **Было:** ~210 случаев
- **Текущее:** ~25 случаев
- **Успех:** -185 замен (88% прогресс)
- **Осталось:** ~25 случаев (в основном CLI интерфейсы и sample_code)

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
| print() в production | ~25 | 📈 **-185** |
| Test coverage | ~20% | 📈 |
| GitHub Workflows | 7 | ✅ |
| PytestCollectionWarning | 0 | ✅ |
| Mypy errors | 771 | ⚠️ Требуется внимание |
| Dev vs Main | синхронизированы | ✅ |

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

### Выполнено

- ✅ **Исправлено:** PytestReturnNotNoneWarning в 7 файлах
- ✅ **Исправлено:** alerting API validation issue
- ✅ **Исправлено:** PytestCollectionWarning — добавлен `__test__ = False`
- ✅ **Миграция print() → logging:** -185 замен (main.py, src/cli/main.py, project_manager.py, rtlsdr_control_panel.py)
- ✅ **Dev/Main синхронизация:** полностью синхронизированы

### Выявленные проблемы

- ⚠️ **Mypy error** — 771 ошибка требует внимания
- ⚠️ **Интеграционные тесты** — требуют запущенного API сервера
- ⚠️ **print() в production** — ~25 случаев (прогресс: -185, 88%)

### Следующие шаги

1. Увеличить test coverage до 30%
2. RTL-SDR V4 end-to-end тесты
3. Завершить миграцию print() → logging (цель: 0 print в production, кроме CLI интерфейсов)
4. Исправить mypy ошибки (приоритет: критические type errors)
5. PostgreSQL migration (опционально)

---

**Владелец проекта:** Дуплей Максим Игоревич
**Лицензия:** Проприетарная (ограниченные права)
