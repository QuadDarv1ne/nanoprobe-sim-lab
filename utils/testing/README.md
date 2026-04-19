# Testing Utilities for Nanoprobe Sim Lab

Набор утилит для тестирования API и производительности проекта.

## 📦 Модули

### 1. API Test Runner (`api_test_runner.py`)

Автоматическое тестирование всех API endpoints с валидацией ответов.

**Возможности:**
- Автоматическое тестирование health endpoints
- Тестирование auth endpoints
- Тестирование public endpoints
- Генерация отчётов (текстовый и JSON)
- Поддержка кастомных base URL
- Verbose режим для отладки

**Использование:**
```bash
# Базовое использование
python utils/testing/api_test_runner.py

# С кастомным base URL
python utils/testing/api_test_runner.py --base-url http://localhost:8000

# С сохранением отчёта в JSON
python utils/testing/api_test_runner.py --output report.json

# Verbose режим
python utils/testing/api_test_runner.py --verbose
```

**Аргументы:**
- `--base-url, -b`: Base URL API (по умолчанию: `http://localhost:8000`)
- `--timeout, -t`: Таймаут запросов в секундах (по умолчанию: `30.0`)
- `--verbose, -v`: Включить подробный вывод
- `--output, -o`: Сохранить отчёт в JSON файл

**Пример вывода:**
```
======================================================================
API TEST REPORT
======================================================================
Base URL: http://localhost:8000
Timestamp: 2026-04-19T10:00:00.000000+00:00
Total Tests: 15
Passed: 14 ✅
Failed: 1 ❌
Success Rate: 93.3%
Total Time: 2.45s
======================================================================

❌ FAILED TESTS:
----------------------------------------------------------------------
  GET /api/v1/some-endpoint
    Expected: 200, Got: 500
    Error: Internal server error
    Time: 0.123s
======================================================================
```

---

### 2. API Performance Profiler (`api_profiler.py`)

Профилирование производительности API endpoints.

**Возможности:**
- Многократное тестирование endpoints
- Статистика времени ответа (min, max, avg, median, P95)
- Расчёт success rate
- Мониторинг статус кодов
- Генерация детализированных отчётов
- Сохранение результатов в JSON

**Использование:**
```bash
# Базовое использование (10 итераций на endpoint)
python utils/testing/api_profiler.py

# С кастомным количеством итераций
python utils/testing/api_profiler.py --iterations 50

# С сохранением отчёта
python utils/testing/api_profiler.py --output profile.json --iterations 100

# Verbose режим
python utils/testing/api_profiler.py --verbose
```

**Аргументы:**
- `--base-url, -b`: Base URL API (по умолчанию: `http://localhost:8000`)
- `--iterations, -i`: Количество итераций на endpoint (по умолчанию: `10`)
- `--timeout, -t`: Таймаут запросов в секундах (по умолчанию: `30.0`)
- `--verbose, -v`: Включить подробный вывод
- `--output, -o`: Сохранить отчёт в JSON файл

**Пример вывода:**
```
================================================================================
API PERFORMANCE PROFILER REPORT
================================================================================
Base URL: http://localhost:8000
Timestamp: 2026-04-19T10:00:00.000000+00:00
Total Requests: 70
Total Time: 5.23s
Average Rate: 13.38 req/s
================================================================================

📊 GET /health
--------------------------------------------------------------------------------
  Iterations: 10
  Success Rate: 100.0%
  Min Time: 12.34ms
  Max Time: 45.67ms
  Avg Time: 23.45ms
  Median: 22.10ms
  P95: 40.00ms

📊 GET /api/v1/scans
--------------------------------------------------------------------------------
  Iterations: 10
  Success Rate: 100.0%
  Min Time: 23.45ms
  Max Time: 67.89ms
  Avg Time: 34.56ms
  Median: 32.10ms
  P95: 55.00ms
...
================================================================================
```

---

## 🧪 Тестирование

Запуск тестов для утилит:

```bash
# Все тесты testing модуля
python -m pytest tests/testing/ -v

# Конкретный модуль
python -m pytest tests/testing/test_api_test_runner.py -v
python -m pytest tests/testing/test_api_profiler.py -v

# С покрытием
python -m pytest tests/testing/ --cov=utils/testing --cov-report=term-missing
```

---

## 📊 Статистика

| Метрика | Значение |
|---------|----------|
| Тесты для API Test Runner | 19 тестов |
| Тесты для API Profiler | 21 тест |
| Общее количество тестов | 40 тестов |
| Coverage | ~95% |

---

## 🔧 Интеграция в CI/CD

### GitHub Actions

```yaml
- name: Run API Tests
  run: python utils/testing/api_test_runner.py --base-url http://localhost:8000

- name: Run API Performance Tests
  run: python utils/testing/api_profiler.py --base-url http://localhost:8000 --iterations 5
```

---

## 📝 Примеры использования

### Проверка здоровья API после деплоя

```bash
python utils/testing/api_test_runner.py \
  --base-url https://api.yourdomain.com \
  --output deploy-check.json
```

### Бенчмарк API перед релизом

```bash
python utils/testing/api_profiler.py \
  --base-url https://api.yourdomain.com \
  --iterations 100 \
  --output release-benchmark.json
```

### Сравнение производительности dev vs prod

```bash
# Dev
python utils/testing/api_profiler.py \
  --base-url http://localhost:8000 \
  --iterations 50 \
  --output dev-profile.json

# Prod
python utils/testing/api_profiler.py \
  --base-url https://api.yourdomain.com \
  --iterations 50 \
  --output prod-profile.json
```

---

## 🚀 Будущие улучшения

- [ ] Поддержка аутентификации (JWT tokens)
- [ ] Тестирование WebSocket endpoints
- [ ] Генерация HTML отчётов
- [ ] Интеграция с Prometheus/Grafana
- [ ] Поддержка нагрузочного тестирования (locust/pytest-benchmark)

---

**Автор:** Nanoprobe Sim Lab Team
**Последнее обновление:** 2026-04-19
