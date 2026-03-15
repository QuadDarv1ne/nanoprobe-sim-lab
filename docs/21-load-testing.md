# Load Testing Guide

**Руководство по нагрузочному тестированию Nanoprobe Sim Lab API**

---

## 📋 Содержание

1. [Быстрый старт](#быстрый-старт)
2. [Настройки](#настройки)
3. [Примеры использования](#примеры-использования)
4. [Интерпретация результатов](#интерпретация-результатов)
5. [Рекомендации](#рекомендации)

---

## 🚀 Быстрый старт

### Требования

```bash
pip install requests
```

### Запуск теста

```bash
# Базовый запуск (10 пользователей, 60 секунд)
python tests/load_test.py

# Быстрый тест (30 секунд)
python tests/load_test.py --duration 30

# Интенсивный тест (50 пользователей, 2 минуты)
python tests/load_test.py --users 50 --duration 120
```

---

## ⚙️ Настройки

### Параметры командной строки

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--url` | Base URL API | `http://localhost:8000` |
| `--users` | Количество одновременных пользователей | `10` |
| `--duration` | Длительность теста (секунды) | `60` |
| `--timeout` | Таймаут запроса (секунды) | `10` |

### Примеры настроек

```bash
# Тест локального API
python tests/load_test.py --url http://localhost:8000

# Тест с 20 пользователями на 90 секунд
python tests/load_test.py --users 20 --duration 90

# Тест production API (с увеличенным таймаутом)
python tests/load_test.py --url https://api.nanoprobe-lab.ru --users 30 --timeout 30
```

---

## 📊 Примеры использования

### 1. Быстрый тест (30 секунд)

```bash
python tests/load_test.py --duration 30
```

**Результат:**
```
🚀 Load Testing: Nanoprobe Sim Lab API
======================================================================
Base URL: http://localhost:8000
Users: 10, Duration: 30с
Start Time: 2026-03-14 15:30:00
======================================================================

📋 Проверка доступности API...
✅ API доступен: {'status': 'healthy', 'timestamp': '...'}

🔵 Тест: Health check
   Пользователей: 10, Длительность: 30с

   ✅ Результаты:
      Запросов: 120 (успешно: 120, ошибок: 0)
      Success Rate: 100.0%
      RPS: 4.00 запросов/сек
      Response Time:
         Min: 12.50ms
         Max: 45.30ms
         Avg: 25.80ms
         Median: 24.10ms
         P95: 38.50ms
         P99: 42.20ms
```

### 2. Стандартный тест (60 секунд)

```bash
python tests/load_test.py
```

**Тестируемые эндпоинты:**
- `GET /health` - Health check
- `GET /health/detailed` - Detailed health
- `GET /api/v1/dashboard/stats` - Dashboard statistics
- `GET /api/v1/scans/` - List scans
- `GET /api/v1/simulations/` - List simulations
- `GET /metrics/realtime` - Realtime metrics

### 3. Интенсивный тест (120 секунд, 50 пользователей)

```bash
python tests/load_test.py --users 50 --duration 120
```

**Для stress testing production среды.**

---

## 📈 Интерпретация результатов

### Метрики

#### RPS (Requests Per Second)

| RPS | Оценка | Описание |
|-----|--------|----------|
| > 100 | ✅ Отлично | Высокая производительность |
| 50-100 | ✅ Хорошо | Нормальная производительность |
| 10-50 | ⚠️ Средне | Требуется оптимизация |
| < 10 | ❌ Низко | Критично низкая производительность |

#### Response Time (Время ответа)

| Avg (ms) | Оценка | Описание |
|----------|--------|----------|
| < 100 | ✅ Отлично | Мгновенный ответ |
| 100-500 | ✅ Хорошо | Быстрый ответ |
| 500-1000 | ⚠️ Средне | Замедленный ответ |
| > 1000 | ❌ Плохо | Медленный ответ |

#### P95 (95-й перцентиль)

**95% запросов быстрее этого значения.**

| P95 (ms) | Оценка |
|----------|--------|
| < 200 | ✅ Отлично |
| 200-500 | ✅ Хорошо |
| 500-1000 | ⚠️ Средне |
| > 1000 | ❌ Плохо |

#### Success Rate

| Success Rate | Оценка |
|--------------|--------|
| ≥ 99% | ✅ Отлично |
| 95-99% | ✅ Хорошо |
| 80-95% | ⚠️ Внимание |
| < 80% | ❌ Критично |

---

## 💡 Рекомендации

### Если Success Rate < 95%

1. **Проверьте логи API:**
   ```bash
   tail -f logs/backend.log
   ```

2. **Проверьте базу данных:**
   ```bash
   python -c "from utils.database import DatabaseManager; db = DatabaseManager('data/nanoprobe.db'); print('DB OK')"
   ```

3. **Проверьте Redis:**
   ```bash
   redis-cli ping
   ```

### Если Response Time > 1000ms

1. **Включите кэширование:**
   - Проверьте Redis подключение
   - Убедитесь, что кэш используется в endpoints

2. **Оптимизируйте запросы:**
   - Проверьте индексы БД
   - Используйте `EXPLAIN` для медленных запросов

3. **Увеличьте ресурсы:**
   - Добавьте RAM
   - Увеличьте CPU лимиты

### Если RPS < 10

1. **Проверьте Rate Limiting:**
   ```bash
   curl -i http://localhost:8000/health
   ```

2. **Проверьте конфигурацию Uvicorn:**
   ```bash
   # workers count
   uvicorn api.main:app --workers 4
   ```

3. **Проверьте базу данных:**
   - Убедитесь, что индексы созданы
   - Проверьте connection pool

---

## 📁 Выходные файлы

### load_test_results.json

**Структура:**
```json
{
  "timestamp": "2026-03-14T15:30:00",
  "config": {
    "base_url": "http://localhost:8000",
    "users": 10,
    "duration": 60
  },
  "results": {
    "GET /health": {
      "endpoint": "/health",
      "method": "GET",
      "total_requests": 120,
      "successful_requests": 120,
      "failed_requests": 0,
      "success_rate": 100.0,
      "avg_response_time": 25.8,
      "p95_response_time": 38.5,
      "requests_per_second": 4.0
    }
  }
}
```

---

## 🔗 Дополнительные ресурсы

- [Locust.io](https://locust.io/) - Продвинутый load testing инструмент
- [Apache JMeter](https://jmeter.apache.org/) - Load testing инструмент
- [k6.io](https://k6.io/) - Modern load testing

---

## 📝 Примеры отчётов

### Отличный результат

```
📈 Общая статистика:
   Всего запросов: 600
   Успешно: 600
   Ошибок: 0
   Success Rate: 100.0%
   Средний RPS: 10.00 запросов/сек
   Среднее время ответа: 25.50ms

💡 Рекомендации:
   ✅ Все метрики в норме!
```

### Требуется оптимизация

```
📈 Общая статистика:
   Всего запросов: 300
   Успешно: 270
   Ошибок: 30
   Success Rate: 90.0%
   Средний RPS: 5.00 запросов/сек
   Среднее время ответа: 1250.00ms

💡 Рекомендации:
   ⚠️  Success Rate ниже 95% - проверьте логи ошибок
   ⚠️  Среднее время ответа > 1с - рассмотрите кэширование
```

---

**Last Updated:** 2026-03-14
**Version:** 1.0.0
