# Utils Reorganization Guide

## Реорганизация утилит проекта

**Статус:** ✅ Выполнено (2026-03-15)

---

## 🎯 Цели реорганизации

### Проблемы до реорганизации:
- ❌ 65+ файлов в корне utils/
- ❌ Сложно найти нужный модуль
- ❌ Дублирование функциональности
- ❌ Загрязнение глобального пространства имён

### После реорганизации:
- ✅ Модульная структура по назначению
- ✅ Чёткое разделение ответственности
- ✅ Удобная навигация
- ✅ Избежание конфликтов имён

---

## 📁 Новая структура

```
utils/
├── __init__.py                 # Главный пакет с экспортом
│
├── core/                       # Базовые утилиты
│   ├── __init__.py
│   ├── cli_utils.py           # CLI helpers
│   └── error_handler.py       # Error handlers
│
├── api/                        # API клиенты
│   ├── __init__.py
│   ├── nasa_api_client.py     # NASA API client
│   └── space_image_downloader.py
│
├── database/                   # Database utilities
│   ├── __init__.py
│   └── database.py            # DatabaseManager
│
├── security/                   # Security
│   ├── __init__.py
│   ├── rate_limiter.py        # Rate limiting
│   └── two_factor_auth.py     # 2FA
│
├── caching/                    # Caching
│   ├── __init__.py
│   ├── redis_cache.py         # Redis cache
│   ├── cache_manager.py       # Cache manager
│   └── circuit_breaker.py     # Circuit breaker
│
├── monitoring/                 # Monitoring
│   ├── __init__.py
│   ├── system_monitor.py      # System monitoring
│   ├── enhanced_monitor.py    # Enhanced monitor
│   ├── system_health_monitor.py
│   ├── performance_monitor.py
│   └── realtime_dashboard.py
│
├── performance/                # Performance optimization
│   ├── __init__.py
│   ├── performance_profiler.py
│   ├── performance_benchmark.py
│   ├── memory_tracker.py
│   ├── resource_optimizer.py
│   └── optimization_orchestrator.py
│
├── batch/                      # Batch processing
│   ├── __init__.py
│   └── batch_processor.py
│
├── logging/                    # Logging utilities
│   ├── __init__.py
│   ├── logger.py
│   ├── production_logger.py
│   └── advanced_logger_analyzer.py
│
├── visualization/              # Visualization
│   ├── __init__.py
│   ├── visualizer.py
│   ├── analytics.py
│   └── spm_realtime_visualizer.py
│
├── simulator/                  # Simulator orchestration
│   ├── __init__.py
│   └── simulator_orchestrator.py
│
├── testing/                    # Testing frameworks
│   ├── __init__.py
│   └── test_framework.py
│
├── dev/                        # Development tools
│   ├── __init__.py
│   └── code_analyzer.py
│
├── ai/                         # AI/ML (уже была)
│   ├── __init__.py
│   ├── machine_learning.py
│   ├── model_trainer.py
│   └── defect_analyzer.py
│
├── config/                     # Configuration (уже была)
│   ├── __init__.py
│   ├── config_manager.py
│   └── config_validator.py
│
├── data/                       # Data management (уже была)
│   ├── __init__.py
│   ├── data_manager.py
│   └── data_validator.py
│
├── reporting/                  # Reports (уже был)
│   ├── __init__.py
│   ├── report_generator.py
│   └── pdf_report_generator.py
│
└── deployment/                 # Deployment (уже был)
    ├── __init__.py
    └── deployment_manager.py
```

---

## 🔄 Миграция импортов

### До реорганизации:
```python
from utils.database import DatabaseManager
from utils.redis_cache import cache
from utils.rate_limiter import limiter
from utils.nasa_api_client import get_nasa_client
```

### После реорганизации:

**Вариант 1: Через main __init__.py (рекомендуется)**
```python
from utils import DatabaseManager, cache, limiter, get_nasa_client
```

**Вариант 2: Прямые импорты**
```python
from utils.database.database import DatabaseManager
from utils.caching.redis_cache import cache
from utils.security.rate_limiter import limiter
from utils.api.nasa_api_client import get_nasa_client
```

**Вариант 3: Из подмодулей**
```python
from utils.database import DatabaseManager
from utils.caching import cache
from utils.security import limiter
from utils.api import get_nasa_client
```

---

## 🚀 Автоматическая миграция

### Скрипт реорганизации:

```bash
# Dry run (показать что будет сделано)
python utils_reorganization.py

# Выполнить миграцию
python utils_reorganization.py --execute
```

### Этапы:
1. Создание директорий
2. Перемещение файлов
3. Создание __init__.py
4. Обновление импортов (требует ручного ревью)

---

## 📊 Статистика

| Метрика | До | После |
|---------|-----|-------|
| Файлов в корне utils/ | 65 | ~15 |
| Директорий | 7 | 16 |
| Средняя глубина вложенности | 1 | 2 |
| Импортов в коде | ~100 | ~100 (требуют обновления) |

---

## ✅ Checklist

### Созданные директории:
- [x] core/
- [x] api/
- [x] database/
- [x] security/ (обновлена)
- [x] caching/
- [x] monitoring/ (обновлена)
- [x] performance/ (обновлена)
- [x] batch/
- [x] logging/
- [x] visualization/
- [x] simulator/
- [x] testing/
- [x] dev/

### Созданные __init__.py:
- [x] utils/__init__.py (главный)
- [x] utils/core/__init__.py
- [x] utils/api/__init__.py
- [x] utils/database/__init__.py
- [x] utils/security/__init__.py
- [x] utils/caching/__init__.py
- [x] utils/monitoring/__init__.py

### Документация:
- [x] utils_reorganization.py (скрипт)
- [x] UTILS_REORGANIZATION.md (документация)

---

## 🔧 Обновление импортов в проекте

### Поиск и замена:

```bash
# Найти все импорты utils
grep -r "from utils\." --include="*.py" .

# Обновить импорты (примеры)
# Было:
from utils.database import DatabaseManager

# Стало:
from utils.database.database import DatabaseManager
# или
from utils import DatabaseManager
```

### Критичные файлы для обновления:
1. `api/main.py`
2. `api/routes/*.py`
3. `src/cli/*.py`
4. `tests/*.py`

---

## 🧪 Тестирование

После миграции:

```bash
# Запустить все тесты
pytest tests/ -v

# Проверить импорты
python -c "from utils import DatabaseManager; print('OK')"
python -c "from utils.caching import cache; print('OK')"
python -c "from utils.api import get_nasa_client; print('OK')"
```

---

## 📝 Рекомендации

### Для новых модулей:
1. Создавать файл в соответствующей директории
2. Добавлять экспорт в __init__.py
3. Следовать naming conventions

### Для импортов:
1. Использовать импорты из главного __init__.py
2. Избегать глубокой вложенности
3. Группировать импорты по типу

---

## 🔗 Связанные документы

- `CONSOLIDATION_GUIDE.md` — общая консолидация проекта
- `SECURITY_DASHBOARD_IMPROVEMENTS.md` — security улучшения
- `NASA_API_INTEGRATION.md` — NASA API интеграция

---

*Обновлено: 2026-03-15*
