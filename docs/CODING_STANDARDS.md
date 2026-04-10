# Стандарты кодирования Nanoprobe Sim Lab

## Типизация

### Обязательная типизация
Все публичные функции и методы должны иметь type hints:

```python
# ✅ Правильно
def process_samples(samples: np.ndarray, sample_rate: float) -> Dict[str, Any]:
    """Обработка сэмплов с возвратом метаданных."""
    ...

# ❌ Неправильно
def process_samples(samples, sample_rate):
    ...
```

### py.typed marker
Проект поддерживает PEP 561 type checking:
- Файл `utils/py.typed` указывает на поддержку типов
- Все публичные модули должны экспортировать типы через `__all__`

## Импорты

### Порядок импортов
1. Стандартная библиотека
2. Сторонние библиотеки
3. Локальные импорты

```python
# Стандартная библиотека
import os
from pathlib import Path
from typing import Dict, List, Optional

# Сторонние
import numpy as np
from fastapi import APIRouter

# Локальные
from utils.database import Database
from api.schemas import ScanCreate
```

### Избегание циклических импортов
- Выносите общие типы в `types.py`
- Используйте dependency injection
- Lazy imports только когда необходимо:

```python
def get_database() -> Database:
    """Lazy import для избежания циклических зависимостей."""
    from utils.database import Database
    return Database()
```

### Проверка на циклические импорты
```bash
python scripts/check_cyclic_imports.py
```

## datetime

### Обязательное использование timezone-aware datetime
```python
# ✅ Правильно
from datetime import datetime, timezone

timestamp = datetime.now(timezone.utc)

# ❌ Неправильно
timestamp = datetime.now()  # Naive datetime
```

## Логирование

### Использование структурного логирования
```python
import logging

logger = logging.getLogger(__name__)

# ✅ Правильно
logger.info("Scan started", extra={"scan_id": scan_id, "user_id": user_id})

# ❌ Неправильно
print(f"Scan {scan_id} started by user {user_id}")
```

## Исключения

### Кастомные исключения
```python
class NanoprobeError(Exception):
    """Базовое исключение для проекта."""
    pass

class DeviceNotFoundError(NanoprobeError):
    """Устройство не найдено."""
    pass
```

### Обработка ошибок
```python
# ✅ Правильно
try:
    result = await process_data(data)
except ValidationError as e:
    logger.error("Validation failed", extra={"errors": e.errors})
    raise HTTPException(status_code=400, detail=str(e))
except Exception as e:
    logger.exception("Unexpected error during processing")
    raise HTTPException(status_code=500, detail="Internal server error")
```

## Тестирование

### Именование тестов
```python
def test_process_samples_returns_valid_spectrum():
    """Тест обработки сэмплов."""
    pass

def test_database_insert_creates_record():
    """Тест вставки в БД."""
    pass
```

### Минимальное покрытие
- Цель: 80%+ coverage
- Критичные модули: 95%+ coverage

## CI/CD

### Pre-commit hooks
```bash
pre-commit install
pre-commit run --all-files
```

### Локальные проверки
```bash
make lint          # flake8
make format        # black
make type-check    # mypy
make test          # pytest
make check-all     # всё вышеперечисленное
```

## Документация

### Docstrings
```python
def calculate_fft(
    samples: np.ndarray,
    sample_rate: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вычисление FFT с Hann-окном.

    Args:
        samples: Входные сэмплы
        sample_rate: Частота дискретизации в Гц

    Returns:
        Tuple[frequencies, powers]: Частоты и мощности

    Raises:
        ValueError: Если samples пустой
    """
    ...
```

## Производительность

### Избегание утечек памяти
- Используйте circular buffers для потоковых данных
- Ограничивайте размер кэшей
- Закрывайте соединения и процессы

```python
# ✅ Правильно
class WaterfallDisplay:
    def __init__(self, max_frames: int = 36000):  # 1 час @ 10 fps
        self.buffer = np.zeros((max_frames, width), dtype=np.float32)

# ❌ Неправильно
class WaterfallDisplay:
    def __init__(self):
        self.frames = []  # Будет расти бесконечно!
```

## Безопасность

### Секреты
- Никогда не коммитьте секреты в git
- Используйте `.env` файлы
- Ротируйте ключи каждые 90 дней

### Валидация ввода
- Всегда валидируйте пользовательский ввод
- Используйте Pydantic для схем
- Санитизируйте пути к файлам
