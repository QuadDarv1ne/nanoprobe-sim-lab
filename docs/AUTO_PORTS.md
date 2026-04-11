# Руководство по автоопределению портов

**Дата:** 2026-04-08
**Версия:** 1.0.0

---

## Обзор

Система автоматического определения свободных портов позволяет запускать несколько экземпляров проекта без конфликтов портов.

---

## Как это работает

### 1. Утилита `utils/port_finder.py`

```python
from utils.port_finder import find_port, find_ports

# Найти порт для одного сервиса
backend_port = find_port("backend", preferred=8000)

# Найти порты для нескольких сервисов
ports = find_ports(["backend", "flask", "nextjs"])
# {"backend": 8000, "flask": 5000, "nextjs": 3000}
```

### 2. Приоритет определения порта

1. **Явно заданный порт** (`--port 8001`)
2. **Переменная окружения** (`BACKEND_PORT=8001`)
3. **Автоопределение** (поиск свободного порта в диапазоне)
4. **Порт по умолчанию** (8000, 5000, 3000)

### 3. Диапазоны поиска

| Сервис | Приоритетный | Fallback диапазон |
|--------|--------------|-------------------|
| Backend (FastAPI) | 8000 | 8000-8049, 8100-8149 |
| Flask Frontend | 5000 | 5000-5049, 5100-5149 |
| Next.js Frontend | 3000 | 3000-3049, 3100-3149 |

---

## Использование

### Запуск через main.py

```bash
# Автоматическое определение портов (по умолчанию)
python main.py flask

# Только Backend с автоопределением
python main.py api-only

# Backend + Next.js
python main.py nextjs
```

### Запуск API напрямую

```bash
# Автоопределение порта (по умолчанию)
python run_api.py

# Задать порт вручную
python run_api.py --port 8001

# Отключить автоопределение
python run_api.py --no-auto-port

# Комбинация
python run_api.py --port 8002 --reload
```

### Запуск Flask Dashboard

```bash
# Автоопределение порта (по умолчанию)
python src/web/web_dashboard_unified.py

# Задать порт вручную
python src/web/web_dashboard_unified.py --port 5001

# Отключить автоопределение
python src/web/web_dashboard_unified.py --no-auto-port
```

---

## Переменные окружения

Добавьте в `.env`:

```bash
# Фиксированные порты (отключает автоопределение для этих сервисов)
BACKEND_PORT=8000
FLASK_PORT=5000
NEXTJS_PORT=3000
```

---

## Программное использование

### Пример 1: Проверка доступности порта

```python
from utils.port_finder import PortFinder

finder = PortFinder()

# Проверить конкретный порт
if finder.is_port_available(8000):
    print("Порт 8000 свободен")
else:
    print("Порт 8000 занят")
```

### Пример 2: Предложение нескольких портов

```python
from utils.port_finder import get_port_finder

finder = get_port_finder()

# Получить 5 свободных портов для backend
suggestions = finder.suggest_ports("backend", count=5)
print(f"Доступные порты: {suggestions}")
```

### Пример 3: Статус порта

```python
from utils.port_finder import get_port_finder

finder = get_port_finder()

status = finder.get_port_status(8000)
# {"port": 8000, "available": True, "service": "backend", "host": "127.0.0.1"}
```

---

## Демо режим

```bash
# Запустить демо утилиты
python utils/port_finder.py
```

Вывод:
```
============================================================
Auto Port Finder - Демо
============================================================

✅ Найденные порты:
  backend        : 8000
  flask          : 5000
  nextjs         : 3000

📊 Статус стандартных портов:
  ✅ backend        : 8000 (свободен)
  ✅ flask          : 5000 (свободен)
  ✅ nextjs         : 3000 (свободен)
  ✅ redis          : 6379 (свободен)

💡 Предложения для backend:
  [8000, 8001, 8002, 8003, 8004]
```

---

## Troubleshooting

### Проблема: Автоопределение не работает

**Решение:**
```bash
# Проверить что port_finder импортируется
python -c "from utils.port_finder import find_port; print('OK')"

# Проверить порты вручную
netstat -ano | findstr ":8000"
```

### Проблема: Конфликт портов

**Решение 1:** Задать порт вручную
```bash
python run_api.py --port 8001
```

**Решение 2:** Установить переменную окружения
```bash
# Windows
set BACKEND_PORT=8001
python run_api.py

# Linux/Mac
BACKEND_PORT=8001 python run_api.py
```

---

## API Reference

### `PortFinder.is_port_available(port: int, host: str) -> bool`
Проверяет доступность конкретного порта.

### `PortFinder.find_available_port(service_name: str, preferred_port: int, host: str) -> int`
Находит свободный порт для сервиса.

### `PortFinder.find_multiple_ports(services: List[str], host: str) -> dict`
Находит порты для нескольких сервисов одновременно.

### `PortFinder.get_port_status(port: int, host: str) -> dict`
Возвращает статус порта.

### `PortFinder.suggest_ports(service_name: str, count: int, host: str) -> List[int]`
Предлагает несколько свободных портов.

---

## Интеграция

### FastAPI (run_api.py)
- `--auto-port` - включить автоопределение (по умолчанию)
- `--no-auto-port` - отключить автоопределение
- `--port N` - задать порт вручную

### Flask (web_dashboard_unified.py)
- `--auto-port` - включить автоопределение (по умолчанию)
- `--no-auto-port` - отключить автоопределение
- `--port N` - задать порт вручную

### Universal Launcher (main.py)
- Автоматическое определение для всех сервисов
- Обновление переменных окружения `BACKEND_PORT`, `FLASK_PORT`, `NEXTJS_PORT`
