# Документация разработчика Nanoprobe Sim Lab

**Версия:** 1.0.0  
**Дата:** 2026-03-11  
**Для кого:** Разработчики, программисты

---

## 📖 Содержание

1. [Архитектура проекта](#архитектура-проекта)
2. [Структура кода](#структура-кода)
3. [Настройка окружения](#настройка-окружения)
4. [Запуск проекта](#запуск-проекта)
5. [Тестирование](#тестирование)
6. [Вклад в проект](#вклад-в-проект)
7. [Стиль кода](#стиль-кода)
8. [Документирование](#документирование)

---

## 🏗️ Архитектура проекта

### Компоненты

```
┌────────────────────────────────────────────────────────┐
│                     Frontend                           │
│  ┌────────────────┐  ┌────────────────────────┐      │
│  │  Flask Web     │  │  HTML Templates        │      │
│  │  (порт 5000)   │  │  Jinja2                │      │
│  └────────────────┘  └────────────────────────┘      │
└────────────────────────────────────────────────────────┘
                        ↓ HTTP
┌────────────────────────────────────────────────────────┐
│                     API Layer                          │
│  ┌────────────────────────────────────────────────┐   │
│  │  FastAPI REST API (порт 8000)                  │   │
│  │  - Authentication (JWT)                        │   │
│  │  - CRUD Operations                             │   │
│  │  - WebSocket Real-time                         │   │
│  └────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────┐
│                  Business Logic                        │
│  ┌────────────────────────────────────────────────┐   │
│  │  utils/*.py (40+ модулей)                      │   │
│  │  - Database Manager (SQLite)                   │   │
│  │  - Defect Analyzer (ML)                        │   │
│  │  - PDF Report Generator                        │   │
│  │  - Surface Comparator                          │   │
│  │  - Batch Processor                             │   │
│  └────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────┐
│                    Data Layer                          │
│  ┌──────────────┐  ┌──────────────┐                  │
│  │  SQLite DB   │  │  File System │                  │
│  │  nanoprobe   │  │  Images,     │                  │
│  │  .db         │  │  Reports     │                  │
│  └──────────────┘  └──────────────┘                  │
└────────────────────────────────────────────────────────┘
```

### Технологический стек

**Backend:**
- Python 3.8+
- FastAPI (REST API)
- Flask (Web UI)
- SQLite (База данных)
- Pydantic (Валидация)

**ML/Data:**
- NumPy, Pandas
- Scikit-learn
- OpenCV, Pillow
- ReportLab (PDF)

**DevOps:**
- Docker, Docker Compose
- GitHub Actions (CI/CD)
- pytest (Тесты)

---

## 📁 Структура кода

```
nanoprobe-sim-lab/
├── api/                          # FastAPI REST API
│   ├── main.py                   # Главное приложение
│   ├── schemas.py                # Pydantic схемы
│   ├── README.md                 # Документация API
│   └── routes/                   # API роуты
│       ├── auth.py               # Аутентификация
│       ├── scans.py              # Сканирования
│       ├── simulations.py        # Симуляции
│       ├── analysis.py           # Анализ дефектов
│       ├── comparison.py         # Сравнение
│       ├── reports.py            # Отчёты
│       └── admin.py              # Администрирование
│
├── src/                          # Исходный код
│   ├── cli/                      # Консольные утилиты
│   │   ├── main.py               # Главная консоль
│   │   └── project_manager.py    # Менеджер проекта
│   └── web/                      # Веб-интерфейс
│       └── web_dashboard.py      # Flask веб-панель
│
├── utils/                        # Общие утилиты
│   ├── database.py               # Менеджер БД
│   ├── defect_analyzer.py        # AI/ML анализ
│   ├── pdf_report_generator.py   # PDF отчёты
│   ├── surface_comparator.py     # Сравнение поверхностей
│   ├── batch_processor.py        # Пакетная обработка
│   └── ...                       # Другие утилиты
│
├── tests/                        # Тесты
│   ├── test_api.py               # Тесты API
│   └── test_utils.py             # Тесты утилит
│
├── docs/                         # Документация
│   ├── API.md                    # API документация
│   ├── ADMIN.md                  # Для администратора
│   └── DEVELOPER.md              # Для разработчиков
│
├── components/                   # C++ компоненты
│   └── cpp-spm-hardware-sim/
│
├── config/                       # Конфигурация
├── data/                         # База данных
├── logs/                         # Логи
├── output/                       # Результаты
└── reports/                      # PDF отчёты
```

---

## ⚙️ Настройка окружения

### 1. Виртуальное окружение

```bash
# Создание
python -m venv venv

# Активация (Windows)
venv\Scripts\activate

# Активация (Linux/macOS)
source venv/bin/activate
```

### 2. Установка зависимостей

```bash
# Основные зависимости
pip install -r requirements.txt

# API зависимости
pip install -r requirements-api.txt

# Dev зависимости
pip install -r requirements-dev.txt  # если есть
```

### 3. Переменные окружения

Создайте файл `.env`:

```bash
# JWT Secret
JWT_SECRET=your-super-secret-key-min-32-characters

# Database
DATABASE_PATH=data/nanoprobe.db

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=info

# CORS
CORS_ORIGINS=["http://localhost:3000","http://localhost:5000"]
```

### 4. Pre-commit хуки (опционально)

```bash
# Установка pre-commit
pip install pre-commit
pre-commit install

# Проверка всех файлов
pre-commit run --all-files
```

---

## 🚀 Запуск проекта

### Быстрый старт

```bash
# Запуск API (терминал 1)
python run_api.py --reload

# Запуск веб-интерфейса (терминал 2)
python start.py web

# Админ CLI
python admin_cli.py status
```

### Отладка

```bash
# Запуск с отладкой
python -m pdb run_api.py

# Профилирование
python -m cProfile -o profile.stats run_api.py

# Мониторинг памяти
python -m memory_profiler run_api.py
```

---

## 🧪 Тестирование

### Запуск тестов

```bash
# Все тесты
pytest tests/ -v

# Только API тесты
pytest tests/test_api.py -v

# С покрытием
pytest tests/ --cov=. --cov-report=html

# Конкретный тест
pytest tests/test_api.py::TestAuth::test_login_success -v
```

### Написание тестов

**Пример теста:**
```python
# tests/test_example.py
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_async_example():
    await asyncio.sleep(1)
    assert True
```

### Покрытие кода

Целевое покрытие: **>80%**

```bash
# Проверка покрытия
coverage run -m pytest tests/
coverage report
coverage html

# Открыть отчёт
start htmlcov/index.html  # Windows
open htmlcov/index.html   # macOS
```

---

## 🤝 Вклад в проект

### Ветвление Git

```
main (production)
  └── dev (development)
        ├── feature/add-new-feature
        ├── bugfix/fix-issue-123
        └── hotfix/urgent-fix
```

### Создание Pull Request

1. Форкните репозиторий
2. Создайте ветку (`git checkout -b feature/amazing-feature`)
3. Закоммитьте изменения (`git commit -m 'Add amazing feature'`)
4. Запушьте (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

### Conventional Commits

```
feat: добавление новой функции
fix: исправление ошибки
docs: обновление документации
style: форматирование кода
refactor: рефакторинг кода
test: добавление тестов
chore: обновление зависимостей
```

**Примеры:**
```bash
git commit -m "feat: добавить JWT аутентификацию"
git commit -m "fix: исправить утечку памяти в database.py"
git commit -m "docs: обновить README.md"
```

---

## 📝 Стиль кода

### Основные правила

**Именование:**
```python
# Классы - PascalCase
class DatabaseManager:
    pass

# Функции - snake_case
def get_scan_results():
    pass

# Константы - UPPER_SNAKE_CASE
MAX_CONNECTIONS = 100

# Приватные методы - _prefix
def _internal_helper():
    pass
```

**Docstrings:**
```python
def calculate_statistics(data: np.ndarray) -> Dict[str, float]:
    """
    Расчёт статистики данных.

    Args:
        data: Массив данных для анализа

    Returns:
        Словарь со статистикой (mean, std, min, max)

    Example:
        >>> data = np.array([1, 2, 3, 4, 5])
        >>> calculate_statistics(data)
        {'mean': 3.0, 'std': 1.41, ...}
    """
    pass
```

**Аннотации типов:**
```python
from typing import List, Dict, Optional, Union

def process_items(
    items: List[str],
    limit: int = 100,
    callback: Optional[callable] = None
) -> Dict[str, Union[int, str]]:
    pass
```

### Линтеры

```bash
# Black (форматирование)
black src/ utils/ api/

# Flake8 (стиль)
flake8 src/ utils/ api/ --max-line-length=100

# MyPy (типы)
mypy src/ utils/ api/ --ignore-missing-imports

# Все сразу
python format_code.py
```

---

## 📚 Документирование

### Встроенная документация

**Docstring формат:**
```python
class SurfaceAnalyzer:
    """
    Анализатор поверхностей для СЗМ данных.

    Предоставляет методы для статистического анализа,
    обнаружения дефектов и сравнения поверхностей.

    Attributes:
        model_name: Название ML модели
        threshold: Порог детектирования
    """

    def analyze(self, surface: np.ndarray) -> Dict:
        """
        Анализ поверхности.

        Args:
            surface: 2D массив данных поверхности

        Returns:
            Словарь с результатами анализа

        Raises:
            ValueError: Если поверхность пустая
        """
        pass
```

### Генерация документации

```bash
# Sphinx (если настроен)
sphinx-apidoc -o docs/source api/
cd docs && make html

# Pydoc (простой вариант)
pydoc -w api.main
```

---

## 🔧 Полезные скрипты

### format_code.py
```bash
python format_code.py  # Форматирование кода
```

### validate_project.py
```bash
python validate_project.py  # Проверка проекта
```

### admin_cli.py
```bash
python admin_cli.py status    # Статус системы
python admin_cli.py backup    # Бэкап
python admin_cli.py users     # Пользователи
```

---

## 🐛 Отладка

### Логирование

```python
from utils.logger import get_logger

logger = get_logger(__name__)

logger.debug("Отладочное сообщение")
logger.info("Информация")
logger.warning("Предупреждение")
logger.error("Ошибка")
logger.critical("Критическая ошибка")
```

### Отладка API

```bash
# Включить debug логи
uvicorn api.main:app --reload --log-level debug

# Просмотр запросов
tail -f logs/api.log | grep "REQUEST"
```

### Профилирование

```python
# В коде
import cProfile
import pstats

def profile_function():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Ваш код
    result = some_function()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
```

---

## 📊 Метрики кода

### Целевые показатели

| Метрика | Значение |
|---------|----------|
| Покрытие тестами | >80% |
| Сложность функции | <10 |
| Длина функции | <50 строк |
| Длина файла | <500 строк |

### Проверка

```bash
# Радон (сложность)
pip install radon
radon cc src/ -a -s
radon mi src/

# Покрытие
pytest --cov=. --cov-report=term-missing
```

---

## 🎯 Roadmap разработки

### Ближайшие задачи

- [ ] Интеграция с Redis для кэширования
- [ ] Celery для фоновых задач
- [ ] GraphQL API
- [ ] Real-time дашборд на React
- [ ] Улучшение ML моделей

### Долгосрочные цели

- Микросервисная архитектура
- Kubernetes оркестрация
- Полное покрытие тестами
- Международная поддержка

---

## 📞 Контакты

**Вопросы и предложения:**
- Email: maksimqwe42@mail.ru
- GitHub Issues: https://github.com/your-username/nanoprobe-sim-lab/issues

---

*Документация разработчика v1.0.0*  
*Последнее обновление: 2026-03-11*
