# 🚀 Onboarding Guide для новых разработчиков

Добро пожаловать в Nanoprobe Sim Lab! Этот гайд поможет вам быстро начать работу.

## 📋 Оглавление

1. [О проекте](#о-проекте)
2. [Первый запуск за 5 минут](#первый-запуск-за-5-минут)
3. [Архитектура проекта](#архитектура-проекта)
4. [Разработка](#разработка)
5. [Тестирование](#тестирование)
6. [SDR/SSTV модуль](#sdrsstv-модуль)
7. [Полезные ссылки](#полезные-ссылки)

---

## О проекте

**Nanoprobe Sim Lab** — это:
- 🔬 СЗМ (Сканирующая Зондовая Микроскопия) симулятор
- 🛰️ SSTV Ground Station для приёма изображений с МКС
- 📊 Анализатор поверхностей с AI/ML
- 🌐 Веб-дашборд для мониторинга

**Технологический стек:**
- Backend: FastAPI (Python 3.13+)
- Frontend: Flask + Socket.IO (в процессе миграции на Next.js)
- Database: SQLite (dev) / PostgreSQL (prod)
- Cache: Redis (опционально)
- SDR: RTL-SDR V4

---

## Первый запуск за 5 минут

### 1. Клонирование и установка

```bash
# Клонирование
git clone https://github.com/your-username/nanoprobe-sim-lab.git
cd nanoprobe-sim-lab

# Создание виртуального окружения
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Установка зависимостей
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Для разработки
```

### 2. Настройка окружения

```bash
# Копирование .env
cp .env.example .env

# Редактирование (опционально)
# .env уже содержит значения по умолчанию
```

### 3. Запуск

```bash
# Быстрый старт (Backend + Frontend)
python start_all.py

# Или отдельно:
# Backend (FastAPI)
python run_api.py

# Frontend (Flask)
python src/web/web_dashboard.py
```

### 4. Проверка

Откройте в браузере:
- Backend API: http://localhost:8000/docs (Swagger UI)
- Frontend Dashboard: http://localhost:5000

---

## Архитектура проекта

```
nanoprobe-sim-lab/
├── api/                    # FastAPI Backend
│   ├── routes/            # API эндпоинты
│   ├── sstv/              # SSTV API
│   ├── main.py            # Точка входа
│   └── middleware/        # Middleware
│
├── components/            # Научные компоненты
│   ├── py-sstv-groundstation/  # SSTV станция
│   ├── py-surface-image-analyzer/  # Анализ поверхностей
│   └── cpp-spm-hardware-sim/  # СЗМ симулятор (C++)
│
├── utils/                 # Утилиты
│   ├── database.py       # Работа с БД
│   ├── caching/          # Redis cache
│   ├── monitoring/       # Мониторинг
│   └── security/         # Безопасность
│
├── frontend/              # Frontend (Flask)
│   ├── templates/        # HTML шаблоны
│   └── static/           # CSS/JS
│
├── tests/                 # Тесты
├── docs/                  # Документация
└── scripts/               # Скрипты обслуживания
```

### Ключевые концепции

1. **Модульность**: Каждая научная функция — отдельный компонент
2. **Кэширование**: Redis для API ответов (stats: 5с, metrics: 1с)
3. **Миграции**: Alembic для управления схемой БД
4. **Мониторинг**: Health checks + Prometheus метрики

---

## Разработка

### Pre-commit hooks

```bash
# Установка
pre-commit install

# Ручной запуск
pre-commit run --all-files
```

### Качество кода

```bash
# Все проверки
make check-all

# Отдельно:
make lint        # flake8
make format      # black
make type-check  # mypy
make test        # pytest
```

### Добавление нового API эндпоинта

1. Создайте роут в `api/routes/your_module.py`:

```python
from fastapi import APIRouter

router = APIRouter(prefix="/your-module", tags=["Your Module"])

@router.get("/health")
async def health_check():
    return {"status": "ok"}
```

2. Зарегистрируйте в `api/main.py`:

```python
from api.routes.your_module import router as your_module_router

app.include_router(your_module_router)
```

### Работа с БД

```python
from utils.database import Database

db = Database("data/nanoprobe.db")

# Инициализация
db.initialize()

# Вставка
db.insert_scan({
    "user_id": 1,
    "status": "completed",
    "data": {"key": "value"}
})

# Запрос
results = db.query("SELECT * FROM scans WHERE status = ?", ("completed",))
```

---

## Тестирование

### Запуск тестов

```bash
# Все тесты
pytest tests/ -v

# С покрытием
pytest tests/ --cov=src --cov=utils --cov-report=html

# Конкретный тест
pytest tests/test_database.py -v

#_integration тесты с mock RTL-SDR
pytest tests/test_integration_rtlsdr.py -v
```

### Написание тестов

```python
def test_database_insert():
    """Тест вставки в БД."""
    db = Database(":memory:")
    db.initialize()

    result = db.insert_scan({"user_id": 1, "status": "test"})
    assert result is not None

    rows = db.query("SELECT * FROM scans")
    assert len(rows) == 1
```

### Mock RTL-SDR

Для тестов без физического устройства:

```python
from unittest.mock import patch, Mock

@patch('rtlsdr.RtlSdr')
def test_sstv_recording(mock_rtlsdr):
    mock_rtlsdr.get_device_count.return_value = 1
    # ... тест
```

---

## SDR/SSTV модуль

### Быстрый старт SSTV

```bash
# Проверка устройства
python main.py --check

# Приём SSTV с МКС
python main.py --realtime-sstv -f iss --duration 120

# Waterfall спектр
python main.py --waterfall -f 145.800
```

### API эндпоинты

```bash
# Health check
curl http://localhost:8000/api/v1/sstv/health

# Проверка устройства
curl http://localhost:8000/api/v1/sstv/device/check

# Расширенный health check
curl http://localhost:8000/api/v1/sstv/health/extended
```

### Структура SSTV компонента

```
components/py-sstv-groundstation/src/
├── sstv_decoder.py      # Декодер SSTV
├── sdr_interface.py     # RTL-SDR интерфейс
├── satellite_tracker.py # Трекинг спутников
├── waterfall_display.py # Waterfall дисплей
└── auto_recorder.py     # Автозапись
```

---

## Полезные ссылки

### Документация
- [API Reference](api_reference.md)
- [Coding Standards](CODING_STANDARDS.md)
- [Startup Guide](10-startup-guide.md)
- [RTL-SDR SSTV Recording](03-rtl-sdr-sstv-recording.md)

### Внешние ресурсы
- [RTL-SDR Blog](https://www.rtl-sdr.com/)
- [Celestrak TLE](https://celestrak.org/)
- [NASA API](https://api.nasa.gov/)
- [ISS SSTV Schedule](https://www.ariss.org/)

### Команда
- Владелец проекта: Максим Дуплей
- Email: maksimqwe72@mail.ru
- Telegram: @your_username

---

## FAQ

**Q: RTL-SDR устройство не определяется**
```bash
# Windows: проверьте Zadig драйверы
python check_zadig_drivers.bat

# Linux: проверьте права
sudo usermod -a -G plugdev $USER
```

**Q: Redis не запускается**
```bash
# Без Redis (используется in-memory fallback)
echo "REDIS_DISABLED=true" >> .env
```

**Q: Тесты падают из-за БД**
```bash
# Пересоздайте БД
rm data/nanoprobe.db
python -c "from utils.database import Database; Database('data/nanoprobe.db').initialize()"
```

---

**Следующие шаги:**
1. ✅ Прочитайте этот гайд
2. ✅ Запустите проект локально
3. ✅ Пройдите туториал [API Examples](api-examples.md)
4. 🚀 Начните вносить свой первый PR!
