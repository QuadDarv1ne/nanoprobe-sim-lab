# Инструкция по установке и запуску Nanoprobe Sim Lab

## 📋 Требования

- Python 3.8+
- Windows/Linux/macOS
- 4GB+ RAM (рекомендуется 8GB)
- 2GB+ свободного места

---

## 🚀 Быстрый старт

### 1. Клонирование репозитория

```bash
git clone https://github.com/your-username/nanoprobe-sim-lab.git
cd nanoprobe-sim-lab
```

### 2. Создание виртуального окружения

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Установка зависимостей

```bash
# Основные зависимости
pip install -r requirements.txt

# Зависимости для API (опционально)
pip install -r requirements-api.txt
```

### 4. Проверка установки

```bash
# Запуск главной консоли
python start.py cli

# Или запуск API
python run_api.py --reload
```

---

## 📦 Варианты запуска

### Вариант 1: Главная консоль (CLI)

```bash
python start.py cli
```

**Доступные команды:**
- Симулятор СЗМ
- Анализатор изображений
- Наземная станция SSTV
- Очистка кэша

### Вариант 2: Менеджер проекта

```bash
python start.py manager
```

**Команды менеджера:**
```bash
# Запустить симулятор СЗМ (Python)
python start.py manager spm-python

# Запустить симулятор СЗМ (C++)
python start.py manager spm-cpp

# Запустить анализатор изображений
python start.py manager analyzer

# Запустить наземную станцию SSTV
python start.py manager sstv

# Собрать C++ компоненты
python start.py manager build

# Очистить кэш проекта
python start.py manager clean-cache

# Показать информацию о проекте
python start.py manager info
```

### Вариант 3: Веб-панель управления

```bash
python start.py web
```

После запуска откройте в браузере: http://localhost:5000

### Вариант 4: FastAPI REST API ⭐ НОВОЕ!

```bash
# Установка API зависимостей
pip install -r requirements-api.txt

# Запуск API сервера
python run_api.py --reload
```

**Документация API:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 🐳 Запуск через Docker

### Требования

- Docker Desktop
- Docker Compose

### Запуск

```bash
# Сборка и запуск
docker-compose -f docker-compose.api.yml up -d

# Проверка статуса
docker-compose ps

# Просмотр логов
docker-compose logs -f api

# Остановка
docker-compose down
```

---

## 🧪 Тестирование

```bash
# Запуск тестов
pytest tests/ -v

# Тесты API
pytest tests/test_api.py -v

# С покрытием
pytest tests/ --cov=. --cov-report=html
```

---

## 🔧 Настройка окружения

### Переменные окружения (опционально)

Создайте файл `.env` в корне проекта:

```bash
# Скопируйте пример
cp .env.example .env

# Отредактируйте .env
# JWT_SECRET=your-secret-key
# DATABASE_PATH=data/nanoprobe.db
# LOG_LEVEL=info
```

---

## 📁 Структура директорий

После запуска создаются следующие директории:

```
nanoprobe-sim-lab/
├── data/           # База данных и файлы данных
├── logs/           # Логи приложения
├── output/         # Результаты сканирований
├── reports/        # PDF отчёты
│   └── pdf/
├── temp/           # Временные файлы
└── cache/          # Кэш
```

---

## 🛠️ Сборка C++ компонентов (опционально)

```bash
# Перейти в директорию компонента
cd components/cpp-spm-hardware-sim

# Создать директорию сборки
mkdir build && cd build

# Конфигурация CMake
cmake ..

# Сборка
make

# Или для Windows с Visual Studio
cmake --build . --config Release
```

---

## 📊 Мониторинг и диагностика

### Проверка здоровья API

```bash
curl http://localhost:8000/health
```

### Логи

```bash
# Просмотр логов API
tail -f logs/api.log

# Просмотр логов Flask
tail -f logs/flask.log
```

### Статистика БД

```bash
python -c "from utils.database import DatabaseManager; db = DatabaseManager(); print(db.get_statistics())"
```

---

## ❓ Решение проблем

### Ошибка: "ModuleNotFoundError"

```bash
# Переустановите зависимости
pip install -r requirements.txt --force-reinstall
```

### Ошибка: "Port already in use"

```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/macOS
lsof -i :8000
kill -9 <PID>
```

### Ошибка: "Database locked"

```bash
# Удалите WAL файлы
rm data/nanoprobe.db-wal data/nanoprobe.db-shm
```

### Ошибка при сборке C++

```bash
# Установите CMake и компилятор
# Windows: Visual Studio Build Tools
# Linux: sudo apt install build-essential cmake
# macOS: xcode-select --install
```

---

## 📞 Контакты

**Школа программирования Maestro7IT**
- Email: maksimqwe42@mail.ru
- Сайт: https://school-maestro7it.ru/

---

## 📚 Дополнительная документация

- [README.md](README.md) - Основная документация
- [api/README.md](api/README.md) - API документация
- [docs/API.md](docs/API.md) - Полная документация API
- [docs/](docs/) - Другая документация

---

*Последнее обновление: 2026-03-11*
