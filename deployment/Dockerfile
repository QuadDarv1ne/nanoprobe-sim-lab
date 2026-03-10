# Dockerfile для Лаборатории моделирования нанозонда
# Версия: 2.0 с поддержкой всех компонентов
FROM python:3.11-slim

# Метаданные
LABEL maintainer="Школа программирования Maestro7IT <maksimqwe42@mail.ru>"
LABEL version="1.0.0"
LABEL description="Nanoprobe Simulation Lab - комплексный научно-образовательный проект"

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    libhdf5-dev \
    libatlas-base-dev \
    libopenblas-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копирование файлов зависимостей
COPY requirements.txt .
COPY pyproject.toml .

# Установка Python зависимостей
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Создание необходимых директорий
RUN mkdir -p data logs output config reports

# Создание .env файла из примера
RUN if [ -f .env.example ] && [ ! -f .env ]; then cp .env.example .env; fi

# Инициализация базы данных
RUN python -c "from utils.database import DatabaseManager; db = DatabaseManager('data/nanoprobe.db')" 2>/dev/null || true

# Открытие портов
EXPOSE 5000 8000 8080

# Переменные окружения
ENV PYTHONPATH=/app
ENV FLASK_ENV=production
ENV DATABASE_PATH=/app/data/nanoprobe.db

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/api/status', timeout=5)" || exit 1

# Команда по умолчанию
CMD ["python", "start.py", "web"]
