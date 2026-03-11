# Dockerfile для Лаборатории моделирования нанозонда
# Версия: 2.1 с multi-stage build и оптимизацией
# Stage 1: Build
FROM python:3.11-slim as builder

# Установка системных зависимостей для компиляции
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

WORKDIR /app

# Копирование файлов зависимостей
COPY requirements.txt pyproject.toml ./

# Установка зависимостей в отдельную директорию
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim as runtime

# Метаданные
LABEL maintainer="Школа программирования Maestro7IT <maksimqwe42@mail.ru>"
LABEL version="2.1.0"
LABEL description="Nanoprobe Simulation Lab - комплексный научно-образовательный проект"
LABEL org.label-schema.schema-version="1.0"
LABEL org.label-schema.name="nanoprobe-sim-lab"
LABEL org.label-schema.version="2.1.0"

# Установка минимальных runtime зависимостей
RUN apt-get update && apt-get install -y \
    libhdf5 \
    libatlas3-base \
    libopenblas0 \
    libffi8 \
    libssl3 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Создание пользователя без root прав
RUN groupadd -r nanoprobe && useradd -r -g nanoprobe nanoprobe

WORKDIR /app

# Копирование зависимостей из builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Копирование исходного кода
COPY . .

# Создание необходимых директорий
RUN mkdir -p data logs output config reports /app/data && \
    chown -R nanoprobe:nanoprobe /app

# Переключение на пользователя без root прав
USER nanoprobe

# Инициализация базы данных
RUN python -c "from utils.database import DatabaseManager; db = DatabaseManager('data/nanoprobe.db')" 2>/dev/null || true

# Открытие портов
EXPOSE 5000 8000 8080

# Переменные окружения
ENV PYTHONPATH=/app
ENV FLASK_ENV=production
ENV DATABASE_PATH=/app/data/nanoprobe.db
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/api/status', timeout=5)" || exit 1

# Команда по умолчанию
CMD ["python", "start.py", "web"]
