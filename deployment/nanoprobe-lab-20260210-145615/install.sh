#!/bin/bash
# Установочный скрипт для Nanoprobe Simulation Lab

echo "Установка Nanoprobe Simulation Lab..."

# Проверка зависимостей
if ! command -v python3 &> /dev/null; then
    echo "Python 3 не найден. Установите Python 3.8+"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "Docker не найден. Установите Docker"
    exit 1
fi

# Создание виртуального окружения
python3 -m venv venv
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt

# Сборка Docker образа
docker build -t nanoprobe-lab .

echo "Установка завершена!"
echo "Для запуска используйте: docker-compose up -d"
