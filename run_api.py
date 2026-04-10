#!/usr/bin/env python
"""
Скрипт запуска FastAPI API для Nanoprobe Sim Lab

Требования:
- Python 3.11, 3.12, 3.13, or 3.14
"""

# Проверка версии Python (требуется 3.11 - 3.14)
import sys

MIN_PYTHON_VERSION = (3, 11)
MAX_PYTHON_VERSION = (3, 14)
if sys.version_info < MIN_PYTHON_VERSION or sys.version_info >= (
    MAX_PYTHON_VERSION[0],
    MAX_PYTHON_VERSION[1] + 1,
):
    print(f"[ERROR] Требуется Python 3.11 - 3.14, текущая версия: {sys.version}")
    print(f"Путь к Python: {sys.executable}")
    print("Установите Python 3.11 - 3.14 с https://www.python.org/downloads/")
    sys.exit(1)

import argparse
import os
from pathlib import Path

import uvicorn

# Установка UTF-8 кодировки для Windows
if sys.platform == "win32":
    os.system("chcp 65001 >nul")
    # Перенастройка stdout для UTF-8
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        # Python < 3.7
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description="Запуск FastAPI API для Nanoprobe Sim Lab")

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Хост для прослушивания (по умолчанию: 0.0.0.0)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=None,  # None для автоопределения
        help="Порт для прослушивания (по умолчанию: автоопределение)",
    )

    parser.add_argument(
        "--auto-port",
        action="store_true",
        default=True,  # Включено по умолчанию
        help="Автоматический выбор свободного порта (по умолчанию: True)",
    )

    parser.add_argument(
        "--no-auto-port",
        action="store_true",
        help="Отключить автоматический выбор порта",
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        help="Автоматическая перезагрузка при изменении файлов",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Уровень логирования",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Количество worker процессов (для production)",
    )

    args = parser.parse_args()

    # Создание необходимых директорий
    Path("data").mkdir(exist_ok=True)
    Path("reports/pdf").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # Автоматическое определение порта
    use_auto_port = args.auto_port and not args.no_auto_port

    if args.port is not None:
        # Порт задан явно
        port = args.port
        print(f"📌 Порт задан вручную: {port}")
    elif use_auto_port:
        # Автоопределение порта
        try:
            from utils.port_finder import find_port

            preferred_port = int(os.getenv("BACKEND_PORT", 8000))
            port = find_port("backend", preferred_port)

            if port != preferred_port:
                print(f"⚠️  Порт {preferred_port} занят, выбран: {port}")
            else:
                print(f"✅ Порт: {port}")

            # Сохраняем выбранный порт в переменную окружения
            os.environ["BACKEND_PORT"] = str(port)
        except Exception as e:
            print(f"⚠️  Автоопределение порта не удалось: {e}")
            print(f"📌 Используем порт по умолчанию: 8000")
            port = 8000
    else:
        # Порт не задан и автоопределение отключено
        port = int(os.getenv("BACKEND_PORT", 8000))
        print(f"📌 Порт из BACKEND_PORT: {port}")

    print("=" * 60)
    print("🚀 Nanoprobe Sim Lab API")
    print("=" * 60)
    print(f"📍 Хост: {args.host}")
    print(f"🔌 Порт: {port}")
    print(f"📊 Документация: http://{args.host}:{port}/docs")
    print(f"📖 ReDoc: http://{args.host}:{port}/redoc")
    print(f"❤️  Health: http://{args.host}:{port}/health")
    print("=" * 60)

    # Конфигурация uvicorn
    config = uvicorn.Config(
        app="api.main:app",
        host=args.host,
        port=port,
        reload=args.reload,
        log_level=args.log_level,
        workers=args.workers if not args.reload else 1,
    )

    server = uvicorn.Server(config)

    try:
        server.run()
    except KeyboardInterrupt:
        print("\n👋 Остановка сервера...")
        sys.exit(0)


if __name__ == "__main__":
    main()
