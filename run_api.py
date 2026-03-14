#!/usr/bin/env python
"""
Скрипт запуска FastAPI API для Nanoprobe Sim Lab
"""

import uvicorn
import argparse
import sys
from pathlib import Path

# Установка UTF-8 кодировки для Windows
if sys.platform == "win32":
    import os
    os.system("chcp 65001 >nul")
    # Перенастройка stdout для UTF-8
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(
        description="Запуск FastAPI API для Nanoprobe Sim Lab"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Хост для прослушивания (по умолчанию: 0.0.0.0)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Порт для прослушивания (по умолчанию: 8000)",
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

    print("=" * 60)
    print("🚀 Nanoprobe Sim Lab API")
    print("=" * 60)
    print(f"📍 Хост: {args.host}")
    print(f"🔌 Порт: {args.port}")
    print(f"📊 Документация: http://{args.host}:{args.port}/docs")
    print(f"📖 ReDoc: http://{args.host}:{args.port}/redoc")
    print(f"❤️  Health: http://{args.host}:{args.port}/health")
    print("=" * 60)

    # Конфигурация uvicorn
    config = uvicorn.Config(
        app="api.main:app",
        host=args.host,
        port=args.port,
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
