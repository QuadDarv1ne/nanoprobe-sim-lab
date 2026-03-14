#!/usr/bin/env python
"""
Универсальный скрипт запуска Nanoprobe Sim Lab
Выбор версии frontend: Flask (v1.0) или Next.js (v2.0)

Использование:
    python start.py              # Интерактивный выбор
    python start.py flask        # Запуск с Flask frontend
    python start.py nextjs       # Запуск с Next.js frontend
    python start.py api-only     # Только Backend API
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

# UTF-8 для Windows
if sys.platform == "win32":
    os.system("chcp 65001 >nul")

PROJECT_ROOT = Path(__file__).parent
BACKEND_PORT = 8000
FLASK_PORT = 5000
NEXTJS_PORT = 3000


def print_banner():
    """Вывод заголовка"""
    print("=" * 70)
    print("  Nanoprobe Sim Lab - Универсальный запуск")
    print("=" * 70)
    print()


def print_versions():
    """Информация о версиях"""
    print("📌 Доступные версии frontend:")
    print()
    print("  1️⃣  Flask Dashboard (v1.0 - Legacy/Stable)")
    print("     • Порт: http://localhost:5000")
    print("     • Технологии: Flask + Jinja2 + Socket.IO")
    print("     • Файл: src/web/web_dashboard.py")
    print()
    print("  2️⃣  Next.js Dashboard (v2.0 - Modern/Production)")
    print("     • Порт: http://localhost:3000")
    print("     • Технологии: Next.js 14 + TypeScript + Tailwind CSS")
    print("     • Папка: frontend/")
    print()
    print("  3️⃣  Только Backend API")
    print("     • Порт: http://localhost:8000")
    print("     • Swagger: http://localhost:8000/docs")
    print()
    print("=" * 70)
    print()


def check_backend_port():
    """Проверка, занят ли порт backend"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        result = sock.connect_ex(('localhost', BACKEND_PORT))
        sock.close()
        return result == 0
    except (socket.error, OSError):
        return False


def start_backend():
    """Запуск Backend (FastAPI)"""
    print("🚀 Запуск Backend (FastAPI)...")
    print(f"   Порт: http://localhost:{BACKEND_PORT}")
    print(f"   Swagger: http://localhost:{BACKEND_PORT}/docs")
    print()
    
    backend_script = PROJECT_ROOT / "run_api.py"
    if not backend_script.exists():
        print(f"❌ Файл {backend_script} не найден!")
        return None
    
    process = subprocess.Popen(
        [sys.executable, str(backend_script), "--reload"],
        cwd=str(PROJECT_ROOT)
    )
    
    # Ожидание запуска
    time.sleep(3)
    
    if check_backend_port():
        print("✅ Backend запущен!")
        print()
        return process
    else:
        print("❌ Не удалось запустить Backend!")
        return None


def start_flask_frontend():
    """Запуск Flask frontend"""
    print("🚀 Запуск Flask Frontend...")
    print(f"   Порт: http://localhost:{FLASK_PORT}")
    print()
    
    flask_script = PROJECT_ROOT / "src" / "web" / "web_dashboard.py"
    if not flask_script.exists():
        print(f"❌ Файл {flask_script} не найден!")
        return None
    
    process = subprocess.Popen(
        [sys.executable, str(flask_script)],
        cwd=str(PROJECT_ROOT)
    )
    
    time.sleep(2)
    print("✅ Flask Frontend запущен!")
    print()
    return process


def start_nextjs_frontend():
    """Запуск Next.js frontend"""
    print("🚀 Запуск Next.js Frontend...")
    print(f"   Порт: http://localhost:{NEXTJS_PORT}")
    print()
    
    frontend_dir = PROJECT_ROOT / "frontend"
    if not frontend_dir.exists():
        print(f"❌ Папка {frontend_dir} не найдена!")
        return None
    
    # Проверка node_modules
    node_modules = frontend_dir / "node_modules"
    if not node_modules.exists():
        print("⚠️  Node modules не найдены. Установка зависимостей...")
        subprocess.run(["npm", "install"], cwd=str(frontend_dir))
    
    process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(frontend_dir)
    )
    
    time.sleep(5)  # Next.js запускается дольше
    print("✅ Next.js Frontend запущен!")
    print()
    return process


def open_browser(version: str):
    """Открытие браузера"""
    urls = {
        "flask": f"http://localhost:{FLASK_PORT}",
        "nextjs": f"http://localhost:{NEXTJS_PORT}",
        "api-only": f"http://localhost:{BACKEND_PORT}/docs"
    }
    
    url = urls.get(version, urls["api-only"])
    
    print(f"🌐 Открытие браузера: {url}")
    webbrowser.open(url)
    time.sleep(1)


def interactive_choice():
    """Интерактивный выбор версии"""
    print_versions()
    
    while True:
        choice = input("Выберите версию (1/2/3 или flask/nextjs/api-only): ").strip().lower()
        
        if choice in ["1", "flask"]:
            return "flask"
        elif choice in ["2", "nextjs"]:
            return "nextjs"
        elif choice in ["3", "api-only", "api"]:
            return "api-only"
        else:
            print("❌ Неверный выбор. Попробуйте ещё раз.")
            print()


def main():
    """Основная функция"""
    print_banner()
    
    # Определение режима запуска
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode not in ["flask", "nextjs", "api-only", "api"]:
            print(f"❌ Неверный аргумент: {mode}")
            print("Используйте: flask, nextjs, или api-only")
            return
    else:
        mode = interactive_choice()
    
    print()
    print("=" * 70)
    print(f"  Режим запуска: {mode.upper()}")
    print("=" * 70)
    print()
    
    # Запуск backend
    backend_process = start_backend()
    if not backend_process:
        return
    
    processes = [backend_process]
    
    # Запуск frontend в зависимости от режима
    if mode == "flask":
        frontend_process = start_flask_frontend()
        if frontend_process:
            processes.append(frontend_process)
        open_browser("flask")
        
    elif mode == "nextjs":
        frontend_process = start_nextjs_frontend()
        if frontend_process:
            processes.append(frontend_process)
        open_browser("nextjs")
        
    elif mode == "api-only":
        print("📡 Запущен только Backend API")
        print("   Swagger UI: http://localhost:8000/docs")
        print()
        open_browser("api-only")
    
    print("=" * 70)
    print("  ✅ Все сервисы запущены!")
    print("=" * 70)
    print()
    print("📌 Полезные ссылки:")
    print(f"   • Backend API: http://localhost:{BACKEND_PORT}/docs")
    if mode == "flask":
        print(f"   • Flask Frontend: http://localhost:{FLASK_PORT}")
    elif mode == "nextjs":
        print(f"   • Next.js Frontend: http://localhost:{NEXTJS_PORT}")
    print()
    print("💡 Для остановки нажмите Ctrl+C")
    print()
    
    # Ожидание остановки
    try:
        for proc in processes:
            proc.wait()
    except KeyboardInterrupt:
        print("\n\n⏹️  Остановка сервисов...")
        for proc in processes:
            proc.terminate()
        print("✅ Все сервисы остановлены.")


if __name__ == "__main__":
    main()
