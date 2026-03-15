#!/usr/bin/env python
"""
Nanoprobe Sim Lab - Universal Launcher v2.0
Универсальный запуск с выбором Frontend

Поддерживаемые режимы:
    python start_universal.py              # Интерактивный выбор
    python start_universal.py flask        # Flask + FastAPI
    python start_universal.py nextjs       # Next.js + FastAPI
    python start_universal.py api-only     # Только Backend API
    python start_universal.py full         # Flask + FastAPI + Sync Manager

Требования:
    Python 3.11, 3.12, 3.13, or 3.14
"""

# Проверка версии Python
import sys
MIN_PYTHON_VERSION = (3, 11)
MAX_PYTHON_VERSION = (3, 14)
if sys.version_info < MIN_PYTHON_VERSION or sys.version_info >= (MAX_PYTHON_VERSION[0], MAX_PYTHON_VERSION[1] + 1):
    print(f"[ERROR] Требуется Python 3.11 - 3.14, текущая версия: {sys.version}")
    sys.exit(1)

import os
import subprocess
import time
import webbrowser
import socket
from pathlib import Path
from typing import Optional, List

if sys.platform == "win32":
    os.system("chcp 65001 >nul")

PROJECT_ROOT = Path(__file__).parent

# Порты
BACKEND_PORT = 8000
FLASK_PORT = 5000
NEXTJS_PORT = 3000

# Скрипты
BACKEND_SCRIPT = PROJECT_ROOT / "run_api.py"
FLASK_SCRIPT = PROJECT_ROOT / "src" / "web" / "web_dashboard_unified.py"
SYNC_SCRIPT = PROJECT_ROOT / "api" / "sync_manager.py"
FRONTEND_DIR = PROJECT_ROOT / "frontend"


# ==================== Утилиты ====================

def check_port(port: int) -> bool:
    """Проверка занятости порта"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result == 0
    except (socket.error, OSError):
        return False


def wait_for_port(port: int, timeout: int = 30) -> bool:
    """Ожидание доступности порта"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if not check_port(port):
            time.sleep(0.5)
        else:
            return True
    return False


def print_banner():
    """Вывод баннера"""
    print()
    print("=" * 70)
    print("  Nanoprobe Sim Lab - Universal Launcher v2.0")
    print("=" * 70)
    print()


def print_versions():
    """Вывод доступных версий"""
    print("Доступные режимы запуска:")
    print()
    print("  1. Flask + FastAPI (Unified)")
    print("     - Backend:  http://localhost:8000")
    print("     - Frontend: http://localhost:5000")
    print("     - Swagger:  http://localhost:8000/docs")
    print("     - Sync:     Включён (автоматическая синхронизация)")
    print()
    print("  2. Next.js + FastAPI")
    print("     - Backend:  http://localhost:8000")
    print("     - Frontend: http://localhost:3000")
    print("     - Swagger:  http://localhost:8000/docs")
    print()
    print("  3. Backend API only")
    print("     - Backend:  http://localhost:8000")
    print("     - Swagger:  http://localhost:8000/docs")
    print()
    print("  4. Full Stack (Flask + FastAPI + Sync Manager)")
    print("     - Backend:  http://localhost:8000")
    print("     - Frontend: http://localhost:5000")
    print("     - Sync:     Расширенная синхронизация")
    print()
    print("=" * 70)
    print()


def interactive_choice() -> str:
    """Интерактивный выбор режима"""
    print_versions()
    while True:
        choice = input("Выберите режим (1/2/3/4 или flask/nextjs/api-only/full): ").strip().lower()
        
        if choice in ["1", "flask"]:
            return "flask"
        elif choice in ["2", "nextjs"]:
            return "nextjs"
        elif choice in ["3", "api-only", "api"]:
            return "api-only"
        elif choice in ["4", "full"]:
            return "full"
        else:
            print("[ERROR] Неверный выбор. Попробуйте снова.")
            print()


# ==================== Запуск сервисов ====================

def start_backend() -> Optional[subprocess.Popen]:
    """Запуск Backend (FastAPI)"""
    print("Запуск Backend (FastAPI)...")
    print(f"   Порт:    http://localhost:{BACKEND_PORT}")
    print(f"   Swagger: http://localhost:{BACKEND_PORT}/docs")
    print()

    if not BACKEND_SCRIPT.exists():
        print(f"[ERROR] {BACKEND_SCRIPT} не найден!")
        return None

    # Проверка uvicorn
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        print("[ERROR] uvicorn не установлен! Выполните: pip install uvicorn")
        return None

    process = subprocess.Popen(
        [sys.executable, str(BACKEND_SCRIPT), "--reload"],
        cwd=str(PROJECT_ROOT)
    )

    if wait_for_port(BACKEND_PORT, timeout=15):
        print("[OK] Backend запущен!")
        print()
        return process
    else:
        print("[ERROR] Не удалось запустить Backend!")
        return None


def start_flask_frontend() -> Optional[subprocess.Popen]:
    """Запуск Flask frontend"""
    print("Запуск Flask Frontend (Unified)...")
    print(f"   Порт: http://localhost:{FLASK_PORT}")
    print()

    if not FLASK_SCRIPT.exists():
        print(f"[ERROR] {FLASK_SCRIPT} не найден!")
        return None

    # Проверка Flask
    try:
        import flask  # noqa: F401
    except ImportError:
        print("[ERROR] Flask не установлен! Выполните: pip install flask flask-socketio")
        return None

    process = subprocess.Popen(
        [sys.executable, str(FLASK_SCRIPT)],
        cwd=str(PROJECT_ROOT)
    )

    time.sleep(3)
    
    if wait_for_port(FLASK_PORT, timeout=10):
        print("[OK] Flask Frontend запущен!")
        print()
        return process
    else:
        print("[WARN] Flask запущен, но порт может быть занят")
        print()
        return process


def start_nextjs_frontend() -> Optional[subprocess.Popen]:
    """Запуск Next.js frontend"""
    print("Запуск Next.js Frontend...")
    print(f"   Порт: http://localhost:{NEXTJS_PORT}")
    print()

    if not FRONTEND_DIR.exists():
        print(f"[ERROR] {FRONTEND_DIR} не найдена!")
        return None

    # Проверка node_modules
    node_modules = FRONTEND_DIR / "node_modules"
    if not node_modules.exists():
        print("[WARN] Установка зависимостей...")
        subprocess.run(["npm", "install"], cwd=str(FRONTEND_DIR))

    process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(FRONTEND_DIR)
    )

    time.sleep(5)
    print("[OK] Next.js Frontend запущен!")
    print()
    return process


def start_sync_manager() -> Optional[subprocess.Popen]:
    """Запуск Sync Manager"""
    print("Запуск Sync Manager...")
    print("   Синхронизация Backend ↔ Frontend")
    print()

    if not SYNC_SCRIPT.exists():
        print(f"[WARN] {SYNC_SCRIPT} не найден")
        return None

    process = subprocess.Popen(
        [sys.executable, str(SYNC_SCRIPT)],
        cwd=str(PROJECT_ROOT)
    )

    time.sleep(2)
    print("[OK] Sync Manager запущен!")
    print()
    return process


def open_browser(mode: str):
    """Открытие браузера"""
    urls = {
        "flask": f"http://localhost:{FLASK_PORT}",
        "nextjs": f"http://localhost:{NEXTJS_PORT}",
        "api-only": f"http://localhost:{BACKEND_PORT}/docs",
        "full": f"http://localhost:{FLASK_PORT}"
    }
    
    url = urls.get(mode, urls["api-only"])
    print(f"Открытие браузера: {url}")
    
    try:
        webbrowser.open(url)
    except Exception:
        pass
    
    time.sleep(1)


# ==================== Менеджер процессов ====================

class ProcessManager:
    """Менеджер процессов"""

    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.running = True

    def add_process(self, process: Optional[subprocess.Popen]):
        """Добавление процесса"""
        if process:
            self.processes.append(process)

    def wait(self):
        """Ожидание завершения процессов"""
        try:
            for proc in self.processes:
                proc.wait()
        except KeyboardInterrupt:
            print("\n\n[INFO] Остановка сервисов...")
            self.stop_all()

    def stop_all(self):
        """Остановка всех процессов"""
        for proc in reversed(self.processes):
            try:
                proc.terminate()
            except Exception:
                proc.kill()
        
        print("[OK] Все сервисы остановлены.")


# ==================== Главная функция ====================

def main():
    """Главная функция"""
    print_banner()

    # Определение режима запуска
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        valid_modes = ["flask", "nextjs", "api-only", "api", "full"]
        if mode not in valid_modes:
            print(f"[ERROR] Неверный аргумент: {mode}")
            print(f"Допустимые: {', '.join(valid_modes)}")
            return
    else:
        mode = interactive_choice()

    print()
    print("=" * 70)
    print(f"  Режим запуска: {mode.upper()}")
    print("=" * 70)
    print()

    # Инициализация менеджера процессов
    pm = ProcessManager()

    # Запуск Backend (обязательно для всех режимов)
    backend_process = start_backend()
    if not backend_process:
        print("[ERROR] Не удалось запустить Backend. Завершение.")
        return
    
    pm.add_process(backend_process)

    # Запуск Frontend в зависимости от режима
    if mode == "flask":
        frontend_process = start_flask_frontend()
        pm.add_process(frontend_process)
        open_browser("flask")

    elif mode == "nextjs":
        frontend_process = start_nextjs_frontend()
        pm.add_process(frontend_process)
        open_browser("nextjs")

    elif mode == "api-only":
        print("Запуск только Backend API")
        print("   Swagger UI: http://localhost:8000/docs")
        print()
        open_browser("api-only")

    elif mode == "full":
        frontend_process = start_flask_frontend()
        pm.add_process(frontend_process)
        
        sync_process = start_sync_manager()
        pm.add_process(sync_process)
        
        open_browser("full")

    # Вывод полезной информации
    print("=" * 70)
    print("  [OK] Все сервисы запущены!")
    print("=" * 70)
    print()
    print("Полезные ссылки:")
    print(f"   - Backend API (Swagger): http://localhost:{BACKEND_PORT}/docs")
    print(f"   - Backend Health:        http://localhost:{BACKEND_PORT}/health")
    
    if mode in ["flask", "full"]:
        print(f"   - Flask Frontend:        http://localhost:{FLASK_PORT}")
    elif mode == "nextjs":
        print(f"   - Next.js Frontend:      http://localhost:{NEXTJS_PORT}")
    
    print()
    print("Нажмите Ctrl+C для остановки всех сервисов")
    print()

    # Ожидание завершения
    pm.wait()


if __name__ == "__main__":
    main()
