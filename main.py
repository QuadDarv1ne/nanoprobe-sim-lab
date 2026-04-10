#!/usr/bin/env python
"""
Nanoprobe Sim Lab - Universal Launcher v3.0 (Unified)

Единая точка входа для запуска всех сервисов проекта.
Заменяет: start.py, start_all.py, start_universal.py

Поддерживаемые режимы:
    python main.py                    # Интерактивный выбор
    python main.py flask              # Flask + FastAPI + Sync
    python main.py nextjs             # Next.js + FastAPI + Sync
    python main.py api-only           # Только Backend API
    python main.py full               # Full Stack с расширенной синхронизацией
    python main.py dev                # Development mode (Flask + reload)

Требования:
    Python 3.11, 3.12, 3.13, or 3.14
"""

# Проверка версии Python
import sys

MIN_PYTHON_VERSION = (3, 11)
MAX_PYTHON_VERSION = (3, 14)
if sys.version_info < MIN_PYTHON_VERSION or sys.version_info >= (
    MAX_PYTHON_VERSION[0],
    MAX_PYTHON_VERSION[1] + 1,
):
    print(f"[ERROR] Требуется Python 3.11 - 3.14, текущая версия: {sys.version}")
    sys.exit(1)

import os
import signal
import socket
import subprocess
import time
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional

# Автоопределение портов
try:
    from utils.port_finder import find_ports

    AUTO_PORT_ENABLED = True
except ImportError:
    AUTO_PORT_ENABLED = False

if sys.platform == "win32":
    os.system("chcp 65001 >nul")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

PROJECT_ROOT = Path(__file__).parent

# Порты (будут обновлены при автоопределении)
BACKEND_PORT = int(os.getenv("BACKEND_PORT", 8000))
FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))
NEXTJS_PORT = int(os.getenv("NEXTJS_PORT", 3000))

# Скрипты
BACKEND_SCRIPT = PROJECT_ROOT / "run_api.py"
FLASK_SCRIPT = PROJECT_ROOT / "src" / "web" / "web_dashboard_unified.py"
SYNC_SCRIPT = PROJECT_ROOT / "api" / "sync_manager.py"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Конфигурация
SYNC_ENABLED_BY_DEFAULT = True
SYNC_INTERVAL = 5  # секунд
HEALTH_CHECK_TIMEOUT = 30
HEALTH_CHECK_INTERVAL = 2


# ==================== Утилиты ====================


def auto_detect_ports(services: List[str] = None) -> Dict[str, int]:
    """
    Автоматическое определение свободных портов

    Args:
        services: Список сервисов ['backend', 'flask', 'nextjs']

    Returns:
        Словарь {service: port}
    """
    global BACKEND_PORT, FLASK_PORT, NEXTJS_PORT

    if not AUTO_PORT_ENABLED:
        print("⚠️  Автоопределение портов недоступно (portFinder не импортирован)")
        return {"backend": BACKEND_PORT, "flask": FLASK_PORT, "nextjs": NEXTJS_PORT}

    if services is None:
        services = ["backend", "flask", "nextjs"]

    try:
        print("🔍 Автоопределение свободных портов...")
        ports = find_ports(services)

        # Обновляем глобальные переменные
        if "backend" in ports:
            BACKEND_PORT = ports["backend"]
            os.environ["BACKEND_PORT"] = str(BACKEND_PORT)

        if "flask" in ports:
            FLASK_PORT = ports["flask"]
            os.environ["FLASK_PORT"] = str(FLASK_PORT)

        if "nextjs" in ports:
            NEXTJS_PORT = ports["nextjs"]
            os.environ["NEXTJS_PORT"] = str(NEXTJS_PORT)

        print("✅ Найденные порты:")
        for service, port in ports.items():
            print(f"   {service:15s}: {port}")
        print()

        return ports

    except Exception as e:
        print(f"⚠️  Автоопределение не удалось: {e}")
        print(f"📌 Используем порта по умолчанию:")
        print(f"   backend: {BACKEND_PORT}")
        print(f"   flask:   {FLASK_PORT}")
        print(f"   nextjs:  {NEXTJS_PORT}")
        print()

        return {"backend": BACKEND_PORT, "flask": FLASK_PORT, "nextjs": NEXTJS_PORT}


def check_port(port: int) -> bool:
    """Проверка занятости порта"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        result = sock.connect_ex(("localhost", port))
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
    print("  Nanoprobe Sim Lab - Universal Launcher v3.0 (Unified)")
    print("=" * 70)
    print()


def print_versions():
    """Вывод доступных режимов"""
    print("Доступные режимы запуска:")
    print()
    print("  1. flask      - Flask + FastAPI + Auto Sync ✅")
    print("  2. nextjs     - Next.js + FastAPI + Auto Sync ✅")
    print("  3. api-only   - Только Backend API")
    print("  4. full       - Full Stack (Flask + FastAPI + Sync Manager)")
    print("  5. dev        - Development mode (Flask + reload)")
    print()
    print("=" * 70)
    print("  💡 Sync Manager запускается автоматически (кроме api-only)")
    print("  🔍 Автоопределение портов включено")
    print("=" * 70)
    print()


def interactive_choice() -> str:
    """Интерактивный выбор режима"""
    print_versions()
    while True:
        choice = input("Выберите режим (1-5 или flask/nextjs/api-only/full/dev): ").strip().lower()

        if choice in ["1", "flask"]:
            return "flask"
        elif choice in ["2", "nextjs"]:
            return "nextjs"
        elif choice in ["3", "api-only", "api"]:
            return "api-only"
        elif choice in ["4", "full"]:
            return "full"
        elif choice in ["5", "dev"]:
            return "dev"
        else:
            print("[ERROR] Неверный выбор. Попробуйте снова.")
            print()


# ==================== Запуск сервисов ====================


def start_backend(reload: bool = False) -> Optional[subprocess.Popen]:
    """Запуск Backend (FastAPI)"""
    print("Запуск Backend (FastAPI)...")
    print(f"   Порт:    http://localhost:{BACKEND_PORT}")
    print(f"   Swagger: http://localhost:{BACKEND_PORT}/docs")
    print()

    if not BACKEND_SCRIPT.exists():
        print(f"[ERROR] {BACKEND_SCRIPT} не найден!")
        return None

    # Проверка uvicorn
    import importlib.util

    if importlib.util.find_spec("uvicorn") is None:
        print("[ERROR] uvicorn не установлен! Выполните: pip install uvicorn")
        return None

    cmd = [sys.executable, str(BACKEND_SCRIPT)]
    if reload:
        cmd.append("--reload")

    process = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))

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
    import importlib.util

    if importlib.util.find_spec("flask") is None:
        print("[ERROR] Flask не установлен! Выполните: pip install flask flask-socketio")
        return None

    process = subprocess.Popen([sys.executable, str(FLASK_SCRIPT)], cwd=str(PROJECT_ROOT))

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

    process = subprocess.Popen(["npm", "run", "dev"], cwd=str(FRONTEND_DIR))

    time.sleep(5)
    print("[OK] Next.js Frontend запущен!")
    print()
    return process


def start_sync_manager() -> Optional[subprocess.Popen]:
    """Запуск Sync Manager (автоматическая синхронизация)"""
    print("Запуск Sync Manager...")
    print("   Автоматическая синхронизация Backend ↔ Frontend")
    print(f"   Интервал: {SYNC_INTERVAL} секунд")
    print()

    if not SYNC_SCRIPT.exists():
        print(f"[WARN] {SYNC_SCRIPT} не найден")
        return None

    process = subprocess.Popen(
        [sys.executable, str(SYNC_SCRIPT)],
        cwd=str(PROJECT_ROOT),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )

    time.sleep(3)
    print("[OK] Sync Manager запущен!")
    print()
    return process


def open_browser(mode: str):
    """Открытие браузера"""
    urls = {
        "flask": f"http://localhost:{FLASK_PORT}",
        "nextjs": f"http://localhost:{NEXTJS_PORT}",
        "api-only": f"http://localhost:{BACKEND_PORT}/docs",
        "full": f"http://localhost:{FLASK_PORT}",
        "dev": f"http://localhost:{FLASK_PORT}",
    }

    url = urls.get(mode, urls["api-only"])
    print(f"Открытие браузера: {url}")

    try:
        webbrowser.open(url)
    except Exception:
        pass

    time.sleep(1)


# ==================== Process Manager ====================


class ProcessManager:
    """Менеджер процессов"""

    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.running = True
        # Регистрация signal handlers
        if sys.platform != "win32":
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Обработка сигналов для корректного завершения"""
        print(f"\n[INFO] Получен сигнал {signum}, остановка сервисов...")
        self.stop_all()
        sys.exit(0)

    def add_process(self, process: Optional[subprocess.Popen]):
        """Добавление процесса"""
        if process:
            self.processes.append(process)

    def wait(self):
        """Ожидание завершения процессов"""
        try:
            for proc in self.processes:
                if proc.poll() is None:  # Только если процесс ещё работает
                    proc.wait()
        except KeyboardInterrupt:
            print("\n\n[INFO] Остановка сервисов...")
            self.stop_all()
        except Exception as e:
            print(f"\n[ERROR] Ошибка ожидания: {e}")
            self.stop_all()

    def stop_all(self):
        """Остановка всех процессов с корректным завершением"""
        if not self.processes:
            return

        print("[INFO] Остановка всех сервисов...")

        # Фаза 1: Отправка SIGTERM
        for proc in reversed(self.processes):
            if proc.poll() is None:  # Только живые процессы
                try:
                    proc.terminate()
                    print(f"  → Отправлен SIGTERM процессу PID {proc.pid}")
                except Exception as e:
                    print(f"  ⚠️  Ошибка terminate PID {proc.pid}: {e}")

        # Фаза 2: Ожидание завершения (до 5 секунд)
        for proc in self.processes:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=5)
                    print(f"  ✓ Процесс PID {proc.pid} завершён")
                except subprocess.TimeoutExpired:
                    print(f"  ⚠️  Процесс PID {proc.pid} не завершился за 5с, отправляем SIGKILL")
                    try:
                        proc.kill()
                        proc.wait(timeout=2)
                        print(f"  ✓ Процесс PID {proc.pid} убит")
                    except Exception as e:
                        print(f"  ✗ Ошибка kill PID {proc.pid}: {e}")
                except Exception as e:
                    print(f"  ✗ Ошибка ожидания PID {proc.pid}: {e}")

        # Фаза 3: Убиваем дочерние процессы (orphan cleanup)
        try:
            import psutil

            parent = psutil.Process()
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass

            # Ждём завершения дочерних
            _, alive = psutil.wait_procs(children, timeout=3)
            for p in alive:
                try:
                    p.kill()
                except psutil.NoSuchProcess:
                    pass
        except ImportError:
            pass  # psutil не установлен, пропускаем
        except Exception as e:
            print(f"  ⚠️  Ошибка очистки дочерних процессов: {e}")

        print("[OK] Все сервисы остановлены.")


# ==================== Главная функция ====================


def main():
    """Главная функция"""
    print_banner()

    # Определение режима запуска
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        valid_modes = ["flask", "nextjs", "api-only", "api", "full", "dev"]
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

    # Автоматическое определение портов
    if mode == "api-only":
        auto_detect_ports(["backend"])
    elif mode == "nextjs":
        auto_detect_ports(["backend", "nextjs"])
    else:
        auto_detect_ports(["backend", "flask"])

    # Инициализация менеджера процессов
    pm = ProcessManager()

    # Запуск Backend (обязательно для всех режимов)
    reload = mode == "dev"
    backend_process = start_backend(reload=reload)
    if not backend_process:
        print("[ERROR] Не удалось запустить Backend. Завершение.")
        return

    pm.add_process(backend_process)

    # Запуск Sync Manager (автоматически для всех режимов кроме api-only)
    if mode != "api-only" and SYNC_ENABLED_BY_DEFAULT:
        sync_process = start_sync_manager()
        pm.add_process(sync_process)

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
        # Sync Manager уже запущен выше
        open_browser("full")

    elif mode == "dev":
        frontend_process = start_flask_frontend()
        pm.add_process(frontend_process)
        open_browser("dev")

    # Вывод полезной информации
    print("=" * 70)
    print("  [OK] Все сервисы запущены!")
    print("=" * 70)
    print()
    print("Полезные ссылки:")
    print(f"   - Backend API (Swagger):   http://localhost:{BACKEND_PORT}/docs")
    print(f"   - Backend Health:          http://localhost:{BACKEND_PORT}/health")
    print(f"   - Sync Manager Status:     http://localhost:{BACKEND_PORT}/api/v1/sync/status")

    if mode in ["flask", "full", "dev"]:
        print(f"   - Flask Frontend:          http://localhost:{FLASK_PORT}")
    elif mode == "nextjs":
        print(f"   - Next.js Frontend:        http://localhost:{NEXTJS_PORT}")

    print()
    print("Синхронизация:")
    if mode != "api-only":
        print("   ✅ Sync Manager запущен (автоматическая синхронизация каждые 5 сек)")
    else:
        print("   ❌ Sync Manager отключен (режим api-only)")

    print()
    print("Порты:")
    print(f"   Backend:  {BACKEND_PORT}")
    if mode in ["flask", "full", "dev"]:
        print(f"   Flask:    {FLASK_PORT}")
    elif mode == "nextjs":
        print(f"   Next.js:  {NEXTJS_PORT}")

    print()
    print("Нажмите Ctrl+C для остановки всех сервисов")
    print()

    # Ожидание завершения
    pm.wait()


if __name__ == "__main__":
    main()
