#!/usr/bin/env python
"""
Синхронизированный запуск Backend + Frontend для Nanoprobe Sim Lab
Автоматическая проверка зависимостей, портов и здоровья сервисов

Интеграция:
- Backend (FastAPI): http://localhost:8000
- Frontend (Flask): http://localhost:5000
- Sync Manager: автоматическая синхронизация метрик и событий

Требования:
- Python 3.11, 3.12, 3.13, or 3.14
"""

import os
import sys

# Проверка версии Python (требуется 3.11 - 3.14)
MIN_PYTHON_VERSION = (3, 11)
MAX_PYTHON_VERSION = (3, 14)
if sys.version_info < MIN_PYTHON_VERSION or sys.version_info >= (MAX_PYTHON_VERSION[0], MAX_PYTHON_VERSION[1] + 1):
    print(f"[ERROR] Требуется Python 3.11 - 3.14, текущая версия: {sys.version}")
    print(f"Путь к Python: {sys.executable}")
    print("Установите Python 3.11 - 3.14 с https://www.python.org/downloads/")
    sys.exit(1)

import time
import signal
import subprocess
import socket
import webbrowser
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

# UTF-8 для Windows
if sys.platform == "win32":
    os.system("chcp 65001 >nul")
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Пути
PROJECT_ROOT = Path(__file__).parent
BACKEND_SCRIPT = PROJECT_ROOT / "run_api.py"
FRONTEND_SCRIPT = PROJECT_ROOT / "src" / "web" / "web_dashboard_unified.py"
SYNC_MANAGER_SCRIPT = PROJECT_ROOT / "api" / "sync_manager.py"
LOG_DIR = PROJECT_ROOT / "logs"
PYTHON_EXECUTABLE = sys.executable

# Конфигурация
BACKEND_HOST = "0.0.0.0"
BACKEND_PORT = 8000
FRONTEND_HOST = "127.0.0.1"
FRONTEND_PORT = 5000
HEALTH_CHECK_TIMEOUT = 30  # секунд
HEALTH_CHECK_INTERVAL = 2  # секунды
SYNC_INTERVAL = 5  # секунд


class ServiceManager:
    """Менеджер сервисов для синхронизированного запуска"""

    def __init__(self):
        """TODO: Add description"""
        self.backend_process: Optional[subprocess.Popen] = None
        self.frontend_process: Optional[subprocess.Popen] = None
        self.sync_process: Optional[subprocess.Popen] = None
        self.running = True

        # Создание директории логов
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        # Настройка обработчика сигналов
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Статус синхронизации
        self.sync_enabled = True
        self.last_sync_time: Optional[datetime] = None
        self.sync_stats: Dict[str, Any] = {}

    def _signal_handler(self, signum, frame):
        """Обработчик сигналов остановки"""
        print(f"\n[INFO] Получен сигнал {signum}, остановка сервисов...")
        self.running = False
        self._stop_all()
        sys.exit(0)

    def _check_port(self, port: int) -> bool:
        """Проверка занятости порта"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            result = sock.connect_ex(('localhost', port))
            sock.close()
            return result == 0  # Порт занят если результат 0
        except (socket.error, OSError):
            return False

    def _wait_for_port(self, port: int, timeout: int = 30) -> bool:
        """Ожидание доступности порта"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if not self._check_port(port):
                time.sleep(0.5)
            else:
                return True
        return False

    def _check_health(self, url: str) -> bool:
        """Проверка health endpoint"""
        try:
            import requests
            response = requests.get(url, timeout=3)
            return response.status_code == 200
        except (requests.RequestException, requests.Timeout):
            return False

    def _sync_backend_frontend(self) -> bool:
        """
        Синхронизация Backend и Frontend

        Получает статистику из Backend и передаёт во Frontend
        """
        try:
            import requests

            # Получение статистики из Backend
            backend_stats_url = f"http://localhost:{BACKEND_PORT}/api/v1/dashboard/stats"
            response = requests.get(backend_stats_url, timeout=5)

            if response.status_code == 200:
                stats = response.json()
                self.sync_stats = stats
                self.last_sync_time = datetime.now()

                # Отправка статистики во Frontend (опционально)
                # Frontend может сам запрашивать данные через reverse proxy

                self._log(f"✅ Синхронизация: {len(stats)} полей данных", "SUCCESS")
                return True
            else:
                self._log(f"⚠️ Backend вернул статус {response.status_code}", "WARNING")
                return False

        except Exception as e:
            self._log(f"⚠️ Ошибка синхронизации: {e}", "WARNING")
            return False

    def _log(self, message: str, level: str = "INFO"):
        """Логирование сообщения"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)

        # Запись в лог
        log_file = LOG_DIR / "startup.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

    def _check_dependencies(self) -> Tuple[bool, list]:
        """Проверка необходимых зависимостей"""
        missing = []

        # Проверка для backend
        backend_deps = ["fastapi", "uvicorn", "pydantic", "sqlalchemy"]
        for dep in backend_deps:
            try:
                __import__(dep.replace("-", "_"))
            except ImportError:
                missing.append(f"backend: {dep}")

        # Проверка для frontend
        frontend_deps = ["flask", "flask_socketio"]
        for dep in frontend_deps:
            try:
                if dep == "flask_socketio":
                    __import__("flask_socketio")
                else:
                    __import__(dep)
            except ImportError:
                missing.append(f"frontend: {dep}")

        # Проверка requests для мониторинга
        try:
            import requests
        except ImportError:
            missing.append("monitoring: requests")

        return len(missing) == 0, missing

    def _start_backend(self) -> bool:
        """Запуск Backend (FastAPI)"""
        self._log("=" * 60)
        self._log("Запуск Backend (FastAPI)...")
        self._log(f"URL: http://{BACKEND_HOST}:{BACKEND_PORT}")
        self._log(f"Docs: http://{BACKEND_HOST}:{BACKEND_PORT}/docs")

        # Проверка порта
        if self._check_port(BACKEND_PORT):
            self._log(f"Порт {BACKEND_PORT} занят!", "ERROR")
            return False

        # Окружение для UTF-8
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        # Запуск процесса
        log_file = LOG_DIR / "backend.log"
        with open(log_file, "w", encoding="utf-8") as f:
            self.backend_process = subprocess.Popen(
                [PYTHON_EXECUTABLE, str(BACKEND_SCRIPT), "--reload"],
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            )

        self._log(f"Backend запущен (PID: {self.backend_process.pid})")

        # Ожидание доступности
        self._log(f"Ожидание готовности Backend (до {HEALTH_CHECK_TIMEOUT} сек)...")
        if not self._wait_for_port(BACKEND_PORT, HEALTH_CHECK_TIMEOUT):
            self._log("Backend не запустился вовремя!", "ERROR")
            return False

        # Проверка health
        if not self._check_health(f"http://localhost:{BACKEND_PORT}/health"):
            self._log("Backend не отвечает на health check!", "ERROR")
            return False

        self._log("✅ Backend готов к работе", "SUCCESS")
        return True

    def _start_frontend(self) -> bool:
        """Запуск Frontend (Flask)"""
        self._log("=" * 60)
        self._log("Запуск Frontend (Flask)...")
        self._log(f"URL: http://{FRONTEND_HOST}:{FRONTEND_PORT}")

        # Проверка порта
        if self._check_port(FRONTEND_PORT):
            self._log(f"Порт {FRONTEND_PORT} занят!", "ERROR")
            return False

        # Окружение для UTF-8
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        # Запуск процесса
        log_file = LOG_DIR / "frontend.log"
        with open(log_file, "w", encoding="utf-8") as f:
            self.frontend_process = subprocess.Popen(
                [PYTHON_EXECUTABLE, str(FRONTEND_SCRIPT), "--no-browser"],
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            )

        self._log(f"Frontend запущен (PID: {self.frontend_process.pid})")

        # Ожидание доступности
        self._log(f"Ожидание готовности Frontend (до {HEALTH_CHECK_TIMEOUT} сек)...")
        if not self._wait_for_port(FRONTEND_PORT, HEALTH_CHECK_TIMEOUT):
            self._log("Frontend не запустился вовремя!", "ERROR")
            return False

        # Проверка доступности
        if not self._check_health(f"http://localhost:{FRONTEND_PORT}/api/status"):
            self._log("Frontend не отвечает на API check!", "WARNING")
            # Это не критично, продолжаем

        self._log("✅ Frontend готов к работе", "SUCCESS")
        return True

    def _stop_all(self):
        """Остановка всех сервисов"""
        self._log("=" * 60)
        self._log("Остановка сервисов...")

        # Остановка frontend
        if self.frontend_process:
            try:
                self._log("Остановка Frontend...")
                self.frontend_process.terminate()
                self.frontend_process.wait(timeout=5)
                self._log("✅ Frontend остановлен")
            except Exception as e:
                self._log(f"Ошибка остановки Frontend: {e}", "ERROR")
                try:
                    self.frontend_process.kill()
                except (subprocess.SubprocessError, OSError):
                    pass

        # Остановка backend
        if self.backend_process:
            try:
                self._log("Остановка Backend...")
                self.backend_process.terminate()
                self.backend_process.wait(timeout=5)
                self._log("✅ Backend остановлен")
            except Exception as e:
                self._log(f"Ошибка остановки Backend: {e}", "ERROR")
                try:
                    self.backend_process.kill()
                except (subprocess.SubprocessError, OSError):
                    pass

        # Остановка sync manager (если запущен)
        if self.sync_process:
            try:
                self._log("Остановка Sync Manager...")
                self.sync_process.terminate()
                self.sync_process.wait(timeout=3)
                self._log("✅ Sync Manager остановлен")
            except Exception as e:
                self._log(f"Ошибка остановки Sync Manager: {e}", "ERROR")
                try:
                    self.sync_process.kill()
                except (subprocess.SubprocessError, OSError):
                    pass

        self._log("Все сервисы остановлены")

    def start_all(self) -> bool:
        """Запуск всех сервисов"""
        print("\n" + "=" * 70)
        print("🚀 NANOPROBE SIM LAB - Синхронизированный запуск")
        print("=" * 70)
        self._log(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"Python: {PYTHON_EXECUTABLE}")
        self._log(f"Project: {PROJECT_ROOT}")

        # Проверка зависимостей
        self._log("Проверка зависимостей...")
        deps_ok, missing = self._check_dependencies()
        if not deps_ok:
            self._log(f"Отсутствуют зависимости: {missing}", "ERROR")
            self._log("Установите: pip install -r requirements.txt -r requirements-api.txt", "ERROR")
            return False
        self._log("✅ Все зависимости установлены")

        # Запуск backend
        if not self._start_backend():
            self._log("Не удалось запустить Backend!", "CRITICAL")
            self._stop_all()
            return False

        # Небольшая задержка перед запуском frontend
        time.sleep(2)

        # Запуск frontend
        if not self._start_frontend():
            self._log("Не удалось запустить Frontend!", "CRITICAL")
            self._stop_all()
            return False

        # Финальная проверка
        self._log("=" * 70)
        self._log("🎉 ВСЕ СЕРВИСЫ ЗАПУЩЕНЫ!")
        self._log("=" * 70)
        self._log(f"📊 Backend API:  http://{BACKEND_HOST}:{BACKEND_PORT}")
        self._log(f"📚 Swagger UI:   http://{BACKEND_HOST}:{BACKEND_PORT}/docs")
        self._log(f"🖥️  Frontend:     http://{FRONTEND_HOST}:{FRONTEND_PORT}")
        self._log(f"🔄 Sync Manager:  Интервал {SYNC_INTERVAL}с")
        self._log("=" * 70)
        self._log("Нажмите Ctrl+C для остановки всех сервисов")

        # Открытие браузера
        try:
            webbrowser.open(f"http://{FRONTEND_HOST}:{FRONTEND_PORT}")
        except (OSError, webbrowser.Error):
            pass

        return True

    def run(self):
        """Основной цикл работы"""
        if not self.start_all():
            sys.exit(1)

        # Мониторинг процессов
        self._log("Мониторинг сервисов...")
        sync_counter = 0

        while self.running:
            time.sleep(1)

            # Проверка backend
            if self.backend_process and self.backend_process.poll() is not None:
                self._log("Backend unexpectedly stopped!", "ERROR")
                self.running = False

            # Проверка frontend
            if self.frontend_process and self.frontend_process.poll() is not None:
                self._log("Frontend unexpectedly stopped!", "ERROR")
                self.running = False

            # Синхронизация каждые SYNC_INTERVAL секунд
            sync_counter += 1
            if sync_counter >= SYNC_INTERVAL and self.sync_enabled:
                sync_counter = 0
                self._sync_backend_frontend()

            # Health check каждые 30 секунд
            if int(time.time()) % 30 == 0:
                if not self._check_health(f"http://localhost:{BACKEND_PORT}/health"):
                    self._log("Backend health check failed!", "WARNING")


def main():
    """Точка входа"""
    manager = ServiceManager()
    manager.run()


if __name__ == "__main__":
    main()
