#!/usr/bin/env python3
"""
Скрипт для одновременного запуска Flask и FastAPI приложений
Удобно для разработки и тестирования интеграции
"""

import os
import sys
import time
import threading
import subprocess
import webbrowser
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent.parent))


class Colors:
    """Цвета для вывода в консоль"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'


class ApplicationRunner:
    """Менеджер для запуска приложений"""

    def __init__(self, fastapi_port=8000, flask_port=5000, fastapi_reload=False, flask_debug=False):
        self.fastapi_port = fastapi_port
        self.flask_port = flask_port
        self.fastapi_reload = fastapi_reload
        self.flask_debug = flask_debug
        
        self.fastapi_process = None
        self.flask_process = None
        self.project_root = Path(__file__).parent.parent

    def start_fastapi(self):
        """Запуск FastAPI приложения"""
        print(f"\n{Colors.GREEN}[INFO]{Colors.END} Запуск FastAPI на порту {self.fastapi_port}...")
        
        cmd = [
            sys.executable, "-m", "uvicorn",
            "api.main:app",
            "--host", "0.0.0.0",
            "--port", str(self.fastapi_port),
        ]
        
        if self.fastapi_reload:
            cmd.append("--reload")
        
        try:
            self.fastapi_process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                env=os.environ.copy()
            )
            time.sleep(2)  # Ждём запуска
            
            if self.fastapi_process.poll() is None:
                print(f"{Colors.GREEN}[OK]{Colors.END} FastAPI запущен (PID: {self.fastapi_process.pid})")
                return True
            else:
                print(f"{Colors.RED}[ERROR]{Colors.END} FastAPI не удалось запустить")
                return False
        except Exception as e:
            print(f"{Colors.RED}[ERROR]{Colors.END} Ошибка запуска FastAPI: {e}")
            return False

    def start_flask(self):
        """Запуск Flask приложения"""
        print(f"\n{Colors.BLUE}[INFO]{Colors.END} Запуск Flask на порту {self.flask_port}...")
        
        # Используем интегрированную версию
        flask_script = self.project_root / "src" / "web" / "web_dashboard_integrated.py"
        
        if not flask_script.exists():
            # Фоллбэк на оригинальную версию
            flask_script = self.project_root / "src" / "web" / "web_dashboard.py"
        
        cmd = [
            sys.executable,
            str(flask_script),
            "--port", str(self.flask_port),
            "--fastapi-url", f"http://localhost:{self.fastapi_port}",
        ]
        
        if self.flask_debug:
            cmd.append("--debug")
        
        cmd.append("--no-browser")  # Не открывать браузер автоматически
        
        try:
            self.flask_process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                env=os.environ.copy()
            )
            time.sleep(2)  # Ждём запуска
            
            if self.flask_process.poll() is None:
                print(f"{Colors.GREEN}[OK]{Colors.END} Flask запущен (PID: {self.flask_process.pid})")
                return True
            else:
                print(f"{Colors.RED}[ERROR]{Colors.END} Flask не удалось запустить")
                return False
        except Exception as e:
            print(f"{Colors.RED}[ERROR]{Colors.END} Ошибка запуска Flask: {e}")
            return False

    def stop_all(self):
        """Остановка всех приложений"""
        print(f"\n{Colors.YELLOW}[INFO]{Colors.END} Остановка приложений...")
        
        if self.fastapi_process:
            print(f"  Остановка FastAPI (PID: {self.fastapi_process.pid})...")
            self.fastapi_process.terminate()
            try:
                self.fastapi_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.fastapi_process.kill()
        
        if self.flask_process:
            print(f"  Остановка Flask (PID: {self.flask_process.pid})...")
            self.flask_process.terminate()
            try:
                self.flask_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.flask_process.kill()
        
        print(f"{Colors.GREEN}[OK]{Colors.END} Все приложения остановлены")

    def run(self, open_browser=False):
        """Запуск обоих приложений"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}  NANOPROBE SIMULATION LAB{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}  Запуск приложений...{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
        
        # Запуск FastAPI
        if not self.start_fastapi():
            self.stop_all()
            return 1
        
        # Запуск Flask
        if not self.start_flask():
            self.stop_all()
            return 1
        
        # Информация для пользователя
        print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.GREEN}  ПРИЛОЖЕНИЯ ЗАПУЩЕНЫ{Colors.END}")
        print(f"{Colors.BOLD}{Colors.GREEN}{'='*60}{Colors.END}")
        print(f"\n  {Colors.BLUE}FastAPI API:{Colors.END}   http://localhost:{self.fastapi_port}")
        print(f"  {Colors.BLUE}Swagger docs:{Colors.END}  http://localhost:{self.fastapi_port}/docs")
        print(f"  {Colors.BLUE}ReDoc docs:{Colors.END}    http://localhost:{self.fastapi_port}/redoc")
        print(f"  {Colors.YELLOW}Flask Web UI:{Colors.END} http://localhost:{self.flask_port}")
        
        if open_browser:
            print(f"\n  {Colors.CYAN}Открытие браузера...{Colors.END}")
            time.sleep(1.5)
            webbrowser.open(f"http://localhost:{self.flask_port}")
        
        print(f"\n  {Colors.YELLOW}Нажмите Ctrl+C для остановки{Colors.END}\n")
        
        # Ожидание завершения
        try:
            while True:
                time.sleep(1)
                
                # Проверка процессов
                if self.fastapi_process.poll() is not None:
                    print(f"{Colors.RED}[ERROR]{Colors.END} FastAPI процесс завершился unexpectedly")
                    break
                
                if self.flask_process.poll() is not None:
                    print(f"{Colors.RED}[ERROR]{Colors.END} Flask процесс завершился unexpectedly")
                    break
                    
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}[INFO]{Colors.END} Получен сигнал остановки...")
        
        self.stop_all()
        return 0


def check_dependencies():
    """Проверка необходимых зависимостей"""
    print(f"{Colors.CYAN}[INFO]{Colors.END} Проверка зависимостей...")
    
    required = ['uvicorn', 'fastapi', 'flask', 'flask_socketio', 'requests']
    missing = []
    
    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"{Colors.RED}[ERROR]{Colors.END} Отсутствуют зависимости: {', '.join(missing)}")
        print(f"  Установите: pip install {' '.join(missing)}")
        return False
    
    print(f"{Colors.GREEN}[OK]{Colors.END} Все зависимости найдены")
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Одновременный запуск Flask и FastAPI приложений',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python start_all.py                    # Запуск без отладки
  python start_all.py --reload --debug   # С reload и debug режимами
  python start_all.py --browser          # С открытием браузера
        """
    )
    
    parser.add_argument(
        '--fastapi-port',
        type=int,
        default=8000,
        help='Порт для FastAPI (по умолчанию: 8000)'
    )
    parser.add_argument(
        '--flask-port',
        type=int,
        default=5000,
        help='Порт для Flask (по умолчанию: 5000)'
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Включить reload режим для FastAPI'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Включить debug режим для Flask'
    )
    parser.add_argument(
        '--browser',
        action='store_true',
        help='Открыть браузер автоматически'
    )
    
    args = parser.parse_args()
    
    # Проверка зависимостей
    if not check_dependencies():
        return 1
    
    # Создание и запуск
    runner = ApplicationRunner(
        fastapi_port=args.fastapi_port,
        flask_port=args.flask_port,
        fastapi_reload=args.reload,
        flask_debug=args.debug
    )
    
    return runner.run(open_browser=args.browser)


if __name__ == "__main__":
    sys.exit(main())
