#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт развёртывания Nanoprobe Simulation Lab в production
Использование: python scripts/deploy.py [--check] [--setup] [--start]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    CYAN = '\033[0;36m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text):
    print(f"\n{Colors.GREEN}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.GREEN}{Colors.BOLD}  {text}{Colors.END}")
    print(f"{Colors.GREEN}{Colors.BOLD}{'='*60}{Colors.END}\n")


def print_step(text):
    print(f"\n{Colors.CYAN}▶ {text}{Colors.END}")


def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


class DeploymentChecker:
    """Проверка готовности к развёртыванию"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.errors = []
        self.warnings = []
        self.success = []

    def check_python_version(self):
        """Проверка версии Python"""
        print_step("Проверка версии Python")

        version = sys.version_info
        if version.major == 3 and version.minor >= 10:
            print_success(f"Python {version.major}.{version.minor}.{version.micro}")
            self.success.append(f"Python {version.major}.{version.minor}.{version.micro}")
            return True
        else:
            error = f"Требуется Python 3.10+, найдено {version.major}.{version.minor}"
            print_error(error)
            self.errors.append(error)
            return False

    def check_dependencies(self):
        """Проверка установленных зависимостей"""
        print_step("Проверка зависимостей")

        required_packages = [
            'fastapi',
            'uvicorn',
            'flask',
            'flask_socketio',
            'gunicorn',
            'requests',
            'numpy',
            'scipy',
            'pillow',
            'reportlab',
            'scikit-learn',
        ]

        missing = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                print_success(f"{package}")
            except ImportError:
                print_warning(f"{package} не найден")
                missing.append(package)

        if missing:
            self.warnings.append(f"Отсутствуют пакеты: {', '.join(missing)}")
            print_warning(f"Установите: pip install {' '.join(missing)}")
            return False

        self.success.append("Все зависимости установлены")
        return True

    def check_env_file(self):
        """Проверка .env файла"""
        print_step("Проверка .env файла")

        env_file = self.project_root / '.env'

        if not env_file.exists():
            error = ".env файл не найден"
            print_error(error)
            self.errors.append(error)
            print_warning("Скопируйте .env.example в .env")
            return False

        # Проверка обязательных переменных
        required_vars = ['JWT_SECRET', 'FLASK_SECRET_KEY']

        with open(env_file, 'r') as f:
            content = f.read()

        missing_vars = []
        for var in required_vars:
            if var not in content:
                missing_vars.append(var)

        if missing_vars:
            error = f"Отсутствуют переменные: {', '.join(missing_vars)}"
            print_error(error)
            self.errors.append(error)
            return False

        # Проверка на дефолтные значения
        default_secrets = ['your-secret-key', 'change-me', 'secret-key']
        with open(env_file, 'r') as f:
            for line in f:
                for default in default_secrets:
                    if default in line.lower():
                        warning = f"Используйте дефолтное значение секрета: {line.strip()}"
                        print_warning(warning)
                        self.warnings.append(warning)

        print_success(".env файл найден")
        self.success.append(".env файл конфигурирован")
        return True

    def check_directories(self):
        """Проверка необходимых директорий"""
        print_step("Проверка директорий")

        required_dirs = ['logs', 'data', 'output', 'reports', 'deployment/nginx/ssl']

        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                print_warning(f"Создание директории: {dir_name}")
                dir_path.mkdir(parents=True, exist_ok=True)

        print_success("Все директории существуют")
        self.success.append("Директории созданы/проверены")
        return True

    def check_ssl_certs(self):
        """Проверка SSL сертификатов"""
        print_step("Проверка SSL сертификатов")

        ssl_dir = self.project_root / 'deployment' / 'nginx' / 'ssl'
        crt_file = ssl_dir / 'nanoprobe-lab.local.crt'
        key_file = ssl_dir / 'nanoprobe-lab.local.key'

        if not crt_file.exists() or not key_file.exists():
            warning = "SSL сертификаты не найдены"
            print_warning(warning)
            self.warnings.append(warning)
            print("  Для генерации выполните:")
            print("  python scripts/generate_ssl_certs.py")
            return False

        print_success("SSL сертификаты найдены")
        self.success.append("SSL сертификаты установлены")
        return True

    def check_ports(self):
        """Проверка доступности портов"""
        print_step("Проверка портов (8000, 5000)")

        import socket

        ports = [8000, 5000]
        available = True

        for port in ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()

            if result == 0:
                print_warning(f"Порт {port} занят")
                self.warnings.append(f"Порт {port} занят")
                available = False
            else:
                print_success(f"Порт {port} свободен")

        if available:
            self.success.append("Порты свободны")

        return available

    def run_all_checks(self):
        """Запуск всех проверок"""
        print_header("Проверка готовности к развёртыванию")

        checks = [
            self.check_python_version,
            self.check_dependencies,
            self.check_env_file,
            self.check_directories,
            self.check_ssl_certs,
            self.check_ports,
        ]

        results = []
        for check in checks:
            try:
                results.append(check())
            except Exception as e:
                print_error(f"Ошибка проверки: {e}")
                results.append(False)

        # Итоги
        print(f"\n{Colors.BOLD}Результаты:{Colors.END}")
        print(f"  Успешно: {sum(1 for r in results if r)}")
        print(f"  Ошибки: {sum(1 for r in results if not r)}")

        if self.warnings:
            print(f"\n{Colors.YELLOW}Предупреждения ({len(self.warnings)}):{Colors.END}")
            for w in self.warnings:
                print(f"  - {w}")

        if self.errors:
            print(f"\n{Colors.RED}Ошибки ({len(self.errors)}):{Colors.END}")
            for e in self.errors:
                print(f"  - {e}")
            return False

        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ Готово к развёртыванию!{Colors.END}\n")
        return True


class DeploymentManager:
    """Менеджер развёртывания"""

    def __init__(self, use_docker=False):
        self.project_root = Path(__file__).parent.parent
        self.use_docker = use_docker

    def setup_ssl(self):
        """Настройка SSL"""
        print_step("Настройка SSL")

        ssl_dir = self.project_root / 'deployment' / 'nginx' / 'ssl'
        ssl_dir.mkdir(parents=True, exist_ok=True)

        # Генерация сертификатов
        script = self.project_root / 'scripts' / 'generate_ssl_certs.py'
        if script.exists():
            subprocess.run([sys.executable, str(script)], cwd=str(self.project_root))
        else:
            print_error("Скрипт генерации SSL не найден")
            return False

        return True

    def start_docker(self):
        """Запуск Docker Compose"""
        print_step("Запуск Docker Compose")

        compose_file = self.project_root / 'deployment' / 'docker' / 'docker-compose.prod.yml'

        if not compose_file.exists():
            print_error("docker-compose.prod.yml не найден")
            return False

        try:
            # Запуск
            subprocess.run(
                ['docker-compose', '-f', str(compose_file), 'up', '-d'],
                cwd=str(self.project_root)
            )
            print_success("Docker контейнеры запущены")
            return True
        except Exception as e:
            print_error(f"Ошибка запуска Docker: {e}")
            return False

    def start_manual(self):
        """Ручной запуск (без Docker)"""
        print_step("Ручной запуск сервисов")

        print("\nДля запуска выполните:")
        print(f"\n{Colors.CYAN}# Терминал 1 - FastAPI{Colors.END}")
        print("gunicorn api.main:app \\")
        print("  -c deployment/gunicorn/gunicorn_conf.py \\")
        print("  --bind 0.0.0.0:8000 \\")
        print("  --workers 4 \\")
        print("  --worker-class uvicorn.workers.UvicornWorker")

        print(f"\n{Colors.CYAN}# Терминал 2 - Flask{Colors.END}")
        print("gunicorn src.web.web_dashboard_integrated:app \\")
        print("  --bind 0.0.0.0:5000 \\")
        print("  --workers 2 \\")
        print("  --threads 4 \\")
        print("  --worker-class gthread")

        print(f"\n{Colors.CYAN}# Терминал 3 - Nginx (если нужен){Colors.END}")
        print("sudo systemctl restart nginx")

        return True

    def verify_deployment(self):
        """Проверка развёртывания"""
        print_step("Проверка развёртывания")

        import time
        import requests

        time.sleep(5)  # Ждём запуска

        endpoints = {
            'FastAPI': 'http://localhost:8000/health',
            'Flask': 'http://localhost:5000/api/system_info',
        }

        all_healthy = True

        for name, url in endpoints.items():
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    print_success(f"{name}: healthy")
                else:
                    print_error(f"{name}: unhealthy ({response.status_code})")
                    all_healthy = False
            except requests.RequestException as e:
                print_error(f"{name}: unreachable ({e})")
                all_healthy = False

        return all_healthy

    def deploy(self):
        """Полное развёртывание"""
        print_header("Развёртывание Nanoprobe Simulation Lab")

        if self.use_docker:
            success = self.start_docker()
        else:
            success = self.start_manual()

        if success:
            self.verify_deployment()

        return success


def main():
    parser = argparse.ArgumentParser(
        description='Развёртывание Nanoprobe Simulation Lab',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python deploy.py --check          # Проверка готовности
  python deploy.py --setup          # Настройка и запуск
  python deploy.py --docker         # Запуск через Docker
  python deploy.py --verify         # Проверка развёртывания
        """
    )

    parser.add_argument(
        '--check',
        action='store_true',
        help='Проверка готовности к развёртыванию'
    )
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Настройка и развёртывание'
    )
    parser.add_argument(
        '--docker',
        action='store_true',
        help='Использовать Docker Compose'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Проверка развёртывания'
    )

    args = parser.parse_args()

    # Если нет аргументов, показываем справку
    if not any([args.check, args.setup, args.docker, args.verify]):
        parser.print_help()
        return 0

    # Проверка готовности
    if args.check:
        checker = DeploymentChecker()
        success = checker.run_all_checks()
        return 0 if success else 1

    # Развёртывание
    if args.setup:
        manager = DeploymentManager(use_docker=args.docker)

        # Предварительная проверка
        checker = DeploymentChecker()
        if not checker.run_all_checks():
            print_error("Проверка не пройдена. Исправьте ошибки перед развёртыванием.")
            return 1

        # Настройка SSL
        manager.setup_ssl()

        # Развёртывание
        success = manager.deploy()
        return 0 if success else 1

    # Проверка развёртывания
    if args.verify:
        manager = DeploymentManager()
        success = manager.verify_deployment()
        return 0 if success else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
