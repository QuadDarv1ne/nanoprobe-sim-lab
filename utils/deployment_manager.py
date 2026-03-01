# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Модуль управления развертыванием для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для управления развертыванием,
контейнеризацией и оркестрацией проекта.
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import yaml
from dataclasses import dataclass, asdict

# Опциональный импорт Docker
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    docker = None

@dataclass
class DeploymentConfig:
    """Конфигурация развертывания"""
    project_name: str
    version: str
    environment: str  # dev, staging, prod
    docker_image: str
    docker_tag: str
    ports: List[str]
    volumes: List[str]
    environment_vars: Dict[str, str]
    dependencies: List[str]

class DeploymentManager:
    """
    Класс менеджера развертывания
    Обеспечивает управление развертыванием, контейнеризацией
    и оркестрацией проекта.
    """


    def __init__(self, project_root: str = "."):
        """
        Инициализирует менеджер развертывания

        Args:
            project_root: Корневая директория проекта
        """
        self.project_root = Path(project_root).resolve()
        self.deployment_dir = self.project_root / "deployment"
        self.deployment_dir.mkdir(exist_ok=True)

        # Проверяем наличие Docker
        self.docker_available = self._check_docker() and DOCKER_AVAILABLE


    def _check_docker(self) -> bool:
        """Проверяет доступность Docker"""
        try:
            subprocess.run(['docker', '--version'],
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False


    def create_virtual_environment(self, env_name: str = "venv",

                                 python_version: str = "3.9") -> bool:
        """
        Создает виртуальное окружение

        Args:
            env_name: Имя виртуального окружения
            python_version: Версия Python

        Returns:
            Успешность создания
        """
        env_path = self.project_root / env_name

        try:
            # Создаем виртуальное окружение
            subprocess.run([sys.executable, '-m', 'venv', str(env_path)],
                         check=True)

            # Активируем и устанавливаем зависимости
            if os.name == 'nt':  # Windows
                pip_path = env_path / 'Scripts' / 'pip.exe'
            else:  # Unix/Linux/Mac
                pip_path = env_path / 'bin' / 'pip'

            # Устанавливаем зависимости
            requirements_path = self.project_root / 'requirements.txt'
            if requirements_path.exists():
                subprocess.run([str(pip_path), 'install', '-r', str(requirements_path)],
                             check=True)

            print(f"✓ Виртуальное окружение создано: {env_path}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Ошибка создания виртуального окружения: {e}")
            return False
        except Exception as e:
            print(f"❌ Неожиданная ошибка: {e}")
            return False


    def generate_dockerfile(self, output_path: str = None) -> str:
        """
        Генерирует Dockerfile для проекта

        Args:
            output_path: Путь для сохранения Dockerfile

        Returns:
            Путь к сгенерированному Dockerfile
        """
        if output_path is None:
            output_path = str(self.deployment_dir / "Dockerfile")

        dockerfile_content = """# Dockerfile для Лаборатории моделирования нанозонда
FROM python:3.9-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    cmake \\
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копирование файлов зависимостей
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Создание необходимых директорий
RUN mkdir -p data logs output

# Открытие портов
EXPOSE 5000 8000

# Команда по умолчанию
CMD ["python", "start.py", "help"]
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(dockerfile_content)

        print(f"✓ Dockerfile сгенерирован: {output_path}")
        return output_path


    def generate_docker_compose(self, output_path: str = None) -> str:
        """
        Генерирует docker-compose.yml для проекта

        Args:
            output_path: Путь для сохранения docker-compose.yml

        Returns:
            Путь к сгенерированному docker-compose.yml
        """
        if output_path is None:
            output_path = str(self.deployment_dir / "docker-compose.yml")

        compose_content = """version: '3.8'

services:
  nanoprobe-lab:
    build: .
    container_name: nanoprobe-simulation-lab
    ports:
      - "5000:5000"
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./output:/app/output
    environment:
      - PYTHONPATH=/app
      - FLASK_ENV=production
    restart: unless-stopped
    networks:
      - nanoprobe-network

  # Дополнительный сервис для мониторинга (опционально)
  monitoring:
    image: prom/prometheus:latest
    container_name: nanoprobe-monitoring
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    networks:
      - nanoprobe-network
    depends_on:
      - nanoprobe-lab

networks:
  nanoprobe-network:
    driver: bridge

volumes:
  nanoprobe-data:
  nanoprobe-logs:
  nanoprobe-output:
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(compose_content)

        print(f"✓ docker-compose.yml сгенерирован: {output_path}")
        return output_path



    def build_docker_image(self, image_name: str = "nanoprobe-lab",
                          tag: str = "latest") -> bool:
        """
        Собирает Docker образ

        Args:
            image_name: Имя образа
            tag: Тег образа

        Returns:
            Успешность сборки
        """
        if not self.docker_available:
            print("❌ Docker недоступен")
            return False

        try:
            # Генерируем Dockerfile если его нет
            dockerfile_path = self.deployment_dir / "Dockerfile"
            if not dockerfile_path.exists():
                self.generate_dockerfile()

            # Собираем образ
            full_image_name = f"{image_name}:{tag}"
            print(f"Сборка Docker образа: {full_image_name}")

            subprocess.run(['docker', 'build', '-t', full_image_name, '.'],
                         cwd=self.project_root, check=True)

            print(f"✓ Docker образ собран: {full_image_name}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Ошибка сборки Docker образа: {e}")
            return False
        except Exception as e:
            print(f"❌ Неожиданная ошибка: {e}")
            return False


    def run_container(self, image_name: str = "nanoprobe-lab",
                     tag: str = "latest",
                     ports: List[str] = None) -> bool:
        """
        Запускает контейнер

        Args:
            image_name: Имя образа
            tag: Тег образа
            ports: Список портов для проброса

        Returns:
            Успешность запуска
        """
        if not self.docker_available:
            print("❌ Docker недоступен")
            return False

        if ports is None:
            ports = ["5000:5000", "8000:8000"]

        try:
            full_image_name = f"{image_name}:{tag}"

            # Останавливаем существующий контейнер
            subprocess.run(['docker', 'stop', 'nanoprobe-lab'],
                         capture_output=True)
            subprocess.run(['docker', 'rm', 'nanoprobe-lab'],
                         capture_output=True)

            # Запускаем новый контейнер
            cmd = ['docker', 'run', '-d', '--name', 'nanoprobe-lab']

            # Добавляем порты
            for port in ports:
                cmd.extend(['-p', port])

            # Добавляем volumes
            cmd.extend([
                '-v', f'{self.project_root}/data:/app/data',
                '-v', f'{self.project_root}/logs:/app/logs',
                '-v', f'{self.project_root}/output:/app/output'
            ])

            # Добавляем образ
            cmd.append(full_image_name)

            subprocess.run(cmd, cwd=self.project_root, check=True)

            print(f"✓ Контейнер запущен: nanoprobe-lab")
            print(f"  Доступен по адресу: http://localhost:5000")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Ошибка запуска контейнера: {e}")
            return False
        except Exception as e:
            print(f"❌ Неожиданная ошибка: {e}")

            return False


    def generate_systemd_service(self, service_name: str = "nanoprobe-lab",
                               user: str = "root",
                               output_path: str = None) -> str:
        """
        Генерирует systemd service файл

        Args:
            service_name: Имя сервиса
            user: Пользователь для запуска
            output_path: Путь для сохранения service файла

        Returns:
            Путь к сгенерированному service файлу
        """
        if output_path is None:
            output_path = str(self.deployment_dir / f"{service_name}.service")

        service_content = f"""[Unit]
Description=Nanoprobe Simulation Lab Service
After=network.target

[Service]
Type=simple
User={user}
WorkingDirectory={self.project_root}
Environment=PYTHONPATH={self.project_root}
ExecStart={sys.executable} {self.project_root}/start.py web
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(service_content)

        print(f"✓ Systemd service файл сгенерирован: {output_path}")
        return output_path


    def create_deployment_package(self, package_name: str = None) -> str:
        """
        Создает пакет развертывания

        Args:
            package_name: Имя пакета

        Returns:
            Путь к созданному пакету
        """
        if package_name is None:
            package_name = f"nanoprobe-lab-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        package_dir = self.deployment_dir / package_name
        package_dir.mkdir(exist_ok=True)

        try:
            # Копируем необходимые файлы
            files_to_copy = [
                'start.py',
                'requirements.txt',
                'README.md',
                'config/',
                'src/',
                'utils/',
                'templates/',
                'components/'
            ]

            for item in files_to_copy:
                src_path = self.project_root / item
                dst_path = package_dir / item

                if src_path.exists():
                    if src_path.is_dir():
                        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src_path, dst_path)

            # Генерируем конфигурационные файлы
            self.generate_dockerfile(str(package_dir / "Dockerfile"))
            self.generate_docker_compose(str(package_dir / "docker-compose.yml"))
            self.generate_systemd_service(output_path=str(package_dir / "nanoprobe-lab.service"))

            # Создаем установочный скрипт
            install_script = package_dir / "install.sh"
            with open(install_script, 'w', encoding='utf-8') as f:
                f.write(f"""#!/bin/bash
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
""")

            # Делаем скрипт исполняемым
            if os.name != 'nt':
                install_script.chmod(0o755)

            print(f"✓ Пакет развертывания создан: {package_dir}")
            return str(package_dir)

        except Exception as e:
            print(f"❌ Ошибка создания пакета: {e}")
            return ""


    def generate_kubernetes_manifests(self, output_dir: str = None) -> str:
        """
        Генерирует манифесты Kubernetes

        Args:
            output_dir: Директория для сохранения манифестов

        Returns:
            Путь к директории с манифестами
        """
        if output_dir is None:
            output_dir = str(self.deployment_dir / "k8s")

        k8s_dir = Path(output_dir)
        k8s_dir.mkdir(exist_ok=True)

        # Deployment
        deployment_content = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: nanoprobe-lab
  labels:
    app: nanoprobe-lab
spec:
  replicas: 1
  selector:
    matchLabels:
      app: nanoprobe-lab
  template:
    metadata:
      labels:
        app: nanoprobe-lab
    spec:
      containers:
      - name: nanoprobe-lab
        image: nanoprobe-lab:latest
        ports:
        - containerPort: 5000
        - containerPort: 8000
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: logs-volume
          mountPath: /app/logs
        - name: output-volume
          mountPath: /app/output
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: nanoprobe-data-pvc
      - name: logs-volume
        persistentVolumeClaim:
          claimName: nanoprobe-logs-pvc
      - name: output-volume
        persistentVolumeClaim:
          claimName: nanoprobe-output-pvc
"""

        with open(k8s_dir / "deployment.yaml", 'w', encoding='utf-8') as f:
            f.write(deployment_content)

        # Service
        service_content = """apiVersion: v1
kind: Service
metadata:
  name: nanoprobe-lab-service
spec:
  selector:
    app: nanoprobe-lab
  ports:
  - name: web
    port: 5000
    targetPort: 5000
  - name: api
    port: 8000
    targetPort: 8000
  type: LoadBalancer
"""

        with open(k8s_dir / "service.yaml", 'w', encoding='utf-8') as f:
            f.write(service_content)

        # PVCs
        pvc_content = """apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: nanoprobe-data-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: nanoprobe-logs-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: nanoprobe-output-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
"""

        with open(k8s_dir / "pvcs.yaml", 'w', encoding='utf-8') as f:
            f.write(pvc_content)

        print(f"✓ Манифесты Kubernetes сгенерированы: {k8s_dir}")
        return str(k8s_dir)


    def get_deployment_status(self) -> Dict[str, Any]:
        """Получает статус развертывания"""
        status = {
            'docker_available': self.docker_available,
            'deployment_dir': str(self.deployment_dir),
            'generated_files': [],
            'running_containers': []
        }

        # Проверяем сгенерированные файлы
        if self.deployment_dir.exists():
            for file_path in self.deployment_dir.iterdir():
                if file_path.is_file():
                    status['generated_files'].append(str(file_path.name))

        # Проверяем запущенные контейнеры
        if self.docker_available:
            try:
                result = subprocess.run(['docker', 'ps', '--format',
                                       '{{.Names}}:{{.Status}}'],
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if line:
                            name, status_str = line.split(':', 1)
                            status['running_containers'].append({
                                'name': name,
                                'status': status_str
                            })
            except Exception:
                pass

        return status

def main():
    """Главная функция для демонстрации менеджера развертывания"""
    print("=== МЕНЕДЖЕР РАЗВЕРТЫВАНИЯ ===")

    # Создаем менеджер развертывания
    deploy_manager = DeploymentManager()

    print("✓ Менеджер развертывания инициализирован")
    print(f"✓ Корневая директория: {deploy_manager.project_root}")
    print(f"✓ Docker доступен: {deploy_manager.docker_available}")
    print(f"✓ Директория развертывания: {deploy_manager.deployment_dir}")

    # Показываем статус
    status = deploy_manager.get_deployment_status()
    print(f"\nСтатус развертывания:")
    print(f"  - Сгенерированные файлы: {len(status['generated_files'])}")
    print(f"  - Запущенные контейнеры: {len(status['running_containers'])}")

    # Создаем виртуальное окружение
    print("\nСоздание виртуального окружения...")
    venv_success = deploy_manager.create_virtual_environment()
    print(f"  Результат: {'Успешно' if venv_success else 'Ошибка'}")

    # Генерируем Dockerfile
    print("\nГенерация Dockerfile...")
    dockerfile_path = deploy_manager.generate_dockerfile()
    print(f"  Создан: {dockerfile_path}")

    # Генерируем docker-compose
    print("\nГенерация docker-compose.yml...")
    compose_path = deploy_manager.generate_docker_compose()
    print(f"  Создан: {compose_path}")

    # Генерируем systemd service
    print("\nГенерация systemd service...")
    service_path = deploy_manager.generate_systemd_service()
    print(f"  Создан: {service_path}")

    # Создаем пакет развертывания
    print("\nСоздание пакета развертывания...")
    package_path = deploy_manager.create_deployment_package()
    print(f"  Создан: {package_path}")

    # Генерируем Kubernetes манифесты
    print("\nГенерация Kubernetes манифестов...")
    k8s_path = deploy_manager.generate_kubernetes_manifests()
    print(f"  Созданы в: {k8s_path}")

    print("\nМенеджер развертывания успешно протестирован")
    print("\nДоступные функции:")
    print("- Создание виртуального окружения: create_virtual_environment()")
    print("- Генерация Dockerfile: generate_dockerfile()")
    print("- Генерация docker-compose: generate_docker_compose()")
    print("- Сборка Docker образа: build_docker_image()")
    print("- Запуск контейнера: run_container()")
    print("- Генерация systemd service: generate_systemd_service()")
    print("- Создание пакета развертывания: create_deployment_package()")
    print("- Генерация Kubernetes манифестов: generate_kubernetes_manifests()")
    print("- Получение статуса: get_deployment_status()")

if __name__ == "__main__":
    main()

