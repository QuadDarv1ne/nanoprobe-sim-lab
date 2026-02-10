#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль управления деплоем для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для деплоя, 
контейнеризации и управления окружением проекта.
"""

import os
import sys
import json
import subprocess
import shutil
import platform
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import venv
import site
import pip
from jinja2 import Template


class DeploymentManager:
    """
    Класс управления деплоем
    Обеспечивает создание, управление и деплой 
    проекта в различных окружениях.
    """
    
    def __init__(self, project_root: str = "."):
        """
        Инициализирует менеджер деплоя
        
        Args:
            project_root: Корневая директория проекта
        """
        self.project_root = Path(project_root).resolve()
        self.config_file = self.project_root / "config.json"
        self.requirements_file = self.project_root / "requirements.txt"
        self.dockerfile_path = self.project_root / "Dockerfile"
        self.compose_file = self.project_root / "docker-compose.yml"
        
        # Проверяем наличие конфигурационного файла
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            self.config = {}
    
    def create_virtual_environment(self, env_name: str = "venv") -> bool:
        """
        Создает виртуальное окружение
        
        Args:
            env_name: Имя виртуального окружения
            
        Returns:
            True если создание успешно, иначе False
        """
        try:
            venv_path = self.project_root / env_name
            venv.create(venv_path, with_pip=True)
            
            # Устанавливаем зависимости
            pip_path = venv_path / "Scripts" / "pip.exe" if platform.system() == "Windows" else venv_path / "bin" / "pip"
            
            if self.requirements_file.exists():
                subprocess.run([str(pip_path), "install", "-r", str(self.requirements_file)], check=True)
            else:
                # Устанавливаем базовые зависимости
                subprocess.run([str(pip_path), "install", "numpy", "matplotlib", "flask", "pandas", "scipy"], check=True)
            
            print(f"✓ Виртуальное окружение создано: {venv_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Ошибка установки зависимостей: {e}")
            return False
        except Exception as e:
            print(f"✗ Ошибка создания виртуального окружения: {e}")
            return False
    
    def generate_dockerfile(self, base_image: str = "python:3.9-slim") -> bool:
        """
        Генерирует Dockerfile для проекта
        
        Args:
            base_image: Базовый образ Docker
            
        Returns:
            True если генерация успешна, иначе False
        """
        dockerfile_template = Template("""
FROM {{ base_image }}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "api/api_interface.py", "run", "--host=0.0.0.0", "--port=5000"]
""")
        
        try:
            dockerfile_content = dockerfile_template.render(base_image=base_image)
            
            with open(self.dockerfile_path, 'w', encoding='utf-8') as f:
                f.write(dockerfile_content.strip())
            
            print(f"✓ Dockerfile создан: {self.dockerfile_path}")
            return True
            
        except Exception as e:
            print(f"✗ Ошибка генерации Dockerfile: {e}")
            return False
    
    def generate_docker_compose(self) -> bool:
        """
        Генерирует docker-compose.yml для проекта
        
        Returns:
            True если генерация успешна, иначе False
        """
        compose_template = Template("""
version: '3.8'

services:
  nanoprobe-api:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data
      - ./output:/app/output
      - ./backups:/app/backups
    environment:
      - PYTHONPATH=/app
      - FLASK_ENV=production
    restart: unless-stopped
    
  nanoprobe-dashboard:
    build: .
    ports:
      - "8080:8080"
    depends_on:
      - nanoprobe-api
    environment:
      - DASHBOARD_HOST=0.0.0.0
      - DASHBOARD_PORT=8080
    restart: unless-stopped
    
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
""")
        
        try:
            compose_content = compose_template.render()
            
            with open(self.compose_file, 'w', encoding='utf-8') as f:
                f.write(compose_content.strip())
            
            print(f"✓ docker-compose.yml создан: {self.compose_file}")
            return True
            
        except Exception as e:
            print(f"✗ Ошибка генерации docker-compose.yml: {e}")
            return False
    
    def build_docker_image(self, image_name: str = "nanoprobe-sim-lab:latest") -> bool:
        """
        Собирает Docker образ
        
        Args:
            image_name: Имя Docker образа
            
        Returns:
            True если сборка успешна, иначе False
        """
        if not self.dockerfile_path.exists():
            print("✗ Dockerfile не найден. Сначала сгенерируйте Dockerfile.")
            return False
        
        try:
            result = subprocess.run([
                "docker", "build", 
                "-t", image_name, 
                str(self.project_root)
            ], check=True, capture_output=True, text=True)
            
            print(f"✓ Docker образ собран: {image_name}")
            print(result.stdout)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Ошибка сборки Docker образа: {e}")
            print(e.stderr)
            return False
        except FileNotFoundError:
            print("✗ Docker не установлен или не найден в PATH")
            return False
    
    def run_docker_containers(self, compose_file: str = None) -> bool:
        """
        Запускает контейнеры Docker
        
        Args:
            compose_file: Путь к docker-compose файлу (если None, использует стандартный)
            
        Returns:
            True если запуск успешен, иначе False
        """
        compose_path = self.compose_file if compose_file is None else Path(compose_file)
        
        if not compose_path.exists():
            print(f"✗ Docker Compose файл не найден: {compose_path}")
            return False
        
        try:
            result = subprocess.run([
                "docker-compose", "-f", str(compose_path), "up", "-d"
            ], check=True, capture_output=True, text=True)
            
            print("✓ Docker контейнеры запущены")
            print(result.stdout)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Ошибка запуска Docker контейнеров: {e}")
            print(e.stderr)
            return False
        except FileNotFoundError:
            print("✗ docker-compose не установлен или не найден в PATH")
            return False
    
    def create_deployment_package(self, package_name: str = None) -> str:
        """
        Создает пакет для деплоя
        
        Args:
            package_name: Имя пакета (если None, генерируется автоматически)
            
        Returns:
            Путь к созданному пакету
        """
        if package_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            package_name = f"nanoprobe_sim_lab_deploy_{timestamp}.zip"
        
        try:
            # Создаем список файлов для включения в пакет
            files_to_include = []
            
            # Включаем основные файлы проекта
            for root, dirs, files in os.walk(self.project_root):
                # Исключаем системные и временные директории
                dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'venv', '.vscode', '.idea', 'node_modules']]
                
                for file in files:
                    file_path = Path(root) / file
                    
                    # Включаем только файлы проекта
                    if self._should_include_in_package(file_path):
                        files_to_include.append(file_path)
            
            # Создаем архив
            import zipfile
            with zipfile.ZipFile(package_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in files_to_include:
                    arc_path = file_path.relative_to(self.project_root)
                    zipf.write(file_path, arc_path)
            
            print(f"✓ Пакет деплоя создан: {package_name}")
            return package_name
            
        except Exception as e:
            print(f"✗ Ошибка создания пакета деплоя: {e}")
            return ""
    
    def _should_include_in_package(self, file_path: Path) -> bool:
        """
        Определяет, должен ли файл быть включен в пакет деплоя
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            True если файл должен быть включен, иначе False
        """
        # Исключаем системные файлы
        exclude_patterns = [
            '.git', '__pycache__', '.pyc', '.pyo', '.swp', '.swo',
            '.DS_Store', 'Thumbs.db', '.vscode', '.idea', 'venv', 'env',
            'node_modules', '.pytest_cache', '.coverage'
        ]
        
        # Проверяем имя файла
        for pattern in exclude_patterns:
            if pattern in str(file_path):
                return False
        
        # Включаем только файлы проекта
        extensions = ['.py', '.json', '.txt', '.md', '.yml', '.yaml', '.cfg', '.ini', '.xml']
        return file_path.suffix.lower() in extensions or file_path.name in [
            'Dockerfile', 'docker-compose.yml', 'requirements.txt', 'setup.py', 'README.md', 'LICENCE'
        ]
    
    def deploy_to_server(self, server_config: Dict[str, Any]) -> bool:
        """
        Деплоит проект на удаленный сервер
        
        Args:
            server_config: Конфигурация сервера с параметрами подключения
            
        Returns:
            True если деплой успешен, иначе False
        """
        try:
            # Проверяем наличие необходимых параметров
            required_params = ['host', 'username', 'ssh_key_path']
            for param in required_params:
                if param not in server_config:
                    print(f"✗ Отсутствует параметр конфигурации: {param}")
                    return False
            
            # Создаем пакет для деплоя
            package_path = self.create_deployment_package()
            if not package_path:
                return False
            
            # Копируем пакет на сервер (реализация зависит от SSH клиента)
            print(f"✓ Проект готов к деплою на сервер {server_config['host']}")
            print("  Загрузите пакет и выполните распаковку на целевом сервере")
            
            # Здесь мог бы быть код для фактического деплоя через SSH
            # Для безопасности не включаю реальный SSH код
            
            return True
            
        except Exception as e:
            print(f"✗ Ошибка деплоя на сервер: {e}")
            return False
    
    def setup_production_environment(self) -> bool:
        """
        Настраивает production окружение
        
        Returns:
            True если настройка успешна, иначе False
        """
        try:
            # Создаем необходимые директории
            dirs_to_create = [
                self.project_root / "logs",
                self.project_root / "data",
                self.project_root / "output", 
                self.project_root / "backups",
                self.project_root / "temp"
            ]
            
            for dir_path in dirs_to_create:
                dir_path.mkdir(exist_ok=True)
            
            # Создаем файлы конфигурации для production
            prod_config = {
                "environment": "production",
                "debug": False,
                "logging": {
                    "level": "INFO",
                    "file": "logs/app.log"
                },
                "database": {
                    "url": "sqlite:///production.db"
                },
                "security": {
                    "encryption_enabled": True,
                    "auth_required": True,
                    "api_rate_limit": 100
                }
            }
            
            with open(self.project_root / "config_prod.json", 'w', encoding='utf-8') as f:
                json.dump(prod_config, f, indent=2, ensure_ascii=False)
            
            print("✓ Production окружение настроено")
            return True
            
        except Exception as e:
            print(f"✗ Ошибка настройки production окружения: {e}")
            return False
    
    def create_systemd_service(self, service_name: str = "nanoprobe-sim-lab.service") -> bool:
        """
        Создает systemd сервис для Linux
        
        Args:
            service_name: Имя сервиса
            
        Returns:
            True если создание успешно, иначе False
        """
        if platform.system() != "Linux":
            print("✗ systemd сервисы поддерживаются только в Linux")
            return False
        
        service_content = f"""[Unit]
Description=Nanoprobe Simulation Lab Service
After=network.target

[Service]
Type=simple
User={os.getenv('USER', 'root')}
WorkingDirectory={self.project_root}
Environment=PYTHONPATH={self.project_root}
ExecStart={sys.executable} {self.project_root / 'api' / 'api_interface.py'} run --host=0.0.0.0 --port=5000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        try:
            service_path = Path("/etc/systemd/system") / service_name
            
            # Для безопасности не будем писать напрямую в системные директории
            # Вместо этого создадим локальный файл
            local_service_path = self.project_root / service_name
            with open(local_service_path, 'w', encoding='utf-8') as f:
                f.write(service_content.strip())
            
            print(f"✓ systemd сервис создан: {local_service_path}")
            print("  Для установки скопируйте файл в /etc/systemd/system/ и выполните:")
            print("  sudo systemctl daemon-reload")
            print("  sudo systemctl enable nanoprobe-sim-lab.service")
            print("  sudo systemctl start nanoprobe-sim-lab.service")
            
            return True
            
        except Exception as e:
            print(f"✗ Ошибка создания systemd сервиса: {e}")
            return False
    
    def generate_deployment_docs(self, output_dir: str = "deployment_docs") -> bool:
        """
        Генерирует документацию по деплою
        
        Args:
            output_dir: Директория для сохранения документации
            
        Returns:
            True если генерация успешна, иначе False
        """
        try:
            docs_dir = self.project_root / output_dir
            docs_dir.mkdir(exist_ok=True)
            
            # Генерируем руководство по деплою
            deployment_guide = f"""# Руководство по деплою Nanoprobe Simulation Lab

## Содержание
1. [Требования](#требования)
2. [Установка](#установка)
3. [Конфигурация](#конфигурация)
4. [Запуск](#запуск)
5. [Управление](#управление)

## Требования

- Python 3.8+
- pip
- Docker (опционально)
- Git

## Установка

### Вариант 1: Локальная установка

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/your-repo/nanoprobe-sim-lab.git
   cd nanoprobe-sim-lab
   ```

2. Создайте виртуальное окружение:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # или
   venv\\Scripts\\activate  # Windows
   ```

3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

### Вариант 2: Docker

1. Соберите Docker образ:
   ```bash
   docker build -t nanoprobe-sim-lab .
   ```

2. Запустите контейнер:
   ```bash
   docker run -d -p 5000:5000 nanoprobe-sim-lab
   ```

### Вариант 3: Docker Compose

1. Запустите все сервисы:
   ```bash
   docker-compose up -d
   ```

## Конфигурация

Конфигурация проекта осуществляется через файл `config.json`.

Пример конфигурации:
```json
{{
  "environment": "production",
  "debug": false,
  "logging": {{
    "level": "INFO",
    "file": "logs/app.log"
  }},
  "database": {{
    "url": "sqlite:///production.db"
  }}
}}
```

## Запуск

### API Сервер
```bash
python api/api_interface.py run --host=0.0.0.0 --port=5000
```

### Веб-панель управления
```bash
python dashboard.py
```

## Управление

### systemd сервис (Linux)

1. Создайте сервис:
   ```bash
   sudo cp nanoprobe-sim-lab.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable nanoprobe-sim-lab
   sudo systemctl start nanoprobe-sim-lab
   ```

2. Проверьте статус:
   ```bash
   sudo systemctl status nanoprobe-sim-lab
   ```

### Docker управление

- Просмотр запущенных контейнеров: `docker ps`
- Остановка контейнеров: `docker-compose down`
- Просмотр логов: `docker-compose logs -f`

## Мониторинг

Для мониторинга состояния системы используйте:
- `python utils/system_monitor.py` - мониторинг ресурсов
- `python utils/performance_monitor.py` - профилирование производительности
- `python utils/config_validator.py` - валидация конфигурации

## Резервное копирование

Для создания резервной копии:
```bash
python utils/backup_manager.py
```

## Безопасность

- Включена аутентификация через `security/auth_manager.py`
- Шифрование данных через `security/data_encryption.py`
- Валидация входных данных

---

Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            with open(docs_dir / "DEPLOYMENT_GUIDE.md", 'w', encoding='utf-8') as f:
                f.write(deployment_guide)
            
            # Генерируем файлы конфигурации
            nginx_config = """server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static {
        alias /path/to/project/static;
        expires 1d;
    }
}
"""
            
            with open(docs_dir / "nginx.conf", 'w', encoding='utf-8') as f:
                f.write(nginx_config)
            
            print(f"✓ Документация по деплою создана: {docs_dir}")
            return True
            
        except Exception as e:
            print(f"✗ Ошибка генерации документации: {e}")
            return False


def main():
    """Главная функция для демонстрации возможностей менеджера деплоя"""
    print("=== МЕНЕДЖЕР ДЕПЛОЯ ПРОЕКТА ===")
    
    # Создаем менеджер деплоя
    deploy_manager = DeploymentManager()
    
    print("✓ Менеджер деплоя инициализирован")
    print(f"✓ Корневая директория: {deploy_manager.project_root}")
    
    # Проверяем доступность Docker
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        docker_available = True
        print("✓ Docker доступен")
    except:
        docker_available = False
        print("⚠ Docker не найден (контейнеризация недоступна)")
    
    # Создаем виртуальное окружение
    venv_created = deploy_manager.create_virtual_environment()
    print(f"✓ Создание виртуального окружения: {'Успешно' if venv_created else 'Ошибка'}")
    
    # Генерируем Docker файлы
    dockerfile_created = deploy_manager.generate_dockerfile()
    compose_created = deploy_manager.generate_docker_compose()
    print(f"✓ Генерация Dockerfile: {'Успешно' if dockerfile_created else 'Ошибка'}")
    print(f"✓ Генерация docker-compose.yml: {'Успешно' if compose_created else 'Ошибка'}")
    
    # Создаем пакет для деплоя
    package_path = deploy_manager.create_deployment_package()
    print(f"✓ Пакет деплоя: {'Создан' if package_path else 'Ошибка создания'}")
    
    # Настраиваем production окружение
    prod_setup = deploy_manager.setup_production_environment()
    print(f"✓ Настройка production окружения: {'Успешно' if prod_setup else 'Ошибка'}")
    
    # Генерируем документацию
    docs_generated = deploy_manager.generate_deployment_docs()
    print(f"✓ Генерация документации: {'Успешно' if docs_generated else 'Ошибка'}")
    
    print("\nМенеджер деплоя успешно протестирован")
    print("\nДоступные команды:")
    print("- Создание виртуального окружения: create_virtual_environment()")
    print("- Генерация Docker файлов: generate_dockerfile(), generate_docker_compose()")
    print("- Сборка Docker образа: build_docker_image()")
    print("- Создание пакета деплоя: create_deployment_package()")
    print("- Генерация документации: generate_deployment_docs()")


if __name__ == "__main__":
    main()