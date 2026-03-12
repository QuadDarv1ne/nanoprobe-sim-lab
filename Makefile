# Nanoprobe Simulation Lab - Makefile
# Для Windows используйте `mingw32-make` или `nmake`

.PHONY: help install install-dev run run-cli run-manager run-web test lint format clean build validate docs

# По умолчанию - показать справку
help:
	@echo "Nanoprobe Simulation Lab - Доступные команды:"
	@echo ""
	@echo "  Установка:"
	@echo "    make install       - Установить основные зависимости"
	@echo "    make install-dev   - Установить зависимости для разработки"
	@echo ""
	@echo "  Запуск:"
	@echo "    make run           - Запустить главную консоль"
	@echo "    make run-cli       - Запустить консольный интерфейс"
	@echo "    make run-manager   - Запустить менеджер проекта"
	@echo "    make run-web       - Запустить веб-панель"
	@echo ""
	@echo "  Тестирование и качество:"
	@echo "    make test          - Запустить тесты"
	@echo "    make test-cov      - Запустить тесты с покрытием"
	@echo "    make lint          - Проверка кода (flake8)"
	@echo "    make format        - Форматирование кода (black)"
	@echo "    make validate      - Валидация проекта"
	@echo ""
	@echo "  Сборка и очистка:"
	@echo "    make build         - Собрать C++ компоненты"
	@echo "    make clean         - Очистить временные файлы"
	@echo "    make docs          - Сгенерировать документацию"
	@echo ""

# Установка зависимостей
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install black flake8 mypy pytest-cov pre-commit

# Запуск компонентов
run:
	python start.py cli

run-cli:
	python src/cli/main.py

run-manager:
	python src/cli/project_manager.py

run-web:
	python src/web/web_dashboard.py

# Тестирование
test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ --cov=src --cov=utils --cov-report=html

# Качество кода
lint:
	flake8 src/ utils/ components/ --max-line-length=100

format:
	black src/ utils/ components/ --line-length=100

format-check:
	black src/ utils/ components/ --line-length=100 --check

type-check:
	mypy src/ utils/ --ignore-missing-imports

# Валидация
validate:
	python validate_project.py

# Сборка C++ компонентов
build:
	@echo "Сборка C++ компонентов..."
	cd components/cpp-spm-hardware-sim && \
	mkdir -p build && cd build && \
	cmake .. && \
	make

# Очистка
clean:
	@echo "Очистка временных файлов..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	-rm -rf coverage.xml .coverage
	@echo "Очистка завершена"

clean-cache:
	python -c "from utils.cache_manager import CacheManager; cm = CacheManager(); cm.auto_cleanup()"

# Документация
docs:
	python utils/documentation_generator.py
	@echo "Документация сгенерирована в docs/"

# Пре-коммит хуки
pre-commit:
	pre-commit install
	pre-commit run --all-files

# Все проверки разом
check-all: lint format-check type-check test validate
	@echo "Все проверки завершены"
