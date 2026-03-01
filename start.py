# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Main entry point for Nanoprobe Simulation Lab
This script provides access to all project components through a unified interface.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def show_help():
    """Display help information"""
    print("="*60)
    print("           ЛАБОРАТОРИЯ МОДЕЛИРОВАНИЯ НАНОЗОНДА")
    print("        Nanoprobe Simulation Lab - Main Entry Point")
    print("="*60)
    print("\nДОСТУПНЫЕ КОМПОНЕНТЫ:")
    print("  cli/main.py          - Главная консольная утилита")
    print("  cli/project_manager.py - Менеджер проекта")
    print("  cli/dashboard.py     - Интерактивная панель управления")
    print("  web/web_dashboard.py - Веб-панель управления")
    print("\nБЫСТРЫЙ СТАРТ:")
    print("  python start.py cli     # Запустить главную консоль")
    print("  python start.py manager # Запустить менеджер проекта")
    print("  python start.py web     # Запустить веб-панель")
    print("  python start.py help    # Показать эту справку")
    print("="*60)


def run_component(script_path: Path, description: str) -> None:
    """Запуск компонента с обработкой ошибок"""
    if not script_path.exists():
        print(f"Файл {script_path} не найден")
        return

    try:
        subprocess.run([sys.executable, str(script_path)], check=True)
    except KeyboardInterrupt:
        print(f"\n{description} остановлен пользователем")
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при запуске {description}: {e}")
    except Exception as e:
        print(f"Неожиданная ошибка при запуске {description}: {e}")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        show_help()
        return

    command = sys.argv[1].lower()

    if command == "cli":
        run_component(Path("src/cli/main.py"), "Консоль")
    elif command == "manager":
        run_component(Path("src/cli/project_manager.py"), "Менеджер проекта")
    elif command == "web":
        run_component(Path("src/web/web_dashboard.py"), "Веб-панель")
    elif command == "help":
        show_help()
    else:
        print(f"Неизвестная команда: {command}")
        show_help()


if __name__ == "__main__":
    main()

