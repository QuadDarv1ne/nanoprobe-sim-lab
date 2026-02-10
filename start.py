#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for Nanoprobe Simulation Lab
This script provides access to all project components through a unified interface.
"""

import sys
import os
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

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == "cli":
        # Run main console
        cli_path = Path("src/cli/main.py")
        if cli_path.exists():
            os.system(f"{sys.executable} {cli_path}")
        else:
            print("Файл main.py не найден")
            
    elif command == "manager":
        # Run project manager
        manager_path = Path("src/cli/project_manager.py")
        if manager_path.exists():
            os.system(f"{sys.executable} {manager_path}")
        else:
            print("Файл project_manager.py не найден")
            
    elif command == "web":
        # Run web dashboard
        web_path = Path("src/web/web_dashboard.py")
        if web_path.exists():
            os.system(f"{sys.executable} {web_path}")
        else:
            print("Файл web_dashboard.py не найден")
            
    elif command == "help":
        show_help()
        
    else:
        print(f"Неизвестная команда: {command}")
        show_help()

if __name__ == "__main__":
    main()