"""
Utils Реорганизация

Автоматическая миграция файлов utils по функциональным папкам.

Целевая структура:
utils/
├── core/           # Базовые утилиты
├── api/            # API клиенты
├── database/       # Database utilities
├── security/       # Security
├── monitoring/     # Monitoring
├── performance/    # Performance
├── caching/        # Caching
├── ai/             # AI/ML (уже есть)
├── config/         # Configuration (уже есть)
├── data/           # Data management (уже есть)
├── reporting/      # Reports (уже есть)
└── deployment/     # Deployment (уже есть)
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List

# Корневая директория utils
UTILS_ROOT = Path(__file__).parent.parent / "utils"

# Маппинг файлов в целевые директории
FILE_MAPPING: Dict[str, str] = {
    # Core utilities
    "cli_utils.py": "core",
    "error_handler.py": "core",
    
    # API clients
    "nasa_api_client.py": "api",
    "space_image_downloader.py": "api",
    
    # Database
    "database.py": "database",
    
    # Security
    "two_factor_auth.py": "security",
    "rate_limiter.py": "security",
    
    # Monitoring
    "system_monitor.py": "monitoring",
    "enhanced_monitor.py": "monitoring",
    "system_health_monitor.py": "monitoring",
    "performance_monitor.py": "monitoring",
    "performance_monitoring_center.py": "monitoring",
    "realtime_dashboard.py": "monitoring",
    
    # Performance
    "performance_profiler.py": "performance",
    "performance_benchmark.py": "performance",
    "performance_analytics_dashboard.py": "performance",
    "performance_verification_framework.py": "performance",
    "profiler.py": "performance",
    "memory_tracker.py": "performance",
    "resource_optimizer.py": "performance",
    "ai_resource_optimizer.py": "performance",
    "optimization_orchestrator.py": "performance",
    "optimization_logging_manager.py": "performance",
    "self_healing_system.py": "performance",
    "automated_optimization_scheduler.py": "performance",
    
    # Caching
    "cache_manager.py": "caching",
    "redis_cache.py": "caching",
    "circuit_breaker.py": "caching",
    
    # Data (уже есть папка, перемещаем связанные файлы)
    "data_manager.py": "data",
    "data_validator.py": "data",
    "data_integrity.py": "data",
    "data_exporter.py": "data",
    "batch_processor.py": "batch",
    
    # Config (уже есть папка)
    "config_manager.py": "config",
    "config_optimizer.py": "config",
    "config_validator.py": "config",
    
    # Reporting
    "report_generator.py": "reporting",
    "pdf_report_generator.py": "reporting",
    "documentation_generator.py": "reporting",
    
    # AI/ML
    "machine_learning.py": "ai",
    "model_trainer.py": "ai",
    "defect_analyzer.py": "ai",
    "pretrained_defect_analyzer.py": "ai",
    "predictive_analytics_engine.py": "ai",
    
    # Deployment
    "deployment_manager.py": "deployment",
    
    # Logging
    "logger.py": "logging",
    "production_logger.py": "logging",
    "advanced_logger_analyzer.py": "logging",
    
    # Visualization
    "visualizer.py": "visualization",
    "analytics.py": "visualization",
    "spm_realtime_visualizer.py": "visualization",
    "surface_comparator.py": "visualization",
    
    # Simulator
    "simulator_orchestrator.py": "simulator",
    
    # Testing
    "test_framework.py": "testing",
    
    # Code analysis
    "code_analyzer.py": "dev",
}

# Файлы которые остаются в корне
ROOT_FILES = [
    "__init__.py",
    "__pycache__",
]


def create_directory_structure():
    """Создание целевой структуры директорий"""
    directories = [
        "core",
        "api",
        "database",
        "security",
        "monitoring",
        "performance",
        "caching",
        "batch",
        "logging",
        "visualization",
        "simulator",
        "testing",
        "dev",
    ]
    
    # Уже существующие
    existing = ["ai", "config", "data", "reporting", "deployment"]
    
    all_dirs = directories + existing
    
    for dir_name in all_dirs:
        dir_path = UTILS_ROOT / dir_name
        dir_path.mkdir(exist_ok=True)
        
        # Создаём __init__.py
        init_file = dir_path / "__init__.py"
        if not init_file.exists():
            init_file.write_text(f'"""\n{dir_name.title()} Module\n"""\n')
        
        print(f"✓ Директория: {dir_name}")


def migrate_files(dry_run: bool = True):
    """
    Миграция файлов по директориям.
    
    Args:
        dry_run: Если True, только показывает что будет сделано
    """
    migrated = 0
    errors = 0
    
    for filename, target_dir in FILE_MAPPING.items():
        src = UTILS_ROOT / filename
        dst = UTILS_ROOT / target_dir / filename
        
        if not src.exists():
            print(f"⚠ Не найдено: {filename}")
            continue
        
        if dry_run:
            print(f"📁 {filename} → {target_dir}/")
        else:
            try:
                # Перемещение файла
                shutil.move(str(src), str(dst))
                print(f"✓ {filename} → {target_dir}/")
                migrated += 1
            except Exception as e:
                print(f"❌ Error moving {filename}: {e}")
                errors += 1
    
    print(f"\n{'(DRY RUN) ' if dry_run else ''}Migrated: {migrated}, Errors: {errors}")


def update_imports():
    """
    Обновление импортов в проекте.
    Автоматически не делается - требует ручного ревью!
    """
    print("""
⚠️  ВНИМАНИЕ!

После миграции необходимо обновить импорты во всём проекте:

Было:
    from utils.database import DatabaseManager

Стало:
    from utils.database.database import DatabaseManager
    или
    from utils.database import DatabaseManager  # если экспортировать в __init__.py

Рекомендуется:
1. Запустить поиск по всем файлам: "from utils\\."
2. Обновить импорты вручную
3. Запустить тесты для проверки
""")


def main():
    """Основной процесс реорганизации"""
    print("=" * 60)
    print("Utils Реорганизация")
    print("=" * 60)
    print()
    
    # Шаг 1: Создание структуры
    print("Шаг 1: Создание директорий...")
    create_directory_structure()
    print()
    
    # Шаг 2: Dry run миграции
    print("Шаг 2: Анализ миграции (dry run)...")
    migrate_files(dry_run=True)
    print()
    
    # Шаг 3: Предупреждение
    update_imports()
    
    print()
    print("=" * 60)
    print("Для выполнения миграции запустите:")
    print("  python utils_reorganization.py --execute")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--execute":
        print("⚠️  Выполнение миграции...")
        confirm = input("Вы уверены? (yes/no): ")
        if confirm.lower() == "yes":
            create_directory_structure()
            migrate_files(dry_run=False)
            update_imports()
            print("\n✅ Миграция завершена!")
        else:
            print("❌ Отменено")
    else:
        main()
