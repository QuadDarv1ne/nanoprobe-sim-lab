#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль управления кэшем для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для автоматической 
очистки и управления кэшем проекта.
"""

import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import json
import tempfile
import gc
import psutil
from dataclasses import dataclass


@dataclass
class CacheInfo:
    """Информация о кэше"""
    path: Path
    size_bytes: int
    file_count: int
    last_accessed: datetime
    cache_type: str


class CacheManager:
    """
    Класс менеджера кэша
    Обеспечивает автоматическую очистку и 
    управление кэшем проекта.
    """
    
    def __init__(self, project_root: str = ".", config_file: str = "cache_config.json"):
        """
        Инициализирует менеджер кэша
        
        Args:
            project_root: Корневая директория проекта
            config_file: Файл конфигурации кэша
        """
        self.project_root = Path(project_root).resolve()
        self.config_file = self.project_root / "config" / config_file
        self.cache_config = self._load_config()
        self.cache_directories = self._get_cache_directories()
    
    def _load_config(self) -> Dict:
        """
        Загружает конфигурацию кэша
        
        Returns:
            Словарь с конфигурацией кэша
        """
        default_config = {
            "cache_directories": [
                "temp",
                "cache",
                ".cache",
                "__pycache__",
                ".pytest_cache",
                ".mypy_cache",
                "logs/cache",
                "output/cache"
            ],
            "max_age_days": 7,
            "max_size_mb": 100,
            "auto_cleanup": True,
            "cleanup_schedule": "daily",
            "excluded_patterns": [
                "*.py",
                "*.json",
                "*.txt",
                "config.*"
            ]
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # Объединяем с дефолтной конфигурацией
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            except Exception:
                return default_config
        else:
            # Создаем дефолтный файл конфигурации
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            return default_config
    
    def _get_cache_directories(self) -> List[Path]:
        """
        Получает список директорий кэша
        
        Returns:
            Список путей к директориям кэша
        """
        cache_dirs = []
        
        for dir_name in self.cache_config["cache_directories"]:
            cache_path = self.project_root / dir_name
            if cache_path.exists():
                cache_dirs.append(cache_path)
        
        # Добавляем системные директории кэша Python
        python_cache_dirs = [
            self.project_root / "__pycache__",
            Path(tempfile.gettempdir()) / "nanoprobe_cache"
        ]
        
        for cache_dir in python_cache_dirs:
            if cache_dir.exists():
                cache_dirs.append(cache_dir)
        
        return cache_dirs
    
    def analyze_cache(self) -> List[CacheInfo]:
        """
        Анализирует кэш проекта
        
        Returns:
            Список информации о кэше
        """
        cache_info_list = []
        
        for cache_dir in self.cache_directories:
            if cache_dir.exists():
                size_bytes = 0
                file_count = 0
                oldest_file_time = None
                
                # Рекурсивно сканируем директорию
                for root, dirs, files in os.walk(cache_dir):
                    for file in files:
                        file_path = Path(root) / file
                        try:
                            stat = file_path.stat()
                            size_bytes += stat.st_size
                            file_count += 1
                            
                            # Определяем время последнего доступа
                            access_time = datetime.fromtimestamp(stat.st_atime)
                            if oldest_file_time is None or access_time < oldest_file_time:
                                oldest_file_time = access_time
                        except (OSError, PermissionError):
                            continue
                
                if file_count > 0:
                    cache_info = CacheInfo(
                        path=cache_dir,
                        size_bytes=size_bytes,
                        file_count=file_count,
                        last_accessed=oldest_file_time or datetime.now(),
                        cache_type=self._determine_cache_type(cache_dir)
                    )
                    cache_info_list.append(cache_info)
        
        return cache_info_list
    
    def _determine_cache_type(self, cache_path: Path) -> str:
        """
        Определяет тип кэша по пути
        
        Args:
            cache_path: Путь к директории кэша
            
        Returns:
            Тип кэша
        """
        path_str = str(cache_path).lower()
        
        if "__pycache__" in path_str:
            return "python_bytecode"
        elif ".pytest_cache" in path_str:
            return "pytest_cache"
        elif ".mypy_cache" in path_str:
            return "mypy_cache"
        elif "temp" in path_str:
            return "temporary_files"
        elif "logs" in path_str:
            return "log_cache"
        else:
            return "general_cache"
    
    def cleanup_cache(self, 
                     max_age_days: Optional[int] = None,
                     max_size_mb: Optional[int] = None,
                     force: bool = False) -> Dict[str, Union[int, List[str]]]:
        """
        Очищает кэш проекта
        
        Args:
            max_age_days: Максимальный возраст файлов в днях
            max_size_mb: Максимальный размер кэша в мегабайтах
            force: Принудительная очистка без проверок
            
        Returns:
            Словарь с результатами очистки
        """
        if max_age_days is None:
            max_age_days = self.cache_config["max_age_days"]
        
        if max_size_mb is None:
            max_size_mb = self.cache_config["max_size_mb"]
        
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        max_size_bytes = max_size_mb * 1024 * 1024
        
        deleted_files = []
        deleted_size = 0
        deleted_count = 0
        
        # Сначала анализируем кэш
        cache_info_list = self.analyze_cache()
        total_size = sum(info.size_bytes for info in cache_info_list)
        
        # Если размер превышает лимит или force=True, очищаем
        if force or total_size > max_size_bytes:
            for cache_info in cache_info_list:
                if cache_info.last_accessed < cutoff_time or force:
                    try:
                        if cache_info.path.is_file():
                            file_size = cache_info.path.stat().st_size
                            cache_info.path.unlink()
                            deleted_files.append(str(cache_info.path))
                            deleted_size += file_size
                            deleted_count += 1
                        elif cache_info.path.is_dir():
                            shutil.rmtree(cache_info.path)
                            deleted_files.append(str(cache_info.path))
                            deleted_size += cache_info.size_bytes
                            deleted_count += cache_info.file_count
                    except (OSError, PermissionError) as e:
                        print(f"Ошибка при удалении {cache_info.path}: {e}")
        
        # Очищаем системный кэш Python
        self._cleanup_python_cache()
        
        # Запускаем сборку мусора
        gc.collect()
        
        return {
            "deleted_files": deleted_count,
            "deleted_size_bytes": deleted_size,
            "deleted_size_mb": round(deleted_size / (1024 * 1024), 2),
            "freed_space_mb": round(deleted_size / (1024 * 1024), 2),
            "deleted_paths": deleted_files[:10]  # Показываем первые 10 удаленных путей
        }
    
    def _cleanup_python_cache(self):
        """Очищает системный кэш Python"""
        try:
            # Очищаем __pycache__ директории
            for root, dirs, files in os.walk(self.project_root):
                for dir_name in dirs[:]:  # Копируем список для безопасного удаления
                    if dir_name == "__pycache__":
                        cache_dir = Path(root) / dir_name
                        try:
                            shutil.rmtree(cache_dir)
                            dirs.remove(dir_name)  # Удаляем из списка для предотвращения повторного обхода
                        except (OSError, PermissionError):
                            pass
        except Exception as e:
            print(f"Ошибка при очистке Python кэша: {e}")
    
    def auto_cleanup(self) -> Dict[str, Union[int, List[str]]]:
        """
        Автоматическая очистка кэша по расписанию
        
        Returns:
            Словарь с результатами очистки
        """
        if not self.cache_config.get("auto_cleanup", True):
            return {"status": "auto_cleanup_disabled"}
        
        # Проверяем расписание
        schedule = self.cache_config.get("cleanup_schedule", "daily")
        last_cleanup_file = self.project_root / ".last_cache_cleanup"
        
        should_cleanup = False
        if schedule == "daily":
            if not last_cleanup_file.exists():
                should_cleanup = True
            else:
                try:
                    last_cleanup_time = datetime.fromtimestamp(last_cleanup_file.stat().st_mtime)
                    if datetime.now() - last_cleanup_time > timedelta(days=1):
                        should_cleanup = True
                except (OSError, ValueError):
                    should_cleanup = True
        elif schedule == "weekly":
            if not last_cleanup_file.exists():
                should_cleanup = True
            else:
                try:
                    last_cleanup_time = datetime.fromtimestamp(last_cleanup_file.stat().st_mtime)
                    if datetime.now() - last_cleanup_time > timedelta(weeks=1):
                        should_cleanup = True
                except (OSError, ValueError):
                    should_cleanup = True
        else:  # immediate или всегда
            should_cleanup = True
        
        if should_cleanup:
            result = self.cleanup_cache()
            # Обновляем время последней очистки
            last_cleanup_file.touch()
            return result
        else:
            return {"status": "no_cleanup_needed"}
    
    def get_cache_statistics(self) -> Dict[str, Union[int, float, str]]:
        """
        Получает статистику кэша
        
        Returns:
            Словарь со статистикой кэша
        """
        cache_info_list = self.analyze_cache()
        
        total_size = sum(info.size_bytes for info in cache_info_list)
        total_files = sum(info.file_count for info in cache_info_list)
        
        # Группируем по типам
        cache_by_type = {}
        for info in cache_info_list:
            if info.cache_type not in cache_by_type:
                cache_by_type[info.cache_type] = {"size": 0, "files": 0}
            cache_by_type[info.cache_type]["size"] += info.size_bytes
            cache_by_type[info.cache_type]["files"] += info.file_count
        
        return {
            "total_cache_size_bytes": total_size,
            "total_cache_size_mb": round(total_size / (1024 * 1024), 2),
            "total_files": total_files,
            "cache_directories_count": len(cache_info_list),
            "cache_by_type": cache_by_type,
            "timestamp": datetime.now().isoformat(),
            "auto_cleanup_enabled": self.cache_config.get("auto_cleanup", True),
            "cleanup_schedule": self.cache_config.get("cleanup_schedule", "daily")
        }
    
    def optimize_memory_usage(self) -> Dict[str, Union[int, float]]:
        """
        Оптимизирует использование памяти
        
        Returns:
            Словарь с результатами оптимизации
        """
        # Получаем информацию о памяти до оптимизации
        process = psutil.Process()
        memory_before = process.memory_info().rss
        
        # Запускаем сборку мусора
        collected = gc.collect()
        
        # Очищаем кэш Python
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Получаем информацию о памяти после оптимизации
        memory_after = process.memory_info().rss
        memory_freed = memory_before - memory_after
        
        return {
            "memory_freed_bytes": memory_freed,
            "memory_freed_mb": round(memory_freed / (1024 * 1024), 2),
            "garbage_collected_objects": collected,
            "current_memory_usage_mb": round(memory_after / (1024 * 1024), 2)
        }
    
    def generate_cleanup_report(self, output_path: str = None) -> str:
        """
        Генерирует отчет об очистке кэша
        
        Args:
            output_path: Путь для сохранения отчета (если None, генерируется автоматически)
            
        Returns:
            Путь к созданному отчету
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"cache_cleanup_report_{timestamp}.json"
        
        # Получаем статистику
        stats = self.get_cache_statistics()
        
        # Выполняем очистку
        cleanup_result = self.cleanup_cache()
        
        # Оптимизируем память
        memory_result = self.optimize_memory_usage()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "cache_statistics": stats,
            "cleanup_results": cleanup_result,
            "memory_optimization": memory_result,
            "configuration": self.cache_config
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        return output_path


def main():
    """Главная функция для демонстрации возможностей менеджера кэша"""
    print("=== МЕНЕДЖЕР КЭША ПРОЕКТА ===")
    
    # Создаем менеджер кэша
    cache_manager = CacheManager()
    
    print("✓ Менеджер кэша инициализирован")
    print(f"✓ Корневая директория: {cache_manager.project_root}")
    print(f"✓ Конфигурационный файл: {cache_manager.config_file}")
    
    # Анализируем кэш
    print("\nАнализ кэша проекта...")
    cache_info = cache_manager.analyze_cache()
    print(f"  - Найдено директорий кэша: {len(cache_info)}")
    
    for info in cache_info:
        print(f"    * {info.path.name}: {info.file_count} файлов, {info.size_bytes / (1024*1024):.2f} MB")
    
    # Получаем статистику
    print("\nСтатистика кэша...")
    stats = cache_manager.get_cache_statistics()
    print(f"  - Общий размер кэша: {stats['total_cache_size_mb']} MB")
    print(f"  - Всего файлов: {stats['total_files']}")
    print(f"  - Автоочистка: {'Включена' if stats['auto_cleanup_enabled'] else 'Выключена'}")
    
    # Оптимизируем память
    print("\nОптимизация использования памяти...")
    memory_result = cache_manager.optimize_memory_usage()
    print(f"  - Освобождено памяти: {memory_result['memory_freed_mb']} MB")
    print(f"  - Собрано объектов: {memory_result['garbage_collected_objects']}")
    
    # Автоматическая очистка
    print("\nАвтоматическая очистка кэша...")
    cleanup_result = cache_manager.auto_cleanup()
    if "status" in cleanup_result:
        print(f"  - Статус: {cleanup_result['status']}")
    else:
        print(f"  - Удалено файлов: {cleanup_result['deleted_files']}")
        print(f"  - Освобождено места: {cleanup_result['freed_space_mb']} MB")
    
    # Генерируем отчет
    print("\nГенерация отчета об очистке...")
    report_path = cache_manager.generate_cleanup_report()
    print(f"  - Отчет сохранен: {report_path}")
    
    print("\nМенеджер кэша успешно протестирован")
    print("\nДоступные функции:")
    print("- Анализ кэша: analyze_cache()")
    print("- Очистка кэша: cleanup_cache()")
    print("- Автоочистка: auto_cleanup()")
    print("- Статистика: get_cache_statistics()")
    print("- Оптимизация памяти: optimize_memory_usage()")
    print("- Генерация отчетов: generate_cleanup_report()")


if __name__ == "__main__":
    main()