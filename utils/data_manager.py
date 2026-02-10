# -*- coding: utf-8 -*-
#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3

"""
Модуль управления данными для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет централизованное управление данными
для всех компонентов проекта.
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Dict, List, Any, Optional
import csv
import os
from datetime import datetime

class DataManager:
    """
    Класс для управления данными проекта
    Обеспечивает централизованное хранение, загрузку и сохранение
    данных для всех компонентов проекта.
    """


    def __init__(self, data_dir: str = "data", output_dir: str = "output"):
        """
        Инициализирует менеджер данных

        Args:
            data_dir: Директория для входных данных
            output_dir: Директория для выходных данных
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)

        # Создаем директории если они не существуют
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


    def save_surface_data(self, surface_data: np.ndarray, filename: str) -> bool:
        """
        Сохраняет данные поверхности

        Args:
            surface_data: Данные поверхности в виде numpy массива
            filename: Имя файла для сохранения

        Returns:
            bool: True если успешно сохранено, иначе False
        """
        try:
            filepath = self.output_dir / filename
            np.savetxt(filepath, surface_data)
            print(f"Данные поверхности сохранены: {filepath}")
            return True
        except Exception as e:
            print(f"Ошибка при сохранении данных поверхности: {e}")
            return False


    def load_surface_data(self, filename: str) -> Optional[np.ndarray]:
        """
        Загружает данные поверхности

        Args:
            filename: Имя файла для загрузки

        Returns:
            Numpy массив с данными поверхности или None при ошибке
        """
        try:
            filepath = self.data_dir / filename
            if not filepath.exists():
                filepath = self.output_dir / filename  # Пробуем в output директории

            if filepath.exists():
                data = np.loadtxt(filepath)
                print(f"Данные поверхности загружены: {filepath}")
                return data
            else:
                print(f"Файл с данными поверхности не найден: {filepath}")
                return None
        except Exception as e:
            print(f"Ошибка при загрузке данных поверхности: {e}")
            return None


    def save_scan_results(self, scan_data: np.ndarray, filename: str) -> bool:
        """
        Сохраняет результаты сканирования

        Args:
            scan_data: Данные сканирования в виде numpy массива
            filename: Имя файла для сохранения

        Returns:
            bool: True если успешно сохранено, иначе False
        """
        try:
            filepath = self.output_dir / filename
            np.savetxt(filepath, scan_data)
            print(f"Результаты сканирования сохранены: {filepath}")
            return True
        except Exception as e:
            print(f"Ошибка при сохранении результатов сканирования: {e}")
            return False


    def load_scan_results(self, filename: str) -> Optional[np.ndarray]:
        """
        Загружает результаты сканирования

        Args:
            filename: Имя файла для загрузки

        Returns:
            Numpy массив с результатами сканирования или None при ошибке
        """
        try:
            filepath = self.data_dir / filename
            if not filepath.exists():
                filepath = self.output_dir / filename  # Пробуем в output директории

            if filepath.exists():
                data = np.loadtxt(filepath)
                print(f"Результаты сканирования загружены: {filepath}")
                return data
            else:
                print(f"Файл с результатами сканирования не найден: {filepath}")
                return None
        except Exception as e:
            print(f"Ошибка при загрузке результатов сканирования: {e}")
            return None


    def save_image_analysis_results(self, results: Dict[str, Any], filename: str) -> bool:
        """
        Сохраняет результаты анализа изображений

        Args:
            results: Словарь с результатами анализа
            filename: Имя файла для сохранения

        Returns:
            bool: True если успешно сохранено, иначе False
        """
        try:
            filepath = self.output_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Результаты анализа изображений сохранены: {filepath}")
            return True
        except Exception as e:
            print(f"Ошибка при сохранении результатов анализа изображений: {e}")
            return False


    def load_image_analysis_results(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Загружает результаты анализа изображений

        Args:
            filename: Имя файла для загрузки

        Returns:
            Словарь с результатами анализа или None при ошибке
        """
        try:
            filepath = self.data_dir / filename
            if not filepath.exists():
                filepath = self.output_dir / filename  # Пробуем в output директории

            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                print(f"Результаты анализа изображений загружены: {filepath}")
                return results
            else:
                print(f"Файл с результатами анализа изображений не найден: {filepath}")
                return None
        except Exception as e:
            print(f"Ошибка при загрузке результатов анализа изображений: {e}")
            return None


    def save_sstv_results(self, image_data, filename: str) -> bool:
        """
        Сохраняет результаты SSTV декодирования

        Args:
            image_data: Данные изображения
            filename: Имя файла для сохранения

        Returns:
            bool: True если успешно сохранено, иначе False
        """
        try:
            filepath = self.output_dir / filename
            # Для простоты сохраняем как numpy массив
            if hasattr(image_data, 'save'):
                # Если это объект PIL Image
                image_data.save(filepath)
            else:
                # Если это numpy массив
                np.save(filepath.with_suffix('.npy'), image_data)
            print(f"Результаты SSTV декодирования сохранены: {filepath}")
            return True
        except Exception as e:
            print(f"Ошибка при сохранении результатов SSTV декодирования: {e}")
            return False


    def save_simulation_metadata(self, metadata: Dict[str, Any], filename: str = "simulation_metadata.json") -> bool:
        """
        Сохраняет метаданные симуляции

        Args:
            metadata: Словарь с метаданными
            filename: Имя файла для сохранения

        Returns:
            bool: True если успешно сохранено, иначе False
        """
        try:
            # Добавляем временную метку
            metadata['timestamp'] = datetime.now().isoformat()

            filepath = self.output_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"Метаданные симуляции сохранены: {filepath}")
            return True
        except Exception as e:
            print(f"Ошибка при сохранении метаданных симуляции: {e}")
            return False


    def load_simulation_metadata(self, filename: str = "simulation_metadata.json") -> Optional[Dict[str, Any]]:
        """
        Загружает метаданные симуляции

        Args:
            filename: Имя файла для загрузки

        Returns:
            Словарь с метаданными или None при ошибке
        """
        try:
            filepath = self.data_dir / filename
            if not filepath.exists():
                filepath = self.output_dir / filename  # Пробуем в output директории

            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                print(f"Метаданные симуляции загружены: {filepath}")
                return metadata
            else:
                print(f"Файл с метаданными симуляции не найден: {filepath}")
                return None
        except Exception as e:
            print(f"Ошибка при загрузке метаданных симуляции: {e}")
            return None


    def export_to_csv(self, data: Union[np.ndarray, pd.DataFrame], filename: str) -> bool:
        """
        Экспортирует данные в CSV формат

        Args:
            data: Данные для экспорта
            filename: Имя файла для экспорта

        Returns:
            bool: True если успешно экспортировано, иначе False
        """
        try:
            filepath = self.output_dir / filename

            if isinstance(data, np.ndarray):
                df = pd.DataFrame(data)
                df.to_csv(filepath, index=False)
            elif isinstance(data, pd.DataFrame):
                data.to_csv(filepath, index=False)
            else:
                print("Неподдерживаемый тип данных для экспорта в CSV")
                return False

            print(f"Данные экспортированы в CSV: {filepath}")
            return True
        except Exception as e:
            print(f"Ошибка при экспорте в CSV: {e}")
            return False


    def get_recent_files(self, extension: str = "", count: int = 5) -> List[Path]:
        """
        Получает список последних файлов с заданным расширением

        Args:
            extension: Расширение файлов (например, '.txt', '.csv')
            count: Количество файлов для возврата

        Returns:
            Список путей к файлам
        """
        files = []

        # Ищем в обеих директориях
        for directory in [self.data_dir, self.output_dir]:
            for file in directory.glob(f"*{extension}"):
                files.append((file.stat().st_mtime, file))

        # Сортируем по времени модификации (по убыванию)
        files.sort(key=lambda x: x[0], reverse=True)

        # Возвращаем только пути к файлам
        return [file[1] for file in files[:count]]


    def cleanup_old_files(self, days_old: int = 30) -> int:
        """
        Удаляет старые файлы из директорий данных

        Args:
            days_old: Файлы старше этого количества дней будут удалены

        Returns:
            Количество удаленных файлов
        """
        import time
        current_time = time.time()
        cutoff_time = current_time - (days_old * 24 * 60 * 60)
        deleted_count = 0

        for directory in [self.data_dir, self.output_dir]:
            for file in directory.glob("*"):
                if file.is_file() and file.stat().st_mtime < cutoff_time:
                    try:
                        file.unlink()
                        deleted_count += 1
                        print(f"Удален старый файл: {file}")
                    except Exception as e:
                        print(f"Ошибка при удалении файла {file}: {e}")

        return deleted_count

def main():
    """Главная функция для демонстрации работы менеджера данных"""
    print("=== МЕНЕДЖЕР ДАННЫХ ПРОЕКТА ===")

    # Создаем менеджер данных
    data_manager = DataManager()

    # Создаем тестовые данные поверхности
    test_surface = np.random.rand(10, 10)

    # Сохраняем и загружаем данные поверхности
    if data_manager.save_surface_data(test_surface, "test_surface.txt"):
        loaded_surface = data_manager.load_surface_data("test_surface.txt")
        if loaded_surface is not None:
            print(f"✓ Данные поверхности успешно сохранены и загружены. Размер: {loaded_surface.shape}")

    # Создаем тестовые результаты анализа
    test_results = {
        "surface_roughness": 0.123,
        "defect_count": 5,
        "analysis_date": datetime.now().isoformat(),
        "quality_score": 0.85
    }

    # Сохраняем и загружаем результаты анализа
    if data_manager.save_image_analysis_results(test_results, "test_analysis.json"):
        loaded_results = data_manager.load_image_analysis_results("test_analysis.json")
        if loaded_results:
            print(f"✓ Результаты анализа успешно сохранены и загружены. Качество: {loaded_results.get('quality_score')}")

    # Проверяем последние файлы
    recent_txt_files = data_manager.get_recent_files(".txt", 3)
    print(f"Последние .txt файлы: {[f.name for f in recent_txt_files]}")

    print("Менеджер данных успешно инициализирован")

if __name__ == "__main__":
    main()

