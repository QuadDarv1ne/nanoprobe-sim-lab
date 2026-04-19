"""Модуль управления данными для проекта Лаборатория моделирования нанозонда."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
            logger.info("Surface data saved: %s", filepath)
            return True
        except Exception as e:
            logger.error("Error saving surface data: %s", e)
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
                logger.info("Surface data loaded: %s", filepath)
                return data
            else:
                logger.warning("Surface data file not found: %s", filepath)
                return None
        except Exception as e:
            logger.error("Error loading surface data: %s", e)
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
            logger.info("Scan results saved: %s", filepath)
            return True
        except Exception as e:
            logger.error("Error saving scan results: %s", e)
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
                logger.info("Scan results loaded: %s", filepath)
                return data
            else:
                logger.warning("Scan results file not found: %s", filepath)
                return None
        except Exception as e:
            logger.error("Error loading scan results: %s", e)
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
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info("Image analysis results saved: %s", filepath)
            return True
        except Exception as e:
            logger.error("Error saving image analysis results: %s", e)
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
                with open(filepath, "r", encoding="utf-8") as f:
                    results = json.load(f)
                logger.info("Image analysis results loaded: %s", filepath)
                return results
            else:
                logger.warning("Image analysis results file not found: %s", filepath)
                return None
        except Exception as e:
            logger.error("Error loading image analysis results: %s", e)
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
            if hasattr(image_data, "save"):
                # Если это объект PIL Image
                image_data.save(filepath)
            else:
                # Если это numpy массив
                np.save(filepath.with_suffix(".npy"), image_data)
            logger.info("SSTV decoding results saved: %s", filepath)
            return True
        except Exception as e:
            logger.error("Error saving SSTV decoding results: %s", e)
            return False

    def save_simulation_metadata(
        self, metadata: Dict[str, Any], filename: str = "simulation_metadata.json"
    ) -> bool:
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
            metadata["timestamp"] = datetime.now(timezone.utc).isoformat()

            filepath = self.output_dir / filename
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info("Simulation metadata saved: %s", filepath)
            return True
        except Exception as e:
            logger.error("Error saving simulation metadata: %s", e)
            return False

    def load_simulation_metadata(
        self, filename: str = "simulation_metadata.json"
    ) -> Optional[Dict[str, Any]]:
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
                with open(filepath, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                logger.info("Simulation metadata loaded: %s", filepath)
                return metadata
            else:
                logger.warning("Simulation metadata file not found: %s", filepath)
                return None
        except Exception as e:
            logger.error("Error loading simulation metadata: %s", e)
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
                logger.error("Unsupported data type for CSV export")
                return False

            logger.info("Data exported to CSV: %s", filepath)
            return True
        except Exception as e:
            logger.error("Error exporting to CSV: %s", e)
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
                        logger.info("Deleted old file: %s", file)
                    except Exception as e:
                        logger.error("Error deleting file %s: %s", file, e)

        return deleted_count


def main():
    """Главная функция для демонстрации работы менеджера данных"""
    logger.info("=== МЕНЕДЖЕР ДАННЫХ ПРОЕКТА ===")

    # Создаем менеджер данных
    data_manager = DataManager()

    # Создаем тестовые данные поверхности
    test_surface = np.random.rand(10, 10)

    # Сохраняем и загружаем данные поверхности
    if data_manager.save_surface_data(test_surface, "test_surface.txt"):
        loaded_surface = data_manager.load_surface_data("test_surface.txt")
        if loaded_surface is not None:
            shape = loaded_surface.shape
            logger.info(f"✓ Данные поверхности успешно сохранены и загружены. Размер: {shape}")

    # Создаем тестовые результаты анализа
    test_results = {
        "surface_roughness": 0.123,
        "defect_count": 5,
        "analysis_date": datetime.now(timezone.utc).isoformat(),
        "quality_score": 0.85,
    }

    # Сохраняем и загружаем результаты анализа
    if data_manager.save_image_analysis_results(test_results, "test_analysis.json"):
        loaded_results = data_manager.load_image_analysis_results("test_analysis.json")
        if loaded_results:
            quality = loaded_results.get("quality_score")
            logger.info(f"✓ Результаты анализа успешно сохранены и загружены. Качество: {quality}")

    # Проверяем последние файлы
    recent_txt_files = data_manager.get_recent_files(".txt", 3)
    logger.info(f"Последние .txt файлы: {[f.name for f in recent_txt_files]}")

    logger.info("Менеджер данных успешно инициализирован")


if __name__ == "__main__":
    main()
