"""Модуль обработки изображений."""

import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime


class ImageProcessor:
    """Класс для обработки изображений поверхности."""

    def __init__(self):
        """Инициализирует процессор изображений."""
        self.image: Optional[np.ndarray] = None
        self.processed_image: Optional[np.ndarray] = None
        self.image_path: Optional[str] = None
        self.metadata: Dict[str, Any] = {}

    def load_image(self, filepath: str) -> bool:
        """
        Загружает изображение из файла

        Args:
            filepath: Путь к файлу изображения

        Returns:
            bool: True если изображение успешно загружено, иначе False
        """
        try:
            path = Path(filepath)
            if not path.exists():
                print(f"Файл не найден: {filepath}")
                return False

            if path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                print(f"Неподдерживаемый формат: {path.suffix}")
                return False

            self.image = cv2.imread(filepath)
            if self.image is not None:
                self.image_path = filepath
                self.metadata['filepath'] = filepath
                self.metadata['loaded_at'] = datetime.now().isoformat()
                self.metadata['original_shape'] = self.image.shape
                return True
            else:
                print(f"Не удалось загрузить изображение: {filepath}")
                return False
        except Exception as e:
            print(f"Ошибка при загрузке изображения: {str(e)}")
            return False

    def apply_noise_reduction(self, method: str = "gaussian") -> Optional[np.ndarray]:
        """
        Применяет методы уменьшения шума к изображению

        Args:
            method: Метод фильтрации ("gaussian", "median", "bilateral")

        Returns:
            np.ndarray: Обработанное изображение или None при ошибке
        """
        if self.image is None:
            print("Сначала загрузите изображение")
            return None

        valid_methods = {"gaussian", "median", "bilateral"}
        if method not in valid_methods:
            print(f"Неизвестный метод фильтрации: {method}. Доступные: {valid_methods}")
            return None

        if method == "gaussian":
            # Применяем гауссовый фильтр для уменьшения шума
            filtered = cv2.GaussianBlur(self.image, (5, 5), 0)
        elif method == "median":
            # Применяем медианный фильтр для удаления шума
            filtered = cv2.medianBlur(self.image, 5)
        elif method == "bilateral":
            # Применяем билатеральный фильтр для сохранения краев
            filtered = cv2.bilateralFilter(self.image, 9, 75, 75)

        self.processed_image = filtered
        self.metadata['filter_applied'] = method
        return filtered

    def detect_edges(self, threshold1: int = 100, threshold2: int = 200) -> Optional[np.ndarray]:
        """
        Обнаруживает края на изображении с помощью алгоритма Canny

        Args:
            threshold1: Первый порог для гистерезиса
            threshold2: Второй порог для гистерезиса

        Returns:
            np.ndarray: Изображение с выделенными краями или None при ошибке
        """
        if self.processed_image is None:
            print("Сначала обработайте изображение")
            return None

        if not (0 < threshold1 < threshold2):
            print(f"Некорректные пороги: threshold1={threshold1}, threshold2={threshold2}")
            return None

        try:
            gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, threshold1, threshold2)
            self.metadata['edges_detected'] = True
            self.metadata['edge_thresholds'] = (threshold1, threshold2)
            return edges
        except Exception as e:
            print(f"Ошибка обнаружения краев: {e}")
            return None

    def get_histogram(self) -> Optional[np.ndarray]:
        """
        Вычисляет гистограмму яркости изображения

        Returns:
            np.ndarray: Гистограмма или None
        """
        if self.image is None:
            return None

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        return hist

    def get_statistics(self) -> Optional[Dict[str, float]]:
        """
        Получает статистику изображения

        Returns:
            Dict[str, float]: Статистика (mean, std, min, max)
        """
        if self.image is None:
            return None

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return {
            'mean': float(np.mean(gray)),
            'std': float(np.std(gray)),
            'min': float(np.min(gray)),
            'max': float(np.max(gray)),
            'shape': self.image.shape
        }

    def save_image(self, filepath: str, image: np.ndarray = None) -> bool:
        """
        Сохраняет изображение в файл

        Args:
            filepath: Путь к файлу
            image: Изображение для сохранения (если None, используется processed_image)

        Returns:
            bool: True если успешно
        """
        img_to_save = image if image is not None else self.processed_image
        if img_to_save is None:
            print("Нет изображения для сохранения")
            return False

        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), img_to_save)
            self.metadata['saved_path'] = str(path)
            print(f"Изображение сохранено: {filepath}")
            return True
        except Exception as e:
            print(f"Ошибка сохранения: {e}")
            return False

    def get_metadata(self) -> Dict[str, Any]:
        """Получает метаданные обработки."""
        return self.metadata.copy()


def calculate_surface_roughness(
    image: np.ndarray,
    method: str = "std"
) -> Dict[str, float]:
    """
    Вычисляет шероховатость поверхности на основе статистики изображения

    Args:
        image: Входное изображение
        method: Метод расчета ("std", "ra", "rq", "rz")

    Returns:
        Dict[str, float]: Параметры шероховатости

    Raises:
        ValueError: Если изображение пустое или неверного формата
    """
    if image is None or image.size == 0:
        raise ValueError("Изображение не должно быть пустым")

    if len(image.shape) == 3:
        if image.shape[2] not in [3, 4]:
            raise ValueError(f"Неподдерживаемое количество каналов: {image.shape[2]}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Нормализуем
    gray_norm = gray.astype(np.float64)
    mean_height = np.mean(gray_norm)

    # Ra - среднее арифметическое отклонение
    ra = np.mean(np.abs(gray_norm - mean_height))

    # Rq - среднеквадратичное отклонение
    rq = np.sqrt(np.mean((gray_norm - mean_height) ** 2))

    # Rz - средняя высота по 10 точкам
    sorted_vals = np.sort(gray_norm.flatten())
    n = len(sorted_vals)
    if n >= 10:
        avg_peaks = np.mean(sorted_vals[-5:])
        avg_valleys = np.mean(sorted_vals[:5])
        rz = avg_peaks - avg_valleys
    else:
        rz = np.max(gray_norm) - np.min(gray_norm)

    results = {
        'ra': float(ra),
        'rq': float(rq),
        'rz': float(rz),
        'mean': float(mean_height),
        'min': float(np.min(gray_norm)),
        'max': float(np.max(gray_norm)),
        'std': float(np.std(gray_norm))
    }

    return results
