#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль обработки изображений
Этот модуль содержит функции для загрузки, предварительной обработки 
и базовой манипуляции с изображениями поверхности.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class ImageProcessor:
    """
    Класс для обработки изображений поверхности
    Обрабатывает загрузку, предварительную обработку и базовую манипуляцию 
    с изображениями поверхности.
    """
    
    def __init__(self):
        """Инициализирует процессор изображений"""
        self.image = None
        self.processed_image = None
    
    def load_image(self, filepath: str) -> bool:
        """
        Загружает изображение из файла
        
        Args:
            filepath: Путь к файлу изображения
            
        Returns:
            bool: True если изображение успешно загружено, иначе False
        """
        try:
            self.image = cv2.imread(filepath)
            if self.image is not None:
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
            
        if method == "gaussian":
            # Применяем гауссовый фильтр для уменьшения шума
            filtered = cv2.GaussianBlur(self.image, (5, 5), 0)
        elif method == "median":
            # Применяем медианный фильтр для удаления шума
            filtered = cv2.medianBlur(self.image, 5)
        elif method == "bilateral":
            # Применяем билатеральный фильтр для сохранения краев
            filtered = cv2.bilateralFilter(self.image, 9, 75, 75)
        else:
            print(f"Неизвестный метод фильтрации: {method}")
            return None
            
        self.processed_image = filtered
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
            
        gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1, threshold2)
        return edges


def calculate_surface_roughness(image: np.ndarray) -> float:
    """
    Вычисляет шероховатость поверхности на основе статистики изображения
    
    Args:
        image: Входное изображение
        
    Returns:
        float: Значение шероховатости поверхности
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Стандартное отклонение как мера шероховатости
    roughness = float(np.std(gray))
    return roughness