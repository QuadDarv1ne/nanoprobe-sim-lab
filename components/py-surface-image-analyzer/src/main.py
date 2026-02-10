#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пример скрипта для анализатора изображений поверхности
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from image_processor import ImageProcessor, calculate_surface_roughness


def main():
    print("=== АНАЛИЗАТОР ИЗОБРАЖЕНИЙ ПОВЕРХНОСТИ ===")
    print("Инициализация анализатора...")
    
    processor = ImageProcessor()
    
    # Здесь будет код для загрузки и анализа изображения
    print("Анализатор изображений готов к работе")
    print("Для использования загрузите изображение и примените фильтры")
    
    # Пример использования
    # processor.load_image("sample_image.jpg")
    # filtered = processor.apply_noise_reduction("gaussian")
    # edges = processor.detect_edges()
    
    print("Пример анализа изображения завершен")


if __name__ == "__main__":
    main()