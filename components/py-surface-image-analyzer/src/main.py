# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Анализатор изображений поверхности
Скрипт для обработки и анализа AFM-изображений
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from image_processor import ImageProcessor, calculate_surface_roughness


def main():
    parser = argparse.ArgumentParser(description='Анализатор изображений поверхности')
    parser.add_argument('--image', '-i', type=str, help='Путь к изображению')
    parser.add_argument('--filter', '-f', type=str, default='gaussian',
                        choices=['gaussian', 'median', 'bilateral'],
                        help='Тип фильтра для шумоподавления')
    parser.add_argument('--edges', '-e', action='store_true', help='Обнаружить края')
    parser.add_argument('--roughness', '-r', action='store_true', help='Рассчитать шероховатость')
    parser.add_argument('--output', '-o', type=str, help='Путь для сохранения результата')
    
    args = parser.parse_args()
    
    print("=== АНАЛИЗАТОР ИЗОБРАЖЕНИЙ ПОВЕРХНОСТИ ===")
    processor = ImageProcessor()
    
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Ошибка: Файл '{image_path}' не найден")
            sys.exit(1)
        
        print(f"Загрузка изображения: {image_path}")
        if not processor.load_image(str(image_path)):
            print("Ошибка загрузки изображения")
            sys.exit(1)
        
        if args.filter:
            print(f"Применение фильтра: {args.filter}")
            processor.apply_noise_reduction(args.filter)
        
        if args.edges:
            print("Обнаружение краев...")
            edges = processor.detect_edges()
            if edges is not None and args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                edges.save(str(output_path))
                print(f"Края сохранены в: {output_path}")
        
        if args.roughness:
            roughness = calculate_surface_roughness(processor.image_data)
            print(f"Шероховатость поверхности: {roughness:.4f}")
        
        print("Анализ завершен")
    else:
        print("Режим демонстрации:")
        print("- Загрузите изображение с помощью --image")
        print("- Примените фильтр с помощью --filter")
        print("- Обнаружьте края с помощью --edges")
        print("- Рассчитайте шероховатость с помощью --roughness")

if __name__ == "__main__":
    main()

