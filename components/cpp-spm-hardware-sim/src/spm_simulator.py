# -*- coding: utf-8 -*-
#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3

"""
Модуль симулятора сканирующего зондового микроскопа (СЗМ)
Этот модуль содержит классы и функции для симуляции работы
сканирующего зондового микроскопа.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import random

class SurfaceModel:
    """
    Класс для моделирования поверхности
    Обрабатывает генерацию и загрузку данных о топографии поверхности.
    Поддерживает как процедурную генерацию, так и загрузку на основе файлов.
    """


    def __init__(self, width: int = 50, height: int = 50):
        """
        Инициализирует модель поверхности

        Args:
            width: Ширина поверхности
            height: Высота поверхности
        """
        self.width = width
        self.height = height
        self.height_map = np.zeros((height, width))
        self.generate_surface()


    def generate_surface(self):
        """Генерирует случайную поверхность с заданными характеристиками"""
        # Генерируем базовую поверхность с нормальным распределением
        self.height_map = np.random.normal(0, 0.5, (self.height, self.width))

        # Добавляем несколько "кратеров"
        self._add_craters()

        # Добавляем несколько "гор"
        self._add_mountains()

        # Нормализуем значения
        self.height_map = (self.height_map - np.min(self.height_map)) / (np.max(self.height_map) - np.min(self.height_map))


    def _add_craters(self, num_craters: int = 3):
        """Добавляет искусственные кратеры на поверхность"""
        for _ in range(num_craters):
            center_x = random.randint(5, self.width - 5)
            center_y = random.randint(5, self.height - 5)
            radius = random.randint(3, 8)

            for y in range(self.height):
                for x in range(self.width):
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if dist <= radius:
                        depth = -0.3 * (1 - dist/radius)
                        self.height_map[y, x] += depth


    def _add_mountains(self, num_mountains: int = 2):
        """Добавляет искусственные горы на поверхность"""
        for _ in range(num_mountains):
            center_x = random.randint(5, self.width - 5)
            center_y = random.randint(5, self.height - 5)
            radius = random.randint(4, 10)

            for y in range(self.height):
                for x in range(self.width):
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if dist <= radius:
                        height = 0.4 * (1 - dist/radius)
                        self.height_map[y, x] += height


    def get_height(self, x: int, y: int) -> float:
        """
        Получает высоту в заданной точке

        Args:
            x: Координата X
            y: Координата Y

        Returns:
            Высота в точке (x,y)
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            return float(self.height_map[y, x])
        else:
            return 0.0  # Значение по умолчанию за пределами поверхности


    def save_to_file(self, filename: str) -> bool:
        """
        Сохраняет модель поверхности в файл

        Args:
            filename: Имя файла для сохранения

        Returns:
            bool: True если успешно сохранено, иначе False
        """
        try:
            np.savetxt(filename, self.height_map)
            print(f"Модель поверхности сохранена в файл: {filename}")
            return True
        except Exception as e:
            print(f"Ошибка при сохранении модели поверхности: {str(e)}")
            return False


    def visualize(self, title: str = "Модель поверхности"):
        """
        Визуализирует модель поверхности

        Args:
            title: Заголовок графика
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(self.height_map, cmap='viridis', interpolation='bilinear')
        plt.colorbar(label='Высота')
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

class ProbeModel:
    """
    Класс для модели зонда
    Симулирует физическое движение зонда и взаимодействие с поверхностью,
    включая механизмы обратной связи.
    """


    def __init__(self):
        """Инициализирует модель зонда"""
        self.x = 0
        self.y = 0
        self.z = 10.0  # Начальная высота зонда
        self.scan_speed = 0.1
        self.max_z = 20.0
        self.min_z = 0.1


    def set_position(self, new_x: float, new_y: float, new_z: float):
        """
        Устанавливает позицию зонда

        Args:
            new_x: Новая X-координата
            new_y: Новая Y-координата
            new_z: Новая Z-координата
        """
        self.x = new_x
        self.y = new_y
        self.z = new_z


    def get_position(self) -> Tuple[float, float, float]:
        """
        Получает текущую позицию зонда

        Returns:
            Кортеж с координатами (x, y, z)
        """
        return (self.x, self.y, self.z)


    def move_to(self, target_x: float, target_y: float, target_z: float = None):
        """
        Перемещает зонд к следующей позиции

        Args:
            target_x: Целевая X-координата
            target_y: Целевая Y-координата
            target_z: Целевая Z-координата (если None, используется адаптивная высота)
        """
        self.x = target_x
        self.y = target_y
        if target_z is not None:
            self.z = target_z


    def adjust_to_surface(self, surface: SurfaceModel) -> float:
        """
        Адаптирует высоту зонда к поверхности

        Args:
            surface: Модель поверхности

        Returns:
            Адаптированная высота зонда
        """
        surface_height = surface.get_height(int(self.x), int(self.y))
        adjusted_z = surface_height + 0.5  # Поддерживаем небольшой зазор

        if adjusted_z > self.max_z:
            adjusted_z = self.max_z
        elif adjusted_z < self.min_z:
            adjusted_z = self.min_z

        self.z = adjusted_z
        return self.z

class SPMController:
    """
    Класс для контроллера СЗМ
    Управляет общим процессом сканирования, координирует движение зонда
    и собирает данные.
    """


    def __init__(self):
        """Инициализирует контроллер СЗМ"""
        self.surface = None
        self.probe = ProbeModel()
        self.scan_data = None
        self.current_x = 0
        self.current_y = 0


    def set_surface(self, surface: SurfaceModel):
        """
        Устанавливает модель поверхности для сканирования

        Args:
            surface: Модель поверхности
        """
        self.surface = surface
        self.scan_data = np.zeros((surface.height, surface.width))


    def scan_surface(self):
        """Выполняет сканирование всей поверхности"""
        if self.surface is None:
            print("Ошибка: Модель поверхности не установлена!")
            return

        print("Начинаем сканирование поверхности...")

        width = self.surface.width
        height = self.surface.height

        for y in range(height):
            # Сканируем строку слева направо
            for x in range(width):
                # Устанавливаем позицию зонда
                self.probe.move_to(x, y)

                # Адаптируем высоту зонда к поверхности
                adjusted_z = self.probe.adjust_to_surface(self.surface)

                # Сохраняем данные сканирования (высота зонда как индикатор рельефа)
                self.scan_data[y, x] = adjusted_z

                self.current_x = x
                self.current_y = y

            # Выводим прогресс каждые несколько строк
            if y % max(1, height // 10) == 0:
                progress = (y * 100) // height
                print(f"Прогресс: {progress}%")

        print("Сканирование завершено!")


    def save_scan_results(self, filename: str) -> bool:
        """
        Сохраняет результаты сканирования в файл

        Args:
            filename: Имя файла для сохранения

        Returns:
            bool: True если успешно сохранено, иначе False
        """
        if self.scan_data is None or self.scan_data.size == 0:
            print("Ошибка: Нет данных сканирования для сохранения!")
            return False

        try:
            np.savetxt(filename, self.scan_data)
            print(f"Результаты сканирования сохранены в файл: {filename}")
            return True
        except Exception as e:
            print(f"Ошибка при сохранении результатов сканирования: {str(e)}")
            return False


    def visualize_scan_results(self, title: str = "Результаты сканирования"):
        """
        Визуализирует результаты сканирования

        Args:
            title: Заголовок графика
        """
        if self.scan_data is None:
            print("Нет данных сканирования для визуализации")
            return

        plt.figure(figsize=(10, 8))
        plt.imshow(self.scan_data, cmap='viridis', interpolation='bilinear')
        plt.colorbar(label='Высота зонда')
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

def main():
    """Главная функция для демонстрации работы симулятора СЗМ"""
    print("=" * 50)
    print("    СИМУЛЯТОР АППАРАТНОГО ОБЕСПЕЧЕНИЯ СЗМ (Python)")
    print("         (Scanning Probe Microscope Simulator)")
    print("=" * 50)

    # Создаем модель поверхности 30x30
    surface = SurfaceModel(30, 30)
    print(f"Создана модель поверхности размером {surface.width}x{surface.height}")

    # Сохраняем модель поверхности
    surface.save_to_file("surface_model_python.txt")

    # Визуализируем модель поверхности
    surface.visualize("Модель поверхности для сканирования")

    # Создаем контроллер СЗМ и устанавливаем поверхность
    controller = SPMController()
    controller.set_surface(surface)

    # Выполняем сканирование
    controller.scan_surface()

    # Сохраняем результаты сканирования
    controller.save_scan_results("scan_results_python.txt")

    # Визуализируем результаты сканирования
    controller.visualize_scan_results("Результаты сканирования СЗМ")

    print("Симуляция завершена. Результаты сохранены.")

if __name__ == "__main__":
    main()

