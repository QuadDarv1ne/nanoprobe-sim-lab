"""Модуль симулятора сканирующего зондового микроскопа (СЗМ)."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional
import random
import json
from multiprocessing import Pool, cpu_count
from datetime import datetime
from pathlib import Path

# Import database for storing results
try:
    sys_path_added = False
    project_root = Path(__file__).parent.parent.parent.parent
    utils_path = project_root / "utils"
    if str(utils_path) not in __import__('sys').path:
        __import__('sys').path.insert(0, str(utils_path))
        sys_path_added = True

    from database import get_database
    if sys_path_added:
        __import__('sys').path.pop(0)
    HAS_DB = True
except (ImportError, Exception):
    HAS_DB = False


def _scan_line(args):
    """Функция для параллельного сканирования строки."""
    y, width, surface_height_map = args
    line_data = []
    for x in range(width):
        height = float(surface_height_map[y, x])
        line_data.append(height + 0.5)
    return y, np.array(line_data)


class SurfaceModel:
    """Класс для моделирования поверхности."""

    def __init__(self, width: int = 50, height: int = 50):
        """
        Инициализирует модель поверхности.

        Args:
            width: Ширина поверхности
            height: Высота поверхности
        """
        self.width = max(width, 10)  # Минимальный размер 10
        self.height = max(height, 10)
        self.height_map = np.zeros((self.height, self.width))
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
        self.height_map = (self.height_map - np.min(self.height_map)) / (
            np.max(self.height_map) - np.min(self.height_map)
        )

    def _add_craters(self, num_craters: int = 3):
        """Добавляет искусственные кратеры на поверхность"""
        if self.width < 10 or self.height < 10:
            return  # Пропускаем для маленьких поверхностей

        for _ in range(num_craters):
            center_x = random.randint(5, self.width - 5)
            center_y = random.randint(5, self.height - 5)
            radius = random.randint(3, 8)

            for y in range(self.height):
                for x in range(self.width):
                    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                    if dist <= radius:
                        depth = -0.3 * (1 - dist / radius)
                        self.height_map[y, x] += depth

    def _add_mountains(self, num_mountains: int = 2):
        """Добавляет искусственные горы на поверхность"""
        if self.width < 10 or self.height < 10:
            return  # Пропускаем для маленьких поверхностей

        for _ in range(num_mountains):
            center_x = random.randint(5, self.width - 5)
            center_y = random.randint(5, self.height - 5)
            radius = random.randint(4, 10)

            for y in range(self.height):
                for x in range(self.width):
                    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                    if dist <= radius:
                        height = 0.4 * (1 - dist / radius)
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
        plt.imshow(self.height_map, cmap="viridis", interpolation="bilinear")
        plt.colorbar(label="Высота")
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
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

    def scan_surface(self, parallel: bool = True, num_processes: int = None):
        """
        Выполняет сканирование всей поверхности.

        Args:
            parallel: Использовать ли многопроцессорное сканирование
            num_processes: Количество процессов (по умолчанию = число CPU)
        """
        if self.surface is None:
            print("Ошибка: Модель поверхности не установлена!")
            return

        width = self.surface.width
        height = self.surface.height

        if parallel and num_processes != 1:
            self._scan_surface_parallel(num_processes)
        else:
            self._scan_surface_sequential()

    def _scan_surface_sequential(self):
        """Последовательное сканирование поверхности."""
        print("Начинаем последовательное сканирование поверхности...")

        width = self.surface.width
        height = self.surface.height

        for y in range(height):
            for x in range(width):
                self.probe.move_to(x, y)
                adjusted_z = self.probe.adjust_to_surface(self.surface)
                self.scan_data[y, x] = adjusted_z
                self.current_x = x
                self.current_y = y

            if y % max(1, height // 10) == 0:
                progress = (y * 100) // height
                print(f"Прогресс: {progress}%")

        print("Сканирование завершено!")

    def _scan_surface_parallel(self, num_processes: int = None):
        """Параллельное сканирование поверхности с использованием multiprocessing."""
        if num_processes is None:
            num_processes = cpu_count()

        print(f"Начинаем параллельное сканирование ({num_processes} процессов)...")

        # Подготавливаем данные для параллельной обработки
        tasks = [
            (y, self.surface.width, self.surface.height_map)
            for y in range(self.surface.height)
        ]

        # Выполняем сканирование параллельно
        with Pool(processes=num_processes) as pool:
            results = pool.map(_scan_line, tasks)

        # Собираем результаты
        for y, line_data in results:
            self.scan_data[y, :] = line_data

        print("Параллельное сканирование завершено!")

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
            
            # Сохраняем в базу данных
            if HAS_DB:
                try:
                    db = get_database()
                    db.add_scan_result(
                        scan_type="spm",
                        surface_type="simulated",
                        width=self.scan_data.shape[1] if len(self.scan_data.shape) > 1 else self.scan_data.shape[0],
                        height=self.scan_data.shape[0],
                        file_path=filename,
                        metadata={
                            'scan_speed': self.probe.scan_speed,
                            'timestamp': datetime.now().isoformat()
                        }
                    )
                    print("Результаты также сохранены в базу данных")
                except Exception as e:
                    print(f"Предупреждение: Не удалось сохранить в БД: {e}")
            
            return True
        except Exception as e:
            print(f"Ошибка при сохранении результатов сканирования: {str(e)}")
            return False

    def visualize_scan_results(self, title: str = "Результаты сканирования", save_path: str = None):
        """
        Визуализирует результаты сканирования

        Args:
            title: Заголовок графика
            save_path: Путь для сохранения изображения (если None, показывается график)
        """
        if self.scan_data is None:
            print("Нет данных сканирования для визуализации")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 2D карта высот
        im1 = ax1.imshow(self.scan_data, cmap="viridis", interpolation="bilinear")
        ax1.set_title("2D карта высот")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        plt.colorbar(im1, ax=ax1, label="Высота")

        # 3D поверхность
        ax2 = fig.add_subplot(122, projection='3d')
        y = np.arange(self.scan_data.shape[0])
        x = np.arange(self.scan_data.shape[1])
        X, Y = np.meshgrid(x, y)
        ax2.plot_surface(X, Y, self.scan_data, cmap='viridis', alpha=0.8)
        ax2.set_title("3D поверхность")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Высота")

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Визуализация сохранена: {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Главная функция для демонстрации работы симулятора СЗМ"""
    import argparse

    parser = argparse.ArgumentParser(description="Симулятор СЗМ")
    parser.add_argument(
        "--size", "-s", type=int, default=50, help="Размер поверхности (по умолчанию: 50)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="output/scan_results.txt", help="Файл для результатов"
    )
    parser.add_argument("--image", "-i", type=str, default="output/scan_visualization.png", help="Файл для визуализации")
    parser.add_argument("--no-visualize", action="store_true", help="Отключить визуализацию")
    parser.add_argument("--parallel", "-p", action="store_true", help="Использовать параллельное сканирование")

    args = parser.parse_args()

    print("=" * 50)
    print("    СИМУЛЯТОР АППАРАТНОГО ОБЕСПЕЧЕНИЯ СЗМ (Python)")
    print("         (Scanning Probe Microscope Simulator)")
    print("=" * 50)

    size = max(args.size, 10)
    surface = SurfaceModel(size, size)
    print(f"Создана модель поверхности размером {surface.width}x{surface.height}")

    # Сохраняем модель поверхности
    Path("output").mkdir(parents=True, exist_ok=True)
    surface.save_to_file("output/surface_model.txt")

    # Контроллер и сканирование
    controller = SPMController()
    controller.set_surface(surface)
    
    print(f"Запуск сканирования (параллельное: {args.parallel})...")
    controller.scan_surface(parallel=args.parallel)

    # Сохранение результатов
    controller.save_scan_results(args.output)
    print(f"Результаты сохранены в: {args.output}")

    # Визуализация
    if not args.no_visualize:
        controller.visualize_scan_results(
            "Результаты сканирования СЗМ",
            save_path=args.image
        )

    # Статистика
    if HAS_DB:
        try:
            db = get_database()
            stats = db.get_statistics()
            print(f"\nСтатистика БД: {stats.get('total_scans', 0)} сканирований всего")
        except Exception:
            pass

    print("\nСимуляция завершена. Результаты сохранены.")


if __name__ == "__main__":
    main()
