#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль визуализации для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет средства для визуализации данных 
из всех компонентов проекта.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple, Union
import matplotlib.animation as animation
from pathlib import Path


class SurfaceVisualizer:
    """
    Класс для визуализации поверхностей
    Обеспечивает 2D и 3D визуализацию данных поверхности 
    из симулятора СЗМ.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Инициализирует визуализатор поверхности
        
        Args:
            figsize: Размер фигуры для отображения
        """
        self.figsize = figsize
    
    def plot_surface_2d(self, surface_data: np.ndarray, title: str = "Поверхность 2D", 
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Создает 2D визуализацию поверхности
        
        Args:
            surface_data: Данные поверхности в виде numpy массива
            title: Заголовок графика
            save_path: Путь для сохранения изображения (опционально)
            
        Returns:
            Объект matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        im = ax.imshow(surface_data, cmap='viridis', interpolation='bilinear')
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Добавляем цветовую шкалу
        plt.colorbar(im, ax=ax, label='Высота')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_surface_3d(self, surface_data: np.ndarray, title: str = "Поверхность 3D", 
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Создает 3D визуализацию поверхности
        
        Args:
            surface_data: Данные поверхности в виде numpy массива
            title: Заголовок графика
            save_path: Путь для сохранения изображения (опционально)
            
        Returns:
            Объект matplotlib Figure
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        rows, cols = surface_data.shape
        x = np.arange(0, cols, 1)
        y = np.arange(0, rows, 1)
        X, Y = np.meshgrid(x, y)
        
        ax.plot_surface(X, Y, surface_data, cmap='viridis', alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (Высота)')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def animate_scan_process(self, scan_data_list: list, title: str = "Процесс сканирования", 
                           save_path: Optional[str] = None) -> animation.FuncAnimation:
        """
        Создает анимацию процесса сканирования
        
        Args:
            scan_data_list: Список данных сканирования на разных этапах
            title: Заголовок анимации
            save_path: Путь для сохранения анимации (опционально)
            
        Returns:
            Объект matplotlib Animation
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        im = ax.imshow(scan_data_list[0], cmap='viridis', interpolation='bilinear')
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        cbar = plt.colorbar(im, ax=ax, label='Высота')
        
        def update(frame):
            im.set_array(scan_data_list[frame])
            ax.set_title(f'{title} - Шаг {frame + 1}')
            return [im]
        
        ani = animation.FuncAnimation(fig, update, frames=len(scan_data_list), 
                                     interval=200, blit=True, repeat=True)
        
        if save_path:
            ani.save(save_path, writer='pillow', fps=5)
        
        return ani


class ImageAnalyzerVisualizer:
    """
    Класс для визуализации результатов анализа изображений
    Обеспечивает визуализацию результатов фильтрации и анализа 
    изображений поверхности.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Инициализирует визуализатор анализа изображений
        
        Args:
            figsize: Размер фигуры для отображения
        """
        self.figsize = figsize
    
    def plot_comparison(self, original: np.ndarray, processed: np.ndarray, 
                      title_original: str = "Оригинальное изображение",
                      title_processed: str = "Обработанное изображение",
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Сравнивает оригинальное и обработанное изображения
        
        Args:
            original: Оригинальное изображение
            processed: Обработанное изображение
            title_original: Заголовок для оригинального изображения
            title_processed: Заголовок для обработанного изображения
            save_path: Путь для сохранения изображения (опционально)
            
        Returns:
            Объект matplotlib Figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        ax1.imshow(original, cmap='gray')
        ax1.set_title(title_original)
        ax1.axis('off')
        
        ax2.imshow(processed, cmap='gray')
        ax2.set_title(title_processed)
        ax2.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_histograms(self, original: np.ndarray, processed: np.ndarray,
                      title: str = "Гистограммы изображений",
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Строит гистограммы для оригинального и обработанного изображений
        
        Args:
            original: Оригинальное изображение
            processed: Обработанное изображение
            title: Заголовок графика
            save_path: Путь для сохранения изображения (опционально)
            
        Returns:
            Объект matplotlib Figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        ax1.hist(original.flatten(), bins=50, alpha=0.7, color='blue')
        ax1.set_title(f"{title} - Оригинал")
        ax1.set_xlabel('Интенсивность')
        ax1.set_ylabel('Частота')
        
        ax2.hist(processed.flatten(), bins=50, alpha=0.7, color='red')
        ax2.set_title(f"{title} - Обработанное")
        ax2.set_xlabel('Интенсивность')
        ax2.set_ylabel('Частота')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def highlight_defects(self, image: np.ndarray, defects_coords: list,
                         title: str = "Обнаруженные дефекты",
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Выделяет обнаруженные дефекты на изображении
        
        Args:
            image: Исходное изображение
            defects_coords: Список координат дефектов [(x1, y1), (x2, y2), ...]
            title: Заголовок графика
            save_path: Путь для сохранения изображения (опционально)
            
        Returns:
            Объект matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.imshow(image, cmap='gray')
        ax.set_title(title)
        
        # Отмечаем дефекты
        for x, y in defects_coords:
            circle = plt.Circle((x, y), radius=3, fill=False, color='red', linewidth=2)
            ax.add_patch(circle)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class SSTVVisualizer:
    """
    Класс для визуализации результатов SSTV
    Обеспечивает визуализацию декодированных изображений и 
    анализ сигналов SSTV.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Инициализирует визуализатор SSTV
        
        Args:
            figsize: Размер фигура для отображения
        """
        self.figsize = figsize
    
    def plot_decoded_image(self, image_data, title: str = "Декодированное SSTV изображение",
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Отображает декодированное SSTV изображение
        
        Args:
            image_data: Данные изображения (numpy array или PIL Image)
            title: Заголовок графика
            save_path: Путь для сохранения изображения (опционально)
            
        Returns:
            Объект matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if hasattr(image_data, 'size'):  # Это PIL Image
            ax.imshow(image_data)
        else:  # Это numpy array
            ax.imshow(image_data, cmap='gray')
        
        ax.set_title(title)
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_signal_spectrum(self, signal: np.ndarray, sample_rate: int = 44100,
                           title: str = "Спектр сигнала SSTV",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Отображает спектр сигнала SSTV
        
        Args:
            signal: Аудиосигнал
            sample_rate: Частота дискретизации
            title: Заголовок графика
            save_path: Путь для сохранения изображения (опционально)
            
        Returns:
            Объект matplotlib Figure
        """
        # Вычисляем FFT
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
        
        # Берем только положительные частоты
        positive_freqs = freqs[:len(freqs)//2]
        magnitude = np.abs(fft[:len(fft)//2])
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(positive_freqs, magnitude)
        ax.set_title(title)
        ax.set_xlabel('Частота (Гц)')
        ax.set_ylabel('Амплитуда')
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class ProjectVisualizer:
    """
    Центральный класс визуализации проекта
    Объединяет все визуализаторы и предоставляет единый интерфейс 
    для визуализации данных из всех компонентов проекта.
    """
    
    def __init__(self):
        """Инициализирует центральный визуализатор проекта"""
        self.surface_viz = SurfaceVisualizer()
        self.analyzer_viz = ImageAnalyzerVisualizer()
        self.sstv_viz = SSTVVisualizer()
    
    def visualize_all_for_report(self, surface_data: Optional[np.ndarray] = None,
                                original_image: Optional[np.ndarray] = None,
                                processed_image: Optional[np.ndarray] = None,
                                sstv_image = None,
                                output_dir: str = "output") -> bool:
        """
        Создает полный отчет визуализации для всех компонентов
        
        Args:
            surface_data: Данные поверхности
            original_image: Оригинальное изображение
            processed_image: Обработанное изображение
            sstv_image: Декодированное SSTV изображение
            output_dir: Директория для сохранения визуализаций
            
        Returns:
            bool: True если успешно создан отчет, иначе False
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        try:
            # Визуализация поверхности
            if surface_data is not None:
                fig1 = self.surface_viz.plot_surface_2d(
                    surface_data, 
                    "Результаты моделирования СЗМ",
                    output_path / "spm_surface_2d.png"
                )
                plt.close(fig1)
                
                fig2 = self.surface_viz.plot_surface_3d(
                    surface_data, 
                    "3D модель поверхности",
                    output_path / "spm_surface_3d.png"
                )
                plt.close(fig2)
            
            # Визуализация анализа изображений
            if original_image is not None and processed_image is not None:
                fig3 = self.analyzer_viz.plot_comparison(
                    original_image, 
                    processed_image,
                    "Оригинальное изображение",
                    "Обработанное изображение",
                    output_path / "image_comparison.png"
                )
                plt.close(fig3)
                
                fig4 = self.analyzer_viz.plot_histograms(
                    original_image,
                    processed_image,
                    "Гистограммы изображений",
                    output_path / "image_histograms.png"
                )
                plt.close(fig4)
            
            # Визуализация SSTV
            if sstv_image is not None:
                fig5 = self.sstv_viz.plot_decoded_image(
                    sstv_image,
                    "Декодированное SSTV изображение",
                    output_path / "sstv_decoded.png"
                )
                plt.close(fig5)
            
            print(f"Визуализации сохранены в директорию: {output_path}")
            return True
            
        except Exception as e:
            print(f"Ошибка при создании визуализаций: {e}")
            return False


def main():
    """Главная функция для демонстрации возможностей визуализатора"""
    print("=== ВИЗУАЛИЗАТОР ПРОЕКТА ===")
    
    # Создаем визуализатор
    viz = ProjectVisualizer()
    
    # Создаем тестовые данные для демонстрации
    print("Создание тестовых данных...")
    
    # Тестовые данные поверхности
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    surface_data = np.sin(np.sqrt(X**2 + Y**2)) * np.exp(-(X**2 + Y**2)/4)
    
    # Тестовые данные изображений
    original_img = np.random.rand(50, 50)
    processed_img = np.random.rand(50, 50)
    
    # Тестовое SSTV изображение
    sstv_img = np.random.rand(320, 240, 3)
    
    # Создаем визуализации
    success = viz.visualize_all_for_report(
        surface_data=surface_data,
        original_image=original_img,
        processed_image=processed_img,
        sstv_image=sstv_img
    )
    
    if success:
        print("✓ Визуализации успешно созданы")
    else:
        print("✗ Ошибка создания визуализаций")


if __name__ == "__main__":
    main()