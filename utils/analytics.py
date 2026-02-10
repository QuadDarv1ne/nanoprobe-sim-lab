# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Модуль аналитики для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для анализа данных
и машинного обучения для результатов симуляции.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json

class SurfaceAnalytics:
    """
    Класс для анализа данных поверхности
    Обеспечивает статистический анализ, кластеризацию и
    предсказательное моделирование для данных поверхности.
    """


    def __init__(self):
        """Инициализирует аналитический модуль для поверхности"""
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        self.regressor = RandomForestRegressor(random_state=42)


    def calculate_surface_properties(self, surface_data: np.ndarray) -> Dict[str, float]:
        """
        Вычисляет свойства поверхности из данных

        Args:
            surface_data: Данные поверхности в виде numpy массива

        Returns:
            Словарь с вычисленными свойствами поверхности
        """
        flat_data = surface_data.flatten()

        properties = {
            "mean_height": float(np.mean(flat_data)),
            "std_height": float(np.std(flat_data)),
            "min_height": float(np.min(flat_data)),
            "max_height": float(np.max(flat_data)),
            "height_range": float(np.max(flat_data) - np.min(flat_data)),
            "surface_roughness_rms": float(np.sqrt(np.mean(flat_data**2))),
            "skewness": float(pd.Series(flat_data).skew()),
            "kurtosis": float(pd.Series(flat_data).kurtosis()),
            "surface_area": float(self._calculate_surface_area(surface_data)),
            "volume": float(np.sum(np.abs(flat_data)))
        }

        return properties


    def _calculate_surface_area(self, surface_data: np.ndarray) -> float:
        """
        Вычисляет площадь поверхности с учетом рельефа

        Args:
            surface_data: Данные поверхности

        Returns:
            Площадь поверхности
        """
        # Приближенное вычисление площади поверхности через градиенты
        grad_x = np.gradient(surface_data, axis=1)
        grad_y = np.gradient(surface_data, axis=0)

        surface_elements = np.sqrt(1 + grad_x**2 + grad_y**2)
        return float(np.sum(surface_elements))


    def cluster_surface_regions(self, surface_data: np.ndarray, n_clusters: int = 3) -> np.ndarray:
        """
        Кластеризует области поверхности по высоте

        Args:
            surface_data: Данные поверхности
            n_clusters: Количество кластеров

        Returns:
            Массив с метками кластеров
        """
        # Подготовка данных для кластеризации
        rows, cols = surface_data.shape
        X = []

        for i in range(rows):
            for j in range(cols):
                X.append([i, j, surface_data[i, j]])  # координаты и высота

        X = np.array(X)

        # Масштабируем данные
        X_scaled = self.scaler.fit_transform(X)

        # Обновляем модель кластеризации
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X_scaled)

        # Возвращаем массив с метками кластеров в форме поверхности
        cluster_map = labels.reshape(rows, cols)
        return cluster_map


    def dimensionality_reduction(self, surface_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Выполняет понижение размерности поверхности с помощью PCA

        Args:
            surface_data: Данные поверхности

        Returns:
            Кортеж с преобразованными данными и объясненной дисперсией
        """
        rows, cols = surface_data.shape
        X = surface_data.reshape(-1, 1)  # Преобразуем в одномерные признаки

        # Добавляем координаты как дополнительные признаки
        coords = []
        for i in range(rows):
            for j in range(cols):
                coords.append([i, j])
        coords = np.array(coords)

        # Объединяем высоту и координаты
        X_combined = np.hstack([coords, X])

        # Масштабируем данные
        X_scaled = self.scaler.fit_transform(X_combined)

        # Применяем PCA
        X_reduced = self.pca.fit_transform(X_scaled)

        return X_reduced, self.pca.explained_variance_ratio_


    def predict_surface_properties(self, features: np.ndarray, target_property: str = "roughness") -> np.ndarray:
        """
        Предсказывает свойства поверхности на основе признаков

        Args:
            features: Массив признаков
            target_property: Целевое свойство для предсказания

        Returns:
            Предсказанные значения
        """
        # Для демонстрации используем случайные данные
        # В реальном приложении здесь будут реальные признаки и целевые значения

        # Создаем искусственные целевые значения для обучения
        y = np.random.rand(features.shape[0])  # Заменить на реальные значения

        # Разделяем данные
        X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

        # Обучаем модель
        self.regressor.fit(X_train, y_train)

        # Делаем предсказания
        predictions = self.regressor.predict(X_test)

        # Возвращаем предсказания и метрики
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        return predictions

class ImageAnalytics:
    """
    Класс для анализа изображений
    Обеспечивает анализ характеристик изображений и
    обнаружение паттернов в данных изображений.
    """


    def __init__(self):
        """Инициализирует аналитический модуль для изображений"""
        pass


    def calculate_image_features(self, image_data: np.ndarray) -> Dict[str, float]:
        """
        Вычисляет признаки изображения

        Args:
            image_data: Данные изображения в виде numpy массива

        Returns:
            Словарь с вычисленными признаками изображения
        """
        if len(image_data.shape) == 3:  # Цветное изображение
            gray = np.mean(image_data, axis=2)
        else:  # Черно-белое изображение
            gray = image_data

        # Преобразуем в одномерный массив для анализа
        flat_data = gray.flatten()

        features = {
            "mean_intensity": float(np.mean(flat_data)),
            "std_intensity": float(np.std(flat_data)),
            "min_intensity": float(np.min(flat_data)),
            "max_intensity": float(np.max(flat_data)),
            "contrast": float(np.std(flat_data) / np.mean(flat_data)) if np.mean(flat_data) != 0 else 0,
            "entropy": self._calculate_entropy(flat_data),
            "homogeneity": self._calculate_homogeneity(gray),
            "energy": float(np.sum(flat_data**2)),
            "correlation": float(np.corrcoef(flat_data[:-1], flat_data[1:])[0, 1]) if len(flat_data) > 1 else 0
        }

        return features


    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Вычисляет энтропию изображения"""
        hist, _ = np.histogram(data, bins=256)
        hist = hist[hist > 0]  # Убираем нулевые значения
        prob = hist / np.sum(hist)
        entropy = -np.sum(prob * np.log2(prob))
        return float(entropy)


    def _calculate_homogeneity(self, image: np.ndarray) -> float:
        """Вычисляет однородность изображения"""
        # Простой метод: среднее значение близости к среднему
        mean_val = np.mean(image)
        homogeneity = 1.0 / (1.0 + np.mean(np.abs(image - mean_val)))
        return float(homogeneity)


    def detect_patterns(self, image_data: np.ndarray) -> Dict[str, Any]:
        """
        Обнаруживает паттерны в изображении

        Args:
            image_data: Данные изображения

        Returns:
            Словарь с обнаруженными паттернами
        """
        if len(image_data.shape) == 3:  # Цветное изображение
            gray = np.mean(image_data, axis=2)
        else:  # Черно-белое изображение
            gray = image_data

        # Обнаружение краев (упрощенный метод)
        gradient_x = np.gradient(gray, axis=1)
        gradient_y = np.gradient(gray, axis=0)
        edges = np.sqrt(gradient_x**2 + gradient_y**2)

        # Подсчет краев
        edge_threshold = np.mean(edges) + 0.5 * np.std(edges)
        edge_pixels = np.sum(edges > edge_threshold)
        total_pixels = gray.size

        # Обнаружение текстур (упрощенный метод)
        texture_measure = np.std(edges)

        patterns = {
            "edge_density": float(edge_pixels / total_pixels),
            "texture_complexity": float(texture_measure),
            "average_edge_strength": float(np.mean(edges)),
            "pattern_regions": self._detect_regions(gray)
        }

        return patterns


    def _detect_regions(self, image: np.ndarray) -> int:
        """Обнаруживает регионы с похожими характеристиками"""
        # Простой метод: подсчет областей с похожими значениями
        # Используем порог для определения "похожести"
        threshold = np.std(image) / 4
        regions = 0

        # Упрощенный подход: считаем количество уникальных значений в окрестности
        # Для настоящей реализации использовать алгоритмы сегментации
        unique_values = len(np.unique(np.round(image / threshold).astype(int)))
        return min(unique_values, 100)  # Ограничение для стабильности

class SSTVAnalytics:
    """
    Класс для анализа SSTV данных
    Обеспечивает анализ характеристик сигналов и
    качество декодирования SSTV.
    """


    def __init__(self):
        """Инициализирует аналитический модуль для SSTV"""
        pass


    def analyze_signal_quality(self, signal_data: np.ndarray, sample_rate: int = 44100) -> Dict[str, float]:
        """
        Анализирует качество SSTV сигнала

        Args:
            signal_data: Данные аудиосигнала
            sample_rate: Частота дискретизации

        Returns:
            Словарь с метриками качества сигнала
        """
        # Вычисляем основные характеристики сигнала
        rms_amplitude = np.sqrt(np.mean(signal_data**2))
        peak_amplitude = np.max(np.abs(signal_data))
        signal_power = np.mean(signal_data**2)

        # Частотный анализ
        fft = np.fft.fft(signal_data)
        frequencies = np.fft.fftfreq(len(signal_data), 1/sample_rate)

        # Берем только положительные частоты
        pos_freq_idx = frequencies > 0
        pos_fft = fft[pos_freq_idx]
        pos_freqs = frequencies[pos_freq_idx]

        # Находим доминирующую частоту
        dominant_freq_idx = np.argmax(np.abs(pos_fft))
        dominant_frequency = pos_freqs[dominant_freq_idx] if len(pos_freqs) > 0 else 0

        # Вычисляем отношение сигнал/шум (приближенно)
        signal_energy = np.sum(signal_data**2)
        noise_estimate = np.std(signal_data[:1000]) if len(signal_data) > 1000 else np.std(signal_data)
        snr = 20 * np.log10(rms_amplitude / noise_estimate) if noise_estimate > 0 else 0

        quality_metrics = {
            "rms_amplitude": float(rms_amplitude),
            "peak_amplitude": float(peak_amplitude),
            "signal_power": float(signal_power),
            "dominant_frequency": float(dominant_frequency),
            "signal_to_noise_ratio_db": float(snr),
            "total_energy": float(signal_energy),
            "zero_crossing_rate": float(self._calculate_zero_crossing_rate(signal_data)),
            "spectral_centroid": float(self._calculate_spectral_centroid(signal_data, sample_rate))
        }

        return quality_metrics


    def _calculate_zero_crossing_rate(self, signal: np.ndarray) -> float:
        """Вычисляет скорость пересечения нуля"""
        zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
        return float(zero_crossings / len(signal))


    def _calculate_spectral_centroid(self, signal: np.ndarray, sample_rate: int) -> float:
        """Вычисляет спектроцентроид сигнала"""
        fft = np.fft.fft(signal)
        magnitude = np.abs(fft[:len(fft)//2])
        freqs = np.fft.fftfreq(len(signal), 1/sample_rate)[:len(fft)//2]

        if np.sum(magnitude) == 0:
            return 0.0

        spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        return float(spectral_centroid)


    def evaluate_decoding_quality(self, original_image: np.ndarray, decoded_image: np.ndarray) -> Dict[str, float]:
        """
        Оценивает качество декодирования SSTV

        Args:
            original_image: Оригинальное изображение
            decoded_image: Декодированное изображение

        Returns:
            Словарь с метриками качества декодирования
        """
        # Приводим изображения к одинаковому размеру
        min_rows = min(original_image.shape[0], decoded_image.shape[0])
        min_cols = min(original_image.shape[1], decoded_image.shape[1])

        orig_crop = original_image[:min_rows, :min_cols]
        dec_crop = decoded_image[:min_rows, :min_cols]

        # MSE (Mean Squared Error)
        mse = np.mean((orig_crop - dec_crop) ** 2)

        # PSNR (Peak Signal-to-Noise Ratio)
        if mse == 0:
            psnr = float('inf')
        else:
            max_pixel = np.max(orig_crop)
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse)) if max_pixel > 0 else 0

        # SSIM (Structural Similarity Index) - упрощенная версия
        mean_orig = np.mean(orig_crop)
        mean_dec = np.mean(dec_crop)
        var_orig = np.var(orig_crop)
        var_dec = np.var(dec_crop)
        covar = np.mean((orig_crop - mean_orig) * (dec_crop - mean_dec))

        c1 = (0.01 * np.max(orig_crop)) ** 2
        c2 = (0.03 * np.max(orig_crop)) ** 2

        ssim = ((2 * mean_orig * mean_dec + c1) * (2 * covar + c2)) / \
               ((mean_orig**2 + mean_dec**2 + c1) * (var_orig + var_dec + c2))

        quality_metrics = {
            "mse": float(mse),
            "psnr": float(psnr),
            "ssim": float(ssim),
            "correlation": float(np.corrcoef(orig_crop.flatten(), dec_crop.flatten())[0, 1]),
            "mean_difference": float(np.mean(np.abs(orig_crop - dec_crop)))
        }

        return quality_metrics

class ProjectAnalytics:
    """
    Центральный класс аналитики проекта
    Объединяет все аналитические модули и предоставляет
    комплексный анализ данных из всех компонентов проекта.
    """


    def __init__(self):
        """Инициализирует центральный аналитический модуль"""
        self.surface_analytics = SurfaceAnalytics()
        self.image_analytics = ImageAnalytics()
        self.sstv_analytics = SSTVAnalytics()


    def generate_comprehensive_report(self, surface_data: Optional[np.ndarray] = None,
    """TODO: Add description"""

                                   image_data: Optional[np.ndarray] = None,
                                   signal_data: Optional[np.ndarray] = None,
                                   sample_rate: int = 44100) -> Dict[str, Any]:
        """
        Генерирует комплексный аналитический отчет

        Args:
            surface_data: Данные поверхности
            image_data: Данные изображения
            signal_data: Данные аудиосигнала
            sample_rate: Частота дискретизации сигнала

        Returns:
            Словарь с комплексным аналитическим отчетом
        """
        report = {
            "timestamp": str(pd.Timestamp.now()),
            "analyses_performed": [],
            "surface_analysis": {},
            "image_analysis": {},
            "sstv_analysis": {}
        }

        if surface_data is not None:
            try:
                report["surface_analysis"] = self.surface_analytics.calculate_surface_properties(surface_data)
                report["analyses_performed"].append("surface")
            except Exception as e:
                print(f"Ошибка анализа поверхности: {e}")

        if image_data is not None:
            try:
                report["image_analysis"] = self.image_analytics.calculate_image_features(image_data)
                report["image_analysis"].update(self.image_analytics.detect_patterns(image_data))
                report["analyses_performed"].append("image")
            except Exception as e:
                print(f"Ошибка анализа изображения: {e}")

        if signal_data is not None:
            try:
                report["sstv_analysis"] = self.sstv_analytics.analyze_signal_quality(signal_data, sample_rate)
                report["analyses_performed"].append("sstv")
            except Exception as e:
                print(f"Ошибка анализа SSTV сигнала: {e}")

        return report

    """TODO: Add description"""


    def visualize_analytics(self, analytics_report: Dict[str, Any],
                          output_path: str = "analytics_report.png"):
        """
        Визуализирует аналитический отчет

        Args:
            analytics_report: Отчет с аналитикой
            output_path: Путь для сохранения визуализации
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Комплексный аналитический отчет проекта', fontsize=16)

        # Визуализация анализа поверхности
        if analytics_report.get("surface_analysis"):
            surface_props = analytics_report["surface_analysis"]
            props_names = list(surface_props.keys())[:6]  # Берем первые 6 свойств
            props_values = [surface_props[prop] for prop in props_names]

            axes[0, 0].bar(range(len(props_names)), props_values)
            axes[0, 0].set_title('Свойства поверхности')
            axes[0, 0].set_xticks(range(len(props_names)))
            axes[0, 0].set_xticklabels(props_names, rotation=45, ha='right')

        # Визуализация анализа изображения
        if analytics_report.get("image_analysis"):
            image_features = analytics_report["image_analysis"]
            feature_names = list(image_features.keys())[:6]  # Берем первые 6 признаков
            feature_values = [image_features[feat] for feat in feature_names]

            axes[0, 1].barh(range(len(feature_names)), feature_values)
            axes[0, 1].set_title('Признаки изображения')
            axes[0, 1].set_yticks(range(len(feature_names)))
            axes[0, 1].set_yticklabels(feature_names)

        # Визуализация анализа SSTV
        if analytics_report.get("sstv_analysis"):
            sstv_metrics = analytics_report["sstv_analysis"]
            metric_names = list(sstv_metrics.keys())[:6]  # Берем первые 6 метрик
            metric_values = [sstv_metrics[metric] for metric in metric_names]

            axes[1, 0].plot(range(len(metric_names)), metric_values, marker='o')
            axes[1, 0].set_title('Метрики качества SSTV')
            axes[1, 0].set_xticks(range(len(metric_names)))
            axes[1, 0].set_xticklabels(metric_names, rotation=45, ha='right')

        # Информация о проведенном анализе
        axes[1, 1].axis('off')
        analysis_info = f"Анализы выполнены: {', '.join(analytics_report.get('analyses_performed', []))}\n"
        analysis_info += f"Время анализа: {analytics_report.get('timestamp', 'N/A')}"
        axes[1, 1].text(0.1, 0.5, analysis_info, fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()


    def save_analytics_report(self, report: Dict[str, Any], filename: str = "analytics_report.json"):
        """
        Сохраняет аналитический отчет в файл

        Args:
            report: Аналитический отчет
            filename: Имя файла для сохранения
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"Аналитический отчет сохранен: {filename}")

def main():
    """Главная функция для демонстрации возможностей аналитического модуля"""
    print("=== АНАЛИТИЧЕСКИЙ МОДУЛЬ ПРОЕКТА ===")

    # Создаем аналитический модуль
    analytics = ProjectAnalytics()

    # Создаем тестовые данные
    print("Создание тестовых данных...")

    # Тестовые данные поверхности
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    surface_data = np.sin(np.sqrt(X**2 + Y**2)) * np.exp(-(X**2 + Y**2)/4)

    # Тестовые данные изображения
    image_data = np.random.rand(50, 50, 3)

    # Тестовые данные сигнала
    t = np.linspace(0, 1, 44100)  # 1 секунда сигнала
    signal_data = np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 2000 * t)

    # Генерируем отчет
    report = analytics.generate_comprehensive_report(
        surface_data=surface_data,
        image_data=image_data,
        signal_data=signal_data
    )

    print("✓ Комплексный аналитический отчет сгенерирован")
    print(f"Выполненные анализы: {report['analyses_performed']}")

    # Сохраняем отчет
    analytics.save_analytics_report(report, "test_analytics_report.json")

    print("Аналитический модуль успешно протестирован")

if __name__ == "__main__":
    main()

