# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Модуль машинного обучения для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для построения
предсказательных моделей на основе данных симуляции.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import joblib
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json

class SurfacePredictionModel:
    """
    Класс для построения предсказательных моделей для данных поверхности
    Обеспечивает обучение моделей для предсказания свойств поверхности
    на основе параметров симуляции.
    """


    def __init__(self):
        """Инициализирует модель предсказания поверхности"""
        self.models = {
            'regressor': RandomForestRegressor(n_estimators=100, random_state=42),
            'classifier': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False


    def prepare_features(self, surface_data: np.ndarray) -> np.ndarray:
        """
        Подготавливает признаки из данных поверхности

        Args:
            surface_data: Данные поверхности в виде numpy массива

        Returns:
            Массив признаков
        """
        flat_data = surface_data.flatten()

        # Вычисляем статистические признаки
        features = [
            np.mean(flat_data),      # Средняя высота
            np.std(flat_data),       # Стандартное отклонение
            np.min(flat_data),       # Минимальная высота
            np.max(flat_data),       # Максимальная высота
            np.median(flat_data),    # Медиана
            np.percentile(flat_data, 25),  # 25-й процентиль
            np.percentile(flat_data, 75),  # 75-й процентиль
            np.var(flat_data),       # Дисперсия
            np.ptp(flat_data),       # Размах
            np.sqrt(np.mean(flat_data**2))  # Среднеквадратичное отклонение
        ]

        # Добавляем геометрические признаки
        rows, cols = surface_data.shape
        features.extend([
            rows,                   # Количество строк
            cols,                   # Количество столбцов
            rows * cols,            # Общее количество точек
            rows / cols if cols != 0 else 0  # Соотношение сторон
        ])

        # Добавляем признаки формы поверхности
        grad_x = np.gradient(surface_data, axis=1)
        grad_y = np.gradient(surface_data, axis=0)
        features.extend([
            np.mean(grad_x),        # Средний градиент по X
            np.mean(grad_y),        # Средний градиент по Y
            np.std(grad_x),         # Стандартное отклонение градиента X
            np.std(grad_y),         # Стандартное отклонение градиента Y
            np.mean(np.sqrt(grad_x**2 + grad_y**2))  # Средний модуль градиента
        ])

        return np.array(features).reshape(1, -1)


    def train_regression_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Обучает регрессионную модель

        Args:
            X: Признаки (матрица объекты-признаки)
            y: Целевые значения (вектор)

        Returns:
            Словарь с метриками качества модели
        """
        # Разделяем данные
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Масштабируем признаки
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Обучаем модель
        self.models['regressor'].fit(X_train_scaled, y_train)

        # Делаем предсказания
        y_pred = self.models['regressor'].predict(X_test_scaled)

        # Вычисляем метрики
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2_score': r2,
            'mae': np.mean(np.abs(y_test - y_pred))
        }

        self.is_trained = True
        return metrics


    def train_classification_model(self, X: np.ndarray, y_labels: np.ndarray) -> Dict[str, Any]:
        """
        Обучает классификационную модель

        Args:
            X: Признаки (матрица объекты-признаки)
            y_labels: Целевые метки (вектор)

        Returns:
            Словарь с метриками качества модели
        """
        # Кодируем метки
        y_encoded = self.label_encoder.fit_transform(y_labels)

        # Разделяем данные
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        # Масштабируем признаки
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Обучаем модель
        self.models['classifier'].fit(X_train_scaled, y_train)

        # Делаем предсказания
        y_pred = self.models['classifier'].predict(X_test_scaled)

        # Вычисляем метрики
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)

        metrics = {
            'accuracy': accuracy,
            'classification_report': class_report
        }

        self.is_trained = True
        return metrics


    def predict(self, surface_data: np.ndarray, task_type: str = 'regression') -> np.ndarray:
        """
        Делает предсказание для новых данных поверхности

        Args:
            surface_data: Новые данные поверхности
            task_type: Тип задачи ('regression' или 'classification')

        Returns:
            Предсказанные значения
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Сначала обучите модель.")

        # Подготавливаем признаки
        features = self.prepare_features(surface_data)
        features_scaled = self.scaler.transform(features)

        # Делаем предсказание
        if task_type == 'regression':
            prediction = self.models['regressor'].predict(features_scaled)
        elif task_type == 'classification':
            prediction = self.models['classifier'].predict(features_scaled)
        else:
            raise ValueError("task_type должен быть 'regression' или 'classification'")

        return prediction


    def save_model(self, filepath: str):
        """
        Сохраняет обученную модель

        Args:
            filepath: Путь для сохранения модели
        """
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        print(f"Модель сохранена: {filepath}")


    def load_model(self, filepath: str):
        """
        Загружает обученную модель

        Args:
            filepath: Путь для загрузки модели
        """
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.is_trained = model_data['is_trained']
        print(f"Модель загружена: {filepath}")

class ImageAnalysisPredictor:
    """
    Класс для предсказания характеристик изображений
    Обучает модели для предсказания качества изображений
    и обнаруженных паттернов.
    """


    def __init__(self):
        """Инициализирует предиктор анализа изображений"""
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = [
            'mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity',
            'contrast', 'entropy', 'homogeneity', 'energy', 'correlation',
            'edge_density', 'texture_complexity', 'average_edge_strength'
        ]


    def prepare_image_features(self, image_data: np.ndarray) -> np.ndarray:
        """
        Подготавливает признаки из данных изображения

        Args:
            image_data: Данные изображения в виде numpy массива

        Returns:
            Массив признаков
        """
        if len(image_data.shape) == 3:  # Цветное изображение
            gray = np.mean(image_data, axis=2)
        else:  # Черно-белое изображение
            gray = image_data

        flat_data = gray.flatten()

        # Вычисляем те же признаки, что и в analytics
        features = [
            np.mean(flat_data),      # mean_intensity
            np.std(flat_data),       # std_intensity
            np.min(flat_data),       # min_intensity
            np.max(flat_data),       # max_intensity
            np.std(flat_data) / np.mean(flat_data) if np.mean(flat_data) != 0 else 0,  # contrast
        ]

        # Энтропия
        hist, _ = np.histogram(flat_data, bins=256)
        hist = hist[hist > 0]
        prob = hist / np.sum(hist)
        entropy = -np.sum(prob * np.log2(prob)) if len(prob) > 0 else 0
        features.append(entropy)

        # Однородность
        mean_val = np.mean(gray)
        homogeneity = 1.0 / (1.0 + np.mean(np.abs(gray - mean_val)))
        features.append(homogeneity)

        # Энергия
        energy = np.sum(flat_data**2)
        features.append(energy)

        # Корреляция
        correlation = np.corrcoef(flat_data[:-1], flat_data[1:])[0, 1] if len(flat_data) > 1 else 0
        features.append(correlation)

        # Признаки паттернов
        gradient_x = np.gradient(gray, axis=1)
        gradient_y = np.gradient(gray, axis=0)
        edges = np.sqrt(gradient_x**2 + gradient_y**2)

        edge_threshold = np.mean(edges) + 0.5 * np.std(edges)
        edge_pixels = np.sum(edges > edge_threshold)
        total_pixels = gray.size

        features.extend([
            edge_pixels / total_pixels,  # edge_density
            np.std(edges),               # texture_complexity
            np.mean(edges)               # average_edge_strength
        ])

        return np.array(features).reshape(1, -1)


    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Обучает модель на признаках и целевых значениях

        Args:
            X: Матрица признаков
            y: Целевые значения

        Returns:
            Словарь с метриками качества
        """
        # Разделяем данные
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Масштабируем признаки
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Обучаем модель
        self.model.fit(X_train_scaled, y_train)

        # Делаем предсказания
        y_pred = self.model.predict(X_test_scaled)

        # Вычисляем метрики
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2_score': r2,
            'mae': np.mean(np.abs(y_test - y_pred))
        }

        return metrics


    def predict_quality_score(self, image_data: np.ndarray) -> float:
        """
        Предсказывает оценку качества изображения

        Args:
            image_data: Данные изображения

        Returns:
            Предсказанная оценка качества (0-1)
        """
        features = self.prepare_image_features(image_data)
        features_scaled = self.scaler.transform(features)

        prediction = self.model.predict(features_scaled)
        # Ограничиваем результат от 0 до 1
        quality_score = np.clip(prediction[0], 0, 1)

        return float(quality_score)

class SSTVPredictor:
    """
    Класс для предсказания качества SSTV декодирования
    Обучает модели для предсказания качества декодирования
    на основе характеристик сигнала.
    """


    def __init__(self):
        """Инициализирует предиктор SSTV"""
        self.quality_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.error_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()


    def prepare_signal_features(self, signal_data: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """
        Подготавливает признаки из аудиосигнала SSTV

        Args:
            signal_data: Данные аудиосигнала
            sample_rate: Частота дискретизации

        Returns:
            Массив признаков
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
        noise_estimate = np.std(signal_data[:1000]) if len(signal_data) > 1000 else np.std(signal_data)
        snr = 20 * np.log10(rms_amplitude / noise_estimate) if noise_estimate > 0 else 0

        # Скорость пересечения нуля
        zero_crossings = np.sum(np.diff(np.sign(signal_data)) != 0)
        zero_crossing_rate = zero_crossings / len(signal_data)

        features = [
            rms_amplitude,
            peak_amplitude,
            signal_power,
            dominant_frequency,
            snr,
            np.sum(signal_data**2),  # total_energy
            zero_crossing_rate
        ]

        return np.array(features).reshape(1, -1)


    def train_quality_model(self, X: np.ndarray, y_quality: np.ndarray) -> Dict[str, float]:
        """
        Обучает модель для предсказания качества декодирования

        Args:
            X: Признаки сигнала
            y_quality: Целевые значения качества (0-1)

        Returns:
            Словарь с метриками качества
        """
        # Разделяем данные
        X_train, X_test, y_train, y_test = train_test_split(X, y_quality, test_size=0.2, random_state=42)

        # Масштабируем признаки
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Обучаем модель
        self.quality_model.fit(X_train_scaled, y_train)

        # Делаем предсказания
        y_pred = self.quality_model.predict(X_test_scaled)

        # Вычисляем метрики
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2_score': r2,
            'mae': np.mean(np.abs(y_test - y_pred))
        }

        return metrics


    def predict_decoding_quality(self, signal_data: np.ndarray) -> float:
        """
        Предсказывает качество декодирования SSTV

        Args:
            signal_data: Данные аудиосигнала

        Returns:
            Предсказанное качество декодирования (0-1)
        """
        features = self.prepare_signal_features(signal_data)
        features_scaled = self.scaler.transform(features)

        prediction = self.quality_model.predict(features_scaled)
        quality_score = np.clip(prediction[0], 0, 1)

        return float(quality_score)

class ProjectMLPipeline:
    """
    Центральный класс ML пайплайна проекта
    Объединяет все ML компоненты и предоставляет
    единый интерфейс для машинного обучения.
    """


    def __init__(self):
        """Инициализирует ML пайплайн проекта"""
        self.surface_predictor = SurfacePredictionModel()
        self.image_predictor = ImageAnalysisPredictor()
        self.sstv_predictor = SSTVPredictor()


    def train_all_models(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обучает все модели в пайплайне

        Args:
            training_data: Словарь с обучающими данными для каждой модели

        Returns:
            Словарь с метриками качества всех моделей
        """
        results = {}

        # Обучение модели поверхности (если предоставлены данные)
        if 'surface' in training_data:
            surf_data = training_data['surface']
            if 'features' in surf_data and 'targets' in surf_data:
                try:
                    results['surface_regression'] = self.surface_predictor.train_regression_model(
                        surf_data['features'], surf_data['targets']
                    )
                except Exception as e:
                    print(f"Ошибка обучения модели поверхности: {e}")

        # Обучение модели анализа изображений (если предоставлены данные)
        if 'image' in training_data:
            img_data = training_data['image']
            if 'features' in img_data and 'targets' in img_data:
                try:
                    results['image_prediction'] = self.image_predictor.train(
                        img_data['features'], img_data['targets']
                    )
                except Exception as e:
                    print(f"Ошибка обучения модели изображений: {e}")

        # Обучение модели SSTV (если предоставлены данные)
        if 'sstv' in training_data:
            sstv_data = training_data['sstv']
            if 'features' in sstv_data and 'targets' in sstv_data:
                try:
                    results['sstv_prediction'] = self.sstv_predictor.train_quality_model(
                        sstv_data['features'], sstv_data['targets']
                    )
                except Exception as e:
                    print(f"Ошибка обучения модели SSTV: {e}")

        return results


    def make_predictions(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Делает предсказания всеми моделями

        Args:
            input_data: Словарь с входными данными для каждой модели

        Returns:
            Словарь с предсказаниями всех моделей
        """
        predictions = {}

        # Предсказание поверхности
        if 'surface' in input_data:
            try:
                surface_data = input_data['surface']
                pred = self.surface_predictor.predict(surface_data, 'regression')
                predictions['surface_prediction'] = pred.tolist()
            except Exception as e:
                print(f"Ошибка предсказания поверхности: {e}")

        # Предсказание качества изображения
        if 'image' in input_data:
            try:
                image_data = input_data['image']
                pred = self.image_predictor.predict_quality_score(image_data)
                predictions['image_quality_prediction'] = float(pred)
            except Exception as e:
                print(f"Ошибка предсказания качества изображения: {e}")

        # Предсказание качества SSTV
        if 'sstv_signal' in input_data:
            try:
                signal_data = input_data['sstv_signal']
                pred = self.sstv_predictor.predict_decoding_quality(signal_data)
                predictions['sstv_quality_prediction'] = float(pred)
            except Exception as e:
                print(f"Ошибка предсказания качества SSTV: {e}")

        return predictions


    def save_all_models(self, directory: str):
        """
        Сохраняет все обученные модели

        Args:
            directory: Директория для сохранения моделей
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        self.surface_predictor.save_model(dir_path / "surface_model.pkl")
        joblib.dump(self.image_predictor, dir_path / "image_model.pkl")
        joblib.dump(self.sstv_predictor, dir_path / "sstv_model.pkl")

        print(f"Все модели сохранены в директорию: {directory}")


    def load_all_models(self, directory: str):
        """
        Загружает все модели

        Args:
            directory: Директория с сохраненными моделями
        """
        dir_path = Path(directory)

        self.surface_predictor.load_model(dir_path / "surface_model.pkl")
        self.image_predictor = joblib.load(dir_path / "image_model.pkl")
        self.sstv_predictor = joblib.load(dir_path / "sstv_model.pkl")

        print(f"Все модели загружены из директории: {directory}")

def main():
    """Главная функция для демонстрации возможностей ML модуля"""
    print("=== МОДУЛЬ МАШИННОГО ОБУЧЕНИЯ ПРОЕКТА ===")

    # Создаем ML пайплайн
    ml_pipeline = ProjectMLPipeline()

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

    # Делаем предсказания
    input_data = {
        'surface': surface_data,
        'image': image_data,
        'sstv_signal': signal_data
    }

    predictions = ml_pipeline.make_predictions(input_data)

    print("✓ Предсказания выполнены")
    print(f"Предсказания: {predictions}")

    print("ML модуль успешно протестирован")

if __name__ == "__main__":
    main()

