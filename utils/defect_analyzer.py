# -*- coding: utf-8 -*-
"""
Модуль AI/ML анализа дефектов для проекта Nanoprobe Simulation Lab
Обнаружение и классификация дефектов на поверхностях с использованием ML
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import json

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy import ndimage
from scipy.stats import zscore


class DefectDetector:
    """
    Детектор дефектов на поверхностях
    Использует методы машинного обучения для обнаружения аномалий
    """

    def __init__(self, model_name: str = "isolation_forest"):
        """
        Инициализация детектора

        Args:
            model_name: Название модели ('isolation_forest', 'kmeans', 'dbscan')
        """
        self.model_name = model_name
        self.scaler = StandardScaler()
        self.model = self._create_model(model_name)
        self.is_trained = False

        # Категории дефектов
        self.defect_categories = {
            0: 'pit',           # Впадина
            1: 'hillock',       # Выступ
            2: 'scratch',       # Царапина
            3: 'particle',      # Частица
            4: 'crack',         # Трещина
            5: 'normal',        # Нормальная область
        }

    def _create_model(self, model_name: str):
        """Создание модели детектирования"""
        if model_name == 'isolation_forest':
            return IsolationForest(
                n_estimators=100,
                contamination=0.1,
                random_state=42,
                bootstrap=True
            )
        elif model_name == 'kmeans':
            return KMeans(n_clusters=6, random_state=42, n_init=10)
        elif model_name == 'dbscan':
            return DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError(f"Неизвестная модель: {model_name}")

    def extract_features(self, image: np.ndarray, patch_size: int = 16) -> List[np.ndarray]:
        """
        Извлечение признаков из изображения

        Args:
            image: Изображение поверхности
            patch_size: Размер патча для анализа

        Returns:
            Список векторов признаков
        """
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        features = []
        positions = []

        height, width = gray.shape

        for i in range(0, height - patch_size, patch_size // 2):
            for j in range(0, width - patch_size, patch_size // 2):
                patch = gray[i:i+patch_size, j:j+patch_size]

                # Статистические признаки
                feat_vector = [
                    np.mean(patch),
                    np.std(patch),
                    np.min(patch),
                    np.max(patch),
                    np.median(patch),
                    np.percentile(patch, 25),
                    np.percentile(patch, 75),
                    np.var(patch),
                ]

                # Градиенты (текстура)
                grad_x, grad_y = np.gradient(patch)
                feat_vector.extend([
                    np.mean(np.abs(grad_x)),
                    np.mean(np.abs(grad_y)),
                    np.std(grad_x),
                    np.std(grad_y),
                ])

                # Лапласиан (обнаружение краев)
                laplacian = ndimage.laplace(patch)
                feat_vector.extend([
                    np.mean(np.abs(laplacian)),
                    np.std(laplacian),
                ])

                # Энтропия (сложность текстуры)
                hist, _ = np.histogram(patch.flatten(), bins=32)
                hist = hist[hist > 0]
                prob = hist / np.sum(hist)
                entropy = -np.sum(prob * np.log2(prob + 1e-10))
                feat_vector.append(entropy)

                features.append(feat_vector)
                positions.append((i + patch_size // 2, j + patch_size // 2))

        return np.array(features), positions

    def detect_defects(
        self,
        image: np.ndarray,
        threshold: float = -0.5
    ) -> Dict[str, Any]:
        """
        Обнаружение дефектов на изображении

        Args:
            image: Изображение поверхности
            threshold: Порог детектирования

        Returns:
            Словарь с результатами детектирования
        """
        # Извлечение признаков
        features, positions = self.extract_features(image)

        # Нормализация
        features_scaled = self.scaler.fit_transform(features)

        # Предсказание
        if self.model_name == 'isolation_forest':
            predictions = self.model.fit_predict(features_scaled)
            scores = self.model.decision_function(features_scaled)
        elif self.model_name == 'kmeans':
            predictions = self.model.fit_predict(features_scaled)
            # Дефекты - наиболее удалённые от центроидов кластеры
            distances = self.model.transform(features_scaled)
            scores = -np.min(distances, axis=1)
            predictions = (scores < np.percentile(scores, 10)).astype(int) * -1
        else:
            predictions = self.model.fit_predict(features_scaled)
            scores = np.zeros(len(features))

        # Обработка результатов
        defect_mask = predictions == -1  # Аномалии помечены как -1

        # Группировка дефектов
        defect_regions = self._group_defects(defect_mask, positions, image.shape)

        # Классификация дефектов
        classified_defects = self._classify_defects(defect_regions, image)

        return {
            'defects_count': len(classified_defects),
            'defects': classified_defects,
            'defect_mask': defect_mask,
            'scores': scores,
            'positions': positions,
            'summary': self._generate_defect_summary(classified_defects)
        }

    def _group_defects(
        self,
        mask: np.ndarray,
        positions: List[Tuple[int, int]],
        image_shape: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """Группировка дефектов по регионам"""
        # Создание карты дефектов
        defect_map = np.zeros(image_shape, dtype=bool)

        for i, (y, x) in enumerate(positions):
            if i < len(mask) and mask[i]:
                defect_map[y, x] = True

        # Морфологические операции
        if CV2_AVAILABLE:
            kernel = np.ones((3, 3), np.uint8)
            defect_map_uint8 = defect_map.astype(np.uint8) * 255
            dilated = cv2.dilate(defect_map_uint8, kernel, iterations=2)
            eroded = cv2.erode(dilated, kernel, iterations=1)
            defect_map = eroded > 0

        # Поиск связных областей
        labeled, n_components = ndimage.label(defect_map)

        regions = []
        for i in range(1, n_components + 1):
            region_mask = labeled == i
            coords = np.where(region_mask)

            if len(coords[0]) > 0:
                regions.append({
                    'y_coords': coords[0],
                    'x_coords': coords[1],
                    'size': len(coords[0]),
                    'centroid': (np.mean(coords[0]), np.mean(coords[1]))
                })

        return regions

    def _classify_defects(
        self,
        regions: List[Dict[str, Any]],
        image: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Классификация дефектов по типам"""
        classified = []

        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        for region in regions:
            y_coords = region['y_coords']
            x_coords = region['x_coords']

            # Извлечение области дефекта
            y_min, y_max = int(y_coords.min()), int(y_coords.max())
            x_min, x_max = int(x_coords.min()), int(x_coords.max())

            defect_patch = gray[y_min:y_max, x_min:x_max]

            # Признаки для классификации
            aspect_ratio = (x_max - x_min) / (y_max - y_min + 1e-10)
            area = region['size']
            mean_intensity = np.mean(defect_patch)
            std_intensity = np.std(defect_patch)

            # Определение типа дефекта
            if aspect_ratio > 3 or aspect_ratio < 0.33:
                defect_type = 'scratch'  # Царапина
            elif area < 10:
                defect_type = 'particle'  # Частица
            elif std_intensity < 0.1 * mean_intensity:
                defect_type = 'pit'  # Впадина
            else:
                defect_type = 'hillock'  # Выступ

            # Вычисление доверительного интервала
            confidence = min(0.95, 0.5 + 0.1 * np.log(area + 1))

            classified.append({
                'type': defect_type,
                'x': float(region['centroid'][1]),
                'y': float(region['centroid'][0]),
                'width': x_max - x_min,
                'height': y_max - y_min,
                'area': area,
                'aspect_ratio': aspect_ratio,
                'mean_intensity': float(mean_intensity),
                'confidence': confidence,
            })

        return classified

    def _generate_defect_summary(self, defects: List[Dict[str, Any]]) -> str:
        """Генерация резюме анализа"""
        if not defects:
            return "Дефекты не обнаружены. Поверхность в хорошем состоянии."

        type_counts = {}
        for defect in defects:
            dtype = defect['type']
            type_counts[dtype] = type_counts.get(dtype, 0) + 1

        summary_parts = [f"Обнаружено дефектов: {len(defects)}"]

        for dtype, count in type_counts.items():
            summary_parts.append(f"  - {dtype}: {count}")

        # Оценка качества поверхности
        defect_density = len(defects) / 100  # Дефектов на 100 пикселей²

        if defect_density < 0.1:
            summary_parts.append("Качество поверхности: отличное")
        elif defect_density < 0.5:
            summary_parts.append("Качество поверхности: хорошее")
        elif defect_density < 1.0:
            summary_parts.append("Качество поверхности: удовлетворительное")
        else:
            summary_parts.append("Качество поверхности: требует улучшения")

        return ". ".join(summary_parts)

    def train(self, training_images: List[np.ndarray], labels: List[int] = None):
        """
        Обучение модели на тренировочных данных

        Args:
            training_images: Список изображений для обучения
            labels: Метки (None для unsupervised обучения)
        """
        all_features = []

        for image in training_images:
            features, _ = self.extract_features(image)
            all_features.extend(features)

        all_features = np.array(all_features)
        all_features_scaled = self.scaler.fit_transform(all_features)

        # Обучение без учителя (по умолчанию)
        if labels is None:
            self.model.fit(all_features_scaled)
        else:
            # Обучение с учителем
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(all_features_scaled, labels)

        self.is_trained = True

    def save_model(self, filepath: str):
        """Сохранение модели"""
        import joblib
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_name': self.model_name,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        print(f"Модель сохранена: {filepath}")

    def load_model(self, filepath: str):
        """Загрузка модели"""
        import joblib
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_name = model_data['model_name']
        self.is_trained = model_data['is_trained']
        print(f"Модель загружена: {filepath}")


class DefectAnalysisPipeline:
    """
    Пайплайн полного анализа дефектов
    Интеграция с базой данных и генерация отчётов
    """

    def __init__(self, db_manager=None):
        """
        Инициализация пайплайна

        Args:
            db_manager: Менеджер базы данных (опционально)
        """
        self.db_manager = db_manager
        self.detector = DefectDetector()
        self.output_dir = Path("output/defect_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_image(
        self,
        image_path: str,
        model_name: str = "isolation_forest",
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Полный анализ изображения на дефекты

        Args:
            image_path: Путь к изображению
            model_name: Название модели
            save_results: Сохранить результаты

        Returns:
            Результаты анализа
        """
        from PIL import Image as PILImage

        # Загрузка изображения
        img = PILImage.open(image_path)
        image_array = np.array(img)

        # Детектирование
        self.detector = DefectDetector(model_name)
        results = self.detector.detect_defects(image_array)

        # Добавление метаданных
        analysis_id = f"defect_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        full_report = {
            'analysis_id': analysis_id,
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'image_name': Path(image_path).name,
            'model_name': model_name,
            **results
        }

        # Сохранение в БД
        if self.db_manager:
            self.db_manager.add_defect_analysis(
                analysis_id=analysis_id,
                image_path=image_path,
                model_name=model_name,
                defects_detected=results['defects_count'],
                defects_data={'defects': results['defects']},
                confidence_score=np.mean([d['confidence'] for d in results['defects']]) if results['defects'] else 0,
            )

        # Сохранение JSON отчёта
        if save_results:
            report_path = self.output_dir / f"{analysis_id}_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(full_report, f, indent=2, ensure_ascii=False, default=str)
            full_report['report_path'] = str(report_path)

        # Сохранение визуализации
        if save_results and results['defects']:
            viz_path = self._visualize_defects(image_array, results['defects'], analysis_id)
            full_report['visualization_path'] = viz_path

        return full_report

    def _visualize_defects(
        self,
        image: np.ndarray,
        defects: List[Dict[str, Any]],
        analysis_id: str
    ) -> str:
        """Визуализация дефектов на изображении"""
        if not CV2_AVAILABLE:
            return ""

        # Конвертация в BGR для OpenCV
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            vis = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Отрисовка дефектов
        colors = {
            'pit': (255, 0, 0),      # Синий
            'hillock': (0, 255, 0),   # Зелёный
            'scratch': (0, 0, 255),   # Красный
            'particle': (255, 255, 0), # Циан
            'crack': (0, 255, 255),   # Жёлтый
        }

        for defect in defects:
            color = colors.get(defect['type'], (255, 255, 255))
            x = int(defect['x'])
            y = int(defect['y'])
            w = int(defect['width'])
            h = int(defect['height'])

            # Рисуем прямоугольник
            cv2.rectangle(vis, (x - w//2, y - h//2), (x + w//2, y + h//2), color, 2)

            # Подпись
            label = f"{defect['type']}: {defect['confidence']:.2f}"
            cv2.putText(vis, label, (x - w//2, y - h//2 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Сохранение
        output_path = self.output_dir / f"{analysis_id}_visualization.png"
        cv2.imwrite(str(output_path), vis)

        return str(output_path)

    def batch_analyze(
        self,
        image_folder: str,
        pattern: str = "*.png",
        model_name: str = "isolation_forest"
    ) -> List[Dict[str, Any]]:
        """
        Пакетный анализ изображений

        Args:
            image_folder: Папка с изображениями
            pattern: Шаблон файлов
            model_name: Название модели

        Returns:
            Список результатов анализа
        """
        folder = Path(image_folder)
        images = list(folder.glob(pattern))

        results = []
        for img_path in images:
            try:
                result = self.analyze_image(str(img_path), model_name)
                results.append(result)
            except Exception as e:
                print(f"Ошибка анализа {img_path}: {e}")
                results.append({'error': str(e), 'image_path': str(img_path)})

        return results


# Глобальная функция для быстрого анализа
def analyze_defects(
    image_path: str,
    model_name: str = "isolation_forest",
    output_dir: str = "output/defect_analysis"
) -> Dict[str, Any]:
    """
    Быстрый анализ дефектов

    Args:
        image_path: Путь к изображению
        model_name: Название модели
        output_dir: Директория для результатов

    Returns:
        Результаты анализа
    """
    pipeline = DefectAnalysisPipeline()
    pipeline.output_dir = Path(output_dir)
    return pipeline.analyze_image(image_path, model_name)


if __name__ == "__main__":
    # Тестирование
    print("=== Тестирование AI анализа дефектов ===")

    # Создание тестового изображения с "дефектами"
    if PIL_AVAILABLE:
        # Базовое изображение (нормальная поверхность)
        test_image = np.random.normal(128, 10, (256, 256)).astype(np.uint8)

        # Добавление "дефектов"
        test_image[50:60, 100:150] = 200  # Выступ
        test_image[150:155, 50:200] = 50  # Царапина
        test_image[200:220, 200:220] = 30  # Впадина

        test_img = Image.fromarray(test_image)
        test_path = "output/defect_analysis/test_surface.png"
        test_img.save(test_path)

        # Анализ
        result = analyze_defects(test_path)

        print(f"\nРезультаты анализа:")
        print(f"  Найдено дефектов: {result['defects_count']}")
        print(f"  Резюме: {result['summary']}")

        for i, defect in enumerate(result['defects'][:5], 1):
            print(f"\n  Дефект #{i}:")
            print(f"    Тип: {defect['type']}")
            print(f"    Координаты: ({defect['x']:.1f}, {defect['y']:.1f})")
            print(f"    Размер: {defect['width']}x{defect['height']}")
            print(f"    Достоверность: {defect['confidence']:.2%}")

        print(f"\n✓ Отчёт: {result.get('report_path', 'Не сохранён')}")
        print(f"✓ Визуализация: {result.get('visualization_path', 'Не сохранена')}")
