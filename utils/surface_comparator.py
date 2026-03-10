# -*- coding: utf-8 -*-
"""
Модуль сравнения изображений поверхностей для проекта Nanoprobe Simulation Lab
Сравнение и анализ различий между AFM/СЗМ изображениями
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
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

try:
    from scipy import ndimage, stats
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import mean_squared_error
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


class SurfaceImageComparator:
    """
    Класс для сравнения изображений поверхностей
    Вычисляет метрики схожести и создаёт карты различий
    """

    def __init__(self, output_dir: str = "output/comparisons"):
        """
        Инициализация компаратора

        Args:
            output_dir: Директория для результатов сравнения
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Загрузка изображения

        Args:
            image_path: Путь к изображению

        Returns:
            Массив numpy или None
        """
        if not PIL_AVAILABLE:
            raise ImportError("Pillow не установлен. pip install Pillow")

        try:
            img = Image.open(image_path)
            return np.array(img)
        except Exception as e:
            print(f"Ошибка загрузки изображения {image_path}: {e}")
            return None

    def preprocess_image(
        self,
        image: np.ndarray,
        resize: Tuple[int, int] = None,
        normalize: bool = True,
        grayscale: bool = True
    ) -> np.ndarray:
        """
        Предобработка изображения

        Args:
            image: Исходное изображение
            resize: Новый размер (width, height)
            normalize: Нормализовать к [0, 1]
            grayscale: Конвертировать в оттенки серого

        Returns:
            Предобработанное изображение
        """
        result = image.copy()

        # Конвертация в оттенки серого
        if grayscale and len(result.shape) == 3:
            result = np.mean(result, axis=2)

        # Изменение размера
        if resize:
            if CV2_AVAILABLE:
                result = cv2.resize(result, resize, interpolation=cv2.INTER_AREA)
            else:
                from skimage.transform import resize
                result = resize(result, (resize[1], resize[0]), anti_aliasing=True)

        # Нормализация
        if normalize:
            min_val = result.min()
            max_val = result.max()
            if max_val - min_val > 0:
                result = (result - min_val) / (max_val - min_val)

        return result

    def compare_images(
        self,
        image1_path: str,
        image2_path: str,
        resize_to: Tuple[int, int] = (512, 512)
    ) -> Dict[str, Any]:
        """
        Сравнение двух изображений

        Args:
            image1_path: Путь к первому изображению
            image2_path: Путь ко второму изображению
            resize_to: Размер для приведения к общему виду

        Returns:
            Словарь с метриками сравнения
        """
        if not SKIMAGE_AVAILABLE:
            raise ImportError("scikit-image не установлен. pip install scikit-image")

        # Загрузка и предобработка
        img1 = self.load_image(image1_path)
        img2 = self.load_image(image2_path)

        if img1 is None or img2 is None:
            return {'error': 'Не удалось загрузить изображения'}

        # Предобработка
        img1_proc = self.preprocess_image(img1, resize_to)
        img2_proc = self.preprocess_image(img2, resize_to)

        # Вычисление метрик
        metrics = {}

        # 1. SSIM (Structural Similarity Index)
        ssim_score, ssim_map = ssim(img1_proc, img2_proc, full=True)
        metrics['ssim'] = float(ssim_score)
        metrics['ssim_map'] = ssim_map

        # 2. MSE (Mean Squared Error)
        mse = mean_squared_error(img1_proc.flatten(), img2_proc.flatten())
        metrics['mse'] = float(mse)

        # 3. PSNR (Peak Signal-to-Noise Ratio)
        if mse > 0:
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        else:
            psnr = float('inf')
        metrics['psnr'] = float(psnr)

        # 4. Корреляция Пирсона
        correlation = np.corrcoef(img1_proc.flatten(), img2_proc.flatten())[0, 1]
        metrics['pearson_correlation'] = float(correlation)

        # 5. MAE (Mean Absolute Error)
        mae = np.mean(np.abs(img1_proc - img2_proc))
        metrics['mae'] = float(mae)

        # 6. NCC (Normalized Cross-Correlation)
        ncc = np.sum((img1_proc - np.mean(img1_proc)) * (img2_proc - np.mean(img2_proc))) / \
              (np.std(img1_proc) * np.std(img2_proc) * img1_proc.size)
        metrics['ncc'] = float(ncc)

        # 7. Гистограммное сравнение
        hist1 = np.histogram(img1_proc.flatten(), bins=256, range=(0, 1))[0]
        hist2 = np.histogram(img2_proc.flatten(), bins=256, range=(0, 1))[0]
        hist_corr = np.corrcoef(hist1, hist2)[0, 1]
        metrics['histogram_correlation'] = float(hist_corr)

        # Карта различий
        diff_map = np.abs(img1_proc - img2_proc)
        metrics['diff_map'] = diff_map
        metrics['diff_mean'] = float(np.mean(diff_map))
        metrics['diff_std'] = float(np.std(diff_map))
        metrics['diff_max'] = float(np.max(diff_map))

        # Статистика различий
        metrics['pixels_different'] = int(np.sum(diff_map > 0.1))  # Порог 10%
        metrics['total_pixels'] = int(diff_map.size)
        metrics['percent_different'] = float(100 * metrics['pixels_different'] / metrics['total_pixels'])

        return metrics

    def create_comparison_report(
        self,
        image1_path: str,
        image2_path: str,
        save_diff_image: bool = True
    ) -> Dict[str, Any]:
        """
        Создание полного отчёта о сравнении

        Args:
            image1_path: Путь к первому изображению
            image2_path: Путь ко второму изображению
            save_diff_image: Сохранить карту различий

        Returns:
            Полный отчёт о сравнении
        """
        comparison_id = f"comp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Сравнение
        metrics = self.compare_images(image1_path, image2_path)

        if 'error' in metrics:
            return {'error': metrics['error']}

        # Сохранение карты различий
        diff_image_path = None
        if save_diff_image and PIL_AVAILABLE:
            diff_map = metrics.get('diff_map', np.zeros((100, 100)))
            # Нормализация для сохранения
            diff_normalized = ((diff_map - diff_map.min()) / (diff_map.max() - diff_map.min()) * 255).astype(np.uint8)
            diff_img = Image.fromarray(diff_normalized, mode='L')

            # Применение цветовой карты
            if CV2_AVAILABLE:
                diff_colored = cv2.applyColorMap(diff_normalized, cv2.COLORMAP_JET)
                diff_image_path = str(self.output_dir / f"{comparison_id}_diff.png")
                cv2.imwrite(diff_image_path, diff_colored)
            else:
                diff_image_path = str(self.output_dir / f"{comparison_id}_diff.png")
                diff_img.save(diff_image_path)

        # Формирование отчёта
        report = {
            'comparison_id': comparison_id,
            'timestamp': datetime.now().isoformat(),
            'image1_path': image1_path,
            'image2_path': image2_path,
            'image1_name': Path(image1_path).name,
            'image2_name': Path(image2_path).name,
            'metrics': {
                'ssim': metrics['ssim'],
                'mse': metrics['mse'],
                'psnr': metrics['psnr'],
                'pearson_correlation': metrics['pearson_correlation'],
                'mae': metrics['mae'],
                'ncc': metrics['ncc'],
                'histogram_correlation': metrics['histogram_correlation'],
                'diff_mean': metrics['diff_mean'],
                'diff_std': metrics['diff_std'],
                'diff_max': metrics['diff_max'],
                'percent_different': metrics['percent_different'],
            },
            'diff_image_path': diff_image_path,
            'summary': self._generate_summary(metrics)
        }

        # Сохранение JSON отчёта
        report_path = self.output_dir / f"{comparison_id}_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        report['json_report_path'] = str(report_path)

        return report

    def _generate_summary(self, metrics: Dict[str, Any]) -> str:
        """
        Генерация текстового резюме сравнения

        Args:
            metrics: Метрики сравнения

        Returns:
            Текстовое резюме
        """
        ssim = metrics.get('ssim', 0)
        percent_diff = metrics.get('percent_different', 100)

        summary_parts = []

        # Оценка по SSIM
        if ssim > 0.95:
            summary_parts.append("Изображения практически идентичны")
        elif ssim > 0.85:
            summary_parts.append("Изображения очень похожи")
        elif ssim > 0.7:
            summary_parts.append("Изображения имеют умеренные различия")
        elif ssim > 0.5:
            summary_parts.append("Изображения существенно различаются")
        else:
            summary_parts.append("Изображения сильно различаются")

        # Оценка по проценту различий
        if percent_diff < 5:
            summary_parts.append(f"менее {percent_diff:.1f}% пикселей различаются")
        elif percent_diff < 20:
            summary_parts.append(f"{percent_diff:.1f}% пикселей имеют небольшие различия")
        else:
            summary_parts.append(f"{percent_diff:.1f}% пикселей существенно различаются")

        return ". ".join(summary_parts) + "."

    def compare_multiple_surfaces(
        self,
        image_paths: List[str],
        reference_index: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Сравнение нескольких поверхностей с эталоном

        Args:
            image_paths: Список путей к изображениям
            reference_index: Индекс эталонного изображения

        Returns:
            Список отчётов о сравнении
        """
        if len(image_paths) < 2:
            raise ValueError("Требуется минимум 2 изображения для сравнения")

        reference = image_paths[reference_index]
        results = []

        for i, path in enumerate(image_paths):
            if i == reference_index:
                continue

            report = self.create_comparison_report(reference, path)
            results.append(report)

        return results

    def batch_compare(
        self,
        folder1: str,
        folder2: str,
        pattern: str = "*.png"
    ) -> Dict[str, Any]:
        """
        Пакетное сравнение изображений из двух папок

        Args:
            folder1: Первая папка
            folder2: Вторая папка
            pattern: Шаблон файлов

        Returns:
            Сводный отчёт
        """
        folder1 = Path(folder1)
        folder2 = Path(folder2)

        files1 = sorted(folder1.glob(pattern))
        files2 = sorted(folder2.glob(pattern))

        results = []

        for f1 in files1:
            matching_f2 = folder2 / f1.name
            if matching_f2.exists():
                report = self.create_comparison_report(str(f1), str(matching_f2))
                if 'error' not in report:
                    results.append(report)

        # Сводная статистика
        if results:
            avg_ssim = np.mean([r['metrics']['ssim'] for r in results])
            avg_diff = np.mean([r['metrics']['percent_different'] for r in results])

            summary = {
                'total_comparisons': len(results),
                'average_ssim': float(avg_ssim),
                'average_percent_different': float(avg_diff),
                'comparisons': results
            }
        else:
            summary = {'error': 'Нет совпадающих файлов для сравнения'}

        return summary


# Глобальная функция для быстрого сравнения
def compare_surfaces(
    image1: str,
    image2: str,
    output_dir: str = "output/comparisons"
) -> Dict[str, Any]:
    """
    Быстрое сравнение двух поверхностей

    Args:
        image1: Путь к первому изображению
        image2: Путь ко второму изображению
        output_dir: Директория для результатов

    Returns:
        Отчёт о сравнении
    """
    comparator = SurfaceImageComparator(output_dir)
    return comparator.create_comparison_report(image1, image2)


if __name__ == "__main__":
    # Тестовое сравнение
    print("=== Тестирование компаратора поверхностей ===")

    # Создание тестовых изображений
    if PIL_AVAILABLE:
        # Тестовое изображение 1
        test_img1 = np.random.rand(256, 256) * 255
        test_img1 = Image.fromarray(test_img1.astype(np.uint8))
        test_img1.save("output/comparisons/test1.png")

        # Тестовое изображение 2 (с небольшими изменениями)
        test_img2 = test_img1.copy()
        test_img2_array = np.array(test_img2) + np.random.randn(256, 256) * 10
        test_img2 = Image.fromarray(np.clip(test_img2_array, 0, 255).astype(np.uint8))
        test_img2.save("output/comparisons/test2.png")

        # Сравнение
        result = compare_surfaces(
            "output/comparisons/test1.png",
            "output/comparisons/test2.png"
        )

        print("\nРезультаты сравнения:")
        print(f"  SSIM: {result['metrics']['ssim']:.4f}")
        print(f"  PSNR: {result['metrics']['psnr']:.2f} dB")
        print(f"  Различий: {result['metrics']['percent_different']:.2f}%")
        print(f"  Резюме: {result['summary']}")
        print(f"\n✓ Отчёт сохранён: {result['json_report_path']}")
        if result['diff_image_path']:
            print(f"✓ Карта различий: {result['diff_image_path']}")
