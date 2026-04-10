"""Модуль загрузки и обработки космических снимков."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests
from PIL import Image

try:
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class SpaceImageDownloader:
    """Загрузка изображений из космических телескопов."""

    def __init__(self, download_dir: str = "data/space_images"):
        """Инициализирует загрузчик."""
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        # API endpoints
        self.apis = {
            "hubble": "https://mast.stsci.edu/api/v0.1/Discovery/api/",
            "nasa": "https://images-api.nasa.gov",
            "esa": "https://esahubble.org/api/v1",
            "jwst": "https://mast.stsci.edu/api/v0.1/Discovery/api/",
        }

    def search_hubble(
        self, target: str = None, observation_type: str = None, pagesize: int = 10
    ) -> List[Dict]:
        """
        Поиск данных Hubble.

        Args:
            target: Название объекта (например, "M31", "Crab Nebula")
            observation_type: Тип наблюдения
            pagesize: Количество результатов

        Returns:
            Список метаданных изображений
        """
        params = {"format": "json", "pagesize": pagesize}

        if target:
            params["target_name"] = target

        if observation_type:
            params["observation_type"] = observation_type

        try:
            response = requests.get(self.apis["hubble"], params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Ошибка поиска Hubble: {e}")
            return []

    def search_nasa(
        self, query: str, year_start: int = None, year_end: int = None, pagesize: int = 10
    ) -> List[Dict]:
        """
        Поиск изображений NASA.

        Args:
            query: Поисковый запрос
            year_start: Начальный год
            year_end: Конечный год
            pagesize: Количество результатов

        Returns:
            Список метаданных изображений
        """
        params = {"q": query, "media_type": "image", "page_size": pagesize}

        if year_start:
            params["year_start"] = year_start
        if year_end:
            params["year_end"] = year_end

        try:
            response = requests.get(f"{self.apis['nasa']}/search", params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("collection", {}).get("items", [])
        except requests.RequestException as e:
            print(f"Ошибка поиска NASA: {e}")
            return []

    def download_image(
        self, url: str, filename: str = None, save_format: str = "png"
    ) -> Optional[Path]:
        """
        Загрузка изображения по URL.

        Args:
            url: URL изображения
            filename: Имя файла
            save_format: Формат сохранения

        Returns:
            Путь к сохранённому файлу или None
        """
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            if filename is None:
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                filename = f"space_image_{timestamp}.{save_format}"

            filepath = self.download_dir / filename
            with open(filepath, "wb") as f:
                f.write(response.content)

            print(f"Изображение сохранено: {filepath}")
            return filepath

        except requests.RequestException as e:
            print(f"Ошибка загрузки: {e}")
            return None

    def download_hubble_product(self, product_id: str, save_format: str = "fits") -> Optional[Path]:
        """
        Загрузка продукта Hubble по ID.

        Args:
            product_id: ID продукта Hubble
            save_format: Формат (fits, png, jpg)

        Returns:
            Путь к файлу или None
        """
        # Hubble использует MAST для загрузки
        url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={product_id}"
        filename = f"hubble_{product_id}.{save_format}"
        return self.download_image(url, filename, save_format)


class SpaceImageProcessor:
    """Обработка космических изображений."""

    def __init__(self):
        """Инициализирует процессор."""
        self.image = None
        self.processed_image = None
        self.metadata = {}

    def load_image(self, filepath: str) -> bool:
        """Загружает изображение."""
        try:
            self.image = Image.open(filepath)
            self.metadata["filepath"] = filepath
            self.metadata["size"] = self.image.size
            self.metadata["mode"] = self.image.mode
            return True
        except Exception as e:
            print(f"Ошибка загрузки: {e}")
            return False

    def load_from_url(self, url: str) -> bool:
        """Загружает изображение из URL."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            self.image = Image.open(BytesIO(response.content))
            self.metadata["url"] = url
            return True
        except Exception as e:
            print(f"Ошибка загрузки URL: {e}")
            return False

    def convert_to_grayscale(self) -> Image.Image:
        """Конвертирует в чёрно-белое."""
        if self.image is None:
            raise ValueError("Изображение не загружено")

        self.processed_image = self.image.convert("L")
        return self.processed_image

    def enhance_contrast(self, factor: float = 1.5) -> Image.Image:
        """Улучшает контраст."""
        from PIL import ImageEnhance

        if self.image is None:
            raise ValueError("Изображение не загружено")

        enhancer = ImageEnhance.Contrast(self.image)
        self.processed_image = enhancer.enhance(factor)
        return self.processed_image

    def enhance_brightness(self, factor: float = 1.2) -> Image.Image:
        """Улучшает яркость."""
        from PIL import ImageEnhance

        if self.image is None:
            raise ValueError("Изображение не загружено")

        enhancer = ImageEnhance.Brightness(self.image)
        self.processed_image = enhancer.enhance(factor)
        return self.processed_image

    def apply_color_map(self, cmap: str = "viridis") -> Image.Image:
        """Применяет цветовую карту к чёрно-белому изображению."""
        if self.image is None:
            raise ValueError("Изображение не загружено")

        # Конвертируем в numpy массив
        img_array = np.array(self.image.convert("L"))

        # Нормализуем
        img_normalized = (img_array - img_array.min()) / (img_array.max() - img_array.min())

        # Применяем цветовую карту matplotlib
        import matplotlib.pyplot as plt

        colormap = plt.get_cmap(cmap)
        colored = (colormap(img_normalized) * 255).astype(np.uint8)

        self.processed_image = Image.fromarray(colored[:, :, :3])
        return self.processed_image

    def save_image(self, filepath: str, quality: int = 95, fmt: str = None) -> bool:
        """
        Сохраняет изображение.

        Args:
            filepath: Путь к файлу
            quality: Качество (1-100 для JPEG)
            fmt: Формат (автоопределение если None)

        Returns:
            True если успешно
        """
        try:
            img = self.processed_image if self.processed_image else self.image

            if img is None:
                raise ValueError("Нет изображения для сохранения")

            # Определяем формат
            if fmt is None:
                fmt = Path(filepath).suffix.lstrip(".").upper()
                if fmt == "JPG":
                    fmt = "JPEG"

            # Сохраняем
            save_kwargs = {}
            if fmt == "JPEG":
                save_kwargs["quality"] = quality
            elif fmt == "PNG":
                save_kwargs["optimize"] = True

            img.save(filepath, fmt, **save_kwargs)
            print(f"Изображение сохранено: {filepath}")
            return True

        except Exception as e:
            print(f"Ошибка сохранения: {e}")
            return False

    def create_print_versions(
        self, base_filename: str, output_dir: str = "output/print"
    ) -> Dict[str, Path]:
        """
        Создаёт версии для печати (цветная и Ч/Б).

        Args:
            base_filename: Базовое имя файла
            output_dir: Директория вывода

        Returns:
            Словарь с путями к файлам
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {}

        if self.image is None:
            raise ValueError("Изображение не загружено")

        # Цветная версия (улучшенная)
        self.enhance_contrast(1.3)
        self.enhance_brightness(1.1)
        color_path = output_path / f"{base_filename}_color.png"
        self.save_image(str(color_path))
        results["color"] = color_path

        # Чёрно-белая версия
        self.convert_to_grayscale()
        bw_path = output_path / f"{base_filename}_bw.png"
        self.save_image(str(bw_path))
        results["black_white"] = bw_path

        # Псевдоцветная (для науки)
        self.image = self.image.convert("L")  # Сначала в Ч/Б
        self.apply_color_map("plasma")
        false_color_path = output_path / f"{base_filename}_false_color.png"
        self.save_image(str(false_color_path))
        results["false_color"] = false_color_path

        print(f"Создано версий для печати: {len(results)}")
        return results


class FITSReader:
    """Чтение FITS файлов (астрономический формат)."""

    def __init__(self):
        """Инициализирует читатель FITS."""
        self.data = None
        self.header = {}

    def read(self, filepath: str) -> bool:
        """
        Читает FITS файл.

        Args:
            filepath: Путь к FITS файлу

        Returns:
            True если успешно
        """
        try:
            # Пробуем astropy
            try:
                from astropy.io import fits

                with fits.open(filepath) as hdul:
                    self.data = hdul[0].data
                    self.header = dict(hdul[0].header)
                return True
            except ImportError:
                # Простая реализация без astropy
                print("astropy не установлен. Используем упрощённое чтение.")
                return self._simple_read(filepath)

        except Exception as e:
            print(f"Ошибка чтения FITS: {e}")
            return False

    def _simple_read(self, filepath: str) -> bool:
        """Упрощённое чтение FITS без astropy."""
        try:
            # Читаем как бинарный файл
            with open(filepath, "rb") as f:
                # Пропускаем заголовок (упрощённо)
                f.seek(2880)  # Первый блок FITS
                # Читаем данные
                self.data = np.frombuffer(f.read(), dtype=np.float32)
            return True
        except Exception as e:
            print(f"Ошибка упрощённого чтения: {e}")
            return False

    def to_image(self) -> Optional[Image.Image]:
        """Конвертирует FITS данные в изображение."""
        if self.data is None:
            return None

        # Нормализуем данные
        data_normalized = (self.data - self.data.min()) / (self.data.max() - self.data.min())
        data_scaled = (data_normalized * 255).astype(np.uint8)

        return Image.fromarray(data_scaled)


def download_and_print_hubble_image(
    target: str, output_dir: str = "output/print"
) -> Dict[str, Path]:
    """
    Полная цепочка: поиск → загрузка → обработка → печать.

    Args:
        target: Название объекта (например, "M31", "Crab Nebula")
        output_dir: Директория для результатов

    Returns:
        Словарь с путями к файлам для печати
    """
    print(f"🔭 Поиск изображений Hubble: {target}")

    # Загрузка
    downloader = SpaceImageDownloader()
    results = downloader.search_hubble(target, pagesize=5)

    if not results:
        print("❌ Изображения не найдены")
        return {}

    print(f"✅ Найдено {len(results)} изображений")

    # Берём первое изображение
    first_result = results[0]

    # Получаем URL
    image_url = None
    if "url" in first_result:
        image_url = first_result["url"]
    elif "product_url" in first_result:
        image_url = first_result["product_url"]

    if not image_url:
        print("❌ Не найден URL изображения")
        return {}

    print(f"📥 Загрузка: {image_url}")
    downloaded_path = downloader.download_image(image_url)

    if not downloaded_path:
        print("❌ Ошибка загрузки")
        return {}

    # Обработка
    print("🎨 Обработка изображений...")
    processor = SpaceImageProcessor()
    processor.load_image(str(downloaded_path))

    # Создаём версии для печати
    base_name = downloaded_path.stem
    print_versions = processor.create_print_versions(base_name, output_dir)

    print("✅ Готово к печати!")
    for version, path in print_versions.items():
        print(f"  - {version}: {path}")

    return print_versions


# Пример использования
if __name__ == "__main__":
    # Загрузка и обработка изображения Hubble
    results = download_and_print_hubble_image("M31")

    if results:
        print("\n🖨️  Файлы готовы для печати:")
        for name, path in results.items():
            print(f"  {name}: {path}")
