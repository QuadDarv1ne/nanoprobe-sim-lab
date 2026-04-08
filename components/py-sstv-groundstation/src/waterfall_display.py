"""
Waterfall дисплей для визуализации спектра в реальном времени.
Генерация спектрограммы из RTL-SDR данных.
"""

import numpy as np
from typing import Optional, List
from datetime import datetime
from pathlib import Path


class WaterfallDisplay:
    """Waterfall дисплей для RTL-SDR."""

    def __init__(self,
                 width: int = 512,
                 height: int = 256,
                 sample_rate: float = 2.4e6,
                 center_freq: float = 145.8e6):
        """
        Инициализация waterfall дисплея.

        Args:
            width: Ширина спектрограммы (FFT bins)
            height: Высота буфера (количество строк)
            sample_rate: Частота дискретизации
            center_freq: Центральная частота
        """
        self.width = width
        self.height = height
        self.sample_rate = sample_rate
        self.center_freq = center_freq

        # Буфер для waterfall (строки спектра)
        self.waterfall_buffer = np.zeros((height, width), dtype=np.float32)
        self.current_row = 0

        # FFT параметры
        self.fft_size = width
        self.freq_bins = np.fft.fftfreq(width, 1/sample_rate) + center_freq
        self._hann_window = np.hanning(width).astype(np.float32)

        # Цветовая палитра (grayscale)
        self.colormap = self._create_colormap()

        # Статистика
        self.min_power = -100
        self.max_power = -20
        self.frames_count = 0

    def _create_colormap(self) -> np.ndarray:
        """
        Создаёт цветовую палитру.

        Returns:
            np.ndarray: Цветовая карта (256x3 RGB)
        """
        # Градиент от чёрного к синему, зелёному, красному, белому
        colormap = np.zeros((256, 3), dtype=np.uint8)

        for i in range(256):
            if i < 64:
                # Чёрный -> Синий
                colormap[i] = [0, 0, i * 4]
            elif i < 128:
                # Синий -> Зелёный
                colormap[i] = [0, (i - 64) * 4, 255]
            elif i < 192:
                # Зелёный -> Красный
                colormap[i] = [(i - 128) * 4, 255, 255 - (i - 128) * 4]
            else:
                # Красный -> Белый
                colormap[i] = [255, 255, 255 - (255 - i) * 4]

        return colormap

    def push_samples(self, samples: np.ndarray) -> Optional[np.ndarray]:
        """
        Добавляет сэмплы и обновляет waterfall.

        Args:
            samples: numpy массив сэмплов

        Returns:
            np.ndarray: Строка спектра (RGB) или None
        """
        if len(samples) < self.fft_size:
            return None

        # Вычисляем FFT с Hann-окном (убирает спектральные утечки)
        windowed = samples[:self.fft_size] * self._hann_window
        fft_data = np.fft.fft(windowed)
        fft_shifted = np.fft.fftshift(fft_data)

        # Мощность сигнала
        power = np.abs(fft_shifted) ** 2

        # Конвертируем в dB
        power_db = 10 * np.log10(power + 1e-10)

        # Скользящий динамический диапазон (не даём min/max застывать)
        alpha = 0.01  # медленное обновление
        self.min_power = self.min_power * (1 - alpha) + np.percentile(power_db, 5) * alpha
        self.max_power = self.max_power * (1 - alpha) + np.percentile(power_db, 95) * alpha

        # Нормализуем к 0-255
        power_normalized = np.clip(
            (power_db - self.min_power) / (self.max_power - self.min_power + 1e-10) * 255,
            0, 255
        ).astype(np.uint8)

        # Добавляем в буфер
        self.waterfall_buffer[self.current_row] = power_normalized

        # Конвертируем в RGB
        rgb_row = self.colormap[power_normalized]

        # Перемещаем строку
        self.current_row = (self.current_row + 1) % self.height
        self.frames_count += 1

        return rgb_row

    def get_image(self) -> np.ndarray:
        """
        Получает текущее изображение waterfall.

        Returns:
            np.ndarray: RGB изображение (height x width x 3)
        """
        # Перестраиваем буфер чтобы текущая строка была внизу
        if self.current_row == 0:
            buffer = self.waterfall_buffer.copy()
        else:
            buffer = np.vstack([
                self.waterfall_buffer[self.current_row:],
                self.waterfall_buffer[:self.current_row]
            ])

        # Конвертируем в RGB
        image = self.colormap[buffer.astype(np.uint8)]

        return image

    def save_image(self, filepath: str) -> bool:
        """
        Сохраняет waterfall изображение.

        Args:
            filepath: Путь к файлу

        Returns:
            bool: True если успешно
        """
        try:
            image = self.get_image()

            # Сохраняем как PNG через PIL
            from PIL import Image
            pil_image = Image.fromarray(image, 'RGB')

            # Добавляем метаданные
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            pil_image.save(str(filepath), 'PNG')

            print(f"Waterfall сохранён: {filepath}")
            return True
        except Exception as e:
            print(f"Ошибка сохранения waterfall: {e}")
            return False

    def reset(self):
        """Сбрасывает буфер и статистику."""
        self.waterfall_buffer = np.zeros((self.height, self.width), dtype=np.float32)
        self.current_row = 0
        self.min_power = -100
        self.max_power = -20
        self.frames_count = 0
        print("Waterfall сброшен")

    def get_frequency_axis(self) -> np.ndarray:
        """
        Получает ось частот.

        Returns:
            np.ndarray: Частоты в МГц
        """
        return self.freq_bins / 1e6

    def get_spectrum_average(self) -> np.ndarray:
        """
        Получает усреднённый спектр.

        Returns:
            np.ndarray: Усреднённая мощность по частотам
        """
        if self.frames_count == 0:
            return np.zeros(self.width)

        return np.mean(self.waterfall_buffer, axis=0)


class WaterfallRecorder:
    """Запись waterfall в файл."""

    def __init__(self, output_dir: str = "output/waterfall"):
        """
        Инициализация рекордера.

        Args:
            output_dir: Директория для сохранения
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.is_recording = False
        self.frames: List[np.ndarray] = []
        self.max_frames = 300  # 5 минут при 1 fps

    def start(self):
        """Начинает запись."""
        self.frames = []
        self.is_recording = True
        print(f"Запись waterfall начата: {self.output_dir}")

    def add_frame(self, frame: np.ndarray):
        """
        Добавляет кадр.

        Args:
            frame: RGB кадр (width x 3)
        """
        if not self.is_recording:
            return

        self.frames.append(frame)

        if len(self.frames) >= self.max_frames:
            self.stop()

    def stop(self) -> Optional[str]:
        """
        Останавливает запись и сохраняет.

        Returns:
            str: Путь к файлу или None
        """
        self.is_recording = False

        if not self.frames:
            return None

        # Сохраняем как анимированный GIF или видео
        try:
            from PIL import Image

            # Конвертируем кадры
            images = [Image.fromarray(frame, 'RGB') for frame in self.frames]

            # Сохраняем
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.output_dir / f"waterfall_{timestamp}.gif"

            images[0].save(
                str(output_path),
                save_all=True,
                append_images=images[1:],
                duration=100,  # 100ms на кадр = 10 fps
                loop=0
            )

            print(f"Waterfall видео сохранено: {output_path}")
            print(f"  Кадров: {len(self.frames)}")
            return str(output_path)
        except Exception as e:
            print(f"Ошибка сохранения waterfall видео: {e}")
            return None
