"""
Оптимизированный Waterfall дисплей с управлением памятью

Улучшения:
- Ограничение по времени (max_waterfall_duration_minutes)
- Потоковая запись в видео через ffmpeg pipe
- Автоматическая очистка старых кадров
- Мониторинг памяти
"""

import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class OptimizedWaterfallDisplay:
    """
    Оптимизированный waterfall дисплей с управлением памятью.

    Поддерживает:
    - Circular buffer с ограничением по времени
    - Потоковую запись через ffmpeg
    - Автоматическую очистку памяти
    - Мониторинг использования памяти
    """

    def __init__(
        self,
        width: int = 512,
        height: int = 256,
        sample_rate: float = 2.4e6,
        center_freq: float = 145.8e6,
        max_duration_minutes: Optional[int] = None,  # Ограничение по времени
        fps: int = 10,
        enable_streaming: bool = False,  # Потоковая запись
        output_path: Optional[str] = None,
    ):
        """
        Инициализация оптимизированного waterfall.

        Args:
            width: Ширина спектрограммы (FFT bins)
            height: Высота буфера (количество строк)
            sample_rate: Частота дискретизации
            center_freq: Центральная частота
            max_duration_minutes: Максимальная длительность (минуты)
            fps: Частота обновления кадров
            enable_streaming: Включить потоковую запись
            output_path: Путь для сохранения видео
        """
        self.width = width
        self.height = height
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.fps = fps

        # Ограничение по времени
        self.max_duration_minutes = max_duration_minutes
        if max_duration_minutes:
            # Вычисляем максимальное количество кадров
            self.max_frames = int(max_duration_minutes * 60 * fps)
        else:
            self.max_frames = height  # По умолчанию = height строк

        # Circular buffer для кадров
        self.waterfall_buffer = np.zeros((self.max_frames, width), dtype=np.float32)
        self.current_row = 0
        self.frames_count = 0
        self.start_time: Optional[datetime] = None

        # FFT параметры
        self.fft_size = width
        self.freq_bins = np.fft.fftfreq(width, 1 / sample_rate) + center_freq
        self._hann_window = np.hanning(width).astype(np.float32)

        # Цветовая палитра
        self.colormap = self._create_colormap()

        # Статистика
        self.min_power = -100
        self.max_power = -20

        # Потоковая запись
        self.enable_streaming = enable_streaming
        self.ffmpeg_process: Optional[subprocess.Popen] = None
        self.output_path = output_path

        if enable_streaming and output_path:
            self._start_ffmpeg_stream()

    def _create_colormap(self) -> np.ndarray:
        """Создаёт цветовую палитру."""
        colormap = np.zeros((256, 3), dtype=np.uint8)

        for i in range(256):
            if i < 64:
                colormap[i] = [0, 0, i * 4]
            elif i < 128:
                colormap[i] = [0, (i - 64) * 4, 255]
            elif i < 192:
                colormap[i] = [(i - 128) * 4, 255, 255 - (i - 128) * 4]
            else:
                colormap[i] = [255, 255, 255 - (255 - i) * 4]

        return colormap

    def _start_ffmpeg_stream(self):
        """Запускает ffmpeg для потоковой записи."""
        if not self.output_path:
            return

        try:
            # Создаём директорию
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

            # ffmpeg pipe для записи видео
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  # Перезаписываем
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-s",
                f"{self.width}x{self.max_frames}",
                "-pix_fmt",
                "rgb24",
                "-r",
                str(self.fps),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "23",
                self.output_path,
            ]

            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

            logger.info(f"FFmpeg stream запущен: {self.output_path}")

        except FileNotFoundError:
            logger.error("ffmpeg не найден. Установите ffmpeg для потоковой записи.")
            self.enable_streaming = False
        except Exception as e:
            logger.error(f"Ошибка запуска ffmpeg: {e}")
            self.enable_streaming = False

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

        # Вычисляем FFT
        windowed = samples[: self.fft_size] * self._hann_window
        fft_data = np.fft.fft(windowed)
        fft_shifted = np.fft.fftshift(fft_data)

        # Мощность
        power = np.abs(fft_shifted) ** 2
        power_db = 10 * np.log10(power + 1e-10)

        # Скользящий динамический диапазон
        alpha = 0.01
        self.min_power = self.min_power * (1 - alpha) + np.percentile(power_db, 5) * alpha
        self.max_power = self.max_power * (1 - alpha) + np.percentile(power_db, 95) * alpha

        # Нормализуем
        power_normalized = np.clip(
            (power_db - self.min_power) / (self.max_power - self.min_power + 1e-10) * 255, 0, 255
        ).astype(np.uint8)

        # Добавляем в circular buffer
        self.waterfall_buffer[self.current_row] = power_normalized

        # Конвертируем в RGB
        rgb_row = self.colormap[power_normalized]

        # Потоковая запись
        if self.enable_streaming and self.ffmpeg_process:
            try:
                # Записываем один кадр в ffmpeg
                rgb_frame = np.tile(rgb_row, (self.max_frames, 1, 1))
                self.ffmpeg_process.stdin.write(rgb_frame.tobytes())
            except BrokenPipeError:
                logger.error("FFmpeg pipe сломан")
                self.enable_streaming = False
            except Exception as e:
                logger.error(f"Ошибка записи в ffmpeg: {e}")

        # Перемещаем строку
        self.current_row = (self.current_row + 1) % self.max_frames
        self.frames_count += 1

        # Устанавливаем start_time
        if self.start_time is None:
            self.start_time = datetime.now(timezone.utc)

        # Проверяем ограничение по времени
        if self.max_duration_minutes and self.start_time:
            elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds() / 60
            if elapsed > self.max_duration_minutes:
                logger.warning(f"Waterfall достиг лимита {self.max_duration_minutes} минут")
                # Автоматически сохраняем и сбрасываем
                self.save_and_reset()

        return rgb_row

    def get_image(self) -> np.ndarray:
        """Получает текущее изображение waterfall."""
        if self.current_row == 0:
            buffer = self.waterfall_buffer.copy()
        else:
            buffer = np.vstack(
                [
                    self.waterfall_buffer[self.current_row :],
                    self.waterfall_buffer[: self.current_row],
                ]
            )

        return self.colormap[buffer.astype(np.uint8)]

    def get_memory_usage_mb(self) -> float:
        """Получает текущее использование памяти в MB."""
        # Размер буфера в байтах
        buffer_size = self.waterfall_buffer.nbytes
        return buffer_size / (1024 * 1024)

    def save_and_reset(self) -> Optional[str]:
        """Сохраняет текущий waterfall и сбрасывает буфер."""
        if self.frames_count == 0:
            return None

        try:
            # Сохраняем изображение
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_path = f"output/waterfall_{timestamp}.png"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            image = self.get_image()
            from PIL import Image

            pil_image = Image.fromarray(image, "RGB")
            pil_image.save(output_path, "PNG")

            logger.info(f"Waterfall сохранён: {output_path}")

            # Останавливаем ffmpeg
            if self.enable_streaming and self.ffmpeg_process:
                self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.wait(timeout=5)

            # Сбрасываем
            self.reset()

            return output_path

        except Exception as e:
            logger.error(f"Ошибка сохранения waterfall: {e}")
            return None

    def reset(self):
        """Сбрасывает буфер."""
        self.waterfall_buffer = np.zeros((self.max_frames, self.width), dtype=np.float32)
        self.current_row = 0
        self.frames_count = 0
        self.start_time = None
        self.min_power = -100
        self.max_power = -20

        logger.info("Waterfall сброшен")

    def get_status(self) -> dict:
        """Получает статус waterfall."""
        return {
            "width": self.width,
            "height": self.max_frames,
            "fps": self.fps,
            "frames_count": self.frames_count,
            "memory_usage_mb": self.get_memory_usage_mb(),
            "max_duration_minutes": self.max_duration_minutes,
            "streaming_enabled": self.enable_streaming,
            "start_time": self.start_time.isoformat() if self.start_time else None,
        }

    def __del__(self):
        """Деструктор - гарантирует закрытие ffmpeg."""
        if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
            try:
                self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.wait(timeout=5)
            except Exception:
                self.ffmpeg_process.kill()


class OptimizedWaterfallRecorder:
    """
    Оптимизированная запись waterfall с управлением памятью.

    Вместо хранения всех кадров в памяти:
    - Пишет напрямую в ffmpeg pipe
    - Или сохраняет батчами на диск
    - Автоматически ограничивает по времени
    """

    def __init__(
        self,
        output_dir: str = "output/waterfall",
        max_duration_minutes: int = 60,  # 1 час по умолчанию
        fps: int = 10,
        use_ffmpeg: bool = True,
    ):
        """
        Инициализация рекордера.

        Args:
            output_dir: Директория для сохранения
            max_duration_minutes: Максимальная длительность записи
            fps: Частота кадров
            use_ffmpeg: Использовать ffmpeg для видео
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_duration_minutes = max_duration_minutes
        self.fps = fps
        self.use_ffmpeg = use_ffmpeg

        self.is_recording = False
        self.start_time: Optional[datetime] = None
        self.frames_written = 0

        # Ffmpeg процесс
        self.ffmpeg_process: Optional[subprocess.Popen] = None
        self.current_output: Optional[str] = None

    def start(self, output_filename: Optional[str] = None):
        """Начинает запись."""
        self.is_recording = True
        self.start_time = datetime.now(timezone.utc)
        self.frames_written = 0

        if not output_filename:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_filename = (
                f"waterfall_{timestamp}.mp4" if self.use_ffmpeg else f"waterfall_{timestamp}.gif"
            )

        self.current_output = str(self.output_dir / output_filename)

        if self.use_ffmpeg:
            self._start_ffmpeg()

        logger.info(f"Запись waterfall начата: {self.current_output}")

    def _start_ffmpeg(self):
        """Запускает ffmpeg для записи."""
        if not self.current_output:
            return

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"512x256",  # Default size, can be changed
            "-pix_fmt",
            "rgb24",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            self.current_output,
        ]

        try:
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            logger.error("ffmpeg не найден")
            self.use_ffmpeg = False

    def add_frame(self, frame: np.ndarray):
        """
        Добавляет кадр.

        Args:
            frame: RGB кадр (height x width x 3)
        """
        if not self.is_recording:
            return

        # Проверяем ограничение по времени
        if self.start_time:
            elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds() / 60
            if elapsed > self.max_duration_minutes:
                logger.warning(f"Достигнут лимит {self.max_duration_minutes} минут")
                self.stop()
                return

        # Записываем в ffmpeg
        if self.use_ffmpeg and self.ffmpeg_process:
            try:
                self.ffmpeg_process.stdin.write(frame.tobytes())
                self.frames_written += 1
            except BrokenPipeError:
                logger.error("FFmpeg pipe сломан")
                self.stop()
            except Exception as e:
                logger.error(f"Ошибка записи кадра: {e}")
        else:
            # Fallback: сохраняем как PNG батчами
            # Не рекомендуется для больших записей
            logger.warning("FFmpeg недоступен, сохранение отдельных кадров")
            frame_path = self.output_dir / f"frame_{self.frames_written:06d}.png"
            from PIL import Image

            Image.fromarray(frame, "RGB").save(str(frame_path))
            self.frames_written += 1

    def stop(self) -> Optional[str]:
        """Останавливает запись."""
        if not self.is_recording:
            return None

        self.is_recording = False

        # Закрываем ffmpeg
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.wait(timeout=10)
                logger.info(f"FFmpeg завершён, exit code: {self.ffmpeg_process.returncode}")
            except Exception as e:
                logger.error(f"Ошибка закрытия ffmpeg: {e}")
                self.ffmpeg_process.kill()

        logger.info(f"Запись waterfall завершена: {self.frames_written} кадров")
        return self.current_output

    def get_status(self) -> dict:
        """Получает статус записи."""
        elapsed = None
        if self.start_time:
            elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()

        return {
            "is_recording": self.is_recording,
            "elapsed_seconds": elapsed,
            "frames_written": self.frames_written,
            "max_duration_minutes": self.max_duration_minutes,
            "output_file": self.current_output,
        }
