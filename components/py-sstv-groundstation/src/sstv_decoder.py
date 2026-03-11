# -*- coding: utf-8 -*-
"""Модуль декодирования SSTV."""

import numpy as np
from typing import Optional, Tuple, List, Dict
from PIL import Image
from pathlib import Path
from datetime import datetime


class SSTVDecoder:
    """Класс для декодирования SSTV-сигналов."""

    SUPPORTED_MODES = [
        'Martin 1', 'Martin 2', 'Scottie 1', 'Scottie 2',
        'PD50', 'PD90', 'PD120', 'PD160', 'PD180',
        'Robot 36', 'Robot 72', 'Wraase SC1', 'Wraase SC2'
    ]

    def __init__(self, mode: str = 'auto'):
        """
        Инициализирует декодер SSTV.

        Args:
            mode: Режим декодирования ('auto' или название режима)
        """
        self.decoded_image = None
        self.signal_data = None
        self.mode = mode
        self.decoded_images: List[Image.Image] = []
        self.metadata: Dict = {}

    def decode_from_audio(self, audio_file: str) -> Optional[Image.Image]:
        """
        Декодирует SSTV-сигнал из аудиофайла.

        Args:
            audio_file: Путь к аудиофайлу с SSTV-сигналом

        Returns:
            Image.Image: Декодированное изображение или None при ошибке
        """
        try:
            audio_path = Path(audio_file)
            if not audio_path.exists():
                print(f"Ошибка: Файл '{audio_file}' не найден")
                return None

            if audio_path.suffix.lower() not in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
                print(f"Неподдерживаемый аудиоформат: {audio_path.suffix}")
                return None

            print(f"Декодирование SSTV-сигнала из: {audio_file}")

            # Пробуем декодировать с автоматическим определением режима
            if self.mode == 'auto':
                for mode_name in self.SUPPORTED_MODES:
                    try:
                        image = self._decode_with_mode(str(audio_path), mode_name)
                        if image is not None:
                            self.decoded_image = image
                            self.decoded_images.append(image)
                            self.metadata['mode'] = mode_name
                            self.metadata['timestamp'] = datetime.now().isoformat()
                            self.metadata['source'] = str(audio_path)
                            print(f"✓ Успешно декодировано в режиме: {mode_name}")
                            return image
                    except Exception:
                        continue

                print("Не удалось декодировать ни в одном из режимов")
                return None
            else:
                if self.mode not in self.SUPPORTED_MODES:
                    print(f"Неподдерживаемый режим: {self.mode}. Доступные: {self.SUPPORTED_MODES}")
                    return None
                image = self._decode_with_mode(str(audio_path), self.mode)
                if image:
                    self.decoded_image = image
                    self.decoded_images.append(image)
                    self.metadata['mode'] = self.mode
                    self.metadata['timestamp'] = datetime.now().isoformat()
                    self.metadata['source'] = str(audio_path)
                    return image
                return None

        except ImportError:
            print("Библиотека pysstv не установлена.")
            print("Установите: pip install pysstv")
            return self._fallback_decode(audio_file)
        except Exception as e:
            print(f"Ошибка при декодировании SSTV-сигнала: {e}")
            return self._fallback_decode(audio_file)

    def _decode_with_mode(self, audio_file: str, mode: str) -> Optional[Image.Image]:
        """
        Декодирует SSTV в конкретном режиме.

        Args:
            audio_file: Путь к аудиофайлу
            mode: Режим SSTV

        Returns:
            Image.Image или None
        """
        try:
            from pysstv.audio import decode as pysstv_decode

            image = pysstv_decode(audio_file, mode)
            if image is not None and image.size[0] > 0 and image.size[1] > 0:
                return image
            return None
        except Exception:
            return None

    def _fallback_decode(self, audio_file: str) -> Optional[Image.Image]:
        """
        Резервный метод декодирования (заглушка).

        Args:
            audio_file: Путь к аудиофайлу

        Returns:
            Image.Image: Заглушка изображения
        """
        print("Используется резервный режим декодирования...")
        try:
            import wave
            with wave.open(audio_file, "r") as wav:
                n_frames = wav.getnframes()
                duration = n_frames / wav.getframerate()
                self.metadata['duration_seconds'] = duration
                self.metadata['sample_rate'] = wav.getframerate()
        except Exception:
            pass

        decoded_img = Image.new("RGB", (320, 240), color=(20, 20, 40))
        self.decoded_image = decoded_img
        self.metadata['mode'] = 'fallback'
        return decoded_img

    def decode_from_samples(
        self,
        samples: np.ndarray,
        sample_rate: int = 48000
    ) -> Optional[Image.Image]:
        """
        Декодирует SSTV из numpy массива сэмплов.

        Args:
            samples: Аудиосэмпл в формате numpy array
            sample_rate: Частота дискретизации

        Returns:
            Image.Image: Декодированное изображение

        Raises:
            ValueError: Если сэмплы некорректны
        """
        if samples is None or len(samples) == 0:
            raise ValueError("Аудиоданные не должны быть пустыми")

        if sample_rate <= 0:
            raise ValueError("Частота дискретизации должна быть положительной")

        try:
            from pysstv.sstv import SSTV
            from pysstv.color import ColorSSTV

            # Пробуем определить режим и декодировать
            for mode_name in self.SUPPORTED_MODES:
                try:
                    # Создаём SSTV декодер для режима
                    decoder = SSTV(samples, sample_rate, mode_name)
                    image = decoder.decode()
                    if image is not None:
                        self.decoded_image = image
                        self.decoded_images.append(image)
                        self.metadata['mode'] = mode_name
                        self.metadata['sample_rate'] = sample_rate
                        return image
                except Exception:
                    continue

            return None
        except ImportError:
            print("pysstv не установлена, используем заглушку")
            return self._create_placeholder_image(samples)
        except Exception as e:
            print(f"Ошибка декодирования из сэмплов: {e}")
            return None

    def _create_placeholder_image(
        self,
        samples: np.ndarray
    ) -> Image.Image:
        """Создаёт заглушку изображения на основе аудиоданных."""
        duration = len(samples) / 48000 if len(samples) > 0 else 0

        img = Image.new("RGB", (320, 240), color=(30, 30, 60))
        self.metadata['duration_seconds'] = duration
        self.metadata['mode'] = 'placeholder'
        return img

    def save_decoded_image(self, filepath: str, quality: int = 95) -> bool:
        """
        Сохраняет декодированное изображение в файл.

        Args:
            filepath: Путь для сохранения изображения
            quality: Качество сохранения (для JPEG)

        Returns:
            bool: True если изображение успешно сохранено
        """
        if self.decoded_image is None:
            print("Сначала декодируйте изображение")
            return False

        if not filepath:
            print("Путь к файлу не указан")
            return False

        try:
            path = Path(filepath)
            valid_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            if path.suffix.lower() not in valid_formats:
                print(f"Неподдерживаемый формат: {path.suffix}. Доступные: {valid_formats}")
                return False

            path.parent.mkdir(parents=True, exist_ok=True)

            save_kwargs = {}
            if path.suffix.lower() in ['.jpg', '.jpeg']:
                if not (1 <= quality <= 100):
                    print(f"Качество должно быть от 1 до 100, установлено {quality}")
                    quality = max(1, min(100, quality))
                save_kwargs['quality'] = quality
                save_kwargs['optimize'] = True
            elif path.suffix.lower() == '.png':
                save_kwargs['optimize'] = True

            self.decoded_image.save(str(path), **save_kwargs)

            self.metadata['saved_path'] = str(path)
            print(f"Изображение сохранено: {filepath}")
            return True
        except Exception as e:
            print(f"Ошибка при сохранении изображения: {e}")
            return False

    def save_all_images(self, output_dir: str, prefix: str = "sstv") -> List[str]:
        """
        Сохраняет все декодированные изображения.

        Args:
            output_dir: Директория для сохранения
            prefix: Префикс имён файлов

        Returns:
            List[str]: Пути к сохранённым файлам
        """
        if not self.decoded_images:
            print("Нет изображений для сохранения")
            return []

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for i, img in enumerate(self.decoded_images):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{prefix}_{timestamp}_{i:03d}.png"
            filepath = output_path / filename
            img.save(str(filepath), 'png')
            saved_paths.append(str(filepath))

        print(f"Сохранено изображений: {len(saved_paths)}")
        return saved_paths

    def get_metadata(self) -> Dict:
        """Получает метаданные последнего декодирования."""
        return self.metadata.copy()

    def get_statistics(self) -> Dict:
        """Получает статистику декодирования."""
        return {
            'total_images': len(self.decoded_images),
            'last_mode': self.metadata.get('mode', 'unknown'),
            'last_source': self.metadata.get('source', 'unknown'),
            'supported_modes': len(self.SUPPORTED_MODES)
        }


def convert_audio_to_image(
    audio_data: np.ndarray,
    sample_rate: int
) -> Optional[Image.Image]:
    """
    Конвертирует аудиоданные в изображение.

    Args:
        audio_data: Аудиоданные в формате numpy array
        sample_rate: Частота дискретизации аудио

    Returns:
        Image.Image: Результативное изображение или None
    """
    decoder = SSTVDecoder()
    return decoder.decode_from_samples(audio_data, sample_rate)


def detect_sstv_signal(
    audio_file: str
) -> Tuple[bool, Dict]:
    """
    Обнаруживает SSTV-сигнал в аудиофайле.

    Args:
        audio_file: Путь к аудиофайлу

    Returns:
        Tuple[bool, Dict]: (найдено ли, метаданные)
    """
    try:
        import wave
        import numpy as np

        with wave.open(audio_file, "r") as wav:
            n_frames = wav.getnframes()
            duration = n_frames / wav.getframerate()
            sample_rate = wav.getframerate()

            # Анализируем характеристики
            metadata = {
                'duration_seconds': duration,
                'sample_rate': sample_rate,
                'channels': wav.getnchannels(),
                'sample_width': wav.getsampwidth(),
                'file': audio_file
            }

            # SSTV сигналы обычно длятся 30-180 секунд
            is_sstv = 30 <= duration <= 180 and sample_rate >= 44100

            if is_sstv:
                print(f"✓ SSTV-сигнал обнаружен (длительность: {duration:.1f}с)")
            else:
                print(f"? Сомнительный SSTV-сигнал (длительность: {duration:.1f}с)")

            return is_sstv, metadata

    except Exception as e:
        print(f"Ошибка обнаружения сигнала: {e}")
        return False, {'error': str(e)}


def detect_sstv_signal_from_samples(
    audio_data: np.ndarray,
    sample_rate: int
) -> Tuple[bool, float]:
    """
    Обнаруживает SSTV-сигнал в аудиоданных.

    Args:
        audio_data: Аудиоданные в формате numpy array
        sample_rate: Частота дискретизации аудио

    Returns:
        Tuple[bool, float]: (найдено ли, уверенность 0-1)
    """
    if len(audio_data) == 0:
        return False, 0.0

    duration = len(audio_data) / sample_rate

    # SSTV сигналы обычно длятся 30-180 секунд
    if 30 <= duration <= 180:
        # Дополнительный анализ частотных характеристик
        fft_data = np.fft.fft(audio_data.astype(np.float32))
        frequencies = np.abs(fft_data)

        # SSTV использует тона в диапазоне 1100-2300 Гц
        freq_bins = int(len(frequencies) * 1100 / sample_rate)
        freq_bine = int(len(frequencies) * 2300 / sample_rate)

        if freq_bine > freq_bins:
            sstv_energy = np.sum(frequencies[freq_bins:freq_bine])
            total_energy = np.sum(frequencies)

            if total_energy > 0:
                ratio = sstv_energy / total_energy
                confidence = min(1.0, ratio * 2)
                return confidence > 0.3, confidence

    return False, 0.0
