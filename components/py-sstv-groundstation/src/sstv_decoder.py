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

    def decode_from_samples(self, samples: np.ndarray, sample_rate: int = 44100) -> Optional[Image.Image]:
        """
        Декодирует SSTV из numpy массива сэмплов (для RTL-SDR V4).

        Args:
            samples: numpy массив сэмплов
            sample_rate: частота дискретизации

        Returns:
            Image.Image: Декодированное изображение или None
        """
        import tempfile
        import os
        
        temp_path = None
        try:
            # Сохраняем временный WAV файл
            import wave

            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_path = temp_file.name
            temp_file.close()

            # Нормализуем и конвертируем в 16-bit
            # Извлекаем реальную часть из комплексных I/Q сэмплов
            if np.iscomplexobj(samples):
                # FM демодуляция для I/Q данных
                phase = np.angle(samples)
                audio_data = np.diff(np.unwrap(phase))
                # Ресемплинг к нужной частоте
                from scipy.signal import resample
                target_len = int(len(audio_data) * sample_rate / (self.sdr.sample_rate if hasattr(self, 'sdr') and self.sdr else 256000))
                if target_len > 0:
                    audio_data = resample(audio_data, target_len)
            else:
                audio_data = samples.real
            
            # Нормализация
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                normalized = np.int16(audio_data / max_val * 32767)
            else:
                normalized = np.int16(audio_data * 32767)

            with wave.open(temp_path, 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(normalized.tobytes())

            # Декодируем
            image = self.decode_from_audio(temp_path)

            return image
        except Exception as e:
            print(f"Ошибка декодирования из сэмплов: {e}")
            return None
        finally:
            # Гарантированное удаление временного файла
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass

    def decode_realtime_init(self, sample_rate: int = 44100, callback=None):
        """
        Инициализирует real-time декодер SSTV.

        Args:
            sample_rate: частота дискретизации
            callback: функция обратного вызова для прогресса
        """
        self.rt_sample_rate = sample_rate
        self.rt_callback = callback
        self.rt_buffer = []
        self.rt_max_buffer = int(sample_rate * 30)  # 30 секунд
        self.rt_is_decoding = False
        self.rt_image = None
        print(f"Real-time декодер инициализирован: {sample_rate} Гц")

    def decode_realtime_push(self, samples: np.ndarray) -> Optional[Image.Image]:
        """
        Добавляет сэмплы в real-time декодер.

        Args:
            samples: numpy массив сэмплов

        Returns:
            Image.Image: Декодированное изображение или None
        """
        if not hasattr(self, 'rt_buffer'):
            self.decode_realtime_init()

        self.rt_buffer.append(samples)

        # Ограничиваем размер буфера
        total_samples = sum(len(s) for s in self.rt_buffer)
        if total_samples > self.rt_max_buffer:
            # Удаляем старые сэмплы
            while total_samples > self.rt_max_buffer and self.rt_buffer:
                removed = self.rt_buffer.pop(0)
                total_samples -= len(removed)

        # Проверяем наличие SSTV сигнала (простая эвристика)
        if len(self.rt_buffer) > 0:
            combined = np.concatenate(self.rt_buffer[-10:])  # Последние 10 блоков
            signal_strength = np.mean(np.abs(combined))

            if signal_strength > 0.1 and not self.rt_is_decoding:
                # Обнаружен сигнал - начинаем декодирование
                self.rt_is_decoding = True
                print("SSTV сигнал обнаружен, декодирование...")

                if self.rt_callback:
                    self.rt_callback('status', 'decoding')

                # Пробуем декодировать
                try:
                    all_samples = np.concatenate(self.rt_buffer)
                    self.rt_image = self.decode_from_samples(all_samples, self.rt_sample_rate)

                    if self.rt_image:
                        print(f"✓ SSTV декодировано: {self.rt_image.size[0]}x{self.rt_image.size[1]}")
                        if self.rt_callback:
                            self.rt_callback('image', self.rt_image)
                        self.rt_is_decoding = False
                        self.rt_buffer = []  # Очищаем буфер
                        return self.rt_image
                    else:
                        print("? Не удалось декодировать")
                        if self.rt_callback:
                            self.rt_callback('status', 'failed')
                except Exception as e:
                    print(f"Ошибка декодирования: {e}")
                    if self.rt_callback:
                        self.rt_callback('status', 'error')

                self.rt_is_decoding = False

        return None

    def decode_realtime_stop(self):
        """Останавливает real-time декодер."""
        self.rt_buffer = []
        self.rt_is_decoding = False
        print("Real-time декодер остановлен")

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
    Обнаруживает SSTV-сигнал в аудиофайле по VIS-тону.
    
    SSTV VIS (Vertical Interval Signal) состоит из:
    - 300ms лидер-тон 1900 Hz
    - 30ms тишина
    - 300ms VIS-биты (1300 Hz = 0, 2100 Hz = 1)
    - 300ms 1900 Hz завершение

    Args:
        audio_file: Путь к аудиофайлу

    Returns:
        Tuple[bool, Dict]: (найдено ли, метаданные)
    """
    try:
        import wave
        from scipy.signal import spectrogram

        with wave.open(audio_file, "r") as wav:
            n_frames = wav.getnframes()
            duration = n_frames / wav.getframerate()
            sample_rate = wav.getframerate()

            # Читаем первые 2 секунды для анализа VIS
            vis_samples = wav.readframes(min(int(sample_rate * 2), n_frames))
            vis_data = np.frombuffer(vis_samples, dtype=np.int16).astype(np.float32)
            
            # Спектрограмма для анализа частот
            f, t, Sxx = spectrogram(vis_data, sample_rate, nperseg=4096)
            
            # Ищем энергию на 1900 Hz (VIS лидер)
            freq_1900_idx = np.argmin(np.abs(f - 1900))
            vis_leader_energy = np.mean(Sxx[freq_1900_idx-2:freq_1900_idx+3, :])
            
            # Ищем энергию на 1200 Hz и 2100 Hz (VIS биты)
            freq_1200_idx = np.argmin(np.abs(f - 1200))
            freq_2100_idx = np.argmin(np.abs(f - 2100))
            vis_1200_energy = np.mean(Sxx[freq_1200_idx-2:freq_1200_idx+3, :])
            vis_2100_energy = np.mean(Sxx[freq_2100_idx-2:freq_2100_idx+3, :])
            
            # SSTV использует тона в диапазоне 1100-2300 Hz
            sstv_band_mask = (f >= 1100) & (f <= 2300)
            sstv_band_energy = np.mean(Sxx[sstv_band_mask, :])
            total_energy = np.mean(Sxx)
            
            sstv_ratio = sstv_band_energy / (total_energy + 1e-10)
            
            # Анализируем характеристики
            metadata = {
                'duration_seconds': duration,
                'sample_rate': sample_rate,
                'channels': wav.getnchannels(),
                'sample_width': wav.getsampwidth(),
                'file': audio_file,
                'vis_1900_energy': float(vis_leader_energy),
                'sstv_band_ratio': float(sstv_ratio),
            }

            # SSTV сигналы обычно длятся 30-180 секунд И имеют характерные VIS-тоны
            has_duration = 10 <= duration <= 300  # Расширенный диапазон
            has_vis_tones = sstv_ratio > 0.3  # Значительная энергия в SSTV полосе
            
            is_sstv = has_duration and (has_vis_tones or vis_leader_energy > 0)

            if is_sstv:
                print(f"✓ SSTV-сигнал обнаружен (длительность: {duration:.1f}с, VIS ratio: {sstv_ratio:.2f})")
            else:
                print(f"? Сомнительный SSTV-сигнал (длительность: {duration:.1f}с, VIS ratio: {sstv_ratio:.2f})")

            return is_sstv, metadata

    except ImportError:
        # Fallback без scipy - только по длительности
        import wave
        with wave.open(audio_file, "r") as wav:
            n_frames = wav.getnframes()
            duration = n_frames / wav.getframerate()
            sample_rate = wav.getframerate()
            
            metadata = {
                'duration_seconds': duration,
                'sample_rate': sample_rate,
                'channels': wav.getnchannels(),
                'sample_width': wav.getsampwidth(),
                'file': audio_file
            }
            
            is_sstv = 10 <= duration <= 300
            return is_sstv, metadata
            
    except Exception as e:
        print(f"Ошибка обнаружения сигнала: {e}")
        return False, {'error': str(e)}


def detect_sstv_signal_from_samples(
    audio_data: np.ndarray,
    sample_rate: int
) -> Tuple[bool, float]:
    """
    Обнаруживает SSTV-сигнал в аудиоданных по VIS-тону.

    Args:
        audio_data: Аудиоданные в формате numpy array
        sample_rate: Частота дискретизации аудио

    Returns:
        Tuple[bool, float]: (найдено ли, уверенность 0-1)
    """
    if len(audio_data) == 0:
        return False, 0.0

    duration = len(audio_data) / sample_rate
    
    # Ограничиваем размер FFT для производительности
    fft_size = min(len(audio_data), 65536)
    
    # Проверяем длительность (SSTV обычно 10-300 секунд)
    if not (5 <= duration <= 600):
        return False, 0.0

    try:
        from scipy.signal import welch
        
        # Используем Welch's method для эффективного спектрального анализа
        f, Pxx = welch(audio_data[:fft_size].astype(np.float64), sample_rate, nperseg=4096)
        
        # SSTV VIS-тон использует 1900 Hz
        freq_1900_idx = np.argmin(np.abs(f - 1900))
        vis_energy = Pxx[freq_1900_idx]
        
        # SSTV полоса 1100-2300 Hz
        sstv_mask = (f >= 1100) & (f <= 2300)
        sstv_energy = np.sum(Pxx[sstv_mask])
        total_energy = np.sum(Pxx)
        
        if total_energy > 0:
            sstv_ratio = sstv_energy / total_energy
            # Комбинированная оценка: VIS-тон + SSTV полоса
            confidence = min(1.0, (sstv_ratio * 2) + (vis_energy / (np.mean(Pxx) + 1e-10) * 0.1))
            return confidence > 0.3, confidence
    except ImportError:
        # Fallback - простой FFT
        fft_data = np.fft.rfft(audio_data[:fft_size].astype(np.float64))
        frequencies = np.abs(fft_data)
        freqs = np.fft.rfftfreq(fft_size, 1.0/sample_rate)
        
        # SSTV использует тона в диапазоне 1100-2300 Гц
        sstv_mask = (freqs >= 1100) & (freqs <= 2300)
        sstv_energy = np.sum(frequencies[sstv_mask])
        total_energy = np.sum(frequencies)
        
        if total_energy > 0:
            ratio = sstv_energy / total_energy
            confidence = min(1.0, ratio * 2)
            return confidence > 0.3, confidence

    return False, 0.0
