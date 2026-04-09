"""
RTL-SDR V4 SSTV Receiver - Полноценный приемник SSTV сигналов
Поддержка RTL-SDR Blog V4 с тюнером R828D
Интеграция с pysstv для декодирования
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)

# Импорт RTL-SDR
try:
    from rtlsdr import RtlSdr
    RTLSDR_AVAILABLE = True
except ImportError:
    RTLSDR_AVAILABLE = False
    RtlSdr = None

# Импорт SSTV
try:
    from pysstv.sstv import SSTV
    from pysstv.color import PD120, PD90, PD180, MartinM1, MartinM2, ScottieS1, ScottieS2, Robot36
    from pysstv.grayscale import Robot36BW, Robot8BW
    SSTV_AVAILABLE = True
except ImportError:
    SSTV_AVAILABLE = False

# SSTV режимы
SSTV_MODES = {
    'PD120': PD120 if SSTV_AVAILABLE else None,
    'PD90': PD90 if SSTV_AVAILABLE else None,
    'PD180': PD180 if SSTV_AVAILABLE else None,
    'MartinM1': MartinM1 if SSTV_AVAILABLE else None,
    'MartinM2': MartinM2 if SSTV_AVAILABLE else None,
    'ScottieS1': ScottieS1 if SSTV_AVAILABLE else None,
    'ScottieS2': ScottieS2 if SSTV_AVAILABLE else None,
    'Robot36': Robot36 if SSTV_AVAILABLE else None,
    'Robot36BW': Robot36BW if SSTV_AVAILABLE else None,
    'Robot8BW': Robot8BW if SSTV_AVAILABLE else None,
}

# Частоты SSTV
SSTV_FREQUENCIES = {
    'iss_sstv': 145.800,  # МКС основная частота
    'noaa_apt': 137.100,  # NOAA APT
    'meteor_m2': 137.100, # Meteor-M2
}


class RTLSDRReceiver:
    """RTL-SDR приемник с поддержкой V4"""

    def __init__(
        self,
        device_index: int = 0,
        frequency: float = 145.800,
        sample_rate: float = 2.4e6,
        gain: int = 49.6,
        bias_tee: bool = False,
        agc: bool = False
    ):
        """
        Инициализация RTL-SDR приемника

        Args:
            device_index: Индекс устройства
            frequency: Частота приема (МГц)
            sample_rate: Частота дискретизации (Гц)
            gain: Усиление (дБ)
            bias_tee: Включить Bias-T
            agc: Автоматическая регулировка усиления
        """
        self.device_index = device_index
        self.frequency = frequency
        self.sample_rate = sample_rate
        self.gain = gain
        self.bias_tee = bias_tee
        self.agc = agc
        
        self.sdr = None
        self.is_running = False
        self.recording_data = []
        
        # Статистика
        self.stats = {
            'samples_received': 0,
            'recording_sessions': 0,
            'total_recording_time': 0,
            'signal_detections': 0,
        }

    def initialize(self) -> bool:
        """Инициализация RTL-SDR устройства"""
        if not RTLSDR_AVAILABLE:
            logger.error("pyrtlsdr не установлен")
            return False

        try:
            self.sdr = RtlSdr(self.device_index)
            self.sdr.rs = self.sample_rate
            self.sdr.fc = self.frequency * 1e6
            self.sdr.gain = self.gain
            
            if hasattr(self.sdr, 'bias_tee'):
                self.sdr.bias_tee = self.bias_tee
            
            if self.agc:
                self.sdr.manual_gain = False
            
            logger.info(f"RTL-SDR инициализирован: {self.frequency} МГц, gain {self.gain} дБ")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации RTL-SDR: {e}")
            return False

    def get_device_info(self) -> Dict[str, Any]:
        """Получить информацию об устройстве"""
        info = {
            'device_index': self.device_index,
            'frequency_mhz': self.frequency,
            'sample_rate_hz': self.sample_rate,
            'gain_db': self.gain,
            'bias_tee': self.bias_tee,
            'agc': self.agc,
            'connected': self.sdr is not None,
        }
        
        if self.sdr:
            try:
                info['tuner_type'] = self.sdr.get_tuner_type()
                info['center_freq'] = self.sdr.fc
                info['sample_rate'] = self.sdr.rs
                info['gain'] = self.sdr.gain
                info['frequency_correction'] = getattr(self.sdr, 'freq_correction', 0)
            except Exception as e:
                logger.warning(f"Ошибка получения информации: {e}")
        
        return info

    def read_samples(self, num_samples: int = 1024) -> Optional[np.ndarray]:
        """Прочитать сэмплы с устройства"""
        if not self.sdr:
            return None
        
        try:
            samples = self.sdr.read_samples(num_samples)
            self.stats['samples_received'] += len(samples)
            return samples
        except Exception as e:
            logger.error(f"Ошибка чтения сэмплов: {e}")
            return None

    def record_audio(
        self,
        duration: float = 60.0,
        sample_rate: int = 48000,
        output_file: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """
        Записать аудио сигнал для SSTV декодирования

        Args:
            duration: Длительность записи (секунды)
            sample_rate: Частота дискретизации аудио
            output_file: Путь для сохранения WAV

        Returns:
            numpy array с аудио данными
        """
        if not self.sdr:
            logger.error("Устройство не инициализировано")
            return None

        self.is_running = True
        self.stats['recording_sessions'] += 1
        start_time = time.time()
        
        logger.info(f"Начало записи: {duration}с, {sample_rate} Гц")
        
        all_samples = []
        samples_per_read = int(self.sample_rate / 10)  # 10 чтений в секунду
        
        try:
            while self.is_running and (time.time() - start_time) < duration:
                samples = self.read_samples(samples_per_read)
                if samples is not None:
                    all_samples.append(samples)
                
                # Небольшая задержка чтобы не перегружать CPU
                time.sleep(0.1)
            
            # Конвертируем в numpy array
            if all_samples:
                audio_data = np.concatenate(all_samples)
                self.stats['total_recording_time'] += time.time() - start_time
                
                # Сохраняем в файл если указано
                if output_file:
                    self._save_wav(audio_data, output_file, sample_rate)
                
                logger.info(f"Запись завершена: {len(audio_data)} сэмплов")
                return audio_data
            
        except KeyboardInterrupt:
            logger.info("Запись прервана пользователем")
        except Exception as e:
            logger.error(f"Ошибка записи: {e}")
        finally:
            self.is_running = False
        
        return None

    def _save_wav(self, audio_data: np.ndarray, output_file: str, sample_rate: int):
        """Сохранить данные в WAV файл"""
        import wave
        import struct
        
        # Нормализуем и конвертируем в 16-bit PCM
        audio_normalized = audio_data.real / np.max(np.abs(audio_data))
        audio_int16 = (audio_normalized * 32767).astype(np.int16)
        
        with wave.open(output_file, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        logger.info(f"WAV сохранен: {output_file}")

    def detect_sstv_signal(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Обнаружить SSTV сигнал в аудио данных

        Args:
            audio_data: numpy array с аудио

        Returns:
            Dict с результатами обнаружения
        """
        if audio_data is None or len(audio_data) == 0:
            return {'detected': False, 'reason': 'No audio data'}

        # Простой детектор VIS кода
        # SSTV VIS коды: 1000-1200 Hz для синхронизации
        from scipy.signal import find_peaks
        import numpy.fft as fft

        # FFT анализ
        fft_data = np.abs(fft.fft(audio_data))
        freqs = fft.fftfreq(len(audio_data), 1/48000)
        
        # Ищем пики в диапазоне VIS частот (1000-1200 Hz)
        vis_mask = (freqs >= 1000) & (freqs <= 1200)
        vis_peaks, _ = find_peaks(fft_data[vis_mask], height=np.mean(fft_data[vis_mask]) * 2)
        
        detected = len(vis_peaks) > 0
        self.stats['signal_detections'] += 1 if detected else 0
        
        return {
            'detected': detected,
            'vis_peaks_count': len(vis_peaks),
            'signal_strength': np.max(fft_data[vis_mask]) if len(vis_peaks) > 0 else 0,
            'audio_duration': len(audio_data) / 48000,
        }

    def get_signal_strength(self) -> float:
        """Получить текущую силу сигнала"""
        if not self.sdr:
            return 0.0
        
        try:
            samples = self.read_samples(1024)
            if samples is not None:
                return np.mean(np.abs(samples)) * 100  # Процент
        except Exception:
            pass
        
        return 0.0

    def get_spectrum(self, num_points: int = 1024) -> tuple:
        """
        Получить спектр сигнала

        Returns:
            (frequencies, power) tuple
        """
        if not self.sdr:
            return None, None
        
        try:
            samples = self.read_samples(num_points)
            if samples is not None:
                fft_data = np.fft.fft(samples)
                power = np.abs(fft_data) ** 2
                freqs = np.fft.fftfreq(len(samples), 1/self.sample_rate)
                
                # Смещаем к центру
                power = np.fft.fftshift(power)
                freqs = np.fft.fftshift(freqs)
                
                # Конвертируем в дБ
                power_db = 10 * np.log10(power + 1e-10)
                
                return freqs + self.frequency * 1e6, power_db
        except Exception as e:
            logger.error(f"Ошибка получения спектра: {e}")
        
        return None, None

    def close(self):
        """Закрыть соединение с устройством"""
        if self.sdr:
            try:
                self.sdr.close()
                logger.info("RTL-SDR устройство закрыто")
            except Exception as e:
                logger.warning(f"Ошибка закрытия: {e}")
            finally:
                self.sdr = None


class SSTVDecoder:
    """Декодер SSTV сигналов"""

    def __init__(self, mode: str = 'auto'):
        self.mode = mode
        self.last_decoded_image = None
        self.decode_stats = {
            'total_attempts': 0,
            'successful_decodes': 0,
            'last_decode_time': None,
            'last_image_size': None,
        }

    def decode_audio(self, audio_data: np.ndarray, sample_rate: int = 48000) -> Optional[Any]:
        """
        Декодировать SSTV из аудио данных

        Args:
            audio_data: numpy array аудио
            sample_rate: Частота дискретизации

        Returns:
            PIL Image или None
        """
        self.decode_stats['total_attempts'] += 1
        
        try:
            # Конвертируем в нужный формат для pysstv
            audio_int16 = (audio_data.real * 32767).astype(np.int16)
            
            # Пытаемся декодировать с разными режимами
            if self.mode == 'auto':
                for mode_name, mode_class in SSTV_MODES.items():
                    if mode_class is None:
                        continue
                    try:
                        image = self._decode_with_mode(audio_int16, sample_rate, mode_class)
                        if image is not None:
                            self._record_success(image)
                            return image
                    except Exception:
                        continue
            else:
                mode_class = SSTV_MODES.get(self.mode)
                if mode_class:
                    image = self._decode_with_mode(audio_int16, sample_rate, mode_class)
                    if image is not None:
                        self._record_success(image)
                        return image
            
            logger.warning("Не удалось декодировать SSTV")
            return None
            
        except Exception as e:
            logger.error(f"Ошибка декодирования: {e}")
            return None

    def _decode_with_mode(self, audio_data, sample_rate, mode_class) -> Optional[Any]:
        """Декодировать с конкретным режимом"""
        # Здесь нужна интеграция с pysstv
        # pysstv обычно работает с файлами, так что сохраняем временный файл
        import tempfile
        import wave
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with wave.open(tmp_path, 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            # Пробуем декодировать через pysstv
            from pysstv.sstv import SSTV
            decoder = SSTV(tmp_path)
            image = decoder.decode()
            return image
        except Exception:
            return None
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _record_success(self, image):
        """Записать успешное декодирование"""
        self.decode_stats['successful_decodes'] += 1
        self.decode_stats['last_decode_time'] = datetime.now().isoformat()
        self.decode_stats['last_image_size'] = image.size if hasattr(image, 'size') else None
        self.last_decoded_image = image


# Singleton instance
_receiver = None
_decoder = None


def get_receiver() -> Optional[RTLSDRReceiver]:
    """Получить singleton receiver"""
    global _receiver
    if _receiver is None:
        _receiver = RTLSDRReceiver(
            frequency=float(os.getenv("SSTV_FREQUENCY", "145.800")),
            gain=float(os.getenv("SSTV_GAIN", "49.6")),
            sample_rate=float(os.getenv("SSTV_SAMPLE_RATE", "2400000")),
        )
    return _receiver


def get_decoder() -> SSTVDecoder:
    """Получить singleton decoder"""
    global _decoder
    if _decoder is None:
        _decoder = SSTVDecoder(mode=os.getenv("SSTV_MODE", "auto"))
    return _decoder
