"""
RTL-SDR V4 SSTV Receiver
Поддержка RTL-SDR Blog V4 (R828D), FM демодуляция, VIS детектор, pysstv декодирование.
"""

import logging
import os
import tempfile
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from rtlsdr import RtlSdr

    RTLSDR_AVAILABLE = True
except ImportError:
    RTLSDR_AVAILABLE = False
    RtlSdr = None

try:
    from pysstv.color import PD90, PD120, PD180, MartinM1, MartinM2, Robot36, ScottieS1, ScottieS2
    from pysstv.grayscale import Robot8BW, Robot36BW

    SSTV_AVAILABLE = True
    SSTV_MODES = {
        "PD120": PD120,
        "PD90": PD90,
        "PD180": PD180,
        "MartinM1": MartinM1,
        "MartinM2": MartinM2,
        "ScottieS1": ScottieS1,
        "ScottieS2": ScottieS2,
        "Robot36": Robot36,
        "Robot36BW": Robot36BW,
        "Robot8BW": Robot8BW,
    }
except ImportError:
    SSTV_AVAILABLE = False
    SSTV_MODES = {}

# Аудио частота для SSTV декодирования
AUDIO_SAMPLE_RATE = 44100  # Гц — стандарт для pysstv

SSTV_FREQUENCIES = {
    "iss_sstv": 145.800,
    "noaa_15": 137.620,
    "noaa_18": 137.9125,
    "noaa_19": 137.100,
    "meteor_m2": 137.900,
}


def _fm_demodulate(
    iq_samples: np.ndarray, src_rate: int, dst_rate: int = AUDIO_SAMPLE_RATE
) -> np.ndarray:
    """
    FM демодуляция I/Q → аудио с anti-aliasing фильтром и ресемплингом.

    Args:
        iq_samples: Комплексные I/Q сэмплы
        src_rate: Исходная частота дискретизации (напр. 2_400_000)
        dst_rate: Целевая частота аудио (44100)

    Returns:
        np.ndarray: Аудио данные float32
    """
    # Дифференцирование фазы — FM демодуляция
    phase = np.angle(iq_samples)
    audio = np.diff(np.unwrap(phase)).astype(np.float32)

    if src_rate != dst_rate:
        from math import gcd

        from scipy.signal import firwin, lfilter, resample_poly

        # Anti-aliasing FIR перед ресемплингом
        nyq = src_rate / 2.0
        cutoff = min(dst_rate / 2.0 * 0.9, nyq * 0.9)
        fir = firwin(127, cutoff / nyq)
        audio = lfilter(fir, 1.0, audio)
        g = gcd(dst_rate, src_rate)
        audio = resample_poly(audio, dst_rate // g, src_rate // g).astype(np.float32)

    # Нормализация
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    return audio


def _detect_vis(audio: np.ndarray, sample_rate: int = AUDIO_SAMPLE_RATE) -> Dict[str, Any]:
    """
    Детектор VIS-кода SSTV.

    SSTV VIS структура:
    - 300ms лидер 1900 Hz
    - 10ms тишина
    - 300ms старт-бит 1200 Hz
    - 8 × 30ms биты (1300 Hz = 0, 1100 Hz = 1)
    - 30ms стоп-бит 1200 Hz

    Returns:
        dict: detected, confidence, vis_code, mode_hint
    """
    if len(audio) < sample_rate * 0.5:
        return {"detected": False, "confidence": 0.0}

    try:
        from scipy.signal import welch

        # Welch PSD для надёжной оценки спектра
        f, pxx = welch(
            audio[: min(len(audio), sample_rate * 2)].astype(np.float64), sample_rate, nperseg=2048
        )

        def _band_power(f_low, f_high):
            mask = (f >= f_low) & (f <= f_high)
            return float(np.mean(pxx[mask])) if mask.any() else 0.0

        p_1900 = _band_power(1850, 1950)  # VIS лидер
        # _p_1200 = _band_power(1150, 1250)  # старт/стоп бит (unused)
        p_1100 = _band_power(1050, 1150)  # бит "1"
        p_1300 = _band_power(1250, 1350)  # бит "0"
        p_total = float(np.mean(pxx)) + 1e-12

        # Отношение SSTV-полосы к общей мощности
        p_sstv = _band_power(1050, 2350)
        sstv_ratio = p_sstv / p_total

        # Наличие лидер-тона — главный признак
        leader_ratio = p_1900 / p_total
        confidence = min(1.0, leader_ratio * 3.0 + sstv_ratio * 0.5)

        detected = confidence > 0.25

        # Грубая оценка режима по соотношению битов
        mode_hint = None
        if detected:
            if p_1100 > p_1300:
                mode_hint = "MartinM1"  # больше единиц → Martin
            else:
                mode_hint = "PD120"  # больше нулей → PD

        return {
            "detected": detected,
            "confidence": round(confidence, 3),
            "leader_power": round(leader_ratio, 4),
            "sstv_ratio": round(sstv_ratio, 3),
            "mode_hint": mode_hint,
        }
    except Exception as e:
        logger.debug(f"VIS detection error: {e}")
        return {"detected": False, "confidence": 0.0}


class RTLSDRReceiver:
    """RTL-SDR V4 приёмник с правильной FM демодуляцией."""

    def __init__(
        self,
        device_index: int = 0,
        frequency: float = 145.800,
        sample_rate: float = 2.4e6,
        gain: float = 30.0,
        bias_tee: bool = False,
        agc: bool = False,
        freq_correction_ppm: float = 0.0,
    ):
        self.device_index = device_index
        self.frequency = frequency
        self.sample_rate = int(sample_rate)
        self.gain = gain
        self.bias_tee = bias_tee
        self.agc = agc
        self.freq_correction_ppm = freq_correction_ppm

        self.sdr: Optional[Any] = None
        self.is_running = False
        self._hann_window: Optional[np.ndarray] = None

        self.stats = {
            "samples_received": 0,
            "recording_sessions": 0,
            "total_recording_time": 0.0,
            "signal_detections": 0,
        }

    def initialize(self) -> bool:
        """Инициализация устройства. Идемпотентна — повторный вызов безопасен."""
        if self.sdr is not None:
            return True  # уже открыто

        if not RTLSDR_AVAILABLE:
            logger.error("pyrtlsdr не установлен: pip install pyrtlsdr")
            return False

        try:
            self.sdr = RtlSdr(self.device_index)
            self.sdr.rs = self.sample_rate
            self.sdr.fc = int(self.frequency * 1e6)
            self.sdr.gain = "auto" if self.agc else self.gain

            if self.freq_correction_ppm != 0.0:
                self.sdr.freq_correction = int(self.freq_correction_ppm)

            if self.bias_tee and hasattr(self.sdr, "bias_tee"):
                self.sdr.bias_tee = True

            # Hann-окно для FFT
            fft_size = 2048
            self._hann_window = np.hanning(fft_size).astype(np.float32)

            logger.info(
                f"RTL-SDR V4 инициализирован: {self.frequency} МГц, "
                f"SR={self.sample_rate/1e6:.1f} MSPS, gain={self.gain} dB"
            )
            return True

        except Exception as e:
            logger.error(f"Ошибка инициализации RTL-SDR: {e}")
            self.sdr = None
            return False

    def get_device_info(self) -> Dict[str, Any]:
        """Информация об устройстве."""
        info: Dict[str, Any] = {
            "available": RTLSDR_AVAILABLE,
            "connected": self.sdr is not None,
            "device_index": self.device_index,
            "frequency_mhz": self.frequency,
            "sample_rate_msps": self.sample_rate / 1e6,
            "gain_db": self.gain,
            "bias_tee": self.bias_tee,
            "agc": self.agc,
            "freq_correction_ppm": self.freq_correction_ppm,
        }
        if self.sdr:
            try:
                info["tuner_type"] = str(self.sdr.get_tuner_type())
                info["actual_gain"] = self.sdr.gain
                info["actual_freq_mhz"] = self.sdr.fc / 1e6
            except Exception:
                pass
        return info

    def read_samples(self, num_samples: int = 8192) -> Optional[np.ndarray]:
        """Читает I/Q сэмплы."""
        if not self.sdr:
            return None
        try:
            samples = self.sdr.read_samples(num_samples)
            self.stats["samples_received"] += len(samples)
            return samples
        except Exception as e:
            logger.error(f"Ошибка чтения сэмплов: {e}")
            return None

    def record_audio(
        self,
        duration: float = 60.0,
        output_file: Optional[str] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Optional[np.ndarray]:
        """
        Запись и FM демодуляция сигнала.

        Читает I/Q через async callback (не sleep-loop), демодулирует FM,
        ресемплирует до AUDIO_SAMPLE_RATE, сохраняет WAV.

        Returns:
            np.ndarray: Аудио float32 или None
        """
        if not self.sdr:
            logger.error("Устройство не инициализировано")
            return None

        self.is_running = True
        self.stats["recording_sessions"] += 1
        start_time = time.time()

        # Размер буфера: кратен 512, ~0.1с при 2.4 MSPS
        buf_size = 256 * 1024  # 256k сэмплов ≈ 0.107с
        iq_chunks = []

        logger.info(f"Запись {duration}с на {self.frequency} МГц...")

        try:

            def _callback(samples, _ctx):
                if not self.is_running:
                    raise StopIteration
                iq_chunks.append(samples.copy())
                elapsed = time.time() - start_time
                if progress_callback:
                    progress_callback(elapsed / duration)
                if elapsed >= duration:
                    raise StopIteration

            try:
                self.sdr.read_samples_async(_callback, buf_size)
            except StopIteration:
                pass

        except Exception as e:
            logger.error(f"Ошибка записи: {e}")
        finally:
            self.is_running = False
            try:
                self.sdr.cancel_read_async()
            except Exception:
                pass

        if not iq_chunks:
            return None

        iq_all = np.concatenate(iq_chunks)
        self.stats["total_recording_time"] += time.time() - start_time

        # FM демодуляция
        audio = _fm_demodulate(iq_all, self.sample_rate, AUDIO_SAMPLE_RATE)

        if output_file:
            self._save_wav(audio, output_file)

        logger.info(f"Запись завершена: {len(audio)/AUDIO_SAMPLE_RATE:.1f}с аудио")
        return audio

    def _save_wav(self, audio: np.ndarray, output_file: str):
        """Сохраняет аудио float32 в 16-bit WAV."""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        with wave.open(output_file, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(AUDIO_SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())
        logger.info(f"WAV сохранён: {output_file}")

    def detect_sstv_signal(self, audio: np.ndarray) -> Dict[str, Any]:
        """Детектирует SSTV VIS-код в аудио."""
        result = _detect_vis(audio, AUDIO_SAMPLE_RATE)
        if result.get("detected"):
            self.stats["signal_detections"] += 1
        return result

    def get_signal_strength(self) -> float:
        """Сила сигнала в dBFS (0 = полная шкала)."""
        if not self.sdr:
            return -100.0
        samples = self.read_samples(4096)
        if samples is None or len(samples) == 0:
            return -100.0
        power = float(np.mean(np.abs(samples) ** 2))
        return float(10 * np.log10(power + 1e-12))

    def get_spectrum(self, num_points: int = 2048) -> tuple:
        """
        Спектр с Hann-окном.

        Returns:
            (frequencies_hz, power_db) или (None, None)
        """
        if not self.sdr:
            return None, None
        samples = self.read_samples(num_points)
        if samples is None:
            return None, None

        window = (
            self._hann_window[:num_points]
            if self._hann_window is not None
            else np.hanning(num_points)
        )
        windowed = samples[:num_points] * window
        fft = np.fft.fftshift(np.fft.fft(windowed))
        power_db = 10 * np.log10(np.abs(fft) ** 2 + 1e-12).astype(np.float32)
        freqs = (
            np.fft.fftshift(np.fft.fftfreq(num_points, 1.0 / self.sample_rate))
            + self.frequency * 1e6
        )
        return freqs, power_db

    def set_frequency(self, freq_mhz: float) -> bool:
        """Меняет частоту на лету."""
        if not self.sdr:
            return False
        try:
            self.sdr.fc = int(freq_mhz * 1e6)
            self.frequency = freq_mhz
            return True
        except Exception as e:
            logger.error(f"Ошибка установки частоты: {e}")
            return False

    def set_gain(self, gain_db: float) -> bool:
        """Меняет усиление на лету."""
        if not self.sdr:
            return False
        try:
            self.sdr.gain = gain_db
            self.gain = gain_db
            return True
        except Exception as e:
            logger.error(f"Ошибка установки усиления: {e}")
            return False

    def close(self):
        """Закрывает устройство."""
        self.is_running = False
        if self.sdr:
            try:
                self.sdr.cancel_read_async()
            except Exception:
                pass
            try:
                self.sdr.close()
            except Exception:
                pass
            finally:
                self.sdr = None
        logger.info("RTL-SDR закрыт")

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, *_):
        self.close()


class SSTVDecoder:
    """Декодер SSTV через pysstv."""

    def __init__(self, mode: str = "auto"):
        self.mode = mode
        self.last_image = None
        self.stats = {
            "attempts": 0,
            "successes": 0,
            "last_time": None,
            "last_size": None,
        }

    def decode_audio(
        self, audio: np.ndarray, sample_rate: int = AUDIO_SAMPLE_RATE
    ) -> Optional[Any]:
        """
        Декодирует SSTV из аудио float32.

        Сохраняет временный WAV и передаёт в pysstv.
        """
        if not SSTV_AVAILABLE:
            logger.error("pysstv не установлен: pip install pysstv")
            return None

        self.stats["attempts"] += 1

        # Нормализация
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
        pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name

            with wave.open(tmp_path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm.tobytes())

            modes_to_try = (
                [SSTV_MODES[self.mode]]
                if self.mode != "auto" and self.mode in SSTV_MODES
                else list(SSTV_MODES.values())
            )

            for mode_cls in modes_to_try:
                if mode_cls is None:
                    continue
                try:
                    img = self._try_decode(tmp_path, sample_rate, mode_cls)
                    if img is not None:
                        self._record_success(img)
                        return img
                except Exception:
                    continue

            logger.warning("SSTV декодирование не удалось ни в одном режиме")
            return None

        except Exception as e:
            logger.error(f"Ошибка декодирования: {e}")
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def _try_decode(self, wav_path: str, sample_rate: int, mode_cls) -> Optional[Any]:
        """Попытка декодирования одним режимом через pysstv API."""
        try:
            # pysstv имеет метод from_samples который принимает WAV файл напрямую
            # или можно использовать SSTV класс с аудио данными
            from pysstv.sstv import SSTV

            # Читаем WAV файл
            with wave.open(wav_path, "r") as wf:
                frames = wf.readframes(wf.getnframes())
                audio_int16 = np.frombuffer(frames, dtype=np.int16)

            # Конвертируем в float32 нормализованный [-1, 1]
            audio_float = audio_int16.astype(np.float32) / 32768.0

            # pysstv ожидает аудио данные как итерируемый объект
            # Используем правильный API: создаём экземпляр SSTV с аудио
            sstv_instance = SSTV(
                audio_float.tolist(),  # Аудио сэмплы как list
                sample_rate,  # Частота дискретизации
                16,  # Битность (хотя данные float32)
            )

            # Декодируем изображение
            img = sstv_instance.decode()

            # Проверяем что изображение получено
            if img is not None and hasattr(img, "size") and img.size[0] > 0:
                return img

        except Exception as e:
            logger.debug(f"Decode attempt failed with {mode_cls.__name__}: {e}")

        return None

    def _record_success(self, img):
        self.stats["successes"] += 1
        self.stats["last_time"] = datetime.now(timezone.utc).isoformat()
        self.stats["last_size"] = getattr(img, "size", None)
        self.last_image = img


# ── Singletons ──────────────────────────────────────────────────────────────

_receiver: Optional[RTLSDRReceiver] = None
_decoder: Optional[SSTVDecoder] = None


def get_receiver() -> RTLSDRReceiver:
    global _receiver
    if _receiver is None:
        _receiver = RTLSDRReceiver(
            frequency=float(os.getenv("SSTV_FREQUENCY", "145.800")),
            gain=float(os.getenv("SSTV_GAIN", "30.0")),
            sample_rate=float(os.getenv("SSTV_SAMPLE_RATE", "2400000")),
            bias_tee=os.getenv("SSTV_BIAS_TEE", "0") == "1",
            agc=os.getenv("SSTV_AGC", "0") == "1",
            freq_correction_ppm=float(os.getenv("SSTV_PPM", "0")),
        )
    return _receiver


def get_decoder() -> SSTVDecoder:
    global _decoder
    if _decoder is None:
        _decoder = SSTVDecoder(mode=os.getenv("SSTV_MODE", "auto"))
    return _decoder
