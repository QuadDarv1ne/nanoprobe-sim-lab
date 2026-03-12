# -*- coding: utf-8 -*-
"""SDR интерфейс для приема SSTV сигналов."""

import numpy as np
from typing import Optional, Tuple, List, Dict, Callable
from pathlib import Path
from datetime import datetime
import threading
import queue
import time
import platform


class SDRInterface:
    """Интерфейс для работы с SDR устройствами."""

    # Частоты для приема SSTV (МГц)
    FREQUENCIES = {
        'iss': 145.800,      # МКС
        'noaa_15': 137.620,  # NOAA 15
        'noaa_18': 137.9125, # NOAA 18
        'noaa_19': 137.100,  # NOAA 19
        'meteor_m2': 137.900,# Метеор-М2
        'vhf_2m': 144.000,   # 2m диапазон
        'uhf_70cm': 430.000, # 70cm диапазон
    }

    # Поддерживаемые устройства
    SUPPORTED_DEVICES = {
        'rtl2832u': 'RTL-SDR (RTL2832U)',
        'rtl2838': 'RTL-SDR (RTL2838)',
        'r820t': 'RTL-SDR (R820T)',
        'r828d': 'RTL-SDR V4 (R828D)',
        'airspy': 'Airspy',
        'hackrf': 'HackRF',
        'sdrplay': 'SDRplay',
        'bladerf': 'bladeRF',
        'unknown': 'Unknown SDR',
    }

    def __init__(
        self,
        device_index: int = 0,
        sample_rate: int = 2400000,  # 2.4 MSPS для RTL-SDR V4
        center_freq: float = 145.800,
        gain: int = 30,
        device_type: str = 'auto'
    ):
        """
        Инициализирует SDR интерфейс.

        Args:
            device_index: Индекс SDR устройства
            sample_rate: Частота дискретизации (по умолчанию 2.4 MSPS)
            center_freq: Центральная частота (МГц)
            gain: Усиление (0-50 dB)
            device_type: Тип устройства ('auto', 'rtl2832u', 'r828d', 'airspy', etc.)
        """
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.gain = gain
        self.device_type = device_type
        self.sdr = None
        self.device_name = None
        self.is_recording = False
        self.is_scanning = False
        self.audio_queue: queue.Queue = queue.Queue()
        self.recording_thread: Optional[threading.Thread] = None
        self.scanning_thread: Optional[threading.Thread] = None
        self.callback: Optional[Callable] = None
        self.recorded_samples: List[np.ndarray] = []
        self.metadata: Dict = {}

    def initialize(self) -> bool:
        """
        Инициализирует SDR устройство с автоопределением типа.

        Returns:
            bool: True если успешно
        """
        try:
            from rtlsdr import RtlSdr

            # Пробуем открыть устройство
            self.sdr = RtlSdr(device_index=self.device_index)
            
            # Определяем тип устройства
            self._detect_device_type()
            
            # Настраиваем параметры в зависимости от устройства
            self._configure_device()
            
            self.metadata['device'] = self.device_name or f"SDR #{self.device_index}"
            self.metadata['sample_rate'] = self.sample_rate
            self.metadata['center_freq'] = self.center_freq
            self.metadata['gain'] = self.gain
            self.metadata['initialized'] = True
            self.metadata['device_type'] = self.device_type

            print(f"✓ SDR инициализирован: {self.device_name}")
            print(f"  Частота: {self.center_freq} МГц, Sample Rate: {self.sample_rate} sps, Gain: {self.gain} dB")
            return True

        except ImportError:
            print("rtlsdr не установлен. Установите: pip install rtlsdr")
            print("Также установите librtlsdr драйверы")
            self.metadata['error'] = 'rtlsdr not installed'
            return False

        except Exception as e:
            print(f"Ошибка инициализации SDR: {e}")
            self.metadata['error'] = str(e)
            return False

    def _detect_device_type(self) -> str:
        """
        Определяет тип подключенного SDR устройства.

        Returns:
            str: Тип устройства
        """
        if self.device_type != 'auto':
            self.device_name = self.SUPPORTED_DEVICES.get(
                self.device_type, f"Unknown ({self.device_type})"
            )
            return self.device_type

        try:
            # Получаем информацию об устройстве
            if hasattr(self.sdr, 'device_name'):
                device_name = self.sdr.device_name.lower()
            elif hasattr(self.sdr, 'get_device_name'):
                device_name = self.sdr.get_device_name().lower()
            else:
                device_name = ''

            # Определяем тип по имени
            if 'r828d' in device_name or 'v4' in device_name:
                self.device_type = 'r828d'
                self.device_name = 'RTL-SDR V4 (R828D)'
            elif 'r820t' in device_name or 'r820t2' in device_name:
                self.device_type = 'r820t'
                self.device_name = 'RTL-SDR (R820T/R820T2)'
            elif 'rtl2838' in device_name:
                self.device_type = 'rtl2838'
                self.device_name = 'RTL-SDR (RTL2838)'
            elif 'rtl2832' in device_name or 'rtl2832u' in device_name:
                self.device_type = 'rtl2832u'
                self.device_name = 'RTL-SDR (RTL2832U)'
            elif 'airspy' in device_name:
                self.device_type = 'airspy'
                self.device_name = 'Airspy'
            elif 'hackrf' in device_name:
                self.device_type = 'hackrf'
                self.device_name = 'HackRF'
            else:
                self.device_type = 'unknown'
                self.device_name = self.SUPPORTED_DEVICES.get('unknown', 'Unknown SDR')

            print(f"Обнаружено устройство: {self.device_name}")
            return self.device_type

        except Exception as e:
            print(f"Не удалось определить тип устройства: {e}")
            self.device_type = 'unknown'
            self.device_name = 'Unknown SDR'
            return self.device_type

    def _configure_device(self):
        """
        Настраивает параметры устройства в зависимости от типа.
        """
        # RTL-SDR V4 (R828D) требует sample rate 2.4 MSPS
        if self.device_type == 'r828d':
            self.sample_rate = 2400000  # 2.4 MSPS
            print("  Оптимизировано для RTL-SDR V4")
        
        # Классические RTL-SDR работают на 1-3 MSPS
        elif self.device_type in ['rtl2832u', 'rtl2838', 'r820t']:
            if self.sample_rate < 1000000:
                self.sample_rate = 2000000  # 2 MSPS по умолчанию
            print("  Оптимизировано для классического RTL-SDR")

        # Применяем настройки
        self.sdr.sample_rate = self.sample_rate
        self.sdr.center_freq = self.center_freq * 1e6
        self.sdr.gain = self.gain
        
        # Для RTL-SDR V4 включаем прямой режим (direct sampling) если нужно
        if self.device_type == 'r828d':
            try:
                # Пробуем включить прямой режим для УКВ
                if self.center_freq < 24:  # Если частота < 24 МГц
                    self.sdr.direct_sampling = 1
                    print("  Включен прямой режим (direct sampling)")
            except Exception:
                pass

    def get_frequency_range(self) -> Tuple[float, float]:
        """
        Возвращает диапазон частот устройства.

        Returns:
            Tuple[float, float]: (min_freq_hz, max_freq_hz)
        """
        if self.device_type == 'r828d':
            # RTL-SDR V4: 24 МГц - 1766 МГц (с прямым режимом до 28 МГц)
            return (24e6, 1766e6)
        elif self.device_type in ['rtl2832u', 'rtl2838', 'r820t']:
            # Классические RTL-SDR: 24 МГц - 1766 МГц
            return (24e6, 1766e6)
        elif self.device_type == 'airspy':
            # Airspy: 24 МГц - 1800 МГц
            return (24e6, 1800e6)
        elif self.device_type == 'hackrf':
            # HackRF: 1 МГц - 6000 МГц
            return (1e6, 6000e6)
        else:
            # По умолчанию
            return (24e6, 1766e6)

    def set_frequency(self, freq_mhz: float) -> bool:
        """
        Устанавливает частоту приема.

        Args:
            freq_mhz: Частота в МГц

        Returns:
            bool: True если успешно
        """
        if self.sdr is None:
            print("SDR не инициализирован")
            return False

        # Проверка диапазона частот для разных устройств
        if self.device_type == 'r828d':
            # RTL-SDR V4: 24 МГц - 1766 МГц (с прямым режимом до 28 МГц)
            if freq_mhz < 24 or freq_mhz > 1766:
                if freq_mhz >= 1.5 and freq_mhz < 24:
                    print(f"Предупреждение: Частота {freq_mhz} МГц требует прямого режима")
                else:
                    print(f"Частота {freq_mhz} МГц вне диапазона устройства")
                    return False
        elif self.device_type in ['rtl2832u', 'rtl2838', 'r820t']:
            # Классические RTL-SDR: 24 МГц - 1766 МГц
            if freq_mhz < 24 or freq_mhz > 1766:
                print(f"Частота {freq_mhz} МГц вне диапазона устройства")
                return False

        try:
            self.sdr.center_freq = freq_mhz * 1e6
            self.center_freq = freq_mhz
            print(f"Частота установлена: {freq_mhz} МГц")
            return True
        except Exception as e:
            print(f"Ошибка установки частоты: {e}")
            return False

    @staticmethod
    def list_devices() -> List[Dict]:
        """
        Возвращает список доступных SDR устройств.

        Returns:
            List[Dict]: Список устройств с информацией
        """
        devices = []
        try:
            from rtlsdr import RtlSdr

            # Получаем количество устройств
            num_devices = RtlSdr.get_device_count()

            for i in range(num_devices):
                try:
                    device = RtlSdr(device_index=i)
                    device_info = {
                        'index': i,
                        'name': device.get_device_name() if hasattr(device, 'get_device_name') else 'Unknown',
                        'serial': device.get_serial_number() if hasattr(device, 'get_serial_number') else 'Unknown',
                        'manufacturer': device.get_manufacturer() if hasattr(device, 'get_manufacturer') else 'Unknown',
                    }
                    devices.append(device_info)
                    device.close()
                except Exception as e:
                    devices.append({
                        'index': i,
                        'name': f'Error: {e}',
                        'serial': 'Unknown',
                        'manufacturer': 'Unknown',
                    })

        except ImportError:
            print("rtlsdr не установлен")
        except Exception as e:
            print(f"Ошибка сканирования устройств: {e}")

        return devices

    def set_gain(self, gain_db: int) -> bool:
        """
        Устанавливает усиление.

        Args:
            gain_db: Усиление в dB (0-50)

        Returns:
            bool: True если успешно
        """
        if self.sdr is None:
            return False

        gain_db = max(0, min(50, gain_db))
        try:
            self.sdr.gain = gain_db
            self.gain = gain_db
            print(f"Усиление установлено: {gain_db} dB")
            return True
        except Exception as e:
            print(f"Ошибка установки усиления: {e}")
            return False

    def set_agc_mode(self, enabled: bool) -> bool:
        """
        Включает/выключает автоматическую регулировку усиления (AGC).

        Args:
            enabled: True для включения AGC

        Returns:
            bool: True если успешно
        """
        if self.sdr is None:
            return False
        try:
            self.sdr.gain = 'auto' if enabled else self.gain
            print(f"AGC {'включен' if enabled else 'выключен'}")
            return True
        except Exception as e:
            print(f"Ошибка установки AGC: {e}")
            return False

    def set_bias_tee(self, enabled: bool) -> bool:
        """
        Включает/выключает Bias-T для питания антенны.

        Args:
            enabled: True для включения Bias-T

        Returns:
            bool: True если успешно
        """
        if self.sdr is None:
            return False
        try:
            # RTL-SDR V4 и новые версии поддерживают Bias-T
            self.sdr.bias_tee = enabled
            print(f"Bias-T {'включен' if enabled else 'выключен'}")
            return True
        except Exception as e:
            print(f"Bias-T не поддерживается устройством: {e}")
            return False

    def read_samples(self, num_samples: int = 1024) -> Optional[np.ndarray]:
        """
        Читает сэмплы с SDR.

        Args:
            num_samples: Количество сэмплов

        Returns:
            np.ndarray: Сэмплы или None
        """
        if self.sdr is None:
            return None

        try:
            samples = self.sdr.read_samples(num_samples)
            return samples
        except Exception as e:
            print(f"Ошибка чтения сэмплов: {e}")
            return None

    def read_samples_batch(self, num_batches: int = 10, batch_size: int = 8192) -> Optional[np.ndarray]:
        """
        Читает пакет сэмплов с SDR (для RTL-SDR V4 с высокой скоростью).

        Args:
            num_batches: Количество пакетов
            batch_size: Размер одного пакета

        Returns:
            np.ndarray: Объединенные сэмплы или None
        """
        if self.sdr is None:
            return None

        try:
            all_samples = []
            for i in range(num_batches):
                samples = self.sdr.read_samples(batch_size)
                if samples is not None:
                    all_samples.append(samples)
            
            if all_samples:
                return np.concatenate(all_samples)
            return None
        except Exception as e:
            print(f"Ошибка чтения пакета сэмплов: {e}")
            return None

    def set_direct_sampling(self, enabled: bool, mode: int = 1) -> bool:
        """
        Включает/выключает прямой режим (direct sampling) для УКВ диапазона.

        Args:
            enabled: True для включения
            mode: 0 = выключено, 1 = I-канал, 2 = Q-канал

        Returns:
            bool: True если успешно
        """
        if self.sdr is None:
            return False
        try:
            self.sdr.direct_sampling = mode if enabled else 0
            print(f"Direct sampling: {'включен (mode=' + str(mode) + ')' if enabled else 'выключен'}")
            return True
        except Exception as e:
            print(f"Direct sampling не поддерживается: {e}")
            return False

    def set_frequency_correction(self, ppm: float) -> bool:
        """
        Устанавливает коррекцию частоты (для TCXO обычно 0).

        Args:
            ppm: Коррекция в ppm

        Returns:
            bool: True если успешно
        """
        if self.sdr is None:
            return False
        try:
            self.sdr.freq_correction = ppm
            print(f"Коррекция частоты: {ppm} ppm")
            return True
        except Exception as e:
            print(f"Ошибка установки коррекции: {e}")
            return False

    def set_manual_gain_mode(self, enabled: bool) -> bool:
        """
        Включает ручной режим усиления.

        Args:
            enabled: True для ручного режима

        Returns:
            bool: True если успешно
        """
        if self.sdr is None:
            return False
        try:
            if enabled:
                self.sdr.gain = self.gain
            else:
                self.sdr.gain = 'auto'
            print(f"Режим усиления: {'ручной' if enabled else 'авто'}")
            return True
        except Exception as e:
            print(f"Ошибка установки режима усиления: {e}")
            return False

    def start_recording(
        self,
        duration_seconds: float = 60,
        output_file: str = None
    ) -> bool:
        """
        Начинает запись сигнала.

        Args:
            duration_seconds: Длительность записи
            output_file: Файл для сохранения

        Returns:
            bool: True если успешно
        """
        if self.sdr is None:
            print("SDR не инициализирован")
            return False

        if self.is_recording:
            print("Запись уже идет")
            return False

        self.is_recording = True
        self.recorded_samples = []

        def record_thread():
            start_time = time.time()
            num_buffers = int(duration_seconds * self.sample_rate / 1024)

            print(f"Начало записи на {duration_seconds}с...")

            for i in range(num_buffers):
                if not self.is_recording:
                    break

                samples = self.read_samples(1024)
                if samples is not None:
                    self.recorded_samples.append(samples)

                    if self.callback:
                        self.callback(samples)

                time.sleep(1024 / self.sample_rate)

            self.is_recording = False
            print(f"Запись завершена. Получено сэмплов: {len(self.recorded_samples)}")

            if output_file:
                self.save_recording(output_file)

        self.recording_thread = threading.Thread(target=record_thread)
        self.recording_thread.daemon = True
        self.recording_thread.start()

        return True

    def stop_recording(self) -> List[np.ndarray]:
        """
        Останавливает запись.

        Returns:
            List[np.ndarray]: Записанные сэмплы
        """
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join(timeout=5)
        return self.recorded_samples

    def save_recording(self, output_file: str) -> bool:
        """
        Сохраняет запись в файл.

        Args:
            output_file: Путь к файлу

        Returns:
            bool: True если успешно
        """
        if not self.recorded_samples:
            print("Нет данных для сохранения")
            return False

        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Конвертируем в numpy массив
            all_samples = np.concatenate(self.recorded_samples)

            # Сохраняем как WAV
            import wave
            import struct

            # Нормализуем и конвертируем в 16-bit
            normalized = np.int16(all_samples.real / np.max(np.abs(all_samples)) * 32767)

            with wave.open(str(output_path), 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(normalized.tobytes())

            print(f"Запись сохранена: {output_file}")
            self.metadata['saved_file'] = str(output_path)
            return True

        except Exception as e:
            print(f"Ошибка сохранения записи: {e}")
            return False

    def start_frequency_scan(
        self,
        freq_range: Tuple[float, float] = (137, 146),
        step_mhz: float = 0.1,
        dwell_time_ms: int = 100
    ) -> bool:
        """
        Запускает сканирование диапазона частот.

        Args:
            freq_range: Диапазон частот (мин, макс) в МГц
            step_mhz: Шаг сканирования в МГц
            dwell_time_ms: Время на частоте в мс

        Returns:
            bool: True если успешно
        """
        if self.sdr is None:
            print("SDR не инициализирован")
            return False

        if self.is_scanning:
            print("Сканирование уже идет")
            return False

        self.is_scanning = True

        def scan_thread():
            freq_min, freq_max = freq_range
            current_freq = freq_min

            print(f"Начало сканирования: {freq_min}-{freq_max} МГц, шаг {step_mhz} МГц")

            while self.is_scanning and current_freq <= freq_max:
                self.set_frequency(current_freq)
                time.sleep(dwell_time_ms / 1000)

                # Читаем сэмплы для анализа
                samples = self.read_samples(1024)
                if samples is not None and self.callback:
                    signal_strength = np.mean(np.abs(samples))
                    self.callback({
                        'frequency': current_freq,
                        'signal_strength': signal_strength,
                        'timestamp': datetime.now().isoformat()
                    })

                current_freq += step_mhz

            self.is_scanning = False
            print("Сканирование завершено")

        self.scanning_thread = threading.Thread(target=scan_thread)
        self.scanning_thread.daemon = True
        self.scanning_thread.start()

        return True

    def stop_scanning(self) -> bool:
        """Останавливает сканирование частот."""
        self.is_scanning = False
        if self.scanning_thread:
            self.scanning_thread.join(timeout=5)
        return True

    def get_signal_strength(self) -> float:
        """
        Получает текущую силу сигнала.

        Returns:
            float: Сила сигнала (dB)
        """
        if self.sdr is None:
            return 0.0

        samples = self.read_samples(1024)
        if samples is None:
            return 0.0

        # Вычисляем среднюю мощность сигнала
        power = np.mean(np.abs(samples) ** 2)
        return 10 * np.log10(power + 1e-10)

    def get_spectrum(self, num_bins: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """
        Получает спектр сигнала.

        Args:
            num_bins: Количество бинов спектра

        Returns:
            Tuple[np.ndarray, np.ndarray]: (частоты, амплитуды)
        """
        if self.sdr is None:
            return np.array([]), np.array([])

        samples = self.read_samples(1024)
        if samples is None:
            return np.array([]), np.array([])

        # Вычисляем FFT
        fft_data = np.fft.fft(samples)
        fft_shifted = np.fft.fftshift(fft_data)
        amplitudes = np.abs(fft_shifted)

        # Вычисляем частоты
        frequencies = np.fft.fftfreq(len(samples), 1/self.sample_rate)
        frequencies = np.fft.fftshift(frequencies) + self.center_freq * 1e6

        # Уменьшаем до num_bins
        bin_size = len(frequencies) // num_bins
        freq_binned = frequencies[::bin_size][:num_bins]
        amp_binned = amplitudes[::bin_size][:num_bins]

        return freq_binned, amp_binned

    def set_callback(self, callback: Callable) -> None:
        """
        Устанавливает callback для данных.

        Args:
            callback: Функция обратного вызова
        """
        self.callback = callback

    def close(self) -> None:
        """Закрывает SDR устройство."""
        self.stop_recording()
        self.stop_scanning()

        if self.sdr is not None:
            try:
                self.sdr.close()
                print("SDR устройство закрыто")
            except Exception as e:
                print(f"Ошибка закрытия SDR: {e}")
            finally:
                self.sdr = None

    def get_status(self) -> Dict:
        """Получает текущий статус SDR."""
        return {
            'initialized': self.sdr is not None,
            'is_recording': self.is_recording,
            'is_scanning': self.is_scanning,
            'center_freq': self.center_freq,
            'sample_rate': self.sample_rate,
            'gain': self.gain,
            'device_index': self.device_index,
            'recorded_buffers': len(self.recorded_samples),
            'metadata': self.metadata
        }

    def __enter__(self):
        """Контекстный менеджер - вход."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Контекстный менеджер - выход."""
        self.close()


class SDRScanner:
    """Сканер частот для поиска SSTV сигналов."""

    def __init__(self, sdr: SDRInterface = None):
        """
        Инициализирует сканер.

        Args:
            sdr: SDR интерфейс
        """
        self.sdr = sdr or SDRInterface()
        self.found_signals: List[Dict] = []
        self.scan_results: Dict = {}

    def scan_frequencies(
        self,
        freq_range: Tuple[float, float],
        step_mhz: float = 0.05,
        threshold_db: float = -80
    ) -> List[Dict]:
        """
        Сканирует диапазон для поиска сигналов.

        Args:
            freq_range: Диапазон частот
            step_mhz: Шаг сканирования
            threshold_db: Порог обнаружения

        Returns:
            List[Dict]: Найденные сигналы
        """
        self.found_signals = []
        self.scan_results = {}

        freq_min, freq_max = freq_range
        current_freq = freq_min

        print(f"Сканирование {freq_min}-{freq_max} МГц...")

        while current_freq <= freq_max:
            self.sdr.set_frequency(current_freq)
            time.sleep(0.1)

            signal_strength = self.sdr.get_signal_strength()
            self.scan_results[current_freq] = signal_strength

            if signal_strength > threshold_db:
                signal_info = {
                    'frequency': current_freq,
                    'strength_db': signal_strength,
                    'timestamp': datetime.now().isoformat()
                }
                self.found_signals.append(signal_info)
                print(f"  Сигнал найден: {current_freq} МГц, {signal_strength:.1f} dB")

            current_freq += step_mhz

        print(f"Найдено сигналов: {len(self.found_signals)}")
        return self.found_signals

    def get_strongest_signal(self) -> Optional[Dict]:
        """Получает самый сильный сигнал."""
        if not self.found_signals:
            return None
        return max(self.found_signals, key=lambda x: x['strength_db'])

    def plot_spectrum(self, output_file: str = "spectrum.png") -> bool:
        """
        Строит график спектра.

        Args:
            output_file: Файл для сохранения графика

        Returns:
            bool: True если успешно
        """
        try:
            import matplotlib.pyplot as plt

            if not self.scan_results:
                print("Нет данных для отображения")
                return False

            frequencies = list(self.scan_results.keys())
            strengths = list(self.scan_results.values())

            plt.figure(figsize=(12, 6))
            plt.plot(frequencies, strengths, 'b-', linewidth=1)
            plt.xlabel('Частота (МГц)')
            plt.ylabel('Сила сигнала (dB)')
            plt.title('Спектр частот')
            plt.grid(True, alpha=0.3)
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Спектр сохранен: {output_file}")
            return True

        except Exception as e:
            print(f"Ошибка построения спектра: {e}")
            return False


def create_sdr(
    device_index: int = 0,
    frequency: str = 'iss',
    sample_rate: int = 2400000,
    gain: int = 30,
    bias_tee: bool = False,
    agc: bool = False
) -> Optional[SDRInterface]:
    """
    Создает и инициализирует SDR интерфейс.

    Args:
        device_index: Индекс устройства
        frequency: Предустановленная частота или значение в МГц
        sample_rate: Частота дискретизации (по умолчанию 2.4 MSPS для V4)
        gain: Усиление в dB (0-50)
        bias_tee: Включить Bias-T для питания антенны
        agc: Включить автоматическую регулировку усиления

    Returns:
        SDRInterface или None
    """
    sdr = SDRInterface(
        device_index=device_index,
        sample_rate=sample_rate,
        center_freq=145.8 if frequency not in SDRInterface.FREQUENCIES else SDRInterface.FREQUENCIES[frequency],
        gain=gain,
        device_type='auto'
    )

    # Устанавливаем частоту
    if frequency in SDRInterface.FREQUENCIES:
        sdr.center_freq = SDRInterface.FREQUENCIES[frequency]
    else:
        try:
            sdr.center_freq = float(frequency)
        except ValueError:
            print(f"Неизвестная частота: {frequency}")
            return None

    if sdr.initialize():
        # Применяем дополнительные настройки для RTL-SDR V4
        if bias_tee:
            sdr.set_bias_tee(True)
        if agc:
            sdr.set_agc_mode(True)
        return sdr

    return None


# Пример использования
if __name__ == '__main__':
    print("=== SDR интерфейс для SSTV ===")

    # Создаем SDR
    sdr = create_sdr(device_index=0, frequency='iss')

    if sdr:
        print(f"Частота: {sdr.center_freq} МГц")
        print(f"Усиление: {sdr.gain} dB")

        # Получаем силу сигнала
        strength = sdr.get_signal_strength()
        print(f"Сила сигнала: {strength:.1f} dB")

        # Записываем 10 секунд
        print("Запись 10 секунд...")
        sdr.start_recording(duration_seconds=10, output_file="recording.wav")
        time.sleep(12)

        # Закрываем
        sdr.close()
    else:
        print("Не удалось инициализировать SDR")
        print("Проверьте подключение RTL-SDR устройства")
