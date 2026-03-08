# -*- coding: utf-8 -*-
"""
Интерфейс для работы с RTL-SDR
Модуль для приема радиосигналов через RTL-SDR V4 и другие SDR-устройства
"""

import numpy as np
from typing import Optional, Tuple, List
import wave
import os
from pathlib import Path


class SDRInterface:
    """Класс для взаимодействия с RTL-SDR устройством."""

    def __init__(self, device_index: int = 0):
        """
        Инициализирует интерфейс RTL-SDR

        Args:
            device_index: Индекс устройства (0 по умолчанию)
        """
        self.device_index = device_index
        self.sdr = None
        self.sample_rate = 1.024e6  # 1.024 MSPS
        self.center_frequency = 145.800e6  # Частота МКС по умолчанию
        self.gain = 30  # Усиление в дБ
        self.is_initialized = False

    def initialize(self) -> bool:
        """
        Инициализирует RTL-SDR устройство

        Returns:
            bool: True если успешно, иначе False
        """
        try:
            from rtlsdr import RtlSdr

            print(f"Инициализация RTL-SDR устройства (индекс: {self.device_index})...")
            self.sdr = RtlSdr(device_index=self.device_index)
            self.is_initialized = True

            # Настройка параметров
            self.sdr.set_sample_rate(self.sample_rate)
            self.sdr.set_center_freq(self.center_frequency)
            self.sdr.set_gain(self.gain)

            print(f"✓ RTL-SDR инициализирован")
            print(f"  Частота: {self.center_frequency / 1e6:.3f} МГц")
            print(f"  Частота дискретизации: {self.sample_rate / 1e6:.3f} MSPS")
            print(f"  Усиление: {self.gain} дБ")

            return True

        except ImportError:
            print("❌ Ошибка: Библиотека rtlsdr не установлена")
            print("   Установите: pip install rtlsdr pyrtlsdr")
            return False

        except Exception as e:
            print(f"❌ Ошибка инициализации RTL-SDR: {str(e)}")
            print("   Убедитесь, что устройство подключено и драйверы установлены")
            return False

    def set_frequency(self, frequency_mhz: float) -> bool:
        """
        Устанавливает частоту приема

        Args:
            frequency_mhz: Частота в МГц

        Returns:
            bool: True если успешно
        """
        if not self.is_initialized:
            print("Сначала инициализируйте устройство")
            return False

        try:
            self.center_frequency = frequency_mhz * 1e6
            self.sdr.set_center_freq(self.center_frequency)
            print(f"Частота установлена: {frequency_mhz:.3f} МГц")
            return True
        except Exception as e:
            print(f"Ошибка установки частоты: {str(e)}")
            return False

    def set_gain(self, gain_db: int) -> bool:
        """
        Устанавливает усиление

        Args:
            gain_db: Усиление в дБ (0-50)

        Returns:
            bool: True если успешно
        """
        if not self.is_initialized:
            print("Сначала инициализируйте устройство")
            return False

        try:
            gain_db = max(0, min(50, gain_db))  # Ограничение 0-50 дБ
            self.gain = gain_db
            self.sdr.set_gain(self.gain)
            print(f"Усиление установлено: {self.gain} дБ")
            return True
        except Exception as e:
            print(f"Ошибка установки усиления: {str(e)}")
            return False

    def set_sample_rate(self, sample_rate_mhz: float) -> bool:
        """
        Устанавливает частоту дискретизации

        Args:
            sample_rate_mhz: Частота дискретизации в MSPS

        Returns:
            bool: True если успешно
        """
        if not self.is_initialized:
            print("Сначала инициализируйте устройство")
            return False

        try:
            self.sample_rate = sample_rate_mhz * 1e6
            self.sdr.set_sample_rate(self.sample_rate)
            print(f"Частота дискретизации: {sample_rate_mhz:.3f} MSPS")
            return True
        except Exception as e:
            print(f"Ошибка установки частоты дискретизации: {str(e)}")
            return False

    def read_samples(self, num_samples: int = 1024) -> Optional[np.ndarray]:
        """
        Читает образцы с устройства

        Args:
            num_samples: Количество образцов для чтения

        Returns:
            np.ndarray: Массив образцов или None при ошибке
        """
        if not self.is_initialized:
            print("Сначала инициализируйте устройство")
            return None

        try:
            samples = self.sdr.read_samples(num_samples)
            return samples
        except Exception as e:
            print(f"Ошибка чтения образцов: {str(e)}")
            return None

    def record_audio(
        self,
        duration_sec: float = 30.0,
        output_file: str = "sstv_recording.wav",
        progress_callback=None,
    ) -> bool:
        """
        Записывает аудио сигнал с RTL-SDR и сохраняет в WAV файл

        Args:
            duration_sec: Длительность записи в секундах
            output_file: Путь для сохранения WAV файла
            progress_callback: Функция обратного вызова для прогресса

        Returns:
            bool: True если запись успешна
        """
        if not self.is_initialized:
            print("Сначала инициализируйте устройство")
            return False

        try:
            print(f"Начало записи ({duration_sec} сек)...")
            print(f"Частота: {self.center_frequency / 1e6:.3f} МГц")
            print(f"Нажмите Ctrl+C для остановки")

            # Параметры аудио
            audio_sample_rate = 48000  # 48 kHz для аудио
            num_channels = 1  # Моно
            sample_width = 2  # 16 бит

            # Расчет количества образцов
            total_samples = int(self.sample_rate * duration_sec)
            chunk_size = int(self.sample_rate * 0.1)  # 100 мс чанки

            audio_samples = []
            samples_read = 0

            while samples_read < total_samples:
                # Чтение образцов с RTL-SDR
                samples = self.read_samples(min(chunk_size, total_samples - samples_read))
                if samples is None:
                    break

                # Конвертация IQ в аудио (FM демодуляция)
                audio_data = self._fm_demodulate(samples, audio_sample_rate)
                audio_samples.append(audio_data)

                samples_read += len(samples)

                # Обновление прогресса
                if progress_callback:
                    progress = (samples_read / total_samples) * 100
                    progress_callback(progress)

                # Проверка на прерывание
                if samples_read % (chunk_size * 10) == 0:
                    elapsed = samples_read / self.sample_rate
                    print(f"Записано: {elapsed:.1f} сек из {duration_sec} сек")

            if not audio_samples:
                print("Нет данных для записи")
                return False

            # Конкатенация и сохранение
            audio_data = np.concatenate(audio_samples)
            audio_data = np.int16(audio_data * 32767)  # Конвертация в 16-bit

            # Сохранение в WAV
            with wave.open(output_file, "w") as wav_file:
                wav_file.setnchannels(num_channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(audio_sample_rate)
                wav_file.writeframes(audio_data.tobytes())

            print(f"✓ Запись сохранена: {output_file}")
            print(f"  Длительность: {len(audio_data) / audio_sample_rate:.1f} сек")
            return True

        except KeyboardInterrupt:
            print("\nЗапись прервана пользователем")
            return False

        except Exception as e:
            print(f"Ошибка записи: {str(e)}")
            return False

    def _fm_demodulate(self, iq_samples: np.ndarray, target_sample_rate: float) -> np.ndarray:
        """
        Выполняет FM демодуляцию IQ образцов в аудио

        Args:
            iq_samples: IQ образцы с RTL-SDR
            target_sample_rate: Целевая частота дискретизации аудио

        Returns:
            np.ndarray: Аудиоданные
        """
        # FM демодуляция через разность фаз
        phase = np.angle(iq_samples)
        audio = np.diff(phase)

        # Нормализация
        audio = audio / np.max(np.abs(audio)) if len(audio) > 0 else np.array([])

        # Децимация до целевой частоты дискретизации
        decimation_factor = int(self.sample_rate / target_sample_rate)
        if decimation_factor > 1 and len(audio) > decimation_factor:
            audio = audio[::decimation_factor]

        return audio

    def scan_frequencies(
        self, start_mhz: float = 145.0, end_mhz: float = 146.0, step_mhz: float = 0.001
    ) -> List[Tuple[float, float]]:
        """
        Сканирует диапазон частот для обнаружения сигналов

        Args:
            start_mhz: Начало диапазона в МГц
            end_mhz: Конец диапазона в МГц
            step_mhz: Шаг сканирования в МГц

        Returns:
            List[Tuple[float, float]]: Список (частота, мощность) обнаруженных сигналов
        """
        if not self.is_initialized:
            print("Сначала инициализируйте устройство")
            return []

        signals = []
        current_freq = start_mhz

        print(f"Сканирование диапазона {start_mhz:.3f} - {end_mhz:.3f} МГц...")

        while current_freq <= end_mhz:
            self.set_frequency(current_freq)

            # Чтение образцов для измерения мощности
            samples = self.read_samples(1024)
            if samples is not None:
                power = np.mean(np.abs(samples) ** 2)
                power_db = 10 * np.log10(power + 1e-10)

                # Порог обнаружения сигнала
                if power_db > -50:  # Порог в дБ
                    signals.append((current_freq, power_db))
                    print(f"  Сигнал обнаружен: {current_freq:.3f} МГц ({power_db:.1f} дБ)")

            current_freq += step_mhz

        return signals

    def close(self):
        """Закрывает соединение с RTL-SDR."""
        if self.sdr:
            try:
                self.sdr.close()
                print("RTL-SDR устройство закрыто")
            except Exception as e:
                print(f"Ошибка закрытия устройства: {str(e)}")
        self.is_initialized = False

    def __enter__(self):
        """Контекстный менеджер: вход."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Контекстный менеджер: выход."""
        self.close()


def list_devices() -> List[dict]:
    """
    Выводит список доступных RTL-SDR устройств

    Returns:
        List[dict]: Список информации об устройствах
    """
    try:
        from rtlsdr import RtlSdr

        num_devices = RtlSdr.get_device_count()
        devices = []

        print(f"Найдено RTL-SDR устройств: {num_devices}")

        for i in range(num_devices):
            device_info = RtlSdr.get_device_strings(i)
            device = {
                "index": i,
                "manufacturer": device_info.get("manufacturer", "Unknown"),
                "product": device_info.get("product", "Unknown"),
                "serial": device_info.get("serial", "Unknown"),
            }
            devices.append(device)
            print(f"  Устройство {i}: {device['manufacturer']} {device['product']}")

        return devices

    except ImportError:
        print("Библиотека rtlsdr не установлена")
        return []
    except Exception as e:
        print(f"Ошибка получения списка устройств: {str(e)}")
        return []
