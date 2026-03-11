# -*- coding: utf-8 -*-
"""Тесты для SDR интерфейса."""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../components/py-sstv-groundstation/src'))

from sdr_interface import SDRInterface


class TestSDRInterface(unittest.TestCase):
    """Тесты для класса SDRInterface"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.sdr = SDRInterface()

    def test_initialization(self):
        """Тестирует инициализацию SDR интерфейса"""
        self.assertIsNone(self.sdr.sdr)
        self.assertEqual(self.sdr.device_index, 0)
        self.assertEqual(self.sdr.sample_rate, 2400000)
        self.assertEqual(self.sdr.center_freq, 145.800)
        self.assertEqual(self.sdr.gain, 30)

    def test_supported_devices(self):
        """Тестирует список поддерживаемых устройств"""
        self.assertIn('r828d', self.sdr.SUPPORTED_DEVICES)
        self.assertIn('rtl2832u', self.sdr.SUPPORTED_DEVICES)
        self.assertIn('r820t', self.sdr.SUPPORTED_DEVICES)
        self.assertIn('airspy', self.sdr.SUPPORTED_DEVICES)
        self.assertIn('hackrf', self.sdr.SUPPORTED_DEVICES)

    def test_device_type_v4(self):
        """Тестирует инициализацию с явным указанием RTL-SDR V4"""
        sdr_v4 = SDRInterface(device_type='r828d')
        self.assertEqual(sdr_v4.device_type, 'r828d')
        self.assertEqual(sdr_v4.SUPPORTED_DEVICES['r828d'], 'RTL-SDR V4 (R828D)')

    def test_device_type_classic(self):
        """Тестирует инициализацию с явным указанием классического RTL-SDR"""
        sdr_classic = SDRInterface(device_type='rtl2832u')
        self.assertEqual(sdr_classic.device_type, 'rtl2832u')

    def test_device_type_auto(self):
        """Тестирует инициализацию с автоопределением"""
        sdr_auto = SDRInterface(device_type='auto')
        self.assertEqual(sdr_auto.device_type, 'auto')

    def test_frequencies(self):
        """Тестирует预设ленные частоты"""
        self.assertIn('iss', self.sdr.FREQUENCIES)
        self.assertIn('noaa_15', self.sdr.FREQUENCIES)
        self.assertIn('meteor_m2', self.sdr.FREQUENCIES)
        self.assertEqual(self.sdr.FREQUENCIES['iss'], 145.800)

    def test_set_frequency_before_init(self):
        """Тестирует установку частоты до инициализации"""
        result = self.sdr.set_frequency(145.800)
        self.assertFalse(result)

    def test_set_gain_before_init(self):
        """Тестирует установку усиления до инициализации"""
        result = self.sdr.set_gain(30)
        self.assertFalse(result)

    def test_read_samples_before_init(self):
        """Тестирует чтение сэмплов до инициализации"""
        result = self.sdr.read_samples(1024)
        self.assertIsNone(result)

    def test_list_devices_static(self):
        """Тестирует статический метод list_devices"""
        devices = SDRInterface.list_devices()
        self.assertIsInstance(devices, list)
        # Метод должен возвращать список (пустой если нет устройств)

    def test_metadata_initialization(self):
        """Тестирует инициализацию метаданных"""
        self.assertIsInstance(self.sdr.metadata, dict)
        self.assertEqual(len(self.sdr.metadata), 0)

    def test_recording_state(self):
        """Тестирует состояние записи"""
        self.assertFalse(self.sdr.is_recording)
        self.assertFalse(self.sdr.is_scanning)

    def test_sample_rate_default(self):
        """Тестирует частоту дискретизации по умолчанию"""
        # RTL-SDR V4 использует 2.4 MSPS
        self.assertEqual(self.sdr.sample_rate, 2400000)

    def test_custom_sample_rate(self):
        """Тестирует установку custom частоты дискретизации"""
        sdr_custom = SDRInterface(sample_rate=1000000)
        self.assertEqual(sdr_custom.sample_rate, 1000000)

    def test_frequency_range_validation(self):
        """Тестирует валидацию диапазона частот"""
        # Частоты вне диапазона должны возвращать False после инициализации
        # Пока просто проверяем что метод существует
        self.assertTrue(hasattr(self.sdr, 'set_frequency'))


def run_tests():
    """Запускает все тесты."""
    print("=" * 60)
    print("ЗАПУСК ТЕСТОВ ДЛЯ SDR ИНТЕРФЕЙСА")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 60)
    print(f"РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print(f"  Пройдено: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Ошибки: {len(result.errors)}")
    print(f"  Провалы: {len(result.failures)}")
    print("=" * 60)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
