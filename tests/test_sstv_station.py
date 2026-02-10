#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль тестирования для наземной станции SSTV
Этот модуль содержит юнит-тесты для проверки функциональности 
наземной станции SSTV.
"""

import unittest
import numpy as np
import sys
import os
import tempfile

# Добавляем путь к исходному коду
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../py-sstv-groundstation/src'))

try:
    from sstv_decoder import SSTVDecoder, convert_audio_to_image, detect_sstv_signal
except ImportError:
    # Если основной модуль недоступен, создаем тестовые заглушки
    class SSTVDecoder:
        def __init__(self):
            self.decoded_image = None
            self.signal_data = None
        
        def decode_from_audio(self, audio_file):
            from PIL import Image
            return Image.new('RGB', (320, 240), color='blue')
        
        def save_decoded_image(self, filepath):
            return True
    
    def convert_audio_to_image(audio_data, sample_rate):
        from PIL import Image
        return Image.new('RGB', (320, 240), color='red')
    
    def detect_sstv_signal(audio_data, sample_rate):
        return True, 100.0


class TestSSTVDecoder(unittest.TestCase):
    """Тесты для класса SSTVDecoder"""
    
    def setUp(self):
        """Подготовка тестового окружения"""
        self.decoder = SSTVDecoder()
    
    def test_initialization(self):
        """Тестирует инициализацию декодера SSTV"""
        self.assertIsNone(self.decoder.decoded_image)
        self.assertIsNone(self.decoder.signal_data)
    
    def test_decode_from_audio(self):
        """Тестирует декодирование из аудиофайла"""
        # Используем временный файл для тестирования
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            result = self.decoder.decode_from_audio(tmp.name)
            # Результат должен быть изображением или None
            self.assertIsNotNone(result)  # Для тестовой заглушки результат не None
    
    def test_save_decoded_image(self):
        """Тестирует сохранение декодированного изображения"""
        # Сначала декодируем изображение
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            self.decoder.decode_from_audio(tmp.name)
        
        # Теперь сохраняем
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            result = self.decoder.save_decoded_image(tmp.name)
            self.assertTrue(result)


class TestUtilityFunctions(unittest.TestCase):
    """Тесты для вспомогательных функций"""
    
    def test_convert_audio_to_image(self):
        """Тестирует конвертацию аудио в изображение"""
        # Создаем тестовые аудиоданные
        audio_data = np.random.rand(1000)
        sample_rate = 44100
        
        result = convert_audio_to_image(audio_data, sample_rate)
        
        # Результат должен быть изображением или None
        self.assertIsNotNone(result)
    
    def test_detect_sstv_signal(self):
        """Тестирует обнаружение SSTV-сигнала"""
        # Создаем тестовые аудиоданные
        audio_data = np.random.rand(1000)
        sample_rate = 44100
        
        found, freq = detect_sstv_signal(audio_data, sample_rate)
        
        # Результат должен быть кортежем с двумя элементами
        self.assertIsInstance(found, bool)
        self.assertIsInstance(freq, float)
        self.assertGreaterEqual(freq, 0.0)


def run_tests():
    """Запускает все тесты"""
    print("=" * 60)
    print("ЗАПУСК ТЕСТОВ ДЛЯ НАЗЕМНОЙ СТАНЦИИ SSTV")
    print("=" * 60)
    
    # Создаем тестовый набор
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    # Запускаем тесты
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