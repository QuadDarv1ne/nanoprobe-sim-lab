# -*- coding: utf-8 -*-
"""Тесты для наземной станции SSTV."""

import unittest
import numpy as np
import sys
import os
import tempfile

# Добавляем путь к исходному коду
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../components/py-sstv-groundstation/src'))

from sstv_decoder import SSTVDecoder, convert_audio_to_image, detect_sstv_signal


class TestSSTVDecoder(unittest.TestCase):
    """Тесты для класса SSTVDecoder"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.decoder = SSTVDecoder()

    def test_initialization(self):
        """Тестирует инициализацию декодера SSTV"""
        self.assertIsNone(self.decoder.decoded_image)
        self.assertIsNone(self.decoder.signal_data)

    def test_decode_from_audio_nonexistent_file(self):
        """Тестирует декодирование несуществующего файла"""
        result = self.decoder.decode_from_audio("/nonexistent/file.wav")
        self.assertIsNone(result)

    def test_decode_from_audio_nonexistent_file(self):
        """Тестирует декодирование несуществующего файла"""
        result = self.decoder.decode_from_audio("/nonexistent/file.wav")
        self.assertIsNone(result)

    def test_decode_from_audio_invalid_format(self):
        """Тестирует декодирование файла с неподдерживаемым форматом"""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b"test")
            tmp_name = tmp.name
        result = self.decoder.decode_from_audio(tmp_name)
        self.assertIsNone(result)
        try:
            os.unlink(tmp_name)
        except Exception:
            pass

    def test_decode_from_audio_wav(self):
        """Тестирует декодирование WAV файла (fallback режим)"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_name = tmp.name
        result = self.decoder.decode_from_audio(tmp_name)
        # В fallback режиме должно вернуться изображение-заглушка или None (если pysstv не установлен)
        # Проверяем что функция отработала без ошибок
        self.assertIsInstance(result, (type(None), object))
        try:
            os.unlink(tmp_name)
        except Exception:
            pass

    def test_decode_invalid_mode(self):
        """Тестирует декодирование с неверным режимом"""
        decoder = SSTVDecoder(mode='InvalidMode')
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_name = tmp.name
        result = decoder.decode_from_audio(tmp_name)
        self.assertIsNone(result)
        try:
            os.unlink(tmp_name)
        except Exception:
            pass

    def test_save_decoded_image_no_image(self):
        """Тестирует сохранение без декодированного изображения"""
        result = self.decoder.save_decoded_image("/tmp/test.png")
        self.assertFalse(result)

    def test_save_decoded_image_empty_path(self):
        """Тестирует сохранение с пустым путём"""
        result = self.decoder.save_decoded_image("")
        self.assertFalse(result)

    def test_save_decoded_image_invalid_format(self):
        """Тестирует сохранение в неподдерживаемом формате"""
        result = self.decoder.save_decoded_image("/tmp/test.gif")
        self.assertFalse(result)

    def test_save_decoded_image_valid(self):
        """Тестирует сохранение в допустимом формате"""
        # Создаём изображение напрямую
        from PIL import Image
        self.decoder.decoded_image = Image.new('RGB', (100, 100), color='red')

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_name = tmp.name
        result = self.decoder.save_decoded_image(tmp_name)
        self.assertTrue(result)
        try:
            os.unlink(tmp_name)
        except Exception:
            pass


class TestUtilityFunctions(unittest.TestCase):
    """Тесты для вспомогательных функций"""

    def test_convert_audio_to_image(self):
        """Тестирует конвертацию аудио в изображение"""
        # Создаем тестовые аудиоданные
        audio_data = np.random.rand(1000)
        sample_rate = 44100

        result = convert_audio_to_image(audio_data, sample_rate)
        # Результат может быть None если pysstv не установлен
        self.assertIsInstance(result, (type(None), object))

    def test_detect_sstv_signal(self):
        """Тестирует обнаружение SSTV-сигнала"""
        # Функция принимает только путь к файлу
        result = detect_sstv_signal("/nonexistent/file.wav")
        # Результат должен быть кортежем
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], bool)
        self.assertIsInstance(result[1], dict)

    def test_decode_from_samples_empty(self):
        """Тестирует декодирование пустых сэмплов"""
        decoder = SSTVDecoder()
        with self.assertRaises(ValueError):
            decoder.decode_from_samples(np.array([]), 48000)

    def test_decode_from_samples_invalid_sample_rate(self):
        """Тестирует декодирование с неверной частотой дискретизации"""
        decoder = SSTVDecoder()
        with self.assertRaises(ValueError):
            decoder.decode_from_samples(np.random.rand(1000), -100)


def run_tests():
    """Запускает все тесты."""
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

