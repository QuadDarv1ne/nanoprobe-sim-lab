# -*- coding: utf-8 -*-
"""
Интеграционные тесты для проекта Лаборатория моделирования нанозонда.
Тестирует взаимодействие между компонентами системы.
"""

import unittest
import sys
import os
import tempfile
import numpy as np
from pathlib import Path

# Добавляем путь к исходному коду
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from api.data_exchange import (
    DataFormatSpec,
    SurfaceDataConverter,
    ScanResultsConverter,
    ImageDataConverter,
    SSTVSignalConverter
)
from utils.database import DatabaseManager
from utils.data_manager import DataManager


class TestDataExchangeIntegration(unittest.TestCase):
    """Интеграционные тесты для обмена данными"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_surface = np.random.rand(50, 50)
        self.test_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        self.test_signal = np.random.rand(44100)

    def tearDown(self):
        """Очистка после теста"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass

    def test_surface_data_conversion_pipeline(self):
        """Тестирует полный цикл конвертации данных поверхности"""
        # Конвертируем в стандартный формат
        standard_data = SurfaceDataConverter.numpy_to_standard(self.test_surface)
        
        # Проверяем валидацию
        self.assertTrue(DataFormatSpec.validate_format(
            standard_data, 
            DataFormatSpec.FORMAT_SURFACE_DATA
        ))
        
        # Конвертируем обратно
        restored_surface = SurfaceDataConverter.standard_to_numpy(standard_data)
        
        # Проверяем целостность данных
        self.assertEqual(self.test_surface.shape, restored_surface.shape)
        np.testing.assert_array_almost_equal(
            self.test_surface, 
            restored_surface,
            decimal=10
        )

    def test_scan_results_conversion_pipeline(self):
        """Тестирует полный цикл конвертации результатов сканирования"""
        surface_id = "test_surface_001"
        
        # Конвертируем в стандартный формат
        standard_data = ScanResultsConverter.numpy_to_standard(
            self.test_surface, 
            surface_id
        )
        
        # Проверяем валидацию
        self.assertTrue(DataFormatSpec.validate_format(
            standard_data,
            DataFormatSpec.FORMAT_SCAN_RESULTS
        ))
        
        # Проверяем метаданные
        self.assertEqual(standard_data['surface_id'], surface_id)
        self.assertIn('timestamp', standard_data)
        
        # Конвертируем обратно
        restored_scan = ScanResultsConverter.standard_to_numpy(standard_data)
        
        # Проверяем целостность данных
        self.assertEqual(self.test_surface.shape, restored_scan.shape)

    def test_image_data_conversion_pipeline(self):
        """Тестирует полный цикл конвертации данных изображения"""
        # Конвертируем в стандартный формат
        standard_data = ImageDataConverter.numpy_to_standard(self.test_image)
        
        # Проверяем валидацию
        self.assertTrue(DataFormatSpec.validate_format(
            standard_data,
            DataFormatSpec.FORMAT_IMAGE_DATA
        ))
        
        # Проверяем метаданные
        self.assertEqual(standard_data['width'], 50)
        self.assertEqual(standard_data['height'], 50)
        self.assertEqual(standard_data['channels'], 3)
        
        # Конвертируем обратно
        restored_image = ImageDataConverter.standard_to_numpy(standard_data)
        
        # Проверяем целостность данных
        self.assertEqual(self.test_image.shape, restored_image.shape)

    def test_sstv_signal_conversion_pipeline(self):
        """Тестирует полный цикл конвертации SSTV сигнала"""
        sample_rate = 44100
        
        # Конвертируем в стандартный формат
        standard_data = SSTVSignalConverter.numpy_to_standard(
            self.test_signal,
            sample_rate
        )
        
        # Проверяем валидацию
        self.assertTrue(DataFormatSpec.validate_format(
            standard_data,
            DataFormatSpec.FORMAT_SSTV_SIGNAL
        ))
        
        # Проверяем метаданные
        self.assertEqual(standard_data['sample_rate'], sample_rate)
        self.assertAlmostEqual(
            standard_data['length_seconds'],
            len(self.test_signal) / sample_rate,
            places=5
        )
        
        # Конвертируем обратно
        restored_signal = SSTVSignalConverter.standard_to_numpy(standard_data)
        
        # Проверяем целостность данных
        self.assertEqual(self.test_signal.shape, restored_signal.shape)

    def test_base64_encoding_decoding(self):
        """Тестирует кодирование и декодирование base64"""
        # Кодируем поверхность
        encoded = SurfaceDataConverter.encode_base64(self.test_surface)
        
        # Декодируем обратно
        decoded = SurfaceDataConverter.decode_base64(
            encoded,
            shape=self.test_surface.shape,
            dtype=str(self.test_surface.dtype)
        )
        
        # Проверяем целостность
        np.testing.assert_array_almost_equal(
            self.test_surface,
            decoded,
            decimal=10
        )


class TestDatabaseIntegration(unittest.TestCase):
    """Интеграционные тесты для базы данных"""

    def setUp(self):
        """Подготовка тестового окружения"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_nanoprobe.db')
        self.db = DatabaseManager(db_path=self.db_path)

    def tearDown(self):
        """Очистка после теста"""
        import shutil
        try:
            self.db.close()
        except Exception:
            pass
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass

    def test_database_statistics(self):
        """Тестирует получение статистики базы данных"""
        # Добавляем несколько записей
        self.db.add_scan_result(
            scan_type='afm',
            surface_type='silicon',
            width=100,
            height=100,
            file_path='/data/scan_001.json'
        )
        self.db.add_scan_result(
            scan_type='stm',
            surface_type='gold',
            width=50,
            height=50,
            file_path='/data/scan_002.json'
        )
        
        # Получаем статистику
        stats = self.db.get_statistics()
        
        # Проверяем статистику
        self.assertIn('total_scans', stats)
        self.assertGreaterEqual(stats['total_scans'], 2)


if __name__ == '__main__':
    unittest.main()
