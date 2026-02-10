#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль форматов обмена данными для проекта Лаборатория моделирования нанозонда
Этот модуль определяет стандартные форматы для обмена данными между компонентами проекта.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path
import struct
import base64


class DataFormatSpec:
    """
    Класс спецификации форматов данных
    Определяет стандартные форматы для обмена данными между компонентами проекта.
    """
    
    # Основные форматы данных
    FORMAT_SURFACE_DATA = "surface_data_v1"
    FORMAT_SCAN_RESULTS = "scan_results_v1"
    FORMAT_IMAGE_DATA = "image_data_v1"
    FORMAT_SSTV_SIGNAL = "sstv_signal_v1"
    FORMAT_SIMULATION_CONFIG = "simulation_config_v1"
    FORMAT_ANALYTICS_REPORT = "analytics_report_v1"
    
    @staticmethod
    def validate_format(data: Dict[str, Any], format_type: str) -> bool:
        """
        Проверяет соответствие данных заданному формату
        
        Args:
            data: Данные для проверки
            format_type: Тип формата
            
        Returns:
            True если данные соответствуют формату, иначе False
        """
        required_fields = {
            DataFormatSpec.FORMAT_SURFACE_DATA: ['data', 'metadata', 'format_version'],
            DataFormatSpec.FORMAT_SCAN_RESULTS: ['scan_data', 'surface_id', 'timestamp'],
            DataFormatSpec.FORMAT_IMAGE_DATA: ['image', 'format', 'width', 'height'],
            DataFormatSpec.FORMAT_SSTV_SIGNAL: ['signal_data', 'sample_rate', 'encoding'],
            DataFormatSpec.FORMAT_SIMULATION_CONFIG: ['parameters', 'components', 'settings'],
            DataFormatSpec.FORMAT_ANALYTICS_REPORT: ['metrics', 'timestamp', 'analysis_type']
        }
        
        if format_type not in required_fields:
            return False
        
        required = required_fields[format_type]
        return all(field in data for field in required)
    
    @staticmethod
    def get_schema(format_type: str) -> Dict[str, Any]:
        """
        Возвращает схему для заданного формата
        
        Args:
            format_type: Тип формата
            
        Returns:
            Словарь с описанием схемы
        """
        schemas = {
            DataFormatSpec.FORMAT_SURFACE_DATA: {
                'type': 'object',
                'properties': {
                    'data': {
                        'type': 'array',
                        'items': {
                            'type': 'array',
                            'items': {'type': 'number'}
                        }
                    },
                    'metadata': {
                        'type': 'object',
                        'properties': {
                            'width': {'type': 'integer'},
                            'height': {'type': 'integer'},
                            'units': {'type': 'string'},
                            'created_at': {'type': 'string'}
                        }
                    },
                    'format_version': {'type': 'string'}
                }
            },
            DataFormatSpec.FORMAT_SCAN_RESULTS: {
                'type': 'object',
                'properties': {
                    'scan_data': {
                        'type': 'array',
                        'items': {
                            'type': 'array',
                            'items': {'type': 'number'}
                        }
                    },
                    'surface_id': {'type': 'string'},
                    'timestamp': {'type': 'string'},
                    'scan_parameters': {'type': 'object'}
                }
            }
        }
        
        return schemas.get(format_type, {})


class SurfaceDataConverter:
    """
    Класс для конвертации данных поверхности
    Обеспечивает преобразование данных поверхности между 
    различными форматами.
    """
    
    @staticmethod
    def numpy_to_standard(surface_array: np.ndarray) -> Dict[str, Any]:
        """
        Преобразует numpy массив поверхности в стандартный формат
        
        Args:
            surface_array: Numpy массив поверхности
            
        Returns:
            Словарь в стандартном формате
        """
        return {
            'data': surface_array.tolist(),
            'metadata': {
                'width': int(surface_array.shape[1]),
                'height': int(surface_array.shape[0]),
                'units': 'nm',
                'created_at': datetime.now().isoformat(),
                'dtype': str(surface_array.dtype)
            },
            'format_version': DataFormatSpec.FORMAT_SURFACE_DATA
        }
    
    @staticmethod
    def standard_to_numpy(surface_data: Dict[str, Any]) -> np.ndarray:
        """
        Преобразует стандартный формат в numpy массив
        
        Args:
            surface_data: Данные поверхности в стандартном формате
            
        Returns:
            Numpy массив поверхности
        """
        if not DataFormatSpec.validate_format(surface_data, DataFormatSpec.FORMAT_SURFACE_DATA):
            raise ValueError("Invalid surface data format")
        
        data = surface_data['data']
        return np.array(data)
    
    @staticmethod
    def encode_base64(surface_array: np.ndarray) -> str:
        """
        Кодирует массив поверхности в base64
        
        Args:
            surface_array: Numpy массив поверхности
            
        Returns:
            Строка в формате base64
        """
        # Преобразуем в bytes
        surface_bytes = surface_array.tobytes()
        # Кодируем в base64
        encoded = base64.b64encode(surface_bytes).decode('utf-8')
        return encoded
    
    @staticmethod
    def decode_base64(encoded_data: str, shape: tuple, dtype: str = 'float64') -> np.ndarray:
        """
        Декодирует base64 строку в массив поверхности
        
        Args:
            encoded_data: Закодированные данные в base64
            shape: Форма массива (rows, cols)
            dtype: Тип данных
            
        Returns:
            Numpy массив поверхности
        """
        # Декодируем из base64
        surface_bytes = base64.b64decode(encoded_data.encode('utf-8'))
        # Создаем numpy массив
        surface_array = np.frombuffer(surface_bytes, dtype=dtype)
        # Изменяем форму
        return surface_array.reshape(shape)


class ScanResultsConverter:
    """
    Класс для конвертации результатов сканирования
    Обеспечивает преобразование результатов сканирования между 
    различными форматами.
    """
    
    @staticmethod
    def numpy_to_standard(scan_array: np.ndarray, surface_id: str) -> Dict[str, Any]:
        """
        Преобразует numpy массив результатов сканирования в стандартный формат
        
        Args:
            scan_array: Numpy массив результатов сканирования
            surface_id: ID поверхности
            
        Returns:
            Словарь в стандартном формате
        """
        return {
            'scan_data': scan_array.tolist(),
            'surface_id': surface_id,
            'timestamp': datetime.now().isoformat(),
            'scan_parameters': {
                'resolution': scan_array.shape,
                'scan_type': 'topography'
            },
            'format_version': DataFormatSpec.FORMAT_SCAN_RESULTS
        }
    
    @staticmethod
    def standard_to_numpy(scan_data: Dict[str, Any]) -> np.ndarray:
        """
        Преобразует стандартный формат результатов сканирования в numpy массив
        
        Args:
            scan_data: Данные сканирования в стандартном формате
            
        Returns:
            Numpy массив результатов сканирования
        """
        if not DataFormatSpec.validate_format(scan_data, DataFormatSpec.FORMAT_SCAN_RESULTS):
            raise ValueError("Invalid scan results format")
        
        data = scan_data['scan_data']
        return np.array(data)


class ImageDataConverter:
    """
    Класс для конвертации данных изображений
    Обеспечивает преобразование данных изображений между 
    различными форматами.
    """
    
    @staticmethod
    def numpy_to_standard(image_array: np.ndarray) -> Dict[str, Any]:
        """
        Преобразует numpy массив изображения в стандартный формат
        
        Args:
            image_array: Numpy массив изображения
            
        Returns:
            Словарь в стандартном формате
        """
        return {
            'image': image_array.tolist(),
            'format': 'numpy_array',
            'width': int(image_array.shape[1]),
            'height': int(image_array.shape[0]),
            'channels': int(image_array.shape[2]) if len(image_array.shape) > 2 else 1,
            'dtype': str(image_array.dtype),
            'timestamp': datetime.now().isoformat(),
            'format_version': DataFormatSpec.FORMAT_IMAGE_DATA
        }
    
    @staticmethod
    def standard_to_numpy(image_data: Dict[str, Any]) -> np.ndarray:
        """
        Преобразует стандартный формат изображения в numpy массив
        
        Args:
            image_data: Данные изображения в стандартном формате
            
        Returns:
            Numpy массив изображения
        """
        if not DataFormatSpec.validate_format(image_data, DataFormatSpec.FORMAT_IMAGE_DATA):
            raise ValueError("Invalid image data format")
        
        return np.array(image_data['image'])


class SSTVSignalConverter:
    """
    Класс для конвертации данных SSTV сигналов
    Обеспечивает преобразование данных SSTV сигналов между 
    различными форматами.
    """
    
    @staticmethod
    def numpy_to_standard(signal_array: np.ndarray, sample_rate: int = 44100) -> Dict[str, Any]:
        """
        Преобразует numpy массив сигнала в стандартный формат
        
        Args:
            signal_array: Numpy массив аудиосигнала
            sample_rate: Частота дискретизации
            
        Returns:
            Словарь в стандартном формате
        """
        return {
            'signal_data': signal_array.tolist(),
            'sample_rate': sample_rate,
            'encoding': 'pcm_f32',
            'length_seconds': len(signal_array) / sample_rate,
            'timestamp': datetime.now().isoformat(),
            'format_version': DataFormatSpec.FORMAT_SSTV_SIGNAL
        }
    
    @staticmethod
    def standard_to_numpy(signal_data: Dict[str, Any]) -> np.ndarray:
        """
        Преобразует стандартный формат сигнала в numpy массив
        
        Args:
            signal_data: Данные сигнала в стандартном формате
            
        Returns:
            Numpy массив аудиосигнала
        """
        if not DataFormatSpec.validate_format(signal_data, DataFormatSpec.FORMAT_SSTV_SIGNAL):
            raise ValueError("Invalid SSTV signal format")
        
        return np.array(signal_data['signal_data'])


class SimulationConfigConverter:
    """
    Класс для конвертации конфигурации симуляции
    Обеспечивает преобразование конфигурации симуляции между 
    различными форматами.
    """
    
    @staticmethod
    def dict_to_standard(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Преобразует словарь конфигурации в стандартный формат
        
        Args:
            config_dict: Словарь с параметрами конфигурации
            
        Returns:
            Словарь в стандартном формате
        """
        return {
            'parameters': config_dict,
            'components': ['spm', 'image_processor', 'sstv_decoder'],
            'settings': {
                'precision': 'high',
                'real_time': False,
                'output_format': 'standard'
            },
            'timestamp': datetime.now().isoformat(),
            'format_version': DataFormatSpec.FORMAT_SIMULATION_CONFIG
        }
    
    @staticmethod
    def standard_to_dict(config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Преобразует стандартный формат конфигурации в словарь
        
        Args:
            config_data: Данные конфигурации в стандартном формате
            
        Returns:
            Словарь с параметрами конфигурации
        """
        if not DataFormatSpec.validate_format(config_data, DataFormatSpec.FORMAT_SIMULATION_CONFIG):
            raise ValueError("Invalid simulation config format")
        
        return config_data['parameters']


class AnalyticsReportConverter:
    """
    Класс для конвертации аналитических отчетов
    Обеспечивает преобразование аналитических отчетов между 
    различными форматами.
    """
    
    @staticmethod
    def dict_to_standard(metrics_dict: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
        """
        Преобразует словарь метрик в стандартный формат отчета
        
        Args:
            metrics_dict: Словарь с метриками
            analysis_type: Тип анализа
            
        Returns:
            Словарь в стандартном формате
        """
        return {
            'metrics': metrics_dict,
            'analysis_type': analysis_type,
            'timestamp': datetime.now().isoformat(),
            'report_metadata': {
                'generator': 'nanoprobe-analytics',
                'version': '1.0'
            },
            'format_version': DataFormatSpec.FORMAT_ANALYTICS_REPORT
        }
    
    @staticmethod
    def standard_to_dict(report_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Преобразует стандартный формат отчета в словарь метрик
        
        Args:
            report_data: Данные отчета в стандартном формате
            
        Returns:
            Словарь с метриками
        """
        if not DataFormatSpec.validate_format(report_data, DataFormatSpec.FORMAT_ANALYTICS_REPORT):
            raise ValueError("Invalid analytics report format")
        
        return report_data['metrics']


class DataExchangeManager:
    """
    Класс менеджера обмена данными
    Обеспечивает централизованное управление конвертацией данных 
    между различными форматами.
    """
    
    def __init__(self):
        """Инициализирует менеджер обмена данными"""
        self.converters = {
            DataFormatSpec.FORMAT_SURFACE_DATA: SurfaceDataConverter,
            DataFormatSpec.FORMAT_SCAN_RESULTS: ScanResultsConverter,
            DataFormatSpec.FORMAT_IMAGE_DATA: ImageDataConverter,
            DataFormatSpec.FORMAT_SSTV_SIGNAL: SSTVSignalConverter,
            DataFormatSpec.FORMAT_SIMULATION_CONFIG: SimulationConfigConverter,
            DataFormatSpec.FORMAT_ANALYTICS_REPORT: AnalyticsReportConverter
        }
    
    def convert(self, data: Any, from_format: str, to_format: str) -> Any:
        """
        Конвертирует данные из одного формата в другой
        
        Args:
            data: Входные данные
            from_format: Исходный формат
            to_format: Целевой формат
            
        Returns:
            Сконвертированные данные
        """
        if from_format == to_format:
            return data
        
        # Сначала преобразуем в стандартный формат
        if from_format in self.converters:
            # Если from_format не стандартный, преобразуем в стандартный
            if not DataFormatSpec.validate_format(data, from_format):
                # Предполагаем, что данные в формате, который может обработать конвертер
                pass
            
            # Для простоты будем считать, что мы конвертируем через numpy
            # В реальном приложении здесь должна быть более сложная логика
            
            if from_format == DataFormatSpec.FORMAT_SURFACE_DATA:
                numpy_data = self.converters[from_format].standard_to_numpy(data)
            elif from_format == DataFormatSpec.FORMAT_SCAN_RESULTS:
                numpy_data = self.converters[from_format].standard_to_numpy(data)
            elif from_format == DataFormatSpec.FORMAT_IMAGE_DATA:
                numpy_data = self.converters[from_format].standard_to_numpy(data)
            elif from_format == DataFormatSpec.FORMAT_SSTV_SIGNAL:
                numpy_data = self.converters[from_format].standard_to_numpy(data)
            else:
                # Для других форматов возвращаем как есть
                return data
        else:
            # Если формат неизвестен, возвращаем как есть
            return data
        
        # Затем преобразуем из стандартного в целевой формат
        if to_format == DataFormatSpec.FORMAT_SURFACE_DATA:
            return self.converters[to_format].numpy_to_standard(numpy_data)
        elif to_format == DataFormatSpec.FORMAT_SCAN_RESULTS:
            # Для результатов сканирования нужен дополнительный параметр
            surface_id = getattr(data, 'surface_id', 'unknown')
            return self.converters[to_format].numpy_to_standard(numpy_data, surface_id)
        elif to_format == DataFormatSpec.FORMAT_IMAGE_DATA:
            return self.converters[to_format].numpy_to_standard(numpy_data)
        elif to_format == DataFormatSpec.FORMAT_SSTV_SIGNAL:
            return self.converters[to_format].numpy_to_standard(numpy_data)
        else:
            # Если целевой формат неизвестен, возвращаем numpy данные
            return numpy_data
    
    def validate(self, data: Any, format_type: str) -> bool:
        """
        Проверяет данные на соответствие формату
        
        Args:
            data: Данные для проверки
            format_type: Тип формата
            
        Returns:
            True если данные соответствуют формату, иначе False
        """
        return DataFormatSpec.validate_format(data, format_type)
    
    def get_supported_formats(self) -> List[str]:
        """
        Возвращает список поддерживаемых форматов
        
        Returns:
            Список поддерживаемых форматов
        """
        return list(self.converters.keys())


def main():
    """Главная функция для демонстрации возможностей модуля обмена данными"""
    print("=== МОДУЛЬ ОБМЕНА ДАННЫМИ ПРОЕКТА ===")
    
    # Создаем менеджер обмена данными
    data_manager = DataExchangeManager()
    
    print("Поддерживаемые форматы:")
    for fmt in data_manager.get_supported_formats():
        print(f"  - {fmt}")
    
    # Создаем тестовые данные
    test_surface = np.random.rand(10, 10)
    
    # Конвертируем в стандартный формат
    surface_standard = SurfaceDataConverter.numpy_to_standard(test_surface)
    print(f"✓ Поверхность сконвертирована в стандартный формат")
    
    # Проверяем формат
    is_valid = DataFormatSpec.validate_format(surface_standard, DataFormatSpec.FORMAT_SURFACE_DATA)
    print(f"✓ Формат действителен: {is_valid}")
    
    # Конвертируем обратно
    surface_back = SurfaceDataConverter.standard_to_numpy(surface_standard)
    print(f"✓ Поверхность восстановлена из стандартного формата")
    
    # Проверяем, что данные совпадают
    arrays_equal = np.array_equal(test_surface, surface_back)
    print(f"✓ Данные совпадают после конвертации: {arrays_equal}")
    
    # Тестируем кодирование в base64
    encoded = SurfaceDataConverter.encode_base64(test_surface)
    decoded = SurfaceDataConverter.decode_base64(encoded, test_surface.shape)
    base64_equal = np.array_equal(test_surface, decoded)
    print(f"✓ Данные совпадают после base64 кодирования/декодирования: {base64_equal}")
    
    print("Модуль обмена данными успешно протестирован")


if __name__ == "__main__":
    main()