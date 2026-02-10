#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль тестирования для симулятора сканирующего зондового микроскопа (СЗМ)
Этот модуль содержит юнит-тесты для проверки функциональности симулятора СЗМ.
"""

import unittest
import numpy as np
import sys
import os

# Добавляем путь к исходному коду
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../cpp-spm-hardware-sim/src'))

from spm_simulator import SurfaceModel, ProbeModel, SPMController


class TestSurfaceModel(unittest.TestCase):
    """Тесты для класса SurfaceModel"""
    
    def setUp(self):
        """Подготовка тестового окружения"""
        self.surface = SurfaceModel(10, 10)
    
    def test_initialization(self):
        """Тестирует инициализацию модели поверхности"""
        self.assertEqual(self.surface.width, 10)
        self.assertEqual(self.surface.height, 10)
        self.assertEqual(self.surface.height_map.shape, (10, 10))
    
    def test_get_height_within_bounds(self):
        """Тестирует получение высоты в пределах границ"""
        height = self.surface.get_height(5, 5)
        self.assertIsInstance(height, float)
        self.assertGreaterEqual(height, -1.0)  # Значение должно быть в разумном диапазоне
    
    def test_get_height_out_of_bounds(self):
        """Тестирует получение высоты за пределами границ"""
        height = self.surface.get_height(-1, -1)
        self.assertEqual(height, 0.0)
        
        height = self.surface.get_height(100, 100)
        self.assertEqual(height, 0.0)
    
    def test_save_to_file(self):
        """Тестирует сохранение модели поверхности в файл"""
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp:
            result = self.surface.save_to_file(tmp.name)
            self.assertTrue(result)


class TestProbeModel(unittest.TestCase):
    """Тесты для класса ProbeModel"""
    
    def setUp(self):
        """Подготовка тестового окружения"""
        self.probe = ProbeModel()
    
    def test_initial_position(self):
        """Тестирует начальную позицию зонда"""
        pos = self.probe.get_position()
        self.assertEqual(pos[0], 0)
        self.assertEqual(pos[1], 0)
        self.assertAlmostEqual(pos[2], 10.0, places=1)
    
    def test_set_position(self):
        """Тестирует установку позиции зонда"""
        self.probe.set_position(5.0, 6.0, 15.0)
        pos = self.probe.get_position()
        self.assertEqual(pos[0], 5.0)
        self.assertEqual(pos[1], 6.0)
        self.assertEqual(pos[2], 15.0)
    
    def test_move_to(self):
        """Тестирует перемещение зонда"""
        self.probe.move_to(3.0, 4.0, 12.0)
        pos = self.probe.get_position()
        self.assertEqual(pos[0], 3.0)
        self.assertEqual(pos[1], 4.0)
        self.assertEqual(pos[2], 12.0)
    
    def test_adjust_to_surface(self):
        """Тестирует адаптацию зонда к поверхности"""
        surface = SurfaceModel(5, 5)
        initial_z = self.probe.z
        adjusted_z = self.probe.adjust_to_surface(surface)
        
        # Адаптированная высота должна быть близка к высоте поверхности + 0.5
        expected_height = surface.get_height(0, 0) + 0.5
        self.assertAlmostEqual(adjusted_z, expected_height, places=1)


class TestSPMController(unittest.TestCase):
    """Тесты для класса SPMController"""
    
    def setUp(self):
        """Подготовка тестового окружения"""
        self.controller = SPMController()
        self.surface = SurfaceModel(5, 5)
        self.controller.set_surface(self.surface)
    
    def test_set_surface(self):
        """Тестирует установку поверхности"""
        self.assertIsNotNone(self.controller.surface)
        self.assertEqual(self.controller.surface.width, 5)
        self.assertEqual(self.controller.surface.height, 5)
        self.assertIsNotNone(self.controller.scan_data)
        self.assertEqual(self.controller.scan_data.shape, (5, 5))
    
    def test_scan_surface(self):
        """Тестирует процесс сканирования поверхности"""
        self.controller.scan_surface()
        
        # После сканирования данные должны быть заполнены
        self.assertFalse(np.all(self.controller.scan_data == 0))
        
        # Все значения должны быть в разумном диапазоне
        self.assertTrue(np.all(self.controller.scan_data >= 0.1))
        self.assertTrue(np.all(self.controller.scan_data <= 20.0))


def run_tests():
    """Запускает все тесты"""
    print("=" * 60)
    print("ЗАПУСК ТЕСТОВ ДЛЯ СИМУЛЯТОРА СЗМ")
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