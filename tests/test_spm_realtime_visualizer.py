"""Тесты для Real-time СЗМ визуализатора."""

import unittest
import numpy as np
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils.spm_realtime_visualizer import (
        StreamingDataBuffer,
        RealTimeSPMWebSocketAdapter,
    )
    VISUALIZER_AVAILABLE = True
except ImportError:
    VISUALIZER_AVAILABLE = False


@unittest.skipUnless(VISUALIZER_AVAILABLE, "Visualizer module required")
class TestStreamingDataBuffer(unittest.TestCase):
    """Тесты для буфера потоковых данных"""

    def setUp(self):
        """Подготовка"""
        self.buffer = StreamingDataBuffer(max_size=100)

    def test_initialization(self):
        """Тест инициализации"""
        self.assertEqual(self.buffer.max_size, 100)
        self.assertEqual(len(self.buffer.buffer), 0)
        self.assertEqual(self.buffer.total_items_added, 0)

    def test_add_frame(self):
        """Тест добавления кадра"""
        frame = np.random.rand(64, 64)
        self.buffer.add_frame(frame)

        self.assertEqual(self.buffer.total_items_added, 1)
        self.assertEqual(len(self.buffer.buffer), 1)

    def test_add_frame_with_timestamp(self):
        """Тест добавления кадра с timestamp"""
        frame = np.random.rand(64, 64)
        timestamp = time.time()
        self.buffer.add_frame(frame, timestamp)

        latest = self.buffer.get_latest_frame()
        self.assertEqual(latest['timestamp'], timestamp)

    def test_get_latest_frame(self):
        """Тест получения последнего кадра"""
        frame1 = np.random.rand(64, 64)
        frame2 = np.random.rand(64, 64)

        self.buffer.add_frame(frame1)
        self.buffer.add_frame(frame2)

        latest = self.buffer.get_latest_frame()
        np.testing.assert_array_equal(latest['data'], frame2)

    def test_get_frames(self):
        """Тест получения нескольких кадров"""
        frames = [np.random.rand(32, 32) for _ in range(5)]
        for frame in frames:
            self.buffer.add_frame(frame)

        retrieved = self.buffer.get_frames(3)
        self.assertEqual(len(retrieved), 3)
        np.testing.assert_array_equal(retrieved[-1]['data'], frames[-1])

    def test_buffer_size_limit(self):
        """Тест ограничения размера буфера"""
        for i in range(150):
            self.buffer.add_frame(np.random.rand(16, 16))

        self.assertEqual(len(self.buffer.buffer), 100)
        self.assertEqual(self.buffer.total_items_added, 150)

    def test_get_fps(self):
        """Тест расчёта FPS"""
        # Пустой буфер
        self.assertEqual(self.buffer.get_fps(), 0)

        # Добавление кадров с известным интервалом
        base_time = time.time()
        for i in range(10):
            self.buffer.add_frame(np.random.rand(16, 16), base_time + i * 0.1)

        fps = self.buffer.get_fps()
        self.assertGreater(fps, 0)
        self.assertLessEqual(fps, 15)

    def test_clear(self):
        """Тест очистки"""
        for i in range(10):
            self.buffer.add_frame(np.random.rand(16, 16))

        self.buffer.clear()

        self.assertEqual(len(self.buffer.buffer), 0)
        self.assertEqual(self.buffer.total_items_added, 0)

    def test_get_stats(self):
        """Тест статистики"""
        self.buffer.add_frame(np.random.rand(16, 16))

        stats = self.buffer.get_stats()

        self.assertIn('current_size', stats)
        self.assertIn('max_size', stats)
        self.assertIn('total_frames', stats)
        self.assertIn('fps', stats)
        self.assertIn('buffer_usage', stats)

        self.assertEqual(stats['current_size'], 1)
        self.assertEqual(stats['total_frames'], 1)


@unittest.skipUnless(VISUALIZER_AVAILABLE, "Visualizer module required")
class TestRealTimeSPMWebSocketAdapter(unittest.TestCase):
    """Тесты для WebSocket адаптера"""

    def setUp(self):
        """Подготовка"""
        self.adapter = RealTimeSPMWebSocketAdapter()

    def test_initialization(self):
        """Тест инициализации"""
        self.assertIsNone(self.adapter.visualizer)
        self.assertEqual(len(self.adapter.connected_clients), 0)
        self.assertEqual(self.adapter.buffer.max_size, 500)

    def test_attach_visualizer(self):
        """Тест подключения визуализатора"""
        mock_visualizer = object()
        self.adapter.attach_visualizer(mock_visualizer)

        self.assertIs(self.adapter.visualizer, mock_visualizer)

    def test_add_remove_client(self):
        """Тест добавления/удаления клиента"""
        self.adapter.add_client("client_1")
        self.assertIn("client_1", self.adapter.connected_clients)

        self.adapter.remove_client("client_1")
        self.assertNotIn("client_1", self.adapter.connected_clients)

    def test_process_frame(self):
        """Тест обработки кадра"""
        frame = np.random.rand(64, 64)
        self.adapter.process_frame(frame)

        stats = self.adapter.get_buffer_stats()
        self.assertEqual(stats['total_frames'], 1)

    def test_get_latest_frame_data(self):
        """Тест получения последнего кадра"""
        frame = np.random.rand(32, 32)
        self.adapter.process_frame(frame)

        latest = self.adapter.get_latest_frame_data()
        self.assertIsNotNone(latest)
        self.assertEqual(latest['data'].shape, (32, 32))

    def test_export_frame_to_json(self):
        """Тест экспорта кадра в JSON"""
        frame = np.random.rand(32, 32) * 100
        self.adapter.process_frame(frame)

        json_str = self.adapter.export_frame_to_json()
        self.assertIsNotNone(json_str)

        import json
        data = json.loads(json_str)

        self.assertIn('frame_id', data)
        self.assertIn('timestamp', data)
        self.assertIn('shape', data)
        self.assertIn('min', data)
        self.assertIn('max', data)
        self.assertEqual(data['shape'], [32, 32])

    def test_export_frame_to_json_invalid_id(self):
        """Тест экспорта несуществующего кадра"""
        result = self.adapter.export_frame_to_json(999)
        self.assertIsNone(result)


@unittest.skipUnless(VISUALIZER_AVAILABLE, "Visualizer module required")
class TestWebSocketAdapterCompression(unittest.TestCase):
    """Тесты сжатия данных"""

    def setUp(self):
        """Подготовка"""
        self.adapter = RealTimeSPMWebSocketAdapter()

    def test_compression_large_frame(self):
        """Тест сжатия большого кадра"""
        large_frame = np.random.rand(256, 256)
        message = self.adapter._create_frame_message(large_frame, time.time())

        # Проверка сжатия
        self.assertEqual(message['shape'], [128, 128])
        self.assertIn('image_base64', message)
        self.assertIsInstance(message['image_base64'], str)

    def test_no_compression_small_frame(self):
        """Тест без сжатия малого кадра"""
        small_frame = np.random.rand(64, 64)
        message = self.adapter._create_frame_message(small_frame, time.time())

        self.assertEqual(message['shape'], [64, 64])

    def test_message_stats(self):
        """Тест статистики в сообщении"""
        frame = np.random.rand(64, 64) * 100
        message = self.adapter._create_frame_message(frame, time.time())

        self.assertIn('stats', message)
        self.assertIn('mean', message['stats'])
        self.assertIn('std', message['stats'])
        self.assertIn('rms', message['stats'])

        expected_mean = float(np.mean(frame))
        self.assertAlmostEqual(message['stats']['mean'], expected_mean, places=5)


@unittest.skipUnless(VISUALIZER_AVAILABLE, "Visualizer module required")
class TestIntegration(unittest.TestCase):
    """Интеграционные тесты"""

    def setUp(self):
        """Подготовка"""
        self.adapter = RealTimeSPMWebSocketAdapter()
        self.adapter.add_client("test_client")

    def test_full_pipeline(self):
        """Тест полного цикла обработки"""
        frames = [np.random.rand(64, 64) * (i + 1) for i in range(10)]

        for frame in frames:
            self.adapter.process_frame(frame)

        stats = self.adapter.get_buffer_stats()
        self.assertEqual(stats['total_frames'], 10)

        latest = self.adapter.get_latest_frame_data()
        self.assertIsNotNone(latest)

        json_str = self.adapter.export_frame_to_json()
        self.assertIsNotNone(json_str)

    def test_multiple_clients(self):
        """Тест работы с несколькими клиентами"""
        clients = [f"client_{i}" for i in range(5)]
        for client in clients:
            self.adapter.add_client(client)

        self.assertEqual(len(self.adapter.connected_clients), 5)

        self.adapter.process_frame(np.random.rand(32, 32))

        for client in clients:
            self.adapter.remove_client(client)

        self.assertEqual(len(self.adapter.connected_clients), 0)


if __name__ == '__main__':
    unittest.main()
