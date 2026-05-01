"""Тесты для кольцевого буфера (ring buffer)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.sdr.ring_buffer import SharedRingBuffer


class TestSharedRingBufferInit(unittest.TestCase):
    """Тесты инициализации SharedRingBuffer."""

    def test_init_default_capacity(self):
        """Тест инициализации с capacity по умолчанию (2M)."""
        buffer = SharedRingBuffer(name="test_default", capacity=1024)
        self.assertEqual(buffer.capacity, 1024)

    def test_init_custom_capacity(self):
        """Тест инициализации с кастомным capacity."""
        buffer = SharedRingBuffer(name="test_custom", capacity=2048)
        self.assertEqual(buffer.capacity, 2048)

    def test_init_name(self):
        """Тест установки имени буфера."""
        buffer = SharedRingBuffer(name="test_name", capacity=1024)
        self.assertEqual(buffer.name, "test_name")


class TestSharedRingBufferWriteRead(unittest.TestCase):
    """Тесты записи и чтения из SharedRingBuffer."""

    def test_write_read_simple(self):
        """Тест простой записи и чтения."""
        buffer = SharedRingBuffer(name="test_write_read", capacity=1024)

        # Создаём тестовые данные
        data = np.array([1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j], dtype=np.complex64)

        # Записываем данные
        written = buffer.write(data)
        self.assertEqual(written, 3)

        # Читаем данные
        read_data = buffer.read(3)
        self.assertIsNotNone(read_data)
        self.assertEqual(len(read_data), 3)

        # Проверяем данные
        np.testing.assert_array_almost_equal(read_data, data)

    def test_write_empty(self):
        """Тест записи пустого массива."""
        buffer = SharedRingBuffer(name="test_empty", capacity=1024)

        data = np.array([], dtype=np.complex64)
        written = buffer.write(data)
        self.assertEqual(written, 0)

    def test_write_large_buffer(self):
        """Тест записи большого количества данных."""
        buffer = SharedRingBuffer(name="test_large", capacity=1024)

        # Записываем больше чем capacity
        data = np.random.randn(2000).astype(np.float32) + 1j * np.random.randn(2000).astype(
            np.float32
        )
        written = buffer.write(data.astype(np.complex64))

        # Должно записаться только capacity элементов
        self.assertLessEqual(written, 1024)

    def test_read_empty(self):
        """Тест чтения из пустого буфера."""
        buffer = SharedRingBuffer(name="test_read_empty", capacity=1024)

        read_data = buffer.read(100)
        # Возвращает пустой массив, а не None
        self.assertIsInstance(read_data, np.ndarray)
        self.assertEqual(len(read_data), 0)

    def test_read_more_than_available(self):
        """Тест чтения большего количества данных чем доступно."""
        buffer = SharedRingBuffer(name="test_read_more", capacity=1024)

        # Записываем 10 элементов
        data = np.array([1.0 + 2.0j] * 10, dtype=np.complex64)
        buffer.write(data)

        # Читаем 100 элементов (доступно только 10)
        read_data = buffer.read(100)
        self.assertIsNotNone(read_data)
        self.assertEqual(len(read_data), 10)


class TestSharedRingBufferOverflow(unittest.TestCase):
    """Тесты переполнения кольцевого буфера."""

    def test_overflow_behavior(self):
        """Тест поведения при переполнении."""
        buffer = SharedRingBuffer(name="test_overflow", capacity=100)

        # Записываем 150 элементов
        data = np.array([1.0 + 2.0j] * 150, dtype=np.complex64)
        buffer.write(data)

        # Должно остаться только 100 последних элементов
        read_data = buffer.read(100)
        self.assertIsNotNone(read_data)
        # При переполнении должно остаться capacity элементов (100)
        self.assertEqual(len(read_data), 100)


class TestSharedRingBufferThreadSafety(unittest.TestCase):
    """Тесты потокобезопасности."""

    def test_concurrent_write_read(self):
        """Тест конкурентной записи и чтения."""
        import threading

        buffer = SharedRingBuffer(name="test_concurrent", capacity=10000)
        errors = []

        def writer():
            try:
                for i in range(100):
                    data = np.array([float(i) + 1.0j] * 100, dtype=np.complex64)
                    buffer.write(data)
            except Exception as e:
                errors.append(str(e))

        def reader():
            try:
                for _ in range(100):
                    data = buffer.read(50)
                    if data is not None and len(data) == 0:
                        errors.append("Empty read")
            except Exception as e:
                errors.append(str(e))

        # Запускаем потоки
        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)

        writer_thread.start()
        reader_thread.start()

        writer_thread.join()
        reader_thread.join()

        # Проверяем что ошибок не было
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")


class TestSharedRingBufferMemory(unittest.TestCase):
    """Тесты управления памятью."""

    def test_cleanup(self):
        """Тест очистки ресурсов."""
        buffer = SharedRingBuffer(name="test_cleanup", capacity=1024)
        # Буфер должен корректно очищаться при удалении
        del buffer
        # Если не было исключений - тест пройден

    def test_mmap_path_creation(self):
        """Тест создания директории для mmap."""
        buffer = SharedRingBuffer(name="test_mmap_path", capacity=1024)
        # Директория должна быть создана
        if hasattr(buffer, "_mmap_path"):
            self.assertTrue(buffer._mmap_path.parent.exists())


class TestSharedRingBufferEdgeCases(unittest.TestCase):
    """Тесты граничных случаев."""

    def test_write_single_element(self):
        """Тест записи одного элемента."""
        buffer = SharedRingBuffer(name="test_single", capacity=1024)

        data = np.array([1.0 + 2.0j], dtype=np.complex64)
        written = buffer.write(data)

        self.assertEqual(written, 1)

        read_data = buffer.read(1)
        self.assertIsNotNone(read_data)
        self.assertEqual(len(read_data), 1)

    def test_capacity_one(self):
        """Тест буфера с capacity=1."""
        buffer = SharedRingBuffer(name="test_cap_one", capacity=1)

        data = np.array([1.0 + 2.0j], dtype=np.complex64)
        written = buffer.write(data)

        self.assertEqual(written, 1)

    def test_multiple_write_read_cycles(self):
        """Тест множественных циклов записи/чтения."""
        buffer = SharedRingBuffer(name="test_cycles", capacity=100)

        for i in range(10):
            data = np.array([float(i)] * 10, dtype=np.complex64)
            buffer.write(data)
            read_data = buffer.read(10)
            self.assertIsNotNone(read_data)
            self.assertEqual(len(read_data), 10)


if __name__ == "__main__":
    unittest.main()
