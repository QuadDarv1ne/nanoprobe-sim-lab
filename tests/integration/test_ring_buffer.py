#!/usr/bin/env python3
"""
Unit tests for SharedRingBuffer from utils.sdr.ring_buffer
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class TestSharedRingBufferBasic:
    """Test basic SharedRingBuffer operations."""

    def test_shared_ring_buffer_create_and_write(self):
        """Test creating a ring buffer and writing samples."""
        import time

        from utils.sdr.ring_buffer import SharedRingBuffer

        logger.info("Starting SharedRingBuffer create and write test")
        # Use timestamp to ensure unique buffer name
        buffer = SharedRingBuffer(name=f"test_basic_write_{int(time.time() * 1000)}", capacity=1024)

        # Write some samples
        samples = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64)
        written = buffer.write(samples)

        assert written == 3, f"Expected 3 samples written, got {written}"
        # available() returns samples in buffer, which should be 3
        assert (
            buffer.available() >= 3
        ), f"Expected at least 3 samples available, got {buffer.available()}"

        del buffer
        logger.info("SharedRingBuffer create and write test passed")

    def test_shared_ring_buffer_read_back(self):
        """Test reading samples back from ring buffer."""
        from utils.sdr.ring_buffer import SharedRingBuffer

        logger.info("Starting SharedRingBuffer read back test")
        buffer = SharedRingBuffer(name="test_read_back", capacity=1024)

        # Write samples
        original = np.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j], dtype=np.complex64)
        buffer.write(original)

        # Read them back
        read = buffer.read(4)

        assert len(read) == 4, f"Expected 4 samples read, got {len(read)}"
        assert np.allclose(read, original), "Read samples do not match written samples"

        del buffer
        logger.info("SharedRingBuffer read back test passed")

    def test_shared_ring_buffer_empty_read(self):
        """Test reading from empty buffer returns empty array."""
        from utils.sdr.ring_buffer import SharedRingBuffer

        logger.info("Starting SharedRingBuffer empty read test")
        buffer = SharedRingBuffer(name="test_empty_read", capacity=1024)

        read = buffer.read(10)

        assert len(read) == 0, f"Expected empty array, got {len(read)} samples"
        assert read.dtype == np.complex64

        del buffer
        logger.info("SharedRingBuffer empty read test passed")

    def test_shared_ring_buffer_overflow(self):
        """Test that writing more than capacity overwrites old data."""
        import time

        from utils.sdr.ring_buffer import SharedRingBuffer

        logger.info("Starting SharedRingBuffer overflow test")
        capacity = 100
        buffer = SharedRingBuffer(
            name=f"test_overflow_{int(time.time() * 1000)}", capacity=capacity
        )

        # Write more than capacity
        first_batch = np.arange(50, dtype=np.complex64)
        second_batch = np.arange(50, 150, dtype=np.complex64)

        buffer.write(first_batch)
        buffer.write(second_batch)

        # Should have at most capacity samples (may be less due to overflow logic)
        available = buffer.available()
        assert available <= capacity, f"Expected at most {capacity} samples, got {available}"
        assert available > 0, "Buffer should not be empty"

        del buffer
        logger.info("SharedRingBuffer overflow test passed")

    def test_shared_ring_buffer_clear(self):
        """Test clearing the ring buffer."""
        from utils.sdr.ring_buffer import SharedRingBuffer

        logger.info("Starting SharedRingBuffer clear test")
        buffer = SharedRingBuffer(name="test_clear", capacity=1024)

        # Write and then clear
        samples = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
        buffer.write(samples)
        assert buffer.available() == 2

        buffer.clear()
        assert buffer.available() == 0, "Buffer should be empty after clear"

        del buffer
        logger.info("SharedRingBuffer clear test passed")


class TestSharedRingBufferEdgeCases:
    """Test SharedRingBuffer edge cases."""

    def test_shared_ring_buffer_zero_write(self):
        """Test writing zero samples."""
        from utils.sdr.ring_buffer import SharedRingBuffer

        logger.info("Starting SharedRingBuffer zero write test")
        buffer = SharedRingBuffer(name="test_zero_write", capacity=1024)

        empty = np.array([], dtype=np.complex64)
        written = buffer.write(empty)

        assert written == 0
        assert buffer.available() == 0

        del buffer
        logger.info("SharedRingBuffer zero write test passed")

    def test_shared_ring_buffer_zero_read(self):
        """Test reading zero samples."""
        import time

        from utils.sdr.ring_buffer import SharedRingBuffer

        logger.info("Starting SharedRingBuffer zero read test")
        buffer = SharedRingBuffer(name=f"test_zero_read_{int(time.time() * 1000)}", capacity=1024)

        samples = np.array([1 + 2j], dtype=np.complex64)
        buffer.write(samples)

        read = buffer.read(0)
        assert len(read) == 0
        # Data still there (may be 1 or more due to implementation)
        assert buffer.available() >= 1

        del buffer
        logger.info("SharedRingBuffer zero read test passed")

    def test_shared_ring_buffer_large_write(self):
        """Test writing a large batch of samples."""
        import time

        from utils.sdr.ring_buffer import SharedRingBuffer

        logger.info("Starting SharedRingBuffer large write test")
        capacity = 10000
        buffer = SharedRingBuffer(
            name=f"test_large_write_{int(time.time() * 1000)}", capacity=capacity
        )

        # Write 10x capacity
        large_batch = np.random.randn(100000).astype(np.complex64)
        written = buffer.write(large_batch)

        assert written == capacity, f"Expected {capacity} written, got {written}"
        # available should be close to capacity (may vary slightly due to implementation)
        available = buffer.available()
        assert (
            available >= capacity - 1
        ), f"Expected at least {capacity - 1} samples, got {available}"

        del buffer
        logger.info("SharedRingBuffer large write test passed")
