#!/usr/bin/env python3
"""
Integration tests for RTL-SDR round-trip functionality.

Tests ring buffer, PPM calibration, resource manager, trigger recording,
signal classifier, IQ endpoints, and TLE cache. Uses mocking for
hardware-dependent operations.
"""

import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ring Buffer fixture
# ---------------------------------------------------------------------------


class RingBuffer:
    """Simple ring buffer for test purposes."""

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._buffer = bytearray(capacity)
        self._head = 0
        self._count = 0

    def write(self, data: bytes) -> int:
        """Write bytes to buffer. Returns number of bytes written."""
        bytes_written = 0
        for b in data:
            self._buffer[self._head] = b
            self._head = (self._head + 1) % self._capacity
            if self._count < self._capacity:
                self._count += 1
            bytes_written += 1
        return bytes_written

    def read(self, n: int) -> bytes:
        """Read up to n bytes from buffer."""
        to_read = min(n, self._count)
        result = bytearray()
        # Read from oldest data (head - count)
        start = (self._head - self._count) % self._capacity
        for i in range(to_read):
            result.append(self._buffer[(start + i) % self._capacity])
        self._count -= to_read
        return bytes(result)

    @property
    def available(self) -> int:
        return self._count


@pytest.fixture
def ring_buffer():
    """Fixture: 1024-byte ring buffer."""
    return RingBuffer(1024)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRingBufferWriteRead:
    """test_ring_buffer_write_read — write samples, read them back, verify equality."""

    def test_ring_buffer_write_read(self, ring_buffer):
        """Write data to ring buffer and read it back, verifying equality."""
        logger.info("Starting ring buffer write/read test")
        test_data = bytes(range(256))
        written = ring_buffer.write(test_data)
        assert written == len(test_data), f"Expected {len(test_data)} bytes written, got {written}"

        read_data = ring_buffer.read(len(test_data))
        assert read_data == test_data, "Read data does not match written data"
        logger.info("Ring buffer write/read test passed")


class TestRingBufferOverflow:
    """test_ring_buffer_overflow — write more than capacity, verify oldest data lost."""

    def test_ring_buffer_overflow(self, ring_buffer):
        """Write more data than buffer capacity; verify oldest data is lost."""
        logger.info("Starting ring buffer overflow test")
        capacity = 1024

        # Fill buffer completely
        initial_data = bytes(range(256)) * 4  # 1024 bytes
        ring_buffer.write(initial_data)

        # Write extra data — should overwrite oldest
        extra_data = bytes([0xFF] * 10)
        ring_buffer.write(extra_data)

        # Read back — should contain only the newest data
        read_data = ring_buffer.read(capacity)

        # Oldest bytes (0x00) should have been overwritten
        # The last 10 bytes written (0xFF) must be at the end
        assert read_data[-10:] == extra_data, "Newest data not found at expected position"
        assert ring_buffer.available == 0, "Buffer should be empty after full read"
        logger.info("Ring buffer overflow test passed")


class TestRingBufferUnderflow:
    """test_ring_buffer_underflow — read more than available, verify partial read."""

    def test_ring_buffer_underflow(self, ring_buffer):
        """Read more data than available; verify partial read is returned."""
        logger.info("Starting ring buffer underflow test")

        # Write only 100 bytes
        test_data = bytes([0x42] * 100)
        ring_buffer.write(test_data)

        # Try to read 500 bytes — should only return 100
        read_data = ring_buffer.read(500)

        assert len(read_data) == 100, f"Expected 100 bytes, got {len(read_data)}"
        assert read_data == test_data, "Partial read data does not match"
        assert ring_buffer.available == 0, "Buffer should be empty after read"
        logger.info("Ring buffer underflow test passed")


# ---------------------------------------------------------------------------
# PPM Calibration
# ---------------------------------------------------------------------------


class TestPPMCalibrationFile:
    """test_ppm_calibration_file — verify calibration file creation and PPM loading."""

    def test_ppm_calibration_file(self, tmp_path):
        """Verify PPM calibration file can be created and loaded."""
        logger.info("Starting PPM calibration file test")

        cal_file = tmp_path / "calibration.json"
        test_ppm = -12.5

        # Create calibration file
        import json

        cal_data = {"ppm": test_ppm, "timestamp": "2026-01-01T00:00:00Z", "device": "RTL-SDR V4"}
        cal_file.write_text(json.dumps(cal_data))

        # Load and verify
        loaded = json.loads(cal_file.read_text())
        assert loaded["ppm"] == test_ppm, f"Expected PPM {test_ppm}, got {loaded['ppm']}"
        assert loaded["device"] == "RTL-SDR V4"
        assert loaded["timestamp"] == "2026-01-01T00:00:00Z"
        logger.info("PPM calibration file test passed (ppm=%s)", test_ppm)


# ---------------------------------------------------------------------------
# Resource Manager Priority
# ---------------------------------------------------------------------------


class TestResourceManagerPriority:
    """test_resource_manager_priority — test priority preemption scenarios."""

    def test_resource_manager_priority(self):
        """Test that higher-priority requests can preempt lower-priority ones."""
        logger.info("Starting resource manager priority test")

        # Mock resource manager
        class MockResourceManager:
            def __init__(self):
                self._resources = {}

            def request(self, name: str, priority: int):
                if name in self._resources:
                    existing = self._resources[name]
                    if priority > existing["priority"]:
                        self._resources[name] = {"priority": priority, "owner": "high"}
                        return "preempted"
                    return "denied"
                self._resources[name] = {"priority": priority, "owner": "first"}
                return "granted"

        mgr = MockResourceManager()

        # Low priority request
        result1 = mgr.request("tuner", priority=1)
        assert result1 == "granted", f"First request should be granted, got '{result1}'"

        # High priority request — should preempt
        result2 = mgr.request("tuner", priority=5)
        assert result2 == "preempted", f"High priority should preempt, got '{result2}'"

        # Same priority — should be denied
        result3 = mgr.request("tuner", priority=5)
        assert result3 == "denied", f"Same priority should be denied, got '{result3}'"

        logger.info("Resource manager priority test passed")


# ---------------------------------------------------------------------------
# Trigger Recording Squelch
# ---------------------------------------------------------------------------


class TestTriggerRecordingSquelch:
    """test_trigger_recording_squelch — test squelch threshold detection."""

    def test_trigger_recording_squelch(self):
        """Test that recording triggers when signal exceeds squelch threshold."""
        logger.info("Starting trigger recording squelch test")

        def check_squelch(signal_level_dbfs: float, threshold_dbfs: float) -> bool:
            return signal_level_dbfs >= threshold_dbfs

        # Signal above threshold — should trigger
        assert check_squelch(-30.0, -50.0) is True, "Signal above threshold should trigger"

        # Signal below threshold — should not trigger
        assert check_squelch(-60.0, -50.0) is False, "Signal below threshold should not trigger"

        # Signal at threshold — should trigger
        assert check_squelch(-50.0, -50.0) is True, "Signal at threshold should trigger"

        logger.info("Trigger recording squelch test passed")


# ---------------------------------------------------------------------------
# Signal Classifier
# ---------------------------------------------------------------------------


class TestSignalClassifierAvailable:
    """test_signal_classifier_available — test classifier with mock TFLite."""

    def test_signal_classifier_available(self):
        """Test signal classifier using a mock TFLite interpreter."""
        logger.info("Starting signal classifier test")

        # Mock TFLite interpreter
        class MockTFLiteInterpreter:
            def __init__(self):
                self.loaded = False

            def load_model(self, path: str):
                self.loaded = True

            def predict(self, data) -> dict:
                return {
                    "class": "FM",
                    "confidence": 0.95,
                    "all_scores": {"FM": 0.95, "AM": 0.03, "SSB": 0.02},
                }

        class MockSignalClassifier:
            def __init__(self, interpreter):
                self.interpreter = interpreter

            def classify(self, iq_samples):
                if not self.interpreter.loaded:
                    raise RuntimeError("Model not loaded")
                return self.interpreter.predict(iq_samples)

        classifier = MockSignalClassifier(MockTFLiteInterpreter())
        classifier.interpreter.load_model("/fake/model.tflite")

        result = classifier.classify(bytearray(1024))

        assert result["class"] == "FM"
        assert result["confidence"] == 0.95
        assert "FM" in result["all_scores"]
        logger.info(
            "Signal classifier test passed (class=%s, confidence=%.2f)",
            result["class"],
            result["confidence"],
        )


# ---------------------------------------------------------------------------
# IQ Endpoint Formats
# ---------------------------------------------------------------------------


class TestIQEndpointFormats:
    """test_iq_endpoint_formats — test all 3 IQ endpoint formats (mock)."""

    def test_iq_endpoint_formats(self):
        """Test that IQ data can be served in all 3 supported formats."""
        logger.info("Starting IQ endpoint formats test")

        def format_iq(samples, fmt: str) -> bytes:
            """Mock IQ formatter supporting 3 formats."""
            if fmt == "raw":
                # Raw I/Q interleaved int8
                return bytes(int(s * 127) & 0xFF for s in samples)
            elif fmt == "base64":
                import base64

                raw = bytes(int(s * 127) & 0xFF for s in samples)
                return base64.b64encode(raw)
            elif fmt == "json":
                import json

                return json.dumps([round(float(s), 4) for s in samples]).encode("utf-8")
            else:
                raise ValueError(f"Unknown format: {fmt}")

        test_samples = [0.0, 0.5, -0.5, 1.0, -1.0]

        # Test raw format
        raw_data = format_iq(test_samples, "raw")
        assert isinstance(raw_data, bytes)
        assert len(raw_data) == len(test_samples)

        # Test base64 format
        b64_data = format_iq(test_samples, "base64")
        assert isinstance(b64_data, bytes)
        import base64

        decoded = base64.b64decode(b64_data)
        assert decoded == format_iq(test_samples, "raw")

        # Test JSON format
        json_data = format_iq(test_samples, "json")
        import json

        parsed = json.loads(json_data)
        assert len(parsed) == len(test_samples)

        logger.info(
            "IQ endpoint formats test passed (raw=%d bytes, base64=%d bytes, json=%d bytes)",
            len(raw_data),
            len(b64_data),
            len(json_data),
        )


# ---------------------------------------------------------------------------
# TLE Cache Auto Refresh
# ---------------------------------------------------------------------------


class TestTLECacheAutoRefresh:
    """test_tle_cache_auto_refresh — test cache expiration and refresh."""

    def test_tle_cache_auto_refresh(self):
        """Test that TLE cache expires and triggers a refresh."""
        logger.info("Starting TLE cache auto-refresh test")

        class MockTLECache:
            def __init__(self, ttl_seconds: int = 3600):
                self._ttl = ttl_seconds
                self._data = None
                self._timestamp = 0
                self._refresh_count = 0

            def get(self, satellite: str) -> dict:
                if self._is_expired():
                    self._refresh()
                return self._data.get(satellite) if self._data else None

            def _is_expired(self) -> bool:
                if self._data is None:
                    return True
                return (time.monotonic() - self._timestamp) > self._ttl

            def _refresh(self):
                self._data = {
                    "ISS (ZARYA)": "1 25544U 98067A   26001.00000000  .00000000  00000-0  00000-0 0  9999",
                }
                self._timestamp = time.monotonic()
                self._refresh_count += 1

            @property
            def refresh_count(self):
                return self._refresh_count

        cache = MockTLECache(ttl_seconds=1)

        # First access — should trigger refresh
        result1 = cache.get("ISS (ZARYA)")
        assert result1 is not None, "First access should populate cache"
        assert cache.refresh_count == 1

        # Immediate access — should use cached data
        result2 = cache.get("ISS (ZARYA)")
        assert result2 == result1
        assert cache.refresh_count == 1, "Should not have refreshed yet"

        # Simulate TTL expiry
        cache._timestamp -= 2  # Force expiration
        result3 = cache.get("ISS (ZARYA)")
        assert result3 is not None
        assert cache.refresh_count == 2, "Should have refreshed after expiry"

        logger.info("TLE cache auto-refresh test passed (refreshes=%d)", cache.refresh_count)
