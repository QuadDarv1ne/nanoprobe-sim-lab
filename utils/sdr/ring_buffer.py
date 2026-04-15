"""
Ring Buffer in Shared Memory

Cross-platform circular buffer for complex64 IQ samples.
Uses POSIX SHM on Linux, mmap file-backed on Windows.
Capacity: 2M complex64 samples (~16MB data + 32B header).
"""

import logging
import struct
import sys
from pathlib import Path
from threading import Lock

import numpy as np

logger = logging.getLogger(__name__)

MAGIC = b"RING"
VERSION = 1
HEADER_SIZE = 32
HEADER_FORMAT = (
    "<4sIIqII"  # magic(4B) + version(4B) + capacity(4B) + head(8B) + tail(8B) + flags(4B)
)


class SharedRingBuffer:
    """Thread-safe ring buffer for complex64 samples in shared memory."""

    def __init__(self, name: str, capacity: int = 2097152):
        """
        Args:
            name: Unique name for the shared memory region.
            capacity: Number of complex64 samples (default 2M).
        """
        self.name = name
        self.capacity = capacity
        self._data_size = capacity * 8  # complex64 = 2 * float32 = 8 bytes
        self._lock = Lock()

        self._shm = None
        self._buf = None

        self._init_memory()

    def _init_memory(self):
        """Initialize shared memory region (platform-specific)."""
        try:
            if sys.platform != "win32":
                self._init_posix_shm()
            else:
                self._init_mmap()
        except Exception as e:
            logger.warning("Shared memory init failed (%s), falling back to mmap", e)
            self._init_mmap()

    def _init_posix_shm(self):
        """POSIX shared memory (Linux/macOS)."""
        from multiprocessing.shared_memory import SharedMemory

        try:
            self._shm = SharedMemory(
                name=self.name, create=True, size=HEADER_SIZE + self._data_size
            )
            logger.info(
                "RingBuffer created POSIX SHM '%s': %d samples (%.1f MB)",
                self.name,
                self.capacity,
                self._data_size / 1e6,
            )
        except FileExistsError:
            logger.info("RingBuffer attaching to existing SHM '%s'", self.name)
            self._shm = SharedMemory(name=self.name, create=False)
        self._buf = self._shm.buf
        self._header = memoryview(self._buf)[:HEADER_SIZE]
        self._data = memoryview(self._buf)[HEADER_SIZE:]
        self._write_header(self.capacity, 0, 0, 0)

    def _init_mmap(self):
        """File-backed mmap fallback (Windows or fallback)."""
        import mmap

        self._mmap_path = Path("output") / "ring_buffers" / f"{self.name}.buf"
        self._mmap_path.parent.mkdir(parents=True, exist_ok=True)

        self._fd = open(self._mmap_path, "r+b" if self._mmap_path.exists() else "w+b")
        self._fd.truncate(HEADER_SIZE + self._data_size)
        self._buf = mmap.mmap(self._fd.fileno(), HEADER_SIZE + self._data_size)
        self._header = self._buf[:HEADER_SIZE]
        self._data = self._buf[HEADER_SIZE:]

        existing = self._read_header()
        if existing is None or existing[0] != self.capacity:
            self._write_header(self.capacity, 0, 0, 0)
            logger.info(
                "RingBuffer created mmap '%s': %d samples (%.1f MB)",
                self.name,
                self.capacity,
                self._data_size / 1e6,
            )
        else:
            logger.info("RingBuffer attached to existing mmap '%s'", self.name)

    def _write_header(self, capacity, head, tail, flags):
        struct.pack_into(
            HEADER_FORMAT, self._header, 0, MAGIC, VERSION, capacity, head, tail, flags
        )

    def _read_header(self):
        if len(self._header) < HEADER_SIZE:
            return None
        magic, version, capacity, head, tail, flags = struct.unpack_from(
            HEADER_FORMAT, self._header, 0
        )
        if magic != MAGIC:
            return None
        return capacity, head, tail, flags

    def write(self, samples: np.ndarray) -> int:
        """
        Write samples to the ring buffer.

        Args:
            samples: complex64 numpy array.

        Returns:
            Number of samples written.
        """
        if len(samples) == 0:
            return 0

        samples = np.ascontiguousarray(samples, dtype=np.complex64)
        n = len(samples)

        with self._lock:
            header = self._read_header()
            if header is None:
                logger.error("RingBuffer: corrupted header")
                return 0

            capacity, head, tail, flags = header

            if n > capacity:
                # Overwrite entire buffer — keep only last `capacity` samples
                data = samples[-capacity:]
                n = len(data)
            else:
                data = samples

            bytes_to_write = data.view(np.float32).tobytes()

            if head + len(data) <= capacity:
                self._data[head * 8 : (head + len(data)) * 8] = bytes_to_write
            else:
                first_part = capacity - head
                self._data[head * 8 :] = bytes_to_write[: first_part * 8]
                self._data[: (len(data) - first_part) * 8] = bytes_to_write[first_part * 8 :]

            new_head = (head + len(data)) % capacity
            # If we've overwritten the tail, advance it
            if new_head <= tail:
                tail = (new_head + 1) % capacity
            self._write_header(capacity, new_head, tail, flags)

        return n

    def read(self, count: int) -> np.ndarray:
        """
        Read samples from the ring buffer.

        Args:
            count: Number of samples to read.

        Returns:
            numpy array of complex64 samples.
        """
        if count <= 0:
            return np.array([], dtype=np.complex64)

        with self._lock:
            header = self._read_header()
            if header is None:
                return np.array([], dtype=np.complex64)

            capacity, head, tail, flags = header
            available_count = (head - tail) % capacity
            if available_count == 0:
                return np.array([], dtype=np.complex64)

            to_read = min(count, available_count)
            result = np.empty(to_read, dtype=np.complex64)

            if tail + to_read <= capacity:
                data_slice = self._data[tail * 8 : (tail + to_read) * 8]
                result.view(np.float32)[:] = np.frombuffer(data_slice, dtype=np.float32)
            else:
                first_part = capacity - tail
                data1 = self._data[tail * 8 :]
                result[:first_part].view(np.float32)[:] = np.frombuffer(data1, dtype=np.float32)
                remaining = to_read - first_part
                data2 = self._data[: remaining * 8]
                result[first_part:].view(np.float32)[:] = np.frombuffer(data2, dtype=np.float32)

            new_tail = (tail + to_read) % capacity
            self._write_header(capacity, head, new_tail, flags)

        return result

    def available(self) -> int:
        """Number of samples available for reading."""
        with self._lock:
            header = self._read_header()
            if header is None:
                return 0
            capacity, head, tail, flags = header
            return (head - tail) % capacity

    def clear(self):
        """Reset head and tail pointers."""
        with self._lock:
            header = self._read_header()
            if header is not None:
                capacity, _, _, flags = header
                self._write_header(capacity, 0, 0, flags)

    def __del__(self):
        """Clean up shared memory resources."""
        try:
            if hasattr(self, "_shm") and self._shm is not None:
                self._shm.close()
                try:
                    self._shm.unlink()
                except Exception:
                    pass  # May already be unlinked
            elif hasattr(self, "_buf") and self._buf is not None:
                if hasattr(self, "_fd") and self._fd is not None:
                    self._buf.close()
                    self._fd.close()
        except Exception as e:
            logger.debug("RingBuffer cleanup: %s", e)
