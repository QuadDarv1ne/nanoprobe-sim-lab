"""
SDR Resource Manager

Singleton manager for shared SDR receiver access with priority-based
queue and preemption support.

Priority levels:
  ISS (100) > weather (80) > SSTV (60) > scan (20)
"""

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Priority constants ──────────────────────────────────────────────────────

PRIORITY_ISS = 100
PRIORITY_WEATHER = 80
PRIORITY_SSTV = 60
PRIORITY_SCAN = 20

_PRIORITY_NAMES: Dict[int, str] = {
    PRIORITY_ISS: "ISS",
    PRIORITY_WEATHER: "weather",
    PRIORITY_SSTV: "SSTV",
    PRIORITY_SCAN: "scan",
}


def priority_name(priority: int) -> str:
    return _PRIORITY_NAMES.get(priority, f"custom({priority})")


# ── Data classes ────────────────────────────────────────────────────────────


@dataclass(order=False)
class SDRTask:
    """Represents a queued SDR access request."""

    task_name: str
    priority: int
    frequency: float
    gain: Optional[float] = None
    queued_at: float = 0.0  # set by manager on enqueue

    def __gt__(self, other):
        """Higher priority tasks sort first; ties break by earlier queue time."""
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.queued_at < other.queued_at


# ── Singleton ───────────────────────────────────────────────────────────────


class SDRResourceManager:
    """Thread-safe singleton managing exclusive SDR receiver access.

    Uses a priority queue with preemption: a higher-priority request can
    displace the currently active lower-priority task.
    """

    _instance: Optional["SDRResourceManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "SDRResourceManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Guard against re-initialization in singleton
        if hasattr(self, "_initialized"):
            return
        self._mutex = threading.Lock()
        self._queue: List[SDRTask] = []  # kept sorted by priority
        self._active_task: Optional[SDRTask] = None
        self._initialized = True
        logger.info("SDRResourceManager initialized")

    # ── Public API ───────────────────────────────────────────────────────

    def request_access(
        self,
        task_name: str,
        priority: int,
        frequency: float,
        gain: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Request exclusive SDR access.

        If no task is active, the request is granted immediately.
        If the new priority exceeds the active task's priority, the current
        task is preempted and the new one takes over.
        Otherwise the request is queued.

        Returns:
            (granted, message) — *granted* is True if access is given now.
        """
        task = SDRTask(
            task_name=task_name,
            priority=priority,
            frequency=frequency,
            gain=gain,
        )

        with self._mutex:
            import time

            task.queued_at = time.time()

            # No active task — grant immediately
            if self._active_task is None:
                self._active_task = task
                logger.info(
                    f"SDR access granted: {task_name} "
                    f"(priority={priority}, freq={frequency} MHz)"
                )
                return True, "Access granted"

            # Already active with same name — refresh frequency/gain
            if self._active_task.task_name == task_name:
                self._active_task.frequency = frequency
                self._active_task.gain = gain
                logger.debug(f"SDR params updated: {task_name}")
                return True, "Access refreshed"

            # Higher priority — preempt current task
            if self._preempt_current_task(priority):
                preempted = self._active_task
                self._active_task = task
                logger.warning(
                    f"Preemption: {task_name} (p={priority}) replaced "
                    f"{preempted.task_name} (p={preempted.priority})"
                )
                return True, f"Preempted {preempted.task_name}"

            # Queue the request
            self._enqueue(task)
            position = self._queue_position(task)
            msg = f"Queued at position {position}"
            logger.info(f"SDR access queued: {task_name} ({msg})")
            return False, msg

    def release_access(self, task_name: str) -> Optional[SDRTask]:
        """Release SDR access held by *task_name*.

        Returns the next queued task (if any) that should now become active,
        or None if the queue is empty.
        """
        with self._mutex:
            if self._active_task and self._active_task.task_name == task_name:
                logger.info(f"SDR released by: {task_name}")
                self._active_task = None
            elif self._active_task:
                logger.warning(
                    f"release_access called by {task_name}, "
                    f"but active task is {self._active_task.task_name}"
                )
                return None

            # Promote next queued task
            if self._queue:
                next_task = self._queue.pop(0)
                self._active_task = next_task
                logger.info(
                    f"Next task promoted from queue: {next_task.task_name} "
                    f"(priority={next_task.priority})"
                )
                return next_task

            return None

    def get_queue_status(self) -> List[Dict[str, Any]]:
        """Return snapshot of the current queue."""
        with self._mutex:
            return [
                {
                    "task_name": t.task_name,
                    "priority": t.priority,
                    "priority_name": priority_name(t.priority),
                    "frequency": t.frequency,
                    "gain": t.gain,
                }
                for t in self._queue
            ]

    def get_active_task(self) -> Optional[Dict[str, Any]]:
        """Return info about the currently active task, or None."""
        with self._mutex:
            if self._active_task is None:
                return None
            t = self._active_task
            return {
                "task_name": t.task_name,
                "priority": t.priority,
                "priority_name": priority_name(t.priority),
                "frequency": t.frequency,
                "gain": t.gain,
            }

    def cancel_task(self, task_name: str) -> Tuple[bool, str]:
        """Cancel a queued or active task.

        Returns (success, message).
        """
        with self._mutex:
            # Cancel active task
            if self._active_task and self._active_task.task_name == task_name:
                self._active_task = None
                logger.info(f"Active task cancelled: {task_name}")
                # Promote next
                if self._queue:
                    next_task = self._queue.pop(0)
                    self._active_task = next_task
                    return True, f"Cancelled {task_name}, promoted {next_task.task_name}"
                return True, f"Cancelled {task_name}, queue empty"

            # Cancel queued task
            for i, t in enumerate(self._queue):
                if t.task_name == task_name:
                    self._queue.pop(i)
                    logger.info(f"Queued task cancelled: {task_name}")
                    return True, f"Cancelled queued task {task_name}"

            return False, f"Task {task_name} not found"

    # ── Internal helpers ─────────────────────────────────────────────────

    def _preempt_current_task(self, new_priority: int) -> bool:
        """Decide whether a new request should preempt the active task.

        Preemption occurs when *new_priority* strictly exceeds the active
        task's priority.
        """
        if self._active_task is None:
            return False
        return new_priority > self._active_task.priority

    def _enqueue(self, task: SDRTask):
        """Insert *task* into the priority-sorted queue."""
        self._queue.append(task)
        self._queue.sort(key=lambda t: (-t.priority, t.queued_at))

    def _queue_position(self, task: SDRTask) -> int:
        """1-based position of *task* in the sorted queue."""
        for i, t in enumerate(self._queue):
            if t is task:
                return i + 1
        return len(self._queue)


# ── Module-level singleton accessor ─────────────────────────────────────────


def get_sdr_resource_manager() -> SDRResourceManager:
    """Return the global SDRResourceManager singleton."""
    return SDRResourceManager()
