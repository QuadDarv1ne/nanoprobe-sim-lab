#!/usr/bin/env python3
"""
Unit tests for SDRResourceManager from utils.sdr.sdr_resource_manager
"""
import logging
import sys
import time
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

# Import the module under test
from utils.sdr.sdr_resource_manager import (
    PRIORITY_ISS,
    PRIORITY_SCAN,
    PRIORITY_SSTV,
    PRIORITY_WEATHER,
    SDRResourceManager,
    SDRTask,
    get_sdr_resource_manager,
    priority_name,
)


class TestSDRTask:
    """Test SDRTask dataclass."""

    def test_task_creation(self):
        """Test creating an SDRTask."""
        task = SDRTask(
            task_name="test_task",
            priority=PRIORITY_SSTV,
            frequency=145.800,
            gain=30.0,
        )

        assert task.task_name == "test_task"
        assert task.priority == PRIORITY_SSTV
        assert task.frequency == 145.800
        assert task.gain == 30.0

    def test_task_priority_comparison(self):
        """Test task comparison by priority."""
        high_priority = SDRTask(task_name="high", priority=PRIORITY_ISS, frequency=145.8)
        low_priority = SDRTask(task_name="low", priority=PRIORITY_SCAN, frequency=137.6)

        # Higher priority should be "greater"
        assert high_priority > low_priority
        assert not (low_priority > high_priority)

    def test_task_time_comparison(self):
        """Test task comparison by queue time when priorities are equal."""
        task1 = SDRTask(task_name="first", priority=PRIORITY_SSTV, frequency=145.8)
        task1.queued_at = time.time()
        time.sleep(0.01)  # Small delay
        task2 = SDRTask(task_name="second", priority=PRIORITY_SSTV, frequency=145.8)
        task2.queued_at = time.time()

        # Earlier task should be "greater" (sorted first)
        assert task1 > task2


class TestPriorityName:
    """Test priority_name helper function."""

    def test_known_priorities(self):
        """Test priority names for known values."""
        assert priority_name(PRIORITY_ISS) == "ISS"
        assert priority_name(PRIORITY_WEATHER) == "weather"
        assert priority_name(PRIORITY_SSTV) == "SSTV"
        assert priority_name(PRIORITY_SCAN) == "scan"

    def test_unknown_priority(self):
        """Test priority name for unknown value."""
        assert priority_name(50) == "custom(50)"
        assert priority_name(999) == "custom(999)"


class TestSDRResourceManager:
    """Test SDRResourceManager singleton."""

    @pytest.fixture(autouse=True)
    def reset_manager(self):
        """Reset the singleton before each test."""
        SDRResourceManager._instance = None
        yield
        # Cleanup after test
        SDRResourceManager._instance = None

    def test_singleton_pattern(self):
        """Test that SDRResourceManager is a singleton."""
        manager1 = get_sdr_resource_manager()
        manager2 = get_sdr_resource_manager()

        assert manager1 is manager2

    def test_request_access_granted_when_idle(self):
        """Test that access is granted when no task is active."""
        manager = get_sdr_resource_manager()

        granted, message = manager.request_access(
            "test_task", priority=PRIORITY_SSTV, frequency=145.800
        )

        assert granted is True
        assert "Access granted" in message
        assert manager.get_active_task() is not None

    def test_request_access_queued_when_busy(self):
        """Test that access is queued when a lower priority task is active."""
        manager = get_sdr_resource_manager()

        # First task gets access
        granted1, _ = manager.request_access(
            "high_priority", priority=PRIORITY_ISS, frequency=145.800
        )
        assert granted1 is True

        # Second task with lower priority should be queued
        granted2, message = manager.request_access(
            "low_priority", priority=PRIORITY_SCAN, frequency=137.620
        )

        assert granted2 is False
        assert "Queued" in message

        status = manager.get_queue_status()
        assert len(status) == 1
        assert status[0]["task_name"] == "low_priority"

    def test_preemption_higher_priority(self):
        """Test that higher priority task preempts lower priority task."""
        manager = get_sdr_resource_manager()

        # Low priority task gets access
        manager.request_access("low", priority=PRIORITY_SCAN, frequency=137.620)
        assert manager.get_active_task()["task_name"] == "low"

        # High priority task should preempt
        granted, message = manager.request_access("high", priority=PRIORITY_ISS, frequency=145.800)

        assert granted is True
        assert "Preempted" in message
        assert manager.get_active_task()["task_name"] == "high"

    def test_release_access_promotes_next(self):
        """Test that releasing access promotes the next queued task."""
        manager = get_sdr_resource_manager()

        # Add scan_task first (gets access)
        manager.request_access("scan_task", priority=PRIORITY_SCAN, frequency=137.620)
        assert manager.get_active_task()["task_name"] == "scan_task"

        # Add weather_task with higher priority - it preempts but scan_task is NOT queued
        # So we need to add another task that gets queued
        manager.request_access("weather_task", priority=PRIORITY_WEATHER, frequency=137.620)
        assert manager.get_active_task()["task_name"] == "weather_task"

        # Add sstv_task - it should be queued since weather_task has higher priority
        granted, msg = manager.request_access(
            "sstv_task", priority=PRIORITY_SSTV, frequency=145.800
        )
        assert granted is False  # Should be queued

        # Release weather_task
        next_task = manager.release_access("weather_task")

        # sstv_task should be promoted (scan_task was preempted, not queued)
        assert next_task is not None
        assert next_task.task_name == "sstv_task"
        assert manager.get_active_task()["task_name"] == "sstv_task"

    def test_cancel_active_task(self):
        """Test cancelling an active task."""
        manager = get_sdr_resource_manager()

        manager.request_access("active_task", priority=PRIORITY_SSTV, frequency=145.800)
        assert manager.get_active_task() is not None

        success, message = manager.cancel_task("active_task")

        assert success is True
        assert manager.get_active_task() is None

    def test_cancel_queued_task(self):
        """Test cancelling a queued task."""
        manager = get_sdr_resource_manager()

        # Create active task
        manager.request_access("active", priority=PRIORITY_ISS, frequency=145.800)

        # Add queued tasks
        manager.request_access("queued1", priority=PRIORITY_SSTV, frequency=145.800)
        manager.request_access("queued2", priority=PRIORITY_SCAN, frequency=137.620)

        # Cancel queued task
        success, message = manager.cancel_task("queued1")

        assert success is True
        status = manager.get_queue_status()
        assert len(status) == 1
        assert status[0]["task_name"] == "queued2"

    def test_queue_status(self):
        """Test queue status reporting."""
        manager = get_sdr_resource_manager()

        manager.request_access("task1", priority=PRIORITY_SSTV, frequency=145.800, gain=30.0)
        manager.request_access("task2", priority=PRIORITY_SCAN, frequency=137.620)

        status = manager.get_queue_status()

        assert len(status) == 1  # Only task2 is queued, task1 is active
        assert status[0]["task_name"] == "task2"
        assert status[0]["priority"] == PRIORITY_SCAN
        assert status[0]["frequency"] == 137.620

    def test_active_task_info(self):
        """Test active task info reporting."""
        manager = get_sdr_resource_manager()

        manager.request_access("active", priority=PRIORITY_WEATHER, frequency=137.620, gain=35.0)

        active = manager.get_active_task()

        assert active is not None
        assert active["task_name"] == "active"
        assert active["priority"] == PRIORITY_WEATHER
        assert active["frequency"] == 137.620
        assert active["gain"] == 35.0
        assert active["priority_name"] == "weather"

    def test_release_by_non_active_task(self):
        """Test that releasing by a non-active task logs a warning."""
        manager = get_sdr_resource_manager()

        manager.request_access("active", priority=PRIORITY_ISS, frequency=145.800)

        # Try to release by wrong task name
        result = manager.release_access("wrong_task")

        assert result is None

    def test_priority_ordering_in_queue(self):
        """Test that queue maintains priority order."""
        manager = get_sdr_resource_manager()

        # Add tasks in random order - scan first
        manager.request_access("scan", priority=PRIORITY_SCAN, frequency=137.0)
        # iss preempts scan
        manager.request_access("iss", priority=PRIORITY_ISS, frequency=145.8)
        # weather preempts iss
        manager.request_access("weather", priority=PRIORITY_WEATHER, frequency=137.6)
        # sstv doesn't preempt weather
        manager.request_access("sstv", priority=PRIORITY_SSTV, frequency=145.8)

        # weather is active (highest priority that ran last), iss and sstv are queued
        status = manager.get_queue_status()
        assert len(status) == 2  # Two are queued, one is active

        # The queue should be sorted by priority (descending)
        priorities = [t["priority"] for t in status]
        assert priorities == sorted(priorities, reverse=True)

    def test_refresh_active_task_params(self):
        """Test refreshing parameters of active task."""
        manager = get_sdr_resource_manager()

        manager.request_access("task", priority=PRIORITY_SSTV, frequency=145.800, gain=30.0)

        # Refresh with new params
        granted, _ = manager.request_access(
            "task", priority=PRIORITY_SSTV, frequency=145.900, gain=35.0
        )

        assert granted is True
        active = manager.get_active_task()
        assert active["frequency"] == 145.900
        assert active["gain"] == 35.0
