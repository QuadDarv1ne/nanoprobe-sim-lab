"""Тесты для utils/monitoring/system_health_monitor.py."""

import json
import os
import queue
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from utils.monitoring.system_health_monitor import HealthAlert, HealthMetric, SystemHealthMonitor

# ==================== Fixtures ====================


@pytest.fixture
def temp_dir(tmp_path):
    """Создает временную директорию для тестов."""
    return str(tmp_path / "health_reports")


@pytest.fixture
def mock_psutil():
    """Фикстура для мока psutil."""
    with patch("utils.monitoring.system_health_monitor.psutil") as mock:
        mock.cpu_percent.return_value = 45.0
        mock.virtual_memory.return_value = MagicMock(
            percent=60.0,
            used=8 * (1024**3),
            available=8 * (1024**3),
            total=16 * (1024**3),
        )
        mock.disk_usage.return_value = MagicMock(
            percent=55.0,
            used=500 * (1024**3),
            free=500 * (1024**3),
        )
        mock.sensors_temperatures.return_value = {}
        mock.pids.return_value = [1, 2, 3, 4, 5]
        mock.net_io_counters.return_value = MagicMock(bytes_sent=1000000, bytes_recv=2000000)
        mock.cpu_count.return_value = 8
        mock.cpu_freq.return_value = MagicMock(current=2.5, min=1.0, max=3.5)
        mock.boot_time.return_value = 1700000000
        yield mock


@pytest.fixture
def monitor(temp_dir, mock_psutil):
    """Создает монитор с временной директорией."""
    return SystemHealthMonitor(output_dir=temp_dir)


# ==================== TestHealthMetric ====================


class TestHealthMetric:
    """Тесты для класса HealthMetric."""

    def test_create_metric_basic(self):
        metric = HealthMetric(
            name="cpu_percent",
            value=45.0,
            unit="%",
            timestamp=datetime.now(timezone.utc),
            severity="info",
            source="test",
        )
        assert metric.name == "cpu_percent"
        assert metric.value == 45.0
        assert metric.unit == "%"
        assert metric.severity == "info"
        assert metric.source == "test"
        assert metric.threshold_low is None
        assert metric.threshold_high is None

    def test_create_metric_with_thresholds(self):
        ts = datetime.now(timezone.utc)
        metric = HealthMetric(
            name="memory_percent",
            value=80.0,
            unit="%",
            timestamp=ts,
            severity="warning",
            source="test",
            threshold_low=70.0,
            threshold_high=90.0,
        )
        assert metric.threshold_low == 70.0
        assert metric.threshold_high == 90.0
        assert metric.value == 80.0

    def test_metric_severity_levels(self):
        ts = datetime.now(timezone.utc)
        severities = ["info", "warning", "error", "critical"]
        for severity in severities:
            metric = HealthMetric(
                name="test",
                value=1.0,
                unit="unit",
                timestamp=ts,
                severity=severity,
                source="test",
            )
            assert metric.severity == severity

    def test_metric_timestamp(self):
        ts = datetime(2026, 4, 13, 12, 0, 0, tzinfo=timezone.utc)
        metric = HealthMetric(
            name="test",
            value=1.0,
            unit="unit",
            timestamp=ts,
            severity="info",
            source="test",
        )
        assert metric.timestamp == ts
        assert metric.timestamp.year == 2026
        assert metric.timestamp.month == 4


# ==================== TestHealthAlert ====================


class TestHealthAlert:
    """Тесты для класса HealthAlert."""

    def test_create_alert_basic(self):
        ts = datetime.now(timezone.utc)
        alert = HealthAlert(
            alert_id="test_001",
            metric_name="cpu_percent",
            current_value=90.0,
            threshold_value=85.0,
            severity="error",
            message="CPU exceeded threshold",
            timestamp=ts,
        )
        assert alert.alert_id == "test_001"
        assert alert.metric_name == "cpu_percent"
        assert alert.current_value == 90.0
        assert alert.threshold_value == 85.0
        assert alert.severity == "error"
        assert alert.message == "CPU exceeded threshold"
        assert alert.resolved is False

    def test_alert_default_resolved(self):
        ts = datetime.now(timezone.utc)
        alert = HealthAlert(
            alert_id="test_002",
            metric_name="memory_percent",
            current_value=80.0,
            threshold_value=75.0,
            severity="warning",
            message="Memory warning",
            timestamp=ts,
        )
        assert alert.resolved is False

    def test_alert_id_format(self):
        ts = datetime.now(timezone.utc)
        alert_id = f"cpu_{ts.strftime('%Y%m%d_%H%M%S')}"
        alert = HealthAlert(
            alert_id=alert_id,
            metric_name="cpu_percent",
            current_value=75.0,
            threshold_value=70.0,
            severity="warning",
            message="CPU warning",
            timestamp=ts,
        )
        assert "cpu_" in alert.alert_id
        assert alert.alert_id.startswith("cpu_")

    def test_alert_severity_levels(self):
        ts = datetime.now(timezone.utc)
        for severity in ["warning", "error", "critical"]:
            alert = HealthAlert(
                alert_id=f"test_{severity}",
                metric_name="test",
                current_value=1.0,
                threshold_value=1.0,
                severity=severity,
                message="Test",
                timestamp=ts,
            )
            assert alert.severity == severity


# ==================== TestSystemHealthMonitorInit ====================


class TestSystemHealthMonitorInit:
    """Тесты для инициализации SystemHealthMonitor."""

    def test_init_creates_output_dir(self, temp_dir):
        monitor = SystemHealthMonitor(output_dir=temp_dir)
        assert monitor is not None
        assert Path(temp_dir).exists()
        assert Path(temp_dir).is_dir()

    def test_init_empty_metrics(self, monitor):
        assert monitor.metrics == []
        assert monitor.current_metrics == {}

    def test_init_empty_alerts(self, monitor):
        assert monitor.alerts == []

    def test_init_health_score_100(self, monitor):
        assert monitor.health_score == 100.0

    def test_init_thresholds_structure(self, monitor):
        expected_keys = [
            "cpu_percent",
            "memory_percent",
            "disk_percent",
            "temperature",
            "process_count",
            "network_latency_ms",
        ]
        for key in expected_keys:
            assert key in monitor.thresholds
            thresholds = monitor.thresholds[key]
            assert "warning" in thresholds
            assert "error" in thresholds
            assert "critical" in thresholds


# ==================== TestAlertHandlers ====================


class TestAlertHandlers:
    """Тесты для обработчиков оповещений."""

    def test_add_alert_handler(self, monitor):
        handler = MagicMock()
        monitor.add_alert_handler(handler)
        assert len(monitor.alert_handlers) == 1
        assert monitor.alert_handlers[0] == handler

    def test_add_notification_channel_email(self, monitor):
        config = {"smtp_server": "smtp.example.com", "from_email": "test@example.com"}
        monitor.add_notification_channel("email", config)
        assert len(monitor.notification_channels) == 1
        assert monitor.notification_channels[0]["type"] == "email"
        assert monitor.notification_channels[0]["config"] == config

    def test_add_notification_channel_webhook(self, monitor):
        config = {"url": "https://hooks.example.com/alert"}
        monitor.add_notification_channel("webhook", config)
        assert len(monitor.notification_channels) == 1
        assert monitor.notification_channels[0]["type"] == "webhook"
        assert monitor.notification_channels[0]["config"] == config

    def test_add_notification_channel_console(self, monitor):
        monitor.add_notification_channel("console", {})
        assert len(monitor.notification_channels) == 1
        assert monitor.notification_channels[0]["type"] == "console"


# ==================== TestGetSystemMetrics ====================


class TestGetSystemMetrics:
    """Тесты для метода get_system_metrics."""

    def test_get_system_metrics_returns_dict(self, monitor):
        metrics = monitor.get_system_metrics()
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

    def test_get_system_metrics_has_cpu(self, monitor):
        metrics = monitor.get_system_metrics()
        assert "cpu_percent" in metrics
        assert metrics["cpu_percent"] == 45.0

    def test_get_system_metrics_has_memory(self, monitor):
        metrics = monitor.get_system_metrics()
        assert "memory_percent" in metrics
        assert metrics["memory_percent"] == 60.0
        assert "memory_used_gb" in metrics
        assert "memory_available_gb" in metrics

    def test_get_system_metrics_has_disk(self, monitor):
        metrics = monitor.get_system_metrics()
        assert "disk_percent" in metrics
        assert metrics["disk_percent"] == 55.0
        assert "disk_used_gb" in metrics
        assert "disk_free_gb" in metrics


# ==================== TestThresholds ====================


class TestThresholds:
    """Тесты для пороговых значений."""

    def test_cpu_thresholds_structure(self, monitor):
        cpu = monitor.thresholds["cpu_percent"]
        assert cpu["warning"] == 70
        assert cpu["error"] == 85
        assert cpu["critical"] == 95
        assert cpu["warning"] < cpu["error"] < cpu["critical"]

    def test_memory_thresholds_structure(self, monitor):
        mem = monitor.thresholds["memory_percent"]
        assert mem["warning"] == 75
        assert mem["error"] == 85
        assert mem["critical"] == 95
        assert mem["warning"] < mem["error"] < mem["critical"]

    def test_disk_thresholds_structure(self, monitor):
        disk = monitor.thresholds["disk_percent"]
        assert disk["warning"] == 80
        assert disk["error"] == 90
        assert disk["critical"] == 95
        assert disk["warning"] < disk["error"] < disk["critical"]

    def test_custom_thresholds(self, temp_dir, mock_psutil):
        monitor = SystemHealthMonitor(output_dir=temp_dir)
        monitor.thresholds["cpu_percent"] = {"warning": 50, "error": 70, "critical": 90}
        assert monitor.thresholds["cpu_percent"]["warning"] == 50
        assert monitor.thresholds["cpu_percent"]["error"] == 70
        assert monitor.thresholds["cpu_percent"]["critical"] == 90


# ==================== TestAlertQueue ====================


class TestAlertQueue:
    """Тесты для очереди оповещений."""

    def test_alert_queue_initially_empty(self, monitor):
        assert monitor.alert_queue.empty()

    def test_alert_queue_put_get(self, monitor):
        ts = datetime.now(timezone.utc)
        alert = HealthAlert(
            alert_id="test_001",
            metric_name="cpu_percent",
            current_value=90.0,
            threshold_value=85.0,
            severity="error",
            message="CPU exceeded",
            timestamp=ts,
        )
        monitor.alert_queue.put(alert)
        assert not monitor.alert_queue.empty()
        retrieved = monitor.alert_queue.get()
        assert retrieved == alert
        assert retrieved.metric_name == "cpu_percent"

    def test_alert_queue_multiple_items(self, monitor):
        ts = datetime.now(timezone.utc)
        alerts = []
        for i in range(3):
            alert = HealthAlert(
                alert_id=f"test_{i}",
                metric_name="cpu_percent",
                current_value=90.0 + i,
                threshold_value=85.0,
                severity="error",
                message=f"Alert {i}",
                timestamp=ts,
            )
            alerts.append(alert)
            monitor.alert_queue.put(alert)

        assert monitor.alert_queue.qsize() == 3
        for expected in alerts:
            retrieved = monitor.alert_queue.get()
            assert retrieved.alert_id == expected.alert_id


# ==================== TestEdgeCases ====================


class TestEdgeCases:
    """Тесты для граничных случаев."""

    def test_monitor_without_external_services(self, temp_dir, mock_psutil):
        monitor = SystemHealthMonitor(output_dir=temp_dir)
        assert len(monitor.alert_handlers) == 0
        assert len(monitor.notification_channels) == 0
        assert monitor.active is False

    def test_monitor_with_custom_output_dir(self, tmp_path):
        custom_dir = str(tmp_path / "custom_reports")
        monitor = SystemHealthMonitor(output_dir=custom_dir)
        assert Path(custom_dir).exists()
        assert Path(custom_dir).is_dir()
        assert str(monitor.output_dir) == custom_dir
