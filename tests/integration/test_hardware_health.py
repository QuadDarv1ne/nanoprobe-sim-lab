#!/usr/bin/env python3
"""
Unit tests for HardwareHealthChecker from utils.sdr.hardware_health
"""
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

# Import the module under test
from utils.sdr.hardware_health import HardwareHealthChecker


class TestHardwareHealthChecker:
    """Test HardwareHealthChecker class."""

    @pytest.fixture(autouse=True)
    def setup_checker(self):
        """Setup the hardware health checker."""
        self.checker = HardwareHealthChecker(device_index=0)
        yield

    def test_initialization(self):
        """Test HardwareHealthChecker initialization."""
        assert self.checker.device_index == 0
        assert self.checker._last_diagnostic == {}
        assert self.checker._dropped_sample_counts == []

    def test_initialization_with_device_index(self):
        """Test initialization with custom device index."""
        checker = HardwareHealthChecker(device_index=2)
        assert checker.device_index == 2

    def test_parse_temperature_valid(self):
        """Test parsing valid temperature from output."""
        output = "rtl_test: trying device 0\nTemp: 42.5 C\n"
        temp = self.checker._parse_temperature(output)
        assert temp == 42.5

    def test_parse_temperature_negative(self):
        """Test parsing negative temperature."""
        output = "temperature -5.2 degrees"
        temp = self.checker._parse_temperature(output)
        assert temp == -5.2

    def test_parse_temperature_no_match(self):
        """Test parsing when no temperature found."""
        output = "No temperature data available"
        temp = self.checker._parse_temperature(output)
        assert temp is None

    def test_parse_temperature_empty(self):
        """Test parsing empty output."""
        temp = self.checker._parse_temperature("")
        assert temp is None

    def test_temperature_status_ok(self):
        """Test temperature status for normal range."""
        status = self.checker._temperature_status(25.0)
        assert status == "ok"

    def test_temperature_status_warning(self):
        """Test temperature status for warning range."""
        status = self.checker._temperature_status(45.0)
        assert status == "warning"

    def test_temperature_status_critical(self):
        """Test temperature status for critical range."""
        status = self.checker._temperature_status(60.0)
        assert status == "critical"

    def test_check_temperature_mocked(self):
        """Test temperature check with mocked subprocess."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stderr="Temp: 35.5 C", stdout="", returncode=0
            )

            result = self.checker.check_temperature()

            assert "temperature_c" in result
            assert result["temperature_c"] == 35.5
            assert result["status"] == "ok"

    def test_check_temperature_unavailable(self):
        """Test temperature check when unavailable."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("rtl_test not found")

            with patch.object(self.checker, "_read_sysfs_temperature") as mock_sysfs:
                mock_sysfs.side_effect = Exception("sysfs failed")

                result = self.checker.check_temperature()

                assert result["temperature_c"] is None
                assert result["status"] == "unavailable"

    def test_read_sysfs_temperature_success(self):
        """Test reading temperature from sysfs."""
        with patch("builtins.open", MagicMock()) as mock_open:
            mock_open.return_value.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value="45000")))

            temp = self.checker._read_sysfs_temperature()

            assert temp == 45.0

    def test_read_sysfs_temperature_failure(self):
        """Test sysfs temperature read failure."""
        with patch("builtins.open", MagicMock(side_effect=FileNotFoundError)):
            temp = self.checker._read_sysfs_temperature()
            assert temp is None

    def test_check_eeprom_mocked(self):
        """Test EEPROM check with mocked subprocess."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="EEPROM: OK", stderr="", returncode=0
            )

            result = self.checker.check_eeprom()

            assert "valid" in result
            assert result["valid"] is True

    def test_check_eeprom_failed(self):
        """Test EEPROM check failure."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="EEPROM: ERROR", stderr="", returncode=1
            )

            result = self.checker.check_eeprom()

            assert result["valid"] is False

    def test_check_dropped_samples(self):
        """Test checking dropped samples."""
        # Record initial count
        self.checker._dropped_sample_counts = [100]

        # Check with no drops
        result = self.checker.check_dropped_samples(1000)

        assert "dropped" in result
        assert result["total_samples"] == 1000

    def test_run_full_diagnostic(self):
        """Test running full diagnostic."""
        with patch.object(self.checker, "check_temperature") as mock_temp:
            with patch.object(self.checker, "check_eeprom") as mock_eeprom:
                with patch.object(self.checker, "check_dropped_samples") as mock_drops:
                    mock_temp.return_value = {"temperature_c": 35.0, "status": "ok"}
                    mock_eeprom.return_value = {"valid": True}
                    mock_drops.return_value = {"dropped": 0, "total_samples": 1000}

                    result = self.checker.run_full_diagnostic()

                    assert "temperature" in result
                    assert "eeprom" in result
                    assert "dropped_samples" in result
                    assert result["overall_status"] == "ok"

    def test_get_health_status_ok(self):
        """Test health status when everything is ok."""
        self.checker._last_diagnostic = {
            "temperature": {"temperature_c": 30.0, "status": "ok"},
            "eeprom": {"valid": True},
            "dropped_samples": {"dropped": 0},
        }

        status = self.checker.get_health_status()

        assert status["overall"] == "ok"

    def test_get_health_status_warning(self):
        """Test health status with warning."""
        self.checker._last_diagnostic = {
            "temperature": {"temperature_c": 45.0, "status": "warning"},
            "eeprom": {"valid": True},
            "dropped_samples": {"dropped": 10},
        }

        status = self.checker.get_health_status()

        assert status["overall"] == "warning"

    def test_get_health_status_critical(self):
        """Test health status with critical issue."""
        self.checker._last_diagnostic = {
            "temperature": {"temperature_c": 65.0, "status": "critical"},
            "eeprom": {"valid": False},
            "dropped_samples": {"dropped": 1000},
        }

        status = self.checker.get_health_status()

        assert status["overall"] == "critical"

    def test_get_health_status_unknown(self):
        """Test health status when no diagnostics run."""
        self.checker._last_diagnostic = {}

        status = self.checker.get_health_status()

        assert status["overall"] == "unknown"


class TestHardwareHealthCheckerEdgeCases:
    """Test edge cases for HardwareHealthChecker."""

    def test_parse_temperature_various_formats(self):
        """Test parsing various temperature formats."""
        checker = HardwareHealthChecker()

        test_cases = [
            ("Temp: 25.5 C", 25.5),
            ("temperature: -10.2", -10.2),
            ("Temperature = 42", 42.0),
            ("temp 33.3 degrees", 33.3),
            ("no temp here", None),
        ]

        for output, expected in test_cases:
            result = checker._parse_temperature(output)
            assert result == expected, f"Failed for '{output}': expected {expected}, got {result}"

    def test_temperature_boundary_values(self):
        """Test temperature status at boundary values."""
        checker = HardwareHealthChecker()

        # OK range: < 40
        assert checker._temperature_status(39.9) == "ok"
        assert checker._temperature_status(0) == "ok"
        assert checker._temperature_status(-10) == "ok"

        # Warning range: 40-55
        assert checker._temperature_status(40.0) == "warning"
        assert checker._temperature_status(45.0) == "warning"
        assert checker._temperature_status(55.0) == "warning"

        # Critical: > 55
        assert checker._temperature_status(55.1) == "critical"
        assert checker._temperature_status(70.0) == "critical"
