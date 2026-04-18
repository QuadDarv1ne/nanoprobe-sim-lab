"""
Tests for RTL-SDR Auto Calibration Module

Тесты для модуля автоматической калибровки PPM.
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from utils.sdr.rtl_sdr_auto_calibration import RTLSDRAutoCalibration, get_rtl_sdr_devices


class TestRTLSDRAutoCalibration:
    """Тесты для RTLSDRAutoCalibration."""

    @pytest.fixture
    def temp_calibration_file(self, tmp_path):
        """Создание временного файла калибровки."""
        cal_file = tmp_path / "device_calibration.json"
        cal_file.write_text("{}")
        return str(cal_file)

    @pytest.fixture
    def calibrator(self, temp_calibration_file):
        """Создание экземпляра калибратора с временным файлом."""
        return RTLSDRAutoCalibration(
            device_index=0,
            calibration_file=temp_calibration_file,
        )

    def test_init(self, calibrator):
        """Тест инициализации."""
        assert calibrator.device_index == 0
        assert calibrator.calibration_data == {}

    def test_load_calibration_empty(self, calibrator):
        """Тест загрузки пустой калибровки."""
        assert calibrator.get_calibration() is None

    def test_load_calibration_with_data(self, temp_calibration_file):
        """Тест загрузки калибровки с данными."""
        data = {
            "0": {
                "ppm": 42.5,
                "method": "rtl_test",
                "confidence": 0.8,
                "timestamp": datetime.now().isoformat(),
            }
        }
        with open(temp_calibration_file, "w") as f:
            json.dump(data, f)

        calibrator = RTLSDRAutoCalibration(
            device_index=0,
            calibration_file=temp_calibration_file,
        )
        assert calibrator.get_calibration() == 42.5

    def test_save_calibration(self, calibrator, temp_calibration_file):
        """Тест сохранения калибровки."""
        calibrator._save_calibration(ppm=45.2, method="rtl_test", confidence=0.85)

        assert os.path.exists(temp_calibration_file)
        with open(temp_calibration_file, "r") as f:
            data = json.load(f)

        assert "0" in data
        assert data["0"]["ppm"] == 45.2
        assert data["0"]["method"] == "rtl_test"
        assert data["0"]["confidence"] == 0.85

    def test_get_calibration(self, calibrator):
        """Тест получения калибровки."""
        assert calibrator.get_calibration() is None

        calibrator._save_calibration(ppm=30.0, method="signal", confidence=0.7)
        assert calibrator.get_calibration() == 30.0

    def test_get_calibration_info(self, calibrator):
        """Тест получения информации о калибровке."""
        info = calibrator.get_calibration_info()
        assert info["device_index"] == 0
        assert info["has_calibration"] is False
        assert info["ppm"] is None

    def test_is_calibration_valid_no_calibration(self, calibrator):
        """Тест проверки валидности без калибровки."""
        assert calibrator.is_calibration_valid() is False

    def test_is_calibration_valid_with_recent_calibration(self, calibrator):
        """Тест проверки валидности с недавней калибровкой."""
        calibrator._save_calibration(ppm=25.0, method="rtl_test", confidence=0.9)
        assert calibrator.is_calibration_valid() is True

    def test_reset_calibration(self, calibrator, temp_calibration_file):
        """Тест сброса калибровки."""
        calibrator._save_calibration(ppm=50.0, method="rtl_test", confidence=0.8)
        assert calibrator.get_calibration() == 50.0

        calibrator.reset_calibration()
        assert calibrator.get_calibration() is None

    @patch("subprocess.run")
    def test_calibrate_with_rtl_test_success(self, mock_run, calibrator):
        """Тест успешной калибровки через rtl_test."""
        mock_run.return_value = MagicMock(
            stdout="",
            stderr="real-time PPM: 35.2",
        )

        ppm = calibrator.calibrate_with_rtl_test()
        assert ppm == 35.2

    @patch("subprocess.run")
    def test_calibrate_with_rtl_test_timeout(self, mock_run, calibrator):
        """Тест таймаута при калибровке rtl_test."""
        mock_run.side_effect = Exception("Timeout")

        ppm = calibrator.calibrate_with_rtl_test()
        assert ppm is None

    @patch("subprocess.run")
    def test_calibrate_with_rtl_test_not_found(self, mock_run, calibrator):
        """Тест отсутствия rtl_test."""
        mock_run.side_effect = FileNotFoundError("rtl_test not found")

        ppm = calibrator.calibrate_with_rtl_test()
        assert ppm is None

    @patch("subprocess.run")
    def test_calibrate_with_signal_success(self, mock_run, calibrator):
        """Тест успешной калибровки по сигналу."""
        mock_run.return_value = MagicMock(
            stdout=b"dummy iq data",
            stderr="",
        )

        # Mock _estimate_ppm_from_signal
        with patch.object(calibrator, "_estimate_ppm_from_signal", return_value=28.5):
            ppm = calibrator.calibrate_with_signal(reference_freq_mhz=100.0)
            assert ppm == 28.5

    def test_calibrate_auto_rtl_test_first(self, calibrator):
        """Тест авто-калибровки с приоритетом rtl_test."""
        # Mock calibrate_with_rtl_test
        with patch.object(calibrator, "calibrate_with_rtl_test", return_value=40.0):
            ppm = calibrator.calibrate_auto()
            assert ppm == 40.0

    def test_calibrate_auto_signal_fallback(self, calibrator):
        """Тест авто-калибровки с фоллбэком на signal."""
        with patch.object(calibrator, "calibrate_with_rtl_test", return_value=None):
            with patch.object(
                calibrator,
                "calibrate_with_signal",
                return_value=32.0,
            ):
                ppm = calibrator.calibrate_auto(reference_freq_mhz=100.0)
                assert ppm == 32.0

    def test_calibrate_auto_all_fail(self, calibrator):
        """Тест авто-калибровки при отказе всех методов."""
        with patch.object(calibrator, "calibrate_with_rtl_test", return_value=None):
            ppm = calibrator.calibrate_auto()
            assert ppm is None

    def test_add_calibration_handler(self, calibrator):
        """Тест добавления обработчика калибровки."""
        handler = MagicMock()
        calibrator.add_calibration_handler(handler)
        assert handler in calibrator._calibration_handlers

    def test_parse_ppm_patterns(self, calibrator):
        """Тест парсинга различных паттернов PPM."""
        test_cases = [
            ("real-time PPM: 42.3", 42.3),
            ("diff: +35.2 ppm", 35.2),
            ("diff: -12.5 ppm", -12.5),
            ("PPM offset: 28.0", 28.0),
            ("actual: 2400123 diff: +15.5 ppm", 15.5),
        ]

        for output, expected in test_cases:
            result = calibrator._parse_ppm_from_rtl_test(output)
            assert result == expected, f"Failed for: {output}"

    def test_parse_ppm_no_match(self, calibrator):
        """Тест парсинга без совпадений."""
        result = calibrator._parse_ppm_from_rtl_test("no ppm data here")
        assert result is None


class TestGetRtlSdrDevices:
    """Тесты для функции get_rtl_sdr_devices."""

    @patch("subprocess.run")
    def test_get_devices_success(self, mock_run):
        """Тест успешного получения списка устройств."""
        mock_run.return_value = MagicMock(
            stdout="Manufacturer: Realtek\nProduct: RTL2838\nSerial: 00000001\n",
            stderr="",
        )

        devices = get_rtl_sdr_devices()
        assert len(devices) == 1
        assert devices[0]["manufacturer"] == "Realtek"
        assert devices[0]["product"] == "RTL2838"

    @patch("subprocess.run")
    def test_get_devices_not_found(self, mock_run):
        """Тест отсутствия rtl_test."""
        mock_run.side_effect = FileNotFoundError("rtl_test not found")

        devices = get_rtl_sdr_devices()
        assert devices == []

    @patch("subprocess.run")
    def test_get_devices_error(self, mock_run):
        """Тест ошибки при получении устройств."""
        mock_run.side_effect = Exception("USB error")

        devices = get_rtl_sdr_devices()
        assert devices == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
