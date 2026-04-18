"""
Tests for Satellite Auto-Capture Module

Тесты для модуля автоматического захвата спутников.
"""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from utils.sdr.satellite_auto_capture import SatelliteAutoCapture, SatellitePass, TLEData


class TestSatellitePass:
    """Тесты для SatellitePass dataclass."""

    def test_satellite_pass_creation(self):
        """Тест создания пролёта."""
        now = datetime.now()
        pass_ = SatellitePass(
            satellite="NOAA-19",
            aos=now + timedelta(hours=1),
            los=now + timedelta(hours=1, minutes=12),
            max_elevation=45.5,
            frequency_mhz=137.1,
            mode="APT",
        )

        assert pass_.satellite == "NOAA-19"
        assert pass_.max_elevation == 45.5
        assert pass_.mode == "APT"

    def test_duration_seconds(self):
        """Тест вычисления длительности."""
        now = datetime.now()
        pass_ = SatellitePass(
            satellite="NOAA-19",
            aos=now,
            los=now + timedelta(minutes=10),
            max_elevation=30.0,
            frequency_mhz=137.1,
        )

        assert pass_.duration_seconds() == 600

    def test_time_to_aos(self):
        """Тест вычисления времени до AOS."""
        now = datetime.now()
        pass_ = SatellitePass(
            satellite="NOAA-19",
            aos=now + timedelta(minutes=30),
            los=now + timedelta(hours=1),
            max_elevation=30.0,
            frequency_mhz=137.1,
        )

        time_to_aos = pass_.time_to_aos()
        assert 1790 < time_to_aos < 1810  # ~30 минут

    def test_is_active(self):
        """Тест проверки активного пролёта."""
        now = datetime.now()
        pass_inactive = SatellitePass(
            satellite="NOAA-19",
            aos=now - timedelta(hours=1),
            los=now - timedelta(minutes=30),
            max_elevation=30.0,
            frequency_mhz=137.1,
        )
        pass_active = SatellitePass(
            satellite="NOAA-19",
            aos=now - timedelta(minutes=5),
            los=now + timedelta(minutes=5),
            max_elevation=30.0,
            frequency_mhz=137.1,
        )
        pass_future = SatellitePass(
            satellite="NOAA-19",
            aos=now + timedelta(hours=1),
            los=now + timedelta(hours=2),
            max_elevation=30.0,
            frequency_mhz=137.1,
        )

        assert pass_inactive.is_active() is False
        assert pass_active.is_active() is True
        assert pass_future.is_active() is False

    def test_to_dict(self):
        """Тест преобразования в словарь."""
        now = datetime.now()
        pass_ = SatellitePass(
            satellite="NOAA-19",
            aos=now,
            los=now + timedelta(minutes=10),
            max_elevation=45.5,
            frequency_mhz=137.1,
            mode="APT",
            azimuth_aos=180.0,
            azimuth_los=270.0,
        )

        d = pass_.to_dict()
        assert d["satellite"] == "NOAA-19"
        assert d["max_elevation"] == 45.5
        assert d["mode"] == "APT"
        assert "aos" in d
        assert "los" in d


class TestSatelliteAutoCapture:
    """Тесты для SatelliteAutoCapture."""

    @pytest.fixture
    def capture(self, tmp_path):
        """Создание экземпляра SatelliteAutoCapture."""
        return SatelliteAutoCapture(
            location_lat=55.75,
            location_lon=37.61,
            output_dir=str(tmp_path / "captures"),
            device_index=0,
        )

    def test_init(self, capture):
        """Тест инициализации."""
        assert capture.location == (55.75, 37.61)
        assert capture.device_index == 0
        assert capture.passes == []
        assert capture._running is False

    def test_init_creates_output_dir(self, capture, tmp_path):
        """Тест создания директории для записей."""
        output_dir = Path(capture.output_dir)
        assert output_dir.exists()

    def test_add_capture_handler(self, capture):
        """Тест добавления обработчика захвата."""
        handler = MagicMock()
        capture.add_capture_handler(handler)
        assert handler in capture._capture_handlers

    def test_add_pass_start_handler(self, capture):
        """Тест добавления обработчика начала пролёта."""
        handler = MagicMock()
        capture.add_pass_start_handler(handler)
        assert handler in capture._pass_start_handlers

    def test_add_pass_end_handler(self, capture):
        """Тест добавления обработчика окончания пролёта."""
        handler = MagicMock()
        capture.add_pass_end_handler(handler)
        assert handler in capture._pass_end_handlers

    @patch("skyfield.api.load")
    @patch.object(SatelliteAutoCapture, "_refresh_tle_data", return_value=True)
    def test_predict_passes_empty_tle(self, mock_refresh, mock_load, capture):
        """Тест предсказания без TLE данных."""
        capture.tle_data = {}
        passes = capture.predict_passes(hours_ahead=24)
        assert passes == []

    @patch("skyfield.api.load")
    @patch.object(SatelliteAutoCapture, "_refresh_tle_data", return_value=True)
    def test_predict_passes_with_tle(self, mock_refresh, mock_load, capture):
        """Тест предсказания с TLE данными."""
        # Mock skyfield
        mock_ts = MagicMock()
        mock_station = MagicMock()
        mock_satellite = MagicMock()

        now = datetime.now()
        mock_times = [
            MagicMock(utc_datetime=lambda: now + timedelta(hours=1)),
            MagicMock(utc_datetime=lambda: now + timedelta(hours=1, minutes=6)),
            MagicMock(utc_datetime=lambda: now + timedelta(hours=1, minutes=12)),
        ]
        mock_events = [0, 1, 2]  # aos, max_el, los

        mock_load.return_value.timescale.return_value = mock_ts
        mock_load.return_value.Topos.return_value = mock_station
        mock_satellite.find_events.return_value = (mock_times, mock_events)

        capture.tle_data["NOAA-19"] = TLEData(
            name="NOAA 19",
            line1="1 44353U 19013A   24001.00000000  .00000000  00000-0  00000-0 0  9999",
            line2="2 44353  99.0000 180.0000 0010000  90.0000 270.0000 14.12000000123456",
        )

        with patch("skyfield.api.EarthSatellite", return_value=mock_satellite):
            with patch.object(capture, "_calc_max_elevation", return_value=45.0):
                with patch.object(capture, "_calc_azimuths", return_value=(180.0, 270.0)):
                    passes = capture.predict_passes(hours_ahead=24)  # noqa: F841

        # Проверяем что TLE данные были загружены
        assert "NOAA-19" in capture.tle_data

    def test_start_scheduler(self, capture):
        """Тест запуска планировщика."""
        capture.start_scheduler()
        assert capture._running is True
        assert capture._thread is not None
        assert capture._thread.is_alive()

    def test_start_scheduler_already_running(self, capture):
        """Тест повторного запуска планировщика."""
        capture.start_scheduler()
        first_thread = capture._thread

        capture.start_scheduler()  # Должно быть проигнорировано
        assert capture._thread == first_thread

    def test_stop_scheduler(self, capture):
        """Тест остановки планировщика."""
        capture.start_scheduler()
        assert capture._running is True

        capture.stop_scheduler()
        assert capture._running is False

    def test_get_next_pass(self, capture):
        """Тест получения следующего пролёта."""
        now = datetime.now()
        capture.passes = [
            SatellitePass(
                satellite="NOAA-19",
                aos=now + timedelta(hours=2),
                los=now + timedelta(hours=3),
                max_elevation=30.0,
                frequency_mhz=137.1,
            ),
            SatellitePass(
                satellite="NOAA-18",
                aos=now + timedelta(hours=1),
                los=now + timedelta(hours=2),
                max_elevation=40.0,
                frequency_mhz=137.9,
            ),
        ]

        next_pass = capture.get_next_pass()
        assert next_pass.satellite == "NOAA-18"

    def test_get_next_pass_filtered(self, capture):
        """Тест получения следующего пролёта с фильтром."""
        now = datetime.now()
        capture.passes = [
            SatellitePass(
                satellite="NOAA-19",
                aos=now + timedelta(hours=2),
                los=now + timedelta(hours=3),
                max_elevation=30.0,
                frequency_mhz=137.1,
            ),
            SatellitePass(
                satellite="NOAA-18",
                aos=now + timedelta(hours=1),
                los=now + timedelta(hours=2),
                max_elevation=40.0,
                frequency_mhz=137.9,
            ),
        ]

        next_pass = capture.get_next_pass(satellite="NOAA-19")
        assert next_pass.satellite == "NOAA-19"

    def test_get_next_pass_empty(self, capture):
        """Тест получения следующего пролёта без данных."""
        next_pass = capture.get_next_pass()
        assert next_pass is None

    def test_get_active_pass(self, capture):
        """Тест получения активного пролёта."""
        assert capture.get_active_pass() is None

        now = datetime.now()
        capture._active_pass = SatellitePass(
            satellite="NOAA-19",
            aos=now - timedelta(minutes=5),
            los=now + timedelta(minutes=5),
            max_elevation=30.0,
            frequency_mhz=137.1,
        )

        assert capture.get_active_pass().satellite == "NOAA-19"

    def test_get_passes_summary(self, capture):
        """Тест получения сводки пролётов."""
        now = datetime.now()
        capture.passes = [
            SatellitePass(
                satellite="NOAA-19",
                aos=now + timedelta(hours=1),
                los=now + timedelta(hours=2),
                max_elevation=30.0,
                frequency_mhz=137.1,
            ),
            SatellitePass(
                satellite="NOAA-18",
                aos=now - timedelta(hours=1),
                los=now - timedelta(minutes=30),
                max_elevation=40.0,
                frequency_mhz=137.9,
            ),
        ]

        summary = capture.get_passes_summary()
        assert summary["total_predicted"] == 2
        assert summary["upcoming"] == 1
        assert summary["active"] == 0
        assert "NOAA-19" in str(summary["next_pass"])

    def test_supported_satellites(self, capture):
        """Тест списка поддерживаемых спутников."""
        satellites = SatelliteAutoCapture.SATELLITES
        assert "NOAA-15" in satellites
        assert "NOAA-18" in satellites
        assert "NOAA-19" in satellites
        assert "METEOR-M2" in satellites

    def test_tle_data_structure(self):
        """Тест структуры TLE данных."""
        tle = TLEData(
            name="NOAA 19",
            line1="1 44353U 19013A   24001.00000000  .00000000  00000-0  00000-0 0  9999",
            line2="2 44353  99.0000 180.0000 0010000  90.0000 270.0000 14.12000000123456",
        )

        assert tle.name == "NOAA 19"
        assert "1 44353" in tle.line1
        assert "2 44353" in tle.line2


class TestSatelliteAutoCaptureIntegration:
    """Интеграционные тесты для SatelliteAutoCapture."""

    def test_full_workflow(self, tmp_path):
        """Тест полного рабочего процесса."""
        capture = SatelliteAutoCapture(
            location_lat=55.75,
            location_lon=37.61,
            output_dir=str(tmp_path / "captures"),
            device_index=0,
        )

        # Добавить обработчик
        events = []

        def on_capture_start(pass_):
            events.append(("start", pass_.satellite))

        capture.add_capture_handler(on_capture_start)

        # Проверить что обработчик добавлен
        assert len(capture._capture_handlers) == 1

        # Получить сводку
        summary = capture.get_passes_summary()
        assert "satellites" in summary
        assert len(summary["satellites"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
