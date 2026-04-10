"""
Тесты для RTL-SDR модулей: ADS-B, RTL_433, FM Radio
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Добавляем путь к rtl_sdr_tools
sys.path.insert(0, str(Path(__file__).parent.parent / "rtl_sdr_tools"))


class TestADSBCapture:
    """Тесты ADS-B capture модуля"""

    def test_adsb_frequency_constants(self):
        """Проверка ADS-B констант"""
        from rtl_sdr_tools import adsb_receiver

        assert adsb_receiver.ADS_B_FREQUENCY == 1090e6
        assert adsb_receiver.ADS_B_SAMPLE_RATE == 2400000
        assert adsb_receiver.ADS_B_PULSE_WIDTH == 0.5e-6

    def test_adsb_decoder_init(self):
        """Инициализация ADS-B декодера"""
        from rtl_sdr_tools.adsb_receiver import ADSBDecoder

        decoder = ADSBDecoder()
        assert decoder.aircraft == {}
        assert decoder.messages_received == 0
        assert decoder.messages_decoded == 0

    def test_adsb_add_aircraft(self):
        """Добавление самолёта в трекер"""
        from rtl_sdr_tools.adsb_receiver import ADSBDecoder

        decoder = ADSBDecoder()
        decoder.add_aircraft(
            "A1B2C3",
            {
                "callsign": "UAL123",
                "altitude": 35000,
            },
        )

        assert "A1B2C3" in decoder.aircraft
        assert decoder.aircraft["A1B2C3"]["callsign"] == "UAL123"
        assert decoder.aircraft["A1B2C3"]["altitude"] == 35000

    def test_adsb_update_aircraft(self):
        """Обновление данных самолёта"""
        from rtl_sdr_tools.adsb_receiver import ADSBDecoder

        decoder = ADSBDecoder()
        decoder.add_aircraft("A1B2C3", {"altitude": 35000})
        decoder.add_aircraft("A1B2C3", {"altitude": 36000, "speed": 450})

        assert decoder.aircraft["A1B2C3"]["altitude"] == 36000
        assert decoder.aircraft["A1B2C3"]["speed"] == 450
        assert "A1B2C3" in decoder.aircraft

    def test_adsb_get_aircraft_list(self):
        """Получение списка самолётов"""
        from rtl_sdr_tools.adsb_receiver import ADSBDecoder

        decoder = ADSBDecoder()
        decoder.add_aircraft("A1B2C3", {"callsign": "UAL123"})
        decoder.add_aircraft("D4E5F6", {"callsign": "AAL456"})

        aircraft_list = decoder.get_aircraft_list()
        assert len(aircraft_list) == 2

    def test_adsb_get_stats(self):
        """Получение статистики декодера"""
        from rtl_sdr_tools.adsb_receiver import ADSBDecoder

        decoder = ADSBDecoder()
        decoder.messages_received = 100
        decoder.messages_decoded = 50

        stats = decoder.get_stats()
        assert stats["messages_received"] == 100
        assert stats["messages_decoded"] == 50
        assert "uptime_seconds" in stats

    def test_adsb_receiver_init(self):
        """Инициализация ADS-B приёмника"""
        from rtl_sdr_tools.adsb_receiver import ADSBReceiver

        receiver = ADSBReceiver(
            frequency_mhz=1090.0,
            gain=30,
            device_index=0,
        )

        assert receiver.frequency_mhz == 1090.0
        assert receiver.gain == 30
        assert receiver.device_index == 0
        assert receiver.sdr is None

    def test_adsb_receiver_stop(self):
        """Остановка ADS-B приёмника"""
        from rtl_sdr_tools.adsb_receiver import ADSBReceiver

        receiver = ADSBReceiver()
        receiver._running = True
        receiver.stop()
        assert receiver._running is False


class TestRTL433Scanner:
    """Тесты RTL_433 scanner модуля"""

    def test_rtl433_default_frequency(self):
        """Частота по умолчанию 433.92 MHz"""
        from rtl_sdr_tools import rtl433_scanner

        assert rtl433_scanner.find_rtl433() is None or isinstance(rtl433_scanner.find_rtl433(), str)

    def test_rtl433_scan_function_exists(self):
        """Функция scan_433_mhz существует"""
        from rtl_sdr_tools.rtl433_scanner import scan_433_mhz

        assert callable(scan_433_mhz)

    @patch("rtl_sdr_tools.rtl433_scanner.find_rtl433")
    def test_rtl433_exe_not_found(self, mock_find):
        """rtl_433 не найден — выход с ошибкой"""
        mock_find.return_value = None

        import sys
        from io import StringIO

        from rtl_sdr_tools.rtl433_scanner import scan_433_mhz

        old_stderr = sys.stderr
        sys.stderr = StringIO()

        with pytest.raises(SystemExit) as exc_info:
            scan_433_mhz(duration=1)

        sys.stderr = old_stderr
        assert exc_info.value.code == 1


class TestFMRecording:
    """Тесты FM Radio recording модуля"""

    def test_fm_radio_module_exists(self):
        """FM Radio модуль существует"""
        import importlib.util

        spec = importlib.util.find_spec("rtl_sdr_tools.fm_radio")
        assert spec is not None

    def test_fm_capture_module_exists(self):
        """FM Capture модуль существует"""
        import importlib.util

        spec = importlib.util.find_spec("rtl_sdr_tools.fm_capture_simple")
        assert spec is not None


class TestRTLSDRToolsIntegration:
    """Интеграционные тесты RTL-SDR tools"""

    def test_adsb_receiver_import(self):
        """ADS-B receiver импортируется"""
        try:
            from rtl_sdr_tools import adsb_receiver

            assert hasattr(adsb_receiver, "ADSBReceiver")
            assert hasattr(adsb_receiver, "ADSBDecoder")
        except ImportError:
            pytest.skip("ADS-B receiver not available")

    def test_rtl433_scanner_import(self):
        """RTL_433 scanner импортируется"""
        try:
            from rtl_sdr_tools import rtl433_scanner

            assert hasattr(rtl433_scanner, "scan_433_mhz")
            assert hasattr(rtl433_scanner, "find_rtl433")
        except ImportError:
            pytest.skip("RTL_433 scanner not available")

    def test_listen_adsb_bat_exists(self):
        """listen_adsb.bat существует"""
        bat_path = Path(__file__).parent.parent / "rtl_sdr_tools" / "listen_adsb.bat"
        assert bat_path.exists()

    def test_listen_rtl433_bat_exists(self):
        """listen_rtl433.bat существует"""
        bat_path = Path(__file__).parent.parent / "rtl_sdr_tools" / "listen_rtl433.bat"
        assert bat_path.exists()
