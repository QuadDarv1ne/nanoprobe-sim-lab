"""
SSTV Health Check endpoints

Проверка здоровья SSTV модуля, диагностика устройств,
деградация функциональности.
"""

import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import psutil
from fastapi import APIRouter

from api.error_handlers import ServiceUnavailableError
from api.routes.sstv.helpers import REDIS_AVAILABLE, SSTV_AVAILABLE, get_app_state, tracker_module

router = APIRouter()


@router.get("/health")
async def sstv_health_check():
    """Проверка здоровья SSTV модуля."""
    recording_process = get_app_state("recording_process")

    rtl_fm_available = shutil.which("rtl_fm") is not None

    tle_file = Path("data/tle_data.json")
    tle_age_hours = None
    if tle_file.exists():
        tle_age_hours = round((time.time() - tle_file.stat().st_mtime) / 3600, 1)

    status = {
        "sstv_decoder": "available" if SSTV_AVAILABLE else "unavailable",
        "satellite_tracker": ("available" if tracker_module is not None else "unavailable"),
        "redis_cache": "available" if REDIS_AVAILABLE else "unavailable",
        "rtl_sdr_recording": ("idle" if recording_process is None else "recording"),
        "rtl_fm_binary": "found" if rtl_fm_available else "not_found",
        "tle_cache_age_hours": tle_age_hours,
        "tle_fresh": tle_age_hours is not None and tle_age_hours < 12,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    all_ok = SSTV_AVAILABLE and tracker_module is not None

    return {"status": "healthy" if all_ok else "degraded", "components": status}


@router.get("/health/extended")
async def sstv_extended_health_check():
    """
    Расширенная проверка здоровья SSTV/RTL-SDR модуля.

    Включает:
    - Device status и serial
    - Signal strength
    - Memory usage
    - Error rate
    - Graceful degradation status
    """
    recording_process = get_app_state("recording_process")

    rtl_fm_available = shutil.which("rtl_fm") is not None
    rtl_test_available = shutil.which("rtl_test") is not None

    device_info = None
    device_status = "unknown"

    try:
        from rtlsdr import RtlSdr

        if hasattr(RtlSdr, "get_device_count"):
            num_devices = RtlSdr.get_device_count()
        else:
            num_devices = 0

        if num_devices > 0:
            sdr = None
            try:
                sdr = RtlSdr(device_index=0)
                device_info = {
                    "name": (
                        sdr.get_device_name() if hasattr(sdr, "get_device_name") else "Unknown"
                    ),
                    "serial": (
                        sdr.get_serial_number() if hasattr(sdr, "get_serial_number") else "Unknown"
                    ),
                    "index": 0,
                }
                device_status = "connected"

                sdr.sample_rate = 2400000
                sdr.center_freq = 145800000
                device_info["sample_rate"] = sdr.sample_rate
                device_info["center_freq"] = sdr.center_freq

            except Exception as e:
                device_status = "error"
                device_info = {"error": str(e)}
            finally:
                if sdr:
                    try:
                        sdr.close()
                    except Exception:
                        pass
        else:
            device_status = "not_found"

    except ImportError:
        device_status = "driver_not_installed"
    except Exception as e:
        device_status = "error"
        device_info = {"error": str(e)}

    memory_usage = {
        "recording_process_mb": None,
        "system_available_mb": None,
    }

    if recording_process and hasattr(recording_process, "pid"):
        try:
            proc = psutil.Process(recording_process.pid)
            memory_info = proc.memory_info()
            memory_usage["recording_process_mb"] = memory_info.rss / (1024 * 1024)
        except Exception:
            pass

    memory_usage["system_available_mb"] = psutil.virtual_memory().available / (1024 * 1024)

    degradation_level = "full"
    capabilities = []

    if device_status == "connected" and SSTV_AVAILABLE:
        capabilities.extend(["realtime_recording", "sstv_decoding", "satellite_tracking"])
    elif device_status == "connected":
        degradation_level = "partial"
        capabilities.append("realtime_recording")
    elif rtl_fm_available:
        degradation_level = "minimal"
        capabilities.append("cli_recording")
    else:
        degradation_level = "offline"

    tle_file = Path("data/tle_data.json")
    tle_age_hours = None
    tle_status = "unknown"

    if tle_file.exists():
        tle_age_hours = round((time.time() - tle_file.stat().st_mtime) / 3600, 1)
        if tle_age_hours < 12:
            tle_status = "fresh"
        elif tle_age_hours < 24:
            tle_status = "acceptable"
        else:
            tle_status = "stale"
    else:
        tle_status = "not_found"

    return {
        "status": "healthy" if degradation_level == "full" else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": {
            "status": device_status,
            "info": device_info,
        },
        "memory": memory_usage,
        "capabilities": capabilities,
        "degradation_level": degradation_level,
        "components": {
            "sstv_decoder": SSTV_AVAILABLE,
            "satellite_tracker": tracker_module is not None,
            "rtl_fm_binary": rtl_fm_available,
            "rtl_test_binary": rtl_test_available,
            "redis_cache": REDIS_AVAILABLE,
        },
        "tle_cache": {
            "status": tle_status,
            "age_hours": tle_age_hours,
        },
        "recommendations": _get_degradation_recommendations(degradation_level, device_status),
    }


def _get_degradation_recommendations(degradation_level: str, device_status: str) -> list:
    """Получение рекомендаций при деградации функциональности."""
    recommendations = []

    if degradation_level == "offline":
        recommendations.append("Подключите RTL-SDR устройство")
        recommendations.append("Установите rtl-sdr утилиты (rtl_fm, rtl_test)")
    elif degradation_level == "minimal":
        recommendations.append("Установите pyrtlsdr для расширенных функций")
        recommendations.append("pip install pyrtlsdr")
    elif degradation_level == "partial":
        recommendations.append("Проверьте SSTV модуль: pip install pysstv")
    elif device_status == "error":
        recommendations.append("Проверьте подключение RTL-SDR устройства")
        recommendations.append("Запустите rtl_test для диагностики")

    return recommendations


@router.get("/device/check")
async def check_rtlsdr_device():
    """
    Проверка подключения RTL-SDR устройства.

    Returns:
        Информация об устройстве (тип, серийный номер, статус)
    """
    try:
        from rtlsdr import RtlSdr
    except ImportError:
        return {
            "status": "error",
            "message": "pyrtlsdr не установлен",
            "devices": [],
        }

    try:
        if hasattr(RtlSdr, "get_device_count"):
            num_devices = RtlSdr.get_device_count()
        else:
            try:
                test = RtlSdr(device_index=0)
                test.close()
                num_devices = 1
            except Exception:
                num_devices = 0

        if num_devices == 0:
            return {
                "status": "no_devices",
                "message": "RTL-SDR устройства не обнаружены",
                "devices": [],
            }

        devices = []
        for i in range(num_devices):
            sdr = None
            try:
                sdr = RtlSdr(device_index=i)
                name = sdr.get_device_name() if hasattr(sdr, "get_device_name") else "Unknown"
                serial = sdr.get_serial_number() if hasattr(sdr, "get_serial_number") else "Unknown"

                is_v4 = "R828D" in name.upper() or "V4" in name.upper()

                devices.append(
                    {
                        "index": i,
                        "name": name,
                        "serial": serial,
                        "is_v4": is_v4,
                        "recommended_sample_rate": 2400000 if is_v4 else 2000000,
                    }
                )
            except Exception as e:
                devices.append({"index": i, "error": str(e)})
            finally:
                if sdr:
                    try:
                        sdr.close()
                    except Exception:
                        pass

        return {"status": "ok", "count": num_devices, "devices": devices}

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "devices": [],
        }
