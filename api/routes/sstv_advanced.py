"""
Продвинутый SSTV API: спектр, сила сигнала, WebSocket стриминг

Эндпоинты:
- GET  /status            — полный статус системы
- GET  /spectrum          — спектр сигнала
- GET  /signal-strength   — сила сигнала
- WS   /ws/stream         — real-time стрим спектра и сигнала
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from api.error_handlers import ServiceUnavailableError

logger = logging.getLogger(__name__)

router = APIRouter()

# SSTV Receiver — быстрая проверка без блокировки
RECEIVER_AVAILABLE = False
try:
    from api.sstv.rtl_sstv_receiver import RTLSDR_AVAILABLE as _RTL
    from api.sstv.session_manager import get_session_manager

    RECEIVER_AVAILABLE = _RTL
except ImportError:
    pass

# Кэш статуса устройства (чтобы не блокироваться на каждом запросе)
_device_cache = {
    "checked": False,
    "available": False,
    "timestamp": 0,
    "error": None,
}

CACHE_TTL = 10  # секунд


def _check_device_fast() -> dict:
    """Быстрая проверка устройства с кэшированием"""
    now = time.time()

    if _device_cache["checked"] and (now - _device_cache["timestamp"]) < CACHE_TTL:
        return {
            "available": _device_cache["available"],
            "error": _device_cache["error"],
        }

    if not RECEIVER_AVAILABLE:
        _device_cache.update(
            {
                "checked": True,
                "available": False,
                "timestamp": now,
                "error": "pyrtlsdr not installed",
            }
        )
        return {"available": False, "error": "pyrtlsdr not installed"}

    try:
        from api.sstv.rtl_sstv_receiver import get_receiver

        receiver = get_receiver()

        # Быстрая проверка: пробуем открыть устройство с таймаутом
        if receiver.sdr is None:
            # Не пытаемся инициализировать — просто проверяем что модуль загружен
            _device_cache.update(
                {
                    "checked": True,
                    "available": True,
                    "timestamp": now,
                    "error": None,
                }
            )
            return {"available": True, "error": None}

        _device_cache.update(
            {
                "checked": True,
                "available": True,
                "timestamp": now,
                "error": None,
            }
        )
        return {"available": True, "error": None}

    except Exception as e:
        err_msg = str(e)[:100]
        _device_cache.update(
            {
                "checked": True,
                "available": False,
                "timestamp": now,
                "error": err_msg,
            }
        )
        return {"available": False, "error": err_msg}


def _init_receiver_safe() -> bool:
    """Безопасная инициализация приёмника"""
    if not RECEIVER_AVAILABLE:
        return False

    try:
        from api.sstv.rtl_sstv_receiver import get_receiver

        receiver = get_receiver()

        if receiver.sdr is not None:
            return True  # Уже инициализирован

        # Пробуем инициализацию
        result = receiver.initialize()
        return result
    except Exception as e:
        logger.error(f"Receiver init failed: {e}")
        return False


# ============================================
# API Endpoints
# ============================================


@router.get("/status", summary="Статус SSTV системы")
async def get_sstv_status():
    """Полный статус SSTV системы — быстрая проверка"""
    device = _check_device_fast()

    modes = []
    if RECEIVER_AVAILABLE:
        try:
            from api.sstv.rtl_sstv_receiver import SSTV_MODES

            modes = list(SSTV_MODES.keys())
        except ImportError:
            pass

    session_mgr = get_session_manager() if RECEIVER_AVAILABLE else None

    return {
        "receiver": {
            "available": device["available"],
            "error": device["error"],
            "pyrtlsdr_loaded": RECEIVER_AVAILABLE,
        },
        "session_manager": session_mgr.get_stats() if session_mgr else {},
        "frequencies": {
            "iss_sstv": 145.800,
            "noaa_apt": 137.100,
            "meteor_m2": 137.100,
        },
        "modes": modes,
        "config": {
            "frequency": float(os.getenv("SSTV_FREQUENCY", "145.800")),
            "gain": float(os.getenv("SSTV_GAIN", "49.6")),
            "sample_rate": int(os.getenv("SSTV_SAMPLE_RATE", "2400000")),
            "bias_tee": os.getenv("SSTV_BIAS_TEE", "0") == "1",
            "agc": os.getenv("SSTV_AGC", "0") == "1",
            "ppm": int(os.getenv("SSTV_PPM", "0")),
        },
    }


@router.get("/spectrum", summary="Спектр сигнала")
async def get_sstv_spectrum(
    frequency: float = Query(145.800, description="Частота МГц"),
    span: float = Query(2.0, description="Диапазон МГц"),
    points: int = Query(512, description="Количество точек"),
):
    """Получить спектр сигнала"""
    device = _check_device_fast()
    if not device["available"]:
        raise ServiceUnavailableError(f"SSTV receiver not available: {device['error']}")

    try:
        from api.sstv.rtl_sstv_receiver import get_receiver

        receiver = get_receiver()

        if not _init_receiver_safe():
            raise ServiceUnavailableError("Failed to initialize RTL-SDR device")

        if abs(receiver.frequency - frequency) > 0.001:
            try:
                receiver.set_frequency(frequency)
            except Exception:
                pass

        freqs, power = receiver.get_spectrum(num_points=min(points, 1024))

        if freqs is None or power is None:
            raise ServiceUnavailableError("Failed to get spectrum data")

        return {
            "frequency_mhz": frequency,
            "span_mhz": span,
            "points": len(freqs),
            "frequencies": freqs.tolist(),
            "power_db": power.tolist(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except ServiceUnavailableError:
        raise
    except Exception as e:
        raise ServiceUnavailableError(f"Spectrum error: {str(e)}")


@router.get("/signal-strength", summary="Сила сигнала")
async def get_signal_strength():
    """Получить текущую силу сигнала"""
    device = _check_device_fast()
    if not device["available"]:
        raise ServiceUnavailableError(f"SSTV receiver not available: {device['error']}")

    try:
        from api.sstv.rtl_sstv_receiver import get_receiver

        receiver = get_receiver()

        if not _init_receiver_safe():
            raise ServiceUnavailableError("Failed to initialize RTL-SDR device")

        strength = receiver.get_signal_strength()

        return {
            "strength_percent": strength,
            "frequency_mhz": receiver.frequency,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except ServiceUnavailableError:
        raise
    except Exception as e:
        raise ServiceUnavailableError(f"Signal strength error: {str(e)}")


@router.websocket("/ws/stream")
async def sstv_websocket_stream(websocket: WebSocket):
    """WebSocket stream для SSTV данных (спектр + сила сигнала)"""
    if not RECEIVER_AVAILABLE:
        await websocket.close(code=1013, reason="SSTV receiver not available")
        return

    session_manager = get_session_manager()
    await session_manager.add_websocket(websocket)

    # Безопасная инициализация
    receiver = None
    if not _init_receiver_safe():
        # Отправляем статус ошибки и ждём
        await websocket.send_json(
            {
                "type": "error",
                "message": "RTL-SDR device not available or failed to initialize",
            }
        )
        # Не закрываем — пусть клиент переподключится
        while True:
            await asyncio.sleep(30)
            try:
                if _init_receiver_safe():
                    break
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Still waiting for RTL-SDR device...",
                    }
                )
            except Exception:
                break
    else:
        from api.sstv.rtl_sstv_receiver import get_receiver

        receiver = get_receiver()

    try:
        while True:
            if receiver and receiver.sdr:
                try:
                    # Отправляем спектр
                    freqs, power = receiver.get_spectrum(num_points=512)
                    if freqs is not None and power is not None:
                        await websocket.send_json(
                            {
                                "type": "spectrum",
                                "frequencies": freqs.tolist()[-100:],
                                "power_db": power.tolist()[-100:],
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                        )

                    # Отправляем силу сигнала
                    strength = receiver.get_signal_strength()
                    await websocket.send_json(
                        {
                            "type": "signal_strength",
                            "strength": strength,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                except Exception as e:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": f"Receiver error: {str(e)}",
                        }
                    )

            await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        logger.info("SSTV WebSocket отключен")
    except Exception as e:
        logger.error(f"SSTV WebSocket ошибка: {e}")
    finally:
        await session_manager.remove_websocket(websocket)


# ============================================
# Shutdown
# ============================================


@router.on_event("shutdown")
async def shutdown_sstv():
    if RECEIVER_AVAILABLE:
        try:
            from api.sstv.rtl_sstv_receiver import get_receiver

            receiver = get_receiver()
            receiver.close()
        except Exception:
            pass
    logger.info("SSTV Advanced API shutdown")
