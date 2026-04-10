"""
API для управления SSTV станцией и расписанием пролётов МКС.
Интеграция с pysstv для декодирования и кэшированием Redis.
"""

import asyncio
import logging
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from subprocess import DEVNULL
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from api.error_handlers import NotFoundError, ServiceUnavailableError, ValidationError
from api.state import get_app_state, get_redis, set_app_state

logger = logging.getLogger(__name__)

# Добавляем корень проекта в path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.caching.redis_cache import RedisCache

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Импорт SSTV компонентов
try:
    # Добавляем путь к компонентам
    components_path = PROJECT_ROOT / "components" / "py-sstv-groundstation" / "src"
    sys.path.insert(0, str(components_path))

    import satellite_tracker
    from sstv_decoder import SSTVDecoder

    tracker_module = satellite_tracker
    SSTV_AVAILABLE = True
except ImportError:
    SSTV_AVAILABLE = False
    tracker_module = None
    SSTVDecoder = None

router = APIRouter()

# Recording state is managed via app_state (see api/state.py)
# No module-level globals needed - avoids scoping bugs


def get_redis_cache() -> Optional[RedisCache]:
    """Получает Redis cache instance из api.state."""
    return get_redis()


def get_satellite_tracker() -> Optional[Any]:
    """Получает SatelliteTracker instance."""
    if tracker_module is None:
        return None

    tracker = get_app_state("satellite_tracker")
    if tracker is not None:
        return tracker

    try:
        # Координаты по умолчанию (Москва)
        lat = float(os.getenv("GROUND_STATION_LAT", "55.75"))
        lon = float(os.getenv("GROUND_STATION_LON", "37.61"))
        tracker = tracker_module.SatelliteTracker(ground_station_lat=lat, ground_station_lon=lon)
        set_app_state("satellite_tracker", tracker)
    except Exception as e:
        logger.warning(f"SatelliteTracker initialization error: {e}")
        tracker = None
        set_app_state("satellite_tracker", tracker)
    return tracker


def get_sstv_decoder() -> Optional[SSTVDecoder]:
    """Получает SSTVDecoder instance."""
    decoder = get_app_state("sstv_decoder")
    if decoder is not None:
        return decoder

    if SSTVDecoder is not None:
        try:
            decoder = SSTVDecoder(mode="auto")
            set_app_state("sstv_decoder", decoder)
        except Exception as e:
            logger.warning(f"SSTVDecoder initialization error: {e}")
            decoder = None
    return decoder


# ============================================================================
# API Endpoints: Расписание пролётов
# ============================================================================


@router.get("/iss/schedule")
async def get_iss_schedule(
    hours_ahead: int = Query(default=24, ge=1, le=72, description="На сколько часов вперёд (1-72)"),
    min_elevation: float = Query(
        default=10.0, ge=0, le=90, description="Минимальная высота над горизонтом (0-90°)"
    ),
):
    """
    Получает расписание пролётов МКС (ISS).

    Returns:
        Список пролётов с временем, высотой и частотой
    """
    tracker = get_satellite_tracker()
    if not tracker:
        raise ServiceUnavailableError("Satellite tracker недоступен")

    # Проверяем кэш
    cache_key = f"iss_schedule:{hours_ahead}:{min_elevation}"
    redis_cache = get_redis_cache()

    if redis_cache and REDIS_AVAILABLE:
        cached = redis_cache.get(cache_key)
        if cached:
            return {"status": "success", "cached": True, "data": cached}

    # Получаем предсказания
    try:
        passes = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: tracker.get_pass_predictions(
                satellite_name="iss", hours_ahead=hours_ahead, min_elevation=min_elevation
            ),
        )

        # Форматируем результат
        result = []
        for pass_info in passes:
            result.append(
                {
                    "aos": pass_info["aos"].isoformat(),
                    "los": pass_info["los"].isoformat(),
                    "max_elevation": pass_info["max_elevation"],
                    "frequency_mhz": pass_info["frequency"],
                    "duration_minutes": pass_info["duration_minutes"],
                    "mode": "SSTV Martin 1" if pass_info["frequency"] == 145.800 else "Unknown",
                }
            )

        # Кэшируем на 5 минут
        if redis_cache and REDIS_AVAILABLE:
            redis_cache.set(cache_key, result, expire=300)

        return {"status": "success", "cached": False, "count": len(result), "data": result}

    except Exception as e:
        logger.error(f"Error getting ISS schedule: {e}")
        raise ServiceUnavailableError("Не удалось получить расписание МКС")


@router.get("/iss/next-pass")
async def get_iss_next_pass(
    min_elevation: float = Query(
        default=10.0, ge=0, le=90, description="Минимальная высота (0-90°)"
    )
):
    """
    Получает следующий пролёт МКС.

    Returns:
        Информация о следующем пролёте
    """
    tracker = get_satellite_tracker()
    if not tracker:
        raise ServiceUnavailableError("Satellite tracker недоступен")

    # Проверяем кэш
    cache_key = "iss_next_pass"
    redis_cache = get_redis_cache()

    if redis_cache and REDIS_AVAILABLE:
        cached = redis_cache.get(cache_key)
        if cached:
            return {"status": "success", "cached": True, "data": cached}

    try:
        next_pass = await asyncio.get_event_loop().run_in_executor(
            None, lambda: tracker.get_next_pass("iss")
        )

        if not next_pass:
            return {
                "status": "success",
                "message": "No passes found in next 24 hours",
                "data": None,
            }

        result = {
            "aos": next_pass["aos"].isoformat(),
            "los": next_pass["los"].isoformat(),
            "max_elevation": round(next_pass["max_elevation"], 1),
            "frequency_mhz": next_pass["frequency"],
            "duration_minutes": round(next_pass["duration_minutes"], 1),
            "time_until_aos": next_pass.get(
                "time_until_aos", str(next_pass["aos"] - datetime.utcnow())
            ),
        }

        # Кэшируем на 2 минуты
        if redis_cache and REDIS_AVAILABLE:
            redis_cache.set(cache_key, result, expire=120)

        return {"status": "success", "cached": False, "data": result}

    except Exception as e:
        logger.error(f"Error getting ISS next pass: {e}")
        raise ServiceUnavailableError("Не удалось получить данные о пролёте МКС")


@router.get("/iss/position")
async def get_iss_current_position():
    """
    Получает текущую позицию МКС.

    Returns:
        Текущая позиция (широта, долгота, высота)
    """
    tracker = get_satellite_tracker()
    if not tracker:
        raise ServiceUnavailableError("Satellite tracker недоступен")

    # Проверяем кэш (позиция обновляется каждые 30 секунд)
    cache_key = "iss_position"
    redis_cache = get_redis_cache()

    if redis_cache and REDIS_AVAILABLE:
        cached = redis_cache.get(cache_key)
        if cached:
            return {"status": "success", "cached": True, "data": cached}

    try:
        position = tracker.get_current_position("iss")

        if not position:
            raise NotFoundError("Позиция МКС недоступна")

        result = {
            "latitude": position["latitude"],
            "longitude": position["longitude"],
            "altitude_km": position["altitude_km"],
            "velocity_kmh": position["velocity_kmh"],
            "footprint_km": position["footprint_km"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Кэшируем на 30 секунд
        if redis_cache and REDIS_AVAILABLE:
            redis_cache.set(cache_key, result, expire=30)

        return {"status": "success", "cached": False, "data": result}

    except Exception as e:
        logger.error(f"Error getting ISS position: {e}")
        raise ServiceUnavailableError("Не удалось получить позицию МКС")


@router.get("/iss/visible")
async def is_iss_visible(
    min_elevation: float = Query(
        default=10.0, ge=0, le=90, description="Минимальная высота для видимости (0-90°)"
    )
):
    """
    Проверяет видимость МКС сейчас.

    Returns:
        Статус видимости и информация
    """
    tracker = get_satellite_tracker()
    if not tracker:
        raise ServiceUnavailableError("Satellite tracker недоступен")

    try:
        visible = tracker.is_satellite_visible("iss", min_elevation)
        position = tracker.get_current_position("iss")

        return {
            "status": "success",
            "visible": visible,
            "elevation": (
                tracker._elevation_from_position(position, __import__("datetime").datetime.utcnow())
                if position
                else 0
            ),
            "message": "ISS видна" if visible else "ISS не видна",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error checking ISS visibility: {e}")
        raise ServiceUnavailableError("Не удалось проверить видимость МКС")


# ============================================================================
# API Endpoints: SSTV Декодирование
# ============================================================================


@router.post("/sstv/decode")
async def decode_sstv_audio(
    background_tasks: BackgroundTasks, audio_path: str = None, mode: str = "auto"
):
    """
    Декодирует SSTV из аудиофайла.

    - **audio_path**: Путь к WAV файлу
    - **mode**: Режим декодирования (auto, Martin 1, Scottie 1, etc.)

    Returns:
        Информация о декодированном изображении
    """
    if not SSTV_AVAILABLE:
        raise ServiceUnavailableError("SSTV декодер недоступен")

    decoder = get_sstv_decoder()
    if not decoder:
        raise ServiceUnavailableError("SSTV декодер не инициализирован")

    try:
        # Если путь не указан, используем тестовый
        if not audio_path:
            test_audio = Path("data/sstv_test_audio.wav")
            if test_audio.exists():
                audio_path = str(test_audio)
            else:
                raise ValidationError("Аудиофайл не предоставлен и тестовый файл не найден")

        # Проверяем существование файла
        if not Path(audio_path).exists():
            raise NotFoundError(f"Аудиофайл не найден: {audio_path}")

        # Декодируем
        decoder.mode = mode
        image = decoder.decode_from_audio(audio_path)

        if not image:
            raise ValidationError("Не удалось декодировать SSTV")

        # Сохраняем результат
        output_dir = Path("output/sstv")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"sstv_decoded_{timestamp}.png"

        image.save(str(output_path), "png")

        metadata = decoder.get_metadata()

        return {
            "status": "success",
            "image_path": str(output_path),
            "image_size": image.size,
            "mode": metadata.get("mode", "unknown"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "download_url": f"/api/v1/sstv/download/{output_path.name}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SSTV decoding error: {e}")
        raise ServiceUnavailableError("Ошибка декодирования SSTV")


@router.get("/sstv/download/{filename}")
async def download_sstv_image(filename: str):
    """Скачивает декодированное SSTV изображение."""
    file_path = Path("output/sstv") / filename

    if not file_path.exists():
        raise NotFoundError(f"Изображение не найдено: {filename}")

    return FileResponse(str(file_path), media_type="image/png", filename=filename)


@router.get("/sstv/modes")
async def get_sstv_modes():
    """
    Получает список поддерживаемых SSTV режимов.

    Returns:
        Список режимов
    """
    if not SSTV_AVAILABLE:
        return {"status": "unavailable", "modes": []}

    decoder = get_sstv_decoder()
    modes = decoder.SUPPORTED_MODES if decoder else []

    return {"status": "success", "count": len(modes), "modes": modes}


# ============================================================================
# API Endpoints: Спутники
# ============================================================================


@router.get("/satellites")
async def get_all_satellites():
    """Получает список всех отслеживаемых спутников."""
    tracker = get_satellite_tracker()
    if not tracker:
        raise ServiceUnavailableError("Satellite tracker недоступен")

    satellites = tracker.get_all_satellites()

    return {"status": "success", "count": len(satellites), "satellites": satellites}


@router.get("/satellites/schedule")
async def get_all_satellites_schedule(hours_ahead: int = 24):
    """
    Получает расписание всех SSTV спутников.

    Returns:
        Расписание пролётов
    """
    tracker = get_satellite_tracker()
    if not tracker:
        raise ServiceUnavailableError("Satellite tracker недоступен")

    hours_ahead = min(hours_ahead, 72)

    try:
        schedule = await asyncio.get_event_loop().run_in_executor(
            None, lambda: tracker.get_sstv_schedule(hours_ahead)
        )

        result = []
        for pass_info in schedule:
            result.append(
                {
                    "satellite": pass_info["satellite"],
                    "aos": pass_info["aos"].isoformat(),
                    "los": pass_info["los"].isoformat(),
                    "max_elevation": pass_info["max_elevation"],
                    "frequency_mhz": pass_info["frequency"],
                    "duration_minutes": pass_info["duration_minutes"],
                }
            )

        return {"status": "success", "count": len(result), "data": result}

    except Exception as e:
        logger.error(f"Error getting satellite passes: {e}")
        raise ServiceUnavailableError("Не удалось получить расписание спутников")


# ============================================================================
# WebSocket: Real-time обновления
# ============================================================================


@router.websocket("/ws/iss")
async def iss_websocket(websocket: WebSocket):
    """
    WebSocket для real-time обновлений позиции МКС.
    Отправляет позицию каждые 5 секунд.
    """
    await websocket.accept()

    tracker = get_satellite_tracker()
    if not tracker:
        await websocket.send_json({"error": "Satellite tracker not available"})
        await websocket.close()
        return

    try:
        while True:
            try:
                # Проверяем сообщения от клиента
                data = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)

                # Обрабатываем команды
                if data == "ping":
                    await websocket.send_text("pong")
                elif data == "stop":
                    break

            except asyncio.TimeoutError:
                # Отправляем позицию
                try:
                    position = tracker.get_current_position("iss")
                    if position:
                        await websocket.send_json(
                            {
                                "type": "position",
                                "data": position,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                except Exception:
                    pass

            except WebSocketDisconnect:
                break

    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@router.post("/tle/refresh")
async def refresh_tle():
    """
    Принудительное обновление TLE данных с CelesTrak.

    Returns:
        Количество обновлённых спутников
    """
    tracker = get_satellite_tracker()
    if not tracker:
        raise ServiceUnavailableError("Satellite tracker недоступен")

    try:
        updated = tracker.update_tle_from_celestrak()
        if updated > 0:
            tracker.save_tle("data/tle_data.json")
            # Сбрасываем кэшированный трекер чтобы следующий запрос получил свежий
            set_app_state("satellite_tracker", None)

        return {
            "status": "success",
            "updated": updated,
            "message": f"Обновлено TLE: {updated} спутников",
        }
    except Exception as e:
        logger.error(f"TLE refresh error: {e}")
        raise ServiceUnavailableError(f"Ошибка обновления TLE: {str(e)}")


@router.get("/tle/status")
async def get_tle_status():
    """Статус TLE данных (возраст, источник)."""
    import time
    from pathlib import Path

    tle_file = Path("data/tle_data.json")

    if tle_file.exists():
        age_hours = (time.time() - tle_file.stat().st_mtime) / 3600
        return {
            "status": "cached",
            "age_hours": round(age_hours, 2),
            "fresh": age_hours < 12,
            "file": str(tle_file),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    return {
        "status": "builtin",
        "age_hours": None,
        "fresh": False,
        "message": "Используются встроенные TLE, рекомендуется обновить",
    }


# ============================================================================
# Health Check
# ============================================================================


@router.get("/health")
async def sstv_health_check():
    """Проверка здоровья SSTV модуля."""
    recording_process = get_app_state("recording_process")

    # Проверяем наличие rtl_fm в PATH
    import shutil

    rtl_fm_available = shutil.which("rtl_fm") is not None

    # Проверяем TLE возраст
    import time as _time
    from pathlib import Path as _Path

    tle_file = _Path("data/tle_data.json")
    tle_age_hours = None
    if tle_file.exists():
        tle_age_hours = round((_time.time() - tle_file.stat().st_mtime) / 3600, 1)

    status = {
        "sstv_decoder": "available" if SSTV_AVAILABLE else "unavailable",
        "satellite_tracker": "available" if tracker_module is not None else "unavailable",
        "redis_cache": "available" if REDIS_AVAILABLE else "unavailable",
        "rtl_sdr_recording": "idle" if recording_process is None else "recording",
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
    import shutil
    from pathlib import Path as _Path

    import psutil

    recording_process = get_app_state("recording_process")

    # Базовая проверка
    rtl_fm_available = shutil.which("rtl_fm") is not None
    rtl_test_available = shutil.which("rtl_test") is not None

    # Проверка устройства
    device_info = None
    device_status = "unknown"

    try:
        from rtlsdr import RtlSdr

        if hasattr(RtlSdr, "get_device_count"):
            num_devices = RtlSdr.get_device_count()
        else:
            num_devices = 0

        if num_devices > 0:
            # Получаем инфо о первом устройстве
            sdr = None
            try:
                sdr = RtlSdr(device_index=0)
                device_info = {
                    "name": sdr.get_device_name() if hasattr(sdr, "get_device_name") else "Unknown",
                    "serial": (
                        sdr.get_serial_number() if hasattr(sdr, "get_serial_number") else "Unknown"
                    ),
                    "index": 0,
                }
                device_status = "connected"

                # Проверяем sample rate
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
                    except:
                        pass
        else:
            device_status = "not_found"

    except ImportError:
        device_status = "driver_not_installed"
    except Exception as e:
        device_status = "error"
        device_info = {"error": str(e)}

    # Проверка памяти
    memory_usage = {
        "recording_process_mb": None,
        "system_available_mb": None,
    }

    if recording_process and hasattr(recording_process, "pid"):
        try:
            proc = psutil.Process(recording_process.pid)
            memory_info = proc.memory_info()
            memory_usage["recording_process_mb"] = memory_info.rss / (1024 * 1024)
        except:
            pass

    memory_usage["system_available_mb"] = psutil.virtual_memory().available / (1024 * 1024)

    # Graceful degradation status
    degradation_level = "full"  # full, partial, minimal, offline
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

    # TLE cache
    tle_file = _Path("data/tle_data.json")
    tle_age_hours = None
    tle_status = "unknown"

    if tle_file.exists():
        tle_age_hours = round((_time.time() - tle_file.stat().st_mtime) / 3600, 1)
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
    """Получение рекомендаций при деградации функциональности"""
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
        return {"status": "error", "message": "pyrtlsdr не установлен", "devices": []}

    try:
        if hasattr(RtlSdr, "get_device_count"):
            num_devices = RtlSdr.get_device_count()
        else:
            # Пробуем открыть устройство 0
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
        logger.error(f"Device check error: {e}")
        return {"status": "error", "message": str(e), "devices": []}


# ============================================================================
# RTL-SDR Recording Control
# ============================================================================


@router.post("/record/start")
async def start_sstv_recording(
    frequency: float = 145.800,
    sample_rate: int = 2048000,
    gain: int = 30,
    duration: int = 600,
    ppm: int = 0,
):
    """
    Запуск записи с RTL-SDR для приёма SSTV.

    - **frequency**: Частота в MHz (по умолчанию 145.800 для МКС)
    - **sample_rate**: Частота дискретизации (по умолчанию 2048000)
    - **gain**: Усиление RTL-SDR в dB (0-50, по умолчанию 30)
    - **duration**: Длительность записи в секундах
    - **ppm**: Коррекция частоты в ppm (для TCXO обычно 0)

    Returns:
        Статус записи
    """
    recording_process = get_app_state("recording_process")
    recording_start_time = get_app_state("recording_start_time")

    if recording_process is not None:
        return {
            "status": "already_recording",
            "started_at": recording_start_time.isoformat() if recording_start_time else None,
            "message": "Запись уже идёт",
        }

    # Создаём директорию для записей
    output_dir = Path("output/sstv/recordings")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"sstv_{frequency}MHz_{timestamp}.wav"

    # Команда для rtl_fm (часть rtl-sdr)
    # rtl_fm принимает gain в tenths of dB (0.1 dB единицы), поэтому умножаем на 10
    cmd = [
        "rtl_fm",
        "-f",
        str(int(frequency * 1e6)),  # Конвертируем MHz в Hz
        "-s",
        str(sample_rate),
        "-g",
        str(gain * 10),  # dB -> tenths of dB для rtl_fm
        "-F",
        "9",
        "-o",
        "4",
        "-p",
        str(ppm),  # ppm коррекция (0 для TCXO)
        "-M",
        "fm",
        str(output_file),
    ]

    try:
        # Проверяем наличие rtl_fm
        result = subprocess.run(["rtl_fm", "-h"], capture_output=True, timeout=5)
    except FileNotFoundError:
        # rtl_fm не найден - симулируем для тестирования
        logger.warning("rtl_fm not found - simulation mode")
        recording_start_time = datetime.now(timezone.utc)
        recording_metadata = {
            "frequency": frequency,
            "sample_rate": sample_rate,
            "gain": gain,
            "duration": duration,
            "output_file": str(output_file),
            "simulated": True,
        }
        set_app_state("recording_start_time", recording_start_time)
        set_app_state("recording_metadata", recording_metadata)

        return {
            "status": "recording_simulated",
            "frequency_mhz": frequency,
            "sample_rate": sample_rate,
            "output_file": str(output_file),
            "started_at": recording_start_time.isoformat(),
            "message": "RTL-SDR не найден. Запись симулируется для тестирования.",
        }

    try:
        # Use DEVNULL instead of PIPE to avoid deadlock when pipe buffers fill up
        # PIPE would need background threads to drain, DEVNULL just discards output
        recording_process = subprocess.Popen(cmd, stdout=DEVNULL, stderr=DEVNULL, stdin=DEVNULL)
        recording_start_time = datetime.now(timezone.utc)
        recording_metadata = {
            "frequency": frequency,
            "sample_rate": sample_rate,
            "gain": gain,
            "duration": duration,
            "output_file": str(output_file),
            "pid": recording_process.pid,
        }
        set_app_state("recording_process", recording_process)
        set_app_state("recording_start_time", recording_start_time)
        set_app_state("recording_metadata", recording_metadata)

        # Планируем остановку через duration секунд
        asyncio.create_task(stop_recording_after(duration))

        return {
            "status": "recording_started",
            "frequency_mhz": frequency,
            "sample_rate": sample_rate,
            "gain": gain,
            "output_file": str(output_file),
            "started_at": recording_start_time.isoformat(),
            "pid": recording_process.pid,
            "message": f"Запись началась. Остановка через {duration} секунд.",
        }

    except Exception as e:
        logger.error(f"Recording start error: {e}")
        if recording_process:
            try:
                recording_process.kill()
                recording_process.wait(timeout=2)
            except Exception:
                pass
        raise ServiceUnavailableError(f"Не удалось начать запись: {str(e)}")


async def stop_recording_after(duration: int):
    """Автоматическая остановка записи через N секунд."""
    await asyncio.sleep(duration)
    await stop_sstv_recording()


@router.post("/record/stop")
async def stop_sstv_recording():
    """Остановка записи SSTV."""
    recording_process = get_app_state("recording_process")
    recording_metadata = get_app_state("recording_metadata", {})

    if recording_process is None and not recording_metadata.get("simulated"):
        return {"status": "not_recording", "message": "Запись не идёт"}

    # Если симуляция
    if recording_metadata.get("simulated"):
        recording_start_time = get_app_state("recording_start_time")
        duration = (
            (datetime.now(timezone.utc) - recording_start_time).total_seconds()
            if recording_start_time
            else 0
        )
        set_app_state("recording_process", None)
        metadata = recording_metadata.copy()
        set_app_state("recording_metadata", {})

        return {
            "status": "recording_stopped_simulated",
            "duration_seconds": round(duration, 2),
            "output_file": metadata.get("output_file"),
            "message": "Симуляция записи остановлена",
        }

    # Останавливаем процесс
    try:
        if recording_process is None:
            raise ServiceUnavailableError("Recording process не найден в state")

        recording_process.send_signal(signal.SIGINT)
        try:
            recording_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Process did not terminate gracefully, killing")
            recording_process.kill()
            recording_process.wait(timeout=2)

        # Process resources automatically cleaned up on termination

        recording_start_time = get_app_state("recording_start_time")
        duration = (
            (datetime.now(timezone.utc) - recording_start_time).total_seconds()
            if recording_start_time
            else 0
        )
        output_file = recording_metadata.get("output_file")

        set_app_state("recording_process", None)
        metadata = recording_metadata.copy()
        set_app_state("recording_metadata", {})

        return {
            "status": "recording_stopped",
            "duration_seconds": round(duration, 2),
            "output_file": output_file,
            "message": "Запись остановлена",
        }

    except Exception as e:
        logger.error(f"Recording stop error: {e}")
        # Принудительная остановка
        if recording_process:
            try:
                recording_process.kill()
                recording_process.wait(timeout=2)
            except Exception:
                pass
        set_app_state("recording_process", None)
        set_app_state("recording_metadata", {})

        raise ServiceUnavailableError(f"Не удалось остановить запись: {str(e)}")


@router.get("/record/status")
async def get_recording_status():
    """Получить статуса записи."""
    recording_process = get_app_state("recording_process")
    recording_metadata = get_app_state("recording_metadata", {})
    recording_start_time = get_app_state("recording_start_time")

    if recording_process is not None or recording_metadata.get("simulated"):
        duration = (
            (datetime.now(timezone.utc) - recording_start_time).total_seconds()
            if recording_start_time
            else 0
        )

        return {
            "status": "recording",
            "recording": True,
            "started_at": recording_start_time.isoformat() if recording_start_time else None,
            "duration_seconds": round(duration, 2),
            "metadata": recording_metadata,
        }
    else:
        return {"status": "idle", "recording": False, "message": "Запись не идёт"}


@router.get("/recordings")
async def list_recordings(limit: int = 20):
    """Получить список записей SSTV (из файловой системы + БД)."""
    output_dir = Path("output/sstv/recordings")

    if not output_dir.exists():
        return {"recordings": []}

    recordings = []
    for file in sorted(output_dir.glob("*.wav"), reverse=True)[:limit]:
        stat = file.stat()
        # Ищем сопутствующий PNG
        png = file.with_suffix(".png")
        meta_json = file.with_suffix(".json")

        # Читаем метаданные сначала
        metadata = {}
        if meta_json.exists():
            try:
                import json as _json

                metadata = _json.loads(meta_json.read_text())
            except Exception:
                pass

        entry = {
            "filename": file.name,
            "path": str(file),
            "size_bytes": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "has_image": png.exists(),
            "image_filename": png.name if png.exists() else None,
            "frequency": metadata.get("frequency", "145.800"),
            "metadata": metadata,
        }
        recordings.append(entry)

    return {"count": len(recordings), "recordings": recordings}


@router.get("/recordings/{filename}")
async def download_recording(filename: str):
    """Скачать запись SSTV (WAV файл)."""
    file_path = Path("output/sstv/recordings") / filename

    if not file_path.exists():
        raise NotFoundError(f"Запись не найдена: {filename}")

    return FileResponse(str(file_path), media_type="audio/wav", filename=filename)


@router.delete("/recordings/{filename}")
async def delete_recording(filename: str):
    """Удалить запись SSTV."""
    wav_path = Path("output/sstv/recordings") / filename

    if not wav_path.exists():
        raise NotFoundError(f"Запись не найдена: {filename}")

    try:
        # Удаляем WAV
        wav_path.unlink()

        # Удаляем PNG если есть
        png_path = wav_path.with_suffix(".png")
        if png_path.exists():
            png_path.unlink()

        # Удаляем JSON метаданные если есть
        json_path = wav_path.with_suffix(".json")
        if json_path.exists():
            json_path.unlink()

        return {
            "status": "success",
            "message": f"Запись {filename} удалена",
            "deleted_files": [
                str(wav_path),
                str(png_path) if png_path.exists() else None,
                str(json_path) if json_path.exists() else None,
            ],
        }
    except Exception as e:
        logger.error(f"Delete recording error: {e}")
        raise ServiceUnavailableError(f"Ошибка удаления записи: {str(e)}")


@router.post("/sstv/decode-recording/{filename}")
async def decode_existing_recording(filename: str, mode: str = "auto"):
    """
    Декодирует уже записанный WAV файл из output/sstv/recordings/.
    Сохраняет результат рядом с WAV как PNG.
    """
    if not SSTV_AVAILABLE:
        raise ServiceUnavailableError("SSTV декодер недоступен")

    wav_path = Path("output/sstv/recordings") / filename
    if not wav_path.exists():
        raise NotFoundError(f"Файл не найден: {filename}")

    decoder = get_sstv_decoder()
    if not decoder:
        raise ServiceUnavailableError("SSTV декодер не инициализирован")

    try:
        decoder.mode = mode
        image = decoder.decode_from_audio(str(wav_path))

        if not image:
            raise ValidationError("Не удалось декодировать SSTV из файла")

        output_path = wav_path.with_suffix(".png")
        image.save(str(output_path), "PNG")

        metadata = decoder.get_metadata()

        # Сохраняем в БД
        try:
            from api.state import get_db_manager

            db = get_db_manager()
            db.add_scan_result(
                scan_type="sstv",
                file_path=str(output_path),
                metadata={
                    "source_wav": filename,
                    "sstv_mode": metadata.get("mode", "unknown"),
                    "image_size": list(image.size),
                },
            )
        except Exception as e:
            logger.debug(f"Could not save SSTV scan to DB: {e}")

        return {
            "status": "success",
            "image_path": str(output_path),
            "image_size": image.size,
            "mode": metadata.get("mode", "unknown"),
            "download_url": f"/api/v1/sstv/download/{output_path.name}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Decode recording error: {e}")
        raise ServiceUnavailableError(f"Ошибка декодирования: {str(e)}")
