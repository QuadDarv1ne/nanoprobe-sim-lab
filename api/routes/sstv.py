"""
API для управления SSTV станцией и расписанием пролётов МКС.
Интеграция с pysstv для декодирования и кэшированием Redis.
"""

import asyncio
import logging
import os
import sys
import subprocess
import signal
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import APIRouter, BackgroundTasks, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import FileResponse

from api.error_handlers import ServiceUnavailableError, NotFoundError, ValidationError

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

# Глобальные объекты (ленивая инициализация)
_sstv_decoder: Optional[SSTVDecoder] = None
_tracker: Optional[tracker_module.SatelliteTracker] = None

# Управление записью RTL-SDR
_recording_process: Optional[subprocess.Popen] = None
_recording_start_time: Optional[datetime] = None
_recording_metadata: Dict[str, Any] = {}


def get_redis_cache() -> Optional[RedisCache]:
    """Получает Redis cache instance из api.state."""
    from api.state import get_redis
    return get_redis()


def get_satellite_tracker() -> Optional[tracker_module.SatelliteTracker]:
    """Получает SatelliteTracker instance."""
    global _tracker
    if _tracker is None and tracker_module is not None:
        try:
            # Координаты по умолчанию (Москва)
            lat = float(os.getenv("GROUND_STATION_LAT", "55.75"))
            lon = float(os.getenv("GROUND_STATION_LON", "37.61"))
            _tracker = tracker_module.SatelliteTracker(
                ground_station_lat=lat,
                ground_station_lon=lon
            )
        except Exception as e:
            logger.warning(f"SatelliteTracker initialization error: {e}")
            _tracker = tracker_module.SatelliteTracker()
    return _tracker


def get_sstv_decoder() -> Optional[SSTVDecoder]:
    """Получает SSTVDecoder instance."""
    global _sstv_decoder
    if _sstv_decoder is None and SSTVDecoder is not None:
        try:
            _sstv_decoder = SSTVDecoder(mode='auto')
        except Exception as e:
            logger.warning(f"SSTVDecoder initialization error: {e}")
            _sstv_decoder = None
    return _sstv_decoder


# ============================================================================
# API Endpoints: Расписание пролётов
# ============================================================================

@router.get("/iss/schedule")
async def get_iss_schedule(
    hours_ahead: int = Query(default=24, ge=1, le=72, description="На сколько часов вперёд (1-72)"),
    min_elevation: float = Query(default=10.0, ge=0, le=90, description="Минимальная высота над горизонтом (0-90°)")
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
            return {
                "status": "success",
                "cached": True,
                "data": cached
            }

    # Получаем предсказания
    try:
        passes = tracker.get_pass_predictions(
            satellite_name='iss',
            hours_ahead=hours_ahead,
            min_elevation=min_elevation
        )

        # Форматируем результат
        result = []
        for pass_info in passes:
            result.append({
                "aos": pass_info['aos'].isoformat(),
                "los": pass_info['los'].isoformat(),
                "max_elevation": pass_info['max_elevation'],
                "frequency_mhz": pass_info['frequency'],
                "duration_minutes": pass_info['duration_minutes'],
                "mode": "SSTV Martin 1" if pass_info['frequency'] == 145.800 else "Unknown"
            })

        # Кэшируем на 5 минут
        if redis_cache and REDIS_AVAILABLE:
            redis_cache.set(cache_key, result, expire=300)
        
        return {
            "status": "success",
            "cached": False,
            "count": len(result),
            "data": result
        }

    except Exception as e:
        logger.error(f"Error getting ISS schedule: {e}")
        raise ServiceUnavailableError("Не удалось получить расписание МКС")


@router.get("/iss/next-pass")
async def get_iss_next_pass(
    min_elevation: float = Query(default=10.0, ge=0, le=90, description="Минимальная высота (0-90°)")
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
            return {
                "status": "success",
                "cached": True,
                "data": cached
            }

    try:
        next_pass = tracker.get_next_pass('iss')

        if not next_pass:
            return {
                "status": "success",
                "message": "No passes found in next 24 hours",
                "data": None
            }

        result = {
            "aos": next_pass['aos'].isoformat(),
            "los": next_pass['los'].isoformat(),
            "max_elevation": next_pass['max_elevation'],
            "frequency_mhz": next_pass['frequency'],
            "duration_minutes": next_pass['duration_minutes'],
            "time_until_aos": str(next_pass['aos'] - datetime.now())
        }

        # Кэшируем на 2 минуты
        if redis_cache and REDIS_AVAILABLE:
            redis_cache.set(cache_key, result, expire=120)

        return {
            "status": "success",
            "cached": False,
            "data": result
        }

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
            return {
                "status": "success",
                "cached": True,
                "data": cached
            }

    try:
        position = tracker.get_current_position('iss')

        if not position:
            raise NotFoundError("Позиция МКС недоступна")

        result = {
            "latitude": position['latitude'],
            "longitude": position['longitude'],
            "altitude_km": position['altitude_km'],
            "velocity_kmh": position['velocity_kmh'],
            "footprint_km": position['footprint_km'],
            "timestamp": datetime.now().isoformat()
        }

        # Кэшируем на 30 секунд
        if redis_cache and REDIS_AVAILABLE:
            redis_cache.set(cache_key, result, expire=30)
        
        return {
            "status": "success",
            "cached": False,
            "data": result
        }

    except Exception as e:
        logger.error(f"Error getting ISS position: {e}")
        raise ServiceUnavailableError("Не удалось получить позицию МКС")


@router.get("/iss/visible")
async def is_iss_visible(
    min_elevation: float = Query(default=10.0, ge=0, le=90, description="Минимальная высота для видимости (0-90°)")
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
        visible = tracker.is_satellite_visible('iss', min_elevation)
        position = tracker.get_current_position('iss')

        return {
            "status": "success",
            "visible": visible,
            "elevation": position['latitude'] if position else 0,
            "message": "ISS видна" if visible else "ISS не видна",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error checking ISS visibility: {e}")
        raise ServiceUnavailableError("Не удалось проверить видимость МКС")


# ============================================================================
# API Endpoints: SSTV Декодирование
# ============================================================================

@router.post("/sstv/decode")
async def decode_sstv_audio(
    background_tasks: BackgroundTasks,
    audio_path: str = None,
    mode: str = 'auto'
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
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = output_dir / f"sstv_decoded_{timestamp}.png"
        
        image.save(str(output_path), 'png')
        
        metadata = decoder.get_metadata()
        
        return {
            "status": "success",
            "image_path": str(output_path),
            "image_size": image.size,
            "mode": metadata.get('mode', 'unknown'),
            "timestamp": datetime.now().isoformat(),
            "download_url": f"/api/v1/sstv/download/{output_path.name}"
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
    
    return FileResponse(
        str(file_path),
        media_type="image/png",
        filename=filename
    )


@router.get("/sstv/modes")
async def get_sstv_modes():
    """
    Получает список поддерживаемых SSTV режимов.
    
    Returns:
        Список режимов
    """
    if not SSTV_AVAILABLE:
        return {
            "status": "unavailable",
            "modes": []
        }
    
    decoder = get_sstv_decoder()
    modes = decoder.SUPPORTED_MODES if decoder else []
    
    return {
        "status": "success",
        "count": len(modes),
        "modes": modes
    }


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
    
    return {
        "status": "success",
        "count": len(satellites),
        "satellites": satellites
    }


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
        schedule = tracker.get_sstv_schedule(hours_ahead)
        
        result = []
        for pass_info in schedule:
            result.append({
                "satellite": pass_info['satellite'],
                "aos": pass_info['aos'].isoformat(),
                "los": pass_info['los'].isoformat(),
                "max_elevation": pass_info['max_elevation'],
                "frequency_mhz": pass_info['frequency'],
                "duration_minutes": pass_info['duration_minutes']
            })
        
        return {
            "status": "success",
            "count": len(result),
            "data": result
        }

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
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=5.0
                )
                
                # Обрабатываем команды
                if data == "ping":
                    await websocket.send_text("pong")
                elif data == "stop":
                    break
                    
            except asyncio.TimeoutError:
                # Отправляем позицию
                try:
                    position = tracker.get_current_position('iss')
                    if position:
                        await websocket.send_json({
                            "type": "position",
                            "data": position,
                            "timestamp": datetime.now().isoformat()
                        })
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


# ============================================================================
# Health Check
# ============================================================================

@router.get("/health")
async def sstv_health_check():
    """Проверка здоровья SSTV модуля."""
    status = {
        "sstv_decoder": "available" if SSTV_AVAILABLE else "unavailable",
        "satellite_tracker": "available" if tracker_module is not None else "unavailable",
        "redis_cache": "available" if REDIS_AVAILABLE else "unavailable",
        "rtl_sdr": "ready" if _recording_process is None else "recording",
        "timestamp": datetime.now().isoformat()
    }

    all_ok = all([
        SSTV_AVAILABLE,
        tracker_module is not None
    ])

    return {
        "status": "healthy" if all_ok else "degraded",
        "components": status
    }


# ============================================================================
# RTL-SDR Recording Control
# ============================================================================

@router.post("/record/start")
async def start_sstv_recording(
    frequency: float = 145.800,
    sample_rate: int = 2048000,
    gain: int = 496,
    duration: int = 600
):
    """
    Запуск записи с RTL-SDR для приёма SSTV.
    
    - **frequency**: Частота в MHz (по умолчанию 145.800 для МКС)
    - **sample_rate**: Частота дискретизации (по умолчанию 2048000)
    - **gain**: Усиление RTL-SDR (0-496)
    - **duration**: Длительность записи в секундах
    
    Returns:
        Статус записи
    """
    global _recording_process, _recording_start_time, _recording_metadata
    
    if _recording_process is not None:
        return {
            "status": "already_recording",
            "started_at": _recording_start_time.isoformat() if _recording_start_time else None,
            "message": "Запись уже идёт"
        }
    
    # Создаём директорию для записей
    output_dir = Path("output/sstv/recordings")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"sstv_{frequency}MHz_{timestamp}.wav"
    
    # Команда для rtl_fm (часть rtl-sdr)
    cmd = [
        "rtl_fm",
        "-f", str(frequency * 1e6),  # Конвертируем MHz в Hz
        "-s", str(sample_rate),
        "-g", str(gain),
        "-F", "9",
        "-o", "4",
        "-p", "0",  # ppm correction (настраивается)
        "-M", "fm",
        output_file
    ]
    
    try:
        # Проверяем наличие rtl_fm
        result = subprocess.run(["rtl_fm", "-h"], capture_output=True, timeout=5)
    except FileNotFoundError:
        # rtl_fm не найден - симулируем для тестирования
        logger.warning("rtl_fm not found - simulation mode")
        _recording_start_time = datetime.now()
        _recording_metadata = {
            "frequency": frequency,
            "sample_rate": sample_rate,
            "gain": gain,
            "duration": duration,
            "output_file": str(output_file),
            "simulated": True
        }
        
        return {
            "status": "recording_simulated",
            "frequency_mhz": frequency,
            "sample_rate": sample_rate,
            "output_file": str(output_file),
            "started_at": _recording_start_time.isoformat(),
            "message": "RTL-SDR не найден. Запись симулируется для тестирования."
        }
    
    try:
        _recording_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        _recording_start_time = datetime.now()
        _recording_metadata = {
            "frequency": frequency,
            "sample_rate": sample_rate,
            "gain": gain,
            "duration": duration,
            "output_file": str(output_file),
            "pid": _recording_process.pid
        }
        
        # Планируем остановку через duration секунд
        asyncio.create_task(stop_recording_after(duration))
        
        return {
            "status": "recording_started",
            "frequency_mhz": frequency,
            "sample_rate": sample_rate,
            "gain": gain,
            "output_file": str(output_file),
            "started_at": _recording_start_time.isoformat(),
            "pid": _recording_process.pid,
            "message": f"Запись началась. Остановка через {duration} секунд."
        }
        
    except Exception as e:
        logger.error(f"Recording start error: {e}")
        raise ServiceUnavailableError(f"Не удалось начать запись: {str(e)}")


async def stop_recording_after(duration: int):
    """Автоматическая остановка записи через N секунд."""
    await asyncio.sleep(duration)
    await stop_sstv_recording()


@router.post("/record/stop")
async def stop_sstv_recording():
    """Остановка записи SSTV."""
    global _recording_process, _recording_start_time, _recording_metadata
    
    if _recording_process is None and not _recording_metadata.get("simulated"):
        return {
            "status": "not_recording",
            "message": "Запись не идёт"
        }
    
    # Если симуляция
    if _recording_metadata.get("simulated"):
        duration = (datetime.now() - _recording_start_time).total_seconds() if _recording_start_time else 0
        _recording_process = None
        metadata = _recording_metadata.copy()
        _recording_metadata = {}
        
        return {
            "status": "recording_stopped_simulated",
            "duration_seconds": round(duration, 2),
            "output_file": metadata.get("output_file"),
            "message": "Симуляция записи остановлена"
        }
    
    # Останавливаем процесс
    try:
        _recording_process.send_signal(signal.SIGINT)
        _recording_process.wait(timeout=5)
        
        duration = (datetime.now() - _recording_start_time).total_seconds() if _recording_start_time else 0
        output_file = _recording_metadata.get("output_file")
        
        _recording_process = None
        metadata = _recording_metadata.copy()
        _recording_metadata = {}
        
        return {
            "status": "recording_stopped",
            "duration_seconds": round(duration, 2),
            "output_file": output_file,
            "message": "Запись остановлена"
        }
        
    except Exception as e:
        logger.error(f"Recording stop error: {e}")
        # Принудительная остановка
        if _recording_process:
            _recording_process.kill()
        _recording_process = None
        _recording_metadata = {}

        raise ServiceUnavailableError(f"Не удалось остановить запись: {str(e)}")


@router.get("/record/status")
async def get_recording_status():
    """Получить статус записи."""
    global _recording_process, _recording_start_time, _recording_metadata
    
    if _recording_process is not None or _recording_metadata.get("simulated"):
        duration = (datetime.now() - _recording_start_time).total_seconds() if _recording_start_time else 0
        
        return {
            "status": "recording",
            "recording": True,
            "started_at": _recording_start_time.isoformat() if _recording_start_time else None,
            "duration_seconds": round(duration, 2),
            "metadata": _recording_metadata
        }
    else:
        return {
            "status": "idle",
            "recording": False,
            "message": "Запись не идёт"
        }


@router.get("/recordings")
async def list_recordings(limit: int = 20):
    """Получить список записей SSTV."""
    output_dir = Path("output/sstv/recordings")
    
    if not output_dir.exists():
        return {"recordings": []}
    
    recordings = []
    for file in sorted(output_dir.glob("*.wav"), reverse=True)[:limit]:
        stat = file.stat()
        recordings.append({
            "filename": file.name,
            "path": str(file),
            "size_bytes": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "frequency": file.name.split('_')[1] if '_' in file.name else "unknown"
        })
    
    return {"recordings": recordings}
