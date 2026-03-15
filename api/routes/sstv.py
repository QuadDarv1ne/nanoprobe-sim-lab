"""
API для управления SSTV станцией и расписанием пролётов МКС.
Интеграция с pysstv для декодирования и кэшированием Redis.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

# Добавляем корень проекта в path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.redis_cache import RedisCache
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

# Глобальные объекты
_redis_cache: Optional[RedisCache] = None
_sstv_decoder: Optional[SSTVDecoder] = None
_tracker: Optional[tracker_module.SatelliteTracker] = None


def get_redis_cache() -> Optional[RedisCache]:
    """Получает Redis cache instance."""
    global _redis_cache
    if _redis_cache is None and REDIS_AVAILABLE:
        try:
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", "6379"))
            _redis_cache = RedisCache(host=redis_host, port=redis_port)
        except Exception:
            _redis_cache = None
    return _redis_cache


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
        except Exception:
            _tracker = tracker_module.SatelliteTracker()
    return _tracker


def get_sstv_decoder() -> Optional[SSTVDecoder]:
    """Получает SSTVDecoder instance."""
    global _sstv_decoder
    if _sstv_decoder is None and SSTVDecoder is not None:
        _sstv_decoder = SSTVDecoder(mode='auto')
    return _sstv_decoder


# ============================================================================
# API Endpoints: Расписание пролётов
# ============================================================================

@router.get("/iss/schedule")
async def get_iss_schedule(
    hours_ahead: int = 24,
    min_elevation: float = 10.0
):
    """
    Получает расписание пролётов МКС (ISS).
    
    - **hours_ahead**: На сколько часов вперёд (максимум 72)
    - **min_elevation**: Минимальная высота над горизонтом (градусы)
    
    Returns:
        Список пролётов с временем, высотой и частотой
    """
    hours_ahead = min(hours_ahead, 72)  # Максимум 3 дня
    
    tracker = get_satellite_tracker()
    if not tracker:
        raise HTTPException(status_code=503, detail="Satellite tracker not available")
    
    # Проверяем кэш
    cache_key = f"iss_schedule:{hours_ahead}:{min_elevation}"
    redis_cache = get_redis_cache()
    
    if redis_cache and REDIS_AVAILABLE:
        cached = await redis_cache.get(cache_key)
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
            await redis_cache.set(cache_key, result, expire=300)
        
        return {
            "status": "success",
            "cached": False,
            "count": len(result),
            "data": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting schedule: {str(e)}")


@router.get("/iss/next-pass")
async def get_iss_next_pass():
    """
    Получает следующий пролёт МКС.
    
    Returns:
        Информация о следующем пролёте
    """
    tracker = get_satellite_tracker()
    if not tracker:
        raise HTTPException(status_code=503, detail="Satellite tracker not available")
    
    # Проверяем кэш
    cache_key = "iss_next_pass"
    redis_cache = get_redis_cache()
    
    if redis_cache and REDIS_AVAILABLE:
        cached = await redis_cache.get(cache_key)
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
            await redis_cache.set(cache_key, result, expire=120)
        
        return {
            "status": "success",
            "cached": False,
            "data": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/iss/position")
async def get_iss_current_position():
    """
    Получает текущую позицию МКС.
    
    Returns:
        Текущая позиция (широта, долгота, высота)
    """
    tracker = get_satellite_tracker()
    if not tracker:
        raise HTTPException(status_code=503, detail="Satellite tracker not available")
    
    # Проверяем кэш (позиция обновляется каждые 30 секунд)
    cache_key = "iss_position"
    redis_cache = get_redis_cache()
    
    if redis_cache and REDIS_AVAILABLE:
        cached = await redis_cache.get(cache_key)
        if cached:
            return {
                "status": "success",
                "cached": True,
                "data": cached
            }
    
    try:
        position = tracker.get_current_position('iss')
        
        if not position:
            raise HTTPException(status_code=404, detail="ISS position not available")
        
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
            await redis_cache.set(cache_key, result, expire=30)
        
        return {
            "status": "success",
            "cached": False,
            "data": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/iss/visible")
async def is_iss_visible(min_elevation: float = 10.0):
    """
    Проверяет видимость МКС сейчас.
    
    Returns:
        Статус видимости и информация
    """
    tracker = get_satellite_tracker()
    if not tracker:
        raise HTTPException(status_code=503, detail="Satellite tracker not available")
    
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
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


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
        raise HTTPException(status_code=503, detail="SSTV decoder not available")
    
    decoder = get_sstv_decoder()
    if not decoder:
        raise HTTPException(status_code=503, detail="SSTV decoder initialization failed")
    
    try:
        # Если путь не указан, используем тестовый
        if not audio_path:
            test_audio = Path("data/sstv_test_audio.wav")
            if test_audio.exists():
                audio_path = str(test_audio)
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No audio file provided and no test file found"
                )
        
        # Проверяем существование файла
        if not Path(audio_path).exists():
            raise HTTPException(status_code=404, detail=f"Audio file not found: {audio_path}")
        
        # Декодируем
        decoder.mode = mode
        image = decoder.decode_from_audio(audio_path)
        
        if not image:
            raise HTTPException(status_code=400, detail="Failed to decode SSTV")
        
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
        raise HTTPException(status_code=500, detail=f"Decoding error: {str(e)}")


@router.get("/sstv/download/{filename}")
async def download_sstv_image(filename: str):
    """Скачивает декодированное SSTV изображение."""
    file_path = Path("output/sstv") / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
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
        raise HTTPException(status_code=503, detail="Satellite tracker not available")
    
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
        raise HTTPException(status_code=503, detail="Satellite tracker not available")
    
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
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


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
