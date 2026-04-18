"""
SSTV Satellite Capture API Router

API для автоматического захвата спутниковых сигналов (NOAA APT, METEOR LRPT).
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/satellites", tags=["SSTV Satellites"])


# Pydantic Models
class SatellitePassResponse(BaseModel):
    """Ответ с информацией о пролёте спутника."""

    satellite: str
    aos: str
    los: str
    max_elevation: float
    frequency_mhz: float
    mode: str
    duration_seconds: int
    azimuth_aos: float
    azimuth_los: float
    time_to_aos: Optional[float] = None


class CaptureConfig(BaseModel):
    """Конфигурация захвата."""

    device_index: int = Field(default=0, ge=0, le=7)
    sample_rate: int = Field(default=2400000, description="Частота дискретизации")
    output_dir: str = Field(default="data/satellite_captures")
    min_elevation: float = Field(default=5.0, ge=0, le=90)
    pre_record_offset: int = Field(default=120, description="Предзапись до AOS (сек)")
    post_record_offset: int = Field(default=60, description="Дописывание после LOS (сек)")


class SchedulerStatus(BaseModel):
    """Статус планировщика."""

    running: bool
    active_pass: Optional[Dict[str, Any]] = None
    upcoming_passes: int = 0
    next_pass: Optional[SatellitePassResponse] = None
    last_update: str


class StartSchedulerRequest(BaseModel):
    """Запрос на запуск планировщика."""

    hours_ahead: int = Field(default=24, ge=1, le=168)


# Helper Functions
def _get_satellite_capture(device_index: int = 0):
    """Получение экземпляра SatelliteAutoCapture."""
    try:
        from utils.sdr.satellite_auto_capture import SatelliteAutoCapture

        return SatelliteAutoCapture(
            location_lat=55.75,
            location_lon=37.61,
            output_dir="data/satellite_captures",
            device_index=device_index,
        )
    except ImportError:
        logger.error("satellite_auto_capture module not found")
        return None
    except Exception as e:
        logger.error("Ошибка инициализации SatelliteAutoCapture: %s", e)
        return None


# API Endpoints
@router.get("/passes", summary="Предсказать пролёты")
async def predict_passes(
    hours_ahead: int = Query(48, ge=1, le=168),
    satellite: Optional[str] = Query(None),
    device_index: int = Query(0, ge=0, le=7),
):
    """Предсказать пролёты спутников на ближайшие N часов."""
    capture = _get_satellite_capture(device_index)
    if not capture:
        raise HTTPException(status_code=500, detail="Модуль захвата недоступен")

    try:
        passes = capture.predict_passes(hours_ahead=hours_ahead)
        if satellite:
            passes = [p for p in passes if p.satellite == satellite]

        now = datetime.now()
        pass_responses = []
        for p in passes:
            pass_responses.append(
                SatellitePassResponse(
                    satellite=p.satellite,
                    aos=p.aos.isoformat(),
                    los=p.los.isoformat(),
                    max_elevation=p.max_elevation,
                    frequency_mhz=p.frequency_mhz,
                    mode=p.mode,
                    duration_seconds=p.duration_seconds(),
                    azimuth_aos=p.azimuth_aos,
                    azimuth_los=p.azimuth_los,
                    time_to_aos=p.time_to_aos() if p.aos > now else None,
                )
            )

        return {"passes": pass_responses, "total_count": len(pass_responses)}
    except Exception as e:
        logger.error("Ошибка предсказания пролётов: %s", e)
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


@router.get("/status", summary="Статус планировщика")
async def get_scheduler_status(device_index: int = Query(0, ge=0, le=7)):
    """Получить статус планировщика автозахвата."""
    capture = _get_satellite_capture(device_index)
    if not capture:
        raise HTTPException(status_code=500, detail="Модуль захвата недоступен")

    try:
        capture.predict_passes(hours_ahead=24)
        summary = capture.get_passes_summary()
        return {
            "running": capture._running,
            "upcoming_passes": summary.get("upcoming", 0),
            "active_pass": summary.get("active_pass"),
            "last_update": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error("Ошибка получения статуса: %s", e)
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


@router.post("/scheduler/start", summary="Запустить планировщик")
async def start_scheduler(
    request: StartSchedulerRequest,
    device_index: int = Query(0, ge=0, le=7),
):
    """Запустить планировщик автозахвата спутников."""
    capture = _get_satellite_capture(device_index)
    if not capture:
        raise HTTPException(status_code=500, detail="Модуль захвата недоступен")

    try:
        capture.predict_passes(hours_ahead=request.hours_ahead)
        capture.start_scheduler()
        return {"success": True, "message": "Планировщик запущен"}
    except Exception as e:
        logger.error("Ошибка запуска планировщика: %s", e)
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


@router.post("/scheduler/stop", summary="Остановить планировщик")
async def stop_scheduler(device_index: int = Query(0, ge=0, le=7)):
    """Остановить планировщик автозахвата спутников."""
    capture = _get_satellite_capture(device_index)
    if not capture:
        raise HTTPException(status_code=500, detail="Модуль захвата недоступен")

    try:
        capture.stop_scheduler()
        return {"success": True, "message": "Планировщик остановлен"}
    except Exception as e:
        logger.error("Ошибка остановки планировщика: %s", e)
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


@router.get("/config", response_model=CaptureConfig, summary="Конфигурация захвата")
async def get_capture_config():
    """Получить текущую конфигурацию захвата спутников."""
    return CaptureConfig(
        device_index=0,
        sample_rate=2400000,
        output_dir="data/satellite_captures",
        min_elevation=5.0,
        pre_record_offset=120,
        post_record_offset=60,
    )


@router.post("/config", summary="Обновить конфигурацию захвата")
async def update_capture_config(config: CaptureConfig):
    """Обновить конфигурацию захвата спутников."""
    logger.info("Обновлена конфигурация захвата: device=%d", config.device_index)
    return {"success": True, "message": "Конфигурация обновлена"}


@router.get("/supported", summary="Поддерживаемые спутники")
async def get_supported_satellites():
    """Получить список поддерживаемых спутников."""
    from utils.sdr.satellite_auto_capture import SatelliteAutoCapture

    satellites = SatelliteAutoCapture.SATELLITES
    return {
        "satellites": [
            {"name": name, "frequency_mhz": cfg["freq"], "mode": cfg["mode"]}
            for name, cfg in satellites.items()
        ]
    }


@router.get("/captures", summary="Список записей")
async def get_captures(
    satellite: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
):
    """Получить список сохранённых записей спутников."""
    output_dir = Path("data/satellite_captures")

    if not output_dir.exists():
        return {"captures": [], "total": 0}

    try:
        captures = []
        for file in output_dir.glob("*.raw"):
            filename = file.name
            parts = filename.replace(".raw", "").split("_")
            if len(parts) >= 4:
                sat_name = parts[0]
                if satellite and sat_name != satellite:
                    continue
                try:
                    timestamp = datetime.strptime(parts[1] + "_" + parts[2], "%Y%m%d_%H%M%S")
                except (ValueError, IndexError):
                    timestamp = datetime.fromtimestamp(file.stat().st_mtime)

                captures.append(
                    {
                        "filename": filename,
                        "satellite": sat_name,
                        "timestamp": timestamp.isoformat(),
                        "size_bytes": file.stat().st_size,
                    }
                )

        captures.sort(key=lambda x: x["timestamp"], reverse=True)
        return {"captures": captures[:limit], "total": len(captures)}
    except Exception as e:
        logger.error("Ошибка получения списка записей: %s", e)
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


@router.delete("/captures/{filename}", summary="Удалить запись")
async def delete_capture(filename: str):
    """Удалить запись спутника."""
    output_dir = Path("data/satellite_captures")
    file_path = output_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Файл не найден")

    try:
        file_path.unlink()
        return {"success": True, "message": f"Файл {filename} удалён"}
    except Exception as e:
        logger.error("Ошибка удаления файла: %s", e)
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")
