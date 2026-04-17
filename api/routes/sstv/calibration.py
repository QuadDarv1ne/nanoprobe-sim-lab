"""SSTV Calibration API Router

Автоматическая калибровка PPM для RTL-SDR устройств.
"""

import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/calibration", tags=["SSTV Calibration"])

# Pydantic Models


class CalibrationRequest(BaseModel):
    """Запрос на автоматическую калибровку PPM."""

    frequency_mhz: float = Field(
        ...,
        description="Известная частота в МГц (например, 100.0 для FM радио)",
        ge=50,
        le=2000,
    )
    duration: int = Field(default=10, description="Длительность теста в секундах", ge=5, le=60)
    device_index: int = Field(default=0, description="Индекс RTL-SDR устройства", ge=0, le=7)


class CalibrationResponse(BaseModel):
    """Ответ с результатами калибровки."""

    success: bool
    ppm_error: float
    frequency_mhz: float
    duration_seconds: int
    timestamp: str
    device_index: int
    message: str


class CalibrationStatus(BaseModel):
    """Статус текущей калибровки."""

    has_calibration: bool
    ppm: int
    is_valid: bool
    last_calibration: str | None = None
    calibration_data: Dict[str, Any] = {}


# Helper Functions


def _get_calibrator():
    """Получение экземпляра RTLSDRCalibrator."""
    try:
        from utils.sdr.rtl_sdr_calibration import RTLSDRCalibrator

        return RTLSDRCalibrator(calibration_file="config/device_calibration.json")
    except ImportError:
        logger.error("rtl_sdr_calibration module not found")
        return None


def _get_calibration_file_path() -> Path:
    """Получение пути к файлу калибровки."""
    return Path("config/device_calibration.json")


# API Endpoints


@router.get("/status", response_model=CalibrationStatus, summary="Получить статус калибровки")
async def get_calibration_status(device_index: int = Query(0, ge=0, le=7)):
    """Получить текущий статус калибровки PPM.

    Возвращает информацию о наличии валидной калибровки, PPM значении
    и дате последней калибровки.
    """
    calibrator = _get_calibrator()
    if not calibrator:
        raise HTTPException(status_code=500, detail="Модуль калибровки недоступен")

    try:
        info = calibrator.get_calibration_info()
        last_calib = None
        if info["data"] and "timestamp" in info["data"]:
            last_calib = info["data"]["timestamp"]

        return CalibrationStatus(
            has_calibration=info["has_calibration"],
            ppm=info["ppm"],
            is_valid=info["is_valid"],
            last_calibration=last_calib,
            calibration_data=info["data"],
        )
    except Exception as e:
        logger.error("Ошибка получения статуса калибровки: %s", e)
        raise HTTPException(status_code=500, detail=f"Ошибка получения статуса: {str(e)}")


@router.post(
    "/automated", response_model=CalibrationResponse, summary="Автоматическая PPM калибровка"
)
async def automated_ppm_calibration(request: CalibrationRequest):
    """Выполнить автоматическую PPM калибровку с использованием rtl_test -p.

    Использует известную частоту (например, FM радио станцию) для точного
    определения PPM отклонения.
    """
    calibrator = _get_calibrator()
    if not calibrator:
        raise HTTPException(
            status_code=500,
            detail="Модуль калибровки недоступен. Убедитесь, что rtl_sdr установлен.",
        )

    try:
        logger.info(
            "Запуск автоматической калибровки: freq=%.3f MHz, duration=%ds",
            request.frequency_mhz,
            request.duration,
        )

        result = calibrator.automated_ppm_calibration(
            known_frequency=request.frequency_mhz * 1e6,
            duration=request.duration,
        )

        if result.get("success"):
            return CalibrationResponse(
                success=True,
                ppm_error=result.get("ppm", 0.0),
                frequency_mhz=request.frequency_mhz,
                duration_seconds=request.duration,
                timestamp=datetime.now(timezone.utc).isoformat(),
                device_index=request.device_index,
                message=f"Калибровка успешна: PPM={result.get('ppm', 0.0):.2f}",
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Неизвестная ошибка калибровки"),
            )

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="rtl_test не найден. Установите rtl-sdr-tools.",
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=408,
            detail="Таймаут калибровки. Увеличьте duration или проверьте устройство.",
        )
    except Exception as e:
        logger.exception("Ошибка калибровки: %s", e)
        raise HTTPException(status_code=500, detail=f"Ошибка калибровки: {str(e)}")


@router.post("/reset", summary="Сбросить калибровку")
async def reset_calibration(device_index: int = Query(0, ge=0, le=7)):
    """Сбросить сохранённую калибровку PPM.

    Удаляет файл config/device_calibration.json.
    """
    calibrator = _get_calibrator()
    if not calibrator:
        raise HTTPException(status_code=500, detail="Модуль калибровки недоступен")

    try:
        calibrator.reset_calibration()
        return {
            "success": True,
            "message": "Калибровка успешно сброшена",
            "device_index": device_index,
        }
    except Exception as e:
        logger.error("Ошибка сброса калибровки: %s", e)
        raise HTTPException(status_code=500, detail=f"Ошибка сброса: {str(e)}")


@router.get("/file", summary="Получить файл калибровки")
async def get_calibration_file():
    """Получить файл калибровки (device_calibration.json)."""
    calib_path = _get_calibration_file_path()
    if not calib_path.exists():
        raise HTTPException(status_code=404, detail="Файл калибровки не найден")

    try:
        import json

        with open(calib_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Ошибка чтения файла калибровки: %s", e)
        raise HTTPException(status_code=500, detail=f"Ошибка чтения файла: {str(e)}")
