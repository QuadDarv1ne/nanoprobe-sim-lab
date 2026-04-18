"""SSTV Calibration API Router

Автоматическая калибровка PPM для RTL-SDR устройств.

Поддерживаемые методы:
- rtl_test: Использование rtl_test -p для оценки PPM
- signal: Калибровка по известному сигналу (FM радио)
- auto: Автоматический выбор метода с фоллбэком
"""

import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/calibration", tags=["SSTV Calibration"])


# Pydantic Models
class CalibrationRequest(BaseModel):
    """Запрос на автоматическую калибровку PPM."""

    method: str = Field(
        default="auto",
        pattern="^(rtl_test|signal|auto)$",
        description="Метод калибровки: rtl_test, signal или auto",
    )
    frequency_mhz: Optional[float] = Field(
        default=None,
        description="Опорная частота в МГц (для метода signal)",
        ge=50,
        le=2000,
    )
    duration: int = Field(
        default=10,
        description="Длительность теста в секундах",
        ge=5,
        le=60,
    )
    device_index: int = Field(
        default=0,
        description="Индекс RTL-SDR устройства",
        ge=0,
        le=7,
    )
    ppm_range: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Ожидаемый диапазон PPM для валидации",
    )


class CalibrationResponse(BaseModel):
    """Ответ с результатами калибровки."""

    success: bool
    ppm_error: float
    frequency_mhz: Optional[float]
    duration_seconds: Optional[int]
    timestamp: str
    device_index: int
    method: str
    confidence: float
    message: str


class CalibrationStatus(BaseModel):
    """Статус текущей калибровки."""

    has_calibration: bool
    ppm: Optional[float]
    is_valid: bool
    method: Optional[str] = None
    confidence: Optional[float] = None
    last_calibration: str | None = None
    calibration_data: Dict[str, Any] = {}


class DeviceInfo(BaseModel):
    """Информация об RTL-SDR устройстве."""

    index: int
    manufacturer: str
    product: str
    serial: str


# Helper Functions
def _get_calibrator(device_index: int = 0):
    """Получение экземпляра RTLSDRAutoCalibration."""
    try:
        from utils.sdr.rtl_sdr_auto_calibration import RTLSDRAutoCalibration

        return RTLSDRAutoCalibration(
            device_index=device_index, calibration_file="config/device_calibration.json"
        )
    except ImportError:
        logger.error("rtl_sdr_auto_calibration module not found")
        return None


def _get_calibration_file_path() -> Path:
    """Получение пути к файлу калибровки."""
    return Path("config/device_calibration.json")


# API Endpoints
@router.get("/status", response_model=CalibrationStatus, summary="Получить статус калибровки")
async def get_calibration_status(device_index: int = Query(0, ge=0, le=7)):
    """
    Получить текущий статус калибровки PPM.

    Возвращает информацию о наличии валидной калибровки, PPM значении
    и дате последней калибровки.
    """
    calibrator = _get_calibrator(device_index)
    if not calibrator:
        raise HTTPException(status_code=500, detail="Модуль калибровки недоступен")

    try:
        info = calibrator.get_calibration_info()
        return CalibrationStatus(
            has_calibration=info["has_calibration"],
            ppm=info["ppm"],
            is_valid=calibrator.is_calibration_valid(),
            method=info.get("method"),
            confidence=info.get("confidence"),
            last_calibration=info.get("timestamp"),
            calibration_data=info,
        )
    except Exception as e:
        logger.error("Ошибка получения статуса калибровки: %s", e)
        raise HTTPException(status_code=500, detail=f"Ошибка получения статуса: {str(e)}")


@router.post(
    "/automated", response_model=CalibrationResponse, summary="Автоматическая PPM калибровка"
)
async def automated_ppm_calibration(request: CalibrationRequest):
    """
    Выполнить автоматическую PPM калибровку.

    Поддерживаемые методы:
    - rtl_test: Использование rtl_test -p для оценки PPM
    - signal: Калибровка по известному сигналу (требуется frequency_mhz)
    - auto: Автоматический выбор метода с фоллбэком
    """
    calibrator = _get_calibrator(request.device_index)
    if not calibrator:
        raise HTTPException(
            status_code=500,
            detail="Модуль калибровки недоступен. Убедитесь, что rtl_sdr установлен.",
        )

    try:
        logger.info(
            "Запуск автоматической калибровки: method=%s, device=%d, "
            "freq=%s MHz, duration=%ds, ppm_range=%d",
            request.method,
            request.device_index,
            request.frequency_mhz,
            request.duration,
            request.ppm_range,
        )

        ppm = None
        method_used = request.method

        if request.method == "rtl_test":
            ppm = calibrator.calibrate_with_rtl_test(
                ppm_range=request.ppm_range,
                duration=request.duration,
            )
        elif request.method == "signal":
            if not request.frequency_mhz:
                raise HTTPException(
                    status_code=400,
                    detail="frequency_mhz требуется для метода signal",
                )
            ppm = calibrator.calibrate_with_signal(
                reference_freq_mhz=request.frequency_mhz,
                duration=request.duration,
            )
        else:  # auto
            ppm = calibrator.calibrate_auto(
                reference_freq_mhz=request.frequency_mhz,
                ppm_range=request.ppm_range,
            )

        if ppm is not None:
            cal_info = calibrator.get_calibration_info()
            return CalibrationResponse(
                success=True,
                ppm_error=cal_info["ppm"],
                frequency_mhz=request.frequency_mhz,
                duration_seconds=request.duration,
                timestamp=datetime.now(timezone.utc).isoformat(),
                device_index=request.device_index,
                method=cal_info.get("method", method_used),
                confidence=cal_info.get("confidence", 0.8),
                message=f"Калибровка успешна: PPM={cal_info['ppm']:.2f}",
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Калибровка не удалась. Проверьте устройство и доступность rtl_test.",
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
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Ошибка калибровки: %s", e)
        raise HTTPException(status_code=500, detail=f"Ошибка калибровки: {str(e)}")


@router.get("/current", summary="Получить текущую калибровку")
async def get_current_calibration(device_index: int = Query(0, ge=0, le=7)):
    """
    Получить текущую калибровку для устройства.

    Возвращает PPM значение и метаданные калибровки.
    """
    calibrator = _get_calibrator(device_index)
    if not calibrator:
        raise HTTPException(status_code=500, detail="Модуль калибровки недоступен")

    try:
        ppm = calibrator.get_calibration()
        if ppm is None:
            raise HTTPException(status_code=404, detail="Калибровка не найдена")

        cal_info = calibrator.get_calibration_info()
        return {
            "ppm": ppm,
            "device": device_index,
            "method": cal_info.get("method"),
            "confidence": cal_info.get("confidence"),
            "timestamp": cal_info.get("timestamp"),
            "is_valid": calibrator.is_calibration_valid(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Ошибка получения калибровки: %s", e)
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


@router.post("/reset", summary="Сбросить калибровку")
async def reset_calibration(device_index: int = Query(0, ge=0, le=7)):
    """
    Сбросить сохранённую калибровку PPM.

    Удаляет калибровочные данные для указанного устройства.
    """
    calibrator = _get_calibrator(device_index)
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
    """
    Получить файл калибровки (device_calibration.json).

    Возвращает содержимое файла калибровки для всех устройств.
    """
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


@router.get("/devices", response_model=list[DeviceInfo], summary="Список устройств")
async def get_rtl_sdr_devices():
    """
    Получить список доступных RTL-SDR устройств.

    Возвращает информацию о каждом устройстве:
    - index: Индекс устройства
    - manufacturer: Производитель
    - product: Название продукта
    - serial: Серийный номер
    """
    try:
        from utils.sdr.rtl_sdr_auto_calibration import get_rtl_sdr_devices as get_devices

        devices = get_devices()

        if not devices:
            return []

        return [
            DeviceInfo(
                index=dev["index"],
                manufacturer=dev.get("manufacturer", ""),
                product=dev.get("product", ""),
                serial=dev.get("serial", ""),
            )
            for dev in devices
        ]
    except Exception as e:
        logger.error("Ошибка получения списка устройств: %s", e)
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")
