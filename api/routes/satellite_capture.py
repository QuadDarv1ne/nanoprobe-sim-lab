"""Satellite Auto-Capture API Router

Автоматический захват спутниковых сигналов:
- NOAA APT (погода)
- Meteor LRPT (погода)

Использует предсказания пролётов спутников для автоматического запуска записи.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/satellite-capture", tags=["Satellite Auto-Capture"])

# Pydantic Models


class NOAACaptureRequest(BaseModel):
    """Запрос на захват NOAA APT."""

    satellite: str = Field(..., description="Название спутника: noaa_15, noaa_18, noaa_19")
    frequency_mhz: float = Field(..., description="Частота в МГц (например, 137.620)")
    duration_seconds: int = Field(
        ..., description="Длительность захвата в секундах", ge=60, le=1200
    )


class MeteorCaptureRequest(BaseModel):
    """Запрос на захват Meteor LRPT."""

    satellite: str = Field(default="meteor_m2_3", description="Название спутника")
    frequency_mhz: float = Field(default=137.1, description="Частота в МГц (по умолчанию 137.1)")
    duration_seconds: int = Field(
        ..., description="Длительность захвата в секундах", ge=60, le=1200
    )


class CaptureStatusResponse(BaseModel):
    """Ответ со статусом захвата."""

    capturing: bool
    satellite: str | None
    frequency_mhz: float | None
    start_time: str | None
    duration_seconds: int | None
    output_file: str | None
    scheduled_passes: int
    message: str


class ScheduleRequest(BaseModel):
    """Запрос на планирование захвата."""

    satellite: str
    pass_info: Dict[str, Any]


# Helper Functions


def _get_noaa_manager():
    """Получение экземпляра NOAACaptureManager."""
    try:
        from utils.sdr.noaa_capture import NOAACaptureManager

        return NOAACaptureManager()
    except ImportError:
        logger.error("noaa_capture module not found")
        return None


def _get_meteor_manager():
    """Получение экземпляра MeteorCaptureManager."""
    try:
        from utils.sdr.meteor_capture import MeteorCaptureManager

        return MeteorCaptureManager()
    except ImportError:
        logger.error("meteor_capture module not found")
        return None


# NOAA Endpoints


@router.post("/noaa/start", response_model=CaptureStatusResponse, summary="Начать захват NOAA APT")
async def start_noaa_capture(request: NOAACaptureRequest):
    """Начать автоматический захват NOAA APT изображения.

    Запускает rtl_fm для записи сигнала с NOAA спутника.
    """
    manager = _get_noaa_manager()
    if not manager:
        raise HTTPException(status_code=500, detail="Модуль NOAA захвата недоступен")

    try:
        success = manager.start_capture(
            frequency=request.frequency_mhz * 1e6,  # Конвертируем в Гц
            duration=request.duration_seconds,
            satellite_name=request.satellite,
        )

        if success:
            status = manager.get_capture_status()
            return CaptureStatusResponse(
                capturing=status["capturing"],
                satellite=status.get("satellite"),
                frequency_mhz=status.get("frequency_mhz"),
                start_time=status.get("start_time"),
                duration_seconds=status.get("duration_seconds"),
                output_file=status.get("output_file"),
                scheduled_passes=status.get("scheduled_passes", 0),
                message=f"Захват NOAA {request.satellite} начат",
            )
        else:
            raise HTTPException(status_code=500, detail="Не удалось начать захват")

    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="rtl_fm не найден. Установите rtl-sdr-tools.")
    except Exception as e:
        logger.exception("Ошибка запуска NOAA захвата: %s", e)
        raise HTTPException(status_code=500, detail=f"Ошибка захвата: {str(e)}")


@router.post(
    "/noaa/stop", response_model=CaptureStatusResponse, summary="Остановить захват NOAA APT"
)
async def stop_noaa_capture():
    """Остановить текущий захват NOAA APT."""
    manager = _get_noaa_manager()
    if not manager:
        raise HTTPException(status_code=500, detail="Модуль NOAA захвата недоступен")

    try:
        output_file = manager.stop_capture()
        status = manager.get_capture_status()
        message = f"Захват сохранён в {output_file}" if output_file else "Захват остановлен"

        return CaptureStatusResponse(
            capturing=status["capturing"],
            satellite=status.get("satellite"),
            frequency_mhz=status.get("frequency_mhz"),
            start_time=status.get("start_time"),
            duration_seconds=status.get("duration_seconds"),
            output_file=output_file,
            scheduled_passes=status.get("scheduled_passes", 0),
            message=message,
        )
    except Exception as e:
        logger.error("Ошибка остановки NOAA захвата: %s", e)
        raise HTTPException(status_code=500, detail=f"Ошибка остановки: {str(e)}")


@router.get("/noaa/status", response_model=CaptureStatusResponse, summary="Статус NOAA захвата")
async def get_noaa_status():
    """Получить текущий статус NOAA захвата."""
    manager = _get_noaa_manager()
    if not manager:
        raise HTTPException(status_code=500, detail="Модуль NOAA захвата недоступен")

    try:
        status = manager.get_capture_status()
        return CaptureStatusResponse(
            capturing=status["capturing"],
            satellite=status.get("satellite"),
            frequency_mhz=status.get("frequency_mhz"),
            start_time=status.get("start_time"),
            duration_seconds=status.get("duration_seconds"),
            output_file=status.get("output_file"),
            scheduled_passes=status.get("scheduled_passes", 0),
            message="NOAA захват активен" if status["capturing"] else "NOAA захват не активен",
        )
    except Exception as e:
        logger.error("Ошибка получения статуса NOAA: %s", e)
        raise HTTPException(status_code=500, detail=f"Ошибка статуса: {str(e)}")


@router.post(
    "/noaa/schedule", response_model=CaptureStatusResponse, summary="Запланировать захват NOAA"
)
async def schedule_noaa_capture(request: ScheduleRequest):
    """Запланировать автоматический захват NOAA при пролёте спутника."""
    manager = _get_noaa_manager()
    if not manager:
        raise HTTPException(status_code=500, detail="Модуль NOAA захвата недоступен")

    try:
        success = manager.schedule_auto_capture(
            satellite_name=request.satellite,
            pass_info=request.pass_info,
        )

        if success:
            status = manager.get_capture_status()
            return CaptureStatusResponse(
                capturing=status["capturing"],
                satellite=request.satellite,
                frequency_mhz=None,
                start_time=None,
                duration_seconds=None,
                output_file=None,
                scheduled_passes=status.get("scheduled_passes", 0),
                message=f"Захват запланирован для {request.satellite}",
            )
        else:
            raise HTTPException(status_code=400, detail="Не удалось запланировать захват")

    except Exception as e:
        logger.error("Ошибка планирования NOAA: %s", e)
        raise HTTPException(status_code=500, detail=f"Ошибка планирования: {str(e)}")


# Meteor Endpoints


@router.post(
    "/meteor/start", response_model=CaptureStatusResponse, summary="Начать захват Meteor LRPT"
)
async def start_meteor_capture(request: MeteorCaptureRequest):
    """Начать автоматический захват Meteor LRPT изображения.

    Запускает rtl_fm для записи сигнала с Meteor спутника.
    """
    manager = _get_meteor_manager()
    if not manager:
        raise HTTPException(status_code=500, detail="Модуль Meteor захвата недоступен")

    try:
        success = manager.start_capture(
            frequency=request.frequency_mhz * 1e6,
            duration=request.duration_seconds,
            satellite_name=request.satellite,
        )

        if success:
            status = manager.get_capture_status()
            return CaptureStatusResponse(
                capturing=status["capturing"],
                satellite=status.get("satellite"),
                frequency_mhz=status.get("frequency_mhz"),
                start_time=status.get("start_time"),
                duration_seconds=status.get("duration_seconds"),
                output_file=status.get("output_file"),
                scheduled_passes=status.get("scheduled_passes", 0),
                message=f"Захват Meteor {request.satellite} начат",
            )
        else:
            raise HTTPException(status_code=500, detail="Не удалось начать захват")

    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="rtl_fm не найден. Установите rtl-sdr-tools.")
    except Exception as e:
        logger.exception("Ошибка запуска Meteor захвата: %s", e)
        raise HTTPException(status_code=500, detail=f"Ошибка захвата: {str(e)}")


@router.post(
    "/meteor/stop", response_model=CaptureStatusResponse, summary="Остановить захват Meteor LRPT"
)
async def stop_meteor_capture():
    """Остановить текущий захват Meteor LRPT."""
    manager = _get_meteor_manager()
    if not manager:
        raise HTTPException(status_code=500, detail="Модуль Meteor захвата недоступен")

    try:
        output_file = manager.stop_capture()
        status = manager.get_capture_status()
        message = f"Захват сохранён в {output_file}" if output_file else "Захват остановлен"

        return CaptureStatusResponse(
            capturing=status["capturing"],
            satellite=status.get("satellite"),
            frequency_mhz=status.get("frequency_mhz"),
            start_time=status.get("start_time"),
            duration_seconds=status.get("duration_seconds"),
            output_file=output_file,
            scheduled_passes=status.get("scheduled_passes", 0),
            message=message,
        )
    except Exception as e:
        logger.error("Ошибка остановки Meteor захвата: %s", e)
        raise HTTPException(status_code=500, detail=f"Ошибка остановки: {str(e)}")


@router.get("/meteor/status", response_model=CaptureStatusResponse, summary="Статус Meteor захвата")
async def get_meteor_status():
    """Получить текущий статус Meteor захвата."""
    manager = _get_meteor_manager()
    if not manager:
        raise HTTPException(status_code=500, detail="Модуль Meteor захвата недоступен")

    try:
        status = manager.get_capture_status()
        return CaptureStatusResponse(
            capturing=status["capturing"],
            satellite=status.get("satellite"),
            frequency_mhz=status.get("frequency_mhz"),
            start_time=status.get("start_time"),
            duration_seconds=status.get("duration_seconds"),
            output_file=status.get("output_file"),
            scheduled_passes=status.get("scheduled_passes", 0),
            message="Meteor захват активен" if status["capturing"] else "Meteor захват не активен",
        )
    except Exception as e:
        logger.error("Ошибка получения статуса Meteor: %s", e)
        raise HTTPException(status_code=500, detail=f"Ошибка статуса: {str(e)}")


@router.post(
    "/meteor/schedule", response_model=CaptureStatusResponse, summary="Запланировать захват Meteor"
)
async def schedule_meteor_capture(request: ScheduleRequest):
    """Запланировать автоматический захват Meteor при пролёте спутника."""
    manager = _get_meteor_manager()
    if not manager:
        raise HTTPException(status_code=500, detail="Модуль Meteor захвата недоступен")

    try:
        success = manager.schedule_auto_capture(
            satellite_name=request.satellite,
            pass_info=request.pass_info,
        )

        if success:
            status = manager.get_capture_status()
            return CaptureStatusResponse(
                capturing=status["capturing"],
                satellite=request.satellite,
                frequency_mhz=None,
                start_time=None,
                duration_seconds=None,
                output_file=None,
                scheduled_passes=status.get("scheduled_passes", 0),
                message=f"Захват запланирован для {request.satellite}",
            )
        else:
            raise HTTPException(status_code=400, detail="Не удалось запланировать захват")

    except Exception as e:
        logger.error("Ошибка планирования Meteor: %s", e)
        raise HTTPException(status_code=500, detail=f"Ошибка планирования: {str(e)}")
