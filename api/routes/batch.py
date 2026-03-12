# -*- coding: utf-8 -*-
"""
API роуты для пакетной обработки
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from typing import List, Optional, Dict, Any
from datetime import datetime

from api.schemas import ErrorResponse, BatchJobCreate, BatchJobResponse
from utils.batch_processor import BatchProcessor


router = APIRouter()


def get_batch_processor() -> BatchProcessor:
    """Зависимость для получения процессора пакетной обработки"""
    return BatchProcessor()


@router.get(
    "/jobs",
    summary="Получить все задания",
    description="Получение списка всех заданий с фильтрацией по статусу",
)
async def get_all_jobs(
    batch_status: Optional[str] = Query(default=None, description="Фильтр по статусу"),
    processor: BatchProcessor = Depends(get_batch_processor),
):
    """Получить все задания"""
    jobs = processor.get_all_jobs(status=batch_status)
    return {"jobs": jobs, "total": len(jobs)}


@router.get(
    "/jobs/{job_id}",
    summary="Получить задание по ID",
    description="Получение детальной информации о задании",
    responses={
        200: {"description": "Успешный ответ"},
        404: {"model": ErrorResponse, "description": "Задание не найдено"},
    },
)
async def get_job(
    job_id: str,
    processor: BatchProcessor = Depends(get_batch_processor),
):
    """Получить задание по ID"""
    job = processor.get_job_status(job_id)
    if "error" in job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=job["error"],
        )
    return job


@router.get(
    "/jobs/{job_id}/stats",
    summary="Получить статистику задания",
    description="Получение подробной статистики задания",
)
async def get_job_stats(
    job_id: str,
    processor: BatchProcessor = Depends(get_batch_processor),
):
    """Получить статистику задания"""
    stats = processor.get_job_stats(job_id)
    if stats is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Задание {job_id} не найдено",
        )
    return stats


@router.get(
    "/statistics",
    summary="Статистика пакетной обработки",
    description="Получение общей статистики по всем заданиям",
)
async def get_statistics(processor: BatchProcessor = Depends(get_batch_processor)):
    """Получить статистику"""
    return processor.get_statistics()


@router.post(
    "/jobs/{job_id}/cancel",
    summary="Отменить задание",
    description="Отмена выполнения задания",
)
async def cancel_job(
    job_id: str,
    processor: BatchProcessor = Depends(get_batch_processor),
):
    """Отменить задание"""
    success = processor.cancel_job(job_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Не удалось отменить задание (возможно, оно уже выполняется или завершено)",
        )
    return {"success": True, "job_id": job_id}


@router.post(
    "/jobs/{job_id}/pause",
    summary="Приостановить задание",
    description="Приостановка выполнения задания",
)
async def pause_job(
    job_id: str,
    processor: BatchProcessor = Depends(get_batch_processor),
):
    """Приостановить задание"""
    success = processor.pause_job(job_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Не удалось приостановить задание (возможно, оно не выполняется)",
        )
    return {"success": True, "job_id": job_id}


@router.post(
    "/jobs/{job_id}/resume",
    summary="Возобновить задание",
    description="Возобновление выполнения задания",
)
async def resume_job(
    job_id: str,
    processor: BatchProcessor = Depends(get_batch_processor),
):
    """Возобновить задание"""
    success = processor.resume_job(job_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Не удалось возобновить задание (возможно, оно не приостановлено)",
        )
    return {"success": True, "job_id": job_id}


@router.post(
    "/process/images",
    summary="Пакетная обработка изображений",
    description="Создание задания на пакетную обработку изображений",
)
async def process_images(
    image_paths: List[str] = Body(..., description="Пути к изображениям"),
    operation: str = Body(default="analyze", description="Тип операции (analyze, resize, convert)"),
    parameters: Dict[str, Any] = Body(default={}, description="Параметры операции"),
    processor: BatchProcessor = Depends(get_batch_processor),
):
    """Пакетная обработка изображений"""
    job_id = processor.process_image_batch(
        image_paths=image_paths,
        operation=operation,
        parameters=parameters,
    )
    return {"success": True, "job_id": job_id}


@router.post(
    "/process/surface",
    summary="Пакетный анализ поверхностей",
    description="Создание задания на анализ поверхностей",
)
async def process_surface(
    surface_data: List[Dict[str, Any]] = Body(..., description="Данные поверхностей"),
    analysis_type: str = Body(default="statistics", description="Тип анализа"),
    processor: BatchProcessor = Depends(get_batch_processor),
):
    """Пакетный анализ поверхностей"""
    job_id = processor.process_surface_analysis_batch(
        surface_data_list=surface_data,
        analysis_type=analysis_type,
    )
    return {"success": True, "job_id": job_id}
