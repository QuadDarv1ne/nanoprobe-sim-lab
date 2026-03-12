# -*- coding: utf-8 -*-
"""
API роуты для сравнения поверхностей
"""

from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime
import uuid
from pathlib import Path

from api.schemas import (
    SurfaceComparisonRequest,
    SurfaceComparisonResponse,
    ComparisonMetrics,
    ErrorResponse,
)
from utils.database import DatabaseManager
from utils.surface_comparator import SurfaceComparator


router = APIRouter()


def get_db() -> DatabaseManager:
    """Зависимость для получения менеджера БД"""
    from api.main import db_manager
    return db_manager


@router.post(
    "",
    response_model=SurfaceComparisonResponse,
    summary="Сравнить две поверхности",
    description="Сравнение двух изображений поверхностей с вычислением метрик (SSIM, PSNR, MSE)",
)
async def compare_surfaces(
    request: SurfaceComparisonRequest,
    db: DatabaseManager = Depends(get_db),
):
    """Сравнить две поверхности"""
    from api.metrics import BusinessMetrics

    try:
        # Проверка существования файлов
        image1_path = Path(request.image1_path)
        image2_path = Path(request.image2_path)

        if not image1_path.exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Файл не найден: {request.image1_path}",
            )

        if not image2_path.exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Файл не найден: {request.image2_path}",
            )

        # Создание компаратора
        comparator = SurfaceComparator()

        # Сравнение
        result = comparator.compare(
            str(image1_path),
            str(image2_path),
            save_results=True,
        )

        # Бизнес-метрики
        BusinessMetrics.inc_comparison()
        
        # Генерация ID
        comparison_id = f"comp_{uuid.uuid4().hex[:8]}"
        
        # Сохранение в БД
        if hasattr(db, 'add_surface_comparison'):
            db.add_surface_comparison(
                comparison_id=comparison_id,
                image1_path=str(image1_path),
                image2_path=str(image2_path),
                similarity_score=result.get('similarity', 0),
                difference_map_path=result.get('difference_map_path'),
                metrics={
                    'ssim': result.get('ssim', 0),
                    'psnr': result.get('psnr', 0),
                    'mse': result.get('mse', 0),
                    'similarity': result.get('similarity', 0),
                },
            )
        
        return SurfaceComparisonResponse(
            comparison_id=comparison_id,
            image1_path=str(image1_path),
            image2_path=str(image2_path),
            similarity_score=result.get('similarity', 0),
            metrics=ComparisonMetrics(
                ssim=result.get('ssim', 0),
                psnr=result.get('psnr', 0),
                mse=result.get('mse', 0),
                similarity=result.get('similarity', 0),
                pearson=result.get('pearson', 0),
            ),
            difference_map_path=result.get('difference_map_path'),
            created_at=datetime.now().isoformat(),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка сравнения поверхностей: {str(e)}",
        )


@router.get(
    "/history",
    summary="История сравнений",
    description="Получить историю всех сравнений поверхностей",
)
async def get_comparison_history(
    limit: int = 50,
    db: DatabaseManager = Depends(get_db),
):
    """История сравнений"""
    try:
        if hasattr(db, 'get_surface_comparisons'):
            comparisons = db.get_surface_comparisons(limit=limit)
            return {
                "items": comparisons,
                "total": len(comparisons),
                "limit": limit,
            }
        else:
            return {"items": [], "total": 0, "limit": limit, "message": "Метод не реализован"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка получения истории: {str(e)}",
        )


@router.get(
    "/{comparison_id}",
    summary="Получить результат сравнения по ID",
)
async def get_comparison(
    comparison_id: str,
    db: DatabaseManager = Depends(get_db),
):
    """Получить результат сравнения по ID"""
    try:
        if hasattr(db, 'get_surface_comparisons'):
            comparisons = db.get_surface_comparisons(limit=100)
            comparison = next((c for c in comparisons if c.get('comparison_id') == comparison_id), None)
            
            if not comparison:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Сравнение с ID {comparison_id} не найдено",
                )
            
            return comparison
        else:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Метод не реализован",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка получения сравнения: {str(e)}",
        )
