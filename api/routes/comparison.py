"""
API роуты для сравнения поверхностей
"""

from fastapi import APIRouter, Depends
from datetime import datetime, timezone
import uuid
from pathlib import Path
import logging
from api.error_handlers import ValidationError, NotFoundError

from api.schemas import (
    SurfaceComparisonRequest,
    SurfaceComparisonResponse,
    ComparisonMetrics,
)
from api.dependencies import get_db
from utils.database import DatabaseManager


logger = logging.getLogger(__name__)


router = APIRouter()


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
    # Lazy import - heavy ML/scipy dependencies
    from utils.surface_comparator import SurfaceComparator
    from api.metrics import BusinessMetrics

    try:
        # Проверка существования файлов
        image1_path = Path(request.image1_path)
        image2_path = Path(request.image2_path)

        if not image1_path.exists():
            raise ValidationError(f"Файл не найден: {request.image1_path}")

        if not image2_path.exists():
            raise ValidationError(f"Файл не найден: {request.image2_path}")

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
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        raise ValidationError(f"Ошибка сравнения поверхностей: {str(e)}")


@router.get(
    "/history",
    summary="История сравнений",
)
async def get_comparison_history(
    limit: int = 50,
    db: DatabaseManager = Depends(get_db),
):
    """История сравнений поверхностей"""
    try:
        comparisons = db.get_surface_comparisons(limit=limit)
        return {"items": comparisons, "total": len(comparisons), "limit": limit}
    except Exception as e:
        logger.error(f"History error: {e}")
        raise ValidationError(f"Ошибка получения истории: {str(e)}")


@router.get(
    "/{comparison_id}",
    summary="Получить результат сравнения по ID",
)
async def get_comparison(
    comparison_id: str,
    db: DatabaseManager = Depends(get_db),
):
    """Получить результат сравнения по ID (UUID или числовой id)"""
    try:
        comparisons = db.get_surface_comparisons(limit=500)
        comparison = next(
            (c for c in comparisons if
             c.get('comparison_id') == comparison_id or str(c.get('id')) == comparison_id),
            None
        )
        if not comparison:
            raise NotFoundError(f"Сравнение с ID {comparison_id} не найдено", resource_type="comparison")
        return comparison
    except (ValidationError, NotFoundError):
        raise
    except Exception as e:
        raise ValidationError(f"Ошибка получения сравнения: {str(e)}")


@router.delete(
    "/{comparison_id}",
    status_code=204,
    summary="Удалить результат сравнения",
)
async def delete_comparison(
    comparison_id: str,
    db: DatabaseManager = Depends(get_db),
):
    """Удалить результат сравнения из БД (принимает comparison_id UUID или числовой id)"""
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM surface_comparisons WHERE comparison_id = ? OR CAST(id AS TEXT) = ?",
            (comparison_id, comparison_id)
        )
        if cursor.rowcount == 0:
            raise NotFoundError(f"Сравнение с ID {comparison_id} не найдено", resource_type="comparison")


@router.get(
    "/{comparison_id}/export",
    summary="Экспорт результата сравнения",
)
async def export_comparison(
    comparison_id: str,
    fmt: str = "json",
    db: DatabaseManager = Depends(get_db),
):
    """Экспорт результата сравнения в JSON или CSV (принимает comparison_id UUID или числовой id)"""
    comparisons = db.get_surface_comparisons(limit=500)
    comparison = next(
        (c for c in comparisons if
         c.get('comparison_id') == comparison_id or str(c.get('id')) == comparison_id),
        None
    )

    if not comparison:
        raise NotFoundError(f"Сравнение с ID {comparison_id} не найдено", resource_type="comparison")

    if fmt == "csv":
        import csv, io
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=comparison.keys())
        writer.writeheader()
        writer.writerow(comparison)
        from fastapi.responses import Response
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=comparison_{comparison_id}.csv"},
        )

    return comparison
