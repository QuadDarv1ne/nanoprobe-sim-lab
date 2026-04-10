"""
API роуты для анализа дефектов и изображений
"""

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends

from api.dependencies import get_db
from api.error_handlers import NotFoundError, ValidationError
from api.schemas import DefectAnalysisRequest, DefectAnalysisResponse, DefectInfo
from utils.database import DatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/defects",
    response_model=DefectAnalysisResponse,
    summary="Анализ дефектов на изображении",
    description="AI/ML анализ дефектов с использованием Isolation Forest или KMeans",
)
async def analyze_image_defects(
    request: DefectAnalysisRequest,
    db: DatabaseManager = Depends(get_db),
):
    """Анализ дефектов на изображении"""
    # Lazy import - ML model loading
    from api.metrics import BusinessMetrics
    from utils.ai.defect_analyzer import DefectAnalysisPipeline

    try:
        # Проверка существования файла
        image_path = Path(request.image_path)
        if not image_path.exists():
            raise ValidationError(
                f"Файл не найден: {request.image_path}",
                details={"path": request.image_path}
            )

        # Создание пайплайна
        pipeline = DefectAnalysisPipeline(db_manager=db)

        # Анализ
        result = pipeline.analyze_image(
            image_path=str(image_path),
            model_name=request.model_name,
            save_results=True,
        )

        # Бизнес-метрики
        defects_list = result.get('defects', [])
        BusinessMetrics.inc_defect_analysis(request.model_name, defects_list)

        # Конвертация дефектов в формат схемы
        defects = [
            DefectInfo(
                type=d.get('type', 'unknown'),
                x=d.get('x', 0),
                y=d.get('y', 0),
                width=d.get('width', 0),
                height=d.get('height', 0),
                area=d.get('area', 0),
                confidence=d.get('confidence', 0),
            )
            for d in result.get('defects', [])
        ]

        # Вычисление средней уверенности
        confidence_score = (
            sum(d.confidence for d in defects) / len(defects)
            if defects else 0.0
        )

        return DefectAnalysisResponse(
            analysis_id=result.get('analysis_id', f"defect_{uuid.uuid4().hex[:8]}"),
            image_path=str(image_path),
            model_name=request.model_name,
            defects_count=len(defects),
            defects=defects,
            confidence_score=confidence_score,
            processing_time_ms=result.get('processing_time_ms', 0),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    except (ValidationError, NotFoundError):
        raise
    except Exception as e:
        logger.error(f"Error analyzing image defects: {e}")
        raise ValidationError(f"Ошибка анализа дефектов: {str(e)}")


@router.get(
    "/defects/history",
    summary="История анализов дефектов",
)
async def get_defect_history(
    limit: int = 50,
    db: DatabaseManager = Depends(get_db),
):
    """История анализов дефектов"""
    try:
        analyses = db.get_defect_analyses(limit=limit)
        return {"items": analyses, "total": len(analyses), "limit": limit}
    except Exception as e:
        logger.error(f"Error getting defect analysis history: {e}")
        raise ValidationError(f"Ошибка получения истории: {str(e)}")


@router.get(
    "/defects/{analysis_id}",
    summary="Получить результат анализа по ID",
)
async def get_defect_analysis(
    analysis_id: str,
    db: DatabaseManager = Depends(get_db),
):
    """Получить результат анализа по ID (UUID или числовой id)"""
    try:
        analyses = db.get_defect_analyses(limit=500)
        analysis = next(
            (a for a in analyses if
             a.get('analysis_id') == analysis_id or str(a.get('id')) == analysis_id),
            None
        )
        if not analysis:
            raise NotFoundError(f"Анализ с ID {analysis_id} не найден")
        return analysis
    except (ValidationError, NotFoundError):
        raise
    except Exception as e:
        logger.error(f"Error getting defect analysis by ID: {e}")
        raise ValidationError(f"Ошибка получения анализа: {str(e)}")


@router.delete(
    "/defects/{analysis_id}",
    status_code=204,
    summary="Удалить результат анализа",
)
async def delete_defect_analysis(
    analysis_id: str,
    db: DatabaseManager = Depends(get_db),
):
    """Удалить результат анализа из БД (принимает analysis_id UUID или числовой id)"""
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM defect_analysis WHERE analysis_id = ? OR CAST(id AS TEXT) = ?",
            (analysis_id, analysis_id)
        )
        if cursor.rowcount == 0:
            raise NotFoundError(f"Анализ с ID {analysis_id} не найден", resource_type="analysis")


@router.get(
    "/defects/{analysis_id}/export",
    summary="Экспорт результата анализа",
)
async def export_defect_analysis(
    analysis_id: str,
    fmt: str = "json",
    db: DatabaseManager = Depends(get_db),
):
    """Экспорт результата анализа в JSON или CSV (принимает analysis_id UUID или числовой id)"""
    analyses = db.get_defect_analyses(limit=500)
    analysis = next(
        (a for a in analyses if
         a.get('analysis_id') == analysis_id or str(a.get('id')) == analysis_id),
        None
    )

    if not analysis:
        raise NotFoundError(f"Анализ с ID {analysis_id} не найден", resource_type="analysis")

    if fmt == "csv":
        import csv
        import io
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=analysis.keys())
        writer.writeheader()
        writer.writerow(analysis)
        from fastapi.responses import Response
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=analysis_{analysis_id}.csv"},
        )

    return analysis
