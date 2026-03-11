# -*- coding: utf-8 -*-
"""
API роуты для анализа дефектов и изображений
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from pathlib import Path
from datetime import datetime
import uuid

from api.schemas import (
    DefectAnalysisRequest,
    DefectAnalysisResponse,
    DefectInfo,
    ErrorResponse,
)
from utils.database import DatabaseManager
from utils.defect_analyzer import DefectAnalysisPipeline, analyze_defects


router = APIRouter()


def get_db() -> DatabaseManager:
    """Зависимость для получения менеджера БД"""
    from api.main import db_manager
    return db_manager


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
    try:
        # Проверка существования файла
        image_path = Path(request.image_path)
        if not image_path.exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Файл не найден: {request.image_path}",
            )
        
        # Создание пайплайна
        pipeline = DefectAnalysisPipeline(db_manager=db)
        
        # Анализ
        result = pipeline.analyze_image(
            image_path=str(image_path),
            model_name=request.model_name,
            save_results=True,
        )
        
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
            timestamp=datetime.now().isoformat(),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка анализа дефектов: {str(e)}",
        )


@router.get(
    "/defects/history",
    summary="История анализов дефектов",
    description="Получить историю всех анализов дефектов",
)
async def get_defect_history(
    limit: int = 50,
    db: DatabaseManager = Depends(get_db),
):
    """История анализов дефектов"""
    try:
        if hasattr(db, 'get_defect_analyses'):
            analyses = db.get_defect_analyses(limit=limit)
            return {
                "items": analyses,
                "total": len(analyses),
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
    "/defects/{analysis_id}",
    summary="Получить результат анализа по ID",
)
async def get_defect_analysis(
    analysis_id: str,
    db: DatabaseManager = Depends(get_db),
):
    """Получить результат анализа по ID"""
    try:
        # Поиск в БД
        if hasattr(db, 'get_defect_analyses'):
            analyses = db.get_defect_analyses(limit=100)
            analysis = next((a for a in analyses if a.get('analysis_id') == analysis_id), None)
            
            if not analysis:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Анализ с ID {analysis_id} не найден",
                )
            
            return analysis
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
            detail=f"Ошибка получения анализа: {str(e)}",
        )
