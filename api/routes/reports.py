# -*- coding: utf-8 -*-
"""
API роуты для генерации PDF отчётов
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from datetime import datetime
import uuid
from pathlib import Path

from api.schemas import (
    PDFReportRequest,
    PDFReportResponse,
    ReportType,
    ErrorResponse,
)
from utils.database import DatabaseManager
from utils.pdf_report_generator import ScientificPDFReport


router = APIRouter()


def get_db() -> DatabaseManager:
    """Зависимость для получения менеджера БД"""
    from api.main import db_manager
    return db_manager


@router.post(
    "",
    response_model=PDFReportResponse,
    summary="Сгенерировать PDF отчёт",
    description="Генерация научных PDF отчётов для различных типов анализа",
)
async def generate_pdf_report(
    request: PDFReportRequest,
    db: DatabaseManager = Depends(get_db),
):
    """Сгенерировать PDF отчёт"""
    try:
        # Создание генератора отчётов
        report_generator = ScientificPDFReport(output_dir="reports/pdf")
        
        report_id = f"report_{uuid.uuid4().hex[:8]}"
        report_path = None
        
        # Генерация отчёта в зависимости от типа
        if request.report_type == ReportType.SURFACE_ANALYSIS:
            report_path = report_generator.generate_surface_analysis_report(
                surface_data=request.data,
                images=request.images or [],
                title=request.title,
                author=request.author,
            )
        
        elif request.report_type == ReportType.DEFECT_ANALYSIS:
            report_path = report_generator.generate_defect_analysis_report(
                defect_data=request.data,
                defect_images=request.images or [],
                title=request.title,
                author=request.author,
            )
        
        elif request.report_type == ReportType.COMPARISON:
            report_path = report_generator.generate_comparison_report(
                comparison_data=request.data,
                comparison_images=request.images or [],
                title=request.title,
                author=request.author,
            )
        
        elif request.report_type == ReportType.SIMULATION:
            report_path = report_generator.generate_simulation_report(
                simulation_data=request.data,
                result_images=request.images or [],
                title=request.title,
                author=request.author,
            )
        
        elif request.report_type == ReportType.BATCH:
            report_path = report_generator.generate_batch_report(
                batch_data=request.data,
                title=request.title,
            )
        
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Неизвестный тип отчёта: {request.report_type}",
            )
        
        if not report_path:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Не удалось сгенерировать отчёт",
            )
        
        # Получение размера файла
        file_size = Path(report_path).stat().st_size
        
        # Сохранение в БД
        if hasattr(db, 'add_pdf_report'):
            db.add_pdf_report(
                report_path=report_path,
                report_type=request.report_type.value,
                title=request.title,
                file_size_bytes=file_size,
            )
        
        return PDFReportResponse(
            report_id=report_id,
            report_path=report_path,
            report_type=request.report_type.value,
            title=request.title,
            file_size_bytes=file_size,
            pages_count=None,  # Можно добавить подсчёт страниц
            created_at=datetime.now().isoformat(),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка генерации отчёта: {str(e)}",
        )


@router.get(
    "",
    summary="Список PDF отчётов",
    description="Получить список всех сгенерированных отчётов",
)
async def get_reports(
    limit: int = 50,
    db: DatabaseManager = Depends(get_db),
):
    """Список отчётов"""
    try:
        # Получение списка из БД (если метод реализован)
        if hasattr(db, 'get_pdf_reports'):
            reports = db.get_pdf_reports(limit=limit)
            return {
                "items": reports,
                "total": len(reports),
                "limit": limit,
            }
        else:
            # Получение файлов из директории
            reports_dir = Path("reports/pdf")
            if reports_dir.exists():
                reports = [
                    {
                        "report_path": str(f),
                        "name": f.name,
                        "size_bytes": f.stat().st_size,
                        "created_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                    }
                    for f in reports_dir.glob("*.pdf")
                ]
                return {
                    "items": reports[:limit],
                    "total": len(reports),
                    "limit": limit,
                }
            else:
                return {"items": [], "total": 0, "limit": limit}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка получения отчётов: {str(e)}",
        )


@router.get(
    "/{report_id}/download",
    summary="Скачать PDF отчёт",
    description="Скачать PDF файл отчёта",
)
async def download_report(
    report_id: str,
    db: DatabaseManager = Depends(get_db),
):
    """Скачать PDF отчёт"""
    try:
        # Поиск отчёта по ID (в БД или по имени файла)
        reports_dir = Path("reports/pdf")
        
        # Поиск по имени файла
        report_file = None
        for f in reports_dir.glob("*.pdf"):
            if report_id in f.name or f.stem == report_id:
                report_file = f
                break
        
        if not report_file:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Отчёт с ID {report_id} не найден",
            )
        
        return FileResponse(
            path=str(report_file),
            filename=report_file.name,
            media_type="application/pdf",
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка скачивания отчёта: {str(e)}",
        )
