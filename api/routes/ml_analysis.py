"""
AI/ML Analysis routes с pre-trained моделями
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pathlib import Path
import logging

from api.error_handlers import ValidationError, DatabaseError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml", tags=["AI/ML Analysis"])


@router.post(
    "/analyze",
    summary="Анализ изображения с pre-trained моделью",
    description="Анализ изображения на дефекты с использованием pre-trained модели",
)
async def analyze_with_pretrained(
    image: UploadFile = File(..., description="Изображение для анализа"),
    model_type: str = Form(default="resnet50", description="Тип модели: resnet50, efficientnet, mobilenet")
):
    """
    Анализ изображения на дефекты

    Поддерживаемые типы дефектов:
    - normal (без дефектов)
    - scratch (царапины)
    - crack (трещины)
    - pit (углубления)
    - inclusion (включения)
    - void (пустоты)
    - contamination (загрязнения)
    - roughness (шероховатость)
    """
    from utils.ai.pretrained_defect_analyzer import get_analyzer

    # Sanitize filename to prevent path traversal
    safe_filename = Path(image.filename).name
    if not safe_filename or safe_filename.startswith('.'):
        raise ValidationError("Invalid filename")

    # Save temporary file
    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / safe_filename

    try:
        with open(temp_path, "wb") as f:
            content = await image.read()
            f.write(content)

        # Анализ
        analyzer = get_analyzer(model_type=model_type)
        result = analyzer.analyze_image(str(temp_path))

        if not result.get('success'):
            raise DatabaseError(result.get('error', 'Analysis failed'))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ML analysis error: {e}")
        raise DatabaseError(f"Analysis error: {str(e)}")
    finally:
        # Очистка временного файла
        if temp_path.exists():
            temp_path.unlink()


@router.get(
    "/models",
    summary="Список доступных моделей",
    description="Получить информацию о доступных ML моделях",
)
async def get_available_models():
    """Информация о доступных моделях"""
    from utils.ai.pretrained_defect_analyzer import PretrainedDefectAnalyzer

    models_info = {}
    for model_type in PretrainedDefectAnalyzer.MODEL_TYPES:
        analyzer = PretrainedDefectAnalyzer(model_type=model_type)
        models_info[model_type] = analyzer.get_model_info()

    return {
        "models": models_info,
        "defect_classes": PretrainedDefectAnalyzer.DEFECT_CLASSES,
    }


@router.post(
    "/fine-tune",
    summary="Дообучение модели",
    description="Fine-tuning модели на кастомных данных",
)
async def fine_tune_model(
    model_type: str = Form(default="resnet50"),
    epochs: int = Form(default=10, ge=1, le=100),
    batch_size: int = Form(default=32, ge=8, le=128),
    validation_split: float = Form(default=0.2, ge=0.1, le=0.5),
):
    """
    Дообучение модели на кастомных данных

    Требуется директория с данными в формате:
    data/train/
        class_1/
            image1.jpg
            image2.jpg
        class_2/
            image3.jpg
    """
    from utils.ai.pretrained_defect_analyzer import get_analyzer

    train_path = Path("data/ml_train")

    if not train_path.exists():
        raise ValidationError(f"Training data not found at {train_path}")

    analyzer = get_analyzer(model_type=model_type)

    if not analyzer.load_model():
        raise DatabaseError("Failed to load model")

    result = analyzer.fine_tune(
        train_data_path=str(train_path),
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split
    )

    if not result.get('success'):
        raise DatabaseError(result.get('error', 'Fine-tuning failed'))

    return result


@router.post(
    "/save-model",
    summary="Сохранение модели",
    description="Сохранение обученной модели",
)
async def save_model(
    model_path: str = Form(..., description="Путь для сохранения модели"),
    model_type: str = Form(default="resnet50"),
):
    """Сохранение модели"""
    from utils.ai.pretrained_defect_analyzer import get_analyzer

    analyzer = get_analyzer(model_type=model_type)

    if not analyzer._model_loaded:
        raise ValidationError("Model not loaded")

    success = analyzer.save_model(model_path)

    if not success:
        raise DatabaseError("Failed to save model")

    return {
        "success": True,
        "model_path": model_path,
        "model_type": model_type,
    }


@router.get(
    "/batch-analyze",
    summary="Пакетный анализ",
    description="Анализ нескольких изображений",
)
async def batch_analyze(
    image_paths: str = Form(..., description="JSON список путей к изображениям"),
    model_type: str = Form(default="resnet50"),
):
    """Пакетный анализ изображений"""
    import json
    from utils.ai.pretrained_defect_analyzer import get_analyzer

    try:
        paths = json.loads(image_paths)
    except json.JSONDecodeError:
        raise ValidationError("Invalid JSON in image_paths")

    analyzer = get_analyzer(model_type=model_type)
    results = analyzer.analyze_batch(paths)

    return {
        "results": results,
        "total": len(results),
        "success_count": sum(1 for r in results if r.get('success')),
    }
