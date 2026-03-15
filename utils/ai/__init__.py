"""
AI/ML Utilities for Nanoprobe Sim Lab

Модули для искусственного интеллекта и машинного обучения:
- defect_analyzer.py - анализ дефектов
- pretrained_defect_analyzer.py - предобученный анализ
- machine_learning.py - ML утилиты
- model_trainer.py - обучение моделей
- ai_resource_optimizer.py - AI оптимизация
- predictive_analytics_engine.py - предиктивная аналитика
- code_analyzer.py - анализ кода
- visualizer.py - визуализация
- spm_realtime_visualizer.py - SЗМ визуализация
- space_image_downloader.py - загрузка изображений
"""

from utils.ai.defect_analyzer import analyze_defects
from utils.ai.pretrained_defect_analyzer import PretrainedDefectAnalyzer
from utils.ai.machine_learning import MLModel

__all__ = [
    'analyze_defects',
    'PretrainedDefectAnalyzer',
    'MLModel',
]
