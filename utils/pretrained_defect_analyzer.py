# -*- coding: utf-8 -*-
"""
AI/ML улучшения для анализа дефектов
Pre-trained модели и transfer learning
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PretrainedDefectAnalyzer:
    """
    Анализатор дефектов на основе pre-trained моделей
    Поддерживает:
    - ResNet50
    - EfficientNet
    - MobileNetV2
    """

    MODEL_TYPES = {
        'resnet50': {'input_size': (224, 224), 'pretrained': True},
        'efficientnet': {'input_size': (224, 224), 'pretrained': True},
        'mobilenet': {'input_size': (224, 224), 'pretrained': True},
    }

    DEFECT_CLASSES = [
        'normal',
        'scratch',
        'crack',
        'pit',
        'inclusion',
        'void',
        'contamination',
        'roughness',
    ]

    def __init__(self, model_type: str = 'resnet50', use_gpu: bool = False):
        """
        Инициализация анализатора

        Args:
            model_type: Тип модели (resnet50, efficientnet, mobilenet)
            use_gpu: Использовать GPU (требует CUDA)
        """
        self.model_type = model_type
        self.use_gpu = use_gpu and self._check_gpu_available()
        self.model = None
        self.preprocessor = None
        self._model_loaded = False

    def _check_gpu_available(self) -> bool:
        """Проверка доступности GPU"""
        try:
            import tensorflow as tf
            return len(tf.config.list_physical_devices('GPU')) > 0
        except ImportError:
            return False

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Загрузка pre-trained модели

        Args:
            model_path: Путь к кастомной модели (опционально)

        Returns:
            bool: Успешность загрузки
        """
        try:
            if model_path and Path(model_path).exists():
                # Загрузка кастомной модели
                return self._load_custom_model(model_path)
            else:
                # Загрузка pre-trained модели
                return self._load_pretrained_model()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _load_pretrained_model(self) -> bool:
        """Загрузка pre-trained модели с дообучением"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers

            # Базовая модель
            if self.model_type == 'resnet50':
                base_model = keras.applications.ResNet50(
                    include_top=False,
                    weights='imagenet',
                    input_shape=(224, 224, 3),
                    pooling='avg'
                )
            elif self.model_type == 'efficientnet':
                base_model = keras.applications.EfficientNetB0(
                    include_top=False,
                    weights='imagenet',
                    input_shape=(224, 224, 3),
                    pooling='avg'
                )
            elif self.model_type == 'mobilenet':
                base_model = keras.applications.MobileNetV2(
                    include_top=False,
                    weights='imagenet',
                    input_shape=(224, 224, 3),
                    pooling='avg'
                )
            else:
                base_model = keras.applications.ResNet50(
                    include_top=False,
                    weights='imagenet',
                    input_shape=(224, 224, 3),
                    pooling='avg'
                )

            # Заморозка базовой модели для transfer learning
            base_model.trainable = False

            # Добавление кастомных слоёв для классификации дефектов
            inputs = keras.Input(shape=(224, 224, 3))
            x = base_model(inputs, training=False)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
            outputs = layers.Dense(len(self.DEFECT_CLASSES), activation='softmax')(x)

            self.model = keras.Model(inputs, outputs)

            # Компиляция модели
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
            )

            # Preprocessor
            if self.model_type == 'resnet50':
                self.preprocessor = keras.applications.resnet50.preprocess_input
            elif self.model_type == 'efficientnet':
                self.preprocessor = keras.applications.efficientnet.preprocess_input
            elif self.model_type == 'mobilenet':
                self.preprocessor = keras.applications.mobilenet_v2.preprocess_input

            self._model_loaded = True
            logger.info(f"Loaded pretrained {self.model_type} model for defect analysis")
            return True

        except Exception as e:
            logger.error(f"Failed to load pretrained model: {e}")
            return False

    def _load_custom_model(self, model_path: str) -> bool:
        """Загрузка кастомной модели"""
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(model_path)
            self.preprocessor = None  # Используем встроенный preprocessing
            self._model_loaded = True
            logger.info(f"Loaded custom model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load custom model: {e}")
            return False

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Анализ изображения на дефекты

        Args:
            image_path: Путь к изображению

        Returns:
            Dict с результатами анализа
        """
        if not self._model_loaded:
            if not self.load_model():
                return {
                    'success': False,
                    'error': 'Model not loaded',
                    'defect_type': None,
                    'confidence': 0.0,
                }

        try:
            import tensorflow as tf
            from tensorflow import keras
            import cv2

            # Загрузка и предобработка изображения
            image = cv2.imread(str(image_path))
            if image is None:
                return {
                    'success': False,
                    'error': 'Failed to load image',
                }

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            image_array = np.expand_dims(image, axis=0)

            # Preprocessing
            if self.preprocessor:
                image_array = self.preprocessor(image_array)

            # Предсказание
            predictions = self.model.predict(image_array, verbose=0)
            probabilities = predictions[0]

            # Получение результатов
            predicted_class_idx = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class_idx])
            defect_type = self.DEFECT_CLASSES[predicted_class_idx]

            # Все вероятности
            all_probabilities = {
                cls: float(prob)
                for cls, prob in zip(self.DEFECT_CLASSES, probabilities)
            }

            return {
                'success': True,
                'defect_type': defect_type,
                'confidence': confidence,
                'all_probabilities': all_probabilities,
                'image_path': str(image_path),
                'timestamp': datetime.now().isoformat(),
                'model_type': self.model_type,
            }

        except Exception as e:
            logger.error(f"Failed to analyze image: {e}")
            return {
                'success': False,
                'error': str(e),
                'defect_type': None,
                'confidence': 0.0,
            }

    def analyze_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Пакетный анализ изображений

        Args:
            image_paths: Список путей к изображениям

        Returns:
            Список результатов анализа
        """
        results = []
        for path in image_paths:
            result = self.analyze_image(path)
            results.append(result)
        return results

    def fine_tune(
        self,
        train_data_path: str,
        epochs: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Дообучение модели на кастомных данных

        Args:
            train_data_path: Путь к обучающим данным
            epochs: Количество эпох
            batch_size: Размер батча
            validation_split: Доля валидационных данных

        Returns:
            История обучения
        """
        if not self._model_loaded:
            return {'success': False, 'error': 'Model not loaded'}

        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras.preprocessing.image import ImageDataGenerator

            # Разморозка последних слоёв базовой модели для fine-tuning
            self.model.trainable = True
            for layer in self.model.layers[:-20]:
                layer.trainable = False

            # Перекомпиляция с меньшим learning rate
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            # Подготовка данных
            datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                validation_split=validation_split
            )

            train_generator = datagen.flow_from_directory(
                train_data_path,
                target_size=(224, 224),
                batch_size=batch_size,
                class_mode='categorical',
                subset='training'
            )

            val_generator = datagen.flow_from_directory(
                train_data_path,
                target_size=(224, 224),
                batch_size=batch_size,
                class_mode='categorical',
                subset='validation'
            )

            # Обучение
            history = self.model.fit(
                train_generator,
                epochs=epochs,
                validation_data=val_generator
            )

            return {
                'success': True,
                'epochs': epochs,
                'final_accuracy': float(history.history['accuracy'][-1]),
                'final_loss': float(history.history['loss'][-1]),
                'val_accuracy': float(history.history['val_accuracy'][-1]),
                'history': {
                    'accuracy': [float(x) for x in history.history['accuracy']],
                    'loss': [float(x) for x in history.history['loss']],
                }
            }

        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            return {'success': False, 'error': str(e)}

    def save_model(self, save_path: str) -> bool:
        """
        Сохранение модели

        Args:
            save_path: Путь для сохранения

        Returns:
            bool: Успешность сохранения
        """
        if not self._model_loaded or self.model is None:
            return False

        try:
            self.model.save(save_path)
            logger.info(f"Model saved to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о модели"""
        return {
            'model_type': self.model_type,
            'loaded': self._model_loaded,
            'use_gpu': self.use_gpu,
            'defect_classes': self.DEFECT_CLASSES,
            'input_size': self.MODEL_TYPES.get(self.model_type, {}).get('input_size'),
        }


# Singleton instance
_analyzer_instance: Optional[PretrainedDefectAnalyzer] = None


def get_analyzer(model_type: str = 'resnet50') -> PretrainedDefectAnalyzer:
    """Получение экземпляра анализатора"""
    global _analyzer_instance
    if _analyzer_instance is None or _analyzer_instance.model_type != model_type:
        _analyzer_instance = PretrainedDefectAnalyzer(model_type=model_type)
    return _analyzer_instance
