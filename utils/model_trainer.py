# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Модуль обучения моделей машинного обучения для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для обучения,
оценки и оптимизации моделей машинного обучения.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from functools import wraps
import warnings
from sklearn.exceptions import ConvergenceWarning
import xgboost as xgb
import lightgbm as lgb

@dataclass
class ModelResult:
    """Результат обучения модели"""
    model: Any
    metrics: Dict[str, float]
    predictions: np.ndarray
    test_scores: Dict[str, float]
    training_time: float
    feature_importance: Optional[np.ndarray] = None

class ModelTrainer:
    """
    Класс тренера моделей
    Обеспечивает обучение, оценку и
    оптимизацию моделей машинного обучения.
    """


    def __init__(self, output_dir: str = "models"):
        """
        Инициализирует тренер моделей

        Args:
            output_dir: Директория для сохранения моделей
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None

    def prepare_data(self,

                    X: Union[np.ndarray, pd.DataFrame],
                    y: Union[np.ndarray, pd.Series],
                    test_size: float = 0.2,
                    scale_features: bool = True,
                    encode_labels: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Подготавливает данные для обучения

        Args:
            X: Признаки
            y: Целевые значения
            test_size: Размер тестовой выборки
            scale_features: Нормализовать ли признаки
            encode_labels: Кодировать ли метки

        Returns:
            Кортеж (X_train, X_test, y_train, y_test)
        """
        # Преобразуем в numpy массивы если нужно
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Разделяем данные
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Нормализуем признаки
        if scale_features:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        # Кодируем метки если это классификация
        if encode_labels and len(np.unique(y)) <= min(len(y), 20):  # Предполагаем классификацию
            y_train = self.label_encoder.fit_transform(y_train)
            y_test = self.label_encoder.transform(y_test)

        return X_train, X_test, y_train, y_test


    def train_regression_model(self,
                             X_train: np.ndarray,
                             y_train: np.ndarray,
                             model_type: str = "random_forest") -> Any:
        """
        Обучает модель регрессии

        Args:
            X_train: Обучающие признаки
            y_train: Обучающие целевые значения
            model_type: Тип модели

        Returns:
            Обученную модель
        """
        if model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "linear":
            model = LinearRegression()
        elif model_type == "xgboost":
            model = xgb.XGBRegressor(random_state=42)
        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}")

        model.fit(X_train, y_train)
        return model

    def train_classification_model(self,
                                 X_train: np.ndarray,
                                 y_train: np.ndarray,
                                 model_type: str = "random_forest") -> Any:
        """
        Обучает модель классификации

        Args:
            X_train: Обучающие признаки
            y_train: Обучающие целевые значения
            model_type: Тип модели

        Returns:
            Обученную модель
        """
        if model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "logistic":
            model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == "xgboost":
            model = xgb.XGBClassifier(random_state=42)
        elif model_type == "lightgbm":
            model = lgb.LGBMClassifier(random_state=42)
        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}")

        model.fit(X_train, y_train)

        return model

    def evaluate_model(self,
                      model: Any,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      model_type: str = "regression") -> Dict[str, float]:
        """
        Оценивает модель

        Args:
            model: Обученная модель
            X_test: Тестовые признаки
            y_test: Тестовые целевые значения
            model_type: Тип модели

        Returns:
            Словарь с метриками
        """
        predictions = model.predict(X_test)

        if model_type == "regression":
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            metrics = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2_score': r2,
                'mae': np.mean(np.abs(y_test - predictions))
            }
        else:  # classification
            accuracy = accuracy_score(y_test, predictions)
            metrics = {
                'accuracy': accuracy,
                'classification_report': classification_report(y_test, predictions, output_dict=True)
            }

        return metrics

    def hyperparameter_tuning(self,
                            X_train: np.ndarray,
                            y_train: np.ndarray,
                            model_type: str = "random_forest",
                            cv_folds: int = 5) -> Tuple[Any, Dict[str, Any]]:
        """
        Подбирает гиперпараметры модели

        Args:
            X_train: Обучающие признаки
            y_train: Обучающие целевые значения
            model_type: Тип модели
            cv_folds: Количество фолдов для кросс-валидации

        Returns:
            Кортеж (лучшая модель, результаты подбора)
        """
        if model_type == "random_forest":
            if len(np.unique(y_train)) <= min(len(y_train), 20):  # Классификация
                model = RandomForestClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
            else:  # Регрессия
                model = RandomForestRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
        elif model_type == "xgboost":
            if len(np.unique(y_train)) <= min(len(y_train), 20):  # Классификация
                model = xgb.XGBClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            else:  # Регрессия
                model = xgb.XGBRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
        else:
            raise ValueError(f"Гиперпараметрическая настройка не поддерживается для: {model_type}")

        grid_search = GridSearchCV(
            model, param_grid, cv=cv_folds, scoring='accuracy' if
            len(np.unique(y_train)) <= min(len(y_train), 20) else 'r2', n_jobs=-1
        )


        grid_search.fit(X_train, y_train)

        return grid_search.best_estimator_, grid_search.cv_results_

    def train_and_evaluate(self,
                          X: Union[np.ndarray, pd.DataFrame],
                          y: Union[np.ndarray, pd.Series],
                          model_type: str = "random_forest",
                          problem_type: str = "auto") -> ModelResult:
        """
        Обучает и оценивает модель

        Args:
            X: Признаки
            y: Целевые значения
            model_type: Тип модели
            problem_type: Тип задачи ('regression', 'classification', 'auto')

        Returns:
            Результат обучения модели
        """
        import time
        start_time = time.time()

        # Определяем тип задачи автоматически
        if problem_type == "auto":
            if len(np.unique(y)) <= min(len(y), 20):
                problem_type = "classification"
            else:
                problem_type = "regression"

        # Подготавливаем данные
        X_train, X_test, y_train, y_test = self.prepare_data(
            X, y, scale_features=True, encode_labels=(problem_type == "classification")
        )

        # Обучаем модель
        if problem_type == "regression":
            model = self.train_regression_model(X_train, y_train, model_type)
        else:
            model = self.train_classification_model(X_train, y_train, model_type)

        # Делаем предсказания
        predictions = model.predict(X_test)

        # Оцениваем модель
        test_scores = self.evaluate_model(model, X_test, y_test, problem_type)

        # Получаем важность признаков (если доступна)
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_

        training_time = time.time() - start_time

        # Создаем результат
        result = ModelResult(
            model=model,
            metrics={},
            predictions=predictions,
            test_scores=test_scores,
            training_time=training_time,
            feature_importance=feature_importance
        )

        return result


    def save_model(self, model: Any, model_name: str, metadata: Dict[str, Any] = None) -> str:
        """
        Сохраняет модель

        Args:
            model: Обученная модель
            model_name: Имя модели
            metadata: Метаданные модели

        Returns:
            Путь к сохраненной модели
        """
        model_path = self.output_dir / f"{model_name}.joblib"
        scaler_path = self.output_dir / f"{model_name}_scaler.joblib"

        # Сохраняем модель
        joblib.dump(model, model_path)

        # Сохраняем скалер
        joblib.dump(self.scaler, scaler_path)

        # Сохраняем метаданные
        if metadata:
            metadata_path = self.output_dir / f"{model_name}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

        return str(model_path)


    def load_model(self, model_name: str) -> Tuple[Any, Any, Optional[Dict]]:
        """
        Загружает модель

        Args:
            model_name: Имя модели

        Returns:
            Кортеж (модель, скалер, метаданные)
        """
        model_path = self.output_dir / f"{model_name}.joblib"
        scaler_path = self.output_dir / f"{model_name}_scaler.joblib"
        metadata_path = self.output_dir / f"{model_name}_metadata.json"

        # Загружаем модель
        model = joblib.load(model_path)

        # Загружаем скалер
        scaler = joblib.load(scaler_path)

        # Загружаем метаданные
        metadata = None
        if metadata_path.exists():

            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

        return model, scaler, metadata

    def cross_validate_model(self,
                           X: np.ndarray,
                           y: np.ndarray,
                           model_type: str = "random_forest",
                           cv_folds: int = 5) -> Dict[str, Any]:
        """
        Проводит кросс-валидацию модели

        Args:
            X: Признаки
            y: Целевые значения
            model_type: Тип модели
            cv_folds: Количество фолдов

        Returns:
            Словарь с результатами кросс-валидации
        """
        # Определяем тип задачи
        if len(np.unique(y)) <= min(len(y), 20):
            if model_type == "random_forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_type == "logistic":
                model = LogisticRegression(random_state=42, max_iter=1000)
            elif model_type == "xgboost":
                model = xgb.XGBClassifier(random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            scoring = 'accuracy'
        else:
            if model_type == "random_forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_type == "linear":
                model = LinearRegression()
            elif model_type == "xgboost":
                model = xgb.XGBRegressor(random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            scoring = 'r2'

        # Выполняем кросс-валидацию
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)

        results = {
            'cv_scores': scores.tolist(),
            'mean_cv_score': scores.mean(),
            'std_cv_score': scores.std(),

            'min_cv_score': scores.min(),
            'max_cv_score': scores.max()
        }

        return results

    def plot_feature_importance(self,
                               model: Any,
                               feature_names: List[str] = None,
                               top_n: int = 10,
                               output_path: str = None) -> str:
        """
        Строит график важности признаков

        Args:
            model: Обученная модель
            feature_names: Названия признаков
            top_n: Количество признаков для отображения
            output_path: Путь для сохранения графика

        Returns:
            Путь к сохраненному графику
        """
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Модель не имеет атрибута feature_importances_")

        importances = model.feature_importances_

        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(importances))]

        # Создаем DataFrame с важностью признаков
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)

        # Строим график
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title('Важность признаков')
        plt.xlabel('Важность')
        plt.ylabel('Признаки')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            output_path = self.output_dir / f"feature_importance_{timestamp}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        plt.close()

        return str(output_path)

    def plot_predictions_vs_actual(self,
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  output_path: str = None) -> str:
        """
        Строит график предсказанных vs фактических значений

        Args:
            y_true: Фактические значения
            y_pred: Предсказанные значения
            output_path: Путь для сохранения графика

        Returns:
            Путь к сохраненному графику
        """
        plt.figure(figsize=(10, 8))

        # Диаграмма рассеяния
        plt.subplot(2, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Фактические значения')
        plt.ylabel('Предсказанные значения')
        plt.title('Предсказанные vs Фактические')

        # Остатки
        residuals = y_true - y_pred
        plt.subplot(2, 2, 2)
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Предсказанные значения')
        plt.ylabel('Остатки')
        plt.title('Диаграмма остатков')

        # Гистограмма остатков
        plt.subplot(2, 2, 3)
        plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Остатки')
        plt.ylabel('Частота')
        plt.title('Распределение остатков')

        # Q-Q plot
        from scipy import stats
        plt.subplot(2, 2, 4)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q plot остатков')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"predictions_vs_actual_{timestamp}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        plt.close()

        return str(output_path)


def model_training_pipeline(func):
    """
    Декоратор для создания пайплайна обучения модели

    Args:
        func: Функция для декорирования
    """
    @wraps(func)

    def wrapper(*args, **kwargs):
            trainer = ModelTrainer()
        print(f"Запуск пайплайна обучения модели: {func.__name__}")

        # Выполняем функцию
        result = func(trainer, *args, **kwargs)

        print(f"Пайплайн обучения модели завершен: {func.__name__}")
        return result

    return wrapper

def main():
    """Главная функция для демонстрации возможностей тренера моделей"""
    print("=== ТРЕНЕР МОДЕЛЕЙ МАШИННОГО ОБУЧЕНИЯ ПРОЕКТА ===")

    # Создаем тренер моделей
    trainer = ModelTrainer()

    print("✓ Тренер моделей инициализирован")
    print(f"✓ Директория вывода: {trainer.output_dir}")

    # Создаем синтетические данные для демонстрации
    print("\nСоздание синтетических данных...")
    np.random.seed(42)

    # Данные для регрессии
    n_samples = 1000
    n_features = 5
    X_reg = np.random.randn(n_samples, n_features)
    y_reg = (2 * X_reg[:, 0] + 3 * X_reg[:, 1] - X_reg[:, 2] + 0.5 * np.random.randn(n_samples))

    # Данные для классификации
    X_clf = np.random.randn(n_samples, n_features)
    y_clf = (X_clf[:, 0] + X_clf[:, 1] > 0).astype(int)

    print(f"  - Регрессионные данные: {X_reg.shape[0]} образцов, {X_reg.shape[1]} признаков")
    print(f"  - Классификационные данные: {X_clf.shape[0]} образцов, {X_clf.shape[1]} признаков")

    # Обучаем регрессионную модель
    print("\nОбучение регрессионной модели...")
    reg_result = trainer.train_and_evaluate(X_reg, y_reg, model_type="random_forest", problem_type="regression")
    print(f"  - Время обучения: {reg_result.training_time:.2f} с")
    print(f"  - R² Score: {reg_result.test_scores['r2_score']:.4f}")
    print(f"  - RMSE: {reg_result.test_scores['rmse']:.4f}")

    # Обучаем классификационную модель
    print("\nОбучение классификационной модели...")
    clf_result = trainer.train_and_evaluate(X_clf, y_clf, model_type="random_forest", problem_type="classification")
    print(f"  - Время обучения: {clf_result.training_time:.2f} с")
    print(f"  - Accuracy: {clf_result.test_scores['accuracy']:.4f}")

    # Проводим кросс-валидацию
    print("\nКросс-валидация регрессионной модели...")
    cv_results = trainer.cross_validate_model(X_reg, y_reg, model_type="random_forest", cv_folds=5)
    print(f"  - Средняя оценка: {cv_results['mean_cv_score']:.4f}")
    print(f"  - Стандартное отклонение: {cv_results['std_cv_score']:.4f}")

    # Подбираем гиперпараметры
    print("\nПодбор гиперпараметров...")
    best_model, tuning_results = trainer.hyperparameter_tuning(
        X_reg[:500], y_reg[:500], model_type="random_forest", cv_folds=3
    )
    print(f"  - Лучшие параметры: {tuning_results['params'][np.argmax(tuning_results['mean_test_score'])]}")
    print(f"  - Лучший скор: {max(tuning_results['mean_test_score']):.4f}")

    # Сохраняем модель
    print("\nСохранение модели...")
    metadata = {
        'created_at': datetime.now().isoformat(),
        'model_type': 'random_forest_regression',
        'features_count': X_reg.shape[1],
        'samples_count': X_reg.shape[0],
        'r2_score': reg_result.test_scores['r2_score']
    }
    model_path = trainer.save_model(reg_result.model, "regression_model", metadata)
    print(f"  - Модель сохранена: {model_path}")

    # Загружаем модель
    print("\nЗагрузка модели...")
    loaded_model, loaded_scaler, loaded_metadata = trainer.load_model("regression_model")
    print(f"  - Модель загружена: {loaded_model is not None}")
    print(f"  - Метаданные: {loaded_metadata['r2_score'] if loaded_metadata else 'N/A'}")

    # Строим графики
    print("\nСоздание графиков важности признаков...")
    feature_names = [f"Feature_{i}" for i in range(X_reg.shape[1])]
    importance_plot_path = trainer.plot_feature_importance(
        reg_result.model, feature_names, top_n=5
    )
    print(f"  - График важности признаков сохранен: {importance_plot_path}")

    print("\nСоздание графика предсказанных vs фактических значений...")

    predictions_vs_actual_path = trainer.plot_predictions_vs_actual(
        y_reg[:100], reg_result.predictions[:100]
    )
    print(f"  - График предсказанных vs фактических значений сохранен: {predictions_vs_actual_path}")

    # Демонстрируем декоратор пайплайна
    print("\nДемонстрация декоратора пайплайна обучения модели...")

    @model_training_pipeline

    def sample_training_pipeline(trainer_instance):
        # Создаем простую модель
            X_simple = np.random.randn(100, 2)
        y_simple = X_simple[:, 0] + X_simple[:, 1] + np.random.randn(100) * 0.1

        result = trainer_instance.train_and_evaluate(
            X_simple, y_simple, model_type="linear", problem_type="regression"
        )
        return result

    pipeline_result = sample_training_pipeline()
    print(f"  - Результат пайплайна: R² = {pipeline_result.test_scores['r2_score']:.4f}")

    print("\nТренер моделей успешно протестирован")
    print("\nДоступные функции:")
    print("- Обучение моделей: train_and_evaluate()")
    print("- Подбор гиперпараметров: hyperparameter_tuning()")
    print("- Кросс-валидация: cross_validate_model()")
    print("- Сохранение моделей: save_model()")
    print("- Загрузка моделей: load_model()")
    print("- Графики важности признаков: plot_feature_importance()")
    print("- Графики предсказаний: plot_predictions_vs_actual()")
    print("- Декоратор пайплайна: @model_training_pipeline")

if __name__ == "__main__":
    main()

