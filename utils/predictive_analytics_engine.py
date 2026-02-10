# -*- coding: utf-8 -*-
#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3

"""
Модуль предиктивной аналитики для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет систему прогнозирования производительности и
предиктивных рекомендаций по оптимизации на основе машинного обучения.
"""

import time
import threading
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
from dataclasses import dataclass
import pickle
import warnings

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.performance_profiler import PerformanceProfiler
from utils.resource_optimizer import ResourceManager
from utils.advanced_logger_analyzer import AdvancedLoggerAnalyzer
from utils.memory_tracker import MemoryTracker
from utils.performance_benchmark import PerformanceBenchmarkSuite
from utils.optimization_orchestrator import OptimizationOrchestrator
from utils.system_health_monitor import SystemHealthMonitor
from utils.performance_analytics_dashboard import PerformanceAnalyticsDashboard
from utils.performance_monitoring_center import PerformanceMonitoringCenter

@dataclass
class PredictionResult:
    """Результат предсказания"""
    metric: str
    predicted_value: float
    confidence: float  # 0-1
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    time_horizon: timedelta
    recommendation: str

@dataclass
class AnomalyDetectionResult:
    """Результат обнаружения аномалий"""
    timestamp: datetime
    metric: str
    observed_value: float
    expected_value: float
    deviation: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    is_anomaly: bool

class PredictiveAnalyticsEngine:
    """
    Класс предиктивной аналитики
    Обеспечивает прогнозирование производительности и предиктивные рекомендации
    на основе машинного обучения и статистического анализа.
    """


    def __init__(self, output_dir: str = "predictive_analytics"):
        """
        Инициализирует движок предиктивной аналитики

        Args:
            output_dir: Директория для сохранения моделей и результатов
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Инициализируем все инструменты оптимизации
        self.performance_profiler = PerformanceProfiler(output_dir="profiles")
        self.resource_manager = ResourceManager()
        self.logger_analyzer = AdvancedLoggerAnalyzer()
        self.memory_tracker = MemoryTracker(output_dir="memory_logs")
        self.benchmark_suite = PerformanceBenchmarkSuite(output_dir="benchmarks")
        self.orchestrator = OptimizationOrchestrator(output_dir="optimization_reports")
        self.health_monitor = SystemHealthMonitor(output_dir="health_reports")
        self.analytics_dashboard = PerformanceAnalyticsDashboard(output_dir="analytics_reports")
        self.monitoring_center = PerformanceMonitoringCenter(output_dir="performance_monitoring")

        # История данных для обучения
        self.data_history = {}
        self.max_history_length = 10000  # Максимум 10k точек на метрику

        # Модели прогнозирования
        self.models = {}
        self.model_training_needed = set()

        # Параметры прогнозирования
        self.prediction_horizons = [5, 15, 30, 60]  # минуты
        self.confidence_threshold = 0.8
        self.anomaly_threshold_multiplier = 2.0

        # Состояние
        self.active = False
        self.learning_thread = None
        self.prediction_thread = None

        # Статистика
        self.stats = {
            'predictions_made': 0,
            'anomalies_detected': 0,
            'recommendations_applied': 0,
            'models_trained': 0
        }


    def add_data_point(self, metric_name: str, value: float, timestamp: Optional[datetime] = None):
        """
        Добавляет точку данных для обучения

        Args:
            metric_name: Название метрики
            value: Значение метрики
            timestamp: Временная метка (опционально)
        """
        if timestamp is None:
            timestamp = datetime.now()

        if metric_name not in self.data_history:
            self.data_history[metric_name] = []

        self.data_history[metric_name].append((timestamp, value))

        # Ограничиваем размер истории
        if len(self.data_history[metric_name]) > self.max_history_length:
            self.data_history[metric_name] = self.data_history[metric_name][-self.max_history_length:]

        # Отмечаем, что модель нуждается в переобучении
        self.model_training_needed.add(metric_name)


    def get_recent_data(self, metric_name: str, minutes: int = 60) -> List[Tuple[datetime, float]]:
        """
        Получает недавние данные для метрики

        Args:
            metric_name: Название метрики
            minutes: Количество минут для выборки

        Returns:
            Список пар (время, значение)
        """
        if metric_name not in self.data_history:
            return []

        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_data = [(t, v) for t, v in self.data_history[metric_name] if t >= cutoff_time]

        return sorted(recent_data, key=lambda x: x[0])  # Сортируем по времени


    def train_linear_model(self, metric_name: str) -> Optional[Dict[str, float]]:
        """
        Обучает линейную модель для метрики

        Args:
            metric_name: Название метрики

        Returns:
            Параметры модели (slope, intercept, r_squared) или None
        """
        data = self.get_recent_data(metric_name, minutes=120)  # 2 часа для обучения

        if len(data) < 10:  # Минимум 10 точек для обучения
            return None

        # Преобразуем время в числовые значения (секунды с начала)
        start_time = data[0][0]
        x_values = [(t - start_time).total_seconds() for t, _ in data]
        y_values = [v for _, v in data]

        # Линейная регрессия
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)

        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'std_error': std_err,
            'sample_size': len(data)
        }


    def predict_value(self, model_params: Dict[str, float], minutes_ahead: int) -> Tuple[float, float]:
        """
        Прогнозирует значение с помощью модели

        Args:
            model_params: Параметры модели
            minutes_ahead: На сколько минут вперед прогнозировать

        Returns:
            (прогнозируемое значение, доверительный интервал)
        """
        seconds_ahead = minutes_ahead * 60
        predicted_value = model_params['intercept'] + model_params['slope'] * seconds_ahead

        # Оценка доверительного интервала
        confidence_interval = model_params['std_error'] * 1.96  # 95% доверительный интервал

        return predicted_value, confidence_interval


    def detect_anomalies(self, metric_name: str, window_minutes: int = 30) -> List[AnomalyDetectionResult]:
        """
        Обнаруживает аномалии в метрике

        Args:
            metric_name: Название метрики
            window_minutes: Окно для анализа

        Returns:
            Список результатов обнаружения аномалий
        """
        recent_data = self.get_recent_data(metric_name, minutes=window_minutes)
        if len(recent_data) < 10:
            return []

        values = [v for _, v in recent_data]
        mean_val = np.mean(values)
        std_val = np.std(values)

        anomalies = []
        for timestamp, observed_value in recent_data[-5:]:  # Проверяем последние 5 значений
            expected_value = mean_val
            deviation = abs(observed_value - expected_value)

            # Определяем уровень серьезности
            if deviation > self.anomaly_threshold_multiplier * 2 * std_val:
                severity = 'critical'
            elif deviation > self.anomaly_threshold_multiplier * 1.5 * std_val:
                severity = 'high'
            elif deviation > self.anomaly_threshold_multiplier * std_val:
                severity = 'medium'
            elif deviation > self.anomaly_threshold_multiplier * 0.5 * std_val:
                severity = 'low'
            else:
                severity = 'normal'

            is_anomaly = severity != 'normal'

            if is_anomaly:
                anomalies.append(AnomalyDetectionResult(
                    timestamp=timestamp,
                    metric=metric_name,
                    observed_value=observed_value,
                    expected_value=expected_value,
                    deviation=deviation,
                    severity=severity,
                    is_anomaly=is_anomaly
                ))

        return anomalies


    def generate_prediction(self, metric_name: str, minutes_ahead: int) -> Optional[PredictionResult]:
        """
        Генерирует прогноз для метрики

        Args:
            metric_name: Название метрики
            minutes_ahead: На сколько минут вперед прогнозировать

        Returns:
            Результат прогноза или None
        """
        # Обучаем модель, если нужно
        if metric_name in self.model_training_needed:
            model_params = self.train_linear_model(metric_name)
            if model_params:
                self.models[metric_name] = model_params
                self.model_training_needed.discard(metric_name)
                self.stats['models_trained'] += 1
            else:
                # Если не можем обучить линейную модель, используем скользящее среднее
                recent_data = self.get_recent_data(metric_name, minutes=30)
                if recent_data:
                    avg_value = np.mean([v for _, v in recent_data])
                    model_params = {
                        'slope': 0,
                        'intercept': avg_value,
                        'r_squared': 0.1,  # Низкий R² для скользящего среднего
                        'std_error': np.std([v for _, v in recent_data]),
                        'sample_size': len(recent_data)
                    }
                    self.models[metric_name] = model_params

        if metric_name not in self.models:
            return None

        model_params = self.models[metric_name]
        predicted_value, confidence_interval = self.predict_value(model_params, minutes_ahead)

        # Определяем направление тренда
        trend_direction = 'stable'
        if model_params['slope'] > 0.01:  # Порог для возрастающего тренда
            trend_direction = 'increasing'
        elif model_params['slope'] < -0.01:  # Порог для убывающего тренда
            trend_direction = 'decreasing'

        # Вычисляем доверие к прогнозу
        confidence = min(1.0, model_params['r_squared'] + 0.1)  # Добавляем небольшой базовый уровень

        # Генерируем рекомендацию
        recommendation = self._generate_recommendation(metric_name, predicted_value, trend_direction)

        result = PredictionResult(
            metric=metric_name,
            predicted_value=predicted_value,
            confidence=confidence,
            trend_direction=trend_direction,
            time_horizon=timedelta(minutes=minutes_ahead),
            recommendation=recommendation
        )

        self.stats['predictions_made'] += 1

        return result


    def _generate_recommendation(self, metric_name: str, predicted_value: float, trend_direction: str) -> str:
        """
        Генерирует рекомендацию на основе прогноза

        Args:
            metric_name: Название метрики
            predicted_value: Прогнозируемое значение
            trend_direction: Направление тренда

        Returns:
            Текст рекомендации
        """
        if 'cpu' in metric_name.lower():
            if predicted_value > 80 and trend_direction == 'increasing':
                return "Прогнозируется высокая загрузка CPU, рекомендуется запустить оптимизацию ресурсов"
            elif predicted_value > 70 and trend_direction == 'increasing':
                return "Возможен рост загрузки CPU, подготовьтесь к оптимизации"
            elif predicted_value < 20 and trend_direction == 'decreasing':
                return "Низкая загрузка CPU, возможно, можно оптимизировать распределение ресурсов"

        elif 'memory' in metric_name.lower() or 'ram' in metric_name.lower():
            if predicted_value > 85 and trend_direction == 'increasing':
                return "Прогнозируется высокое использование памяти, рекомендуется запустить оптимизацию памяти"
            elif predicted_value > 75 and trend_direction == 'increasing':
                return "Возможен рост использования памяти, подготовьтесь к оптимизации"

        elif 'efficiency' in metric_name.lower() or 'score' in metric_name.lower():
            if predicted_value < 70 and trend_direction == 'decreasing':
                return "Прогнозируется снижение эффективности, рекомендуется запустить комплексную оптимизацию"
            elif predicted_value < 80 and trend_direction == 'decreasing':
                return "Возможен спад эффективности, подготовьтесь к оптимизации"

        return "Прогноз в нормальных пределах, специальных действий не требуется"


    def get_predictive_insights(self) -> Dict[str, Any]:
        """
        Получает предиктивные инсайты

        Returns:
            Словарь с предиктивными инсайтами
        """
        insights = {
            'predictions': {},
            'anomalies': {},
            'trends': {},
            'recommendations': [],
            'confidence_levels': {},
            'timestamp': datetime.now().isoformat()
        }

        # Метрики для прогнозирования
        metrics_to_predict = [
            'cpu_percent', 'memory_percent', 'resource_efficiency',
            'optimization_score', 'active_processes'
        ]

        for metric in metrics_to_predict:
            predictions = {}
            for horizon in [5, 15, 30]:  # Прогнозы на 5, 15 и 30 минут
                pred_result = self.generate_prediction(metric, horizon)
                if pred_result and pred_result.confidence > 0.5:
                    predictions[f'{horizon}_min'] = {
                        'predicted_value': pred_result.predicted_value,
                        'confidence': pred_result.confidence,
                        'trend': pred_result.trend_direction,
                        'recommendation': pred_result.recommendation
                    }

            if predictions:
                insights['predictions'][metric] = predictions
                insights['confidence_levels'][metric] = np.mean([p['confidence'] for p in predictions.values()])

        # Обнаружение аномалий
        for metric in metrics_to_predict[:3]:  # Проверяем только основные метрики
            anomalies = self.detect_anomalies(metric)
            if anomalies:
                insights['anomalies'][metric] = [
                    {
                        'timestamp': a.timestamp.isoformat(),
                        'severity': a.severity,
                        'observed': a.observed_value,
                        'expected': a.expected_value,
                        'deviation': a.deviation
                    }
                    for a in anomalies
                ]
                insights['anomalies'][metric] = insights['anomalies'][metric][-5:]  # Только последние 5

        # Анализ трендов
        for metric in metrics_to_predict:
            if metric in self.models:
                model = self.models[metric]
                insights['trends'][metric] = {
                    'slope': model['slope'],
                    'r_squared': model['r_squared'],
                    'trend_direction': 'increasing' if model['slope'] > 0.01 else 'decreasing' if model['slope'] < -0.01 else 'stable'
                }

        # Генерация общих рекомендаций
        high_priority_recs = []
        for metric, preds in insights['predictions'].items():
            for timeframe, pred_data in preds.items():
                if pred_data['confidence'] > 0.7:  # Высокая достоверность
                    high_priority_recs.append(pred_data['recommendation'])

        insights['recommendations'] = high_priority_recs

        return insights


    def auto_apply_recommendations(self) -> Dict[str, Any]:
        """
        Автоматически применяет рекомендации на основе прогнозов

        Returns:
            Результаты примененных рекомендаций
        """
        insights = self.get_predictive_insights()
        applied_actions = []

        # Проверяем прогнозы на высокую загрузку CPU
        if 'cpu_percent' in insights['predictions']:
            cpu_preds = insights['predictions']['cpu_percent']
            for timeframe, pred_data in cpu_preds.items():
                if pred_data['predicted_value'] > 85 and pred_data['trend'] == 'increasing':
                    print(f"⚠️ Прогнозируется высокая загрузка CPU ({pred_data['predicted_value']:.1f}%), запуск оптимизации...")
                    result = self.resource_manager.optimize_cpu_usage()
                    applied_actions.append({
                        'action': 'cpu_optimization',
                        'trigger': f"Predicted high CPU load: {pred_data['predicted_value']:.1f}%",
                        'result': result
                    })
                    self.stats['recommendations_applied'] += 1
                    break  # Применяем только одно действие на метрику

        # Проверяем прогнозы на высокое использование памяти
        if 'memory_percent' in insights['predictions']:
            mem_preds = insights['predictions']['memory_percent']
            for timeframe, pred_data in mem_preds.items():
                if pred_data['predicted_value'] > 90 and pred_data['trend'] == 'increasing':
                    print(f"⚠️ Прогнозируется высокое использование памяти ({pred_data['predicted_value']:.1f}%), запуск оптимизации...")
                    result = self.memory_tracker.perform_memory_optimization()
                    applied_actions.append({
                        'action': 'memory_optimization',
                        'trigger': f"Predicted high memory usage: {pred_data['predicted_value']:.1f}%",
                        'result': result
                    })
                    self.stats['recommendations_applied'] += 1
                    break

        # Проверяем прогнозы на снижение эффективности
        if 'resource_efficiency' in insights['predictions']:
            eff_preds = insights['predictions']['resource_efficiency']
            for timeframe, pred_data in eff_preds.items():
                if pred_data['predicted_value'] < 70 and pred_data['trend'] == 'decreasing':
                    print(f"⚠️ Прогнозируется снижение эффективности ({pred_data['predicted_value']:.1f}%), запуск комплексной оптимизации...")
                    result = self.orchestrator.start_comprehensive_optimization(["core_utils"])
                    applied_actions.append({
                        'action': 'comprehensive_optimization',
                        'trigger': f"Predicted low efficiency: {pred_data['predicted_value']:.1f}%",
                        'result': result
                    })
                    self.stats['recommendations_applied'] += 1
                    break

        # Проверяем аномалии
        for metric, anomalies in insights.get('anomalies', {}).items():
            for anomaly in anomalies[-2:]:  # Проверяем только последние 2 аномалии
                if anomaly['severity'] in ['high', 'critical']:
                    print(f"🚨 Обнаружена критическая аномалия в {metric}, запуск диагностики...")
                    applied_actions.append({
                        'action': 'diagnostic_scan',
                        'trigger': f"Critical anomaly in {metric}: {anomaly['severity']}",
                        'result': f"Anomaly detected at {anomaly['timestamp']}"
                    })
                    self.stats['recommendations_applied'] += 1

        return {
            'applied_actions': applied_actions,
            'total_actions': len(applied_actions),
            'timestamp': datetime.now().isoformat()
        }


    def start_predictive_monitoring(self, collection_interval: float = 10.0, prediction_interval: float = 60.0):
        """
        Запускает предиктивный мониторинг

        Args:
            collection_interval: Интервал сбора данных (секунды)
            prediction_interval: Интервал прогнозирования (секунды)
        """
        if self.active:
            return

        self.active = True

        def data_collection_loop():
    """TODO: Add description"""

            while self.active:
                try:
                    # Собираем текущие метрики
                    metrics = self.monitoring_center.get_current_metrics()

                    # Добавляем точки данных
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            self.add_data_point(metric_name, value)

                    # Обнаруживаем аномалии
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            anomalies = self.detect_anomalies(metric_name, window_minutes=10)
                            if anomalies:
                                self.stats['anomalies_detected'] += len(anomalies)

                    time.sleep(collection_interval)
                except Exception as e:
                    print(f"Ошибка в цикле сбора данных: {e}")
                    time.sleep(collection_interval)

    """TODO: Add description"""

        def prediction_loop():
            """TODO: Add description"""
            while self.active:
                try:
                    # Применяем рекомендации на основе прогнозов
                    recommendations = self.auto_apply_recommendations()

                    time.sleep(prediction_interval)
                except Exception as e:
                    print(f"Ошибка в цикле прогнозирования: {e}")
                    time.sleep(prediction_interval)

        # Запускаем потоки
        self.learning_thread = threading.Thread(target=data_collection_loop, daemon=True)
        self.prediction_thread = threading.Thread(target=prediction_loop, daemon=True)

        self.learning_thread.start()
        self.prediction_thread.start()

        print("🧠 Предиктивный мониторинг запущен")


    def stop_predictive_monitoring(self):
        """Останавливает предиктивный мониторинг"""
        self.active = False
        if self.learning_thread:
            self.learning_thread.join(timeout=2.0)
        if self.prediction_thread:
            self.prediction_thread.join(timeout=2.0)

        print("🛑 Предиктивный мониторинг остановлен")


    def save_models(self, filepath: Optional[str] = None):
        """
        Сохраняет обученные модели

        Args:
            filepath: Путь для сохранения моделей (опционально)
        """
        if filepath is None:
            filepath = str(self.output_dir / f"predictive_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")

        models_data = {
            'models': self.models,
            'data_history': self.data_history,
            'stats': self.stats,
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'wb') as f:
            pickle.dump(models_data, f)

        print(f"💾 Модели сохранены: {filepath}")


    def load_models(self, filepath: str):
        """
        Загружает обученные модели

        Args:
            filepath: Путь к файлу моделей
        """
        with open(filepath, 'rb') as f:
            models_data = pickle.load(f)

        self.models = models_data.get('models', {})
        self.data_history = models_data.get('data_history', {})
        self.stats.update(models_data.get('stats', {}))

        print(f"📂 Модели загружены: {filepath}")


    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Получает сводку по предиктивной аналитике

        Returns:
            Сводка по предиктивной аналитике
        """
        insights = self.get_predictive_insights()

        summary = {
            'stats': self.stats,
            'insights': insights,
            'model_count': len(self.models),
            'metrics_tracked': list(self.data_history.keys()),
            'prediction_accuracy': {
                metric: model['r_squared'] for metric, model in self.models.items()
            },
            'timestamp': datetime.now().isoformat()
        }

        return summary

def main():
    """Главная функция для демонстрации возможностей предиктивной аналитики"""
    print("=== ПРЕДИКТИВНАЯ АНАЛИТИКА ПРОИЗВОДИТЕЛЬНОСТИ ===")
    print("🧠 Инициализация движка предиктивной аналитики...")

    # Создаем движок предиктивной аналитики
    engine = PredictiveAnalyticsEngine(output_dir="predictive_analytics")

    print("✅ Движок инициализирован")

    # Добавляем немного тестовых данных для начала обучения
    print("📊 Добавление начальных данных для обучения...")
    base_time = datetime.now()
    for i in range(20):
        offset_time = base_time - timedelta(minutes=i*5)
        engine.add_data_point('cpu_percent', 30 + np.random.normal(0, 5), offset_time)
        engine.add_data_point('memory_percent', 45 + np.random.normal(0, 3), offset_time)
        engine.add_data_point('resource_efficiency', 85 + np.random.normal(0, 2), offset_time)

    print("✅ Начальные данные добавлены")

    # Пробуем сделать прогноз
    print("\n🔮 Генерация первых прогнозов...")
    cpu_pred = engine.generate_prediction('cpu_percent', 10)
    if cpu_pred:
        print(f"   Прогноз CPU через 10 мин: {cpu_pred.predicted_value:.2f}% (доверие: {cpu_pred.confidence:.2f})")

    mem_pred = engine.generate_prediction('memory_percent', 10)
    if mem_pred:
        print(f"   Прогноз памяти через 10 мин: {mem_pred.predicted_value:.2f}% (доверие: {mem_pred.confidence:.2f})")

    # Получаем инсайты
    print("\n💡 Получение предиктивных инсайтов...")
    insights = engine.get_predictive_insights()

    print(f"   Прогнозы для метрик: {list(insights['predictions'].keys())}")
    print(f"   Обнаруженные аномалии: {list(insights['anomalies'].keys())}")
    print(f"   Рекомендаций сгенерировано: {len(insights['recommendations'])}")

    # Показываем статистику
    print(f"\n📊 Статистика:")
    print(f"   • Прогнозов сделано: {engine.stats['predictions_made']}")
    print(f"   • Аномалий обнаружено: {engine.stats['anomalies_detected']}")
    print(f"   • Обучено моделей: {engine.stats['models_trained']}")

    print(f"\n🔗 Доступные функции:")
    print("   • Прогнозирование: engine.generate_prediction()")
    print("   • Обнаружение аномалий: engine.detect_anomalies()")
    print("   • Инсайты: engine.get_predictive_insights()")
    print("   • Авто-рекомендации: engine.auto_apply_recommendations()")
    print("   • Мониторинг: engine.start_predictive_monitoring()")

    print("\n🎉 Предиктивная аналитика готова к использованию!")

if __name__ == "__main__":
    main()

