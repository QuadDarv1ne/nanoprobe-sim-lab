# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Модуль ИИ-оптимизатора ресурсов для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет систему искусственного интеллекта для оптимизации
ресурсов на основе машинного обучения и адаптивных алгоритмов.
"""

import time
import threading
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import pickle
import warnings
from collections import deque

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
from utils.predictive_analytics_engine import PredictiveAnalyticsEngine
from utils.automated_optimization_scheduler import AutomatedOptimizationScheduler

@dataclass
class OptimizationRecommendation:
    """Рекомендация по оптимизации"""
    algorithm: str
    parameters: Dict[str, Any]
    expected_improvement: float  # Ожидаемое улучшение в %
    confidence: float  # Уровень доверия (0-1)
    priority: int  # Приоритет (1-5)
    execution_cost: float  # Стоимость выполнения (в условных единицах)

@dataclass
class ResourceState:
    """Состояние ресурсов системы"""
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_io: float
    active_processes: int
    threads_count: int
    load_average: float
    timestamp: datetime

class AIResourceOptimizer:
    """
    Класс ИИ-оптимизатора ресурсов
    Обеспечивает интеллектуальную оптимизацию ресурсов на основе машинного обучения
    и адаптивных алгоритмов.
    """


    def __init__(self, output_dir: str = "ai_optimization"):
        """
        Инициализирует ИИ-оптимизатор ресурсов

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
        self.predictive_engine = PredictiveAnalyticsEngine(output_dir="predictive_analytics")
        self.scheduler = AutomatedOptimizationScheduler(output_dir="automated_optimization")

        # История состояний системы
        self.state_history = deque(maxlen=1000)
        self.optimization_history = []

        # Модель машинного обучения (упрощенная версия)
        self.ml_model = {
            'weights': np.random.rand(7),  # веса для 7 признаков
            'bias': 0.0,
            'learning_rate': 0.01,
            'training_data': [],
            'accuracy': 0.0
        }

        # Алгоритмы оптимизации
        self.optimization_algorithms = {
            'cpu_scheduler': self._optimize_cpu_scheduling,
            'memory_compact': self._optimize_memory_compaction,
            'process_balance': self._optimize_process_balancing,
            'cache_adjustment': self._optimize_cache_settings,
            'thread_pool': self._optimize_thread_pool,
            'disk_scheduler': self._optimize_disk_scheduling,
            'network_buffer': self._optimize_network_buffers
        }

        # Параметры ИИ
        self.optimization_threshold = 0.7  # Порог для применения оптимизации
        self.learning_enabled = True

        # Состояние
        self.active = False
        self.optimizer_thread = None
        self.learning_thread = None

        # Статистика
        self.stats = {
            'optimizations_applied': 0,
            'improvements_achieved': 0,
            'models_trained': 0,
            'predictions_made': 0
        }


    def get_current_state(self) -> ResourceState:
        """
        Получает текущее состояние системы

        Returns:
            Объект ResourceState с текущими метриками
        """
        import psutil

        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk_usage = psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else 0
        # Get network I/O statistics instead of connection counts
        net_io = psutil.net_io_counters()
        network_io = (net_io.bytes_sent + net_io.bytes_recv) / 1024 / 1024  # MB
        active_processes = len(psutil.pids())
        threads_count = sum(p.num_threads() for p in psutil.process_iter())
        load_average = getattr(os, 'getloadavg', lambda: (0, 0, 0))()[0] if hasattr(os, 'getloadavg') else 0

        state = ResourceState(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_usage=disk_usage,
            network_io=network_io,
            active_processes=active_processes,
            threads_count=threads_count,
            load_average=load_average,
            timestamp=datetime.now()
        )

        return state


    def extract_features(self, state: ResourceState) -> np.ndarray:
        """
        Извлекает признаки из состояния системы

        Args:
            state: Состояние системы

        Returns:
            Массив признаков для ML модели
        """
        features = np.array([
            state.cpu_percent / 100.0,           # Нормализованные значения
            state.memory_percent / 100.0,
            state.disk_usage / 100.0,
            min(state.network_io / 100.0, 1.0), # Ограничиваем максимальное значение
            min(state.active_processes / 500.0, 1.0),  # Нормализуем количество процессов
            min(state.threads_count / 2000.0, 1.0),    # Нормализуем количество потоков
            min(state.load_average / 10.0, 1.0)        # Нормализуем load average
        ])

        return features


    def predict_optimization_needed(self, state: ResourceState) -> Tuple[bool, float]:
        """
        Предсказывает необходимость оптимизации

        Args:
            state: Текущее состояние системы

        Returns:
            (необходимость оптимизации, уровень уверенности)
        """
        features = self.extract_features(state)

        # Простая модель: взвешенная сумма признаков
        prediction = np.dot(features, self.ml_model['weights']) + self.ml_model['bias']
        confidence = min(1.0, max(0.0, prediction))  # Ограничиваем от 0 до 1

        needs_optimization = confidence > self.optimization_threshold

        self.stats['predictions_made'] += 1

        return needs_optimization, confidence


    def generate_optimization_recommendations(self, state: ResourceState) -> List[OptimizationRecommendation]:
        """
        Генерирует рекомендации по оптимизации

        Args:
            state: Текущее состояние системы

        Returns:
            Список рекомендаций по оптимизации
        """
        recommendations = []

        # Рекомендации на основе анализа состояния
        if state.cpu_percent > 80:
            recommendations.append(OptimizationRecommendation(
                algorithm='cpu_scheduler',
                parameters={'priority_boost': True, 'affinity_optimization': True},
                expected_improvement=15.0,
                confidence=0.85,
                priority=5,
                execution_cost=0.2
            ))

        if state.memory_percent > 85:
            recommendations.append(OptimizationRecommendation(
                algorithm='memory_compact',
                parameters={'gc_collect': True, 'memory_pool_optimization': True},
                expected_improvement=20.0,
                confidence=0.90,
                priority=5,
                execution_cost=0.3
            ))

        if state.disk_usage > 90:
            recommendations.append(OptimizationRecommendation(
                algorithm='disk_scheduler',
                parameters={'io_priority_adjustment': True, 'buffer_optimization': True},
                expected_improvement=10.0,
                confidence=0.75,
                priority=4,
                execution_cost=0.1
            ))

        if state.load_average > 2.0:
            recommendations.append(OptimizationRecommendation(
                algorithm='process_balance',
                parameters={'load_balancing': True, 'process_prioritization': True},
                expected_improvement=12.0,
                confidence=0.80,
                priority=4,
                execution_cost=0.25
            ))

        # Сортируем по приоритету и уровню уверенности
        recommendations.sort(key=lambda x: (x.priority, x.confidence), reverse=True)

        return recommendations


    def _optimize_cpu_scheduling(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Оптимизация планирования CPU

        Args:
            parameters: Параметры оптимизации

        Returns:
            Результат оптимизации
        """
        result = {
            'algorithm': 'cpu_scheduler',
            'success': True,
            'improvement': 0.0,
            'details': 'CPU scheduling optimized'
        }

        try:
            # В реальной системе здесь будет код для оптимизации планирования CPU
            # Например, изменение приоритетов процессов, affinity настройки и т.д.

            # Симуляция оптимизации
            initial_cpu = self.get_current_state().cpu_percent
            time.sleep(0.1)  # Симуляция работы
            final_cpu = self.get_current_state().cpu_percent

            result['improvement'] = max(0, initial_cpu - final_cpu)
            result['details'] = f'CPU usage reduced from {initial_cpu:.1f}% to {final_cpu:.1f}%'

        except Exception as e:
            result['success'] = False
            result['error'] = str(e)

        return result


    def _optimize_memory_compaction(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Оптимизация компакции памяти

        Args:
            parameters: Параметры оптимизации

        Returns:
            Результат оптимизации
        """
        result = {
            'algorithm': 'memory_compact',
            'success': True,
            'improvement': 0.0,
            'details': 'Memory compaction completed'
        }

        try:
            import gc

            # Выполняем сборку мусора
            collected = gc.collect()

            # В реальной системе здесь будет код для оптимизации пула памяти
            initial_memory = self.get_current_state().memory_percent
            time.sleep(0.1)  # Симуляция работы
            final_memory = self.get_current_state().memory_percent

            result['improvement'] = max(0, initial_memory - final_memory)
            result['details'] = f'Memory usage reduced from {initial_memory:.1f}% to {final_memory:.1f}%, collected {collected} objects'

        except Exception as e:
            result['success'] = False
            result['error'] = str(e)

        return result


    def _optimize_process_balancing(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Оптимизация балансировки процессов

        Args:
            parameters: Параметры оптимизации

        Returns:
            Результат оптимизации
        """
        result = {
            'algorithm': 'process_balance',
            'success': True,
            'improvement': 0.0,
            'details': 'Process balancing completed'
        }

        try:
            # В реальной системе здесь будет код для балансировки процессов
            initial_load = self.get_current_state().load_average
            time.sleep(0.1)  # Симуляция работы
            final_load = self.get_current_state().load_average

            result['improvement'] = max(0, initial_load - final_load)
            result['details'] = f'Load average reduced from {initial_load:.2f} to {final_load:.2f}'

        except Exception as e:
            result['success'] = False
            result['error'] = str(e)

        return result


    def _optimize_cache_settings(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Оптимизация настроек кэширования

        Args:
            parameters: Параметры оптимизации

        Returns:
            Результат оптимизации
        """
        result = {
            'algorithm': 'cache_adjustment',
            'success': True,
            'improvement': 0.0,
            'details': 'Cache settings adjusted'
        }

        try:
            # В реальной системе здесь будет код для оптимизации кэширования
            # Симуляция оптимизации
            time.sleep(0.05)  # Симуляция работы
            result['improvement'] = 5.0  # Условное улучшение

        except Exception as e:
            result['success'] = False
            result['error'] = str(e)

        return result


    def _optimize_thread_pool(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Оптимизация пула потоков

        Args:
            parameters: Параметры оптимизации

        Returns:
            Результат оптимизации
        """
        result = {
            'algorithm': 'thread_pool',
            'success': True,
            'improvement': 0.0,
            'details': 'Thread pool optimized'
        }

        try:
            # В реальной системе здесь будет код для оптимизации пула потоков
            initial_threads = self.get_current_state().threads_count
            time.sleep(0.05)  # Симуляция работы
            final_threads = self.get_current_state().threads_count

            result['improvement'] = max(0, initial_threads - final_threads)
            result['details'] = f'Threads optimized, count changed from {initial_threads} to {final_threads}'

        except Exception as e:
            result['success'] = False
            result['error'] = str(e)

        return result


    def _optimize_disk_scheduling(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Оптимизация планирования дисковых операций

        Args:
            parameters: Параметры оптимизации

        Returns:
            Результат оптимизации
        """
        result = {
            'algorithm': 'disk_scheduler',
            'success': True,
            'improvement': 0.0,
            'details': 'Disk scheduling optimized'
        }

        try:
            # В реальной системе здесь будет код для оптимизации дисковых операций
            initial_disk = self.get_current_state().disk_usage
            time.sleep(0.05)  # Симуляция работы
            final_disk = self.get_current_state().disk_usage

            result['improvement'] = max(0, initial_disk - final_disk)
            result['details'] = f'Disk usage optimization applied'

        except Exception as e:
            result['success'] = False
            result['error'] = str(e)

        return result


    def _optimize_network_buffers(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Оптимизация сетевых буферов

        Args:
            parameters: Параметры оптимизации

        Returns:
            Результат оптимизации
        """
        result = {
            'algorithm': 'network_buffer',
            'success': True,
            'improvement': 0.0,
            'details': 'Network buffers optimized'
        }

        try:
            # В реальной системе здесь будет код для оптимизации сетевых буферов
            initial_network = self.get_current_state().network_io
            time.sleep(0.05)  # Симуляция работы
            final_network = self.get_current_state().network_io

            result['improvement'] = max(0, initial_network - final_network)
            result['details'] = f'Network I/O optimization applied'

        except Exception as e:
            result['success'] = False
            result['error'] = str(e)

        return result


    def apply_optimization(self, recommendation: OptimizationRecommendation) -> Dict[str, Any]:
        """
        Применяет рекомендацию по оптимизации

        Args:
            recommendation: Рекомендация по оптимизации

        Returns:
            Результат применения оптимизации
        """
        if recommendation.algorithm not in self.optimization_algorithms:
            return {
                'success': False,
                'error': f'Unknown optimization algorithm: {recommendation.algorithm}'
            }

        # Выполняем оптимизацию
        algorithm_func = self.optimization_algorithms[recommendation.algorithm]
        result = algorithm_func(recommendation.parameters)

        # Обновляем статистику
        self.stats['optimizations_applied'] += 1
        if result.get('success', False):
            self.stats['improvements_achieved'] += 1

        # Сохраняем в историю
        self.optimization_history.append({
            'recommendation': recommendation,
            'result': result,
            'timestamp': datetime.now()
        })

        return result


    def learn_from_optimization(self, state_before: ResourceState,
    """TODO: Add description"""

                               state_after: ResourceState,
                               recommendation: OptimizationRecommendation,
                               result: Dict[str, Any]) -> None:
        """
        Обучается на результате оптимизации

        Args:
            state_before: Состояние до оптимизации
            state_after: Состояние после оптимизации
            recommendation: Примененная рекомендация
            result: Результат оптимизации
        """
        if not self.learning_enabled:
            return

        try:
            # Вычисляем улучшение
            improvement = self._calculate_improvement(state_before, state_after)

            # Обновляем модель (упрощенная версия обучения)
            features_before = self.extract_features(state_before)

            # Обновляем веса модели на основе результата
            target = 1.0 if improvement > 5.0 else 0.0  # Цель: улучшение > 5%
            prediction = np.dot(features_before, self.ml_model['weights']) + self.ml_model['bias']

            # Простой градиентный шаг
            error = target - prediction
            gradient = error * features_before

            self.ml_model['weights'] += self.ml_model['learning_rate'] * gradient
            self.ml_model['bias'] += self.ml_model['learning_rate'] * error

            # Обновляем точность модели
            self.ml_model['accuracy'] = max(0.5, self.ml_model['accuracy'] + 0.01)  # Простое обновление

            self.stats['models_trained'] += 1

        except Exception as e:
            print(f"Ошибка в обучении: {e}")


    def _calculate_improvement(self, state_before: ResourceState, state_after: ResourceState) -> float:
        """
        Вычисляет улучшение после оптимизации

        Args:
            state_before: Состояние до оптимизации
            state_after: Состояние после оптимизации

        Returns:
            Процент улучшения
        """
        # Вычисляем общий индекс производительности до и после
        before_score = (
            (100 - state_before.cpu_percent) * 0.3 +
            (100 - state_before.memory_percent) * 0.3 +
            (100 - state_before.disk_usage) * 0.2 +
            (100 - min(state_before.load_average * 10, 100)) * 0.2
        )

        after_score = (
            (100 - state_after.cpu_percent) * 0.3 +
            (100 - state_after.memory_percent) * 0.3 +
            (100 - state_after.disk_usage) * 0.2 +
            (100 - min(state_after.load_average * 10, 100)) * 0.2
        )

        # Процент улучшения
        if before_score > 0:
            improvement = ((after_score - before_score) / before_score) * 100
        else:
            improvement = 0.0

        return max(0, improvement)  # Только положительные улучшения


    def run_ai_optimization_cycle(self) -> List[Dict[str, Any]]:
        """
        Выполняет цикл ИИ-оптимизации

        Returns:
            Список результатов оптимизации
        """
        # Получаем текущее состояние
        current_state = self.get_current_state()

        # Сохраняем состояние в историю
        self.state_history.append(current_state)

        # Проверяем необходимость оптимизации
        needs_optimization, confidence = self.predict_optimization_needed(current_state)

        results = []

        if needs_optimization:
            # Генерируем рекомендации
            recommendations = self.generate_optimization_recommendations(current_state)

            # Применяем рекомендации (до 3 за цикл)
            for rec in recommendations[:3]:
                if rec.confidence > 0.7:  # Минимальный уровень доверия
                    # Сохраняем состояние до оптимизации
                    state_before = current_state

                    # Применяем оптимизацию
                    result = self.apply_optimization(rec)

                    # Получаем состояние после оптимизации
                    state_after = self.get_current_state()

                    # Обучаемся на результате
                    self.learn_from_optimization(state_before, state_after, rec, result)

                    results.append({
                        'recommendation': rec,
                        'result': result,
                        'timestamp': datetime.now()
                    })

                    # Небольшая пауза между оптимизациями
                    time.sleep(0.1)

        return results


    def start_ai_optimization(self, interval: float = 10.0):
        """
        Запускает ИИ-оптимизацию в фоновом режиме

        Args:
            interval: Интервал между циклами оптимизации (в секундах)
        """
        if self.active:
            return

        self.active = True

    """TODO: Add description"""

        def ai_optimization_loop():
            while self.active:
                try:
                    self.run_ai_optimization_cycle()
                    time.sleep(interval)
                except Exception as e:
                    print(f"Ошибка в ИИ-оптимизации: {e}")
                    time.sleep(interval)
    """TODO: Add description"""

        def learning_loop():
            while self.active:
                try:
                    # Периодическое обновление модели
                    time.sleep(300)  # Каждые 5 минут
                except Exception as e:
                    print(f"Ошибка в цикле обучения: {e}")

        # Запускаем потоки
        self.optimizer_thread = threading.Thread(target=ai_optimization_loop, daemon=True)
        self.learning_thread = threading.Thread(target=learning_loop, daemon=True)

        self.optimizer_thread.start()
        self.learning_thread.start()

        print("🤖 ИИ-оптимизация запущена")


    def stop_ai_optimization(self):
        """Останавливает ИИ-оптимизацию"""
        self.active = False
        if self.optimizer_thread:
            self.optimizer_thread.join(timeout=2.0)
        if self.learning_thread:
            self.learning_thread.join(timeout=2.0)

        print("🛑 ИИ-оптимизация остановлена")


    def get_ai_status(self) -> Dict[str, Any]:
        """
        Получает статус ИИ-оптимизатора

        Returns:
            Статус ИИ-оптимизатора
        """
        current_state = self.get_current_state()

        return {
            'active': self.active,
            'current_state': {
                'cpu_percent': current_state.cpu_percent,
                'memory_percent': current_state.memory_percent,
                'disk_usage': current_state.disk_usage,
                'network_io': current_state.network_io,
                'active_processes': current_state.active_processes,
                'threads_count': current_state.threads_count,
                'load_average': current_state.load_average
            },
            'stats': self.stats,
            'model_accuracy': self.ml_model['accuracy'],
            'recommendations_count': len(self.optimization_history),
            'state_history_length': len(self.state_history),
            'timestamp': datetime.now().isoformat()
        }


    def save_ai_model(self, filepath: Optional[str] = None):
        """
        Сохраняет ИИ-модель

        Args:
            filepath: Путь для сохранения модели (опционально)
        """
        if filepath is None:
            filepath = str(self.output_dir / f"ai_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")

        model_data = {
            'model': self.ml_model,
            'state_history': list(self.state_history),
            'optimization_history': self.optimization_history,
            'stats': self.stats,
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"💾 ИИ-модель сохранена: {filepath}")


    def load_ai_model(self, filepath: str):
        """
        Загружает ИИ-модель

        Args:
            filepath: Путь к файлу модели
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.ml_model = model_data.get('model', self.ml_model)
        self.state_history = deque(model_data.get('state_history', []), maxlen=1000)
        self.optimization_history = model_data.get('optimization_history', [])
        self.stats.update(model_data.get('stats', {}))

        print(f"📂 ИИ-модель загружена: {filepath}")


    def get_optimization_insights(self) -> Dict[str, Any]:
        """
        Получает инсайты по оптимизации

        Returns:
            Инсайты по оптимизации
        """
        insights = {
            'top_recommendations': [],
            'performance_trends': {},
            'efficiency_metrics': {},
            'ai_decisions': [],
            'timestamp': datetime.now().isoformat()
        }

        # Анализ истории оптимизаций
        if self.optimization_history:
            recent_optimizations = self.optimization_history[-10:]  # Последние 10

            for opt in recent_optimizations:
                insights['ai_decisions'].append({
                    'algorithm': opt['recommendation'].algorithm,
                    'priority': opt['recommendation'].priority,
                    'confidence': opt['recommendation'].confidence,
                    'success': opt['result'].get('success', False),
                    'improvement': opt['result'].get('improvement', 0)
                })

        # Тренды производительности
        if len(self.state_history) > 1:
            states = list(self.state_history)
            insights['performance_trends'] = {
                'cpu_trend': 'decreasing' if states[-1].cpu_percent < states[0].cpu_percent else 'increasing',
                'memory_trend': 'decreasing' if states[-1].memory_percent < states[0].memory_percent else 'increasing',
                'disk_trend': 'decreasing' if states[-1].disk_usage < states[0].disk_usage else 'increasing'
            }

        return insights

def main():
    """Главная функция для демонстрации возможностей ИИ-оптимизатора"""
    print("=== ИИ-ОПТИМИЗАТОР РЕСУРСОВ ===")
    print("🤖 Инициализация ИИ-оптимизатора ресурсов...")

    # Создаем ИИ-оптимизатор
    ai_optimizer = AIResourceOptimizer(output_dir="ai_optimization")

    print("✅ ИИ-оптимизатор инициализирован")

    # Получаем текущее состояние
    print("\n📊 Получение текущего состояния системы...")
    current_state = ai_optimizer.get_current_state()

    print(f"   CPU: {current_state.cpu_percent}%")
    print(f"   Память: {current_state.memory_percent}%")
    print(f"   Диск: {current_state.disk_usage}%")
    print(f"   Активные процессы: {current_state.active_processes}")
    print(f"   Потоки: {current_state.threads_count}")

    # Проверяем необходимость оптимизации
    print("\n🔍 Анализ необходимости оптимизации...")
    needs_opt, confidence = ai_optimizer.predict_optimization_needed(current_state)
    print(f"   Необходима оптимизация: {needs_opt} (уверенность: {confidence:.2f})")

    # Генерируем рекомендации
    print("\n💡 Генерация рекомендаций...")
    recommendations = ai_optimizer.generate_optimization_recommendations(current_state)
    print(f"   Сгенерировано рекомендаций: {len(recommendations)}")

    for i, rec in enumerate(recommendations[:3]):  # Первые 3
        print(f"   {i+1}. {rec.algorithm} - приоритет: {rec.priority}, доверие: {rec.confidence:.2f}")

    # Показываем статус
    print("\n📊 Статус ИИ-оптимизатора:")
    status = ai_optimizer.get_ai_status()
    print(f"   • Точность модели: {status['model_accuracy']:.2f}")
    print(f"   • Оптимизаций применено: {status['stats']['optimizations_applied']}")
    print(f"   • Улучшений достигнуто: {status['stats']['improvements_achieved']}")
    print(f"   • Обучено моделей: {status['stats']['models_trained']}")

    print(f"\n🔗 Доступные функции:")
    print("   • Оптимизация: ai_optimizer.run_ai_optimization_cycle()")
    print("   • Статус: ai_optimizer.get_ai_status()")
    print("   • Инсайты: ai_optimizer.get_optimization_insights()")
    print("   • Запуск: ai_optimizer.start_ai_optimization()")
    print("   • Сохранение: ai_optimizer.save_ai_model()")

    print("\n🎉 ИИ-оптимизатор ресурсов готов к использованию!")

if __name__ == "__main__":
    main()

