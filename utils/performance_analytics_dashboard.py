# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Модуль аналитики производительности для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет аналитическую панель для мониторинга и анализа
производительности проекта, объединяя данные из всех инструментов оптимизации.
"""

import time
import threading
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import statistics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import numpy as np

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

@dataclass
class PerformanceInsight:
    """Инсайт по производительности"""
    category: str  # performance, memory, cpu, efficiency
    title: str
    description: str
    severity: str  # low, medium, high, critical
    value: float
    recommendation: str
    timestamp: datetime

class PerformanceAnalyticsDashboard:
    """
    Класс аналитической панели производительности
    Объединяет данные из всех инструментов оптимизации в единую аналитическую систему.
    """


    def __init__(self, output_dir: str = "analytics_reports"):
        """
        Инициализирует аналитическую панель

        Args:
            output_dir: Директория для сохранения аналитических отчетов
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

        # Хранилище аналитических данных
        self.insights = []
        self.performance_history = []
        self.trend_analysis = {}
        self.active = False
        self.dashboard_thread = None

        # Параметры анализа
        self.analysis_window_hours = 24  # Окно анализа в часах
        self.refresh_interval = 60  # Интервал обновления в секундах


    def start_analytics_monitoring(self, interval: float = 60.0):
        """
        Запускает мониторинг аналитики

        Args:
            interval: Интервал между сбором аналитики (в секундах)
        """
        if self.active:
            return

        self.active = True
        self.refresh_interval = interval

        def analytics_monitor():

            while self.active:
                try:
                    # Собираем аналитику
                    self._collect_analytics_data()

                    # Генерируем инсайты
                    self._generate_insights()

                    # Анализируем тренды
                    self._analyze_trends()

                    time.sleep(interval)

                except Exception as e:
                    print(f"Ошибка в аналитическом мониторинге: {e}")
                    time.sleep(interval)

        self.dashboard_thread = threading.Thread(target=analytics_monitor, daemon=True)
        self.dashboard_thread.start()


    def stop_analytics_monitoring(self):
        """Останавливает мониторинг аналитики"""
        self.active = False
        if self.dashboard_thread:
            self.dashboard_thread.join(timeout=5)


    def _collect_analytics_data(self):
        """Собирает данные для аналитики"""
        try:
            # Собираем текущие метрики
            current_data = {
                'timestamp': datetime.now(),
                'performance_metrics': self._get_performance_metrics(),
                'resource_metrics': self._get_resource_metrics(),
                'memory_metrics': self._get_memory_metrics(),
                'health_metrics': self.health_monitor.get_current_health_status(),
                'benchmark_results': self._get_recent_benchmarks(),
                'log_analysis': self._get_recent_log_analysis()
            }

            self.performance_history.append(current_data)

            # Ограничиваем размер истории (хранить только последние 24 часа)
            cutoff_time = datetime.now() - timedelta(hours=self.analysis_window_hours)
            self.performance_history = [
                data for data in self.performance_history
                if data['timestamp'] > cutoff_time
            ]

        except Exception as e:
            print(f"Ошибка сбора аналитических данных: {e}")


    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Получает метрики производительности"""
        return {
            'efficiency_score': self.resource_manager.get_resource_efficiency_score(),
            'recent_optimizations': len(self.performance_profiler.metrics),
            'avg_execution_time': self._calculate_avg_execution_time(),
            'memory_efficiency': self._calculate_memory_efficiency()
        }


    def _get_resource_metrics(self) -> Dict[str, Any]:
        """Получает метрики ресурсов"""
        current_resources = self.resource_manager.get_current_resources()
        return {
            'cpu_percent': current_resources.get('cpu_percent', 0),
            'memory_rss_mb': current_resources.get('memory_rss_mb', 0),
            'memory_percent': current_resources.get('memory_percent', 0),
            'num_threads': current_resources.get('num_threads', 0),
            'io_read_bytes': current_resources.get('io_counters', {}).get('read_bytes', 0),
            'io_write_bytes': current_resources.get('io_counters', {}).get('write_bytes', 0)
        }


    def _get_memory_metrics(self) -> Dict[str, Any]:
        """Получает метрики памяти"""
        if self.memory_tracker.snapshots:
            latest_snapshot = self.memory_tracker.snapshots[-1]
            return {
                'rss_mb': latest_snapshot.rss_mb,
                'vms_mb': latest_snapshot.vms_mb,
                'percent': latest_snapshot.percent,
                'total_snapshots': len(self.memory_tracker.snapshots),
                'leak_detections': len(self.memory_tracker.leak_detections)
            }
        else:
            return {
                'rss_mb': 0,
                'vms_mb': 0,
                'percent': 0,
                'total_snapshots': 0,
                'leak_detections': 0
            }


    def _calculate_avg_execution_time(self) -> float:
        """Рассчитывает среднее время выполнения"""
        if not self.performance_profiler.metrics:
            return 0.0

        execution_times = [
            metric.value for metric in self.performance_profiler.metrics
            if 'execution_time' in metric.name
        ]

        if execution_times:
            return statistics.mean(execution_times)
        else:
            return 0.0


    def _calculate_memory_efficiency(self) -> float:
        """Рассчитывает эффективность использования памяти"""
        if not self.performance_profiler.metrics:
            return 100.0

        memory_usages = [
            metric.value for metric in self.performance_profiler.metrics
            if 'memory' in metric.name
        ]

        if memory_usages:
            avg_memory = statistics.mean(memory_usages)
            # Чем меньше использование памяти, тем выше эффективность (до определенного порога)
            efficiency = max(0, 100 - (avg_memory / 10))  # Нормализуем к 0-100
            return efficiency
        else:
            return 100.0


    def _get_recent_benchmarks(self) -> List[Dict[str, Any]]:
        """Получает недавние результаты бенчмарков"""
        # Возвращаем последние 10 результатов бенчмарков
        recent_results = []

        # Это просто заглушка - в реальной системе мы бы собирали реальные данные
        if hasattr(self.benchmark_suite, 'results') and self.benchmark_suite.results:
            for result in self.benchmark_suite.results[-10:]:
                recent_results.append({
                    'name': result.name,
                    'execution_time': result.execution_time,
                    'memory_used_mb': result.memory_used_mb,
                    'timestamp': result.timestamp.isoformat()
                })

        return recent_results


    def _get_recent_log_analysis(self) -> Dict[str, Any]:
        """Получает анализ недавних логов"""
        # Просто возвращаем базовую информацию
        return {
            'analyzer_status': 'ready',
            'recent_analysis_count': 0,
            'error_detection_enabled': True
        }


    def _generate_insights(self):
        """Генерирует инсайты по производительности"""
        if not self.performance_history:
            return

        latest_data = self.performance_history[-1]

        # Генерируем инсайты на основе текущих метрик
        insights = []

        # Инсайт по эффективности
        efficiency = latest_data['performance_metrics'].get('efficiency_score', 0)
        if efficiency < 60:
            insights.append(PerformanceInsight(
                category='efficiency',
                title='Низкая эффективность использования ресурсов',
                description=f'Текущая эффективность составляет {efficiency:.1f}%, что ниже оптимального уровня',
                severity='high',
                value=efficiency,
                recommendation='Запустите комплексную оптимизацию ресурсов через orchestrator',
                timestamp=datetime.now()
            ))
        elif efficiency < 80:
            insights.append(PerformanceInsight(
                category='efficiency',
                title='Средняя эффективность использования ресурсов',
                description=f'Текущая эффективность составляет {efficiency:.1f}%, есть резервы для улучшения',
                severity='medium',
                value=efficiency,
                recommendation='Проверьте настройки оптимизации ресурсов',
                timestamp=datetime.now()
            ))

        # Инсайт по использованию CPU
        cpu_usage = latest_data['resource_metrics'].get('cpu_percent', 0)
        if cpu_usage > 85:
            insights.append(PerformanceInsight(
                category='cpu',
                title='Высокая загрузка CPU',
                description=f'Загрузка CPU составляет {cpu_usage:.1f}%, что может указывать на перегрузку системы',
                severity='high',
                value=cpu_usage,
                recommendation='Оптимизируйте алгоритмы или рассмотрите распараллеливание вычислений',
                timestamp=datetime.now()
            ))
        elif cpu_usage > 70:
            insights.append(PerformanceInsight(
                category='cpu',
                title='Повышенная загрузка CPU',
                description=f'Загрузка CPU составляет {cpu_usage:.1f}%, следите за производительностью',
                severity='medium',
                value=cpu_usage,
                recommendation='Мониторьте использование CPU и оптимизируйте при необходимости',
                timestamp=datetime.now()
            ))

        # Инсайт по использованию памяти
        memory_usage = latest_data['resource_metrics'].get('memory_percent', 0)
        if memory_usage > 85:
            insights.append(PerformanceInsight(
                category='memory',
                title='Высокое использование памяти',
                description=f'Использование памяти составляет {memory_usage:.1f}%, что может вызвать проблемы',
                severity='high',
                value=memory_usage,
                recommendation='Проверьте утечки памяти и оптимизируйте использование',
                timestamp=datetime.now()
            ))
        elif memory_usage > 70:
            insights.append(PerformanceInsight(
                category='memory',
                title='Повышенное использование памяти',
                description=f'Использование памяти составляет {memory_usage:.1f}%, следите за потреблением',
                severity='medium',
                value=memory_usage,
                recommendation='Мониторьте использование памяти и очищайте кэш при необходимости',
                timestamp=datetime.now()
            ))

        # Добавляем инсайты в историю
        self.insights.extend(insights)

        # Ограничиваем количество инсайтов
        self.insights = self.insights[-50:]  # Храним последние 50 инсайтов


    def _analyze_trends(self):
        """Анализирует тренды производительности"""
        if len(self.performance_history) < 2:
            return

        # Анализ трендов по основным метрикам
        metrics_over_time = {
            'efficiency': [],
            'cpu_percent': [],
            'memory_percent': [],
            'rss_mb': [],
            'execution_time': []
        }

        for data in self.performance_history:
            metrics_over_time['efficiency'].append(data['performance_metrics'].get('efficiency_score', 0))
            metrics_over_time['cpu_percent'].append(data['resource_metrics'].get('cpu_percent', 0))
            metrics_over_time['memory_percent'].append(data['resource_metrics'].get('memory_percent', 0))
            metrics_over_time['rss_mb'].append(data['memory_metrics'].get('rss_mb', 0))
            metrics_over_time['execution_time'].append(data['performance_metrics'].get('avg_execution_time', 0))

        trends = {}

        for metric_name, values in metrics_over_time.items():
            if len(values) >= 2:
                # Рассчитываем тренд (угол наклона линии регрессии)
                x = list(range(len(values)))
                y = values

                # Простая линейная регрессия
                if len(x) > 1:
                    slope = np.polyfit(x, y, 1)[0] if len(set(y)) > 1 else 0
                    trend_direction = 'increasing' if slope > 0.1 else 'decreasing' if slope < -0.1 else 'stable'

                    trends[metric_name] = {
                        'slope': slope,
                        'direction': trend_direction,
                        'current_value': values[-1],
                        'previous_value': values[-2] if len(values) > 1 else values[0]
                    }

        self.trend_analysis = {
            'timestamp': datetime.now(),
            'trends': trends,
            'analysis_period': self.analysis_window_hours
        }


    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Получает сводку по производительности

        Returns:
            Словарь с основными показателями производительности
        """
        if not self.performance_history:
            return {'status': 'no_data', 'message': 'Нет данных для анализа'}

        latest_data = self.performance_history[-1]

        # Рассчитываем средние значения за период
        efficiency_values = [d['performance_metrics'].get('efficiency_score', 0) for d in self.performance_history]
        cpu_values = [d['resource_metrics'].get('cpu_percent', 0) for d in self.performance_history]
        memory_values = [d['resource_metrics'].get('memory_percent', 0) for d in self.performance_history]

        return {
            'current_status': 'active',
            'period_hours': self.analysis_window_hours,
            'data_points_collected': len(self.performance_history),
            'current_metrics': {
                'efficiency_score': latest_data['performance_metrics'].get('efficiency_score', 0),
                'cpu_percent': latest_data['resource_metrics'].get('cpu_percent', 0),
                'memory_percent': latest_data['resource_metrics'].get('memory_percent', 0),
                'rss_mb': latest_data['memory_metrics'].get('rss_mb', 0),
                'health_score': latest_data['health_metrics'].get('health_score', 0)
            },
            'average_metrics': {
                'efficiency_score': statistics.mean(efficiency_values) if efficiency_values else 0,
                'cpu_percent': statistics.mean(cpu_values) if cpu_values else 0,
                'memory_percent': statistics.mean(memory_values) if memory_values else 0
            },
            'peak_metrics': {
                'max_cpu_percent': max(cpu_values) if cpu_values else 0,
                'max_memory_percent': max(memory_values) if memory_values else 0,
                'max_rss_mb': max([d['memory_metrics'].get('rss_mb', 0) for d in self.performance_history]) if self.performance_history else 0
            },
            'insights_count': len(self.insights),
            'critical_insights': len([i for i in self.insights if i.severity == 'critical']),
            'high_insights': len([i for i in self.insights if i.severity == 'high']),
            'trend_analysis_available': bool(self.trend_analysis)
        }


    def get_actionable_insights(self, severity_filter: str = 'high') -> List[PerformanceInsight]:
        """
        Получает действия, которые можно предпринять на основе инсайтов

        Args:
            severity_filter: Фильтр по уровню серьезности ('low', 'medium', 'high', 'critical')

        Returns:
            Список инсайтов, требующих внимания
        """
        filtered_insights = [
            insight for insight in self.insights
            if insight.severity in ['critical', 'high', 'medium']
            if severity_filter == 'all' or insight.severity == severity_filter
            or (severity_filter == 'high' and insight.severity in ['critical', 'high'])
            or (severity_filter == 'medium' and insight.severity in ['critical', 'high', 'medium'])
        ]

        # Сортируем по серьезности и актуальности
        sorted_insights = sorted(
            filtered_insights,
            key=lambda x: (
                {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}[x.severity],
                -x.timestamp.timestamp()  # Новые первыми
            ),
            reverse=True
        )

        return sorted_insights


    def generate_analytics_report(self, output_path: str = None) -> str:
        """
        Генерирует аналитический отчет

        Args:
            output_path: Путь для сохранения отчета

        Returns:
            Путь к сохраненному отчету
        """
        if output_path is None:
            output_path = str(self.output_dir / f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        report = {
            'generation_time': datetime.now().isoformat(),
            'summary': self.get_performance_summary(),
            'trend_analysis': self.trend_analysis,
            'top_insights': [self._insight_to_dict(insight) for insight in self.get_actionable_insights('high')],
            'all_insights_count': len(self.insights),
            'recent_performance_history': [
                {
                    'timestamp': data['timestamp'].isoformat(),
                    'efficiency_score': data['performance_metrics'].get('efficiency_score', 0),
                    'cpu_percent': data['resource_metrics'].get('cpu_percent', 0),
                    'memory_percent': data['resource_metrics'].get('memory_percent', 0)
                }
                for data in self.performance_history[-24:]  # Последние 24 точки
            ],
            'recommendations': self._generate_comprehensive_recommendations()
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        return output_path


    def _insight_to_dict(self, insight: PerformanceInsight) -> Dict[str, Any]:
        """Конвертирует инсайт в словарь"""
        return {
            'category': insight.category,
            'title': insight.title,
            'description': insight.description,
            'severity': insight.severity,
            'value': insight.value,
            'recommendation': insight.recommendation,
            'timestamp': insight.timestamp.isoformat()
        }


    def _generate_comprehensive_recommendations(self) -> List[str]:
        """Генерирует комплексные рекомендации"""
        recommendations = []

        summary = self.get_performance_summary()
        current = summary.get('current_metrics', {})

        if current.get('efficiency_score', 0) < 70:
            recommendations.append("Запустите комплексную оптимизацию через OptimizationOrchestrator для повышения эффективности")

        if current.get('cpu_percent', 0) > 80:
            recommendations.append("Оптимизируйте алгоритмы или распределите нагрузку для снижения загрузки CPU")

        if current.get('memory_percent', 0) > 80:
            recommendations.append("Проверьте утечки памяти и оптимизируйте использование памяти")

        if self.trend_analysis and 'trends' in self.trend_analysis:
            trends = self.trend_analysis['trends']

            if 'cpu_percent' in trends and trends['cpu_percent']['direction'] == 'increasing':
                recommendations.append("Загрузка CPU имеет положительный тренд - требуется оптимизация")

            if 'memory_percent' in trends and trends['memory_percent']['direction'] == 'increasing':
                recommendations.append("Использование памяти растет - мониторьте и оптимизируйте")

        # Добавляем рекомендации из инсайтов
        actionable_insights = self.get_actionable_insights('high')
        for insight in actionable_insights[:3]:  # Берем топ-3
            recommendations.append(insight.recommendation)

        if not recommendations:
            recommendations.append("Система работает в оптимальном режиме. Продолжайте текущую стратегию мониторинга.")

        return recommendations


    def visualize_performance_dashboard(self, output_path: str = None) -> str:
        """
        Создает визуализацию аналитической панели

        Args:
            output_path: Путь для сохранения визуализации

        Returns:
            Путь к сохраненной визуализации
        """
        if output_path is None:
            output_path = str(self.output_dir / f"performance_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

        if not self.performance_history:
            print("Нет данных для визуализации")
            return ""

        # Подготовка данных для визуализации
        timestamps = [data['timestamp'] for data in self.performance_history]
        efficiency_scores = [data['performance_metrics'].get('efficiency_score', 0) for data in self.performance_history]
        cpu_percents = [data['resource_metrics'].get('cpu_percent', 0) for data in self.performance_history]
        memory_percents = [data['resource_metrics'].get('memory_percent', 0) for data in self.performance_history]
        rss_mb_values = [data['memory_metrics'].get('rss_mb', 0) for data in self.performance_history]

        # Создаем фигуру с подграфиками
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Аналитическая панель производительности - Nanoprobe Simulation Lab', fontsize=16, fontweight='bold')

        # 1. Эффективность использования ресурсов
        axes[0, 0].plot(timestamps, efficiency_scores, marker='o', linewidth=2, markersize=4, color='green', label='Эффективность')
        axes[0, 0].axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='Порог хорошей эффективности')
        axes[0, 0].axhline(y=60, color='red', linestyle='--', alpha=0.7, label='Порог низкой эффективности')
        axes[0, 0].set_title('Эффективность использования ресурсов')
        axes[0, 0].set_ylabel('Эффективность (%)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. Загрузка CPU
        axes[0, 1].plot(timestamps, cpu_percents, marker='s', linewidth=2, markersize=4, color='red', label='CPU %')
        axes[0, 1].axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='Порог высокой загрузки')
        axes[0, 1].axhline(y=95, color='red', linestyle='--', alpha=0.7, label='Порог критической загрузки')
        axes[0, 1].set_title('Загрузка CPU')
        axes[0, 1].set_ylabel('Загрузка (%)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. Использование памяти
        axes[1, 0].plot(timestamps, memory_percents, marker='^', linewidth=2, markersize=4, color='blue', label='Память %')
        axes[1, 0].plot(timestamps, [rss_mb_values[i]/10 for i in range(len(rss_mb_values))] if rss_mb_values else [],
                        linestyle=':', color='purple', label='RSS (деленная на 10)')
        axes[1, 0].axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='Порог высокого использования')
        axes[1, 0].set_title('Использование памяти')
        axes[1, 0].set_ylabel('Процент / MB (шкала RSS/10)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 4. Комбинированный график
        ax4 = axes[1, 1]
        ax4_twin = ax4.twinx()

        # Основной график - эффективность
        line1 = ax4.plot(timestamps, efficiency_scores, marker='o', linewidth=2, markersize=4, color='green', label='Эффективность')[0]

        # Вторичная ось - RSS память
        line2 = ax4_twin.plot(timestamps, rss_mb_values, marker='d', linewidth=2, markersize=4, color='purple', label='RSS память')[0]

        ax4.set_ylabel('Эффективность (%)', color='green')
        ax4_twin.set_ylabel('RSS память (MB)', color='purple')
        ax4.set_title('Эффективность vs Потребление памяти')
        ax4.grid(True, alpha=0.3)

        # Легенда для обоих графиков
        ax4.legend([line1, line2], ['Эффективность', 'RSS память'], loc='upper left')
        ax4_twin.legend(loc='upper right')
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path


    def get_optimization_suggestions(self) -> Dict[str, List[str]]:
        """
        Получает предложения по оптимизации из всех инструментов

        Returns:
            Словарь с предложениями по категориям
        """
        suggestions = {
            'performance': self.performance_profiler.get_optimization_recommendations(),
            'resources': self.resource_manager.suggest_optimizations(),
            'memory': self.memory_tracker.get_memory_optimization_recommendations(),
            'system_health': self.health_monitor.get_health_recommendations(),
            'analytics': self._generate_comprehensive_recommendations()
        }

        return suggestions

def main():
    """Главная функция для демонстрации возможностей аналитической панели"""
    print("=== АНАЛИТИЧЕСКАЯ ПАНЕЛЬ ПРОИЗВОДИТЕЛЬНОСТИ ===")

    # Создаем аналитическую панель
    dashboard = PerformanceAnalyticsDashboard()

    print("✓ Аналитическая панель инициализирована")
    print(f"✓ Директория вывода: {dashboard.output_dir}")

    # Запускаем краткосрочный мониторинг для сбора данных
    print("\nЗапуск мониторинга аналитики...")
    dashboard.start_analytics_monitoring(interval=10)  # Обновление каждые 10 секунд

    # Ждем немного для сбора данных
    print("Сбор данных в течение 30 секунд...")
    time.sleep(30)

    # Останавливаем мониторинг
    dashboard.stop_analytics_monitoring()
    print("✓ Мониторинг остановлен")

    # Получаем сводку
    print("\nПолучение сводки по производительности...")
    summary = dashboard.get_performance_summary()
    print(f"✓ Статус: {summary['current_status']}")
    print(f"✓ Собрано точек данных: {summary['data_points_collected']}")
    print(f"✓ Текущая эффективность: {summary['current_metrics']['efficiency_score']:.2f}%")
    print(f"✓ Загрузка CPU: {summary['current_metrics']['cpu_percent']:.2f}%")
    print(f"✓ Использование памяти: {summary['current_metrics']['memory_percent']:.2f}%")

    # Получаем инсайты
    print("\nПолучение ключевых инсайтов...")
    insights = dashboard.get_actionable_insights('high')
    print(f"✓ Найдено инсайтов: {len(insights)}")
    for i, insight in enumerate(insights[:3], 1):  # Показываем топ-3
        print(f"  {i}. [{insight.severity.upper()}] {insight.title}")
        print(f"     Рекомендация: {insight.recommendation}")

    # Генерируем отчет
    print("\nГенерация аналитического отчета...")
    report_path = dashboard.generate_analytics_report()
    print(f"✓ Отчет сохранен: {report_path}")

    # Создаем визуализацию
    print("\nСоздание визуализации панели...")
    viz_path = dashboard.visualize_performance_dashboard()
    if viz_path:
        print(f"✓ Визуализация сохранена: {viz_path}")

    # Получаем предложения по оптимизации
    print("\nПолучение предложений по оптимизации...")
    suggestions = dashboard.get_optimization_suggestions()
    print("Категории предложений:")
    for category, recs in suggestions.items():
        print(f"  {category}: {len(recs)} предложений")

    print("\nАналитическая панель успешно протестирована")
    print("\nДоступные функции:")
    print("- Мониторинг: dashboard.start_analytics_monitoring()")
    print("- Сводка: dashboard.get_performance_summary()")
    print("- Инсайты: dashboard.get_actionable_insights()")
    print("- Отчеты: dashboard.generate_analytics_report()")
    print("- Визуализация: dashboard.visualize_performance_dashboard()")
    print("- Рекомендации: dashboard.get_optimization_suggestions()")
    print("- Тренды: dashboard.trend_analysis")

if __name__ == "__main__":
    main()

