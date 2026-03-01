# -*- coding: utf-8 -*-
#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3

"""
Модуль центра мониторинга производительности для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет централизованную систему мониторинга и управления
всеми аспектами производительности и оптимизации проекта.
"""

import time
import threading
import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import psutil
import statistics
from dataclasses import dataclass, asdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque

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
from utils.realtime_dashboard import RealTimeDashboard

@dataclass
class PerformanceAlert:
    """Оповещение о производительности"""
    timestamp: datetime
    severity: str  # 'low', 'medium', 'high', 'critical'
    category: str  # 'cpu', 'memory', 'disk', 'network', 'performance', 'optimization'
    message: str
    value: float
    threshold: float

@dataclass
class PerformanceTrend:
    """Тренд производительности"""
    metric: str
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    rate_of_change: float
    confidence: float  # 0-1
    duration: timedelta

class PerformanceMonitoringCenter:
    """
    Класс центра мониторинга производительности
    Обеспечивает комплексный мониторинг, анализ и управление производительностью проекта.
    """


    def __init__(self, output_dir: str = "performance_monitoring"):
        """
        Инициализирует центр мониторинга

        Args:
            output_dir: Директория для сохранения данных мониторинга
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
        self.realtime_dashboard = RealTimeDashboard(port=8081)  # Порт для мониторинга

        # История метрик
        self.metric_history = defaultdict(lambda: deque(maxlen=1000))  # Хранит последние 1000 значений
        self.alerts_history = []
        self.trends_history = []

        # Пороговые значения
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_usage': 90.0,
            'response_time_ms': 1000.0,
            'error_rate': 0.05,
            'resource_efficiency': 70.0,
            'optimization_score': 75.0
        }

        # Состояние мониторинга
        self.monitoring_active = False
        self.monitoring_thread = None
        self.alert_handlers = []
        self.data_exporters = []

        # Статистика
        self.stats = {
            'total_checks': 0,
            'alerts_generated': 0,
            'optimizations_applied': 0,
            'performance_improvements': 0
        }


    def add_alert_handler(self, handler: Callable[[PerformanceAlert], None]):
        """
        Добавляет обработчик оповещений

        Args:
            handler: Функция-обработчик оповещений
        """
        self.alert_handlers.append(handler)


    def add_data_exporter(self, exporter: Callable[[Dict[str, Any]], None]):
        """
        Добавляет экспортер данных

        Args:
            exporter: Функция-экспортер данных
        """
        self.data_exporters.append(exporter)


    def set_threshold(self, metric: str, value: float):
        """
        Устанавливает пороговое значение для метрики

        Args:
            metric: Название метрики
            value: Пороговое значение
        """
        self.thresholds[metric] = value


    def get_current_metrics(self) -> Dict[str, float]:
        """
        Получает текущие метрики производительности

        Returns:
            Словарь с текущими метриками
        """
        # Системные метрики
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk_usage = psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else 0

        # Метрики из инструментов оптимизации
        resource_efficiency = self.resource_manager.get_resource_efficiency_score()

        # Оценка оптимизации
        optimization_score = min(100, max(0, resource_efficiency + 10))

        # Собираем все метрики
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'disk_usage': disk_usage,
            'resource_efficiency': resource_efficiency,
            'optimization_score': optimization_score,
            'active_processes': len(psutil.pids()),
            'load_average': getattr(os, 'getloadavg', lambda: (0, 0, 0))()[0] if hasattr(os, 'getloadavg') else 0,
            'network_connections': len(psutil.net_connections()),
            'threads_count': sum(p.num_threads() for p in psutil.process_iter())
        }

        return metrics


    def check_for_alerts(self, metrics: Dict[str, float]) -> List[PerformanceAlert]:
        """
        Проверяет наличие оповещений по метрикам

        Args:
            metrics: Словарь с метриками

        Returns:
            Список оповещений
        """
        alerts = []

        for metric_name, current_value in metrics.items():
            if isinstance(current_value, (int, float)) and metric_name in self.thresholds:
                threshold_value = self.thresholds[metric_name]

                if current_value > threshold_value:
                    # Определяем уровень серьезности
                    severity = 'low'
                    if current_value > threshold_value * 1.2:
                        severity = 'medium'
                    if current_value > threshold_value * 1.5:
                        severity = 'high'
                    if current_value > threshold_value * 2.0:
                        severity = 'critical'

                    # Определяем категорию
                    category = 'performance'
                    if 'cpu' in metric_name:
                        category = 'cpu'
                    elif 'memory' in metric_name:
                        category = 'memory'
                    elif 'disk' in metric_name:
                        category = 'disk'
                    elif 'resource' in metric_name:
                        category = 'optimization'
                    elif 'optimization' in metric_name:
                        category = 'optimization'

                    alert = PerformanceAlert(
                        timestamp=datetime.now(),
                        severity=severity,
                        category=category,
                        message=f"Метрика {metric_name} превысила порог: {current_value:.2f} > {threshold_value:.2f}",
                        value=current_value,
                        threshold=threshold_value
                    )

                    alerts.append(alert)

        return alerts


    def analyze_trends(self, metric_name: str, window_minutes: int = 30) -> Optional[PerformanceTrend]:
        """
        Анализирует тренды для метрики

        Args:
            metric_name: Название метрики
            window_minutes: Окно анализа в минутах

        Returns:
            Объект тренда или None
        """
        if metric_name not in self.metric_history or len(self.metric_history[metric_name]) < 10:
            return None

        # Получаем последние значения за указанное время
        history_values = list(self.metric_history[metric_name])

        if len(history_values) < 2:
            return None

        # Вычисляем изменения
        recent_values = history_values[-min(len(history_values), window_minutes * 2):]  # 2 значения в минуту
        if len(recent_values) < 2:
            return None

        # Вычисляем направление тренда
        start_value = recent_values[0]
        end_value = recent_values[-1]

        rate_of_change = (end_value - start_value) / len(recent_values)

        if abs(rate_of_change) < 0.1:  # Порог для стабильности
            trend_direction = 'stable'
        elif rate_of_change > 0:
            trend_direction = 'increasing'
        else:
            trend_direction = 'decreasing'

        # Вычисляем уверенность (на основе стандартного отклонения)
        std_dev = statistics.stdev(recent_values) if len(recent_values) > 1 else 0
        confidence = 1.0 - min(1.0, std_dev / max(abs(start_value), 0.1))

        trend = PerformanceTrend(
            metric=metric_name,
            trend_direction=trend_direction,
            rate_of_change=rate_of_change,
            confidence=confidence,
            duration=timedelta(minutes=len(recent_values)//2)  # Приблизительно
        )

        return trend


    def collect_and_process_metrics(self):
        """Собирает и обрабатывает метрики"""
        # Собираем текущие метрики
        current_metrics = self.get_current_metrics()

        # Сохраняем в историю
        for key, value in current_metrics.items():
            if isinstance(value, (int, float)):
                self.metric_history[key].append(value)

        # Проверяем оповещения
        alerts = self.check_for_alerts(current_metrics)

        # Обрабатываем оповещения
        for alert in alerts:
            self.alerts_history.append(alert)
            self.stats['alerts_generated'] += 1

            # Вызываем обработчики
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    print(f"Ошибка в обработчике оповещений: {e}")

        # Анализируем тренды
        for metric_name in current_metrics.keys():
            if isinstance(current_metrics[metric_name], (int, float)):
                trend = self.analyze_trends(metric_name)
                if trend:
                    self.trends_history.append(trend)

        # Экспортируем данные
        for exporter in self.data_exporters:
            try:
                exporter(current_metrics)
            except Exception as e:
                print(f"Ошибка в экспортере данных: {e}")

        self.stats['total_checks'] += 1

        return current_metrics, alerts


    def start_monitoring(self, interval: float = 5.0):
        """
        Запускает мониторинг в фоновом режиме

        Args:
            interval: Интервал между проверками (в секундах)
        """
        if self.monitoring_active:
            return

        self.monitoring_active = True

        def monitor():

            while self.monitoring_active:
                try:
                    self.collect_and_process_metrics()
                    time.sleep(interval)
                except Exception as e:
                    print(f"Ошибка в мониторинге: {e}")
                    time.sleep(interval)

        self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self.monitoring_thread.start()

        print("✅ Мониторинг производительности запущен")


    def stop_monitoring(self):
        """Останавливает мониторинг"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)

        print("🛑 Мониторинг производительности остановлен")


    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Получает сводку производительности

        Returns:
            Словарь с общей информацией о производительности
        """
        current_metrics = self.get_current_metrics()

        # Получаем последние тренды
        recent_trends = []
        for trend in self.trends_history[-10:]:  # Последние 10 трендов
            recent_trends.append({
                'metric': trend.metric,
                'direction': trend.trend_direction,
                'rate': trend.rate_of_change,
                'confidence': trend.confidence
            })

        # Получаем последние оповещения
        recent_alerts = []
        for alert in self.alerts_history[-10:]:  # Последние 10 оповещений
            recent_alerts.append({
                'severity': alert.severity,
                'category': alert.category,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat()
            })

        summary = {
            'current_metrics': current_metrics,
            'stats': self.stats,
            'recent_trends': recent_trends,
            'recent_alerts': recent_alerts,
            'health_status': self.health_monitor.get_current_health_status(),
            'optimization_status': self.analytics_dashboard.get_performance_summary(),
            'timestamp': datetime.now().isoformat()
        }

        return summary


    def generate_performance_report(self, output_path: Optional[str] = None) -> str:
        """
        Генерирует отчет о производительности

        Args:
            output_path: Путь для сохранения отчета (опционально)

        Returns:
            Путь к созданному отчету
        """
        if output_path is None:
            filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_path = str(self.output_dir / filename)

        summary = self.get_performance_summary()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

        print(f"📊 Отчет о производительности сохранен: {output_path}")
        return output_path


    def generate_visualization_report(self, output_path: Optional[str] = None) -> str:
        """
        Генерирует визуальный отчет о производительности

        Args:
            output_path: Путь для сохранения отчета (опционально)

        Returns:
            Путь к созданному отчету
        """
        if output_path is None:
            filename = f"performance_viz_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            output_path = str(self.output_dir / filename)

        # Подготовка данных для визуализации
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Monitoring Report - Nanoprobe Simulation Lab', fontsize=16)

        # CPU Usage
        if 'cpu_percent' in self.metric_history and len(self.metric_history['cpu_percent']) > 1:
            cpu_data = list(self.metric_history['cpu_percent'])
            ax1.plot(cpu_data, label='CPU %', color='red')
            ax1.axhline(y=self.thresholds['cpu_percent'], color='red', linestyle='--', alpha=0.5, label='Threshold')
            ax1.set_title('CPU Usage Over Time')
            ax1.set_xlabel('Measurements')
            ax1.set_ylabel('CPU %')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Memory Usage
        if 'memory_percent' in self.metric_history and len(self.metric_history['memory_percent']) > 1:
            memory_data = list(self.metric_history['memory_percent'])
            ax2.plot(memory_data, label='Memory %', color='blue')
            ax2.axhline(y=self.thresholds['memory_percent'], color='blue', linestyle='--', alpha=0.5, label='Threshold')
            ax2.set_title('Memory Usage Over Time')
            ax2.set_xlabel('Measurements')
            ax2.set_ylabel('Memory %')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # Resource Efficiency
        if 'resource_efficiency' in self.metric_history and len(self.metric_history['resource_efficiency']) > 1:
            eff_data = list(self.metric_history['resource_efficiency'])
            ax3.plot(eff_data, label='Efficiency %', color='green')
            ax3.axhline(y=self.thresholds['resource_efficiency'], color='green', linestyle='--', alpha=0.5, label='Threshold')
            ax3.set_title('Resource Efficiency Over Time')
            ax3.set_xlabel('Measurements')
            ax3.set_ylabel('Efficiency %')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # Alert Distribution
        if self.alerts_history:
            alert_categories = [alert.category for alert in self.alerts_history[-50:]]  # Последние 50
            alert_severities = [alert.severity for alert in self.alerts_history[-50:]]

            if alert_categories:
                # Категории оповещений
                category_counts = {}
                for cat in alert_categories:
                    category_counts[cat] = category_counts.get(cat, 0) + 1

                ax4.bar(category_counts.keys(), category_counts.values(), alpha=0.7, color='orange')
                ax4.set_title('Alert Distribution by Category')
                ax4.set_xlabel('Category')
                ax4.set_ylabel('Count')
                ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 Визуальный отчет о производительности сохранен: {output_path}")
        return output_path


    def apply_optimizations_based_on_metrics(self) -> Dict[str, Any]:
        """
        Применяет оптимизации на основе текущих метрик

        Returns:
            Результаты примененных оптимизаций
        """
        current_metrics = self.get_current_metrics()
        results = {
            'optimizations_applied': [],
            'improvements_detected': 0,
            'before_metrics': current_metrics.copy()
        }

        # Применяем оптимизации в зависимости от метрик
        optimizations_performed = []

        # Высокая загрузка CPU
        if current_metrics.get('cpu_percent', 0) > self.thresholds['cpu_percent']:
            print("⚠️ Высокая загрузка CPU, применяем оптимизации...")
            cpu_opt_results = self.resource_manager.optimize_cpu_usage()
            optimizations_performed.append({
                'type': 'cpu_optimization',
                'results': cpu_opt_results,
                'trigger_metric': 'cpu_percent',
                'value': current_metrics['cpu_percent']
            })

        # Высокое использование памяти
        if current_metrics.get('memory_percent', 0) > self.thresholds['memory_percent']:
            print("⚠️ Высокое использование памяти, применяем оптимизации...")
            mem_opt_results = self.resource_manager.optimize_memory_usage()
            optimizations_performed.append({
                'type': 'memory_optimization',
                'results': mem_opt_results,
                'trigger_metric': 'memory_percent',
                'value': current_metrics['memory_percent']
            })

        # Запускаем комплексную оптимизацию
        print("🔄 Запуск комплексной оптимизации...")
        comp_opt_results = self.orchestrator.start_comprehensive_optimization([
            "core_utils", "spm_simulator", "image_analyzer"
        ])
        optimizations_performed.append({
            'type': 'comprehensive_optimization',
            'results': comp_opt_results,
            'trigger_metric': 'overall_performance',
            'value': 'N/A'
        })

        # Обновляем статистику
        self.stats['optimizations_applied'] += len(optimizations_performed)

        # Проверяем улучшения
        after_metrics = self.get_current_metrics()
        results['after_metrics'] = after_metrics
        results['optimizations_applied'] = optimizations_performed

        # Подсчитываем улучшения
        for metric_name in ['cpu_percent', 'memory_percent', 'resource_efficiency', 'optimization_score']:
            before_val = results['before_metrics'].get(metric_name, 0)
            after_val = results['after_metrics'].get(metric_name, 0)

            if metric_name in ['cpu_percent', 'memory_percent']:
                # Для этих метрик улучшение - снижение значения
                if after_val < before_val:
                    results['improvements_detected'] += 1
            else:
                # Для этих метрик улучшение - увеличение значения
                if after_val > before_val:
                    results['improvements_detected'] += 1

        self.stats['performance_improvements'] += results['improvements_detected']

        return results


    def export_to_csv(self, output_path: Optional[str] = None) -> str:
        """
        Экспортирует историю метрик в CSV

        Args:
            output_path: Путь для сохранения CSV (опционально)

        Returns:
            Путь к созданному файлу
        """
        if output_path is None:
            filename = f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            output_path = str(self.output_dir / filename)

        # Подготовка данных для CSV
        all_metrics = []
        if self.metric_history:
            # Определяем общую длину (берем самую длинную серию)
            max_len = max(len(series) for series in self.metric_history.values()) if self.metric_history else 0

            # Создаем строки данных
            for i in range(max_len):
                row = {'index': i}
                for metric_name, series in self.metric_history.items():
                    if i < len(series):
                        row[metric_name] = series[i]
                    else:
                        row[metric_name] = None  # Fill with None if not enough data

                all_metrics.append(row)

        # Сохраняем в CSV
        if all_metrics:
            df = pd.DataFrame(all_metrics)
            df.to_csv(output_path, index=False)

        print(f"📊 История метрик экспортирована в CSV: {output_path}")
        return output_path

def default_alert_handler(alert: PerformanceAlert):
    """Обработчик оповещений по умолчанию"""
    severity_colors = {
        'low': '\033[92m',      # Green
        'medium': '\033[93m',   # Yellow
        'high': '\033[91m',     # Red
        'critical': '\033[95m'  # Magenta
    }
    reset_color = '\033[0m'

    color = severity_colors.get(alert.severity, '')
    print(f"{color}[{alert.severity.upper()} - {alert.category}] {alert.message}{reset_color}")

def main():
    """Главная функция для демонстрации возможностей центра мониторинга"""
    print("=== ЦЕНТР МОНИТОРИНГА ПРОИЗВОДИТЕЛЬНОСТИ ===")
    print("🚀 Инициализация центра мониторинга...")

    # Создаем центр мониторинга
    pmc = PerformanceMonitoringCenter(output_dir="performance_monitoring")

    # Добавляем обработчик оповещений
    pmc.add_alert_handler(default_alert_handler)

    print("✅ Центр мониторинга инициализирован")

    # Запускаем мониторинг
    print("🔄 Запуск мониторинга...")
    pmc.start_monitoring(interval=3.0)  # Каждые 3 секунды

    # Запускаем реал-тайм дашборд в фоне
    print("📊 Запуск реал-тайм дашборда...")
    pmc.realtime_dashboard.start_monitoring(interval=2.0)

    try:
        print("\n⏳ Сбор данных в течение 30 секунд...")
        time.sleep(30)

        # Применяем оптимизации
        print("\n🔧 Применение оптимизаций на основе собранных метрик...")
        opt_results = pmc.apply_optimizations_based_on_metrics()
        print(f"✅ Применено оптимизаций: {len(opt_results['optimizations_applied'])}")
        print(f"📈 Обнаружено улучшений: {opt_results['improvements_detected']}")

        # Генерируем отчеты
        print("\n📝 Генерация отчетов...")
        report_path = pmc.generate_performance_report()
        viz_report_path = pmc.generate_visualization_report()
        csv_path = pmc.export_to_csv()

        # Показываем сводку
        print("\n📋 Сводка производительности:")
        summary = pmc.get_performance_summary()
        print(f"  • Всего проверок: {summary['stats']['total_checks']}")
        print(f"  • Сгенерировано оповещений: {summary['stats']['alerts_generated']}")
        print(f"  • Применено оптимизаций: {summary['stats']['optimizations_applied']}")
        print(f"  • Улучшений производительности: {summary['stats']['performance_improvements']}")

        print(f"\n📊 Отчеты сохранены:")
        print(f"  • JSON: {report_path}")
        print(f"  • Визуализация: {viz_report_path}")
        print(f"  • CSV: {csv_path}")

        print(f"\n🔗 Реал-тайм дашборд доступен на http://localhost:8081")

        print("\nНажмите Ctrl+C для остановки мониторинга...")
        while True:
            time.sleep(10)

            # Периодически показываем статус
            current_metrics = pmc.get_current_metrics()
            print(f"\n📊 Текущее состояние (CPU: {current_metrics['cpu_percent']:.1f}%, "
                  f"MEM: {current_metrics['memory_percent']:.1f}%, "
                  f"EFF: {current_metrics['resource_efficiency']:.1f}%)")

    except KeyboardInterrupt:
        print("\n🛑 Остановка мониторинга...")
        pmc.stop_monitoring()
        pmc.realtime_dashboard.stop_monitoring()
        print("✅ Мониторинг остановлен")

if __name__ == "__main__":
    main()

