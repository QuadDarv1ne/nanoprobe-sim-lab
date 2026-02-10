#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль автоматического планировщика оптимизации для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет систему автоматического планирования и выполнения
операций оптимизации на основе предиктивной аналитики и текущего состояния системы.
"""

import time
import threading
import json
import schedule
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import queue

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


@dataclass
class OptimizationJob:
    """Задание на оптимизацию"""
    id: str
    name: str
    scheduled_time: datetime
    optimization_type: str
    priority: int  # 1-5, 5 - highest priority
    target_metrics: List[str]
    trigger_condition: str
    trigger_value: float
    executed: bool = False
    execution_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None


@dataclass
class ScheduledOptimization:
    """Запланированная оптимизация"""
    job: OptimizationJob
    scheduled_at: datetime
    status: str  # 'scheduled', 'running', 'completed', 'failed', 'cancelled'


class AutomatedOptimizationScheduler:
    """
    Класс автоматического планировщика оптимизации
    Обеспечивает автоматическое планирование и выполнение оптимизаций
    на основе предиктивной аналитики и текущего состояния системы.
    """
    
    def __init__(self, output_dir: str = "automated_optimization"):
        """
        Инициализирует планировщик оптимизации
        
        Args:
            output_dir: Директория для сохранения логов и результатов
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
        
        # Очередь задач
        self.job_queue = queue.PriorityQueue()
        self.scheduled_jobs = {}
        self.executed_jobs = []
        self.failed_jobs = []
        
        # Правила автоматического планирования
        self.auto_rules = []
        
        # Состояние
        self.running = False
        self.scheduler_thread = None
        self.monitoring_thread = None
        
        # Статистика
        self.stats = {
            'jobs_scheduled': 0,
            'jobs_executed': 0,
            'jobs_failed': 0,
            'auto_triggers_fired': 0,
            'optimization_cycles': 0
        }
        
        # Настройка логирования
        self.logger = logging.getLogger('OptimizationScheduler')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.output_dir / 'scheduler.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def add_auto_rule(self, name: str, condition: Callable[[Dict[str, Any]], bool], 
                     optimization_func: Callable[[], Dict[str, Any]], 
                     priority: int = 3, description: str = ""):
        """
        Добавляет правило автоматического выполнения оптимизации
        
        Args:
            name: Название правила
            condition: Функция условия (возвращает True если нужно выполнить)
            optimization_func: Функция выполнения оптимизации
            priority: Приоритет (1-5)
            description: Описание правила
        """
        rule = {
            'name': name,
            'condition': condition,
            'optimization_func': optimization_func,
            'priority': priority,
            'description': description,
            'last_triggered': None
        }
        self.auto_rules.append(rule)
        self.logger.info(f"Добавлено автоправило: {name}")
    
    def create_optimization_job(self, name: str, optimization_type: str, 
                              priority: int = 3, target_metrics: List[str] = None,
                              trigger_condition: str = "", trigger_value: float = 0.0) -> OptimizationJob:
        """
        Создает задание на оптимизацию
        
        Args:
            name: Название задания
            optimization_type: Тип оптимизации ('cpu', 'memory', 'resource', 'comprehensive', etc.)
            priority: Приоритет (1-5)
            target_metrics: Целевые метрики
            trigger_condition: Условие срабатывания
            trigger_value: Значение для условия
            
        Returns:
            Созданное задание на оптимизацию
        """
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.scheduled_jobs)}"
        
        if target_metrics is None:
            target_metrics = []
        
        job = OptimizationJob(
            id=job_id,
            name=name,
            scheduled_time=datetime.now(),
            optimization_type=optimization_type,
            priority=priority,
            target_metrics=target_metrics,
            trigger_condition=trigger_condition,
            trigger_value=trigger_value
        )
        
        return job
    
    def schedule_job(self, job: OptimizationJob):
        """
        Планирует выполнение задания
        
        Args:
            job: Задание на оптимизацию
        """
        # Добавляем в очередь с приоритетом (меньше число - выше приоритет)
        priority = 5 - job.priority  # Инвертируем, так как очередь мин-куча
        self.job_queue.put((priority, job.id, job))
        
        scheduled_item = ScheduledOptimization(
            job=job,
            scheduled_at=datetime.now(),
            status='scheduled'
        )
        
        self.scheduled_jobs[job.id] = scheduled_item
        self.stats['jobs_scheduled'] += 1
        
        self.logger.info(f"Запланировано задание: {job.name} (ID: {job.id}), приоритет: {job.priority}")
    
    def execute_job(self, job: OptimizationJob) -> Dict[str, Any]:
        """
        Выполняет задание на оптимизацию
        
        Args:
            job: Задание на оптимизацию
            
        Returns:
            Результат выполнения
        """
        self.logger.info(f"Выполнение задания: {job.name} (ID: {job.id})")
        
        start_time = datetime.now()
        result = {}
        
        try:
            # Выполняем оптимизацию в зависимости от типа
            if job.optimization_type == 'cpu':
                result = self.resource_manager.optimize_cpu_usage()
            elif job.optimization_type == 'memory':
                result = self.memory_tracker.perform_memory_optimization()
            elif job.optimization_type == 'resource':
                result = self.resource_manager.optimize_all_resources()
            elif job.optimization_type == 'comprehensive':
                result = self.orchestrator.start_comprehensive_optimization(['core_utils'])
            elif job.optimization_type == 'profiling':
                result = self.performance_profiler.profile_function(lambda: print("Profiling test"))()
            elif job.optimization_type == 'benchmarking':
                result = self.benchmark_suite.benchmark_function("test", lambda x: x**2, 1000, iterations=5)
            else:
                # Общий случай - пробуем выполнить как комплексную оптимизацию
                result = self.orchestrator.start_comprehensive_optimization(['core_utils'])
            
            # Обновляем статус задания
            if job.id in self.scheduled_jobs:
                self.scheduled_jobs[job.id].status = 'completed'
                self.scheduled_jobs[job.id].execution_time = datetime.now()
                self.scheduled_jobs[job.id].result = result
            
            job.executed = True
            job.execution_time = datetime.now()
            job.result = result
            
            self.executed_jobs.append(job)
            self.stats['jobs_executed'] += 1
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Задание {job.name} выполнено за {execution_time:.2f}с")
            
            return {
                'success': True,
                'result': result,
                'execution_time': execution_time,
                'job_id': job.id
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка выполнения задания {job.name} (ID: {job.id}): {str(e)}")
            
            # Обновляем статус задания как неудачное
            if job.id in self.scheduled_jobs:
                self.scheduled_jobs[job.id].status = 'failed'
            
            self.failed_jobs.append(job)
            self.stats['jobs_failed'] += 1
            
            return {
                'success': False,
                'error': str(e),
                'job_id': job.id
            }
    
    def run_scheduler_cycle(self):
        """Выполняет один цикл планировщика"""
        # Проверяем автоматические правила
        current_metrics = self.monitoring_center.get_current_metrics()
        
        for rule in self.auto_rules:
            try:
                if rule['condition'](current_metrics):
                    if rule['last_triggered'] is None or \
                       (datetime.now() - rule['last_triggered']).seconds > 60:  # Не чаще раз в минуту
                        
                        self.logger.info(f"Срабатывание автоправила: {rule['name']}")
                        
                        # Выполняем оптимизацию
                        result = rule['optimization_func']()
                        
                        rule['last_triggered'] = datetime.now()
                        self.stats['auto_triggers_fired'] += 1
                        
                        self.logger.info(f"Автоправило {rule['name']} выполнено: {result}")
                        
            except Exception as e:
                self.logger.error(f"Ошибка в автоправиле {rule['name']}: {str(e)}")
        
        # Выполняем запланированные задания
        while not self.job_queue.empty():
            try:
                priority, job_id, job = self.job_queue.get_nowait()
                
                if job.id in self.scheduled_jobs:
                    self.scheduled_jobs[job.id].status = 'running'
                
                result = self.execute_job(job)
                
            except queue.Empty:
                break  # Очередь пуста
    
    def start_scheduler(self, interval: float = 5.0):
        """
        Запускает планировщик в фоновом режиме
        
        Args:
            interval: Интервал между циклами планировщика (в секундах)
        """
        if self.running:
            return
        
        self.running = True
        
        def scheduler_loop():
            while self.running:
                try:
                    self.run_scheduler_cycle()
                    self.stats['optimization_cycles'] += 1
                    time.sleep(interval)
                except Exception as e:
                    self.logger.error(f"Ошибка в цикле планировщика: {str(e)}")
                    time.sleep(interval)
        
        self.scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        print("⏰ Планировщик оптимизации запущен")
        self.logger.info("Планировщик оптимизации запущен")
    
    def stop_scheduler(self):
        """Останавливает планировщик"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=2.0)
        
        print("🛑 Планировщик оптимизации остановлен")
        self.logger.info("Планировщик оптимизации остановлен")
    
    def setup_default_rules(self):
        """Настройка стандартных правил автоматической оптимизации"""
        # Правило 1: Высокая загрузка CPU -> оптимизация CPU
        def cpu_high_condition(metrics):
            return metrics.get('cpu_percent', 0) > 80
        
        def cpu_optimization():
            return self.resource_manager.optimize_cpu_usage()
        
        self.add_auto_rule(
            name="high_cpu_optimization",
            condition=cpu_high_condition,
            optimization_func=cpu_optimization,
            priority=5,
            description="Оптимизация CPU при высокой загрузке (>80%)"
        )
        
        # Правило 2: Высокое использование памяти -> оптимизация памяти
        def memory_high_condition(metrics):
            return metrics.get('memory_percent', 0) > 85
        
        def memory_optimization():
            return self.memory_tracker.perform_memory_optimization()
        
        self.add_auto_rule(
            name="high_memory_optimization",
            condition=memory_high_condition,
            optimization_func=memory_optimization,
            priority=5,
            description="Оптимизация памяти при высоком использовании (>85%)"
        )
        
        # Правило 3: Низкая эффективность ресурсов -> комплексная оптимизация
        def low_efficiency_condition(metrics):
            return metrics.get('resource_efficiency', 100) < 70
        
        def efficiency_optimization():
            return self.orchestrator.start_comprehensive_optimization(['core_utils'])
        
        self.add_auto_rule(
            name="low_efficiency_optimization",
            condition=low_efficiency_condition,
            optimization_func=efficiency_optimization,
            priority=4,
            description="Комплексная оптимизация при низкой эффективности (<70%)"
        )
        
        # Правило 4: Подозрительная активность -> профилирование
        def suspicious_activity_condition(metrics):
            return (metrics.get('cpu_percent', 0) > 90 or 
                   metrics.get('memory_percent', 0) > 95 or
                   metrics.get('active_processes', 0) > 200)  # Подозрительное количество процессов
        
        def diagnostic_optimization():
            return self.performance_profiler.profile_function(lambda: print("Diagnostic scan"))()
        
        self.add_auto_rule(
            name="suspicious_activity_monitoring",
            condition=suspicious_activity_condition,
            optimization_func=diagnostic_optimization,
            priority=5,
            description="Диагностическое профилирование при подозрительной активности"
        )
        
        print(f"✅ Установлено {len(self.auto_rules)} стандартных правил автоматической оптимизации")
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """
        Получает статус планировщика
        
        Returns:
            Словарь с информацией о состоянии планировщика
        """
        return {
            'running': self.running,
            'stats': self.stats,
            'queued_jobs': self.job_queue.qsize(),
            'scheduled_jobs': len(self.scheduled_jobs),
            'executed_jobs': len(self.executed_jobs),
            'failed_jobs': len(self.failed_jobs),
            'auto_rules_count': len(self.auto_rules),
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_optimization_report(self, output_path: Optional[str] = None) -> str:
        """
        Генерирует отчет об оптимизациях
        
        Args:
            output_path: Путь для сохранения отчета (опционально)
            
        Returns:
            Путь к созданному отчету
        """
        if output_path is None:
            filename = f"optimization_schedule_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_path = str(self.output_dir / filename)
        
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'optimization_schedule_report'
            },
            'status': self.get_scheduler_status(),
            'recent_executed_jobs': [
                {
                    'id': job.id,
                    'name': job.name,
                    'type': job.optimization_type,
                    'executed_at': job.execution_time.isoformat() if job.execution_time else None,
                    'result_keys': list(job.result.keys()) if job.result else []
                }
                for job in self.executed_jobs[-10:]  # Последние 10 выполненных заданий
            ],
            'recent_failed_jobs': [
                {
                    'id': job.id,
                    'name': job.name,
                    'type': job.optimization_type,
                }
                for job in self.failed_jobs[-5:]  # Последние 5 неудачных заданий
            ],
            'auto_rules': [
                {
                    'name': rule['name'],
                    'description': rule['description'],
                    'priority': rule['priority'],
                    'last_triggered': rule['last_triggered'].isoformat() if rule['last_triggered'] else None
                }
                for rule in self.auto_rules
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"📊 Отчет об оптимизациях сохранен: {output_path}")
        return output_path
    
    def add_predictive_optimization_job(self, prediction_result: Dict[str, Any]):
        """
        Добавляет задание на оптимизацию на основе прогноза
        
        Args:
            prediction_result: Результат прогноза из предиктивного движка
        """
        if not prediction_result.get('recommendation'):
            return
        
        # Определяем тип оптимизации на основе рекомендации
        recommendation = prediction_result['recommendation'].lower()
        
        if 'cpu' in recommendation:
            opt_type = 'cpu'
            priority = 4
        elif 'memory' in recommendation:
            opt_type = 'memory'
            priority = 4
        elif 'resource' in recommendation or 'efficiency' in recommendation:
            opt_type = 'resource'
            priority = 3
        elif 'comprehensive' in recommendation:
            opt_type = 'comprehensive'
            priority = 5
        else:
            opt_type = 'comprehensive'
            priority = 3
        
        # Создаем задание
        job = self.create_optimization_job(
            name=f"Predictive_Opt_{prediction_result.get('metric', 'unknown')}",
            optimization_type=opt_type,
            priority=priority,
            target_metrics=[prediction_result.get('metric', '')],
            trigger_condition='predicted',
            trigger_value=prediction_result.get('predicted_value', 0)
        )
        
        # Планируем выполнение
        self.schedule_job(job)
        
        self.logger.info(f"Добавлено предиктивное задание: {job.name} на основе прогноза")
    
    def integrate_with_predictive_engine(self):
        """
        Интегрирует планировщик с предиктивным движком
        """
        def predictive_monitoring():
            while self.running:
                try:
                    # Получаем предиктивные инсайты
                    insights = self.predictive_engine.get_predictive_insights()
                    
                    # Обрабатываем прогнозы и создаем задания
                    for metric, predictions in insights.get('predictions', {}).items():
                        for timeframe, pred_data in predictions.items():
                            if pred_data['confidence'] > 0.7:  # Высокая достоверность
                                prediction_result = {
                                    'metric': metric,
                                    'predicted_value': pred_data['predicted_value'],
                                    'confidence': pred_data['confidence'],
                                    'recommendation': pred_data['recommendation']
                                }
                                
                                self.add_predictive_optimization_job(prediction_result)
                    
                    time.sleep(120)  # Проверяем прогнозы каждые 2 минуты
                    
                except Exception as e:
                    self.logger.error(f"Ошибка в предиктивном мониторинге: {str(e)}")
                    time.sleep(120)
        
        # Запускаем предиктивный мониторинг в отдельном потоке
        pred_thread = threading.Thread(target=predictive_monitoring, daemon=True)
        pred_thread.start()
        
        print("🔮 Интеграция с предиктивным движком завершена")


def main():
    """Главная функция для демонстрации возможностей планировщика"""
    print("=== АВТОМАТИЧЕСКИЙ ПЛАНИРОВЩИК ОПТИМИЗАЦИИ ===")
    print("⏰ Инициализация планировщика оптимизации...")
    
    # Создаем планировщик
    scheduler = AutomatedOptimizationScheduler(output_dir="automated_optimization")
    
    # Настраиваем стандартные правила
    scheduler.setup_default_rules()
    
    # Пример добавления ручного задания
    print("\n➕ Добавление тестового задания...")
    test_job = scheduler.create_optimization_job(
        name="Тестовая оптимизация ресурсов",
        optimization_type="resource",
        priority=4,
        target_metrics=["cpu_percent", "memory_percent"],
        trigger_condition="manual"
    )
    scheduler.schedule_job(test_job)
    
    print(f"✅ Задание '{test_job.name}' запланировано")
    
    # Показываем статус
    status = scheduler.get_scheduler_status()
    print(f"\n📊 Текущий статус:")
    print(f"   • Планировщик запущен: {status['running']}")
    print(f"   • Запланировано заданий: {status['scheduled_jobs']}")
    print(f"   • Выполнено заданий: {status['stats']['jobs_executed']}")
    print(f"   • Автоматических правил: {status['auto_rules_count']}")
    
    # Интеграция с предиктивным движком
    print("\n🔄 Интеграция с предиктивным движком...")
    scheduler.integrate_with_predictive_engine()
    
    print(f"\n🔗 Доступные функции:")
    print("   • Планирование заданий: scheduler.schedule_job()")
    print("   • Статус: scheduler.get_scheduler_status()")
    print("   • Отчеты: scheduler.generate_optimization_report()")
    print("   • Запуск: scheduler.start_scheduler()")
    print("   • Автоправила: scheduler.setup_default_rules()")
    
    print("\n🎉 Планировщик оптимизации готов к работе!")
    
    # Показываем примеры использования
    print(f"\n💡 Примеры использования:")
    print("   # Создать задание на оптимизацию CPU")
    print("   job = scheduler.create_optimization_job('CPU Optimization', 'cpu', priority=5)")
    print("   scheduler.schedule_job(job)")
    print("")
    print("   # Запустить планировщик")
    print("   scheduler.start_scheduler()")
    print("")
    print("   # Сгенерировать отчет")
    print("   scheduler.generate_optimization_report()")


if __name__ == "__main__":
    main()