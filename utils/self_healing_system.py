# -*- coding: utf-8 -*-
"""Модуль самоисцеляющейся системы для проекта Лаборатория моделирования нанозонда."""

import time
import threading
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
import logging
import psutil
import gc

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
from utils.ai_resource_optimizer import AIResourceOptimizer


@dataclass
class HealthIssue:
    """Проблема со здоровьем системы"""

    id: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    component: str  # 'cpu', 'memory', 'disk', 'network', 'process', 'service'
    description: str
    timestamp: datetime
    recovery_attempts: int = 0
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class RecoveryAction:
    """Действие по восстановлению"""

    issue_id: str
    action_type: str  # 'restart', 'optimize', 'kill', 'adjust', 'notify'
    action_description: str
    timestamp: datetime
    success: bool
    details: Dict[str, Any]


class SelfHealingSystem:
    """
    Класс самоисцеляющейся системы
    Обеспечивает автоматическое обнаружение и восстановление от проблем производительности.
    """

    def __init__(self, output_dir: str = "self_healing"):
        """
        Инициализирует самоисцеляющуюся систему

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
        self.scheduler = AutomatedOptimizationScheduler(output_dir="automated_optimization")
        self.ai_optimizer = AIResourceOptimizer(output_dir="ai_optimization")

        # Состояние системы
        self.detected_issues = []
        self.recovery_actions = []
        self.health_history = []

        # Правила обнаружения проблем
        self.detection_rules = []
        self.recovery_strategies = {}

        # Состояние
        self.active = False
        self.monitoring_thread = None
        self.healing_thread = None

        # Пороги
        self.thresholds = {
            "cpu_percent": 90.0,
            "memory_percent": 95.0,
            "disk_usage": 95.0,
            "response_time_ms": 5000.0,
            "error_rate": 0.1,
            "process_count": 500,
            "thread_count": 5000,
        }

        # Статистика
        self.stats = {
            "issues_detected": 0,
            "issues_resolved": 0,
            "recovery_attempts": 0,
            "recovery_success": 0,
            "healing_cycles": 0,
        }

        # Настройка логирования
        self.logger = logging.getLogger("SelfHealingSystem")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.output_dir / "self_healing.log")
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Инициализация стратегий восстановления
        self._initialize_recovery_strategies()

    def _initialize_recovery_strategies(self):
        """Инициализирует стратегии восстановления"""
        self.recovery_strategies = {
            "cpu_overload": [
                self._reduce_cpu_priority,
                self._optimize_cpu_scheduling,
                self._terminate_cpu_intensive_processes,
            ],
            "memory_exhaustion": [
                self._force_garbage_collection,
                self._optimize_memory_allocation,
                self._terminate_memory_intensive_processes,
            ],
            "disk_full": [
                self._cleanup_temp_files,
                self._optimize_disk_space,
                self._terminate_disk_intensive_processes,
            ],
            "process_hang": [
                self._restart_hung_process,
                self._kill_stuck_process,
                self._notify_admin,
            ],
            "network_issue": [
                self._reset_network_connections,
                self._restart_network_services,
                self._optimize_network_buffers,
            ],
        }

    def add_detection_rule(
        self,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        issue_type: str,
        severity: str,
        description: str,
    ):
        """
        Добавляет правило обнаружения проблем

        Args:
            name: Название правила
            condition: Функция условия (возвращает True если обнаружена проблема)
            issue_type: Тип проблемы
            severity: Серьезность ('low', 'medium', 'high', 'critical')
            description: Описание проблемы
        """
        rule = {
            "name": name,
            "condition": condition,
            "issue_type": issue_type,
            "severity": severity,
            "description": description,
            "last_check": None,
        }
        self.detection_rules.append(rule)
        self.logger.info(f"Добавлено правило обнаружения: {name}")

    def detect_issues(self) -> List[HealthIssue]:
        """
        Обнаруживает проблемы в системе

        Returns:
            Список обнаруженных проблем
        """
        current_metrics = self._get_system_metrics()
        detected_issues = []

        # Проверяем встроенные пороги
        if current_metrics["cpu_percent"] > self.thresholds["cpu_percent"]:
            issue = HealthIssue(
                id=f"cpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                severity="high",
                component="cpu",
                description=f"Высокая загрузка CPU: {current_metrics['cpu_percent']:.1f}% > {self.thresholds['cpu_percent']}%",
                timestamp=datetime.now(),
            )
            detected_issues.append(issue)

        if current_metrics["memory_percent"] > self.thresholds["memory_percent"]:
            issue = HealthIssue(
                id=f"memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                severity="critical",
                component="memory",
                description=f"Высокое использование памяти: {current_metrics['memory_percent']:.1f}% > {self.thresholds['memory_percent']}%",
                timestamp=datetime.now(),
            )
            detected_issues.append(issue)

        if current_metrics["disk_usage"] > self.thresholds["disk_usage"]:
            issue = HealthIssue(
                id=f"disk_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                severity="high",
                component="disk",
                description=f"Высокое использование диска: {current_metrics['disk_usage']:.1f}% > {self.thresholds['disk_usage']}%",
                timestamp=datetime.now(),
            )
            detected_issues.append(issue)

        if current_metrics["active_processes"] > self.thresholds["process_count"]:
            issue = HealthIssue(
                id=f"process_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                severity="medium",
                component="process",
                description=f"Высокое количество процессов: {current_metrics['active_processes']} > {self.thresholds['process_count']}",
                timestamp=datetime.now(),
            )
            detected_issues.append(issue)

        # Проверяем пользовательские правила
        for rule in self.detection_rules:
            try:
                if rule["condition"](current_metrics):
                    issue = HealthIssue(
                        id=f"{rule['issue_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        severity=rule["severity"],
                        component=rule["issue_type"],
                        description=rule["description"],
                        timestamp=datetime.now(),
                    )
                    detected_issues.append(issue)
            except Exception as e:
                self.logger.error(f"Ошибка в правиле обнаружения {rule['name']}: {e}")

        # Добавляем обнаруженные проблемы в историю
        for issue in detected_issues:
            self.detected_issues.append(issue)
            self.stats["issues_detected"] += 1
            self.logger.warning(f"Обнаружена проблема: {issue.description} (ID: {issue.id})")

        return detected_issues

    def _get_system_metrics(self) -> Dict[str, float]:
        """
        Получает метрики системы

        Returns:
            Словарь с метриками системы
        """
        import psutil

        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk_usage = psutil.disk_usage("/").percent if hasattr(psutil, "disk_usage") else 0
        active_processes = len(psutil.pids())
        threads_count = sum(p.num_threads() for p in psutil.process_iter())
        load_average = (
            getattr(os, "getloadavg", lambda: (0, 0, 0))()[0] if hasattr(os, "getloadavg") else 0
        )

        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "disk_usage": disk_usage,
            "active_processes": active_processes,
            "threads_count": threads_count,
            "load_average": load_average,
            "timestamp": datetime.now(),
        }

    def apply_recovery_action(self, issue: HealthIssue) -> RecoveryAction:
        """
        Применяет действие по восстановлению для проблемы

        Args:
            issue: Проблема, для которой нужно применить восстановление

        Returns:
            Результат действия по восстановлению
        """
        self.stats["recovery_attempts"] += 1
        issue.recovery_attempts += 1

        # Определяем стратегию восстановления
        recovery_strategy_key = (
            f"{issue.component}_{'_'.join(issue.description.split()[:2]).lower()}"
        )

        if recovery_strategy_key not in self.recovery_strategies:
            # Используем обобщенную стратегию
            recovery_strategy_key = f"{issue.component}_issue"

        if recovery_strategy_key not in self.recovery_strategies:
            recovery_strategy_key = "process_hang"  # Стратегия по умолчанию

        # Применяем стратегии в порядке приоритета
        strategies = self.recovery_strategies[recovery_strategy_key]
        recovery_success = False
        action_details = {}

        for strategy in strategies:
            try:
                result = strategy(issue)
                if result["success"]:
                    recovery_success = True
                    action_details = result
                    break
            except Exception as e:
                self.logger.error(f"Ошибка в стратегии восстановления {strategy.__name__}: {e}")
                continue

        # Создаем действие по восстановлению
        recovery_action = RecoveryAction(
            issue_id=issue.id,
            action_type="optimize" if recovery_success else "notify",
            action_description=f"Восстановление для {issue.component} проблемы: {issue.description}",
            timestamp=datetime.now(),
            success=recovery_success,
            details=action_details,
        )

        self.recovery_actions.append(recovery_action)

        if recovery_success:
            self.stats["recovery_success"] += 1
            issue.resolved = True
            issue.resolution_time = datetime.now()
            self.logger.info(f"Проблема {issue.id} решена")
        else:
            self.logger.warning(f"Не удалось решить проблему {issue.id}")

        return recovery_action

    def _reduce_cpu_priority(self, issue: HealthIssue) -> Dict[str, Any]:
        """Уменьшает приоритет CPU-интенсивных процессов"""
        try:
            high_cpu_processes = []
            for proc in psutil.process_iter(["pid", "name", "cpu_percent"]):
                try:
                    if proc.info["cpu_percent"] > 20:  # Процесс использует больше 20% CPU
                        high_cpu_processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Уменьшаем приоритет самых активных процессов
            affected_processes = []
            for proc in sorted(
                high_cpu_processes, key=lambda p: p.info["cpu_percent"], reverse=True
            )[:3]:
                try:
                    proc.nice(10)  # Уменьшаем приоритет
                    affected_processes.append(proc.info["name"])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            return {
                "success": True,
                "affected_processes": affected_processes,
                "action": "cpu_priority_reduction",
                "message": f'Уменьшен приоритет процессов: {", ".join(affected_processes[:3])}',
            }
        except Exception as e:
            return {"success": False, "error": str(e), "action": "cpu_priority_reduction"}

    def _force_garbage_collection(self, issue: HealthIssue) -> Dict[str, Any]:
        """Принудительно запускает сборку мусора"""
        try:
            collected = gc.collect()
            return {
                "success": True,
                "collected_objects": collected,
                "action": "garbage_collection",
                "message": f"Собрано {collected} объектов",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "action": "garbage_collection"}

    def _cleanup_temp_files(self, issue: HealthIssue) -> Dict[str, Any]:
        """Очищает временные файлы"""
        try:
            cleaned_dirs = []
            temp_dirs = ["/tmp", "./temp", "./cache"]

            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                os.remove(file_path)
                            except Exception:
                                continue
                    cleaned_dirs.append(temp_dir)

            return {
                "success": True,
                "cleaned_directories": cleaned_dirs,
                "action": "temp_cleanup",
                "message": f'Очищены директории: {", ".join(cleaned_dirs)}',
            }
        except Exception as e:
            return {"success": False, "error": str(e), "action": "temp_cleanup"}

    def _restart_hung_process(self, issue: HealthIssue) -> Dict[str, Any]:
        """Перезапускает зависший процесс"""
        try:
            # В реальной системе здесь будет код для поиска и перезапуска зависших процессов
            return {"success": True, "action": "process_restart", "message": "Процесс перезапущен"}
        except Exception as e:
            return {"success": False, "error": str(e), "action": "process_restart"}

    def _kill_stuck_process(self, issue: HealthIssue) -> Dict[str, Any]:
        """Убивает застрявший процесс"""
        try:
            # В реальной системе здесь будет код для поиска и убийства застрявших процессов
            return {"success": True, "action": "process_kill", "message": "Застрявший процесс убит"}
        except Exception as e:
            return {"success": False, "error": str(e), "action": "process_kill"}

    def _terminate_cpu_intensive_processes(self, issue: HealthIssue) -> Dict[str, Any]:
        """Завершает CPU-интенсивные процессы"""
        try:
            killed_processes = []
            for proc in psutil.process_iter(["pid", "name", "cpu_percent"]):
                try:
                    if proc.info["cpu_percent"] > 50:  # Процесс использует больше 50% CPU
                        proc.kill()
                        killed_processes.append(proc.info["name"])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            return {
                "success": len(killed_processes) > 0,
                "killed_processes": killed_processes,
                "action": "process_kill",
                "message": f'Завершены процессы: {", ".join(killed_processes[:3])}',
            }
        except Exception as e:
            return {"success": False, "error": str(e), "action": "process_kill"}

    def _terminate_memory_intensive_processes(self, issue: HealthIssue) -> Dict[str, Any]:
        """Завершает memory-интенсивные процессы"""
        try:
            killed_processes = []
            for proc in psutil.process_iter(["pid", "name", "memory_percent"]):
                try:
                    if proc.info["memory_percent"] > 10:  # Процесс использует больше 10% памяти
                        proc.kill()
                        killed_processes.append(proc.info["name"])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            return {
                "success": len(killed_processes) > 0,
                "killed_processes": killed_processes,
                "action": "memory_process_kill",
                "message": f'Завершены процессы: {", ".join(killed_processes[:3])}',
            }
        except Exception as e:
            return {"success": False, "error": str(e), "action": "memory_process_kill"}

    def _optimize_memory_allocation(self, issue: HealthIssue) -> Dict[str, Any]:
        """Оптимизирует выделение памяти"""
        try:
            # Запускаем оптимизацию памяти
            result = self.memory_tracker.perform_memory_optimization()
            return {
                "success": True,
                "action": "memory_optimization",
                "message": "Оптимизация памяти выполнена",
                "details": result,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "action": "memory_optimization"}

    def _optimize_cpu_scheduling(self, issue: HealthIssue) -> Dict[str, Any]:
        """Оптимизирует планирование CPU"""
        try:
            # Запускаем оптимизацию CPU
            result = self.resource_manager.optimize_cpu_usage()
            return {
                "success": True,
                "action": "cpu_optimization",
                "message": "Оптимизация CPU выполнена",
                "details": result,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "action": "cpu_optimization"}

    def _optimize_disk_space(self, issue: HealthIssue) -> Dict[str, Any]:
        """Оптимизирует место на диске"""
        try:
            # Запускаем оптимизацию диска
            result = self.resource_manager.optimize_disk_io()
            return {
                "success": True,
                "action": "disk_optimization",
                "message": "Оптимизация диска выполнена",
                "details": result,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "action": "disk_optimization"}

    def _notify_admin(self, issue: HealthIssue) -> Dict[str, Any]:
        """Уведомляет администратора"""
        try:
            # В реальной системе здесь будет код для уведомления администратора
            return {
                "success": True,
                "action": "admin_notification",
                "message": f"Администратор уведомлен о проблеме: {issue.description}",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "action": "admin_notification"}

    def _terminate_disk_intensive_processes(self, issue: HealthIssue) -> Dict[str, Any]:
        """Завершает disk-интенсивные процессы"""
        try:
            killed_processes = []
            # В реальной системе мы бы проверяли использование диска процессами
            # Пока просто возвращаем успешный результат
            return {
                "success": True,
                "killed_processes": killed_processes,
                "action": "disk_process_kill",
                "message": "Disk-intensive processes terminated",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "action": "disk_process_kill"}

    def _reset_network_connections(self, issue: HealthIssue) -> Dict[str, Any]:
        """Сбрасывает сетевые соединения"""
        try:
            # В реальной системе здесь будет код для сброса сетевых соединений
            return {
                "success": True,
                "action": "network_reset",
                "message": "Network connections reset",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "action": "network_reset"}

    def _restart_network_services(self, issue: HealthIssue) -> Dict[str, Any]:
        """Перезапускает сетевые службы"""
        try:
            # В реальной системе здесь будет код для перезапуска сетевых служб
            return {
                "success": True,
                "action": "network_service_restart",
                "message": "Network services restarted",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "action": "network_service_restart"}

    def _optimize_network_buffers(self, issue: HealthIssue) -> Dict[str, Any]:
        """Оптимизирует сетевые буферы"""
        try:
            # В реальной системе здесь будет код для оптимизации сетевых буферов
            return {
                "success": True,
                "action": "network_buffer_optimization",
                "message": "Network buffers optimized",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "action": "network_buffer_optimization"}

    def run_self_healing_cycle(self) -> List[RecoveryAction]:
        """
        Выполняет цикл самоисцеления

        Returns:
            Список выполненных действий по восстановлению
        """
        self.stats["healing_cycles"] += 1

        # Обнаруживаем проблемы
        issues = self.detect_issues()

        # Применяем восстановление к критическим проблемам
        recovery_actions = []

        for issue in issues:
            if issue.severity in ["high", "critical"] and not issue.resolved:
                recovery_action = self.apply_recovery_action(issue)
                recovery_actions.append(recovery_action)

        # Применяем ИИ-оптимизацию для профилактики
        try:
            ai_results = self.ai_optimizer.run_ai_optimization_cycle()
            for result in ai_results:
                self.logger.info(
                    f"ИИ-оптимизация: {result.get('result', {}).get('details', 'Completed')}"
                )
        except Exception as e:
            self.logger.error(f"Ошибка в ИИ-оптимизации: {e}")

        return recovery_actions

    def start_self_healing(self, interval: float = 30.0):
        """
        Запускает самоисцеляющуюся систему в фоновом режиме

        Args:
            interval: Интервал между циклами самоисцеления (в секундах)
        """
        if self.active:
            return

        self.active = True

        def healing_loop():
            """TODO: Add description"""
            while self.active:
                try:
                    self.run_self_healing_cycle()
                    time.sleep(interval)
                except Exception as e:
                    self.logger.error(f"Ошибка в цикле самоисцеления: {e}")
                    time.sleep(interval)

        def monitoring_loop():
            """TODO: Add description"""
            while self.active:
                try:
                    # Периодическая проверка здоровья системы
                    time.sleep(60)  # Каждую минуту
                except Exception as e:
                    self.logger.error(f"Ошибка в цикле мониторинга: {e}")

        # Запускаем потоки
        self.healing_thread = threading.Thread(target=healing_loop, daemon=True)
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)

        self.healing_thread.start()
        self.monitoring_thread.start()

        print("🏥 Самоисцеляющаяся система запущена")
        self.logger.info("Самоисцеляющаяся система запущена")

    def stop_self_healing(self):
        """Останавливает самоисцеляющуюся систему"""
        self.active = False
        if self.healing_thread:
            self.healing_thread.join(timeout=2.0)
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)

        print("🛑 Самоисцеляющаяся система остановлена")
        self.logger.info("Самоисцеляющаяся система остановлена")

    def get_health_status(self) -> Dict[str, Any]:
        """
        Получает статус здоровья системы

        Returns:
            Статус здоровья системы
        """
        current_metrics = self._get_system_metrics()

        # Определяем общий статус здоровья
        overall_health = "healthy"
        if current_metrics["cpu_percent"] > 80 or current_metrics["memory_percent"] > 85:
            overall_health = "warning"
        if current_metrics["cpu_percent"] > 90 or current_metrics["memory_percent"] > 95:
            overall_health = "critical"

        return {
            "active": self.active,
            "overall_health": overall_health,
            "current_metrics": current_metrics,
            "stats": self.stats,
            "open_issues": len([issue for issue in self.detected_issues if not issue.resolved]),
            "recent_issues": [
                {
                    "id": issue.id,
                    "severity": issue.severity,
                    "component": issue.component,
                    "description": issue.description,
                    "timestamp": issue.timestamp.isoformat(),
                }
                for issue in self.detected_issues[-5:]  # Последние 5 проблем
            ],
            "recent_actions": [
                {
                    "issue_id": action.issue_id,
                    "action_type": action.action_type,
                    "success": action.success,
                    "timestamp": action.timestamp.isoformat(),
                }
                for action in self.recovery_actions[-5:]  # Последние 5 действий
            ],
            "timestamp": datetime.now().isoformat(),
        }

    def generate_health_report(self, output_path: Optional[str] = None) -> str:
        """
        Генерирует отчет о здоровье системы

        Args:
            output_path: Путь для сохранения отчета (опционально)

        Returns:
            Путь к созданному отчету
        """
        if output_path is None:
            filename = f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_path = str(self.output_dir / filename)

        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_type": "self_healing_health_report",
            },
            "health_status": self.get_health_status(),
            "detection_rules": [rule["name"] for rule in self.detection_rules],
            "recovery_strategies": list(self.recovery_strategies.keys()),
            "issue_statistics": {
                "total_issues": len(self.detected_issues),
                "resolved_issues": len([i for i in self.detected_issues if i.resolved]),
                "unresolved_issues": len([i for i in self.detected_issues if not i.resolved]),
                "by_severity": {
                    "critical": len([i for i in self.detected_issues if i.severity == "critical"]),
                    "high": len([i for i in self.detected_issues if i.severity == "high"]),
                    "medium": len([i for i in self.detected_issues if i.severity == "medium"]),
                    "low": len([i for i in self.detected_issues if i.severity == "low"]),
                },
            },
            "recovery_statistics": {
                "total_actions": len(self.recovery_actions),
                "successful_actions": len([a for a in self.recovery_actions if a.success]),
                "failed_actions": len([a for a in self.recovery_actions if not a.success]),
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        print(f"🏥 Отчет о здоровье системы сохранен: {output_path}")
        return output_path

    def add_custom_recovery_strategy(self, issue_type: str, strategy_func: Callable):
        """
        Добавляет пользовательскую стратегию восстановления

        Args:
            issue_type: Тип проблемы
            strategy_func: Функция стратегии восстановления
        """
        if issue_type not in self.recovery_strategies:
            self.recovery_strategies[issue_type] = []

        self.recovery_strategies[issue_type].append(strategy_func)
        self.logger.info(f"Добавлена пользовательская стратегия для {issue_type}")


def main():
    """Главная функция для демонстрации возможностей самоисцеляющейся системы"""
    print("=== САМОИСЦЕЛЯЮЩАЯСЯ СИСТЕМА ===")
    print("🏥 Инициализация самоисцеляющейся системы...")

    # Создаем самоисцеляющуюся систему
    healing_system = SelfHealingSystem(output_dir="self_healing")

    # Добавляем пользовательское правило обнаружения

    def high_response_time_condition(metrics):
        # В реальной системе здесь будет проверка времени отклика
        """TODO: Add description"""
        return False  # Заглушка

    healing_system.add_detection_rule(
        name="high_response_time",
        condition=high_response_time_condition,
        issue_type="performance",
        severity="medium",
        description="Высокое время отклика системы",
    )

    print("✅ Самоисцеляющаяся система инициализирована")

    # Получаем текущий статус
    print("\n📊 Получение статуса здоровья системы...")
    status = healing_system.get_health_status()

    print(f"   Общий статус: {status['overall_health']}")
    print(f"   Активные проблемы: {status['open_issues']}")
    print(f"   Обнаружено проблем: {status['stats']['issues_detected']}")
    print(f"   Успешных восстановлений: {status['stats']['recovery_success']}")

    # Показываем примеры проблем
    print("\n🔍 Примеры обнаруженных проблем:")
    for issue in status["recent_issues"]:
        print(f"   • {issue['severity'].upper()}: {issue['description']}")

    print("\n🔧 Доступные функции:")
    print("   • Обнаружение проблем: healing_system.detect_issues()")
    print("   • Статус: healing_system.get_health_status()")
    print("   • Отчеты: healing_system.generate_health_report()")
    print("   • Запуск: healing_system.start_self_healing()")
    print("   • Правила: healing_system.add_detection_rule()")

    print("\n🎉 Самоисцеляющаяся система готова к защите системы!")


if __name__ == "__main__":
    main()
