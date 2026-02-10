#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль отслеживания использования памяти для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для мониторинга, анализа и оптимизации использования памяти.
"""

import psutil
import gc
import tracemalloc
import threading
import time
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dataclasses import dataclass
from functools import wraps
try:
    import objgraph
except ImportError:
    objgraph = None
    print("Warning: objgraph not installed. Install with 'pip install objgraph'")


@dataclass
class MemorySnapshot:
    """Снимок памяти"""
    timestamp: datetime
    rss_mb: float
    vms_mb: float
    percent: float
    shared_mb: float
    text_mb: float
    lib_mb: float
    data_mb: float
    dirty_mb: float
    heap_size: Optional[float] = None
    heap_usage: Optional[float] = None


@dataclass
class MemoryLeakDetection:
    """Обнаружение утечки памяти"""
    object_type: str
    growth_rate: float  # Объектов в секунду
    current_count: int
    baseline_count: int
    duration_seconds: int
    severity: str  # low, medium, high, critical


class MemoryTracker:
    """
    Класс отслеживания памяти
    Обеспечивает мониторинг, анализ и обнаружение утечек памяти.
    """
    
    def __init__(self, output_dir: str = "memory_logs"):
        """
        Инициализирует трекер памяти
        
        Args:
            output_dir: Директория для сохранения логов памяти
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.snapshots = []
        self.tracking = False
        self.tracking_thread = None
        self.baseline_snapshots = []
        self.leak_detections = []
        self.tracemalloc_enabled = False
        self.current_process = psutil.Process()
        self.monitoring_history = {
            'timestamps': [],
            'rss_mb': [],
            'vms_mb': [],
            'percent': [],
            'gc_collections': []
        }
    
    def take_snapshot(self) -> MemorySnapshot:
        """
        Делает снимок текущего использования памяти
        
        Returns:
            Снимок памяти
        """
        memory_info = self.current_process.memory_info()
        memory_percent = self.current_process.memory_percent()
        
        # Конвертируем байты в мегабайты
        rss_mb = memory_info.rss / (1024 * 1024)
        vms_mb = memory_info.vms / (1024 * 1024)
        shared_mb = getattr(memory_info, 'shared', 0) / (1024 * 1024)
        text_mb = getattr(memory_info, 'text', 0) / (1024 * 1024)
        lib_mb = getattr(memory_info, 'lib', 0) / (1024 * 1024)
        data_mb = getattr(memory_info, 'data', 0) / (1024 * 1024)
        dirty_mb = getattr(memory_info, 'dirty', 0) / (1024 * 1024)
        
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            rss_mb=rss_mb,
            vms_mb=vms_mb,
            percent=memory_percent,
            shared_mb=shared_mb,
            text_mb=text_mb,
            lib_mb=lib_mb,
            data_mb=data_mb,
            dirty_mb=dirty_mb
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def start_tracking(self, interval: float = 1.0):
        """
        Начинает отслеживание использования памяти
        
        Args:
            interval: Интервал между замерами (в секундах)
        """
        if self.tracking:
            return
        
        self.tracking = True
        
        def track():
            while self.tracking:
                try:
                    snapshot = self.take_snapshot()
                    
                    # Сохраняем в историю для мониторинга
                    self.monitoring_history['timestamps'].append(snapshot.timestamp)
                    self.monitoring_history['rss_mb'].append(snapshot.rss_mb)
                    self.monitoring_history['vms_mb'].append(snapshot.vms_mb)
                    self.monitoring_history['percent'].append(snapshot.percent)
                    self.monitoring_history['gc_collections'].append(gc.get_count())
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    print(f"Ошибка отслеживания памяти: {e}")
                    time.sleep(interval)
        
        self.tracking_thread = threading.Thread(target=track, daemon=True)
        self.tracking_thread.start()
    
    def stop_tracking(self):
        """Останавливает отслеживание памяти"""
        self.tracking = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=2)
    
    def set_baseline(self):
        """Устанавливает базовую линию для сравнения"""
        self.baseline_snapshots = self.snapshots.copy()
        print(f"✓ Базовая линия установлена на основе {len(self.baseline_snapshots)} снимков")
    
    def get_current_memory_usage(self) -> Dict[str, float]:
        """
        Получает текущее использование памяти
        
        Returns:
            Словарь с информацией о памяти
        """
        memory_info = self.current_process.memory_info()
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024),
            'percent': self.current_process.memory_percent(),
            'num_threads': self.current_process.num_threads(),
            'num_fds': self.current_process.num_fds() if hasattr(self.current_process, 'num_fds') else 0
        }
    
    def detect_memory_leaks(self, threshold_growth: float = 1.0) -> List[MemoryLeakDetection]:
        """
        Обнаруживает потенциальные утечки памяти
        
        Args:
            threshold_growth: Порог роста в MB/мин для обнаружения утечки
            
        Returns:
            Список обнаруженных утечек
        """
        if len(self.snapshots) < 2:
            return []
        
        leaks = []
        
        # Рассчитываем средний рост RSS за последний период
        recent_snapshots = self.snapshots[-10:] if len(self.snapshots) >= 10 else self.snapshots
        if len(recent_snapshots) < 2:
            return []
        
        time_diff = (recent_snapshots[-1].timestamp - recent_snapshots[0].timestamp).total_seconds()
        if time_diff <= 0:
            return []
        
        rss_diff = recent_snapshots[-1].rss_mb - recent_snapshots[0].rss_mb
        growth_rate_mb_per_min = (rss_diff / time_diff) * 60 if time_diff > 0 else 0
        
        if abs(growth_rate_mb_per_min) > threshold_growth:
            severity = "low"
            if abs(growth_rate_mb_per_min) > threshold_growth * 3:
                severity = "medium"
            if abs(growth_rate_mb_per_min) > threshold_growth * 5:
                severity = "high"
            if abs(growth_rate_mb_per_min) > threshold_growth * 10:
                severity = "critical"
            
            leak = MemoryLeakDetection(
                object_type="RSS Memory",
                growth_rate=growth_rate_mb_per_min,
                current_count=int(recent_snapshots[-1].rss_mb),
                baseline_count=int(recent_snapshots[0].rss_mb),
                duration_seconds=int(time_diff),
                severity=severity
            )
            leaks.append(leak)
        
        # Также проверяем с помощью objgraph
        if objgraph is not None:
            try:
                # Получаем топ-10 типов объектов
                obj_counts = objgraph.most_common_types(limit=10)
                for obj_type, count in obj_counts:
                    # Здесь можно добавить логику для обнаружения роста конкретных типов объектов
                    # Пока просто добавляем как потенциальную информацию
                    pass
            except Exception as e:
                print(f"Ошибка при использовании objgraph: {e}")
        
        self.leak_detections.extend(leaks)
        return leaks
    
    def trigger_garbage_collection(self) -> Dict[str, int]:
        """
        Вызывает сборку мусора и возвращает статистику
        
        Returns:
            Статистика сборки мусора
        """
        before_memory = self.get_current_memory_usage()['rss_mb']
        collected = gc.collect()
        after_memory = self.get_current_memory_usage()['rss_mb']
        
        return {
            'collected_objects': collected,
            'memory_freed_mb': before_memory - after_memory,
            'before_collection_mb': before_memory,
            'after_collection_mb': after_memory
        }
    
    def start_trace_malloc(self):
        """Начинает трассировку выделения памяти"""
        if not self.tracemalloc_enabled:
            tracemalloc.start()
            self.tracemalloc_enabled = True
            print("✓ Трассировка выделения памяти запущена")
    
    def stop_trace_malloc(self):
        """Останавливает трассировку выделения памяти"""
        if self.tracemalloc_enabled:
            tracemalloc.stop()
            self.tracemalloc_enabled = False
            print("✓ Трассировка выделения памяти остановлена")
    
    def get_trace_malloc_stats(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Получает статистику трассировки выделения памяти
        
        Args:
            limit: Ограничение количества результатов
            
        Returns:
            Список статистики по выделению памяти
        """
        if not self.tracemalloc_enabled:
            return []
        
        try:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            result = []
            for stat in top_stats[:limit]:
                result.append({
                    'filename': stat.traceback.format()[0] if stat.traceback else 'Unknown',
                    'size_mb': stat.size / (1024 * 1024),
                    'count': stat.count
                })
            
            return result
        except Exception as e:
            print(f"Ошибка получения статистики трассировки: {e}")
            return []
    
    def analyze_memory_usage(self) -> Dict[str, Any]:
        """
        Анализирует использование памяти
        
        Returns:
            Результаты анализа
        """
        if not self.snapshots:
            return {}
        
        # Собираем статистику
        rss_values = [s.rss_mb for s in self.snapshots]
        percent_values = [s.percent for s in self.snapshots]
        
        analysis = {
            'time_range': {
                'start': self.snapshots[0].timestamp.isoformat(),
                'end': self.snapshots[-1].timestamp.isoformat(),
                'duration_minutes': (self.snapshots[-1].timestamp - self.snapshots[0].timestamp).total_seconds() / 60
            },
            'rss_stats': {
                'min_mb': min(rss_values),
                'max_mb': max(rss_values),
                'avg_mb': sum(rss_values) / len(rss_values),
                'current_mb': rss_values[-1],
                'growth_mb': rss_values[-1] - rss_values[0] if len(rss_values) > 1 else 0
            },
            'percent_stats': {
                'min_percent': min(percent_values),
                'max_percent': max(percent_values),
                'avg_percent': sum(percent_values) / len(percent_values),
                'current_percent': percent_values[-1]
            },
            'total_snapshots': len(self.snapshots),
            'leak_detections': len(self.leak_detections),
            'baseline_comparison': self.compare_with_baseline()
        }
        
        return analysis
    
    def compare_with_baseline(self) -> Dict[str, float]:
        """
        Сравнивает текущее использование памяти с базовой линией
        
        Returns:
            Результаты сравнения
        """
        if not self.baseline_snapshots or not self.snapshots:
            return {}
        
        baseline_avg = sum(s.rss_mb for s in self.baseline_snapshots) / len(self.baseline_snapshots)
        current_avg = sum(s.rss_mb for s in self.snapshots[-10:]) / min(len(self.snapshots), 10)
        
        return {
            'baseline_avg_mb': baseline_avg,
            'current_avg_mb': current_avg,
            'difference_mb': current_avg - baseline_avg,
            'difference_percent': ((current_avg - baseline_avg) / baseline_avg) * 100 if baseline_avg > 0 else 0
        }
    
    def visualize_memory_usage(self, output_path: str = None) -> str:
        """
        Визуализирует использование памяти
        
        Args:
            output_path: Путь для сохранения визуализации
            
        Returns:
            Путь к сохраненной визуализации
        """
        if not self.monitoring_history['timestamps']:
            print("Нет данных для визуализации")
            return ""
        
        if output_path is None:
            output_path = str(self.output_dir / f"memory_usage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Анализ использования памяти', fontsize=16)
        
        timestamps = self.monitoring_history['timestamps']
        
        # RSS память
        axes[0, 0].plot(timestamps, self.monitoring_history['rss_mb'], label='RSS (Resident Set Size)', color='blue')
        axes[0, 0].set_title('Использование RSS памяти')
        axes[0, 0].set_ylabel('MB')
        axes[0, 0].grid(True)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # VMS память
        axes[0, 1].plot(timestamps, self.monitoring_history['vms_mb'], label='VMS (Virtual Memory Size)', color='red')
        axes[0, 1].set_title('Использование VMS памяти')
        axes[0, 1].set_ylabel('MB')
        axes[0, 1].grid(True)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Процент использования
        axes[1, 0].plot(timestamps, self.monitoring_history['percent'], label='Процент использования', color='green')
        axes[1, 0].set_title('Процент использования памяти')
        axes[1, 0].set_ylabel('Процент')
        axes[1, 0].grid(True)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Сборка мусора
        gc_counts = self.monitoring_history['gc_collections']
        if gc_counts:
            axes[1, 1].plot(range(len(gc_counts)), [counts[0] for counts in gc_counts], label='Generation 0', color='purple')
            axes[1, 1].plot(range(len(gc_counts)), [counts[1] for counts in gc_counts], label='Generation 1', color='orange')
            axes[1, 1].plot(range(len(gc_counts)), [counts[2] for counts in gc_counts], label='Generation 2', color='brown')
            axes[1, 1].set_title('Счетчики сборки мусора')
            axes[1, 1].set_ylabel('Объектов')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def save_memory_report(self, output_path: str = None) -> str:
        """
        Сохраняет отчет об использовании памяти
        
        Args:
            output_path: Путь для сохранения отчета
            
        Returns:
            Путь к сохраненному отчету
        """
        if output_path is None:
            output_path = str(self.output_dir / f"memory_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        analysis = self.analyze_memory_usage()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'analysis': analysis,
            'snapshots_count': len(self.snapshots),
            'leak_detections_count': len(self.leak_detections),
            'leak_detections': [
                {
                    'object_type': ld.object_type,
                    'growth_rate': ld.growth_rate,
                    'current_count': ld.current_count,
                    'baseline_count': ld.baseline_count,
                    'duration_seconds': ld.duration_seconds,
                    'severity': ld.severity
                }
                for ld in self.leak_detections
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        return output_path
    
    def get_memory_optimization_recommendations(self) -> List[str]:
        """
        Получает рекомендации по оптимизации памяти
        
        Returns:
            Список рекомендаций
        """
        recommendations = []
        
        if not self.snapshots:
            return ["Нет данных для анализа оптимизации памяти"]
        
        current_memory = self.snapshots[-1].rss_mb
        max_memory = max(s.rss_mb for s in self.snapshots)
        
        if current_memory > 1000:  # больше 1GB
            recommendations.append("Текущее использование памяти превышает 1GB. Рассмотрите оптимизацию.")
        
        if max_memory > 2000:  # больше 2GB
            recommendations.append("Максимальное использование памяти превышает 2GB. Необходима срочная оптимизация.")
        
        # Проверяем рост памяти
        if len(self.snapshots) >= 10:
            initial_memory = self.snapshots[0].rss_mb
            final_memory = self.snapshots[-1].rss_mb
            growth = ((final_memory - initial_memory) / initial_memory) * 100 if initial_memory > 0 else 0
            
            if growth > 50:  # Рост более чем на 50%
                recommendations.append(f"Обнаружен значительный рост памяти: {growth:.2f}%. Возможно наличие утечки.")
        
        # Проверяем обнаруженные утечки
        critical_leaks = [ld for ld in self.leak_detections if ld.severity in ['high', 'critical']]
        if critical_leaks:
            recommendations.append(f"Обнаружены критические утечки памяти: {len(critical_leaks)}. Требуется немедленное вмешательство.")
        
        if not recommendations:
            recommendations.append("Использование памяти в норме. Рекомендаций по оптимизации нет.")
        
        return recommendations


class MemoryDecorator:
    """
    Декоратор для отслеживания использования памяти функций
    """
    
    def __init__(self, memory_tracker: MemoryTracker):
        """
        Инициализирует декоратор
        
        Args:
            memory_tracker: Экземпляр трекера памяти
        """
        self.memory_tracker = memory_tracker
    
    def __call__(self, func: Callable) -> Callable:
        """
        Декорирует функцию для отслеживания использования памяти
        
        Args:
            func: Функция для декорирования
            
        Returns:
            Обернутая функция
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Замеряем память до выполнения
            before_memory = self.memory_tracker.get_current_memory_usage()
            
            # Выполняем функцию
            result = func(*args, **kwargs)
            
            # Замеряем память после выполнения
            after_memory = self.memory_tracker.get_current_memory_usage()
            
            # Рассчитываем разницу
            memory_diff = after_memory['rss_mb'] - before_memory['rss_mb']
            
            print(f"\n=== Отслеживание памяти для {func.__name__} ===")
            print(f"Память до: {before_memory['rss_mb']:.2f} MB")
            print(f"Память после: {after_memory['rss_mb']:.2f} MB")
            print(f"Разница: {memory_diff:+.2f} MB")
            print(f"Использование: {after_memory['percent']:.2f}%")
            
            return result
        
        return wrapper


def main():
    """Главная функция для демонстрации возможностей трекера памяти"""
    print("=== ТРЕКЕР ИСПОЛЬЗОВАНИЯ ПАМЯТИ ===")
    
    # Создаем трекер памяти
    memory_tracker = MemoryTracker()
    
    print("✓ Трекер памяти инициализирован")
    print(f"✓ Директория вывода: {memory_tracker.output_dir}")
    
    # Начинаем отслеживание
    print("\nЗапуск отслеживания памяти...")
    memory_tracker.start_tracking(interval=0.5)
    
    # Делаем несколько снимков
    print("Создание снимков памяти...")
    for i in range(5):
        snapshot = memory_tracker.take_snapshot()
        print(f"  Снимок {i+1}: {snapshot.rss_mb:.2f} MB")
        time.sleep(0.2)
    
    # Устанавливаем базовую линию
    memory_tracker.set_baseline()
    
    # Создаем декоратор
    mem_decorator = MemoryDecorator(memory_tracker)
    
    # Пример функции для тестирования
    @mem_decorator
    def example_function():
        """Пример функции для отслеживания памяти"""
        # Создаем некоторое количество объектов
        data = [i for i in range(100000)]
        return len(data)
    
    # Вызываем функцию
    print("\nВызов примерной функции с отслеживанием памяти...")
    result = example_function()
    print(f"Результат: {result}")
    
    # Вызываем сборку мусора
    print("\nВызов сборки мусора...")
    gc_result = memory_tracker.trigger_garbage_collection()
    print(f"Собрано объектов: {gc_result['collected_objects']}")
    print(f"Освобождено памяти: {gc_result['memory_freed_mb']:.2f} MB")
    
    # Начинаем трассировку выделения памяти
    print("\nЗапуск трассировки выделения памяти...")
    memory_tracker.start_trace_malloc()
    
    # Делаем что-то, что выделяет память
    large_list = [i**2 for i in range(50000)]
    
    # Получаем статистику трассировки
    trace_stats = memory_tracker.get_trace_malloc_stats(limit=5)
    print("Топ-5 источников выделения памяти:")
    for i, stat in enumerate(trace_stats, 1):
        print(f"  {i}. {stat['filename']}: {stat['size_mb']:.2f} MB ({stat['count']} allocations)")
    
    # Останавливаем трассировку
    memory_tracker.stop_trace_malloc()
    
    # Обнаруживаем утечки
    print("\nПоиск утечек памяти...")
    leaks = memory_tracker.detect_memory_leaks(threshold_growth=0.5)
    print(f"Обнаружено утечек: {len(leaks)}")
    for leak in leaks:
        print(f"  - {leak.object_type}: {leak.growth_rate:+.2f} MB/min (уровень: {leak.severity})")
    
    # Останавливаем отслеживание
    memory_tracker.stop_tracking()
    
    # Анализируем использование
    print("\nАнализ использования памяти...")
    analysis = memory_tracker.analyze_memory_usage()
    print(f"Всего снимков: {analysis['total_snapshots']}")
    print(f"Текущее использование RSS: {analysis['rss_stats']['current_mb']:.2f} MB")
    print(f"Рост памяти: {analysis['rss_stats']['growth_mb']:+.2f} MB")
    print(f"Максимальное использование: {analysis['rss_stats']['max_mb']:.2f} MB")
    
    # Получаем рекомендации
    print("\nПолучение рекомендаций по оптимизации...")
    recommendations = memory_tracker.get_memory_optimization_recommendations()
    print("Рекомендации:")
    for rec in recommendations:
        print(f"  - {rec}")
    
    # Визуализируем
    print("\nСоздание визуализации...")
    viz_path = memory_tracker.visualize_memory_usage()
    if viz_path:
        print(f"✓ Визуализация сохранена: {viz_path}")
    
    # Сохраняем отчет
    print("\nСоздание отчета...")
    report_path = memory_tracker.save_memory_report()
    print(f"✓ Отчет сохранен: {report_path}")
    
    print("\nТрекер памяти успешно протестирован")
    print("\nДоступные функции:")
    print("- Отслеживание памяти: memory_tracker.start_tracking()")
    print("- Снимки памяти: memory_tracker.take_snapshot()")
    print("- Установка базовой линии: memory_tracker.set_baseline()")
    print("- Обнаружение утечек: memory_tracker.detect_memory_leaks()")
    print("- Сборка мусора: memory_tracker.trigger_garbage_collection()")
    print("- Трассировка памяти: memory_tracker.start_trace_malloc()")
    print("- Анализ использования: memory_tracker.analyze_memory_usage()")
    print("- Визуализация: memory_tracker.visualize_memory_usage()")
    print("- Отчеты: memory_tracker.save_memory_report()")
    print("- Рекомендации: memory_tracker.get_memory_optimization_recommendations()")
    print("- Декоратор для функций: MemoryDecorator")


if __name__ == "__main__":
    main()