# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Модуль мониторинга производительности для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для мониторинга
производительности и эффективности симуляций.
"""

import time
import psutil
import threading
from datetime import datetime
from typing import Dict, List, Callable, Any, Optional
from pathlib import Path
import json
import statistics

class PerformanceMonitor:
    """
    Класс для мониторинга производительности
    Отслеживает использование CPU, памяти, диска и других
    ресурсов во время выполнения симуляций.
    """


    def __init__(self):
        """Инициализирует монитор производительности"""
        self.monitoring = False
        self.monitoring_thread = None
        self.metrics_history = {
            'cpu_percent': [],
            'memory_percent': [],
            'disk_io': [],
            'network_io': [],
            'timestamps': []
        }
        self.start_time = None
        self.end_time = None


    def start_monitoring(self, interval: float = 1.0):
        """
        Запускает мониторинг ресурсов

        Args:
            interval: Интервал между измерениями в секундах
        """
        if self.monitoring:
            print("Мониторинг уже запущен")
            return

        self.monitoring = True
        self.start_time = time.time()
        self.metrics_history = {
            'cpu_percent': [],
            'memory_percent': [],
            'disk_io': [],
            'network_io': [],
            'timestamps': []
        }

        self.monitoring_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        print("Мониторинг производительности запущен")


    def stop_monitoring(self):
        """Останавливает мониторинг ресурсов"""
        self.monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)  # Ожидаем завершения потока
        self.end_time = time.time()
        print("Мониторинг производительности остановлен")


    def _monitor_loop(self, interval: float):
        """
        Основной цикл мониторинга

        Args:
            interval: Интервал между измерениями
        """
        while self.monitoring:
            try:
                # Сбор метрик
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent

                # Статистика ввода-вывода
                disk_io_counters = psutil.disk_io_counters()
                network_io_counters = psutil.net_io_counters()

                disk_read_bytes = disk_io_counters.read_bytes if disk_io_counters else 0
                disk_write_bytes = disk_io_counters.write_bytes if disk_io_counters else 0
                net_sent_bytes = network_io_counters.bytes_sent if network_io_counters else 0
                net_recv_bytes = network_io_counters.bytes_recv if network_io_counters else 0

                # Сохранение метрик
                self.metrics_history['cpu_percent'].append(cpu_percent)
                self.metrics_history['memory_percent'].append(memory_percent)
                self.metrics_history['disk_io'].append({
                    'read': disk_read_bytes,
                    'write': disk_write_bytes
                })
                self.metrics_history['network_io'].append({
                    'sent': net_sent_bytes,
                    'recv': net_recv_bytes
                })
                self.metrics_history['timestamps'].append(time.time())

                time.sleep(interval)
            except Exception as e:
                print(f"Ошибка в цикле мониторинга: {e}")
                break


    def get_current_metrics(self) -> Dict[str, float]:
        """
        Получает текущие метрики производительности

        Returns:
            Словарь с текущими метриками
        """
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'process_count': len(psutil.pids()),
            'disk_usage_percent': psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else 0
        }


    def get_average_metrics(self) -> Dict[str, float]:
        """
        Получает усредненные метрики за период мониторинга

        Returns:
            Словарь с усредненными метриками
        """
        if not self.metrics_history['cpu_percent']:
            return {}

        avg_cpu = statistics.mean(self.metrics_history['cpu_percent'])
        avg_memory = statistics.mean(self.metrics_history['memory_percent'])

        # Рассчитываем среднюю скорость ввода-вывода
        if len(self.metrics_history['disk_io']) > 1:
            total_disk_read = self.metrics_history['disk_io'][-1]['read'] - self.metrics_history['disk_io'][0]['read']
            total_disk_write = self.metrics_history['disk_io'][-1]['write'] - self.metrics_history['disk_io'][0]['write']
            total_net_sent = self.metrics_history['network_io'][-1]['sent'] - self.metrics_history['network_io'][0]['sent']
            total_net_recv = self.metrics_history['network_io'][-1]['recv'] - self.metrics_history['network_io'][0]['recv']

            duration = self.metrics_history['timestamps'][-1] - self.metrics_history['timestamps'][0] if len(self.metrics_history['timestamps']) > 1 else 1

            avg_disk_read_rate = total_disk_read / duration if duration > 0 else 0
            avg_disk_write_rate = total_disk_write / duration if duration > 0 else 0
            avg_net_sent_rate = total_net_sent / duration if duration > 0 else 0
            avg_net_recv_rate = total_net_recv / duration if duration > 0 else 0
        else:
            avg_disk_read_rate = avg_disk_write_rate = avg_net_sent_rate = avg_net_recv_rate = 0

        return {
            'avg_cpu_percent': round(avg_cpu, 2),
            'avg_memory_percent': round(avg_memory, 2),
            'max_cpu_percent': max(self.metrics_history['cpu_percent']),
            'max_memory_percent': max(self.metrics_history['memory_percent']),
            'avg_disk_read_rate_bps': avg_disk_read_rate,
            'avg_disk_write_rate_bps': avg_disk_write_rate,
            'avg_network_sent_rate_bps': avg_net_sent_rate,
            'avg_network_recv_rate_bps': avg_net_recv_rate,
            'monitoring_duration_sec': round(duration, 2) if 'duration' in locals() else 0
        }


    def measure_function_performance(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Измеряет производительность выполнения функции

        Args:
            func: Функция для измерения
            *args: Аргументы функции
            **kwargs: Ключевые аргументы функции

        Returns:
            Словарь с метриками производительности
        """
        # Запускаем мониторинг
        self.start_monitoring(interval=0.5)

        start_time = time.perf_counter()
        start_resources = self.get_current_metrics()

        # Выполняем функцию
        result = func(*args, **kwargs)

        end_resources = self.get_current_metrics()
        end_time = time.perf_counter()

        # Останавливаем мониторинг
        self.stop_monitoring()

        # Собираем результаты
        performance_metrics = {
            'execution_time_sec': round(end_time - start_time, 4),
            'start_resources': start_resources,
            'end_resources': end_resources,
            'average_metrics': self.get_average_metrics(),
            'function_result': result
        }

        return performance_metrics


    def visualize_performance(self, output_path: Optional[str] = None):
        """
        Визуализирует метрики производительности

        Args:
            output_path: Путь для сохранения графика (опционально)
        """
        import matplotlib.pyplot as plt
        
        if not self.metrics_history['timestamps']:
            print("Нет данных для визуализации")
            return

        timestamps = [t - self.metrics_history['timestamps'][0] for t in self.metrics_history['timestamps']]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Метрики производительности', fontsize=16)

        # CPU загрузка
        axes[0, 0].plot(timestamps, self.metrics_history['cpu_percent'], 'b-', linewidth=1)
        axes[0, 0].set_title('Загрузка CPU (%)')
        axes[0, 0].set_xlabel('Время (сек)')
        axes[0, 0].set_ylabel('Загрузка CPU (%)')
        axes[0, 0].grid(True)

        # Загрузка памяти
        axes[0, 1].plot(timestamps, self.metrics_history['memory_percent'], 'r-', linewidth=1)
        axes[0, 1].set_title('Загрузка памяти (%)')
        axes[0, 1].set_xlabel('Время (сек)')
        axes[0, 1].set_ylabel('Загрузка памяти (%)')
        axes[0, 1].grid(True)

        # Диск I/O
        if self.metrics_history['disk_io']:
            disk_reads = [io['read'] for io in self.metrics_history['disk_io']]
            disk_writes = [io['write'] for io in self.metrics_history['disk_io']]
            axes[1, 0].plot(timestamps, disk_reads, 'g-', label='Чтение', linewidth=1)
            axes[1, 0].plot(timestamps, disk_writes, 'orange', label='Запись', linewidth=1)
            axes[1, 0].set_title('Диск I/O (байты)')
            axes[1, 0].set_xlabel('Время (сек)')
            axes[1, 0].set_ylabel('Байты')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

        # Сеть I/O
        if self.metrics_history['network_io']:
            net_sent = [io['sent'] for io in self.metrics_history['network_io']]
            net_recv = [io['recv'] for io in self.metrics_history['network_io']]
            axes[1, 1].plot(timestamps, net_sent, 'purple', label='Отправлено', linewidth=1)
            axes[1, 1].plot(timestamps, net_recv, 'cyan', label='Получено', linewidth=1)
            axes[1, 1].set_title('Сеть I/O (байты)')
            axes[1, 1].set_xlabel('Время (сек)')
            axes[1, 1].set_ylabel('Байты')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"График производительности сохранен: {output_path}")

        plt.show()


    def save_performance_report(self, output_path: str):
        """
        Сохраняет отчет о производительности

        Args:
            output_path: Путь для сохранения отчета
        """
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'start_time': self.start_time,
            'end_time': self.end_time,
            'metrics_history': self.metrics_history,
            'average_metrics': self.get_average_metrics()
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)

        print(f"Отчет о производительности сохранен: {output_path}")

class SimulationProfiler:
    """
    Класс для профилирования симуляций
    Обеспечивает детальное профилирование производительности
    различных этапов симуляции.
    """


    def __init__(self):
        """Инициализирует профилировщик симуляций"""
        self.performance_monitor = PerformanceMonitor()
        self.profile_results = {}


    def profile_simulation_stage(self, stage_name: str, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Профилирует отдельный этап симуляции

        Args:
            stage_name: Название этапа
            func: Функция этапа
            *args: Аргументы функции
            **kwargs: Ключевые аргументы

        Returns:
            Словарь с результатами профилирования
        """
        print(f"Начало профилирования этапа: {stage_name}")

        # Измеряем производительность
        perf_metrics = self.performance_monitor.measure_function_performance(func, *args, **kwargs)

        # Сохраняем результаты
        self.profile_results[stage_name] = perf_metrics

        print(f"Завершено профилирование этапа: {stage_name}")
        return perf_metrics


    def profile_full_simulation(self, simulation_func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Профилирует полную симуляцию

        Args:
            simulation_func: Функция симуляции
            *args: Аргументы функции
            **kwargs: Ключевые аргументы

        Returns:
            Словарь с результатами профилирования
        """
        print("Начало профилирования полной симуляции")

        # Измеряем производительность всей симуляции
        perf_metrics = self.performance_monitor.measure_function_performance(simulation_func, *args, **kwargs)

        # Сохраняем как общий результат
        self.profile_results['full_simulation'] = perf_metrics

        print("Завершено профилирование полной симуляции")
        return perf_metrics


    def get_optimization_recommendations(self) -> List[str]:
        """
        Получает рекомендации по оптимизации

        Returns:
            Список рекомендаций
        """
        recommendations = []

        if 'full_simulation' in self.profile_results:
            avg_metrics = self.profile_results['full_simulation'].get('average_metrics', {})

            # Проверяем высокую загрузку CPU
            avg_cpu = avg_metrics.get('avg_cpu_percent', 0)
            if avg_cpu > 80:
                recommendations.append("Высокая загрузка CPU (>80%) - рассмотрите оптимизацию алгоритмов")

            # Проверяем высокое использование памяти
            avg_memory = avg_metrics.get('avg_memory_percent', 0)
            if avg_memory > 80:
                recommendations.append("Высокое использование памяти (>80%) - рассмотрите оптимизацию использования памяти")

            # Проверяем длительное время выполнения
            exec_time = self.profile_results['full_simulation'].get('execution_time_sec', 0)
            if exec_time > 60:  # Больше 1 минуты
                recommendations.append("Длительное время выполнения (>60 сек) - рассмотрите параллелизацию или оптимизацию")

        return recommendations


    def generate_performance_report(self, output_path: str = "performance_report.json"):
        """
        Генерирует полный отчет о производительности

        Args:
            output_path: Путь для сохранения отчета
        """
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'profile_results': self.profile_results,
            'recommendations': self.get_optimization_recommendations()
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)

        print(f"Полный отчет о производительности сохранен: {output_path}")

        # Также создаем краткий текстовый отчет
        txt_report_path = output_path.replace('.json', '.txt')
        with open(txt_report_path, 'w', encoding='utf-8') as f:
            f.write("ОТЧЕТ О ПРОИЗВОДИТЕЛЬНОСТИ СИМУЛЯЦИИ\n")
            f.write("=" * 50 + "\n\n")

            if 'full_simulation' in self.profile_results:
                perf = self.profile_results['full_simulation']
                f.write(f"Время выполнения: {perf['execution_time_sec']} сек\n")
                avg_metrics = perf.get('average_metrics', {})
                if 'avg_cpu_percent' in avg_metrics:
                    f.write(f"Средняя загрузка CPU: {avg_metrics['avg_cpu_percent']}%\n")
                if 'avg_memory_percent' in avg_metrics:
                    f.write(f"Средняя загрузка памяти: {avg_metrics['avg_memory_percent']}%\n")

            f.write(f"\nРекомендации по оптимизации:\n")
            recommendations = self.get_optimization_recommendations()
            for rec in recommendations:
                f.write(f"- {rec}\n")

        print(f"Текстовый отчет о производительности сохранен: {txt_report_path}")

def main():
    """Главная функция для демонстрации возможностей монитора производительности"""
    print("=== МОНИТОР ПРОИЗВОДИТЕЛЬНОСТИ ПРОЕКТА ===")

    # Создаем монитор производительности
    monitor = PerformanceMonitor()

    # Тестируем функцию измерения производительности

    def test_function():
        """Тестовая функция для измерения производительности"""
        # Симуляция нагрузки
        total = 0
        for i in range(1000000):
            total += i * 0.001
        return total

    print("Тестирование измерения производительности функции...")
    perf_result = monitor.measure_function_performance(test_function)

    print(f"Время выполнения: {perf_result['execution_time_sec']} сек")
    print(f"Средняя загрузка CPU: {perf_result['average_metrics'].get('avg_cpu_percent', 'N/A')}%")
    print(f"Средняя загрузка памяти: {perf_result['average_metrics'].get('avg_memory_percent', 'N/A')}%")

    # Создаем профилировщик
    profiler = SimulationProfiler()

    # Профилируем тестовую функцию
    profiler.profile_simulation_stage("тестовый_этап", test_function)

    # Получаем рекомендации
    recommendations = profiler.get_optimization_recommendations()
    print(f"Рекомендации: {recommendations}")

    # Сохраняем отчет
    profiler.generate_performance_report()

    print("Монитор производительности успешно протестирован")

if __name__ == "__main__":
    main()

