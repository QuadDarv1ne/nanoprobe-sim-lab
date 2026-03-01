# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Модуль мониторинга системы для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для мониторинга
состояния системы и производительности проекта.
"""

import psutil
import time
import threading
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import deque

class SystemMonitor:
    """
    Класс мониторинга системы
    Обеспечивает мониторинг состояния системы,
    ресурсов и производительности проекта.
    """


    def __init__(self, update_interval: float = 1.0):
        """
        Инициализирует монитор системы

        Args:
            update_interval: Интервал обновления в секундах
        """
        self.update_interval = update_interval
        self.monitoring = False
        self.monitoring_thread = None

        # История метрик
        self.history = {
            'cpu_percent': deque(maxlen=300),  # Последние 5 минут при 1с интервале
            'memory_percent': deque(maxlen=300),
            'disk_usage': deque(maxlen=300),
            'network_sent': deque(maxlen=300),
            'network_recv': deque(maxlen=300),
            'timestamps': deque(maxlen=300)
        }

        # Текущие метрики
        self.current_metrics = {}
        self.start_time = None


    def start_monitoring(self):
        """Запускает мониторинг системы"""
        if self.monitoring:
            return

        self.monitoring = True
        self.start_time = datetime.now()
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_thread.start()


    def stop_monitoring(self):
        """Останавливает мониторинг системы"""
        self.monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)


    def _monitor_loop(self):
        """Основной цикл мониторинга"""
        while self.monitoring:
            try:
                # Сбор метрик
                timestamp = datetime.now()

                cpu_percent = psutil.cpu_percent(interval=None)
                memory_percent = psutil.virtual_memory().percent
                disk_usage = psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else 0

                # Статистика сети
                net_io = psutil.net_io_counters()
                network_sent = net_io.bytes_sent
                network_recv = net_io.bytes_recv

                # Сохранение метрик
                self.history['cpu_percent'].append(cpu_percent)
                self.history['memory_percent'].append(memory_percent)
                self.history['disk_usage'].append(disk_usage)
                self.history['network_sent'].append(network_sent)
                self.history['network_recv'].append(network_recv)
                self.history['timestamps'].append(timestamp)

                # Обновление текущих метрик
                self.current_metrics = {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
                    'disk_usage_percent': disk_usage,
                    'process_count': len(psutil.pids()),
                    'uptime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
                }

                time.sleep(self.update_interval)

            except Exception as e:
                print(f"Ошибка в цикле мониторинга: {e}")
                time.sleep(self.update_interval)


    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Получает текущие метрики системы

        Returns:
            Словарь с текущими метриками
        """
        return self.current_metrics.copy()


    def get_system_health(self) -> Dict[str, Any]:
        """
        Оценивает здоровье системы

        Returns:
            Словарь с оценкой здоровья системы
        """
        metrics = self.get_current_metrics()

        health_score = 100
        issues = []

        # Проверяем загрузку CPU
        if metrics.get('cpu_percent', 0) > 80:
            health_score -= 20
            issues.append("Высокая загрузка CPU")
        elif metrics.get('cpu_percent', 0) > 60:
            health_score -= 10

        # Проверяем использование памяти
        if metrics.get('memory_percent', 0) > 80:
            health_score -= 20
            issues.append("Высокое использование памяти")
        elif metrics.get('memory_percent', 0) > 60:
            health_score -= 10

        # Проверяем использование диска
        if metrics.get('disk_usage_percent', 0) > 80:
            health_score -= 20
            issues.append("Высокое использование диска")
        elif metrics.get('disk_usage_percent', 0) > 60:
            health_score -= 10

        # Определяем уровень здоровья
        if health_score >= 80:
            health_level = "Отлично"
        elif health_score >= 60:
            health_level = "Хорошо"
        elif health_score >= 40:
            health_level = "Удовлетворительно"
        else:
            health_level = "Плохо"

        return {
            'health_score': max(0, health_score),
            'health_level': health_level,
            'issues': issues,
            'timestamp': datetime.now().isoformat()
        }


    def get_resource_usage_trend(self) -> Dict[str, List[float]]:
        """
        Получает тренды использования ресурсов

        Returns:
            Словарь с трендами использования ресурсов
        """
        return {
            'cpu_percent': list(self.history['cpu_percent']),
            'memory_percent': list(self.history['memory_percent']),
            'disk_usage': list(self.history['disk_usage']),
            'timestamps': list(self.history['timestamps'])
        }


    def generate_report(self, output_path: str = None) -> str:
        """
        Генерирует отчет о состоянии системы

        Args:
            output_path: Путь для сохранения отчета (если None, генерируется автоматически)

        Returns:
            Путь к созданному отчету
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"system_monitor_report_{timestamp}.json"

        report = {
            'timestamp': datetime.now().isoformat(),
            'report_type': 'system_monitor',
            'current_metrics': self.get_current_metrics(),
            'system_health': self.get_system_health(),
            'resource_trends': self.get_resource_usage_trend(),
            'monitoring_duration': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        return output_path

class MonitoringDashboard:
    """
    Класс интерактивной панели мониторинга
    Предоставляет графический интерфейс для мониторинга
    состояния системы и производительности.
    """


    def __init__(self, system_monitor: SystemMonitor):
        """
        Инициализирует панель мониторинга

        Args:
            system_monitor: Экземпляр SystemMonitor
        """
        self.system_monitor = system_monitor
        self.root = None
        self.fig = None
        self.ani = None
        self.is_running = False

        # Метрики для отображения
        self.cpu_label = None
        self.memory_label = None
        self.disk_label = None
        self.health_label = None
        self.status_label = None

        # Графики
        self.ax_cpu = None
        self.ax_memory = None
        self.ax_disk = None


    def create_gui(self):
        """Создает графический интерфейс панели мониторинга"""
        import tkinter as tk
        from tkinter import ttk, scrolledtext
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.animation import FuncAnimation
        import matplotlib.pyplot as plt
        
        self.root = tk.Tk()
        self.root.title("Панель мониторинга системы - Лаборатория моделирования нанозонда")
        self.root.geometry("1200x800")

        # Основной фрейм
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Фрейм для метрик
        metrics_frame = ttk.LabelFrame(main_frame, text="Текущие метрики", padding=10)
        metrics_frame.pack(fill=tk.X, padx=5, pady=5)

        # Метки для метрик
        ttk.Label(metrics_frame, text="CPU:").grid(row=0, column=0, sticky=tk.W)
        self.cpu_label = ttk.Label(metrics_frame, text="0%")
        self.cpu_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))

        ttk.Label(metrics_frame, text="Память:").grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        self.memory_label = ttk.Label(metrics_frame, text="0%")
        self.memory_label.grid(row=0, column=3, sticky=tk.W, padx=(10, 0))

        ttk.Label(metrics_frame, text="Диск:").grid(row=0, column=4, sticky=tk.W, padx=(20, 0))
        self.disk_label = ttk.Label(metrics_frame, text="0%")
        self.disk_label.grid(row=0, column=5, sticky=tk.W, padx=(10, 0))

        ttk.Label(metrics_frame, text="Здоровье системы:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.health_label = ttk.Label(metrics_frame, text="Неизвестно")
        self.health_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))

        ttk.Label(metrics_frame, text="Статус:").grid(row=1, column=2, sticky=tk.W, padx=(20, 0), pady=(10, 0))
        self.status_label = ttk.Label(metrics_frame, text="Остановлен")
        self.status_label.grid(row=1, column=3, sticky=tk.W, padx=(10, 0), pady=(10, 0))

        # Фрейм для графиков
        plots_frame = ttk.LabelFrame(main_frame, text="Графики ресурсов", padding=10)
        plots_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Создаем фигуру matplotlib
        self.fig, ((self.ax_cpu, self.ax_memory), (self.ax_disk, self.ax_network)) = plt.subplots(2, 2, figsize=(12, 8))

        # Настройка графиков
        self.ax_cpu.set_title("Загрузка CPU (%)")
        self.ax_memory.set_title("Использование памяти (%)")
        self.ax_disk.set_title("Использование диска (%)")
        self.ax_network.set_title("Сеть (байт/сек)")

        # Настройка осей
        for ax in [self.ax_cpu, self.ax_memory, self.ax_disk, self.ax_network]:
            ax.set_xlabel("Время")
            ax.grid(True)

        # Встраиваем matplotlib в tkinter
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(self.fig, plots_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Фрейм для кнопок управления
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Кнопки управления
        self.start_button = ttk.Button(control_frame, text="Запустить", command=self.start_monitoring)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(control_frame, text="Остановить", command=self.stop_monitoring)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.stop_button.config(state=tk.DISABLED)

        self.refresh_button = ttk.Button(control_frame, text="Обновить", command=self.update_display)
        self.refresh_button.pack(side=tk.LEFT, padx=5)

        self.report_button = ttk.Button(control_frame, text="Создать отчет", command=self.generate_report)
        self.report_button.pack(side=tk.LEFT, padx=5)


    def start_monitoring(self):
        """Запускает мониторинг"""
        if not self.is_running:
            self.system_monitor.start_monitoring()
            self.is_running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="Работает")

            # Запускаем анимацию графиков
            self.ani = FuncAnimation(self.fig, self.animate_plots, interval=1000, blit=False)


    def stop_monitoring(self):
        """Останавливает мониторинг"""
        if self.is_running:
            self.system_monitor.stop_monitoring()
            self.is_running = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_label.config(text="Остановлен")

            # Останавливаем анимацию
            if self.ani:
                self.ani.event_source.stop()


    def animate_plots(self, frame):
        """Анимирует графики"""
        if not self.is_running:
            return

        # Получаем текущие данные
        trends = self.system_monitor.get_resource_usage_trend()

        # Очищаем графики
        self.ax_cpu.clear()
        self.ax_memory.clear()
        self.ax_disk.clear()
        self.ax_network.clear()

        # Рисуем графики
        if trends['timestamps']:
            times = trends['timestamps']
            cpu_data = trends['cpu_percent']
            mem_data = trends['memory_percent']
            disk_data = trends['disk_usage']

            self.ax_cpu.plot(times, cpu_data, 'b-', linewidth=1)
            self.ax_cpu.set_title("Загрузка CPU (%)")
            self.ax_cpu.grid(True)

            self.ax_memory.plot(times, mem_data, 'r-', linewidth=1)
            self.ax_memory.set_title("Использование памяти (%)")
            self.ax_memory.grid(True)

            self.ax_disk.plot(times, disk_data, 'g-', linewidth=1)
            self.ax_disk.set_title("Использование диска (%)")
            self.ax_disk.grid(True)

            # Для графика сети вычисляем производную (скорость передачи)
            if len(trends['network_sent']) > 1:
                network_times = times[1:] if len(times) > len(trends['network_sent']) else times
                sent_rates = [trends['network_sent'][i+1] - trends['network_sent'][i]
                             for i in range(len(trends['network_sent'])-1)]
                recv_rates = [trends['network_recv'][i+1] - trends['network_recv'][i]
                             for i in range(len(trends['network_recv'])-1)]

                self.ax_network.plot(network_times[:-1], sent_rates, 'c-', label='Отправлено', linewidth=1)
                self.ax_network.plot(network_times[:-1], recv_rates, 'm-', label='Получено', linewidth=1)
                self.ax_network.set_title("Скорость передачи данных (байт/сек)")
                self.ax_network.legend()
                self.ax_network.grid(True)

        # Обновляем отображение метрик
        self.update_display()


    def update_display(self):
        """Обновляет отображение метрик"""
        metrics = self.system_monitor.get_current_metrics()
        health = self.system_monitor.get_system_health()

        # Обновляем метки
        if self.cpu_label:
            cpu_percent = metrics.get('cpu_percent', 0)
            self.cpu_label.config(text=f"{cpu_percent}%")

        if self.memory_label:
            memory_percent = metrics.get('memory_percent', 0)
            memory_gb = metrics.get('memory_available_gb', 0)
            self.memory_label.config(text=f"{memory_percent}% ({memory_gb} GB свободно)")

        if self.disk_label:
            disk_usage = metrics.get('disk_usage_percent', 0)
            self.disk_label.config(text=f"{disk_usage}%")

        if self.health_label:
            health_level = health.get('health_level', 'Неизвестно')
            health_score = health.get('health_score', 0)
            self.health_label.config(text=f"{health_level} ({health_score}/100)")


    def generate_report(self):
        """Генерирует отчет о мониторинге"""
        report_path = self.system_monitor.generate_report()
        # Показываем сообщение о создании отчета
        tk.messagebox.showinfo("Отчет создан", f"Отчет сохранен: {report_path}")


    def run(self):
        """Запускает панель мониторинга"""
        self.create_gui()
        self.root.mainloop()

class HealthCheckManager:
    """
    Класс менеджера проверки здоровья
    Обеспечивает комплексную проверку состояния
    системы и компонентов проекта.
    """


    def __init__(self, system_monitor: SystemMonitor):
        """
        Инициализирует менеджер проверки здоровья

        Args:
            system_monitor: Экземпляр SystemMonitor
        """
        self.system_monitor = system_monitor
        self.check_results = {}


    def run_comprehensive_health_check(self) -> Dict[str, Any]:
        """
        Запускает комплексную проверку здоровья системы

        Returns:
            Словарь с результатами проверки
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'overall_health': 'unknown',
            'issues_found': 0,
            'recommendations': []
        }

        # Проверка системы
        results['checks']['system'] = self._check_system_health()

        # Проверка ресурсов
        results['checks']['resources'] = self._check_resource_usage()

        # Проверка проектных директорий
        results['checks']['project_directories'] = self._check_project_directories()

        # Проверка конфигурации
        results['checks']['configuration'] = self._check_project_configuration()

        # Подсчет проблем
        for check_name, check_result in results['checks'].items():
            if not check_result['healthy']:
                results['issues_found'] += 1

        # Определение общего состояния
        healthy_checks = sum(1 for check in results['checks'].values() if check['healthy'])
        total_checks = len(results['checks'])

        if healthy_checks == total_checks:
            results['overall_health'] = 'excellent'
        elif healthy_checks >= total_checks * 0.8:
            results['overall_health'] = 'good'
        elif healthy_checks >= total_checks * 0.6:
            results['overall_health'] = 'fair'
        else:
            results['overall_health'] = 'poor'

        # Генерация рекомендаций
        results['recommendations'] = self._generate_recommendations(results)

        return results


    def _check_system_health(self) -> Dict[str, Any]:
        """Проверяет здоровье системы"""
        system_health = self.system_monitor.get_system_health()

        return {
            'healthy': system_health['health_score'] >= 60,
            'score': system_health['health_score'],
            'level': system_health['health_level'],
            'issues': system_health['issues'],
            'details': system_health
        }


    def _check_resource_usage(self) -> Dict[str, Any]:
        """Проверяет использование ресурсов"""
        metrics = self.system_monitor.get_current_metrics()

        issues = []
        if metrics.get('cpu_percent', 0) > 80:
            issues.append(f"Высокая загрузка CPU: {metrics.get('cpu_percent', 0)}%")
        if metrics.get('memory_percent', 0) > 80:
            issues.append(f"Высокое использование памяти: {metrics.get('memory_percent', 0)}%")
        if metrics.get('disk_usage_percent', 0) > 80:
            issues.append(f"Высокое использование диска: {metrics.get('disk_usage_percent', 0)}%")

        return {
            'healthy': len(issues) == 0,
            'cpu_percent': metrics.get('cpu_percent', 0),
            'memory_percent': metrics.get('memory_percent', 0),
            'disk_usage_percent': metrics.get('disk_usage_percent', 0),
            'issues': issues
        }


    def _check_project_directories(self) -> Dict[str, Any]:
        """Проверяет директории проекта"""
        project_dirs = [
            'cpp-spm-hardware-sim',
            'py-surface-image-analyzer',
            'py-sstv-groundstation',
            'utils',
            'api',
            'security',
            'tests',
            'docs'
        ]

        issues = []
        for dir_name in project_dirs:
            if not Path(dir_name).exists():
                issues.append(f"Отсутствует директория: {dir_name}")

        return {
            'healthy': len(issues) == 0,
            'directories_checked': len(project_dirs),
            'issues': issues
        }


    def _check_project_configuration(self) -> Dict[str, Any]:
        """Проверяет конфигурацию проекта"""
        config_files = ['config.json', 'requirements.txt', 'CMakeLists.txt']

        issues = []
        for config_file in config_files:
            if not Path(config_file).exists():
                issues.append(f"Отсутствует конфигурационный файл: {config_file}")

        return {
            'healthy': len(issues) == 0,
            'files_checked': len(config_files),
            'issues': issues
        }


    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Генерирует рекомендации на основе результатов проверки"""
        recommendations = []

        if results['issues_found'] > 0:
            if results['checks']['resources']['cpu_percent'] > 80:
                recommendations.append("Рассмотрите оптимизацию кода для снижения загрузки CPU")
            if results['checks']['resources']['memory_percent'] > 80:
                recommendations.append("Рассмотрите оптимизацию использования памяти")
            if results['checks']['resources']['disk_usage_percent'] > 80:
                recommendations.append("Рассмотрите очистку дискового пространства")
            if not results['checks']['project_directories']['healthy']:
                recommendations.append("Проверьте целостность файлов проекта")
            if not results['checks']['configuration']['healthy']:
                recommendations.append("Проверьте наличие конфигурационных файлов")

        return recommendations


    def generate_health_report(self, output_path: str = None) -> str:
        """
        Генерирует отчет о здоровье системы

        Args:
            output_path: Путь для сохранения отчета

        Returns:
            Путь к созданному отчету
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"system_health_report_{timestamp}.json"

        health_check_results = self.run_comprehensive_health_check()

        report = {
            'timestamp': datetime.now().isoformat(),
            'report_type': 'system_health',
            'health_check_results': health_check_results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        return output_path

def main():
    """Главная функция для демонстрации возможностей мониторинга системы"""
    print("=== МОНИТОРИНГ СИСТЕМЫ ПРОЕКТА ===")

    # Создаем монитор системы
    system_monitor = SystemMonitor(update_interval=1.0)

    print("✓ Монитор системы инициализирован")
    print("✓ Интервал обновления: 1 секунда")

    # Получаем текущие метрики
    system_monitor.start_monitoring()
    time.sleep(2)  # Ждем немного для сбора данных

    current_metrics = system_monitor.get_current_metrics()
    print(f"✓ Текущие метрики: {current_metrics}")

    system_health = system_monitor.get_system_health()
    print(f"✓ Здоровье системы: {system_health['health_level']} ({system_health['health_score']}/100)")

    # Создаем менеджер проверки здоровья
    health_manager = HealthCheckManager(system_monitor)
    health_results = health_manager.run_comprehensive_health_check()
    print(f"✓ Результаты проверки здоровья: {health_results['overall_health']}")
    print(f"✓ Найдено проблем: {health_results['issues_found']}")

    # Останавливаем мониторинг
    system_monitor.stop_monitoring()

    # Генерируем отчет
    report_path = system_monitor.generate_report()
    print(f"✓ Отчет о мониторинге создан: {report_path}")

    health_report_path = health_manager.generate_health_report()
    print(f"✓ Отчет о здоровье системы создан: {health_report_path}")

    print("Мониторинг системы успешно протестирован")

if __name__ == "__main__":
    main()

