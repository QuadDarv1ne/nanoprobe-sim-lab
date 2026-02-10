#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль реального времени для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет интерактивную веб-панель для мониторинга
производительности и оптимизационных метрик в реальном времени.
"""

import time
import threading
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import psutil
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from dataclasses import dataclass
import pandas as pd
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
import io
import mimetypes

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


@dataclass
class RealTimeMetric:
    """Метрика реального времени"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_sent: float
    network_recv: float
    active_processes: int
    resource_efficiency: float
    optimization_score: float


class RealTimeDashboard:
    """
    Класс реал-тайм дашборда
    Обеспечивает визуализацию и мониторинг производительности в реальном времени.
    """
    
    def __init__(self, port: int = 8080):
        """
        Инициализирует реал-тайм дашборд
        
        Args:
            port: Порт для веб-сервера дашборда
        """
        self.port = port
        self.metrics_history = []
        self.max_history = 100  # Максимум 100 точек истории
        self.is_running = False
        self.dashboard_thread = None
        
        # Инициализируем все инструменты оптимизации
        self.performance_profiler = PerformanceProfiler(output_dir="profiles")
        self.resource_manager = ResourceManager()
        self.logger_analyzer = AdvancedLoggerAnalyzer()
        self.memory_tracker = MemoryTracker(output_dir="memory_logs")
        self.benchmark_suite = PerformanceBenchmarkSuite(output_dir="benchmarks")
        self.orchestrator = OptimizationOrchestrator(output_dir="optimization_reports")
        self.health_monitor = SystemHealthMonitor(output_dir="health_reports")
        self.analytics_dashboard = PerformanceAnalyticsDashboard(output_dir="analytics_reports")
        
        # Текущие метрики
        self.current_metrics = {}
    
    def collect_realtime_metrics(self) -> RealTimeMetric:
        """
        Собирает метрики в реальном времени
        
        Returns:
            Объект с текущими метриками
        """
        # Системные метрики
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk_usage = psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else 0
        
        # Статистика сети
        net_io = psutil.net_io_counters()
        network_sent = net_io.bytes_sent
        network_recv = net_io.bytes_recv
        
        # Количество активных процессов
        active_processes = len(psutil.pids())
        
        # Эффективность ресурсов из менеджера
        resource_efficiency = self.resource_manager.get_resource_efficiency_score()
        
        # Оценка оптимизации
        optimization_score = min(100, max(0, resource_efficiency + 10))  # Базовая оценка
        
        metric = RealTimeMetric(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_usage=disk_usage,
            network_sent=network_sent,
            network_recv=network_recv,
            active_processes=active_processes,
            resource_efficiency=resource_efficiency,
            optimization_score=optimization_score
        )
        
        # Сохраняем в историю
        self.metrics_history.append(metric)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)
        
        # Обновляем текущие метрики
        self.current_metrics = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'disk_usage': disk_usage,
            'active_processes': active_processes,
            'resource_efficiency': resource_efficiency,
            'optimization_score': optimization_score,
            'timestamp': metric.timestamp.isoformat()
        }
        
        return metric
    
    def get_current_status(self) -> Dict[str, Any]:
        """
        Получает текущий статус системы
        
        Returns:
            Словарь с текущими метриками
        """
        return self.current_metrics.copy()
    
    def get_metrics_history(self, last_n: int = 50) -> List[Dict[str, Any]]:
        """
        Получает историю метрик
        
        Args:
            last_n: Количество последних метрик для возврата
            
        Returns:
            Список метрик за последние N измерений
        """
        history = self.metrics_history[-last_n:] if len(self.metrics_history) >= last_n else self.metrics_history
        return [
            {
                'timestamp': m.timestamp.isoformat(),
                'cpu_percent': m.cpu_percent,
                'memory_percent': m.memory_percent,
                'disk_usage': m.disk_usage,
                'network_sent': m.network_sent,
                'network_recv': m.network_recv,
                'active_processes': m.active_processes,
                'resource_efficiency': m.resource_efficiency,
                'optimization_score': m.optimization_score
            }
            for m in history
        ]
    
    def generate_dashboard_html(self) -> str:
        """
        Генерирует HTML для дашборда
        
        Returns:
            HTML-код дашборда
        """
        # Получаем последние метрики
        current = self.get_current_status()
        history = self.get_metrics_history(last_n=30)
        
        # Создаем графики
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CPU & Memory', 'Disk Usage', 'Network Activity', 'Optimization Score'),
            specs=[[{"secondary_y": True}, {"secondary_y": True}], 
                   [{"secondary_y": True}, {"secondary_y": True}]]
        )
        
        if history:
            timestamps = [item['timestamp'] for item in history]
            cpu_data = [item['cpu_percent'] for item in history]
            memory_data = [item['memory_percent'] for item in history]
            disk_data = [item['disk_usage'] for item in history]
            opt_data = [item['optimization_score'] for item in history]
            
            # CPU и Memory
            fig.add_trace(go.Scatter(x=timestamps, y=cpu_data, name='CPU %', line=dict(color='red')), row=1, col=1)
            fig.add_trace(go.Scatter(x=timestamps, y=memory_data, name='Memory %', line=dict(color='blue')), row=1, col=1)
            
            # Disk Usage
            fig.add_trace(go.Scatter(x=timestamps, y=disk_data, name='Disk %', line=dict(color='green')), row=1, col=2)
            
            # Optimization Score
            fig.add_trace(go.Scatter(x=timestamps, y=opt_data, name='Opt Score', line=dict(color='orange')), row=2, col=2)
        
        fig.update_layout(height=600, title_text="Real-Time Performance Dashboard")
        
        # Генерируем график как HTML
        chart_html = fig.to_html(include_plotlyjs='cdn', div_id="main-chart")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Real-Time Performance Dashboard - Nanoprobe Simulation Lab</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 20px; }}
                .metric-card {{ background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
                .metric-label {{ font-size: 14px; color: #7f8c8d; }}
                .chart-container {{ background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .controls {{ margin: 20px 0; text-align: center; }}
                button {{ padding: 10px 20px; margin: 0 5px; background-color: #3498db; color: white; border: none; border-radius: 5px; cursor: pointer; }}
                button:hover {{ background-color: #2980b9; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Real-Time Performance Dashboard</h1>
                <p>Nanoprobe Simulation Lab - Optimization Monitoring System</p>
            </div>
            
            <div class="controls">
                <button onclick="refreshData()">Refresh Data</button>
                <button onclick="startAutoRefresh()">Start Auto-Refresh</button>
                <button onclick="stopAutoRefresh()">Stop Auto-Refresh</button>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{current.get('cpu_percent', 0):.1f}%</div>
                    <div class="metric-label">CPU Usage</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{current.get('memory_percent', 0):.1f}%</div>
                    <div class="metric-label">Memory Usage</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{current.get('disk_usage', 0):.1f}%</div>
                    <div class="metric-label">Disk Usage</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{current.get('resource_efficiency', 0):.1f}%</div>
                    <div class="metric-label">Resource Efficiency</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{current.get('optimization_score', 0):.1f}%</div>
                    <div class="metric-label">Optimization Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{current.get('active_processes', 0)}</div>
                    <div class="metric-label">Active Processes</div>
                </div>
            </div>
            
            <div class="chart-container">
                <div id="main-chart">{chart_html}</div>
            </div>
            
            <script>
                let autoRefreshInterval = null;
                
                function refreshData() {{
                    fetch('/api/metrics')
                        .then(response => response.json())
                        .then(data => {{
                            // Обновляем значения метрик
                            document.querySelector('.metrics-grid .metric-value:nth-child(1)').textContent = data.cpu_percent.toFixed(1) + '%';
                            document.querySelector('.metrics-grid .metric-value:nth-child(2)').textContent = data.memory_percent.toFixed(1) + '%';
                            document.querySelector('.metrics-grid .metric-value:nth-child(3)').textContent = data.disk_usage.toFixed(1) + '%';
                            document.querySelector('.metrics-grid .metric-value:nth-child(4)').textContent = data.resource_efficiency.toFixed(1) + '%';
                            document.querySelector('.metrics-grid .metric-value:nth-child(5)').textContent = data.optimization_score.toFixed(1) + '%';
                            document.querySelector('.metrics-grid .metric-value:nth-child(6)').textContent = data.active_processes;
                            
                            // Обновляем timestamp
                            console.log('Metrics updated at:', data.timestamp);
                        }})
                        .catch(error => console.error('Error fetching metrics:', error));
                }}
                
                function startAutoRefresh() {{
                    if (autoRefreshInterval) return;
                    autoRefreshInterval = setInterval(refreshData, 2000);
                }}
                
                function stopAutoRefresh() {{
                    if (autoRefreshInterval) {{
                        clearInterval(autoRefreshInterval);
                        autoRefreshInterval = null;
                    }}
                }}
                
                // Начинаем автообновление при загрузке
                startAutoRefresh();
            </script>
        </body>
        </html>
        """
        
        return html_content
    
    def start_server(self):
        """Запускает веб-сервер дашборда"""
        class DashboardHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/' or self.path == '/dashboard':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    
                    html = self.server.dashboard_instance.generate_dashboard_html()
                    self.wfile.write(html.encode())
                elif self.path == '/api/metrics':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    
                    metrics = self.server.dashboard_instance.get_current_status()
                    self.wfile.write(json.dumps(metrics).encode())
                elif self.path == '/api/history':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    
                    history = self.server.dashboard_instance.get_metrics_history()
                    self.wfile.write(json.dumps(history).encode())
                else:
                    # Попробовать обработать как статический файл
                    file_path = os.path.join(os.getcwd(), self.path.lstrip('/'))
                    if os.path.exists(file_path) and os.path.isfile(file_path):
                        mime_type, _ = mimetypes.guess_type(file_path)
                        if mime_type:
                            self.send_response(200)
                            self.send_header('Content-type', mime_type)
                            self.end_headers()
                            
                            with open(file_path, 'rb') as f:
                                self.wfile.write(f.read())
                    else:
                        self.send_response(404)
                        self.end_headers()
            
            def log_message(self, format, *args):
                # Подавляем стандартные логи сервера
                pass
        
        # Создаем сервер
        Handler = DashboardHandler
        httpd = HTTPServer(("localhost", self.port), Handler)
        httpd.dashboard_instance = self  # Передаем экземпляр дашборда
        
        print(f"🚀 Real-Time Dashboard запущен на http://localhost:{self.port}")
        print(f"📊 Открываю веб-браузер...")
        
        # Открываем в браузере
        webbrowser.open(f"http://localhost:{self.port}")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Сервер остановлен")
            httpd.shutdown()
    
    def start_monitoring(self, interval: float = 2.0):
        """
        Запускает мониторинг в фоновом режиме
        
        Args:
            interval: Интервал между сбором метрик (в секундах)
        """
        if self.is_running:
            return
        
        self.is_running = True
        
        def monitor():
            while self.is_running:
                try:
                    self.collect_realtime_metrics()
                    time.sleep(interval)
                except Exception as e:
                    print(f"Ошибка в мониторинге: {e}")
                    time.sleep(interval)
        
        self.dashboard_thread = threading.Thread(target=monitor, daemon=True)
        self.dashboard_thread.start()
    
    def stop_monitoring(self):
        """Останавливает мониторинг"""
        self.is_running = False
        if self.dashboard_thread:
            self.dashboard_thread.join(timeout=2.0)


def main():
    """Главная функция для демонстрации возможностей дашборда"""
    print("=== РЕАЛ-ТАЙМ ДАШБОРД ПРОИЗВОДИТЕЛЬНОСТИ ===")
    
    # Создаем дашборд
    dashboard = RealTimeDashboard(port=8080)
    
    # Запускаем мониторинг
    print("🔄 Запуск мониторинга...")
    dashboard.start_monitoring(interval=2.0)
    
    print("📊 Запуск веб-сервера дашборда...")
    try:
        dashboard.start_server()
    except KeyboardInterrupt:
        print("\n🛑 Остановка дашборда...")
        dashboard.stop_monitoring()


if __name__ == "__main__":
    main()