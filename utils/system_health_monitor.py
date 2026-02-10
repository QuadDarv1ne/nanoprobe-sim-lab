#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль мониторинга здоровья системы для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для мониторинга здоровья системы,
обнаружения аномалий и отправки уведомлений о проблемах.
"""

import time
import threading
import queue
import smtplib
import sys
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import json
import psutil
import logging
from dataclasses import dataclass
import requests
import socket
import os


@dataclass
class HealthMetric:
    """Метрика здоровья системы"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    severity: str  # info, warning, error, critical
    source: str
    threshold_low: Optional[float] = None
    threshold_high: Optional[float] = None


@dataclass
class HealthAlert:
    """Оповещение о здоровье системы"""
    alert_id: str
    metric_name: str
    current_value: float
    threshold_value: float
    severity: str
    message: str
    timestamp: datetime
    resolved: bool = False


class SystemHealthMonitor:
    """
    Класс мониторинга здоровья системы
    Обеспечивает постоянный мониторинг состояния системы,
    обнаружение аномалий и отправку уведомлений.
    """
    
    def __init__(self, output_dir: str = "health_reports"):
        """
        Инициализирует монитор здоровья системы
        
        Args:
            output_dir: Директория для сохранения отчетов о здоровье
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.metrics = []
        self.alerts = []
        self.active = False
        self.monitoring_thread = None
        self.alert_handlers = []  # Функции для обработки оповещений
        self.notification_channels = []  # Каналы уведомлений
        
        # Пороговые значения по умолчанию
        self.thresholds = {
            'cpu_percent': {'warning': 70, 'error': 85, 'critical': 95},
            'memory_percent': {'warning': 75, 'error': 85, 'critical': 95},
            'disk_percent': {'warning': 80, 'error': 90, 'critical': 95},
            'temperature': {'warning': 70, 'error': 80, 'critical': 90},  # Celsius
            'process_count': {'warning': 200, 'error': 500, 'critical': 1000},
            'network_latency_ms': {'warning': 100, 'error': 500, 'critical': 1000}
        }
        
        # Текущие значения метрик
        self.current_metrics = {}
        self.health_score = 100.0
        
        # Очередь для обработки оповещений
        self.alert_queue = queue.Queue()
        
    def add_alert_handler(self, handler: Callable[[HealthAlert], None]):
        """
        Добавляет обработчик оповещений
        
        Args:
            handler: Функция для обработки оповещений
        """
        self.alert_handlers.append(handler)
    
    def add_notification_channel(self, channel: str, config: Dict[str, Any]):
        """
        Добавляет канал уведомлений
        
        Args:
            channel: Тип канала ('email', 'webhook', 'console')
            config: Конфигурация канала
        """
        self.notification_channels.append({
            'type': channel,
            'config': config
        })
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Получает текущие системные метрики"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            temperature = 0  # Будет получено ниже, если возможно
            
            # Попытка получить температуру
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    # Берем температуру CPU, если доступна
                    if 'coretemp' in temps:
                        temp_sensors = temps['coretemp']
                        if temp_sensors:
                            temperature = temp_sensors[0].current
                    elif 'cpu_thermal' in temps:
                        temp_sensors = temps['cpu_thermal']
                        if temp_sensors:
                            temperature = temp_sensors[0].current
                    else:
                        # Берем первую доступную температуру
                        for sensor_name, sensor_list in temps.items():
                            if sensor_list:
                                temperature = sensor_list[0].current
                                break
            except AttributeError:
                # Сенсоры температуры могут не быть доступны на некоторых системах
                pass
            
            # Получаем информацию о процессах
            process_count = len(psutil.pids())
            
            # Получаем сетевые метрики
            network_io = psutil.net_io_counters()
            network_sent = network_io.bytes_sent if network_io else 0
            network_recv = network_io.bytes_recv if network_io else 0
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent if disk else 0,
                'disk_used_gb': disk.used / (1024**3) if disk else 0,
                'disk_free_gb': disk.free / (1024**3) if disk else 0,
                'temperature_celsius': temperature,
                'process_count': process_count,
                'network_sent_bytes': network_sent,
                'network_recv_bytes': network_recv,
                'timestamp': datetime.now()
            }
        except Exception as e:
            print(f"Ошибка получения метрик системы: {e}")
            return {}
    
    def evaluate_metric_severity(self, metric_name: str, value: float) -> str:
        """
        Оценивает уровень серьезности метрики
        
        Args:
            metric_name: Название метрики
            value: Значение метрики
            
        Returns:
            Уровень серьезности ('info', 'warning', 'error', 'critical')
        """
        if metric_name not in self.thresholds:
            return 'info'
        
        thresholds = self.thresholds[metric_name]
        
        if value <= thresholds['warning']:
            return 'info'
        elif value <= thresholds['error']:
            return 'warning'
        elif value <= thresholds['critical']:
            return 'error'
        else:
            return 'critical'
    
    def check_for_alerts(self, metrics: Dict[str, float]) -> List[HealthAlert]:
        """
        Проверяет метрики на наличие оповещений
        
        Args:
            metrics: Словарь с текущими метриками
            
        Returns:
            Список оповещений
        """
        alerts = []
        
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)) and metric_name in self.thresholds:
                severity = self.evaluate_metric_severity(metric_name, value)
                
                if severity in ['warning', 'error', 'critical']:
                    # Определяем пороговое значение для данного уровня серьезности
                    thresholds = self.thresholds[metric_name]
                    if severity == 'warning':
                        threshold_val = thresholds['warning']
                    elif severity == 'error':
                        threshold_val = thresholds['error']
                    else:  # critical
                        threshold_val = thresholds['critical']
                    
                    alert = HealthAlert(
                        alert_id=f"{metric_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        metric_name=metric_name,
                        current_value=value,
                        threshold_value=threshold_val,
                        severity=severity,
                        message=f"Метрика {metric_name} превысила порог: {value} > {threshold_val}",
                        timestamp=datetime.now()
                    )
                    
                    alerts.append(alert)
                    
                    # Добавляем в очередь оповещений
                    self.alert_queue.put(alert)
        
        return alerts
    
    def process_alerts(self):
        """Обрабатывает оповещения из очереди"""
        while not self.alert_queue.empty():
            try:
                alert = self.alert_queue.get_nowait()
                
                # Добавляем в историю оповещений
                self.alerts.append(alert)
                
                # Вызываем обработчики
                for handler in self.alert_handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        print(f"Ошибка в обработчике оповещений: {e}")
                
                # Отправляем уведомления
                self.send_notifications(alert)
                
            except queue.Empty:
                break
    
    def send_notifications(self, alert: HealthAlert):
        """Отправляет уведомления об оповещении"""
        for channel in self.notification_channels:
            try:
                if channel['type'] == 'email':
                    self._send_email_notification(alert, channel['config'])
                elif channel['type'] == 'webhook':
                    self._send_webhook_notification(alert, channel['config'])
                elif channel['type'] == 'console':
                    self._send_console_notification(alert)
            except Exception as e:
                print(f"Ошибка отправки уведомления через {channel['type']}: {e}")
    
    def _send_email_notification(self, alert: HealthAlert, config: Dict[str, Any]):
        """Отправляет уведомление по email"""
        try:
            msg = MIMEMultipart()
            msg['From'] = config.get('from_email', '')
            msg['To'] = ', '.join(config.get('to_emails', []))
            msg['Subject'] = f"[{alert.severity.upper()}] Системное оповещение: {alert.metric_name}"
            
            body = f"""
Системное оповещение:
- Метрика: {alert.metric_name}
- Текущее значение: {alert.current_value}
- Пороговое значение: {alert.threshold_value}
- Серьезность: {alert.severity}
- Время: {alert.timestamp}
- Сообщение: {alert.message}
"""
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            server = smtplib.SMTP(config.get('smtp_server', 'localhost'), config.get('smtp_port', 587))
            server.starttls()
            server.login(config.get('username', ''), config.get('password', ''))
            text = msg.as_string()
            server.sendmail(msg['From'], config.get('to_emails', []), text)
            server.quit()
            
        except Exception as e:
            print(f"Ошибка отправки email уведомления: {e}")
    
    def _send_webhook_notification(self, alert: HealthAlert, config: Dict[str, Any]):
        """Отправляет уведомление через webhook"""
        try:
            payload = {
                'alert_id': alert.alert_id,
                'metric_name': alert.metric_name,
                'current_value': alert.current_value,
                'threshold_value': alert.threshold_value,
                'severity': alert.severity,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat()
            }
            
            response = requests.post(config.get('url', ''), json=payload)
            response.raise_for_status()
            
        except Exception as e:
            print(f"Ошибка отправки webhook уведомления: {e}")
    
    def _send_console_notification(self, alert: HealthAlert):
        """Отправляет уведомление в консоль"""
        print(f"[{alert.severity.upper()}] {alert.message} (Value: {alert.current_value})")
    
    def calculate_health_score(self) -> float:
        """
        Рассчитывает общий показатель здоровья системы
        
        Returns:
            Оценка здоровья (0-100)
        """
        if not self.current_metrics:
            return 100.0
        
        metrics = self.current_metrics
        
        # Рассчитываем вклад каждой метрики в общую оценку
        weights = {
            'cpu_percent': 0.25,
            'memory_percent': 0.25,
            'disk_percent': 0.20,
            'temperature_celsius': 0.15,
            'process_count': 0.15
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric_name, weight in weights.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                
                # Нормализуем значение (чем меньше, тем лучше здоровье)
                if metric_name in self.thresholds:
                    max_threshold = self.thresholds[metric_name]['critical']
                    normalized = min(100, (value / max_threshold) * 100)
                    # Чем выше нормализованное значение, тем хуже здоровье
                    metric_score = max(0, 100 - normalized)
                else:
                    metric_score = 100  # Если нет порога, считаем здоровым
                
                score += metric_score * weight
                total_weight += weight
        
        if total_weight > 0:
            score = score / total_weight
        
        # Инвертируем, чтобы высокий балл означал хорошее здоровье
        self.health_score = 100 - score
        return max(0, min(100, self.health_score))
    
    def start_monitoring(self, interval: float = 30.0):
        """
        Запускает мониторинг здоровья системы
        
        Args:
            interval: Интервал между проверками (в секундах)
        """
        if self.active:
            return
        
        self.active = True
        
        def monitor():
            while self.active:
                try:
                    # Получаем текущие метрики
                    metrics = self.get_system_metrics()
                    if metrics:
                        self.current_metrics = metrics
                        
                        # Создаем объекты HealthMetric
                        for metric_name, value in metrics.items():
                            if isinstance(value, (int, float)):
                                severity = self.evaluate_metric_severity(metric_name, value)
                                
                                # Определяем пороги для метрики
                                thresholds = self.thresholds.get(metric_name, {})
                                metric_obj = HealthMetric(
                                    name=metric_name,
                                    value=value,
                                    unit='%' if 'percent' in metric_name else 
                                         'GB' if 'gb' in metric_name else
                                         '°C' if 'temperature' in metric_name else
                                         'count' if 'count' in metric_name else
                                         'bytes' if 'bytes' in metric_name else
                                         'ms' if 'latency' in metric_name else
                                         'unknown',
                                    timestamp=metrics.get('timestamp', datetime.now()),
                                    severity=severity,
                                    source='system_monitor',
                                    threshold_low=thresholds.get('warning'),
                                    threshold_high=thresholds.get('critical')
                                )
                                
                                self.metrics.append(metric_obj)
                        
                        # Проверяем на наличие оповещений
                        alerts = self.check_for_alerts(metrics)
                        
                        # Обновляем оценку здоровья
                        self.calculate_health_score()
                        
                        # Обрабатываем оповещения
                        self.process_alerts()
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    print(f"Ошибка в мониторинге здоровья системы: {e}")
                    time.sleep(interval)
        
        self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Останавливает мониторинг здоровья системы"""
        self.active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
    
    def get_current_health_status(self) -> Dict[str, Any]:
        """
        Получает текущий статус здоровья системы
        
        Returns:
            Словарь с текущим статусом
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'health_score': self.health_score,
            'current_metrics': self.current_metrics,
            'active_alerts': len([a for a in self.alerts if not a.resolved]),
            'total_alerts': len(self.alerts),
            'recent_alerts': [a for a in self.alerts[-5:] if not a.resolved],
            'system_info': self._get_system_info()
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Получает информацию о системе"""
        try:
            return {
                'cpu_count': psutil.cpu_count(logical=True),
                'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
                'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat(),
                'hostname': socket.gethostname(),
                'platform': f"{os.name}-{sys.platform}"
            }
        except:
            return {'info': 'Could not retrieve system info'}
    
    def generate_health_report(self, output_path: str = None) -> str:
        """
        Генерирует отчет о здоровье системы
        
        Args:
            output_path: Путь для сохранения отчета
            
        Returns:
            Путь к сохраненному отчету
        """
        if output_path is None:
            output_path = str(self.output_dir / f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        report = {
            'generation_time': datetime.now().isoformat(),
            'health_score': self.health_score,
            'metrics_count': len(self.metrics),
            'alerts_count': len(self.alerts),
            'resolved_alerts': len([a for a in self.alerts if a.resolved]),
            'unresolved_alerts': len([a for a in self.alerts if not a.resolved]),
            'recent_metrics': [
                {
                    'name': m.name,
                    'value': m.value,
                    'unit': m.unit,
                    'severity': m.severity,
                    'timestamp': m.timestamp.isoformat()
                }
                for m in self.metrics[-20:]  # Последние 20 метрик
            ],
            'recent_alerts': [
                {
                    'id': a.alert_id,
                    'metric': a.metric_name,
                    'value': a.current_value,
                    'threshold': a.threshold_value,
                    'severity': a.severity,
                    'message': a.message,
                    'timestamp': a.timestamp.isoformat(),
                    'resolved': a.resolved
                }
                for a in self.alerts[-10:]  # Последние 10 оповещений
            ],
            'current_status': self.get_current_health_status(),
            'system_info': self._get_system_info()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        return output_path
    
    def get_health_recommendations(self) -> List[str]:
        """
        Получает рекомендации по улучшению здоровья системы
        
        Returns:
            Список рекомендаций
        """
        recommendations = []
        
        if not self.current_metrics:
            return ["Система не отвечает, невозможно получить метрики"]
        
        metrics = self.current_metrics
        
        # Рекомендации по CPU
        cpu_usage = metrics.get('cpu_percent', 0)
        if cpu_usage > 80:
            recommendations.append("Высокая загрузка CPU (>80%). Рассмотрите оптимизацию процессов или масштабирование.")
        elif cpu_usage > 60:
            recommendations.append("Загрузка CPU выше нормы (>60%). Следите за производительностью.")
        
        # Рекомендации по памяти
        memory_usage = metrics.get('memory_percent', 0)
        if memory_usage > 85:
            recommendations.append("Высокое использование памяти (>85%). Рассмотрите очистку кэша или увеличение объема памяти.")
        elif memory_usage > 70:
            recommendations.append("Использование памяти выше нормы (>70%). Следите за утечками памяти.")
        
        # Рекомендации по диску
        disk_usage = metrics.get('disk_percent', 0)
        if disk_usage > 90:
            recommendations.append("Критическое использование диска (>90%). Освободите место на диске срочно.")
        elif disk_usage > 80:
            recommendations.append("Высокое использование диска (>80%). Рассмотрите очистку старых файлов.")
        
        # Рекомендации по температуре
        temp = metrics.get('temperature_celsius', 0)
        if temp > 80:
            recommendations.append("Высокая температура системы (>80°C). Проверьте систему охлаждения.")
        elif temp > 70:
            recommendations.append("Температура системы выше нормы (>70°C). Следите за охлаждением.")
        
        # Рекомендации по процессам
        proc_count = metrics.get('process_count', 0)
        if proc_count > 500:
            recommendations.append("Очень большое количество процессов (>500). Проверьте систему на наличие лишних процессов.")
        elif proc_count > 300:
            recommendations.append("Высокое количество процессов (>300). Рассмотрите оптимизацию запущенных служб.")
        
        # Рекомендации на основе оценки здоровья
        if self.health_score < 60:
            recommendations.append("Общее здоровье системы низкое (<60). Требуется комплексная диагностика и оптимизация.")
        elif self.health_score < 80:
            recommendations.append("Общее здоровье системы ниже среднего (<80). Рассмотрите профилактические меры.")
        
        if not recommendations:
            recommendations.append("Система работает в нормальном режиме. Здоровье системы хорошее.")
        
        return recommendations


def main():
    """Главная функция для демонстрации возможностей монитора здоровья"""
    print("=== МОНИТОР ЗДОРОВЬЯ СИСТЕМЫ ===")
    
    # Создаем монитор
    health_monitor = SystemHealthMonitor()
    
    print("✓ Монитор здоровья системы инициализирован")
    print(f"✓ Директория вывода: {health_monitor.output_dir}")
    
    # Добавляем обработчик оповещений
    def alert_handler(alert):
        print(f"🚨 ОПОВЕЩЕНИЕ: {alert.message} (Уровень: {alert.severity})")
    
    health_monitor.add_alert_handler(alert_handler)
    
    # Добавляем консольный канал уведомлений
    health_monitor.add_notification_channel('console', {})
    
    # Запускаем мониторинг
    print("\nЗапуск мониторинга здоровья системы...")
    health_monitor.start_monitoring(interval=10)  # Проверка каждые 10 секунд
    
    # Ждем немного для сбора данных
    print("Сбор данных в течение 30 секунд...")
    time.sleep(30)
    
    # Останавливаем мониторинг
    health_monitor.stop_monitoring()
    print("✓ Мониторинг остановлен")
    
    # Получаем текущий статус
    print("\nПолучение текущего статуса системы...")
    status = health_monitor.get_current_health_status()
    print(f"✓ Оценка здоровья системы: {status['health_score']:.2f}")
    print(f"✓ Активных оповещений: {status['active_alerts']}")
    print(f"✓ Всего оповещений: {status['total_alerts']}")
    
    # Генерируем отчет
    print("\nГенерация отчета о здоровье...")
    report_path = health_monitor.generate_health_report()
    print(f"✓ Отчет сохранен: {report_path}")
    
    # Получаем рекомендации
    print("\nПолучение рекомендаций по здоровью системы...")
    recommendations = health_monitor.get_health_recommendations()
    print("Рекомендации:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    print("\nМонитор здоровья системы успешно протестирован")
    print("\nДоступные функции:")
    print("- Мониторинг: health_monitor.start_monitoring()")
    print("- Текущий статус: health_monitor.get_current_health_status()")
    print("- Отчеты: health_monitor.generate_health_report()")
    print("- Рекомендации: health_monitor.get_health_recommendations()")
    print("- Обработчики оповещений: health_monitor.add_alert_handler()")
    print("- Каналы уведомлений: health_monitor.add_notification_channel()")


if __name__ == "__main__":
    main()