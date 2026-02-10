#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль продвинутого анализа логов для проекта Лаборатория моделирования нанозонда
Этот модуль предоставляет инструменты для анализа, фильтрации и визуализации логов проекта.
"""

import os
import re
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import statistics
import threading
import time


@dataclass
class LogEntry:
    """Запись лога"""
    timestamp: datetime
    level: str
    component: str
    message: str
    module: str = ""
    function: str = ""
    thread_id: str = ""


@dataclass
class LogAnalysisResult:
    """Результат анализа логов"""
    total_entries: int
    error_count: int
    warning_count: int
    info_count: int
    debug_count: int
    unique_components: List[str]
    time_range: Tuple[datetime, datetime]
    analysis_timestamp: datetime


class AdvancedLoggerAnalyzer:
    """
    Класс продвинутого анализа логов
    Обеспечивает анализ, фильтрацию, статистику и визуализацию логов проекта.
    """
    
    def __init__(self, log_directory: str = "logs"):
        """
        Инициализирует анализатор логов
        
        Args:
            log_directory: Директория с логами
        """
        self.log_directory = Path(log_directory)
        self.log_entries = []
        self.analysis_results = {}
        self.patterns = {
            'timestamp': r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)\]',
            'level': r'(DEBUG|INFO|WARNING|ERROR|CRITICAL)',
            'component': r'\[([^\]]+)\]',
            'message': r'- (.+)$'
        }
        self.level_colors = {
            'DEBUG': '#808080',
            'INFO': '#008000',
            'WARNING': '#FFA500',
            'ERROR': '#FF0000',
            'CRITICAL': '#8B0000'
        }
        self.real_time_monitoring = False
        self.monitoring_thread = None
    
    def parse_log_file(self, file_path: str) -> List[LogEntry]:
        """
        Парсит файл лога
        
        Args:
            file_path: Путь к файлу лога
            
        Returns:
            Список записей лога
        """
        entries = []
        file_path = Path(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                entry = self.parse_log_line(line.strip())
                if entry:
                    entries.append(entry)
        
        except Exception as e:
            print(f"Ошибка парсинга файла {file_path}: {e}")
        
        return entries
    
    def parse_log_line(self, line: str) -> Optional[LogEntry]:
        """
        Парсит одну строку лога
        
        Args:
            line: Строка лога
            
        Returns:
            Запись лога или None если не удалось распарсить
        """
        try:
            # Пример формата: [2023-12-01 10:30:45] - INFO - component_name - Message text
            timestamp_match = re.search(self.patterns['timestamp'], line)
            level_match = re.search(self.patterns['level'], line)
            component_match = re.search(r'- ([A-Za-z_]+(?:\.[A-Za-z_]+)*) -', line)
            message_match = re.search(self.patterns['message'], line)
            
            if timestamp_match and level_match:
                timestamp = datetime.strptime(timestamp_match.group(1).split('.')[0], '%Y-%m-%d %H:%M:%S')
                level = level_match.group(1)
                component = component_match.group(1) if component_match else "Unknown"
                message = message_match.group(1) if message_match else line
                
                return LogEntry(
                    timestamp=timestamp,
                    level=level,
                    component=component,
                    message=message
                )
        
        except Exception:
            # Если стандартный формат не подходит, пробуем другие варианты
            pass
        
        return None
    
    def scan_log_directory(self, pattern: str = "*.log") -> List[Path]:
        """
        Сканирует директорию логов
        
        Args:
            pattern: Паттерн для поиска файлов
            
        Returns:
            Список найденных файлов логов
        """
        if not self.log_directory.exists():
            return []
        
        log_files = []
        for file_path in self.log_directory.rglob(pattern):
            log_files.append(file_path)
        
        return sorted(log_files, reverse=True)  # Сортируем по убыванию (новые первыми)
    
    def load_all_logs(self, pattern: str = "*.log") -> List[LogEntry]:
        """
        Загружает все логи из директории
        
        Args:
            pattern: Паттерн для поиска файлов
            
        Returns:
            Список всех записей логов
        """
        log_files = self.scan_log_directory(pattern)
        all_entries = []
        
        for log_file in log_files:
            print(f"Загрузка логов из: {log_file}")
            entries = self.parse_log_file(log_file)
            all_entries.extend(entries)
        
        # Сортируем по времени
        all_entries.sort(key=lambda x: x.timestamp)
        self.log_entries = all_entries
        
        return all_entries
    
    def filter_logs(self, level: str = None, component: str = None, 
                   start_time: datetime = None, end_time: datetime = None,
                   search_term: str = None) -> List[LogEntry]:
        """
        Фильтрует логи по различным критериям
        
        Args:
            level: Уровень лога (DEBUG, INFO, WARNING, ERROR)
            component: Компонент
            start_time: Начальное время
            end_time: Конечное время
            search_term: Поисковый термин
            
        Returns:
            Отфильтрованный список записей
        """
        filtered_entries = self.log_entries.copy()
        
        if level:
            filtered_entries = [entry for entry in filtered_entries 
                              if entry.level.upper() == level.upper()]
        
        if component:
            filtered_entries = [entry for entry in filtered_entries 
                              if component.lower() in entry.component.lower()]
        
        if start_time:
            filtered_entries = [entry for entry in filtered_entries 
                              if entry.timestamp >= start_time]
        
        if end_time:
            filtered_entries = [entry for entry in filtered_entries 
                              if entry.timestamp <= end_time]
        
        if search_term:
            search_term_lower = search_term.lower()
            filtered_entries = [entry for entry in filtered_entries 
                              if search_term_lower in entry.message.lower() or 
                                 search_term_lower in entry.component.lower()]
        
        return filtered_entries
    
    def analyze_logs(self, logs: List[LogEntry] = None) -> LogAnalysisResult:
        """
        Анализирует логи и возвращает статистику
        
        Args:
            logs: Список логов для анализа (если None, использует все загруженные)
            
        Returns:
            Результат анализа
        """
        if logs is None:
            logs = self.log_entries
        
        if not logs:
            return LogAnalysisResult(
                total_entries=0,
                error_count=0,
                warning_count=0,
                info_count=0,
                debug_count=0,
                unique_components=[],
                time_range=(datetime.now(), datetime.now()),
                analysis_timestamp=datetime.now()
            )
        
        # Подсчет по уровням
        level_counts = Counter(entry.level.upper() for entry in logs)
        
        # Уникальные компоненты
        unique_components = list(set(entry.component for entry in logs))
        
        # Диапазон времени
        timestamps = [entry.timestamp for entry in logs]
        time_range = (min(timestamps), max(timestamps))
        
        return LogAnalysisResult(
            total_entries=len(logs),
            error_count=level_counts.get('ERROR', 0) + level_counts.get('CRITICAL', 0),
            warning_count=level_counts.get('WARNING', 0),
            info_count=level_counts.get('INFO', 0),
            debug_count=level_counts.get('DEBUG', 0),
            unique_components=unique_components,
            time_range=time_range,
            analysis_timestamp=datetime.now()
        )
    
    def generate_statistics(self, logs: List[LogEntry] = None) -> Dict[str, Any]:
        """
        Генерирует статистику по логам
        
        Args:
            logs: Список логов для анализа
            
        Returns:
            Словарь со статистикой
        """
        if logs is None:
            logs = self.log_entries
        
        if not logs:
            return {}
        
        # Статистика по уровням
        level_counts = Counter(entry.level for entry in logs)
        
        # Статистика по компонентам
        component_counts = Counter(entry.component for entry in logs)
        
        # Статистика по времени
        timestamps = [entry.timestamp for entry in logs]
        time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                     for i in range(len(timestamps)-1)] if len(timestamps) > 1 else []
        
        # Часто встречающиеся сообщения
        message_counts = Counter(entry.message for entry in logs)
        
        # Временные интервалы (группировка по часам)
        hourly_counts = defaultdict(int)
        for entry in logs:
            hour_key = entry.timestamp.strftime('%Y-%m-%d %H:00')
            hourly_counts[hour_key] += 1
        
        return {
            'level_distribution': dict(level_counts),
            'component_distribution': dict(component_counts),
            'top_components': component_counts.most_common(10),
            'top_messages': message_counts.most_common(10),
            'hourly_activity': dict(hourly_counts),
            'avg_interval_between_logs': statistics.mean(time_diffs) if time_diffs else 0,
            'min_interval': min(time_diffs) if time_diffs else 0,
            'max_interval': max(time_diffs) if time_diffs else 0,
            'timestamp_stats': {
                'first_log': min(timestamps).isoformat() if timestamps else None,
                'last_log': max(timestamps).isoformat() if timestamps else None,
                'total_duration': (max(timestamps) - min(timestamps)).total_seconds() if timestamps else 0
            }
        }
    
    def detect_anomalies(self, logs: List[LogEntry] = None) -> List[Dict[str, Any]]:
        """
        Обнаруживает аномалии в логах
        
        Args:
            logs: Список логов для анализа
            
        Returns:
            Список обнаруженных аномалий
        """
        if logs is None:
            logs = self.log_entries
        
        anomalies = []
        
        if not logs:
            return anomalies
        
        # 1. Высокая частота ошибок
        error_entries = [entry for entry in logs if entry.level in ['ERROR', 'CRITICAL']]
        if len(error_entries) > len(logs) * 0.1:  # Если больше 10% ошибок
            anomalies.append({
                'type': 'high_error_rate',
                'severity': 'high',
                'description': f'Высокий процент ошибок: {len(error_entries)}/{len(logs)} ({len(error_entries)/len(logs)*100:.2f}%)',
                'timestamp': datetime.now()
            })
        
        # 2. Повторяющиеся сообщения
        message_counts = Counter(entry.message for entry in logs)
        repeated_messages = [msg for msg, count in message_counts.items() if count > 10]
        for msg in repeated_messages:
            anomalies.append({
                'type': 'repeated_messages',
                'severity': 'medium',
                'description': f'Повторяющееся сообщение: {msg}',
                'count': message_counts[msg],
                'timestamp': datetime.now()
            })
        
        # 3. Аномалии по времени (очень короткие интервалы)
        timestamps = sorted([entry.timestamp for entry in logs])
        time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                     for i in range(len(timestamps)-1)] if len(timestamps) > 1 else []
        
        if time_diffs:
            avg_interval = statistics.mean(time_diffs)
            threshold = avg_interval * 0.1  # 10% от среднего интервала
            rapid_logs = [i for i, diff in enumerate(time_diffs) if diff < threshold and diff > 0]
            
            if len(rapid_logs) > 10:  # Если много быстрых записей
                anomalies.append({
                    'type': 'rapid_logging',
                    'severity': 'medium',
                    'description': f'Обнаружено {len(rapid_logs)} случаев быстрого логирования',
                    'timestamp': datetime.now()
                })
        
        return anomalies
    
    def visualize_logs(self, output_path: str = None, logs: List[LogEntry] = None) -> str:
        """
        Визуализирует логи
        
        Args:
            output_path: Путь для сохранения визуализации
            logs: Список логов для визуализации
            
        Returns:
            Путь к сохраненной визуализации
        """
        if logs is None:
            logs = self.log_entries
        
        if not logs:
            print("Нет данных для визуализации")
            return ""
        
        if output_path is None:
            output_path = f"log_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        # Подготовка данных
        df = pd.DataFrame([{
            'timestamp': entry.timestamp,
            'level': entry.level,
            'component': entry.component,
            'message': entry.message
        } for entry in logs])
        
        df['hour'] = df['timestamp'].dt.hour
        df['date'] = df['timestamp'].dt.date
        
        # Создаем фигуру с несколькими подграфиками
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Анализ логов проекта', fontsize=16)
        
        # 1. Распределение по уровням
        level_counts = df['level'].value_counts()
        axes[0, 0].bar(level_counts.index, level_counts.values, 
                       color=[self.level_colors.get(level, '#000000') for level in level_counts.index])
        axes[0, 0].set_title('Распределение по уровням логов')
        axes[0, 0].set_ylabel('Количество')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Активность по компонентам (топ-10)
        component_counts = df['component'].value_counts().head(10)
        axes[0, 1].bar(range(len(component_counts)), component_counts.values)
        axes[0, 1].set_title('Активность по компонентам (топ-10)')
        axes[0, 1].set_ylabel('Количество')
        axes[0, 1].set_xticks(range(len(component_counts)))
        axes[0, 1].set_xticklabels(component_counts.index, rotation=45, ha='right')
        
        # 3. Активность по времени (часы)
        hourly_activity = df.groupby('hour').size()
        axes[1, 0].plot(hourly_activity.index, hourly_activity.values, marker='o')
        axes[1, 0].set_title('Активность по часам')
        axes[1, 0].set_xlabel('Час')
        axes[1, 0].set_ylabel('Количество записей')
        axes[1, 0].set_xticks(range(0, 24, 2))
        
        # 4. Активность по дням
        daily_activity = df.groupby('date').size()
        axes[1, 1].plot(range(len(daily_activity)), daily_activity.values, marker='o')
        axes[1, 1].set_title('Активность по дням')
        axes[1, 1].set_xlabel('День')
        axes[1, 1].set_ylabel('Количество записей')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def export_filtered_logs(self, logs: List[LogEntry], output_path: str, 
                           format_type: str = 'csv') -> str:
        """
        Экспортирует отфильтрованные логи
        
        Args:
            logs: Список логов для экспорта
            output_path: Путь для сохранения
            format_type: Формат ('csv', 'json', 'txt')
            
        Returns:
            Путь к сохраненному файлу
        """
        if not logs:
            print("Нет логов для экспорта")
            return ""
        
        if format_type.lower() == 'csv':
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['timestamp', 'level', 'component', 'message']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for entry in logs:
                    writer.writerow({
                        'timestamp': entry.timestamp.isoformat(),
                        'level': entry.level,
                        'component': entry.component,
                        'message': entry.message
                    })
        
        elif format_type.lower() == 'json':
            data = [{
                'timestamp': entry.timestamp.isoformat(),
                'level': entry.level,
                'component': entry.component,
                'message': entry.message
            } for entry in logs]
            
            with open(output_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=2, ensure_ascii=False, default=str)
        
        elif format_type.lower() == 'txt':
            with open(output_path, 'w', encoding='utf-8') as txtfile:
                for entry in logs:
                    txtfile.write(f"[{entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] - "
                                f"{entry.level} - {entry.component} - {entry.message}\n")
        
        return output_path
    
    def start_real_time_monitoring(self, log_file_path: str, callback: callable = None):
        """
        Начинает мониторинг лог-файла в реальном времени
        
        Args:
            log_file_path: Путь к лог-файлу для мониторинга
            callback: Функция обратного вызова для обработки новых записей
        """
        if self.real_time_monitoring:
            return
        
        self.real_time_monitoring = True
        
        def monitor():
            file_path = Path(log_file_path)
            if not file_path.exists():
                print(f"Файл {log_file_path} не существует")
                return
            
            # Начинаем читать с конца файла
            with open(file_path, 'r', encoding='utf-8') as f:
                f.seek(0, 2)  # Перемещаемся в конец файла
                while self.real_time_monitoring:
                    line = f.readline()
                    if line:
                        entry = self.parse_log_line(line.strip())
                        if entry:
                            self.log_entries.append(entry)
                            if callback:
                                callback(entry)
                    else:
                        time.sleep(0.1)  # Ждем новую строку
        
        self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self.monitoring_thread.start()
    
    def stop_real_time_monitoring(self):
        """Останавливает мониторинг в реальном времени"""
        self.real_time_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2)
    
    def generate_alerts(self, logs: List[LogEntry] = None) -> List[Dict[str, Any]]:
        """
        Генерирует оповещения на основе логов
        
        Args:
            logs: Список логов для анализа
            
        Returns:
            Список оповещений
        """
        if logs is None:
            logs = self.log_entries
        
        alerts = []
        
        # Оповещения об ошибках
        error_entries = [entry for entry in logs if entry.level in ['ERROR', 'CRITICAL']]
        if len(error_entries) > 0:
            alerts.append({
                'type': 'errors_detected',
                'severity': 'high',
                'count': len(error_entries),
                'message': f'Обнаружено {len(error_entries)} ошибок',
                'timestamp': datetime.now()
            })
        
        # Оповещения о предупреждениях
        warning_entries = [entry for entry in logs if entry.level == 'WARNING']
        if len(warning_entries) > 5:  # Если больше 5 предупреждений
            alerts.append({
                'type': 'warnings_detected',
                'severity': 'medium',
                'count': len(warning_entries),
                'message': f'Обнаружено {len(warning_entries)} предупреждений',
                'timestamp': datetime.now()
            })
        
        # Оповещения о высокой активности
        if len(logs) > 1000:  # Если много записей за период
            alerts.append({
                'type': 'high_activity',
                'severity': 'low',
                'count': len(logs),
                'message': f'Высокая активность: {len(logs)} записей',
                'timestamp': datetime.now()
            })
        
        return alerts


def main():
    """Главная функция для демонстрации возможностей анализатора логов"""
    print("=== ПРОДВИНУТЫЙ АНАЛИЗАТОР ЛОГОВ ===")
    
    # Создаем анализатор логов
    analyzer = AdvancedLoggerAnalyzer()
    
    print("✓ Анализатор логов инициализирован")
    print(f"✓ Директория логов: {analyzer.log_directory}")
    
    # Создаем тестовые логи если они не существуют
    test_logs_dir = Path("logs")
    test_logs_dir.mkdir(exist_ok=True)
    
    # Создаем тестовый лог-файл
    test_log_file = test_logs_dir / "test_app.log"
    with open(test_log_file, 'w', encoding='utf-8') as f:
        for i in range(100):
            timestamp = datetime.now() - timedelta(minutes=i)
            level = "INFO" if i % 10 != 0 else ("ERROR" if i % 5 == 0 else "WARNING")
            component = f"Component_{i % 5}"
            message = f"Test message {i} - Some event occurred in {component}"
            f.write(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] - {level} - {component} - {message}\n")
    
    print(f"✓ Создан тестовый лог-файл: {test_log_file}")
    
    # Загружаем логи
    print("\nЗагрузка логов...")
    logs = analyzer.load_all_logs()
    print(f"✓ Загружено {len(logs)} записей логов")
    
    # Анализируем логи
    print("\nАнализ логов...")
    analysis_result = analyzer.analyze_logs()
    print(f"  Всего записей: {analysis_result.total_entries}")
    print(f"  Ошибок: {analysis_result.error_count}")
    print(f"  Предупреждений: {analysis_result.warning_count}")
    print(f"  Информационных: {analysis_result.info_count}")
    print(f"  Компонентов: {len(analysis_result.unique_components)}")
    
    # Генерируем статистику
    print("\nГенерация статистики...")
    stats = analyzer.generate_statistics()
    print(f"  Уровни логов: {stats['level_distribution']}")
    print(f"  Топ компонентов: {stats['top_components'][:3]}")
    
    # Обнаруживаем аномалии
    print("\nПоиск аномалий...")
    anomalies = analyzer.detect_anomalies()
    print(f"  Найдено аномалий: {len(anomalies)}")
    for anomaly in anomalies[:3]:  # Показываем первые 3
        print(f"    - {anomaly['type']}: {anomaly['description']}")
    
    # Генерируем оповещения
    print("\nГенерация оповещений...")
    alerts = analyzer.generate_alerts()
    print(f"  Оповещений: {len(alerts)}")
    for alert in alerts[:3]:  # Показываем первые 3
        print(f"    - {alert['type']}: {alert['message']}")
    
    # Фильтруем логи
    print("\nФильтрация логов (только ошибки)...")
    error_logs = analyzer.filter_logs(level="ERROR")
    print(f"  Найдено ошибок: {len(error_logs)}")
    
    # Визуализируем логи
    print("\nСоздание визуализации...")
    viz_path = analyzer.visualize_logs()
    if viz_path:
        print(f"✓ Визуализация сохранена: {viz_path}")
    
    # Экспортируем отфильтрованные логи
    print("\nЭкспорт отфильтрованных логов...")
    export_path = analyzer.export_filtered_logs(error_logs, "filtered_errors.csv", "csv")
    print(f"✓ Экспортировано {len(error_logs)} ошибок: {export_path}")
    
    # Пример фильтрации по компоненту
    print("\nФильтрация по компоненту...")
    component_logs = analyzer.filter_logs(component="Component_1")
    print(f"  Записей для Component_1: {len(component_logs)}")
    
    # Пример фильтрации по поисковому запросу
    print("\nФильтрация по поисковому запросу...")
    search_logs = analyzer.filter_logs(search_term="event")
    print(f"  Записей с 'event': {len(search_logs)}")
    
    print("\nАнализатор логов успешно протестирован")
    print("\nДоступные функции:")
    print("- Загрузка логов: analyzer.load_all_logs()")
    print("- Анализ логов: analyzer.analyze_logs()")
    print("- Фильтрация: analyzer.filter_logs()")
    print("- Статистика: analyzer.generate_statistics()")
    print("- Обнаружение аномалий: analyzer.detect_anomalies()")
    print("- Визуализация: analyzer.visualize_logs()")
    print("- Экспорт: analyzer.export_filtered_logs()")
    print("- Мониторинг в реальном времени: analyzer.start_real_time_monitoring()")
    print("- Генерация оповещений: analyzer.generate_alerts()")


if __name__ == "__main__":
    main()