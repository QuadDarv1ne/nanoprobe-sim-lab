#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт мониторинга ошибок и улучшения проекта Nanoprobe Simulation Lab
Этот скрипт позволяет отслеживать ошибки, производительность и 
предоставляет рекомендации по улучшению проекта.
"""

import os
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
import subprocess
import psutil
import time
from typing import Dict, List, Any

# Добавляем путь к проекту
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.system_monitor import SystemMonitor
from utils.code_analyzer import CodeAnalyzer
from utils.config_validator import ConfigValidator
from utils.test_framework import TestFramework


class ProjectMonitor:
    """
    Класс для мониторинга проекта, отслеживания ошибок и улучшения системы
    """
    
    def __init__(self):
        """Инициализация мониторинга проекта"""
        self.system_monitor = SystemMonitor()
        self.code_analyzer = CodeAnalyzer(str(project_root))
        self.config_validator = ConfigValidator()
        self.test_framework = TestFramework(str(project_root))
        self.monitoring_log = []
        
    def log_message(self, message: str, level: str = "INFO"):
        """Логирование сообщений"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message
        }
        self.monitoring_log.append(log_entry)
        print(f"[{level}] {timestamp}: {message}")
        
    def check_system_health(self):
        """Проверка общего состояния системы"""
        self.log_message("Проверка состояния системы...")
        
        try:
            health = self.system_monitor.get_system_health()
            self.log_message(f"Состояние системы: {health['health_score']}/100", 
                           "INFO" if health['health_score'] >= 70 else "WARNING")
            
            metrics = self.system_monitor.get_current_metrics()
            self.log_message(f"Загрузка CPU: {metrics['cpu_percent']}%", "INFO")
            self.log_message(f"Использование памяти: {metrics['memory_percent']}%", "INFO")
            self.log_message(f"Загрузка диска: {metrics['disk_usage']}%", "INFO")
            
            return health
        except Exception as e:
            self.log_message(f"Ошибка проверки состояния системы: {str(e)}", "ERROR")
            return None
    
    def analyze_code_quality(self):
        """Анализ качества кода"""
        self.log_message("Анализ качества кода...")
        
        try:
            # Анализируем проект
            report = self.code_analyzer.analyze_project(
                include_patterns=['*.py'],
                exclude_patterns=['venv', '__pycache__', '.git', 'node_modules']
            )
            
            issues = report.get('issues', [])
            metrics = report.get('metrics', {})
            
            self.log_message(f"Найдено проблем: {len(issues)}", 
                           "WARNING" if len(issues) > 0 else "INFO")
            
            # Показываем топ-5 проблем
            for issue in issues[:5]:
                self.log_message(f"Проблема: {issue['file_path']}:{issue['line_number']} - {issue['message']}", "WARNING")
            
            if len(issues) > 5:
                self.log_message(f"... и еще {len(issues) - 5} проблем", "WARNING")
            
            return report
        except Exception as e:
            self.log_message(f"Ошибка анализа кода: {str(e)}", "ERROR")
            return None
    
    def validate_configurations(self):
        """Проверка конфигураций проекта"""
        self.log_message("Проверка конфигураций...")
        
        try:
            config_path = project_root / "config" / "config.json"
            if config_path.exists():
                result = self.config_validator.validate_json_config(str(config_path))
                if result['valid']:
                    self.log_message("Конфигурационный файл валиден", "INFO")
                else:
                    self.log_message(f"Ошибки в конфигурации: {result['errors']}", "ERROR")
                return result
            else:
                self.log_message("Конфигурационный файл не найден", "WARNING")
                return None
        except Exception as e:
            self.log_message(f"Ошибка проверки конфигурации: {str(e)}", "ERROR")
            return None
    
    def run_tests(self):
        """Запуск тестов проекта"""
        self.log_message("Запуск тестов...")
        
        try:
            # Находим все тесты
            test_files = self.test_framework.discover_tests()
            self.log_message(f"Найдено тестовых файлов: {len(test_files)}", "INFO")
            
            if test_files:
                # Запускаем юнит-тесты
                results = self.test_framework.run_unittests()
                passed = results.get('passed', 0)
                total = results.get('total_tests', 0)
                failed = results.get('failures', 0) + results.get('errors', 0)
                
                self.log_message(f"Тесты: {passed}/{total} пройдено, {failed} ошибок", 
                               "ERROR" if failed > 0 else "INFO")
                
                return results
            else:
                self.log_message("Тестовые файлы не найдены", "WARNING")
                return None
        except Exception as e:
            self.log_message(f"Ошибка запуска тестов: {str(e)}", "ERROR")
            return None
    
    def check_performance(self):
        """Проверка производительности"""
        self.log_message("Проверка производительности...")
        
        try:
            # Проверяем использование ресурсов
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            self.log_message(f"Использование памяти процессом: {memory_info.rss / 1024 / 1024:.2f} MB", "INFO")
            
            # Проверяем время загрузки модулей
            start_time = time.time()
            import importlib.util
            # Попробуем импортировать основные модули
            modules_to_check = [
                'src.cli.main',
                'components.cpp-spm-hardware-sim.src.spm_simulator',
                'components.py-surface-image-analyzer.src.image_processor',
                'components.py-sstv-groundstation.src.sstv_decoder'
            ]
            
            for module_path in modules_to_check:
                try:
                    # Преобразуем путь в имя модуля
                    module_name = module_path.replace('/', '.').replace('\\', '.')
                    start_module_load = time.time()
                    
                    # Импортируем модуль
                    spec = importlib.util.spec_from_file_location(
                        module_name, 
                        project_root / f"{module_path.replace('.', '/')}.py"
                    )
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                    
                    load_time = time.time() - start_module_load
                    self.log_message(f"Загрузка модуля {module_name}: {load_time:.3f} сек", "INFO")
                except Exception as e:
                    self.log_message(f"Ошибка загрузки модуля {module_path}: {str(e)}", "ERROR")
            
            total_time = time.time() - start_time
            self.log_message(f"Общее время проверки производительности: {total_time:.3f} сек", "INFO")
            
            return {"load_times": {}, "memory_usage": memory_info.rss}
        except Exception as e:
            self.log_message(f"Ошибка проверки производительности: {str(e)}", "ERROR")
            return None
    
    def generate_improvement_recommendations(self):
        """Генерация рекомендаций по улучшению"""
        self.log_message("Генерация рекомендаций по улучшению...")
        
        recommendations = []
        
        # Проверяем наличие документации
        docs_exists = (project_root / "docs").exists()
        if not docs_exists:
            recommendations.append("Создать документацию проекта в папке docs/")
        
        # Проверяем наличие тестов
        test_files = list(project_root.glob("tests/**/*.py"))
        if len(test_files) < 5:
            recommendations.append("Добавить больше тестов в папку tests/")
        
        # Проверяем наличие CI/CD
        ci_exists = (project_root / ".github/workflows").exists() or (project_root / ".gitlab-ci.yml").exists()
        if not ci_exists:
            recommendations.append("Настроить CI/CD pipeline для автоматических тестов")
        
        # Проверяем качество кода
        code_analysis = self.analyze_code_quality()
        if code_analysis and code_analysis.get('issues'):
            recommendations.append(f"Исправить {len(code_analysis['issues'])} проблем с качеством кода")
        
        # Проверяем зависимости
        req_file = project_root / "requirements.txt"
        if req_file.exists():
            with open(req_file, 'r') as f:
                req_content = f.read()
                if '==' not in req_content:
                    recommendations.append("Зафиксировать версии зависимостей в requirements.txt")
        
        for rec in recommendations:
            self.log_message(f"Рекомендация: {rec}", "INFO")
        
        return recommendations
    
    def run_full_monitoring(self):
        """Запуск полного мониторинга проекта"""
        self.log_message("="*60, "INFO")
        self.log_message("ЗАПУСК МОНИТОРИНГА ПРОЕКТА NANOPROBE SIMULATION LAB", "INFO")
        self.log_message("="*60, "INFO")
        
        # Запускаем все проверки
        health = self.check_system_health()
        code_analysis = self.analyze_code_quality()
        config_validation = self.validate_configurations()
        test_results = self.run_tests()
        performance = self.check_performance()
        recommendations = self.generate_improvement_recommendations()
        
        # Сводка
        self.log_message("="*60, "INFO")
        self.log_message("СВОДКА МОНИТОРИНГА", "INFO")
        self.log_message("="*60, "INFO")
        
        if health:
            self.log_message(f"Состояние системы: {health['health_score']}/100", 
                           "INFO" if health['health_score'] >= 70 else "WARNING")
        
        if code_analysis:
            self.log_message(f"Проблем в коде: {len(code_analysis.get('issues', []))}", 
                           "WARNING" if len(code_analysis.get('issues', [])) > 0 else "INFO")
        
        if test_results:
            passed = test_results.get('passed', 0)
            total = test_results.get('total_tests', 0)
            self.log_message(f"Тестов пройдено: {passed}/{total}", 
                           "INFO" if passed == total and total > 0 else "WARNING")
        
        self.log_message(f"Рекомендаций по улучшению: {len(recommendations)}", "INFO")
        
        # Сохраняем отчет
        self.save_monitoring_report()
        
        return {
            "health": health,
            "code_analysis": code_analysis,
            "config_validation": config_validation,
            "test_results": test_results,
            "performance": performance,
            "recommendations": recommendations
        }
    
    def save_monitoring_report(self):
        """Сохранение отчета мониторинга"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = project_root / "reports" / f"monitoring_report_{timestamp}.json"
        
        # Создаем папку отчетов если не существует
        report_path.parent.mkdir(exist_ok=True)
        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "project_name": "Nanoprobe Simulation Lab",
            "monitoring_log": self.monitoring_log,
            "summary": {
                "total_messages": len(self.monitoring_log),
                "errors": len([log for log in self.monitoring_log if log['level'] == 'ERROR']),
                "warnings": len([log for log in self.monitoring_log if log['level'] == 'WARNING']),
                "infos": len([log for log in self.monitoring_log if log['level'] == 'INFO'])
            }
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        self.log_message(f"Отчет мониторинга сохранен: {report_path}", "INFO")


def main():
    """Основная функция запуска мониторинга"""
    print("Запуск мониторинга проекта Nanoprobe Simulation Lab...")
    
    monitor = ProjectMonitor()
    results = monitor.run_full_monitoring()
    
    print("\nМониторинг завершен!")
    print(f"Найдено рекомендаций по улучшению: {len(results['recommendations'])}")
    
    # Показываем топ-5 рекомендаций
    print("\nТоп-5 рекомендаций:")
    for i, rec in enumerate(results['recommendations'][:5], 1):
        print(f"{i}. {rec}")
    
    if len(results['recommendations']) > 5:
        print(f"... и еще {len(results['recommendations']) - 5} рекомендаций")


if __name__ == "__main__":
    main()