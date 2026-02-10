#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Веб-панель управления проектом "Лаборатория моделирования нанозонда"
Этот модуль предоставляет веб-интерфейс для управления всеми аспектами проекта,
включая симулятор СЗМ, анализатор изображений и наземную станцию SSTV.
"""

import os
import sys
import json
import time
import threading
import webbrowser
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
import eventlet

# Добавляем путь к utils для импорта служебных модулей
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.logger import Logger
from utils.error_handler import ErrorHandler
from utils.performance_monitor import PerformanceMonitor
from utils.system_monitor import SystemMonitor
from utils.config_manager import ConfigManager
from utils.cache_manager import CacheManager
from utils.data_manager import DataManager


class WebDashboard:
    """
    Класс веб-панели управления проектом
    Предоставляет веб-интерфейс для управления всеми компонентами проекта
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 5000):
        """
        Инициализация веб-панели
        
        Args:
            host: Хост сервера
            port: Порт сервера
        """
        self.host = host
        self.port = port
        
        # Инициализация Flask приложения
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'nanoprobe_simulation_lab_secret_key'
        
        # Инициализация SocketIO
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins="*",
            ping_timeout=60,
            ping_interval=25
        )
        
        # Инициализация служебных компонентов
        self.logger = Logger("WebDashboard")
        self.error_handler = ErrorHandler()
        self.performance_monitor = PerformanceMonitor()
        self.system_monitor = SystemMonitor()
        self.config_manager = ConfigManager()
        self.cache_manager = CacheManager()
        self.data_manager = DataManager()
        
        # Регистрация маршрутов
        self._register_routes()
        
        # Регистрация обработчиков SocketIO
        self._register_socket_handlers()
        
        self.logger.log_system_event("Веб-панель инициализирована", "INFO")
    
    def _register_routes(self):
        """Регистрация HTTP маршрутов"""
        
        @self.app.route('/')
        def index():
            """Главная страница веб-панели"""
            return render_template('dashboard.html')
        
        @self.app.route('/api/system_info')
        def api_system_info():
            """API для получения информации о системе"""
            try:
                system_info = self.system_monitor.get_current_metrics()
                
                # Добавляем информацию о проекте
                system_info["project_info"] = {
                    "name": "Nanoprobe Simulation Lab",
                    "version": "1.0.0",
                    "status": "running",
                    "uptime": self._get_uptime()
                }
                
                return jsonify(system_info)
            except Exception as e:
                self.error_handler.log_error(f"Ошибка получения информации о системе: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/performance_data')
        def api_performance_data():
            """API для получения данных о производительности"""
            try:
                performance_data = self.performance_monitor.get_current_metrics()
                return jsonify(performance_data)
            except Exception as e:
                self.error_handler.log_error(f"Ошибка получения данных о производительности: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/component_status')
        def api_component_status():
            """API для получения статуса компонентов"""
            try:
                # Здесь будет информация о статусе всех компонентов проекта
                component_status = {
                    "spm_simulator": {"status": "ready", "processes": 0},
                    "image_analyzer": {"status": "ready", "processes": 0},
                    "sstv_station": {"status": "ready", "processes": 0},
                    "web_dashboard": {"status": "running", "processes": 1}
                }
                return jsonify(component_status)
            except Exception as e:
                self.error_handler.log_error(f"Ошибка получения статуса компонентов: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/logs')
        def api_logs():
            """API для получения логов"""
            try:
                # Получаем последние N записей из логов
                log_file = "logs/web_dashboard.log"
                if os.path.exists(log_file):
                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        recent_logs = lines[-50:]  # Последние 50 строк
                else:
                    recent_logs = ["Лог-файл не найден"]
                
                return jsonify({"logs": recent_logs})
            except Exception as e:
                self.error_handler.log_error(f"Ошибка получения логов: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/config', methods=['GET', 'POST'])
        def api_config():
            """API для управления конфигурацией"""
            try:
                if request.method == 'GET':
                    # Возвращаем текущую конфигурацию
                    config = self.config_manager.get_config()
                    return jsonify(config)
                elif request.method == 'POST':
                    # Обновляем конфигурацию
                    new_config = request.json
                    self.config_manager.update_config(new_config)
                    return jsonify({"status": "success", "message": "Конфигурация обновлена"})
            except Exception as e:
                self.error_handler.log_error(f"Ошибка управления конфигурацией: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/cache_stats')
        def api_cache_stats():
            """API для получения статистики кэша"""
            try:
                cache_stats = self.cache_manager.get_cache_statistics()
                return jsonify(cache_stats)
            except Exception as e:
                self.error_handler.log_error(f"Ошибка получения статистики кэша: {e}")
                return jsonify({"error": str(e)}), 500
    
    def _register_socket_handlers(self):
        """Регистрация обработчиков SocketIO событий"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Обработка подключения клиента"""
            self.logger.log_system_event("Клиент подключен к веб-панели", "INFO")
            emit('connection_response', {'data': 'Connected to Nanoprobe Simulation Lab Dashboard'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Обработка отключения клиента"""
            self.logger.log_system_event("Клиент отключен от веб-панели", "INFO")
        
        @self.socketio.on('request_system_update')
        def handle_system_update_request():
            """Обработка запроса на обновление информации о системе"""
            try:
                system_info = self.system_monitor.get_current_metrics()
                system_info["project_info"] = {
                    "name": "Nanoprobe Simulation Lab",
                    "version": "1.0.0",
                    "status": "running",
                    "uptime": self._get_uptime()
                }
                emit('system_update', system_info)
            except Exception as e:
                self.error_handler.log_error(f"Ошибка отправки обновления системы: {e}")
        
        @self.socketio.on('request_performance_update')
        def handle_performance_update_request():
            """Обработка запроса на обновление информации о производительности"""
            try:
                performance_data = self.performance_monitor.get_current_metrics()
                emit('performance_update', performance_data)
            except Exception as e:
                self.error_handler.log_error(f"Ошибка отправки обновления производительности: {e}")
        
        @self.socketio.on('execute_command')
        def handle_execute_command(data):
            """Обработка команды выполнения"""
            try:
                command = data.get('command', '')
                params = data.get('params', {})
                
                # Логика выполнения команд
                if command == 'start_simulation':
                    result = self._execute_start_simulation(params)
                elif command == 'stop_simulation':
                    result = self._execute_stop_simulation(params)
                elif command == 'analyze_image':
                    result = self._execute_analyze_image(params)
                elif command == 'cleanup_cache':
                    result = self._execute_cleanup_cache(params)
                else:
                    result = {"status": "error", "message": f"Неизвестная команда: {command}"}
                
                emit('command_result', result)
            except Exception as e:
                self.error_handler.log_error(f"Ошибка выполнения команды: {e}")
                emit('command_result', {"status": "error", "message": str(e)})
    
    def _get_uptime(self) -> str:
        """Получение времени работы системы"""
        # В реальной реализации это будет вычисляться с момента запуска
        start_time = getattr(self, '_start_time', datetime.now())
        uptime = datetime.now() - start_time
        return str(uptime).split('.')[0]  # Убираем микросекунды
    
    def _execute_start_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Выполнение команды запуска симуляции"""
        try:
            # Здесь будет логика запуска симуляции
            simulation_id = f"sim_{int(time.time())}"
            self.logger.log_system_event(f"Запуск симуляции: {simulation_id}", "INFO")
            
            return {
                "status": "success",
                "message": f"Симуляция запущена: {simulation_id}",
                "simulation_id": simulation_id
            }
        except Exception as e:
            self.error_handler.log_error(f"Ошибка запуска симуляции: {e}")
            return {"status": "error", "message": str(e)}
    
    def _execute_stop_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Выполнение команды остановки симуляции"""
        try:
            simulation_id = params.get('simulation_id', '')
            self.logger.log_system_event(f"Остановка симуляции: {simulation_id}", "INFO")
            
            return {
                "status": "success",
                "message": f"Симуляция остановлена: {simulation_id}"
            }
        except Exception as e:
            self.error_handler.log_error(f"Ошибка остановки симуляции: {e}")
            return {"status": "error", "message": str(e)}
    
    def _execute_analyze_image(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Выполнение команды анализа изображения"""
        try:
            image_path = params.get('image_path', '')
            self.logger.log_system_event(f"Анализ изображения: {image_path}", "INFO")
            
            # Здесь будет логика анализа изображения
            return {
                "status": "success",
                "message": f"Изображение проанализировано: {image_path}",
                "results": {"analysis_complete": True}
            }
        except Exception as e:
            self.error_handler.log_error(f"Ошибка анализа изображения: {e}")
            return {"status": "error", "message": str(e)}
    
    def _execute_cleanup_cache(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Выполнение команды очистки кэша"""
        try:
            force = params.get('force', False)
            self.logger.log_system_event("Запуск очистки кэша", "INFO")
            
            cleanup_result = self.cache_manager.cleanup_cache(force=force)
            
            return {
                "status": "success",
                "message": "Кэш очищен",
                "cleanup_result": cleanup_result
            }
        except Exception as e:
            self.error_handler.log_error(f"Ошибка очистки кэша: {e}")
            return {"status": "error", "message": str(e)}
    
    def start_server(self, open_browser: bool = True):
        """
        Запуск веб-сервера
        
        Args:
            open_browser: Открывать ли браузер автоматически
        """
        self._start_time = datetime.now()
        self.logger.log_system_event(f"Запуск веб-панели на http://{self.host}:{self.port}", "INFO")
        
        print("="*60)
        print(f"Сервер запущен на http://{self.host}:{self.port}")
        print("Нажмите Ctrl+C для остановки сервера")
        print("="*60)
        
        # Открываем браузер в отдельном потоке
        if open_browser:
            def open_browser_func():
                """Open browser in a separate thread"""
                import time
                time.sleep(2)  # Ждем запуска сервера
                webbrowser.open(f"http://{self.host}:{self.port}")

            browser_thread = threading.Thread(target=open_browser_func)
            browser_thread.daemon = True
            browser_thread.start()

        try:
            self.socketio.run(self.app, host=self.host, port=self.port, debug=False)
        except KeyboardInterrupt:
            print("\nОстановка веб-панели...")
            self.logger.log_system_event("Веб-панель остановлена", "INFO")
        except Exception as e:
            self.error_handler.log_error(f"Ошибка запуска веб-панели: {e}")
            print(f"Ошибка: {e}")


def main():
    """Главная функция запуска веб-панели"""
    import argparse

    parser = argparse.ArgumentParser(description='Веб-панель управления проектом')
    parser.add_argument('--host', default='127.0.0.1', help='Хост сервера')
    parser.add_argument('--port', type=int, default=5000, help='Порт сервера')
    parser.add_argument('--no-browser', action='store_true', help='Не открывать браузер автоматически')

    args = parser.parse_args()

    # Создаем и запускаем веб-панель
    dashboard = WebDashboard(host=args.host, port=args.port)
    dashboard.start_server(open_browser=not args.no_browser)


if __name__ == "__main__":
    main()