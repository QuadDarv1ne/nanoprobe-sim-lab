# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Веб-панель управления для Лаборатории моделирования нанозонда
Этот скрипт создает веб-интерфейс для управления всеми компонентами проекта.
"""

import os
import sys
import atexit
import json
import threading
import webbrowser
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Импорты Flask и связанных библиотек
try:
    from flask import Flask, render_template, request, jsonify, redirect, url_for
    from flask_socketio import SocketIO, emit
except ImportError:
    print("Установите необходимые зависимости:")
    print("pip install flask flask-socketio")
    sys.exit(1)

# Импорты из нашего проекта
from utils.config_manager import ConfigManager
from utils.logger import setup_project_logging
from utils.data_manager import DataManager
from utils.analytics import ProjectAnalytics
from utils.system_monitor import SystemMonitor
from utils.cache_manager import CacheManager
from utils.error_handler import ErrorHandler

# Configuration paths
CONFIG_PATH = project_root / "config" / "config.json"
TEMPLATES_PATH = project_root / "templates"

class WebDashboard:
    """
    Класс веб-панели управления
    Обеспечивает веб-интерфейс для управления проектом.
    """


    def __init__(self, host: str = '127.0.0.1', port: int = 5000):
        """
        Инициализирует веб-панель

        Args:
            host: Хост для запуска сервера
            port: Порт для запуска сервера
        """
        self.host = host
        self.port = port
        self.app = Flask(__name__, template_folder=str(TEMPLATES_PATH))
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        # Инициализация компонентов проекта
        self.config_manager = ConfigManager(str(CONFIG_PATH))
        self.logger = setup_project_logging(self.config_manager)
        self.data_manager = DataManager()
        self.analytics = ProjectAnalytics()
        self.system_monitor = SystemMonitor()
        self.cache_manager = CacheManager(str(project_root))
        self.error_handler = ErrorHandler()

        # Состояние запущенных процессов
        self.running_processes = {}

        # Регистрируем автоматическую очистку кэша при завершении
        atexit.register(self._auto_cleanup_on_exit)

        # Настройка маршрутов
        self._setup_routes()

        # Настройка SocketIO
        self._setup_socketio()


    def _setup_routes(self):
        """Настройка маршрутов Flask"""

        @self.app.route('/')
        def index():
            """Главная страница"""
            return render_template('dashboard.html')

        @self.app.route('/api/status')
        def api_status():
            """API для получения статуса системы"""
            try:
                status = {
                    'project_info': self._get_project_info(),
                    'system_metrics': self._get_system_metrics(),
                    'cache_info': self._get_cache_info(),
                    'running_processes': self._get_running_processes(),
                    'timestamp': datetime.now().isoformat()
                }
                return jsonify(status)
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/components')
        def api_components():
            """API для получения информации о компонентах"""
            try:
                components = self._get_components_info()
                return jsonify(components)
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/actions/<action>', methods=['POST'])
        def api_actions(action):
            """API для выполнения действий"""
            try:
                data = request.get_json() or {}
                result = self._execute_action(action, data)
                return jsonify(result)
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/logs')
        def api_logs():
            """API для получения логов"""
            try:
                limit = int(request.args.get('limit', 50))
                logs = self._get_recent_logs(limit)
                return jsonify(logs)
            except Exception as e:
                return jsonify({'error': str(e)}), 500


    def _setup_socketio(self):
        """Настройка SocketIO для реального времени"""

        @self.socketio.on('connect')
        def handle_connect():
            """Обработка подключения клиента"""
            print("Клиент подключен к веб-панели")
            emit('status_update', {'message': 'Подключено к серверу'})

        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Обработка отключения клиента"""
            print("Клиент отключен от веб-панели")

        @self.socketio.on('request_update')
        def handle_update_request():
            """Обработка запроса на обновление данных"""
            status = self._get_realtime_status()
            emit('status_update', status)


    def _get_project_info(self) -> Dict[str, Any]:
        """Получает информацию о проекте"""
        try:
            config = self.config_manager.config  # Исправлено: config вместо get_config()
            return {
                'name': config.get('project', {}).get('name', 'Nanoprobe Simulation Lab'),
                'version': config.get('project', {}).get('version', '1.0.0'),
                'description': config.get('project', {}).get('description', ''),
                'author': config.get('project', {}).get('author', ''),
                'components_count': len(config.get('components', {}))
            }
        except Exception as e:
            self.error_handler.log_error(f"Ошибка получения информации о проекте: {e}")
            return {'error': str(e)}


    def _get_system_metrics(self) -> Dict[str, Any]:
        """Получает системные метрики"""
        try:
            metrics = self.system_monitor.get_current_metrics()  # Исправлено: get_current_metrics()
            return {
                'cpu_percent': metrics.get('cpu_percent', 0),
                'memory_percent': metrics.get('memory_percent', 0),
                'disk_usage': metrics.get('disk_usage', 0),
                'uptime': metrics.get('uptime', 0)
            }
        except Exception as e:
            self.error_handler.log_error(f"Ошибка получения системных метрик: {e}")
            return {'error': str(e)}


    def _get_cache_info(self) -> Dict[str, Any]:
        """Получает информацию о кэше"""
        try:
            stats = self.cache_manager.get_cache_statistics()
            return {
                'total_size_mb': stats.get('total_cache_size_mb', 0),
                'total_files': stats.get('total_files', 0),
                'directories_count': stats.get('cache_directories_count', 0),
                'auto_cleanup_enabled': stats.get('auto_cleanup_enabled', False)
            }
        except Exception as e:
            self.error_handler.log_error(f"Ошибка получения информации о кэше: {e}")
            return {'error': str(e)}


    def _get_running_processes(self) -> Dict[str, Any]:
        """Получает информацию о запущенных процессах"""
        return self.running_processes


    def _get_components_info(self) -> List[Dict[str, Any]]:
        """Получает информацию о компонентах проекта"""
        try:
            config = self.config_manager.config  # Исправлено: config вместо get_config()
            components = []

            for name, info in config.get('components', {}).items():
                components.append({
                    'name': info.get('name', name),
                    'description': info.get('description', ''),
                    'language': info.get('language', ''),
                    'path': info.get('path', ''),
                    'status': 'available'  # В реальной реализации можно проверять доступность
                })

            return components
        except Exception as e:
            self.error_handler.log_error(f"Ошибка получения информации о компонентах: {e}")
            return []


    def _execute_action(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Выполняет действие"""
        try:
            if action == 'start_component':
                component = data.get('component')
                if component:
                    result = self._start_component(component)
                    return {'success': True, 'message': f'Компонент {component} запущен', 'result': result}
                else:
                    return {'success': False, 'error': 'Не указан компонент'}

            elif action == 'stop_component':
                component = data.get('component')
                if component:
                    result = self._stop_component(component)
                    return {'success': True, 'message': f'Компонент {component} остановлен', 'result': result}
                else:
                    return {'success': False, 'error': 'Не указан компонент'}

            elif action == 'clean_cache':
                result = self.cache_manager.cleanup_cache()
                return {'success': True, 'message': 'Кэш очищен', 'result': result}

            elif action == 'get_analytics':
                result = self.analytics.generate_project_report()
                return {'success': True, 'message': 'Аналитика получена', 'result': result}

            else:
                return {'success': False, 'error': f'Неизвестное действие: {action}'}

        except Exception as e:
            self.error_handler.log_error(f"Ошибка выполнения действия {action}: {e}")
            return {'success': False, 'error': str(e)}


    def _start_component(self, component_name: str) -> Dict[str, Any]:
        """Запускает компонент"""
        # В реальной реализации здесь будет код запуска компонентов
        self.running_processes[component_name] = {
            'status': 'running',
            'start_time': datetime.now().isoformat(),
            'pid': os.getpid()  # В реальности будет PID запущенного процесса
        }

        self.logger.log_general_activity(f"Запуск компонента: {component_name}", "INFO")
        return {'status': 'started', 'component': component_name}


    def _stop_component(self, component_name: str) -> Dict[str, Any]:
        """Останавливает компонент"""
        if component_name in self.running_processes:
            del self.running_processes[component_name]
            self.logger.log_general_activity(f"Остановка компонента: {component_name}", "INFO")
            return {'status': 'stopped', 'component': component_name}
        else:
            return {'status': 'not_running', 'component': component_name}


    def _get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Получает последние записи логов"""
        # В реальной реализации будет чтение из файлов логов
        return [
            {
                'timestamp': datetime.now().isoformat(),
                'level': 'INFO',
                'component': 'WebDashboard',
                'message': 'Веб-панель запущена'
            }
        ][:limit]


    def _get_realtime_status(self) -> Dict[str, Any]:
        """Получает статус в реальном времени"""
        return {
            'system_metrics': self._get_system_metrics(),
            'cache_info': self._get_cache_info(),
            'running_processes': self._get_running_processes(),
            'timestamp': datetime.now().isoformat()
        }


    def _auto_cleanup_on_exit(self):
        """Внутренняя функция автоматической очистки при завершении"""
        print("\n" + "="*50)
        print("Автоматическая очистка кэша через WebDashboard...")
        try:
            # Останавливаем SocketIO сервер корректно
            if hasattr(self, 'socketio'):
                self.socketio.stop()

            # Выполняем очистку кэша
            stats = self.cache_manager.get_cache_statistics()
            print(f"Текущий размер кэша: {stats['total_cache_size_mb']} MB")

            result = self.cache_manager.auto_cleanup()

            if "status" in result:
                print(f"Статус: {result['status']}")
            else:
                print(f"Удалено файлов: {result['deleted_files']}")
                print(f"Освобождено места: {result['freed_space_mb']} MB")

            # Оптимизация памяти
            memory_result = self.cache_manager.optimize_memory_usage()
            print(f"Освобождено памяти: {memory_result['memory_freed_mb']} MB")

            print("✓ Автоматическая очистка кэша выполнена успешно")
        except Exception as e:
            print(f"❌ Ошибка при автоматической очистке кэша: {e}")
        print("="*50)


    def start_server(self, open_browser: bool = True):
        """
        Запускает веб-сервер

        Args:
            open_browser: Открыть браузер автоматически
        """
        print("="*60)
        print("           ВЕБ-ПАНЕЛЬ УПРАВЛЕНИЯ")
        print("    Лаборатория моделирования нанозонда")
        print("="*60)
        print(f"Сервер запущен на http://{self.host}:{self.port}")
        print("Нажмите Ctrl+C для остановки сервера")
        print("="*60)

        # Открываем браузер в отдельном потоке
        if open_browser:
            def open_browser_func():
    """TODO: Add description"""

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

