#!/usr/bin/env python3
"""
Веб-панель управления проектом "Лаборатория моделирования нанозонда"
Унифицированная версия с полной интеграцией FastAPI

Этот модуль предоставляет веб-интерфейс с:
- Reverse proxy к FastAPI API
- WebSocket для real-time данных
- Аутентификация через FastAPI
- Кэширование Redis
- SSTV Ground Station
- СЗМ симулятор
- Анализ изображений

Требования:
- Python 3.11, 3.12, 3.13, or 3.14
- Flask, Flask-SocketIO
- FastAPI (для backend)
"""

# Проверка версии Python
import sys
MIN_PYTHON_VERSION = (3, 11)
MAX_PYTHON_VERSION = (3, 14)
if sys.version_info < MIN_PYTHON_VERSION or sys.version_info >= (MAX_PYTHON_VERSION[0], MAX_PYTHON_VERSION[1] + 1):
    print(f"[ERROR] Требуется Python 3.11 - 3.14, текущая версия: {sys.version}")
    sys.exit(1)

import os
import time
import threading
import webbrowser
import subprocess
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

# UTF-8 для Windows
if sys.platform == "win32":
    os.system("chcp 65001 >nul")
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_socketio import SocketIO, emit
from functools import wraps
import requests

# Добавляем путь к utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Project root
project_root = Path(__file__).parent.parent.parent

# Импорт утилит
from utils.logger import NanoprobeLogger
from utils.error_handler import ErrorHandler
from utils.performance_monitor import PerformanceMonitor
from utils.system_monitor import SystemMonitor
from utils.config_manager import ConfigManager
from utils.cache_manager import CacheManager
from utils.data_manager import DataManager
from utils.data_exporter import DataExporter
from utils.database import DatabaseManager
from utils.surface_comparator import compare_surfaces as compare_surfaces_util
from utils.defect_analyzer import analyze_defects as analyze_defects_util

# Reverse proxy интеграция
try:
    from api.reverse_proxy import register_proxy, FASTAPI_URL, JWT_SECRET
    PROXY_AVAILABLE = True
except ImportError:
    PROXY_AVAILABLE = False
    FASTAPI_URL = os.getenv('FASTAPI_URL', 'http://localhost:8000')
    JWT_SECRET = os.getenv('JWT_SECRET', 'nanoprobe-secret-key-change-in-production')


# ==================== Декораторы ====================

def login_required(f):
    """Декоратор для защиты маршрутов"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            if request.is_json:
                return jsonify({'error': 'Требуется аутентификация'}), 401
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function


# ==================== Класс веб-панели ====================

class UnifiedWebDashboard:
    """
    Унифицированная веб-панель управления с интеграцией FastAPI
    
    Функционал:
    - Reverse proxy к FastAPI API
    - WebSocket real-time обновления
    - Аутентификация через FastAPI
    - SSTV Ground Station
    - СЗМ симулятор
    - Анализ изображений
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5000):
        """
        Инициализация веб-панели

        Args:
            host: Хост сервера (по умолчанию 127.0.0.1)
            port: Порт сервера (по умолчанию 5000)
        """
        self.host = host
        self.port = port
        self.fastapi_url = FASTAPI_URL

        # Project paths
        template_folder = project_root / "templates"
        static_folder = project_root / "static"

        # Flask приложение
        self.app = Flask(
            __name__,
            template_folder=str(template_folder),
            static_folder=str(static_folder) if static_folder.exists() else None
        )
        
        self.app.config["SECRET_KEY"] = os.getenv(
            'FLASK_SECRET_KEY',
            'nanoprobe_unified_dashboard_secret_key_change_in_production'
        )
        self.app.config['SESSION_TYPE'] = 'filesystem'
        self.app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 час

        # SocketIO
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins="*",
            ping_timeout=60,
            ping_interval=25,
            async_mode='threading'
        )

        # Логирование
        self.logger = NanoprobeLogger()
        self.error_handler = ErrorHandler()

        # Мониторинг
        self.performance_monitor = PerformanceMonitor()
        self.system_monitor = SystemMonitor()
        self.config_manager = ConfigManager()
        self.cache_manager = CacheManager()
        self.data_manager = DataManager()
        self.data_exporter = DataExporter(output_dir="output")
        self.database = DatabaseManager(db_path="data/nanoprobe.db")

        # Время запуска
        self._start_time = datetime.now()

        # WebSocket подключения
        self.active_websockets: List = []

        # Регистрация reverse proxy
        if PROXY_AVAILABLE:
            register_proxy(self.app)
            self.logger.log_system_event("Reverse proxy зарегистрирован", "INFO")
        else:
            self.logger.log_system_event("Reverse proxy недоступен", "WARNING")

        # Регистрация маршрутов
        self._register_routes()

        # Регистрация WebSocket обработчиков
        self._register_socket_handlers()

        self.logger.log_system_event("Унифицированная веб-панель инициализирована", "INFO")

    def _register_routes(self):
        """Регистрация HTTP маршрутов"""

        # ==================== Основные страницы ====================

        @self.app.route("/")
        def index():
            """Главная страница"""
            return render_template("dashboard_unified.html")

        @self.app.route("/login")
        def login_page():
            """Страница входа"""
            return render_template("login.html")

        @self.app.route("/sstv")
        def sstv_station():
            """SSTV Ground Station"""
            return render_template("sstv_station.html")

        # ==================== Аутентификация ====================

        @self.app.route("/api/auth/login", methods=["POST"])
        def api_login():
            """
            Аутентификация через FastAPI
            
            POST параметры:
            - username: Имя пользователя
            - password: Пароль
            
            Returns:
                JSON с токенами или ошибкой
            """
            try:
                username = request.form.get('username')
                password = request.form.get('password')

                if not username or not password:
                    return jsonify({'error': 'Требуется имя пользователя и пароль'}), 400

                # Запрос к FastAPI
                response = requests.post(
                    f"{self.fastapi_url}/api/v1/auth/login",
                    data={'username': username, 'password': password},
                    timeout=10
                )

                if response.status_code == 200:
                    tokens = response.json()
                    session['logged_in'] = True
                    session['username'] = username
                    session['access_token'] = tokens.get('access_token')
                    session['refresh_token'] = tokens.get('refresh_token')
                    session.permanent = True

                    self.logger.log_user_action(username, "Вход в систему выполнен")

                    return jsonify({
                        'success': True,
                        'username': username,
                        'message': 'Вход выполнен успешно'
                    })
                else:
                    error_data = response.json() if response.content else {}
                    return jsonify({
                        'error': error_data.get('detail', 'Неверное имя пользователя или пароль')
                    }), 401

            except requests.exceptions.RequestException as e:
                self.logger.log_error(f"Login error: {e}")
                return jsonify({'error': 'Сервер аутентификации недоступен'}), 503
            except Exception as e:
                self.logger.log_error(f"Unexpected login error: {e}")
                return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

        @self.app.route("/api/auth/logout", methods=["POST"])
        @login_required
        def api_logout():
            """Выход из системы"""
            username = session.get('username', 'Unknown')
            
            # Очистка сессии
            session.clear()
            
            self.logger.log_user_action(username, "Выход из системы")
            
            return jsonify({'success': True, 'message': 'Выход выполнен'})

        @self.app.route("/api/auth/refresh", methods=["POST"])
        def api_refresh_token():
            """Обновление токена доступа"""
            try:
                refresh_token = session.get('refresh_token')
                if not refresh_token:
                    return jsonify({'error': 'Токен обновления не найден'}), 401

                response = requests.post(
                    f"{self.fastapi_url}/api/v1/auth/refresh",
                    json={'refresh_token': refresh_token},
                    timeout=10
                )

                if response.status_code == 200:
                    tokens = response.json()
                    session['access_token'] = tokens.get('access_token')
                    if 'refresh_token' in tokens:
                        session['refresh_token'] = tokens['refresh_token']
                    
                    return jsonify({'success': True})
                else:
                    session.clear()
                    return jsonify({'error': 'Не удалось обновить токен'}), 401

            except Exception as e:
                self.logger.log_error(f"Token refresh error: {e}")
                return jsonify({'error': 'Ошибка обновления токена'}), 500

        # ==================== API маршруты ====================

        @self.app.route("/api/system_info")
        def api_system_info():
            """Информация о системе"""
            try:
                system_info = self.system_monitor.get_current_metrics()
                return jsonify(system_info)
            except Exception as e:
                self.logger.log_error(f"System info error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route("/api/stats")
        @login_required
        def api_stats():
            """Статистика дашборда"""
            try:
                # Получение статистики из FastAPI
                try:
                    response = requests.get(
                        f"{self.fastapi_url}/api/v1/dashboard/stats",
                        headers={'Authorization': f'Bearer {session.get("access_token")}'},
                        timeout=5
                    )
                    if response.status_code == 200:
                        return jsonify(response.json())
                except:
                    pass

                # Локальное получение статистики
                db = self.database
                stats = {
                    'scans_count': db.count_scans() if db else 0,
                    'simulations_count': db.count_simulations() if db else 0,
                    'analysis_count': db.count_analysis_results() if db else 0,
                    'comparisons_count': db.count_comparisons() if db else 0,
                    'reports_count': db.count_reports() if db else 0,
                    'uptime': str(datetime.now() - self._start_time)
                }
                return jsonify(stats)

            except Exception as e:
                self.logger.log_error(f"Stats error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route("/api/health")
        def api_health():
            """Проверка здоровья сервисов"""
            health = {
                'flask': 'ok',
                'fastapi': 'unknown',
                'database': 'unknown',
                'sync_manager': 'unknown',
                'timestamp': datetime.now().isoformat()
            }

            # Проверка FastAPI
            try:
                response = requests.get(f"{self.fastapi_url}/health", timeout=3)
                health['fastapi'] = 'ok' if response.status_code == 200 else 'error'
            except:
                health['fastapi'] = 'error'

            # Проверка БД
            try:
                if self.database:
                    health['database'] = 'ok'
            except:
                health['database'] = 'error'

            # Проверка Sync Manager
            try:
                response = requests.get(f"{self.fastapi_url}/api/v1/sync/status", timeout=3)
                if response.status_code == 200:
                    sync_data = response.json()
                    health['sync_manager'] = 'ok' if sync_data.get('running') else 'standby'
                    health['sync_last_update'] = sync_data.get('last_sync_time')
            except:
                health['sync_manager'] = 'not_available'

            all_ok = all(v == 'ok' for k, v in health.items() if k not in ['timestamp', 'sync_last_update'])
            health['status'] = 'healthy' if all_ok else 'degraded'

            return jsonify(health)

        # ==================== Sync Manager ====================

        @self.app.route("/api/sync/status")
        def api_sync_status():
            """Статус Sync Manager"""
            try:
                response = requests.get(f"{self.fastapi_url}/api/v1/sync/status", timeout=5)
                if response.status_code == 200:
                    return jsonify(response.json())
                else:
                    return jsonify({
                        'running': False,
                        'message': 'Sync Manager недоступен'
                    })
            except Exception as e:
                return jsonify({
                    'running': False,
                    'message': f'Ошибка: {str(e)}'
                })

        # ==================== СЗМ операции ====================

        @self.app.route("/api/spm/simulate", methods=["POST"])
        @login_required
        def api_spim_simulate():
            """Запуск симуляции СЗМ"""
            try:
                data = request.json
                scan_type = data.get('scan_type', 'afm')
                surface_type = data.get('surface_type', 'flat')
                resolution = data.get('resolution', 64)

                # Логирование
                username = session.get('username', 'Unknown')
                self.logger.log_user_action(
                    username,
                    f"Запуск СЗМ симуляции: {scan_type}, {surface_type}, {resolution}x{resolution}"
                )

                # Здесь должна быть логика симуляции
                return jsonify({
                    'success': True,
                    'message': f'Симуляция запущена: {scan_type}',
                    'scan_id': int(time.time())
                })

            except Exception as e:
                self.logger.log_error(f"SPM simulation error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route("/api/spm/scan", methods=["POST"])
        @login_required
        def api_spim_scan():
            """Сканирование поверхности"""
            try:
                data = request.json
                scan_type = data.get('scan_type', 'afm')
                surface_type = data.get('surface_type', 'flat')

                username = session.get('username', 'Unknown')
                self.logger.log_user_action(username, f"Сканирование: {scan_type}, {surface_type}")

                return jsonify({
                    'success': True,
                    'message': 'Сканирование выполнено',
                    'scan_id': int(time.time())
                })

            except Exception as e:
                self.logger.log_error(f"SPM scan error: {e}")
                return jsonify({'error': str(e)}), 500

        # ==================== Анализ изображений ====================

        @self.app.route("/api/analysis/compare", methods=["POST"])
        @login_required
        def api_compare_surfaces():
            """Сравнение поверхностей"""
            try:
                data = request.json
                image1_path = data.get('image1')
                image2_path = data.get('image2')

                if not image1_path or not image2_path:
                    return jsonify({'error': 'Требуется два изображения'}), 400

                result = compare_surfaces_util(image1_path, image2_path)

                return jsonify({
                    'success': True,
                    'comparison': result
                })

            except Exception as e:
                self.logger.log_error(f"Surface comparison error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route("/api/analysis/defects", methods=["POST"])
        @login_required
        def api_analyze_defects():
            """Анализ дефектов"""
            try:
                data = request.json
                image_path = data.get('image')

                if not image_path:
                    return jsonify({'error': 'Требуется изображение'}), 400

                result = analyze_defects_util(image_path)

                return jsonify({
                    'success': True,
                    'analysis': result
                })

            except Exception as e:
                self.logger.log_error(f"Defect analysis error: {e}")
                return jsonify({'error': str(e)}), 500

        # ==================== Экспорт данных ====================

        @self.app.route("/api/export/json", methods=["GET"])
        @login_required
        def api_export_json():
            """Экспорт данных в JSON"""
            try:
                export_type = request.args.get('type', 'all')
                output_file = self.data_exporter.export_to_json(export_type)

                return jsonify({
                    'success': True,
                    'file': str(output_file)
                })

            except Exception as e:
                self.logger.log_error(f"JSON export error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route("/api/export/csv", methods=["GET"])
        @login_required
        def api_export_csv():
            """Экспорт данных в CSV"""
            try:
                export_type = request.args.get('type', 'scans')
                output_file = self.data_exporter.export_to_csv(export_type)

                return jsonify({
                    'success': True,
                    'file': str(output_file)
                })

            except Exception as e:
                self.logger.log_error(f"CSV export error: {e}")
                return jsonify({'error': str(e)}), 500

    def _register_socket_handlers(self):
        """Регистрация обработчиков WebSocket"""

        @self.socketio.on('connect')
        def handle_connect():
            """Подключение клиента"""
            self.active_websockets.append(request.sid)
            self.logger.log_system_event(f"WebSocket подключён: {request.sid}")
            emit('connected', {'sid': request.sid})

        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Отключение клиента"""
            if request.sid in self.active_websockets:
                self.active_websockets.remove(request.sid)
            self.logger.log_system_event(f"WebSocket отключён: {request.sid}")

        @self.socketio.on('subscribe_metrics')
        def handle_subscribe_metrics():
            """Подписка на метрики"""
            self.logger.log_system_event(f"Клиент {request.sid} подписан на метрики")
            
            # Отправка текущих метрик
            try:
                metrics = self.system_monitor.get_current_metrics()
                emit('metrics_update', metrics)
            except Exception as e:
                self.logger.log_error(f"Metrics send error: {e}")

        @self.socketio.on('subscribe_stats')
        def handle_subscribe_stats():
            """Подписка на статистику"""
            self.logger.log_system_event(f"Клиент {request.sid} подписан на статистику")
            
            # Отправка текущей статистики
            try:
                stats = {
                    'scans_count': self.database.count_scans() if self.database else 0,
                    'simulations_count': self.database.count_simulations() if self.database else 0,
                    'analysis_count': self.database.count_analysis_results() if self.database else 0,
                }
                emit('stats_update', stats)
            except Exception as e:
                self.logger.log_error(f"Stats send error: {e}")

    def _background_metrics_updater(self):
        """Фоновое обновление метрик (WebSocket)"""
        while True:
            time.sleep(5)  # Каждые 5 секунд
            
            try:
                metrics = self.system_monitor.get_current_metrics()
                
                # Отправка всем подключённым клиентам
                self.socketio.emit('metrics_update', metrics)
                
            except Exception as e:
                self.logger.log_error(f"Background metrics error: {e}")

    def start(self, open_browser: bool = True):
        """
        Запуск веб-панели

        Args:
            open_browser: Открыть браузер автоматически
        """
        self.logger.log_system_event(f"Запуск веб-панели на {self.host}:{self.port}", "INFO")

        # Запуск фонового обновления метрик
        metrics_thread = threading.Thread(target=self._background_metrics_updater, daemon=True)
        metrics_thread.start()

        # Открытие браузера
        if open_browser:
            def open_browser_delayed():
                time.sleep(2)
                webbrowser.open(f"http://{self.host}:{self.port}")
            
            browser_thread = threading.Thread(target=open_browser_delayed, daemon=True)
            browser_thread.start()

        # Запуск Flask + SocketIO
        print("=" * 70)
        print("  Унифицированная веб-панель управления")
        print("=" * 70)
        print(f"  URL: http://{self.host}:{self.port}")
        print(f"  FastAPI: {self.fastapi_url}")
        print(f"  Reverse Proxy: {'Включён' if PROXY_AVAILABLE else 'Отключён'}")
        print("=" * 70)
        print()

        self.socketio.run(
            self.app,
            host=self.host,
            port=self.port,
            debug=False,
            log_output=False,
            allow_unsafe_werkzeug=True
        )


# ==================== Точка входа ====================

def main():
    """Точка входа для запуска веб-панели"""
    import argparse

    parser = argparse.ArgumentParser(description='Унифицированная веб-панель Nanoprobe Sim Lab')
    parser.add_argument('--host', default='127.0.0.1', help='Хост сервера')
    parser.add_argument('--port', type=int, default=5000, help='Порт сервера')
    parser.add_argument('--no-browser', action='store_true', help='Не открывать браузер')
    
    args = parser.parse_args()

    dashboard = UnifiedWebDashboard(host=args.host, port=args.port)
    dashboard.start(open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
