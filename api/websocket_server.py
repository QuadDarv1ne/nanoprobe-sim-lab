"""WebSocket сервер для realtime обновлений дашборда."""

from flask import Flask, request
from flask_socketio import SocketIO, emit
from datetime import datetime
import threading
import time
import psutil

from utils.config.config_manager import ConfigManager
from utils.logger import setup_project_logging


class WebSocketServer:
    """Сервер WebSocket для realtime обновлений."""

    def __init__(self, app: Flask = None):
        """Инициализирует WebSocket сервер."""
        self.app = app or Flask(__name__)
        self.app.config['SECRET_KEY'] = 'nanoprobe-secret-key'
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins="*",
            async_mode='eventlet',
            logger=False,
            engineio_logger=False
        )
        self.config_manager = ConfigManager()
        self.logger = setup_project_logging(self.config_manager)
        self.running = False
        self._connected_clients = set()
        self._setup_routes()
        self._setup_error_handlers()

    def _setup_routes(self):
        """Настраивает WebSocket маршруты."""

        @self.socketio.on('connect')
        def handle_connect():
            """Обработчик подключения клиента."""
            self._connected_clients.add(request.sid)
            self.logger.info(f"Клиент подключился: {request.sid}")
            emit('status', {'status': 'connected', 'timestamp': datetime.now().isoformat()})

        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Обработчик отключения клиента."""
            self._connected_clients.discard(request.sid)
            self.logger.info(f"Клиент отключился: {request.sid}")

        @self.socketio.on('request_metrics')
        def handle_request_metrics():
            """Запрос метрик системы."""
            metrics = self._get_system_metrics()
            emit('metrics', metrics)

        @self.socketio.on('start_simulation')
        def handle_start_simulation(data):
            """Запуск симуляции."""
            simulation_id = data.get('simulation_id', 'default')
            emit('simulation_started', {
                'simulation_id': simulation_id,
                'timestamp': datetime.now().isoformat()
            })

        @self.socketio.on('stop_simulation')
        def handle_stop_simulation(data):
            """Остановка симуляции."""
            simulation_id = data.get('simulation_id', 'default')
            emit('simulation_stopped', {
                'simulation_id': simulation_id,
                'timestamp': datetime.now().isoformat()
            })

    def _setup_error_handlers(self):
        """Настраивает обработчики ошибок."""
        @self.socketio.on('error')
        def handle_error(error):
            """Обработчик ошибок WebSocket."""
            self.logger.error(f"WebSocket ошибка: {error}")
            emit('error', {'message': str(error)})

    def _get_system_metrics(self) -> dict:
        """Получает метрики системы."""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'timestamp': datetime.now().isoformat()
        }

    def get_connected_clients_count(self) -> int:
        """Возвращает количество подключенных клиентов."""
        return len(self._connected_clients)

    def broadcast_metrics(self, interval: float = 1.0):
        """Транслирует метрики системы всем подключенным клиентам."""
        while self.running:
            try:
                metrics = self._get_system_metrics()
                self.socketio.emit('metrics', metrics)
            except Exception as e:
                self.logger.error(f"Ошибка трансляции метрик: {e}")
            time.sleep(interval)

    def start(self, host: str = '0.0.0.0', port: int = 5001, debug: bool = False):
        """Запускает WebSocket сервер."""
        self.running = True

        # Запускаем поток трансляции метрик
        metrics_thread = threading.Thread(target=self.broadcast_metrics, daemon=True)
        metrics_thread.start()

        self.logger.info(f"WebSocket сервер запущен на {host}:{port}")
        try:
            self.socketio.run(self.app, host=host, port=port, debug=debug)
        except Exception as e:
            self.logger.error(f"Ошибка WebSocket сервера: {e}")
            self.running = False

    def stop(self):
        """Останавливает WebSocket сервер."""
        self.running = False
        self.logger.info(f"WebSocket сервер остановлен. Подключено клиентов: {len(self._connected_clients)}")
        self.socketio.stop()


# Для прямого запуска
if __name__ == '__main__':
    server = WebSocketServer()
    server.start()
