"""API интерфейс для проекта Лаборатория моделирования нанозонда."""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json
import uuid
from datetime import datetime
from pathlib import Path
import numpy as np
from typing import Dict, Any, Tuple
import threading
import time

from utils.config_manager import ConfigManager
from utils.data_manager import DataManager
from utils.logger import NanoprobeLogger, setup_project_logging
from cpp_spm_hardware_sim.src.spm_simulator import SurfaceModel, SPMController
from py_surface_image_analyzer.src.image_processor import ImageProcessor
from py_sstv_groundstation.src.sstv_decoder import SSTVDecoder
from .validators import DataValidator, ResponseBuilder, ValidationError

class NanoprobeAPI:
    """
    Класс API интерфейса проекта
    Обеспечивает REST API для взаимодействия между компонентами и внешними системами.
    """

    active_simulations: Dict[str, Dict[str, Any]]
    results_cache: Dict[str, Dict[str, Any]]

    def __init__(self) -> None:
        """Инициализирует API интерфейс"""
        self.app = Flask(__name__)
        CORS(self.app)  # Разрешаем кросс-доменные запросы

        # Инициализация компонентов
        self.config_manager = ConfigManager()
        self.data_manager = DataManager()
        self.logger_manager: NanoprobeLogger = setup_project_logging(self.config_manager)

        # Компоненты симуляции
        self.spm_controller = SPMController()
        self.image_processor = ImageProcessor()
        self.sstv_decoder = SSTVDecoder()

        # Состояния
        self.active_simulations = {}
        self.results_cache = {}

        # Настраиваем маршруты
        self.setup_routes()


    def setup_routes(self) -> None:
        """Настраивает маршруты API"""
        # Маршруты для СЗМ симуляции
        self.app.add_url_rule('/api/spm/create-surface', 'create_surface', self.create_surface, methods=['POST'])
        self.app.add_url_rule('/api/spm/scan-surface', 'scan_surface', self.scan_surface, methods=['POST'])
        self.app.add_url_rule('/api/spm/get-results', 'get_spm_results', self.get_spm_results, methods=['GET'])

        # Маршруты для анализа изображений
        self.app.add_url_rule('/api/image/process', 'process_image', self.process_image, methods=['POST'])
        self.app.add_url_rule('/api/image/analyze', 'analyze_image', self.analyze_image, methods=['POST'])
        self.app.add_url_rule('/api/image/get-results', 'get_image_results', self.get_image_results, methods=['GET'])

        # Маршруты для SSTV
        self.app.add_url_rule('/api/sstv/decode', 'decode_sstv', self.decode_sstv, methods=['POST'])
        self.app.add_url_rule('/api/sstv/get-results', 'get_sstv_results', self.get_sstv_results, methods=['GET'])

        # Маршруты для управления симуляцией
        self.app.add_url_rule('/api/simulation/start', 'start_simulation', self.start_simulation, methods=['POST'])
        self.app.add_url_rule('/api/simulation/status/<simulation_id>', 'get_simulation_status', self.get_simulation_status, methods=['GET'])
        self.app.add_url_rule('/api/simulation/stop/<simulation_id>', 'stop_simulation', self.stop_simulation, methods=['POST'])
        self.app.add_url_rule('/api/simulation/results/<simulation_id>', 'get_simulation_results', self.get_simulation_results, methods=['GET'])

        # Маршруты для данных
        self.app.add_url_rule('/api/data/upload', 'upload_data', self.upload_data, methods=['POST'])
        self.app.add_url_rule('/api/data/download/<filename>', 'download_data', self.download_data, methods=['GET'])
        self.app.add_url_rule('/api/data/list', 'list_data', self.list_data, methods=['GET'])

        # Маршрут для получения информации о системе
        self.app.add_url_rule('/api/system/info', 'get_system_info', self.get_system_info, methods=['GET'])
        self.app.add_url_rule('/api/system/status', 'get_system_status', self.get_system_status, methods=['GET'])


    def create_surface(self) -> Tuple[Response, int]:
        """
        Создает новую поверхность для симуляции.

        Returns:
            JSON ответ с результатом операции
        """
        try:
            data = request.get_json()

            # Валидация входных данных
            is_valid, error_message = DataValidator.validate_surface_params(data)
            if not is_valid:
                self.logger_manager.log_spm_event(f"Ошибка валидации: {error_message}", "WARNING")
                return jsonify(ResponseBuilder.validation_error("surface_params", error_message)), 400

            width = data.get('width', 50)
            height = data.get('height', 50)
            surface_type = data.get('type', 'random')

            # Создаем поверхность
            surface = SurfaceModel(width, height)

            # Сохраняем результат
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"surface_{timestamp}.txt"
            surface.saveToFile(filename)

            # Логируем действие
            self.logger_manager.log_spm_event(f"Создана поверхность {filename}, размер: {width}x{height}", "INFO")

            return jsonify(ResponseBuilder.success({
                'surface_id': filename,
                'dimensions': {'width': width, 'height': height},
                'type': surface_type
            }, f"Поверхность {width}x{height} создана"))

        except ValidationError as e:
            self.logger_manager.log_spm_event(f"Ошибка валидации: {e.message}", "ERROR")
            return jsonify(ResponseBuilder.validation_error(e.field, e.message)), 400
        except Exception as e:
            self.logger_manager.log_spm_event(f"Критическая ошибка: {str(e)}", "ERROR")
            return jsonify(ResponseBuilder.error(str(e), "INTERNAL_ERROR")), 500


    def scan_surface(self) -> Tuple[Response, int]:
        """
        Выполняет сканирование поверхности СЗМ.

        Returns:
            JSON ответ с результатом сканирования
        """
        try:
            data = request.get_json()

            # Валидация входных данных
            is_valid, error_message = DataValidator.validate_scan_params(data)
            if not is_valid:
                return jsonify(ResponseBuilder.validation_error("scan_params", error_message)), 400

            surface_id = data.get('surface_id')
            scan_speed = data.get('scan_speed', 1.0)

            # Загружаем поверхность (в реальной реализации)
            # surface = load_surface(surface_id)
            # self.spm_controller.setSurface(surface)

            # Выполняем сканирование
            # self.spm_controller.scanSurface()

            # Для демонстрации создаем фиктивные результаты
            width, height = 50, 50
            scan_results = np.random.rand(height, width).tolist()

            # Сохраняем результаты
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_filename = f"scan_results_{timestamp}.json"

            with open(results_filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'scan_results': scan_results,
                    'surface_id': surface_id,
                    'scan_speed': scan_speed,
                    'timestamp': timestamp
                }, f, ensure_ascii=False)

            # Логируем действие
            self.logger_manager.log_spm_event(f"Выполнено сканирование поверхности {surface_id}", "INFO")

            return jsonify(ResponseBuilder.success({
                'results_file': results_filename,
                'surface_id': surface_id,
                'scan_speed': scan_speed
            }, f"Сканирование {surface_id} завершено"))

        except ValidationError as e:
            return jsonify(ResponseBuilder.validation_error(e.field, e.message)), 400
        except Exception as e:
            self.logger_manager.log_spm_event(f"Ошибка сканирования: {str(e)}", "ERROR")
            return jsonify(ResponseBuilder.error(str(e), "SCAN_ERROR")), 500


    def process_image(self) -> Tuple[Response, int]:
        """
        Обрабатывает изображение.

        Returns:
            JSON ответ с результатом обработки
        """
        try:
            data = request.get_json()

            # Валидация входных данных
            is_valid, error_message = DataValidator.validate_image_data(data)
            if not is_valid:
                return jsonify(ResponseBuilder.validation_error("image_data", error_message)), 400

            image_data = data.get('image_data')  # base64 encoded
            filter_type = data.get('filter', 'gaussian')

            # Декодируем изображение из base64
            # image = self.decode_base64_image(image_data)

            # Применяем фильтр
            # processed_image = self.image_processor.apply_noise_reduction(filter_type)

            # Для демонстрации возвращаем фиктивный результат
            processed_data = np.random.rand(100, 100).tolist()

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_filename = f"processed_image_{timestamp}.json"

            with open(results_filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'processed_image': processed_data,
                    'filter_type': filter_type,
                    'timestamp': timestamp
                }, f, ensure_ascii=False)

            # Логируем действие
            self.logger_manager.log_analyzer_event(f"Обработано изображение с фильтром {filter_type}", "INFO")

            return jsonify(ResponseBuilder.success({
                'results_file': results_filename,
                'filter_type': filter_type
            }, f"Изображение обработано фильтром {filter_type}"))

        except ValidationError as e:
            return jsonify(ResponseBuilder.validation_error(e.field, e.message)), 400
        except Exception as e:
            self.logger_manager.log_analyzer_event(f"Ошибка обработки изображения: {str(e)}", "ERROR")
            return jsonify(ResponseBuilder.error(str(e), "IMAGE_PROCESSING_ERROR")), 500


    def decode_sstv(self) -> Tuple[Response, int]:
        """
        Декодирует SSTV сигнал.

        Returns:
            JSON ответ с результатом декодирования
        """
        try:
            data = request.get_json()

            # Валидация входных данных
            is_valid, error_message = DataValidator.validate_audio_data(data)
            if not is_valid:
                return jsonify(ResponseBuilder.validation_error("audio_data", error_message)), 400

            audio_data = data.get('audio_data')  # base64 encoded
            mode = data.get('mode', 'MartinM1')

            # Декодируем аудио из base64
            # audio = self.decode_base64_audio(audio_data)

            # Декодируем SSTV
            # decoded_image = self.sstv_decoder.decode_from_audio(audio)

            # Для демонстрации возвращаем фиктивное изображение
            decoded_image_data = np.random.rand(320, 240, 3).tolist()

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_filename = f"sstv_decoded_{timestamp}.json"

            with open(results_filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'decoded_image': decoded_image_data,
                    'mode': mode,
                    'timestamp': timestamp
                }, f, ensure_ascii=False)

            # Логируем действие
            self.logger_manager.log_sstv_event(f"Декодирован SSTV сигнал в режиме {mode}", "INFO")

            return jsonify({
                'status': 'success',
                'results_file': results_filename,
                'timestamp': timestamp
            })

        except Exception as e:
            self.logger_manager.log_sstv_event(f"Ошибка декодирования SSTV: {e}", "ERROR")
            return jsonify({'status': 'error', 'message': str(e)}), 500


    def start_simulation(self) -> Tuple[Response, int]:
        """
        Запускает новую симуляцию

        Returns:
            JSON ответ с результатом запуска
        """
        try:
            data = request.get_json()

            simulation_id = str(uuid.uuid4())
            simulation_type = data.get('type', 'spm')
            parameters = data.get('parameters', {})

            # Создаем запись о симуляции
            self.active_simulations[simulation_id] = {
                'type': simulation_type,
                'parameters': parameters,
                'status': 'running',
                'start_time': datetime.now().isoformat(),
                'progress': 0
            }

            # Логируем начало симуляции
            self.logger_manager.log_simulation_event(f"Начата симуляция {simulation_id}, тип: {simulation_type}", "INFO")

            # Запускаем симуляцию в отдельном потоке
            thread = threading.Thread(
                target=self._run_simulation,
                args=(simulation_id, simulation_type, parameters)
            )
            thread.daemon = True
            thread.start()

            return jsonify({
                'status': 'success',
                'simulation_id': simulation_id,
                'message': f'Simulation {simulation_type} started'
            })

        except Exception as e:
            self.logger_manager.log_simulation_event(f"Ошибка запуска симуляции: {e}", "ERROR")
            return jsonify({'status': 'error', 'message': str(e)}), 500


    def _run_simulation(self, simulation_id: str, simulation_type: str, parameters: Dict[str, Any]) -> None:
        """
        Выполняет симуляцию в отдельном потоке

        Args:
            simulation_id: ID симуляции
            simulation_type: Тип симуляции
            parameters: Параметры симуляции
        """
        try:
            # Обновляем статус
            self.active_simulations[simulation_id]['progress'] = 10

            # Симуляция выполнения
            for i in range(10):
                time.sleep(0.5)  # Имитация работы
                progress = int(((i + 1) / 10) * 80) + 10
                self.active_simulations[simulation_id]['progress'] = progress

            # Генерируем результаты
            results = {
                'simulation_id': simulation_id,
                'type': simulation_type,
                'parameters': parameters,
                'results': self._generate_simulation_results(simulation_type, parameters),
                'end_time': datetime.now().isoformat()
            }

            # Сохраняем результаты
            self.results_cache[simulation_id] = results

            # Обновляем статус
            self.active_simulations[simulation_id]['status'] = 'completed'
            self.active_simulations[simulation_id]['progress'] = 100
            self.active_simulations[simulation_id]['end_time'] = datetime.now().isoformat()

            # Логируем завершение
            self.logger_manager.log_simulation_event(f"Симуляция {simulation_id} завершена", "INFO")

        except Exception as e:
            self.logger_manager.log_simulation_event(f"Ошибка в симуляции {simulation_id}: {e}", "ERROR")
            self.active_simulations[simulation_id]['status'] = 'error'
            self.active_simulations[simulation_id]['error'] = str(e)


    def _generate_simulation_results(self, simulation_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Генерирует результаты симуляции

        Args:
            simulation_type: Тип симуляции
            parameters: Параметры симуляции

        Returns:
            Словарь с результатами
        """
        if simulation_type == 'spm':
            return {
                'surface_analysis': {
                    'roughness': np.random.uniform(0.1, 0.5),
                    'features_detected': np.random.randint(5, 20)
                },
                'scan_quality': np.random.uniform(0.8, 1.0)
            }
        elif simulation_type == 'image_analysis':
            return {
                'image_quality': np.random.uniform(0.7, 1.0),
                'defects_found': np.random.randint(0, 10)
            }
        elif simulation_type == 'sstv':
            return {
                'signal_quality': np.random.uniform(0.6, 1.0),
                'decoding_success': True
            }
        else:
            return {'general_results': 'Simulation completed successfully'}


    def get_simulation_status(self, simulation_id: str) -> Tuple[Response, int]:
        """
        Возвращает статус симуляции

        Args:
            simulation_id: ID симуляции

        Returns:
            JSON ответ со статусом симуляции
        """
        if simulation_id not in self.active_simulations:
            return jsonify({'status': 'error', 'message': 'Simulation not found'}), 404

        return jsonify(self.active_simulations[simulation_id])


    def get_simulation_results(self, simulation_id: str) -> Tuple[Response, int]:
        """
        Возвращает результаты симуляции

        Args:
            simulation_id: ID симуляции

        Returns:
            JSON ответ с результатами симуляции
        """
        if simulation_id in self.results_cache:
            return jsonify(self.results_cache[simulation_id])
        else:
            # Проверяем, завершена ли симуляция
            if simulation_id in self.active_simulations:
                status = self.active_simulations[simulation_id]['status']
                if status == 'completed':
                    return jsonify({'status': 'pending', 'message': 'Results not yet available'})
                else:
                    return jsonify({'status': status, 'message': f'Simulation is {status}'})
            else:
                return jsonify({'status': 'error', 'message': 'Simulation not found'}), 404


    def upload_data(self) -> Tuple[Response, int]:
        """
        Загружает данные в систему

        Returns:
            JSON ответ с результатом загрузки
        """
        try:
            if 'file' not in request.files:
                return jsonify({'status': 'error', 'message': 'No file provided'}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({'status': 'error', 'message': 'No file selected'}), 400

            # Сохраняем файл
            upload_path = Path('uploads') / file.filename
            upload_path.parent.mkdir(exist_ok=True)
            file.save(str(upload_path))

            # Логируем загрузку
            self.logger_manager.log_system_event(f"Загружен файл: {file.filename}", "INFO")

            return jsonify({
                'status': 'success',
                'filename': file.filename,
                'path': str(upload_path)
            })

        except Exception as e:
            self.logger_manager.log_system_event(f"Ошибка загрузки данных: {e}", "ERROR")
            return jsonify({'status': 'error', 'message': str(e)}), 500


    def list_data(self) -> Tuple[Response, int]:
        """
        Возвращает список доступных данных

        Returns:
            JSON ответ со списком файлов
        """
        try:
            # Собираем список файлов из различных директорий
            data_dirs = ['data', 'output', 'uploads']
            file_list = []

            for dir_name in data_dirs:
                dir_path = Path(dir_name)
                if dir_path.exists():
                    for file_path in dir_path.rglob('*'):
                        if file_path.is_file():
                            file_list.append({
                                'name': file_path.name,
                                'path': str(file_path),
                                'size': file_path.stat().st_size,
                                'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                            })

            return jsonify({
                'status': 'success',
                'files': file_list,
                'total_count': len(file_list)
            })

        except Exception as e:
            self.logger_manager.log_system_event(f"Ошибка получения списка данных: {e}", "ERROR")
            return jsonify({'status': 'error', 'message': str(e)}), 500


    def get_system_info(self) -> Response:
        """
        Возвращает информацию о системе

        Returns:
            JSON ответ с информацией о системе
        """
        import platform
        import psutil

        info = {
            'system': {
                'platform': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'architecture': platform.architecture()[0],
                'processor': platform.processor()
            },
            'hardware': {
                'cpu_count': psutil.cpu_count(),
                'cpu_percent': psutil.cpu_percent(),
                'memory_total': psutil.virtual_memory().total,
                'memory_available': psutil.virtual_memory().available,
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else 0
            },
            'software': {
                'python_version': platform.python_version(),
                'nanoprobe_api_version': '1.0.0'
            },
            'status': 'operational',
            'uptime': time.time()  # Время работы с момента запуска
        }

        return jsonify(info)


    def get_system_status(self) -> Response:
        """
        Возвращает статус системы

        Returns:
            JSON ответ со статусом системы
        """
        active_count = len([s for s in self.active_simulations.values() if s['status'] == 'running'])

        status = {
            'overall_status': 'operational',
            'active_simulations': active_count,
            'total_simulations': len(self.active_simulations),
            'cached_results': len(self.results_cache),
            'components': {
                'spm_controller': 'ready',
                'image_processor': 'ready',
                'sstv_decoder': 'ready'
            }
        }

        return jsonify(status)


    def run(self, host: str = 'localhost', port: int = 5000, debug: bool = False) -> None:
        """
        Запускает API сервер

        Args:
            host: Хост для прослушивания
            port: Порт для прослушивания
            debug: Режим отладки
        """
        print(f"Запуск API сервера на {host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)

def main():
    """Главная функция для запуска API сервера"""
    print("=== API ИНТЕРФЕЙС ПРОЕКТА ===")

    # Создаем API интерфейс
    api = NanoprobeAPI()

    print("✓ API интерфейс инициализирован")
    print("Доступные маршруты:")
    print("  - /api/spm/create-surface - Создание поверхности")
    print("  - /api/spm/scan-surface - Сканирование поверхности")
    print("  - /api/image/process - Обработка изображений")
    print("  - /api/sstv/decode - Декодирование SSTV")
    print("  - /api/simulation/start - Запуск симуляции")
    print("  - /api/system/info - Информация о системе")
    print("  - /api/system/status - Статус системы")

    # Для демонстрации запускаем сервер
    # api.run(debug=True)  # Закомментировано для безопасности

if __name__ == "__main__":
    main()

