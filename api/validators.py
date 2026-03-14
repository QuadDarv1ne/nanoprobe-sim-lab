"""Модуль валидации данных для API проекта Nanoprobe Simulation Lab."""

from typing import Dict, Any, Optional


class ValidationError(Exception):
    """Исключение валидации."""

    def __init__(self, message: str, field: str = None):
        self.message = message
        self.field = field
        super().__init__(self.message)


class DataValidator:
    """Валидатор входных данных API."""

    @staticmethod
    def validate_surface_params(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Валидирует параметры создания поверхности."""
        if not data:
            return True, None  # Пустые данные = используем значения по умолчанию

        width = data.get('width', 50)
        height = data.get('height', 50)
        surface_type = data.get('type', 'random')

        if not isinstance(width, int) or width < 1 or width > 1000:
            return False, "width должен быть целым числом от 1 до 1000"

        if not isinstance(height, int) or height < 1 or height > 1000:
            return False, "height должен быть целым числом от 1 до 1000"

        valid_types = ['random', 'flat', 'gaussian', 'periodic']
        if surface_type not in valid_types:
            return False, f"type должен быть одним из: {valid_types}"

        return True, None

    @staticmethod
    def validate_scan_params(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Валидирует параметры сканирования."""
        if not data:
            return False, "Отсутствуют данные запроса"

        if 'surface_id' not in data:
            return False, "surface_id обязателен"

        if not isinstance(data['surface_id'], str):
            return False, "surface_id должен быть строкой"

        scan_speed = data.get('scan_speed', 1.0)
        if not isinstance(scan_speed, (int, float)) or scan_speed <= 0:
            return False, "scan_speed должен быть положительным числом"

        return True, None

    @staticmethod
    def validate_image_data(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Валидирует данные изображения."""
        if not data:
            return False, "Отсутствуют данные запроса"

        if 'image_data' not in data:
            return False, "image_data обязателен"

        if not isinstance(data['image_data'], str):
            return False, "image_data должен быть строкой (base64)"

        valid_filters = ['gaussian', 'median', 'bilateral', 'none']
        filter_type = data.get('filter', 'gaussian')
        if filter_type not in valid_filters:
            return False, f"filter должен быть одним из: {valid_filters}"

        return True, None

    @staticmethod
    def validate_audio_data(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Валидирует аудио данные для SSTV."""
        if not data:
            return False, "Отсутствуют данные запроса"

        if 'audio_data' not in data and 'audio_path' not in data:
            return False, "audio_data или audio_path обязательны"

        return True, None

    @staticmethod
    def validate_simulation_params(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Валидирует параметры симуляции."""
        if not data:
            return False, "Отсутствуют данные запроса"

        if 'simulation_type' not in data:
            return False, "simulation_type обязателен"

        valid_types = ['spm', 'image', 'sstv', 'combined']
        if data['simulation_type'] not in valid_types:
            return False, f"simulation_type должен быть одним из: {valid_types}"

        duration = data.get('duration', 60)
        if not isinstance(duration, (int, float)) or duration <= 0 or duration > 3600:
            return False, "duration должен быть числом от 1 до 3600 секунд"

        return True, None

    @staticmethod
    def validate_file_upload(file) -> Tuple[bool, Optional[str]]:
        """Валидирует загружаемый файл."""
        if not file:
            return False, "Файл не предоставлен"

        allowed_extensions = {'.txt', '.json', '.csv', '.png', '.jpg', '.jpeg', '.bmp', '.wav', '.mp3'}
        filename = file.filename.lower()
        ext = '.' + filename.rsplit('.', 1)[-1] if '.' in filename else ''

        if ext not in allowed_extensions:
            return False, f"Недопустимый тип файла. Разрешены: {allowed_extensions}"

        return True, None


class ResponseBuilder:
    """Конструктор ответов API."""

    @staticmethod
    def success(data: Any = None, message: str = None) -> Dict[str, Any]:
        """Создает успешный ответ."""
        response = {
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        }
        if data:
            response['data'] = data
        if message:
            response['message'] = message
        return response

    @staticmethod
    def error(message: str, error_code: str = None, details: Dict = None) -> Dict[str, Any]:
        """Создает ответ об ошибке."""
        response = {
            'status': 'error',
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        if error_code:
            response['error_code'] = error_code
        if details:
            response['details'] = details
        return response

    @staticmethod
    def validation_error(field: str, message: str) -> Dict[str, Any]:
        """Создает ответ об ошибке валидации."""
        return ResponseBuilder.error(
            message=f"Ошибка валидации: {field} - {message}",
            error_code="VALIDATION_ERROR",
            details={'field': field}
        )
