"""
Reverse proxy для интеграции Flask и FastAPI
Позволяет Flask приложению проксировать запросы к FastAPI
"""

import logging
import os
import requests
from flask import Blueprint, request, jsonify, session

logger = logging.getLogger(__name__)

# Blueprint для reverse proxy
api_proxy = Blueprint('api_proxy', __name__, url_prefix='/api/v1')

# Конфигурация
FASTAPI_URL = os.getenv('FASTAPI_URL', 'http://localhost:8000')

# Безопасный JWT secret (импортируем из централизованного источника)
try:
    from api.security.jwt_config import get_jwt_secret
    JWT_SECRET = get_jwt_secret()
except ImportError:
    # Fallback: используем ENV переменную, иначе генерируем безопасный ключ
    import secrets
    JWT_SECRET = os.getenv('JWT_SECRET') or secrets.token_hex(32)
    if not os.getenv('JWT_SECRET'):
        logger.warning("JWT_SECRET не установлен, используется сгенерированный ключ")


def get_token_from_session():
    """Получение токена из сессии Flask"""
    return session.get('access_token')


def get_auth_headers():
    """Получение заголовков авторизации для FastAPI"""
    token = get_token_from_session()
    if token:
        return {'Authorization': f'Bearer {token}'}
    return {}


def proxy_request(method, endpoint, **kwargs):
    """
    Универсальная функция проксирования запросов к FastAPI

    Args:
        method: HTTP метод
        endpoint: Endpoint URL
        **kwargs: Дополнительные параметры для requests

    Returns:
        Response от FastAPI
    """
    url = f"{FASTAPI_URL}{endpoint}"
    headers = get_auth_headers()

    # Добавляем заголовки из исходного запроса
    exclude_headers = ['Host', 'Content-Length', 'Cookie']
    for key, value in request.headers:
        if key not in exclude_headers:
            headers[key] = value

    try:
        # Проксирование запроса
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            params=kwargs.get('params'),
            json=kwargs.get('json'),
            data=kwargs.get('data'),
            timeout=30
        )

        return response
    except requests.RequestException:
        return None


# ==================== Auth endpoints ====================

@api_proxy.route('/auth/login', methods=['POST'])
def proxy_login():
    """
    Проксирование запроса аутентификации
    После успешного входа сохраняем токен в сессии
    """
    response = proxy_request('POST', '/api/v1/auth/login', data=request.form)

    if response is None:
        return jsonify({'error': 'FastAPI недоступен'}), 503

    if response.status_code == 200:
        tokens = response.json()
        # Сохранение токенов в сессии
        session['access_token'] = tokens.get('access_token')
        session['refresh_token'] = tokens.get('refresh_token')
        session['logged_in'] = True

        # Декодирование токена для получения информации о пользователе
        try:
            payload = jwt.decode(
                tokens.get('access_token'),
                JWT_SECRET,
                algorithms=['HS256']
            )
            session['user_id'] = payload.get('sub')
            session['username'] = payload.get('username')
        except (jwt.PyJWTError, KeyError, TypeError):
            pass

        return jsonify(tokens)

    return jsonify(response.json()), response.status_code


@api_proxy.route('/auth/refresh', methods=['POST'])
def proxy_refresh():
    """Проксирование запроса обновления токена"""
    refresh_token = session.get('refresh_token')

    if not refresh_token:
        return jsonify({'error': 'Refresh token не найден'}), 401

    response = proxy_request(
        'POST',
        '/api/v1/auth/refresh',
        json={'refresh_token': refresh_token}
    )

    if response is None:
        return jsonify({'error': 'FastAPI недоступен'}), 503

    if response.status_code == 200:
        tokens = response.json()
        session['access_token'] = tokens.get('access_token')
        if 'refresh_token' in tokens:
            session['refresh_token'] = tokens['refresh_token']
        return jsonify(tokens)

    return jsonify(response.json()), response.status_code


@api_proxy.route('/auth/logout', methods=['POST'])
def proxy_logout():
    """Выход из системы"""
    # Опционально: можно отправить запрос на FastAPI для инвалидации токена
    session.clear()
    return jsonify({'status': 'success'})


@api_proxy.route('/auth/me', methods=['GET'])
def proxy_current_user():
    """Получение информации о текущем пользователе"""
    response = proxy_request('GET', '/api/v1/auth/me')

    if response is None:
        return jsonify({'error': 'FastAPI недоступен'}), 503

    return jsonify(response.json()), response.status_code


# ==================== Scans endpoints ====================

@api_proxy.route('/scans', methods=['GET'])
def proxy_get_scans():
    """Проксирование получения списка сканирований"""
    response = proxy_request('GET', '/api/v1/scans', params=request.args)

    if response is None:
        return jsonify({'error': 'FastAPI недоступен'}), 503

    return jsonify(response.json()), response.status_code


@api_proxy.route('/scans/<int:scan_id>', methods=['GET'])
def proxy_get_scan(scan_id):
    """Проксирование получения сканирования по ID"""
    response = proxy_request('GET', f'/api/v1/scans/{scan_id}')

    if response is None:
        return jsonify({'error': 'FastAPI недоступен'}), 503

    return jsonify(response.json()), response.status_code


@api_proxy.route('/scans', methods=['POST'])
def proxy_create_scan():
    """Проксирование создания сканирования"""
    response = proxy_request('POST', '/api/v1/scans', json=request.json)

    if response is None:
        return jsonify({'error': 'FastAPI недоступен'}), 503

    return jsonify(response.json()), response.status_code


@api_proxy.route('/scans/<int:scan_id>', methods=['DELETE'])
def proxy_delete_scan(scan_id):
    """Проксирование удаления сканирования"""
    response = proxy_request('DELETE', f'/api/v1/scans/{scan_id}')

    if response is None:
        return jsonify({'error': 'FastAPI недоступен'}), 503

    if response.status_code == 204:
        return jsonify({'status': 'success'}), 204

    return jsonify(response.json()), response.status_code


@api_proxy.route('/scans/search/<query>', methods=['GET'])
def proxy_search_scans(query):
    """Проксирование поиска сканирований"""
    response = proxy_request('GET', f'/api/v1/scans/search/{query}', params=request.args)

    if response is None:
        return jsonify({'error': 'FastAPI недоступен'}), 503

    return jsonify(response.json()), response.status_code


# ==================== Simulations endpoints ====================

@api_proxy.route('/simulations', methods=['GET'])
def proxy_get_simulations():
    """Проксирование получения списка симуляций"""
    response = proxy_request('GET', '/api/v1/simulations', params=request.args)

    if response is None:
        return jsonify({'error': 'FastAPI недоступен'}), 503

    return jsonify(response.json()), response.status_code


@api_proxy.route('/simulations/<string:simulation_id>', methods=['GET'])
def proxy_get_simulation(simulation_id):
    """Проксирование получения симуляции по ID"""
    response = proxy_request('GET', f'/api/v1/simulations/{simulation_id}')

    if response is None:
        return jsonify({'error': 'FastAPI недоступен'}), 503

    return jsonify(response.json()), response.status_code


@api_proxy.route('/simulations', methods=['POST'])
def proxy_create_simulation():
    """Проксирование создания симуляции"""
    response = proxy_request('POST', '/api/v1/simulations', json=request.json)

    if response is None:
        return jsonify({'error': 'FastAPI недоступен'}), 503

    return jsonify(response.json()), response.status_code


@api_proxy.route('/simulations/<string:simulation_id>', methods=['PUT'])
def proxy_update_simulation(simulation_id):
    """Проксирование обновления симуляции"""
    response = proxy_request('PUT', f'/api/v1/simulations/{simulation_id}', json=request.json)

    if response is None:
        return jsonify({'error': 'FastAPI недоступен'}), 503

    return jsonify(response.json()), response.status_code


@api_proxy.route('/simulations/<string:simulation_id>', methods=['DELETE'])
def proxy_delete_simulation(simulation_id):
    """Проксирование удаления симуляции"""
    response = proxy_request('DELETE', f'/api/v1/simulations/{simulation_id}')

    if response is None:
        return jsonify({'error': 'FastAPI недоступен'}), 503

    return jsonify(response.json()), response.status_code


# ==================== Analysis endpoints ====================

@api_proxy.route('/analysis/defects', methods=['POST'])
def proxy_analyze_defects():
    """Проксирование анализа дефектов"""
    response = proxy_request('POST', '/api/v1/analysis/defects', json=request.json)

    if response is None:
        return jsonify({'error': 'FastAPI недоступен'}), 503

    return jsonify(response.json()), response.status_code


@api_proxy.route('/analysis/history', methods=['GET'])
def proxy_get_analysis_history():
    """Проксирование получения истории анализов"""
    response = proxy_request('GET', '/api/v1/analysis/history', params=request.args)

    if response is None:
        return jsonify({'error': 'FastAPI недоступен'}), 503

    return jsonify(response.json()), response.status_code


# ==================== Comparison endpoints ====================

@api_proxy.route('/comparison/surfaces', methods=['POST'])
def proxy_compare_surfaces():
    """Проксирование сравнения поверхностей"""
    response = proxy_request('POST', '/api/v1/comparison/surfaces', json=request.json)

    if response is None:
        return jsonify({'error': 'FastAPI недоступен'}), 503

    return jsonify(response.json()), response.status_code


@api_proxy.route('/comparison/history', methods=['GET'])
def proxy_get_comparison_history():
    """Проксирование получения истории сравнений"""
    response = proxy_request('GET', '/api/v1/comparison/history', params=request.args)

    if response is None:
        return jsonify({'error': 'FastAPI недоступен'}), 503

    return jsonify(response.json()), response.status_code


# ==================== Reports endpoints ====================

@api_proxy.route('/reports', methods=['GET'])
def proxy_get_reports():
    """Проксирование получения списка отчётов"""
    response = proxy_request('GET', '/api/v1/reports', params=request.args)

    if response is None:
        return jsonify({'error': 'FastAPI недоступен'}), 503

    return jsonify(response.json()), response.status_code


@api_proxy.route('/reports/<int:report_id>', methods=['GET'])
def proxy_get_report(report_id):
    """Проксирование получения отчёта по ID"""
    response = proxy_request('GET', f'/api/v1/reports/{report_id}')

    if response is None:
        return jsonify({'error': 'FastAPI недоступен'}), 503

    return jsonify(response.json()), response.status_code


@api_proxy.route('/reports/generate', methods=['POST'])
def proxy_generate_report():
    """Проксирование генерации отчёта"""
    response = proxy_request('POST', '/api/v1/reports/generate', json=request.json)

    if response is None:
        return jsonify({'error': 'FastAPI недоступен'}), 503

    return jsonify(response.json()), response.status_code


@api_proxy.route('/reports/<int:report_id>/download', methods=['GET'])
def proxy_download_report(report_id):
    """Проксирование скачивания отчёта"""
    response = proxy_request('GET', f'/api/v1/reports/{report_id}/download')

    if response is None:
        return jsonify({'error': 'FastAPI недоступен'}), 503

    return response.content, response.status_code, {'Content-Type': 'application/pdf'}


# ==================== Admin endpoints ====================

@api_proxy.route('/admin/stats', methods=['GET'])
def proxy_get_stats():
    """Проксирование получения общей статистики"""
    response = proxy_request('GET', '/api/v1/admin/stats')

    if response is None:
        return jsonify({'error': 'FastAPI недоступен'}), 503

    return jsonify(response.json()), response.status_code


@api_proxy.route('/admin/health', methods=['GET'])
def proxy_health_check():
    """Проксирование проверки здоровья FastAPI"""
    try:
        response = requests.get(f'{FASTAPI_URL}/health', timeout=5)
        return jsonify(response.json()), response.status_code
    except requests.RequestException:
        return jsonify({'status': 'unhealthy', 'error': 'FastAPI недоступен'}), 503


def register_proxy(app):
    """
    Регистрация blueprint reverse proxy во Flask приложении

    Args:
        app: Flask приложение
    """
    app.register_blueprint(api_proxy)
    logger.info(f"Reverse proxy registered: {FASTAPI_URL}")
