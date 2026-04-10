"""
Модуль интеграции Flask и FastAPI приложений
Обеспечивает взаимодействие между Flask веб-интерфейсом и FastAPI REST API
"""

import logging
import os
import requests
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import jwt
import json

logger = logging.getLogger(__name__)


class FlaskFastAPIIntegration:
    """
    Класс интеграции между Flask и FastAPI приложениями
    Предоставляет единый интерфейс для взаимодействия с обоими приложениями
    """

    def __init__(
        self,
        fastapi_url: str = "http://localhost:8000",
        flask_url: str = "http://localhost:5000",
        jwt_secret: str = None
    ):
        """
        Инициализация интеграции

        Args:
            fastapi_url: URL FastAPI приложения
            flask_url: URL Flask приложения
            jwt_secret: Секретный ключ для JWT токенов
        """
        self.fastapi_url = fastapi_url.rstrip('/')
        self.flask_url = flask_url.rstrip('/')
        self.jwt_secret = jwt_secret or os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
        self._token_cache: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

    # ==================== Аутентификация ====================

    def login(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Аутентификация через FastAPI и получение токена

        Args:
            username: Имя пользователя
            password: Пароль

        Returns:
            Токены доступа или None при ошибке
        """
        try:
            response = requests.post(
                f"{self.fastapi_url}/api/v1/auth/login",
                data={"username": username, "password": password},
                timeout=10
            )

            if response.status_code == 200:
                tokens = response.json()
                self._token_cache = tokens.get("access_token")

                # Декодирование токена для получения срока действия
                try:
                    payload = jwt.decode(
                        self._token_cache,
                        self.jwt_secret,
                        algorithms=["HS256"]
                    )
                    self._token_expiry = datetime.fromtimestamp(payload.get("exp", 0))
                except (jwt.PyJWTError, KeyError, TypeError, ValueError):
                    self._token_expiry = datetime.now(timezone.utc)

                return tokens
            return None
        except requests.RequestException as e:
            logger.error(f"Ошибка аутентификации: {e}")
            return None

    def get_token(self) -> Optional[str]:
        """Получение текущего access токена"""
        if self._token_cache and self._token_expiry:
            if datetime.now(timezone.utc) < self._token_expiry:
                return self._token_cache
        return None

    def get_auth_headers(self) -> Dict[str, str]:
        """Получение заголовков для авторизованных запросов"""
        token = self.get_token()
        if token:
            return {"Authorization": f"Bearer {token}"}
        return {}

    # ==================== FastAPI вызовы ====================

    def call_fastapi(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict] = None,
        use_auth: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Вызов FastAPI endpoint

        Args:
            endpoint: Endpoint URL (например, "/api/v1/scans")
            method: HTTP метод
            data: Данные для запроса
            use_auth: Использовать ли аутентификацию

        Returns:
            Ответ API или None при ошибке
        """
        try:
            url = f"{self.fastapi_url}{endpoint}"
            headers = self.get_auth_headers() if use_auth else {}

            if method.upper() == "GET":
                response = requests.get(url, headers=headers, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(url, json=data, headers=headers, timeout=30)
            elif method.upper() == "PUT":
                response = requests.put(url, json=data, headers=headers, timeout=30)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=headers, timeout=30)
            else:
                raise ValueError(f"Неподдерживаемый метод: {method}")

            if response.status_code in (200, 201, 204):
                try:
                    return response.json()
                except json.JSONDecodeError:
                    return {"status": "success"}
            else:
                logger.error(f"Ошибка FastAPI [{response.status_code}]: {response.text}")
                return None

        except requests.RequestException as e:
            logger.error(f"Ошибка вызова FastAPI: {e}")
            return None

    # ==================== Flask вызовы ====================

    def call_flask(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Выов Flask endpoint

        Args:
            endpoint: Endpoint URL (например, "/api/database/stats")
            method: HTTP метод
            data: Данные для запроса

        Returns:
            Ответ API или None при ошибке
        """
        try:
            url = f"{self.flask_url}{endpoint}"
            headers = self.get_auth_headers()

            if method.upper() == "GET":
                response = requests.get(url, headers=headers, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(url, json=data, headers=headers, timeout=30)
            else:
                raise ValueError(f"Неподдерживаемый метод: {method}")

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Ошибка Flask [{response.status_code}]: {response.text}")
                return None

        except requests.RequestException as e:
            logger.error(f"Ошибка вызова Flask: {e}")
            return None

    # ==================== Сканирования ====================

    def get_scans(self, scan_type: Optional[str] = None, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Получение списка сканирований из FastAPI"""
        params = {"limit": limit, "offset": offset}
        if scan_type:
            params["scan_type"] = scan_type

        result = self.call_fastapi("/api/v1/scans", method="GET", data=params)
        return result.get("items", []) if result else []

    def get_scan_by_id(self, scan_id: int) -> Optional[Dict]:
        """Получение сканирования по ID из FastAPI"""
        return self.call_fastapi(f"/api/v1/scans/{scan_id}")

    def create_scan(self, scan_data: Dict) -> Optional[Dict]:
        """Создание нового сканирования через FastAPI"""
        return self.call_fastapi("/api/v1/scans", method="POST", data=scan_data)

    def delete_scan(self, scan_id: int) -> bool:
        """Удаление сканирования через FastAPI"""
        result = self.call_fastapi(f"/api/v1/scans/{scan_id}", method="DELETE")
        return result is not None

    # ==================== Симуляции ====================

    def get_simulations(self, status: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Получение списка симуляций из FastAPI"""
        params = {"limit": limit}
        if status:
            params["status"] = status

        result = self.call_fastapi("/api/v1/simulations", method="GET", data=params)
        return result.get("items", []) if result else []

    def create_simulation(self, sim_data: Dict) -> Optional[Dict]:
        """Создание новой симуляции через FastAPI"""
        return self.call_fastapi("/api/v1/simulations", method="POST", data=sim_data)

    def update_simulation(self, simulation_id: str, status: str) -> Optional[Dict]:
        """Обновление статуса симуляции"""
        return self.call_fastapi(f"/api/v1/simulations/{simulation_id}", method="PUT", data={"status": status})

    # ==================== Анализ дефектов ====================

    def analyze_defects(self, image_path: str, model_name: str = "isolation_forest") -> Optional[Dict]:
        """
        Анализ дефектов через FastAPI

        Args:
            image_path: Путь к изображению
            model_name: Название модели

        Returns:
            Результаты анализа
        """
        return self.call_fastapi(
            "/api/v1/analysis/defects",
            method="POST",
            data={"image_path": image_path, "model_name": model_name}
        )

    def get_defect_history(self, limit: int = 50) -> List[Dict]:
        """Получение истории анализов дефектов"""
        result = self.call_fastapi(f"/api/v1/analysis/history?limit={limit}")
        return result.get("items", []) if result else []

    # ==================== Сравнение поверхностей ====================

    def compare_surfaces(self, image1_path: str, image2_path: str) -> Optional[Dict]:
        """
        Сравнение поверхностей через FastAPI

        Args:
            image1_path: Путь к первому изображению
            image2_path: Путь ко второму изображению

        Returns:
            Результаты сравнения
        """
        return self.call_fastapi(
            "/api/v1/comparison/surfaces",
            method="POST",
            data={"image1_path": image1_path, "image2_path": image2_path}
        )

    def get_comparison_history(self, limit: int = 50) -> List[Dict]:
        """Получение истории сравнений поверхностей"""
        result = self.call_fastapi(f"/api/v1/comparison/history?limit={limit}")
        return result.get("items", []) if result else []

    # ==================== PDF отчёты ====================

    def generate_pdf_report(
        self,
        report_type: str,
        title: str,
        source_ids: List[int] = None
    ) -> Optional[Dict]:
        """
        Генерация PDF отчёта через FastAPI

        Args:
            report_type: Тип отчёта (surface, defect, comparison, simulation)
            title: Заголовок отчёта
            source_ids: ID исходных данных

        Returns:
            Информация о созданном отчёте
        """
        return self.call_fastapi(
            "/api/v1/reports/generate",
            method="POST",
            data={
                "report_type": report_type,
                "title": title,
                "source_ids": source_ids or []
            }
        )

    def get_reports(self, limit: int = 50) -> List[Dict]:
        """Получение списка отчётов"""
        result = self.call_fastapi(f"/api/v1/reports?limit={limit}")
        return result.get("items", []) if result else []

    # ==================== Flask специфичные вызовы ====================

    def get_flask_system_info(self) -> Optional[Dict]:
        """Получение информации о системе из Flask"""
        return self.call_flask("/api/system_info")

    def get_flask_performance_data(self) -> Optional[Dict]:
        """Получение данных о производительности из Flask"""
        return self.call_flask("/api/performance_data")

    def get_flask_component_status(self) -> Optional[Dict]:
        """Получение статуса компонентов из Flask"""
        return self.call_flask("/api/component_status")

    def get_flask_logs(self) -> Optional[Dict]:
        """Получение логов из Flask"""
        return self.call_flask("/api/logs")

    def get_flask_config(self) -> Optional[Dict]:
        """Получение конфигурации из Flask"""
        return self.call_flask("/api/config")

    def update_flask_config(self, new_config: Dict) -> Optional[Dict]:
        """Обновление конфигурации во Flask"""
        return self.call_flask("/api/config", method="POST", data=new_config)

    def get_flask_database_stats(self) -> Optional[Dict]:
        """Получение статистики БД из Flask"""
        return self.call_flask("/api/database/stats")

    def get_flask_scans(self, scan_type: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Получение сканирований из Flask"""
        params = {"limit": limit}
        if scan_type:
            params["type"] = scan_type

        result = self.call_flask("/api/database/scans", data=params)
        return result.get("scans", []) if result else []

    def get_flask_simulations(self, status: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Получение симуляций из Flask"""
        params = {"limit": limit}
        if status:
            params["status"] = status

        result = self.call_flask("/api/database/simulations", data=params)
        return result.get("simulations", []) if result else []

    def compare_surfaces_flask(self, image1_path: str, image2_path: str) -> Optional[Dict]:
        """Сравнение поверхностей через Flask"""
        return self.call_flask(
            "/api/surface/compare",
            method="POST",
            data={"image1_path": image1_path, "image2_path": image2_path}
        )

    def analyze_defects_flask(self, image_path: str, model_name: str = "isolation_forest") -> Optional[Dict]:
        """Анализ дефектов через Flask"""
        return self.call_flask(
            "/api/defect/analyze",
            method="POST",
            data={"image_path": image_path, "model_name": model_name}
        )

    # ==================== Утилиты ====================

    def health_check(self) -> Dict[str, Any]:
        """
        Проверка здоровья обоих приложений

        Returns:
            Статус обоих приложений
        """
        result = {
            "fastapi": {"status": "unknown", "response_time_ms": None},
            "flask": {"status": "unknown", "response_time_ms": None},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Проверка FastAPI
        try:
            start = datetime.now(timezone.utc)
            response = requests.get(f"{self.fastapi_url}/health", timeout=5)
            elapsed = (datetime.now(timezone.utc) - start).total_seconds() * 1000

            if response.status_code == 200:
                result["fastapi"] = {"status": "healthy", "response_time_ms": round(elapsed, 2)}
            else:
                result["fastapi"] = {"status": "unhealthy", "response_time_ms": round(elapsed, 2)}
        except requests.RequestException as e:
            result["fastapi"] = {"status": "unreachable", "error": str(e)}

        # Проверка Flask (через системную информацию)
        try:
            start = datetime.now(timezone.utc)
            response = requests.get(f"{self.flask_url}/api/system_info", timeout=5)
            elapsed = (datetime.now(timezone.utc) - start).total_seconds() * 1000

            if response.status_code == 200:
                result["flask"] = {"status": "healthy", "response_time_ms": round(elapsed, 2)}
            else:
                result["flask"] = {"status": "unhealthy", "response_time_ms": round(elapsed, 2)}
        except requests.RequestException as e:
            result["flask"] = {"status": "unreachable", "error": str(e)}

        return result

    def sync_data(self, data_type: str = "all") -> Dict[str, Any]:
        """
        Синхронизация данных между Flask и FastAPI

        Args:
            data_type: Тип данных для синхронизации (scans, simulations, all)

        Returns:
            Результаты синхронизации
        """
        result = {
            "synced": [],
            "errors": [],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Оба приложения используют одну БД, поэтому синхронизация не требуется
        # Этот метод для проверки консистентности данных

        if data_type in ("scans", "all"):
            try:
                fastapi_scans = self.get_scans(limit=10)
                flask_scans = self.get_flask_scans(limit=10)

                if len(fastapi_scans) == len(flask_scans):
                    result["synced"].append("scans")
                else:
                    result["errors"].append(f"Несовпадение количества сканирований: FastAPI={len(fastapi_scans)}, Flask={len(flask_scans)}")
            except Exception as e:
                result["errors"].append(f"Ошибка синхронизации сканирований: {e}")

        if data_type in ("simulations", "all"):
            try:
                fastapi_sims = self.get_simulations(limit=10)
                flask_sims = self.get_flask_simulations(limit=10)

                if len(fastapi_sims) == len(flask_sims):
                    result["synced"].append("simulations")
                else:
                    result["errors"].append(f"Несовпадение количества симуляций: FastAPI={len(fastapi_sims)}, Flask={len(flask_sims)}")
            except Exception as e:
                result["errors"].append(f"Ошибка синхронизации симуляций: {e}")

        return result


# Глобальный экземпляр интеграции
_integration_instance: Optional[FlaskFastAPIIntegration] = None


def get_integration(
    fastapi_url: str = None,
    flask_url: str = None,
    jwt_secret: str = None
) -> FlaskFastAPIIntegration:
    """
    Получение экземпляра интеграции (singleton)

    Args:
        fastapi_url: URL FastAPI (опционально)
        flask_url: URL Flask (опционально)
        jwt_secret: JWT секрет (опционально)

    Returns:
        Экземпляр интеграции
    """
    global _integration_instance

    if _integration_instance is None:
        _integration_instance = FlaskFastAPIIntegration(
            fastapi_url=fastapi_url or "http://localhost:8000",
            flask_url=flask_url or "http://localhost:5000",
            jwt_secret=jwt_secret or os.getenv("JWT_SECRET")
        )

    return _integration_instance


# Функции быстрого доступа
def login(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Быстрый вход через интеграцию"""
    return get_integration().login(username, password)


def health_check() -> Dict[str, Any]:
    """Быстрая проверка здоровья"""
    return get_integration().health_check()


def get_scans(limit: int = 100) -> List[Dict]:
    """Быстрое получение сканирований"""
    return get_integration().get_scans(limit=limit)


def get_simulations(limit: int = 50) -> List[Dict]:
    """Быстрое получение симуляций"""
    return get_integration().get_simulations(limit=limit)


if __name__ == "__main__":
    # Тестирование интеграции
    print("=== Тестирование интеграции Flask + FastAPI ===\n")

    integration = FlaskFastAPIIntegration()

    # Проверка здоровья
    print("1. Проверка здоровья приложений...")
    health = integration.health_check()
    print(f"   FastAPI: {health['fastapi']['status']}")
    print(f"   Flask: {health['flask']['status']}")

    # Тест аутентификации (требуется valid user)
    print("\n2. Тест аутентификации...")
    print("   (пропустите, если нет тестового пользователя)")
    # tokens = integration.login("admin", "admin123")
    # if tokens:
    #     print(f"   ✓ Токен получен: {tokens['access_token'][:50]}...")

    # Тест получения сканирований
    print("\n3. Тест получения сканирований...")
    scans = integration.get_scans(limit=5)
    print(f"   Получено сканирований: {len(scans)}")

    # Тест получения симуляций
    print("\n4. Тест получения симуляций...")
    simulations = integration.get_simulations(limit=5)
    print(f"   Получено симуляций: {len(simulations)}")

    # Синхронизация
    print("\n5. Проверка синхронизации...")
    sync_result = integration.sync_data()
    print(f"   Синхронизировано: {sync_result['synced']}")
    if sync_result['errors']:
        print(f"   Ошибки: {sync_result['errors']}")

    print("\n=== Тестирование завершено ===")
