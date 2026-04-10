"""
Утилита автоматического определения свободных портов

Использует эвристический поиск для нахождения доступных портов
в заданном диапазоне с приоритетом стандартных портов проекта.
"""

import logging

# Важно: импортируем стандартные модули до добавления проекта в path
import socket
from typing import List, Optional

logger = logging.getLogger(__name__)


class PortFinder:
    """Менеджер автоматического определения свободных портов"""

    # Стандартные порты проекта (приоритетные)
    DEFAULT_PORTS = {
        "backend": 8000,
        "flask": 5000,
        "nextjs": 3000,
        "redis": 6379,
    }

    # Диапазоны для fallback поиска
    PORT_RANGES = {
        "backend": list(range(8000, 8050)) + list(range(8100, 8150)),
        "flask": list(range(5000, 5050)) + list(range(5100, 5150)),
        "nextjs": list(range(3000, 3050)) + list(range(3100, 3150)),
    }

    def __init__(self):
        self._used_ports: set = set()
        self._scan_used_ports()

    def _scan_used_ports(self):
        """Сканирование используемых портов на системе"""
        try:
            import re
            import subprocess

            if socket.os.name == "nt":  # Windows
                result = subprocess.run(
                    ["netstat", "-ano"], capture_output=True, text=True, timeout=10
                )
                # Парсинг: TCP    0.0.0.0:8000    0.0.0.0:0    LISTENING
                pattern = r"[:\s](\d{1,5})\s+.*LISTEN"
                self._used_ports.update(
                    int(m) for m in re.findall(pattern, result.stdout) if 1024 <= int(m) <= 65535
                )
            else:  # Linux/Mac
                result = subprocess.run(["ss", "-tuln"], capture_output=True, text=True, timeout=10)
                pattern = r":(\d{1,5})\s"
                self._used_ports.update(
                    int(m) for m in re.findall(pattern, result.stdout) if 1024 <= int(m) <= 65535
                )
        except Exception as e:
            logger.warning(f"Failed to scan used ports: {e}")

    def is_port_available(self, port: int, host: str = "127.0.0.1") -> bool:
        """Проверка доступности конкретного порта"""
        if port in self._used_ports:
            return False

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                available = result != 0
                if available:
                    self._used_ports.add(port)  # Резервируем
                return available
        except (socket.error, OSError):
            return False

    def find_available_port(
        self,
        service_name: str = "backend",
        preferred_port: Optional[int] = None,
        host: str = "127.0.0.1",
    ) -> int:
        """
        Поиск свободного порта для сервиса

        Args:
            service_name: Имя сервиса (backend, flask, nextjs)
            preferred_port: Предпочтительный порт (если None - из DEFAULT_PORTS)
            host: Хост для проверки

        Returns:
            Найденный свободный порт

        Raises:
            RuntimeError: Если не удалось найти свободный порт
        """
        # Определяем предпочтительный порт
        if preferred_port is None:
            preferred_port = self.DEFAULT_PORTS.get(service_name, 8000)

        # Проверяем предпочтительный порт первым
        if self.is_port_available(preferred_port, host):
            logger.info(f"Found available port for {service_name}: {preferred_port} (preferred)")
            return preferred_port

        # Ищем в диапазоне
        port_range = self.PORT_RANGES.get(service_name, list(range(8000, 8100)))

        for port in port_range:
            if port == preferred_port:
                continue  # Уже проверили
            if self.is_port_available(port, host):
                logger.info(f"Found available port for {service_name}: {port} (fallback)")
                return port

        raise RuntimeError(
            f"Не удалось найти свободный порт для {service_name}. "
            f"Проверены порты: {len(port_range)}"
        )

    def find_multiple_ports(self, services: List[str], host: str = "127.0.0.1") -> dict:
        """
        Поиск свободных портов для нескольких сервисов одновременно

        Args:
            services: Список имён сервисов
            host: Хост для проверки

        Returns:
            Словарь {service_name: port}
        """
        result = {}

        for service in services:
            try:
                port = self.find_available_port(service, host=host)
                result[service] = port
            except RuntimeError as e:
                logger.error(f"Failed to find port for {service}: {e}")
                raise

        return result

    def get_port_status(self, port: int, host: str = "127.0.0.1") -> dict:
        """
        Получение статуса порта

        Returns:
            {"port": 8000, "available": True, "service": "backend"}
        """
        available = self.is_port_available(port, host)

        # Определяем какой сервис обычно использует этот порт
        service = "unknown"
        for svc, p in self.DEFAULT_PORTS.items():
            if p == port:
                service = svc
                break

        return {"port": port, "available": available, "service": service, "host": host}

    def suggest_ports(
        self, service_name: str = "backend", count: int = 5, host: str = "127.0.0.1"
    ) -> List[int]:
        """Предложение нескольких свободных портов"""
        available = []
        port_range = self.PORT_RANGES.get(service_name, list(range(8000, 8100)))

        for port in port_range:
            if self.is_port_available(port, host):
                available.append(port)
                if len(available) >= count:
                    break

        return available


# Глобальный инстанс для удобства использования
_port_finder: Optional[PortFinder] = None


def get_port_finder() -> PortFinder:
    """Получение глобального PortFinder (singleton)"""
    global _port_finder
    if _port_finder is None:
        _port_finder = PortFinder()
    return _port_finder


def find_port(service_name: str = "backend", preferred: Optional[int] = None) -> int:
    """Удобная обёртка для быстрого поиска порта"""
    return get_port_finder().find_available_port(service_name, preferred)


def find_ports(services: List[str]) -> dict:
    """Удобная обёртка для поиска нескольких портов"""
    return get_port_finder().find_multiple_ports(services)


if __name__ == "__main__":
    # Демо режим при прямом запуске
    pass

    finder = PortFinder()

    print("=" * 60)
    print("Auto Port Finder - Демо")
    print("=" * 60)

    # Поиск портов для всех сервисов
    try:
        ports = finder.find_multiple_ports(["backend", "flask", "nextjs"])
        print("\n✅ Найденные порты:")
        for service, port in ports.items():
            print(f"  {service:15s}: {port}")
    except RuntimeError as e:
        print(f"\n❌ Ошибка: {e}")

    # Статус стандартных портов
    print("\n📊 Статус стандартных портов:")
    for service, port in finder.DEFAULT_PORTS.items():
        status = finder.get_port_status(port)
        icon = "✅" if status["available"] else "❌"
        print(f"  {icon} {service:15s}: {port} ({'свободен' if status['available'] else 'занят'})")

    # Предложения для backend
    print("\n💡 Предложения для backend:")
    suggestions = finder.suggest_ports("backend", count=5)
    print(f"  {suggestions}")
