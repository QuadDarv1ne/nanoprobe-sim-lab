"""
Улучшенное обнаружение и управление RTL-SDR устройствами.

Поддерживаемые функции:
- Автоматическое обнаружение всех подключённых устройств
- Определение модели (V4 vs другие версии)
- Проверка доступности и состояния
- Multi-device support
"""

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class RTLSDRModel(Enum):
    """Модели RTL-SDR устройств."""

    V4 = "RTL-SDR V4"
    V3 = "RTL-SDR V3"
    V2 = "RTL-SDR V2"
    V1 = "RTL-SDR V1"
    UNKNOWN = "Unknown"


@dataclass
class RTLSDRDevice:
    """Информация об RTL-SDR устройстве."""

    index: int
    manufacturer: str
    product: str
    serial: str
    model: RTLSDRModel = RTLSDRModel.UNKNOWN
    available: bool = True
    sample_rate: Optional[float] = None
    gain: Optional[float] = None
    frequency: Optional[float] = None
    temperature: Optional[float] = None
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Конвертация в словарь."""
        return {
            "index": self.index,
            "manufacturer": self.manufacturer,
            "product": self.product,
            "serial": self.serial,
            "model": self.model.value,
            "available": self.available,
            "sample_rate": self.sample_rate,
            "gain": self.gain,
            "frequency": self.frequency,
            "temperature": self.temperature,
            "errors": self.errors,
        }


class RTLSDRDeviceDetector:
    """Детектор RTL-SDR устройств."""

    def __init__(self):
        """Инициализация детектора."""
        self.devices: Dict[int, RTLSDRDevice] = {}

    def detect_all(self) -> List[RTLSDRDevice]:
        """
        Автоматически найти все подключённые RTL-SDR устройства.

        Returns:
            Список обнаруженных устройств
        """
        logger.info("Поиск RTL-SDR устройств...")
        self.devices = {}

        try:
            # Получаем список доступных устройств
            device_count = self._get_device_count()

            if device_count == 0:
                logger.warning("RTL-SDR устройства не найдены")
                return []

            logger.info("Найдено %d устройство(ей)", device_count)

            # Получаем информацию о каждом устройстве
            for i in range(device_count):
                device = self._detect_device(i)
                if device:
                    self.devices[i] = device

            return list(self.devices.values())

        except Exception as e:
            logger.error("Ошибка при обнаружении устройств: %s", e)
            return []

    def _get_device_count(self) -> int:
        """Получить количество подключённых устройств."""
        try:
            result = subprocess.run(
                ["rtl_test", "-l"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = result.stderr + result.stdout

            # Ищем строку с количеством устройств
            match = re.search(r"Found (\d+) RTL-SDR device", output, re.IGNORECASE)
            if match:
                return int(match.group(1))

            # Альтернативный парсинг
            if "Found" in output and "device" in output:
                return 1

            return 0

        except FileNotFoundError:
            logger.error("rtl_test не найден. Установите rtl-sdr драйверы")
            return 0
        except subprocess.TimeoutExpired:
            logger.error("Тайм-аут при получении списка устройств")
            return 0
        except Exception as e:
            logger.error("Ошибка: %s", e)
            return 0

    def _detect_device(self, index: int) -> Optional[RTLSDRDevice]:
        """
        Получить информацию об устройстве по индексу.

        Args:
            index: Индекс устройства

        Returns:
            Информация об устройстве или None
        """
        try:
            result = subprocess.run(
                ["rtl_test", "-d", str(index)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = result.stderr + result.stdout

            device = RTLSDRDevice(
                index=index,
                manufacturer="",
                product="",
                serial="",
            )

            # Парсим информацию
            manufacturer_match = re.search(r"Manufacturer:\s*(\S+)", output, re.IGNORECASE)
            product_match = re.search(r"Product:\s*(.+)", output, re.IGNORECASE)
            serial_match = re.search(r"Serial:\s*(\S+)", output, re.IGNORECASE)

            if manufacturer_match:
                device.manufacturer = manufacturer_match.group(1).strip()

            if product_match:
                device.product = product_match.group(1).strip()
                # Определяем модель по названию продукта
                device.model = self._identify_model(device.product)

            if serial_match:
                device.serial = serial_match.group(1).strip()

            # Проверяем доступность
            device.available = self._check_device_availability(index)

            # Проверяем температуру (для V4)
            if device.model == RTLSDRModel.V4:
                device.temperature = self._get_device_temperature(index)

            return device

        except FileNotFoundError:
            logger.error("rtl_test не найден")
            return None
        except subprocess.TimeoutExpired:
            logger.error("Тайм-аут при получении информации об устройстве %d", index)
            device = RTLSDRDevice(index=index, manufacturer="", product="", serial="")
            device.errors.append("Тайм-аут")
            device.available = False
            return device
        except Exception as e:
            logger.error("Ошибка при обнаружении устройства %d: %s", index, e)
            device = RTLSDRDevice(index=index, manufacturer="", product="", serial="")
            device.errors.append(str(e))
            device.available = False
            return device

    def _identify_model(self, product_name: str) -> RTLSDRModel:
        """
        Определить модель устройства по названию продукта.

        Args:
            product_name: Название продукта

        Returns:
            Модель устройства
        """
        product_lower = product_name.lower()

        if "rtl-SDR V4" in product_lower or "v4" in product_lower:
            return RTLSDRModel.V4
        elif "rtl-SDR V3" in product_lower or "v3" in product_lower:
            return RTLSDRModel.V3
        elif "rtl-SDR V2" in product_lower or "v2" in product_lower:
            return RTLSDRModel.V2
        elif "rtl-SDR V1" in product_lower or "v1" in product_lower:
            return RTLSDRModel.V1
        else:
            return RTLSDRModel.UNKNOWN

    def _check_device_availability(self, index: int) -> bool:
        """
        Проверить доступность устройства.

        Args:
            index: Индекс устройства

        Returns:
            True если устройство доступно
        """
        try:
            result = subprocess.run(
                ["rtl_test", "-d", str(index), "-t", "1"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            # Если нет критических ошибок, устройство доступно
            return "error" not in result.stderr.lower()
        except Exception:
            return False

    def _get_device_temperature(self, index: int) -> Optional[float]:
        """
        Получить температуру устройства (для RTL-SDR V4).

        Args:
            index: Индекс устройства

        Returns:
            Температура в градусах Цельсия или None
        """
        try:
            # RTL-SDR V4 поддерживает чтение температуры
            result = subprocess.run(
                ["rtl_eeprom", "-d", str(index), "--config-only"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = result.stdout + result.stderr

            # Парсим температуру из вывода
            temp_match = re.search(r"temperature[:\s]+(-?\d+\.?\d*)", output, re.IGNORECASE)
            if temp_match:
                return float(temp_match.group(1))

            return None

        except Exception as e:
            logger.debug("Не удалось получить температуру: %s", e)
            return None

    def get_v4_devices(self) -> List[RTLSDRDevice]:
        """
        Получить только RTL-SDR V4 устройства.

        Returns:
            Список V4 устройств
        """
        return [d for d in self.devices.values() if d.model == RTLSDRModel.V4]

    def get_available_devices(self) -> List[RTLSDRDevice]:
        """
        Получить доступные устройства.

        Returns:
            Список доступных устройств
        """
        return [d for d in self.devices.values() if d.available]

    def select_best_device(self) -> Optional[RTLSDRDevice]:
        """
        Выбрать лучшее доступное устройство (приоритет V4 > V3 > V2 > V1).

        Returns:
            Лучшее устройство или None
        """
        available = self.get_available_devices()
        if not available:
            return None

        # Сортируем по приоритету модели
        priority = {
            RTLSDRModel.V4: 4,
            RTLSDRModel.V3: 3,
            RTLSDRModel.V2: 2,
            RTLSDRModel.V1: 1,
            RTLSDRModel.UNKNOWN: 0,
        }

        return max(available, key=lambda d: priority.get(d.model, 0))


def get_rtl_sdr_devices_detailed() -> List[Dict]:
    """
    Улучшенная версия get_rtl_sdr_devices с полной информацией.

    Returns:
        Список устройств с детальной информацией
    """
    detector = RTLSDRDeviceDetector()
    devices = detector.detect_all()
    return [d.to_dict() for d in devices]


def main():
    """CLI интерфейс для обнаружения устройств."""
    logging.basicConfig(level=logging.INFO)

    logger.info("=== RTL-SDR Device Detector ===")

    detector = RTLSDRDeviceDetector()
    devices = detector.detect_all()

    if not devices:
        logger.info("RTL-SDR устройства не найдены")
        logger.info("Убедитесь, что:")
        logger.info("  1. Устройство подключено к USB")
        logger.info("  2. Установлены драйверы (libusb, rtl-sdr)")
        logger.info("  3. Устройство не занято другим приложением")
        return

    logger.info("Найдено %d устройство(ей):", len(devices))

    for device in devices:
        logger.info("Устройство %d:", device.index)
        logger.info("  Производитель: %s", device.manufacturer)
        logger.info("  Продукт: %s", device.product)
        logger.info("  Модель: %s", device.model.value)
        logger.info("  Серийный номер: %s", device.serial)
        logger.info("  Доступно: %s", "Да" if device.available else "Нет")
        if device.temperature is not None:
            logger.info("  Температура: %.1f°C", device.temperature)
        if device.errors:
            logger.info("  Ошибки: %s", ", ".join(device.errors))
        logger.info("")

    # Показать лучшее устройство
    best = detector.select_best_device()
    if best:
        logger.info("Рекомендуемое устройство: %s (индекс %d)", best.model.value, best.index)


if __name__ == "__main__":
    main()
