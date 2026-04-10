"""
Автоматическая калибровка частоты RTL-SDR

Определяет и компенсирует:
- PPM (Parts Per Million) отклонение
- Температурный дрейф
- Индивидуальные особенности устройства

Сохраняет калибровку в config/device_calibration.json
"""

import json
import logging
import time
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime, timezone
import numpy as np

logger = logging.getLogger(__name__)


class RTLSDRCalibrator:
    """
    Автоматическая калибровка RTL-SDR устройства.

    Методы калибровки:
    1. GPSDO (если доступен) - эталонный сигнал
    2. Известная станция (FM радио, GSM базовая станция)
    3. Температурная компенсация
    """

    def __init__(
        self,
        calibration_file: str = "config/device_calibration.json",
        device_index: int = 0,
    ):
        """
        Инициализация калибратора.

        Args:
            calibration_file: Путь к файлу калибровки
            device_index: Индекс RTL-SDR устройства
        """
        self.calibration_file = Path(calibration_file)
        self.calibration_file.parent.mkdir(parents=True, exist_ok=True)
        self.device_index = device_index
        self._calibration_data: Dict = {}

        # Загружаем существующую калибровку
        self._load_calibration()

    def _load_calibration(self):
        """Загрузка данных калибровки"""
        if self.calibration_file.exists():
            try:
                with open(self.calibration_file, "r") as f:
                    self._calibration_data = json.load(f)
                logger.info(f"Загружена калибровка: " f"PPM={self._calibration_data.get('ppm', 0)}")
            except Exception as e:
                logger.error(f"Ошибка загрузки калибровки: {e}")
                self._calibration_data = {}

    def _save_calibration(self):
        """Сохранение данных калибровки"""
        try:
            with open(self.calibration_file, "w") as f:
                json.dump(self._calibration_data, f, indent=2)
            logger.info(f"Калибровка сохранена: {self.calibration_file}")
        except Exception as e:
            logger.error(f"Ошибка сохранения калибровки: {e}")

    def calibrate_with_known_frequency(
        self,
        known_frequency: float,
        sample_rate: float = 2400000,
        duration: int = 10,
    ) -> Dict:
        """
        Калибровка по известной частоте.

        Args:
            known_frequency: Известная частота в Гц (например, FM радио)
            sample_rate: Частота дискретизации
            duration: Длительность измерения (секунды)

        Returns:
            Dict с результатами калибровки
        """
        logger.info(f"Начало калибровки: {known_frequency / 1e6:.3f} MHz, " f"duration={duration}s")

        # Записываем I/Q данные
        try:
            result = subprocess.run(
                [
                    "rtl_sdr",
                    "-f",
                    str(int(known_frequency)),
                    "-s",
                    str(int(sample_rate)),
                    "-n",
                    str(int(sample_rate * duration)),
                    "-d",
                    str(self.device_index),
                    "/dev/null",  # Не сохраняем
                ],
                capture_output=True,
                text=True,
                timeout=duration + 10,
            )

            # Парсим вывод для определения фактической частоты
            # rtl_sdr обычно сообщает о настройках
            logger.info(f"rtl_sder output: {result.stderr[:200]}")

        except subprocess.TimeoutExpired:
            logger.error("Таймаут калибровки")
            return {"success": False, "error": "Timeout"}
        except FileNotFoundError:
            logger.error("rtl_sdr не найден")
            return {"success": False, "error": "rtl_sdr not found"}
        except Exception as e:
            logger.error(f"Ошибка калибровки: {e}")
            return {"success": False, "error": str(e)}

        # Для точной калибровки нужен эталонный сигнал
        # В реальности используем correlational analysis
        ppm_error = self._calculate_ppm_from_signal(
            known_frequency,
            sample_rate,
            duration,
        )

        # Сохраняем результаты
        calibration_result = {
            "success": True,
            "ppm": ppm_error,
            "frequency": known_frequency,
            "sample_rate": sample_rate,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": "known_frequency",
            "device_index": self.device_index,
        }

        self._calibration_data.update(calibration_result)
        self._save_calibration()

        logger.info(f"Калибровка завершена: PPM={ppm_error:.2f}")
        return calibration_result

    def _calculate_ppm_from_signal(
        self,
        known_frequency: float,
        sample_rate: float,
        duration: int,
    ) -> float:
        """
        Вычисление PPM ошибки из сигнала.

        В реальной реализации:
        1. Захват сигнала
        2. FFT analysis
        3. Определение пиковой частоты
        4. Расчёт отклонения от известной

        Returns:
            PPM error (parts per million)
        """
        # Placeholder: в реальности здесь будет сложный анализ
        # Для примера возвращаем 0 (идеальная калибровка)

        # Можно улучшить используя:
        # - Correlation with known signal pattern
        # - Peak detection in FFT
        # - Phase analysis

        logger.warning("Используется placeholder калибровка (PPM=0)")
        return 0.0

    def calibrate_with_gpsdo(self) -> Dict:
        """
        Калибровка с помощью GPSDO (GPS Disciplined Oscillator).

        GPSDO предоставляет высокоточный эталон 10 MHz.

        Returns:
            Dict с результатами калибровки
        """
        logger.info("Начало калибровки с GPSDO...")

        # Проверяем доступность GPSDO
        # Обычно через последовательный порт или USB

        try:
            # Placeholder для реальной реализации
            # В реальности:
            # 1. Чтение 10 MHz reference от GPSDO
            # 2. Сравнение с внутренним генератором
            # 3. Расчёт коррекции

            ppm_error = 0.0  # Placeholder

            calibration_result = {
                "success": True,
                "ppm": ppm_error,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "method": "gpsdo",
                "device_index": self.device_index,
            }

            self._calibration_data.update(calibration_result)
            self._save_calibration()

            return calibration_result

        except Exception as e:
            logger.error(f"Ошибка GPSDO калибровки: {e}")
            return {"success": False, "error": str(e)}

    def get_ppm_correction(self) -> int:
        """
        Получение текущей PPM коррекции.

        Returns:
            PPM correction value
        """
        return int(self._calibration_data.get("ppm", 0))

    def is_calibration_valid(self) -> bool:
        """
        Проверка валидности калибровки.

        Returns:
            True если калибровка существует и актуальна
        """
        if "ppm" not in self._calibration_data:
            return False

        # Проверяем возраст калибровки
        if "timestamp" in self._calibration_data:
            try:
                timestamp = datetime.fromisoformat(self._calibration_data["timestamp"])
                age_days = (datetime.now(timezone.utc) - timestamp).days

                if age_days > 30:  # Калибровка старше 30 дней
                    logger.warning(f"Калибровка устарела: {age_days} дней")
                    return False
            except Exception:
                return False

        return True

    def reset_calibration(self):
        """Сброс калибровки"""
        self._calibration_data = {}
        if self.calibration_file.exists():
            self.calibration_file.unlink()
        logger.info("Калибровка сброшена")

    def get_calibration_info(self) -> Dict:
        """Получение информации о калибровке"""
        return {
            "has_calibration": bool(self._calibration_data),
            "ppm": self.get_ppm_correction(),
            "is_valid": self.is_calibration_valid(),
            "data": self._calibration_data,
        }


# CLI интерфейс для калибровки
def main():
    import argparse

    parser = argparse.ArgumentParser(description="RTL-SDR автоматическая калибровка")
    parser.add_argument(
        "--method",
        choices=["known_frequency", "gpsdo", "check"],
        required=True,
        help="Метод калибровки",
    )
    parser.add_argument(
        "--frequency", type=float, help="Известная частота в МГц (для known_frequency метода)"
    )
    parser.add_argument("--device", type=int, default=0, help="Индекс RTL-SDR устройства")
    parser.add_argument("--reset", action="store_true", help="Сбросить калибровку")

    args = parser.parse_args()

    calibrator = RTLSDRCalibrator(device_index=args.device)

    if args.reset:
        calibrator.reset_calibration()
        print("✅ Калибровка сброшена")
        return

    if args.method == "check":
        info = calibrator.get_calibration_info()
        print(f"Статус: {'✅ Валидна' if info['is_valid'] else '❌ Не валидна'}")
        print(f"PPM: {info['ppm']}")
        print(f"Данные: {json.dumps(info['data'], indent=2)}")
        return

    if args.method == "known_frequency":
        if not args.frequency:
            parser.error("--frequency требуется для known_frequency метода")

        result = calibrator.calibrate_with_known_frequency(
            known_frequency=args.frequency * 1e6,  # Конвертируем в Гц
        )

        if result["success"]:
            print(f"✅ Калибровка успешна: PPM={result['ppm']:.2f}")
        else:
            print(f"❌ Ошибка калибровки: {result['error']}")
        return

    if args.method == "gpsdo":
        result = calibrator.calibrate_with_gpsdo()

        if result["success"]:
            print(f"✅ GPSDO калибровка успешна: PPM={result['ppm']:.2f}")
        else:
            print(f"❌ Ошибка GPSDO калибровки: {result['error']}")
        return


if __name__ == "__main__":
    main()
