"""
Localized error messages for SDR operations.

Supported languages: en, ru, zh, es
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_ERROR_MESSAGES: Dict[str, Dict[str, str]] = {
    "DEVICE_NOT_FOUND": {
        "en": "SDR device not found. Check USB connection and drivers.",
        "ru": "SDR-устройство не найдено. Проверьте USB-подключение и драйверы.",
        "zh": "未找到SDR设备。请检查USB连接和驱动程序。",
        "es": "Dispositivo SDR no encontrado. Verifique la conexión USB y los controladores.",
    },
    "DEVICE_BUSY": {
        "en": "SDR device is busy. Another application may be using it.",
        "ru": "SDR-устройство занято. Возможно, другое приложение использует его.",
        "zh": "SDR设备正忙。另一个应用程序可能正在使用它。",
        "es": "El dispositivo SDR está ocupado. Otra aplicación puede estar usándolo.",
    },
    "INVALID_FREQUENCY": {
        "en": "Invalid frequency value. Must be within tuner range.",
        "ru": "Неверное значение частоты. Должно быть в диапазоне тюнера.",
        "zh": "频率值无效。必须在调谐器范围内。",
        "es": "Valor de frecuencia no válido. Debe estar dentro del rango del sintonizador.",
    },
    "PPM_INVALID": {
        "en": "Invalid PPM calibration value. Must be between -100 and 100.",
        "ru": "Неверное значение калибровки PPM. Должно быть от -100 до 100.",
        "zh": "PPM校准值无效。必须在-100到100之间。",
        "es": "Valor de calibración PPM no válido. Debe estar entre -100 y 100.",
    },
    "SAMPLE_RATE_ERROR": {
        "en": "Sample rate error. Check device capabilities and configuration.",
        "ru": "Ошибка частоты дискретизации. Проверьте возможности устройства и конфигурацию.",
        "zh": "采样率错误。请检查设备功能和配置。",
        "es": (
            "Error de tasa de muestreo. "
            "Verifique las capacidades y configuración del dispositivo."
        ),
    },
    "TUNER_ERROR": {
        "en": "Tuner error. Device may be disconnected or malfunctioning.",
        "ru": ("Ошибка тюнера. " "Устройство может быть отключено или неисправно."),
        "zh": "调谐器错误。设备可能已断开连接或出现故障。",
        "es": (
            "Error del sintonizador. " "El dispositivo puede estar desconectado o funcionar mal."
        ),
    },
    "EEPROM_ERROR": {
        "en": "EEPROM read/write error. Device configuration may be corrupted.",
        "ru": "Ошибка чтения/записи EEPROM. Конфигурация устройства может быть повреждена.",
        "zh": "EEPROM读写错误。设备配置可能已损坏。",
        "es": (
            "Error de lectura/escritura EEPROM. "
            "La configuración del dispositivo puede estar corrupta."
        ),
    },
    "OVERHEAT": {
        "en": "Device overheating detected. Reduce gain or improve cooling.",
        "ru": ("Обнаружен перегрев устройства. " "Снизьте усиление или улучшите охлаждение."),
        "zh": "检测到设备过热。请降低增益或 улучшить散热。",
        "es": (
            "Sobrecalentamiento del dispositivo detectado. "
            "Reduzca la ganancia o mejore la refrigeración."
        ),
    },
    "NO_SIGNAL": {
        "en": "No signal detected. Check antenna and frequency settings.",
        "ru": "Сигнал не обнаружен. Проверьте антенну и настройки частоты.",
        "zh": "未检测到信号。请检查天线和频率设置。",
        "es": "No se detectó señal. Verifique la antena y la configuración de frecuencia.",
    },
    "CALIBRATION_EXPIRED": {
        "en": "Calibration data has expired. Please recalibrate the device.",
        "ru": "Данные калибровки истекли. Пожалуйста, откалибруйте устройство заново.",
        "zh": "校准数据已过期。请重新校准设备。",
        "es": "Los datos de calibración han expirado. Recalibre el dispositivo.",
    },
    "RING_BUFFER_FULL": {
        "en": "Ring buffer is full. Oldest data is being overwritten.",
        "ru": "Кольцевой буфер заполнен. Самые старые данные перезаписываются.",
        "zh": "环形缓冲区已满。最旧的数据正在被覆盖。",
        "es": "El buffer circular está lleno. Los datos más antiguos se están sobrescribiendo.",
    },
    "TRIGGER_FAILED": {
        "en": "Recording trigger failed. Check squelch settings and signal level.",
        "ru": "Запуск записи не удался. Проверьте настройки шумоподавления и уровень сигнала.",
        "zh": "录制触发失败。请检查静噪设置和信号电平。",
        "es": (
            "Falló el disparo de grabación. "
            "Verifique la configuración de squelch y el nivel de señal."
        ),
    },
}


class SDRErrorLocalizer:
    """Provides localized error messages for SDR operations."""

    def __init__(self, default_lang: str = "en"):
        self._lang = (
            default_lang if default_lang in _ERROR_MESSAGES.get("DEVICE_NOT_FOUND", {}) else "en"
        )
        logger.info("SDRErrorLocalizer initialized with language: %s", self._lang)

    def get_error(self, code: str, lang: Optional[str] = None) -> str:
        """Return localized error message for the given error code.

        Args:
            code: Error code (e.g., 'DEVICE_NOT_FOUND').
            lang: Override language for this call. Uses default if None.

        Returns:
            Localized error message string, or a fallback if code/lang not found.
        """
        language = lang or self._lang

        if code not in _ERROR_MESSAGES:
            logger.warning("Unknown error code requested: %s", code)
            return f"Unknown error: {code}"

        messages = _ERROR_MESSAGES[code]

        if language not in messages:
            logger.warning(
                "No translation for '%s' in language '%s', falling back to 'en'", code, language
            )
            return messages.get("en", f"Unknown error: {code}")

        return messages[language]

    def set_language(self, lang: str) -> None:
        """Change the default localization language.

        Args:
            lang: Language code (e.g., 'en', 'ru', 'zh', 'es').
        """
        available = self.get_available_languages()
        if lang not in available:
            logger.error("Language '%s' not supported. Available: %s", lang, available)
            raise ValueError(f"Language '{lang}' not supported. Available: {available}")

        old_lang = self._lang
        self._lang = lang
        logger.info("SDRErrorLocalizer language changed: %s -> %s", old_lang, lang)

    def get_available_languages(self) -> List[str]:
        """Return list of supported language codes."""
        return ["en", "ru", "zh", "es"]
