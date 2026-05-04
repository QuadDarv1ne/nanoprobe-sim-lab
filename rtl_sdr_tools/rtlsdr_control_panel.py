#!/usr/bin/env python
"""🚀 RTL-SDR V4 Control Panel
Интерактивная панель управления RTL-SDR
"""
import logging
import time
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


def print_header():
    """Вывод заголовка панели."""
    logger.info("=" * 70)
    logger.info(" 🚀 RTL-SDR V4 CONTROL PANEL")
    logger.info(f" {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)


def print_menu():
    """Вывод меню."""
    logger.info("")
    logger.info("📋 МЕНЮ:")
    logger.info(" [1] Проверка устройства")
    logger.info(" [2] Спектральный анализ")
    logger.info(" [3] Запись сигнала")
    logger.info(" [4] Приём SSTV (МКС 145.800 МГц)")
    logger.info(" [5] Сканирование диапазона")
    logger.info(" [6] Информация об устройстве")
    logger.info(" [7] Мониторинг сигнала")
    logger.info(" [0] Выход")
    logger.info("")


def check_device():
    """Проверка устройства"""
    logger.info("")
    logger.info("=" * 70)
    logger.info(" ПРОВЕРКА УСТРОЙСТВА")
    logger.info("=" * 70)
    try:
        from rtlsdr import RtlSdr

        count = RtlSdr.get_device_count()
        logger.info(f"\n✅ Найдено устройств: {count}")
        if count > 0:
            logger.info("\nПопытка открытия устройства...")
            sdr = RtlSdr()
            logger.info("✅ Устройство открыто успешно!")
            logger.info(f" Tuner: {sdr.get_tuner_type()}")
            sdr.close()
        else:
            logger.info("\n❌ Устройства не найдены!")
            logger.info("\n💡 Решение:")
            logger.info(" 1. Проверьте USB подключение")
            logger.info(" 2. Запустите Zadig и установите WinUSB драйвер")
    except Exception as e:
        logger.error(f"\n❌ Ошибка: {e}")
        logger.info("\n💡 Запустите check_zadig_drivers.bat для диагностики")


def spectrum_analysis():
    """Спектральный анализ"""
    logger.info("")
    logger.info("=" * 70)
    logger.info(" СПЕКТРАЛЬНЫЙ АНАЛИЗ")
    logger.info("=" * 70)
    try:
        from api.sstv.rtl_sstv_receiver import RTLSDRReceiver

        freq = float(input("\nЧастота (МГц) [145.800]: ") or 145.800)
        points = int(input("Количество точек [4096]: ") or 4096)
        receiver = RTLSDRReceiver(frequency=freq, gain=49.6)
        if not receiver.initialize():
            logger.error("❌ Не удалось инициализировать устройство")
            return
        logger.info(f"\n📊 Анализ спектра на {freq} МГц...")
        freqs, power = receiver.get_spectrum(points)
        if freqs is not None:
            logger.info(f"\n✅ Спектр получен:")
            logger.info(f" Диапазон: {freqs[0]/1e6:.3f} - {freqs[-1]/1e6:.3f} МГц")
            logger.info(f" Мощность: {np.min(power):.1f} - {np.max(power):.1f} дБ")
            logger.info(f" Средний SNR: {np.max(power) - np.mean(power):.1f} дБ")
            # Визуализация
            logger.info("\n📈 Спектр (визуализация):")
            max_power = np.max(power)
            for i in range(0, len(power), len(power) // 50):
                bar_len = int((power[i] - np.min(power)) / (max_power - np.min(power)) * 40)
                logger.info(f" {'█' * bar_len} {freqs[i]/1e6:.3f} МГц")
        else:
            logger.error("❌ Не удалось получить спектр")
        receiver.close()
    except Exception as e:
        logger.error(f"\n❌ Ошибка: {e}")
        import traceback

        traceback.print_exc()


def record_signal():
    """Запись сигнала"""
    logger.info("")
    logger.info("=" * 70)
    logger.info(" ЗАПИСЬ СИГНАЛА")
    logger.info("=" * 70)
    try:
        from api.sstv.rtl_sstv_receiver import RTLSDRReceiver

        freq = float(input("\nЧастота (МГц) [145.800]: ") or 145.800)
        duration = float(input("Длительность (сек) [10]: ") or 10.0)
        receiver = RTLSDRReceiver(frequency=freq, gain=49.6)
        if not receiver.initialize():
            logger.error("❌ Не удалось инициализировать устройство")
            return
        logger.info(f"\n🎙️ Запись {duration} сек на {freq} МГц...")
        samples = receiver.record_audio(duration=duration, sample_rate=48000)
        if samples is not None:
            filename = f"recording_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.wav"
            receiver._save_wav(samples, filename, 48000)
            logger.info(f"\n✅ Запись сохранена: {filename}")
            logger.info(f" Длительность: {len(samples)/48000:.2f} сек")
        else:
            logger.error("❌ Ошибка записи")
        receiver.close()
    except Exception as e:
        logger.error(f"\n❌ Ошибка: {e}")


def receive_sstv():
    """Приём SSTV"""
    logger.info("")
    logger.info("=" * 70)
    logger.info(" ПРИЁМ SSTV (МКС)")
    logger.info("=" * 70)
    try:
        from api.sstv.rtl_sstv_receiver import RTLSDRReceiver, SSTVDecoder

        duration = float(input("\nДлительность приёма (сек) [60]: ") or 60.0)
        receiver = RTLSDRReceiver(frequency=145.800, gain=49.6)
        decoder = SSTVDecoder(mode="auto")
        if not receiver.initialize():
            logger.error("❌ Не удалось инициализировать устройство")
            return
        logger.info(f"\n🛰️ Приём SSTV {duration} сек...")
        logger.info(" Частота: 145.800 МГц (МКС)")
        logger.info(" Усиление: 49.6 dB")
        samples = receiver.record_audio(duration=duration, sample_rate=48000)
        if samples is not None:
            logger.info("\n🔍 Декодирование SSTV...")
            image = decoder.decode_audio(samples, sample_rate=48000)
            if image:
                filename = f"sstv_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.png"
                image.save(filename)
                logger.info(f"\n✅ SSTV изображение сохранено: {filename}")
                logger.info(f" Размер: {image.size}")
            else:
                logger.info("\n❌ SSTV сигнал не обнаружен")
                logger.info(" 💡 Попробуйте увеличить время приёма")
        else:
            logger.error("❌ Ошибка записи")
        receiver.close()
    except Exception as e:
        logger.error(f"\n❌ Ошибка: {e}")
        import traceback

        traceback.print_exc()


def scan_range():
    """Сканирование диапазона"""
    logger.info("")
    logger.info("=" * 70)
    logger.info(" СКАНИРОВАНИЕ ДИАПАЗОНА")
    logger.info("=" * 70)
    try:
        from api.sstv.rtl_sstv_receiver import RTLSDRReceiver

        start = float(input("\nНачальная частота (МГц) [145.0]: ") or 145.0)
        end = float(input("Конечная частота (МГц) [146.0]: ") or 146.0)
        step = float(input("Шаг (МГц) [0.1]: ") or 0.1)
        receiver = RTLSDRReceiver(frequency=start, gain=49.6)
        if not receiver.initialize():
            logger.error("❌ Не удалось инициализировать устройство")
            return
        logger.info(f"\n🔍 Сканирование {start}-{end} МГц (шаг {step} МГц)...")
        freq = start
        max_strength = 0
        max_freq = start
        while freq <= end:
            receiver.sdr.fc = freq * 1e6
            time.sleep(0.1)
            strength = receiver.get_signal_strength()
            bar_len = int(strength / 2)
            logger.info(f" {'█' * bar_len} {freq:.3f} МГц: {strength:.1f}%")
            if strength > max_strength:
                max_strength = strength
                max_freq = freq
            freq += step
        logger.info(f"\n📊 Максимальная активность:")
        logger.info(f" Частота: {max_freq:.3f} МГц")
        logger.info(f" Сила: {max_strength:.1f}%")
        receiver.close()
    except Exception as e:
        logger.error(f"\n❌ Ошибка: {e}")


def device_info():
    """Информация об устройстве"""
    logger.info("")
    logger.info("=" * 70)
    logger.info(" ИНФОРМАЦИЯ ОБ УСТРОЙСТВЕ")
    logger.info("=" * 70)
    try:
        from api.sstv.rtl_sstv_receiver import RTLSDRReceiver

        receiver = RTLSDRReceiver()
        if not receiver.initialize():
            logger.error("❌ Не удалось инициализировать устройство")
            return
        info = receiver.get_device_info()
        logger.info("\n📡 Устройство:")
        for key, value in info.items():
            logger.info(f" {key}: {value}")
        receiver.close()
    except Exception as e:
        logger.error(f"\n❌ Ошибка: {e}")


def signal_monitor():
    """Мониторинг сигнала"""
    logger.info("")
    logger.info("=" * 70)
    logger.info(" МОНИТОРИНГ СИГНАЛА")
    logger.info("=" * 70)
    try:
        from api.sstv.rtl_sstv_receiver import RTLSDRReceiver

        freq = float(input("\nЧастота мониторинга (МГц) [145.800]: ") or 145.800)
        duration = int(input("Длительность (сек) [30]: ") or 30)
        receiver = RTLSDRReceiver(frequency=freq, gain=49.6)
        if not receiver.initialize():
            logger.error("❌ Не удалось инициализировать устройство")
            return
        logger.info(f"\n📈 Мониторинг {duration} сек...")
        logger.info(" Нажмите Ctrl+C для остановки\n")
        start_time = time.time()
        max_strength = 0
        try:
            while time.time() - start_time < duration:
                strength = receiver.get_signal_strength()
                bar_len = int(strength / 2)
                timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
                logger.info(f" [{timestamp}] {'█' * bar_len} {strength:.1f}%")
                if strength > max_strength:
                    max_strength = strength
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\n\n⏹️ Мониторинг остановлен")
            logger.info(f"\n📊 Результат:")
            logger.info(f" Максимальная сила: {max_strength:.1f}%")
        receiver.close()
    except Exception as e:
        logger.error(f"\n❌ Ошибка: {e}")


def main():
    """Главная функция."""
    print_header()
    while True:
        print_menu()
        choice = input("Выберите действие [0-7]: ").strip()
        if choice == "0":
            logger.info("\n👋 До свидания!")
            break
        elif choice == "1":
            check_device()
        elif choice == "2":
            spectrum_analysis()
        elif choice == "3":
            record_signal()
        elif choice == "4":
            receive_sstv()
        elif choice == "5":
            scan_range()
        elif choice == "6":
            device_info()
        elif choice == "7":
            signal_monitor()
        else:
            logger.warning("\n❌ Неверный выбор. Попробуйте снова.")
            input("\nНажмите Enter для продолжения...")


if __name__ == "__main__":
    main()
