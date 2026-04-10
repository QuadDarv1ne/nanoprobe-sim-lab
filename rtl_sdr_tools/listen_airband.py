"""
Скрипт для приёма авиасигналов (118-137 MHz) с помощью RTL-SDR V4
Использует AM модуляцию для декодирования авиасвязи
"""

import sys

from rtlsdr import RtlSdr

# Частота авиасвязи (можно менять)
FREQUENCY_MHZ = 118.5  # Типичная частота авиасвязи
SAMPLE_RATE = 2.4e6  # 2.4 MS/s
GAIN = 40  # Усиление (0-49.6)
DURATION_SEC = 10  # Длительность приёма


def test_airband():
    """Тест приёма авиасигналов"""
    print(f"📡 Инициализация RTL-SDR V4...")
    print(f"   Частота: {FREQUENCY_MHZ} MHz")
    print(f"   Усиление: {GAIN} dB")
    print(f"   Дискретизация: {SAMPLE_RATE/1e6} MS/s")

    try:
        sdr = RtlSdr()

        # Настройка параметров
        sdr.sample_rate = SAMPLE_RATE
        sdr.center_freq = FREQUENCY_MHZ * 1e6
        sdr.freq_correction = 60  # Коррекция частоты (типично для RTL-SDR)
        sdr.gain = GAIN

        print(f"\n✅ RTL-SDR V4 инициализирован успешно!")
        print(f"   Тип тюнера: {sdr.get_tuner_type()}")

        # Чтение образцов
        print(f"\n📻 Приём данных ({DURATION_SEC} секунд)...")
        samples = sdr.read_samples(SAMPLE_RATE * DURATION_SEC)

        # Статистика
        print(f"\n📊 Получено образцов: {len(samples)}")
        print(f"   Амплитуда (мин): {min(abs(samples)):.4f}")
        print(f"   Амплитуда (макс): {max(abs(samples)):.4f}")
        print(f"   Амплитуда (средн): {sum(abs(samples))/len(samples):.4f}")

        # Проверка на наличие сигналов
        power = sum(abs(s) ** 2 for s in samples) / len(samples)
        print(f"\n⚡ Мощность сигнала: {power:.4f}")

        if power > 0.1:
            print("   🟢 Сигнал обнаружен!")
        else:
            print("   🟡 Сигнал слабый или отсутствует")

        # Сохранение в файл для анализа
        output_file = "airband_samples.dat"
        samples.tofile(output_file)
        print(f"\n💾 Данные сохранены в: {output_file}")

        sdr.close()
        print("\n✅ Приём завершён!")

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        print("\nВозможные причины:")
        print("  1. RTL-SDR не подключён к USB")
        print("  2. Драйверы Zadig не установлены")
        print("  3. Устройство заблокировано другим приложением")
        sys.exit(1)


if __name__ == "__main__":
    test_airband()
