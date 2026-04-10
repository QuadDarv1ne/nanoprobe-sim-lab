"""
Скрипт для приёма FM радио (87.5-108 MHz) с помощью RTL-SDR V4
Использует WFM (широкополосную FM) модуляцию
"""

import sys

from rtlsdr import RtlSdr

# Частота FM радиостанции (можно менять)
FREQUENCY_MHZ = 101.7  # Наше Радио (Москва)
SAMPLE_RATE = 2.4e6  # 2.4 MS/s
GAIN = 49.6  # Максимальное усиление
DURATION_SEC = 5  # Длительность приёма


def test_fm_radio():
    """Тест приёма FM радио"""
    print(f"📻 Приём FM радио...")
    print(f"   Частота: {FREQUENCY_MHZ} MHz")
    print(f"   Усиление: {GAIN} dB")
    print(f"   Дискретизация: {SAMPLE_RATE/1e6} MS/s")
    print(f"   Длительность: {DURATION_SEC} секунд")

    try:
        sdr = RtlSdr()

        # Настройка параметров
        sdr.sample_rate = SAMPLE_RATE
        sdr.center_freq = FREQUENCY_MHZ * 1e6
        sdr.freq_correction = 60  # Коррекция частоты
        sdr.gain = GAIN

        print(f"\n✅ RTL-SDR V4 инициализирован!")
        print(f"   Тюнер: {sdr.get_tuner_type()}")

        # Чтение образцов
        print(f"\n📡 Приём FM сигнала...")
        samples = sdr.read_samples(SAMPLE_RATE * DURATION_SEC)

        # Статистика
        amplitudes = [abs(s) for s in samples]
        print(f"\n📊 Получено образцов: {len(samples)}")
        print(f"   Амплитуда (мин): {min(amplitudes):.4f}")
        print(f"   Амплитуда (макс): {max(amplitudes):.4f}")
        print(f"   Амплитуда (средн): {sum(amplitudes)/len(amplitudes):.4f}")

        # Мощность сигнала
        power = sum(abs(s) ** 2 for s in samples) / len(samples)
        print(f"\n⚡ Мощность сигнала: {power:.4f}")

        if power > 0.05:
            print("   🟢 СИЛЬНЫЙ СИГНАЛ - FM радиостанция обнаружена!")
        elif power > 0.01:
            print("   🟡 Средний сигнал")
        else:
            print("   🔴 Сигнал слабый")

        # Сохранение в файл
        output_file = "fm_radio_samples.dat"
        samples.tofile(output_file)
        print(f"\n💾 Данные сохранены в: {output_file}")
        print(f"   Размер файла: {len(samples) * 8 / 1024:.1f} KB")

        sdr.close()
        print("\n✅ Приём завершён!")
        print("\n📝 Для прослушивания сконвертируй в WAV:")
        cmd = (
            f"   rtl_fm -f {FREQUENCY_MHZ}M -M wbfm -s 256k "
            f"-r 48k -g {GAIN} | ffplay -ar 48000 -f s16le -"
        )
        print(cmd)

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test_fm_radio()
