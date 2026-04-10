"""
Быстрое сканирование авиачастот (118-137 MHz)
Проверка мощности сигнала на известных частотах
"""

import time

from rtlsdr import RtlSdr

# Известные авиачастоты (Москва)
FREQUENCIES = {
    118.1: "Шереметьево Подход",
    118.5: "Тестовая",
    119.1: "Внуково Подход",
    120.4: "Домодедово Башня",
    124.35: "Шереметьево Башня",
    126.7: "Москва-Центр",
    132.0: "Москва-Контроль",
}

GAIN = 40
DURATION = 2  # секунд на частоту


def scan_frequencies():
    """Быстрое сканирование списка частот"""
    print(f"✈️ Сканирование авиачастот...")
    print(f"   Усиление: {GAIN} dB")
    print(f"   Время на частоту: {DURATION} сек")
    print()

    sdr = RtlSdr()
    sdr.sample_rate = 2.4e6
    sdr.freq_correction = 60
    sdr.gain = GAIN

    results = []

    for freq_mhz, name in FREQUENCIES.items():
        print(f"📡 {freq_mhz:.2f} MHz - {name}...", end=" ")

        sdr.center_freq = freq_mhz * 1e6
        time.sleep(0.5)  # Стабилизация

        samples = sdr.read_samples(2.4e6 * DURATION)
        power = sum(abs(s) ** 2 for s in samples) / len(samples)

        # Индикатор силы сигнала
        if power > 0.1:
            indicator = "🟢 СИЛЬНЫЙ"
        elif power > 0.01:
            indicator = "🟡 Средний"
        else:
            indicator = "🔴 Слабый"

        print(f"{power:.4f} - {indicator}")
        results.append((freq_mhz, name, power))

    sdr.close()

    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТЫ:")
    print("=" * 60)

    # Сортировка по мощности
    results.sort(key=lambda x: x[2], reverse=True)

    for freq, name, power in results:
        if power > 0.1:
            indicator = "🟢"
        elif power > 0.01:
            indicator = "🟡"
        else:
            indicator = "🔴"
        print(f"  {indicator} {freq:.2f} MHz - {name:25s} ({power:.4f})")

    # Рекомендация
    best = results[0]
    print(f"\n🎯 Рекомендация: Слушать {best[0]:.2f} MHz ({best[1]})")
    print(f"   Команда: listen_airband.bat {best[0]}")


if __name__ == "__main__":
    scan_frequencies()
