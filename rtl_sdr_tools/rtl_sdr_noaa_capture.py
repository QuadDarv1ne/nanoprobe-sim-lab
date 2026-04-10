"""
RTL-SDR V4 NOAA APT Capture
Приём метеорологических изображений со спутников NOAA.

NOAA спутники передают реальные фото Земли в формате APT:
- NOAA 15: 137.620 MHz
- NOAA 18: 137.9125 MHz
- NOAA 19: 137.100 MHz

APT формат:
- 2400 Hz субнесущая
- AM модуляция
- 4160 строк, 909 пикселей/строка
- Канал A: видимый свет, Канал B: инфракрасный
- Проход спутника: ~10-15 минут

Использование:
    python rtl_sdr_noaa_capture.py              # NOAA 19, 15 минут
    python rtl_sdr_noaa_capture.py --sat noaa18 # NOAA 18
    python rtl_sdr_noaa_capture.py -d 600       # 10 минут
"""

import argparse
import os
import time
import wave

import numpy as np
from rtlsdr import RtlSdr
from scipy.signal import firwin, lfilter, resample_poly

# NOAA спутники
SATELLITES = {
    "noaa15": {"freq": 137.620, "name": "NOAA 15"},
    "noaa18": {"freq": 137.9125, "name": "NOAA 18"},
    "noaa19": {"freq": 137.100, "name": "NOAA 19"},
}


def fm_demodulate_noaa(iq_samples, sample_rate, target_rate=11025):
    """
    FM демодуляция для NOAA APT.
    NOAA использует более низкую частоту для декодирования.
    """
    print("📻 FM демодуляция...", flush=True)

    # Дифференцирование фазы
    phase = np.angle(iq_samples)
    audio = np.diff(np.unwrap(phase)).astype(np.float64)

    # Anti-aliasing фильтр
    nyquist = sample_rate / 2.0
    cutoff = min(target_rate / 2.0 * 0.9, nyquist * 0.8)
    fir = firwin(127, cutoff / nyquist)
    audio = lfilter(fir, 1.0, audio)

    # Ресемплинг
    print(f"   Ресемплинг {sample_rate} → {target_rate} Hz...", flush=True)
    audio_resampled = resample_poly(audio, target_rate, sample_rate).astype(np.float32)

    # Нормализация
    peak = np.max(np.abs(audio_resampled))
    if peak > 0:
        audio_resampled /= peak

    print(f"   ✓ Аудио: {len(audio_resampled)} сэмплов @ {target_rate} Hz", flush=True)
    print(f"   ✓ Длительность: {len(audio_resampled)/target_rate:.1f} сек", flush=True)

    return audio_resampled


def check_noaa_signal(audio, sample_rate=11025):
    """Проверяем наличие NOAA APT сигнала (2400 Hz субнесущая)."""
    print("\n📊 Анализ NOAA APT...", flush=True)

    try:
        from scipy.signal import welch

        # Берём сегмент для анализа
        segment = audio[: min(len(audio), sample_rate * 10)]
        f, pxx = welch(segment, sample_rate, nperseg=2048)

        # NOAA APT использует 2400 Hz субнесущую
        p_2400 = np.mean(pxx[(f >= 2350) & (f <= 2450)])
        p_1200 = np.mean(pxx[(f >= 1150) & (f <= 1250)])  # Синхронизация
        p_2000 = np.mean(pxx[(f >= 1950) & (f <= 2050)])  # Видимый канал
        p_total = np.mean(pxx) + 1e-12

        print(f"   2400 Hz (субнесущая APT): {p_2400/p_total:.4f}", flush=True)
        print(f"   2000 Hz (канал A - видимый): {p_2000/p_total:.4f}", flush=True)
        print(f"   1200 Hz (синхронизация): {p_1200/p_total:.4f}", flush=True)

        # Если 2400 Hz доминирует — это NOAA APT
        noaa_detected = p_2400 > p_total * 0.1

        if noaa_detected:
            print("   ✅ NOAA APT сигнал ОБНАРУЖЕН!", flush=True)
            print("   🛰️  Спутник передаёт данные!", flush=True)
        else:
            print("   ℹ️  NOAA APT не обнаружен", flush=True)
            print("   ⏰ Возможно спутник не в зоне видимости", flush=True)

        return noaa_detected, f, pxx

    except Exception as e:
        print(f"   ⚠️  Анализ: {e}", flush=True)
        return False, None, None


def decode_noaa_apt(audio, output_image="data/noaa/decoded.png"):
    """
    Простой декодер NOAA APT.
    APT формат: AM модуляция на 2400 Hz, 4160 строк × 909 пикселей.
    """
    print("\n🔍 Декодирование NOAA APT...", flush=True)

    try:
        # Детектируем огибающую AM сигнала на 2400 Hz
        from scipy.signal import hilbert

        print("   Демодуляция AM 2400 Hz...", flush=True)

        # Узкополосный фильтр на 2400 Hz
        sample_rate = 11025
        nyq = sample_rate / 2.0
        low = (2400 - 200) / nyq
        high = (2400 + 200) / nyq

        from scipy.signal import butter, filtfilt

        b, a = butter(4, [low, high], btype="band")
        filtered = filtfilt(b, a, audio)

        # Детектирование огибающей (AM демодуляция)
        envelope = np.abs(hilbert(filtered))

        # Нормализация
        envelope = (envelope - np.min(envelope)) / (np.max(envelope) - np.min(envelope) + 1e-12)

        print("   Извлечение строк...", flush=True)

        # NOAA APT: 4160 строк, ~2 секунды на проход
        # Каждая строка: 909 пикселей данных
        # Синхроимпульсы: 1200 Hz и 2400 Hz маркеры

        # Простой подход: берём среднюю огибающую как изображение
        # Это грубый декодер, для лучшего нужно синхронизироваться по sync pulse

        height = 1000  # Примерная высота
        width = 909  # NOAA APT ширина

        if len(envelope) < height * width:
            print(f"   ⚠️  Недостаточно данных ({len(envelope)} < {height*width})", flush=True)
            return False

        # Извлекаем пиксели (грубо, без синхронизации)
        pixels = envelope[: height * width]
        pixels = pixels.reshape((height, width))

        # Конвертируем в изображение
        from PIL import Image

        img_array = (pixels * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode="L")

        # Сохраняем
        os.makedirs(os.path.dirname(output_image), exist_ok=True)
        img.save(output_image)

        print(f"   ✅ Изображение сохранено: {output_image}", flush=True)
        print(f"   Размер: {img.size}", flush=True)

        return True

    except Exception as e:
        print(f"   ❌ Ошибка декодирования: {e}", flush=True)
        import traceback

        traceback.print_exc()
        return False


def capture_noaa(frequency, gain=30.0, duration=600.0, satellite_name="NOAA"):
    """Захват сигнала NOAA."""

    print("=" * 60, flush=True)
    print(f"🛰️  RTL-SDR V4 NOAA APT Capture", flush=True)
    print("=" * 60, flush=True)
    print(f"📡 Спутник: {satellite_name}", flush=True)
    print(f"📊 Частота: {frequency} MHz", flush=True)
    print(f"🔊 Gain: {gain} dB", flush=True)
    print(f"⏱️  Длительность: {duration} сек ({duration/60:.1f} мин)", flush=True)
    print("=" * 60, flush=True)

    # Инициализация RTL-SDR
    print("\n🔌 Подключение RTL-SDR...", flush=True)
    try:
        sdr = RtlSdr()
        sdr.rs = 2400000  # 2.4 MSPS для NOAA
        sdr.fc = int(frequency * 1e6)
        sdr.gain = gain

        print(f"✅ Tuner: {sdr.get_tuner_type()}", flush=True)
    except Exception as e:
        print(f"❌ Ошибка: {e}", flush=True)
        return False

    # Запись
    print(f"\n⏺️  Запись {duration} сек...", flush=True)
    print("   (NOAA проходит ~10-15 мин, нужно поймать пролёт)", flush=True)

    num_samples = int(2.4e6 * duration)
    chunks = []
    chunk_size = 256 * 1024
    start = time.time()
    last_report = 0

    try:
        while len(chunks) * chunk_size < num_samples:
            samples = sdr.read_samples(chunk_size)
            chunks.append(samples)

            elapsed = time.time() - start
            if int(elapsed) > last_report:
                progress = min(100, (len(chunks) * chunk_size / num_samples) * 100)
                print(f"   {progress:.0f}% ({elapsed:.0f}/{duration:.0f}с)", flush=True, end="\r")
                last_report = int(elapsed)
    except KeyboardInterrupt:
        print("\n⏹️  Остановка по Ctrl+C", flush=True)
    except Exception as e:
        print(f"\n❌ Ошибка: {e}", flush=True)
        sdr.close()
        return False
    finally:
        try:
            sdr.close()
        except Exception:
            pass

    total_time = time.time() - start
    iq_all = np.concatenate(chunks)

    print(f"\n✅ Записано: {len(iq_all)/1e6:.1f}M I/Q ({total_time:.1f}с)", flush=True)

    # FM демодуляция
    audio = fm_demodulate_noaa(iq_all, 2400000, 11025)

    # Сохраняем WAV
    wav_file = f'data/noaa/{satellite_name.lower().replace(" ", "_")}_{int(time.time())}.wav'
    os.makedirs("data/noaa", exist_ok=True)

    print(f"\n💾 Сохранение WAV...", flush=True)
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(wav_file, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(11025)
        wf.writeframes(pcm.tobytes())

    print(f"✅ WAV: {wav_file}", flush=True)
    print(f"   {len(audio)/11025:.1f} сек", flush=True)

    # Проверка на NOAA APT
    noaa_detected, f, pxx = check_noaa_signal(audio, 11025)

    # Спектр
    print("\n📊 Спектр...", flush=True)
    fft_size = 4096
    hann = np.hanning(fft_size)
    fft_result = np.fft.fftshift(np.fft.fft(iq_all[:fft_size] * hann))
    power_db = 10 * np.log10(np.abs(fft_result) ** 2 + 1e-12)
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, 1.0 / 2400000)) / 1e6 + frequency

    peak_idx = np.argmax(power_db)
    print(f"   Пик: {freqs[peak_idx]:.3f} MHz ({power_db[peak_idx]:.1f} dB)", flush=True)

    # Если обнаружен NOAA сигнал — пробуем декодировать
    if noaa_detected:
        print("\n" + "=" * 60, flush=True)
        decode_noaa_apt(audio, f'data/noaa/{satellite_name.lower().replace(" ", "_")}_decoded.png')

    print("\n🔌 Готово!", flush=True)
    return noaa_detected


def main():
    parser = argparse.ArgumentParser(description="RTL-SDR NOAA APT Capture")
    parser.add_argument(
        "--sat",
        "--satellite",
        type=str,
        default="noaa19",
        choices=["noaa15", "noaa18", "noaa19"],
        help="Спутник (default: noaa19)",
    )
    parser.add_argument(
        "-f", "--frequency", type=float, default=None, help="Частота MHz (переопределяет --sat)"
    )
    parser.add_argument("-g", "--gain", type=float, default=30.0, help="Усиление dB (default: 30)")
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=600.0,
        help="Длительность сек (default: 600 = 10 мин)",
    )

    args = parser.parse_args()

    if args.frequency:
        frequency = args.frequency
        satellite_name = f"Custom {frequency} MHz"
    else:
        sat = SATELLITES[args.sat]
        frequency = sat["freq"]
        satellite_name = sat["name"]

    capture_noaa(frequency, args.gain, args.duration, satellite_name)


if __name__ == "__main__":
    main()
