"""
RTL-SDR V4 SSTV Capture - Приём изображений с МКС
Записывает сигнал с МКС и декодирует SSTV изображение.

Примеры:
    python rtl_sdr_sstv_capture.py                  # МКС 145.800 MHz, 120 сек
    python rtl_sdr_sstv_capture.py -d 60            # 60 секунд
    python rtl_sdr_sstv_capture.py -g 40 -d 90      # Gain 40 dB, 90 сек
    python rtl_sdr_sstv_capture.py --no-decode      # Только запись, без декодирования
"""

import sys
import os
import time
import wave
import argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

# RTL-SDR
try:
    from rtlsdr import RtlSdr
    RTLSDR_AVAILABLE = True
except ImportError:
    RTLSDR_AVAILABLE = False
    print("❌ rtlsdr не установлен: pip install pyrtlsdr==0.2.93")
    sys.exit(1)

# SSTV Decoder
try:
    from pysstv.sstv import SSTV
    SSTV_AVAILABLE = True
except ImportError:
    SSTV_AVAILABLE = False
    print("⚠️  pysstv не установлен: pip install pysstv (декодирование отключено)")


def fm_demodulate(iq_samples: np.ndarray) -> np.ndarray:
    """FM демодуляция: I/Q → аудио."""
    phase = np.angle(iq_samples)
    audio = np.diff(np.unwrap(phase)).astype(np.float32)

    # Нормализация
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak

    return audio


def save_wav(audio: np.ndarray, output_file: str, sample_rate: int = 44100):
    """Сохраняет аудио в WAV (16-bit PCM)."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)

    with wave.open(output_file, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    print(f"💾 WAV сохранён: {output_file} ({len(audio)/sample_rate:.1f} сек)")


def try_decode_sstv(wav_file: str, output_image: str) -> bool:
    """Пытается декодить SSTV из WAV файла."""
    if not SSTV_AVAILABLE:
        print("❌ pysstv не установлен")
        return False

    if not Path(wav_file).exists():
        print(f"❌ Файл не найден: {wav_file}")
        return False

    print(f"\n🔍 Попытка декодирования SSTV...")

    try:
        # Читаем WAV
        with wave.open(wav_file, 'r') as wf:
            frames = wf.readframes(wf.getnframes())
            audio_int16 = np.frombuffer(frames, dtype=np.int16)

        # Конвертируем в float
        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Создаём SSTV декодер
        sstv = SSTV(audio_float.tolist(), wf.getframerate(), 16)

        # Декодируем
        print("   Декодирование... (может занять 10-60 сек)")
        img = sstv.decode()

        if img and hasattr(img, 'size') and img.size[0] > 0:
            # Сохраняем изображение
            Path(output_image).parent.mkdir(parents=True, exist_ok=True)
            img.save(output_image)
            print(f"✅ SSTV декодировано: {img.size[0]}x{img.size[1]}")
            print(f"🖼️  Изображение: {output_image}")
            return True
        else:
            print("⚠️  SSTV не найден в записи")
            return False

    except Exception as e:
        print(f"❌ Ошибка декодирования: {e}")
        return False


def record_sstv(
    frequency: float = 145.800,
    gain: float = 20.0,
    duration: float = 120.0,
    sample_rate: int = 2400000,
    output_wav: str = None,
    decode: bool = True,
):
    """Запись SSTV с МКС."""

    if output_wav is None:
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        output_wav = f"data/sstv/iss_{timestamp}.wav"

    output_image = output_wav.replace('.wav', '.png')

    print("=" * 60)
    print("🛰️  RTL-SDR V4 SSTV Capture - МКС")
    print("=" * 60)
    print(f"📡 Частота: {frequency} MHz")
    print(f"📊 Sample rate: {sample_rate / 1e6:.1f} MSPS")
    print(f"🔊 Gain: {gain} dB")
    print(f"⏱️  Длительность: {duration} сек")
    print(f"💾 Выход: {output_wav}")
    print(f"🔍 Декодирование: {'включено' if decode else 'отключено'}")
    print("=" * 60)

    # Инициализация RTL-SDR
    print("\n🔌 Подключение RTL-SDR...")
    try:
        sdr = RtlSdr()
        sdr.rs = sample_rate
        sdr.fc = int(frequency * 1e6)
        sdr.gain = gain

        tuner = sdr.get_tuner_type()
        print(f"✅ RTL-SDR подключен (Tuner: {tuner})")
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return False

    # Запись
    print(f"\n⏺️  Запись {duration} сек...")
    start_time = time.time()

    iq_chunks = []
    buf_size = 256 * 1024  # 256k chunks

    try:
        def callback(samples, ctx):
            elapsed = time.time() - start_time
            iq_chunks.append(samples.copy())

            # Прогресс
            progress = min(100, (elapsed / duration) * 100)
            if int(progress) % 10 == 0:
                print(f"   📊 {progress:.0f}% ({elapsed:.0f}/{duration:.0f} сек)", end='\r')

            if elapsed >= duration:
                raise StopIteration

        sdr.read_samples_async(callback, buf_size)

    except StopIteration:
        pass
    except Exception as e:
        print(f"\n❌ Ошибка записи: {e}")
        sdr.close()
        return False
    finally:
        try:
            sdr.cancel_read_async()
        except Exception:
            pass
        sdr.close()

    print(f"\n✅ Запись завершена: {time.time() - start_time:.1f} сек")

    if not iq_chunks:
        print("❌ Нет данных")
        return False

    # Объединяем chunks
    iq_all = np.concatenate(iq_chunks)
    print(f"📦 I/Q сэмплов: {len(iq_all)} ({len(iq_all)/sample_rate:.1f} сек)")

    # FM демодуляция
    print("📻 FM демодуляция...")
    audio = fm_demodulate(iq_all)
    print(f"🔊 Аудио: {len(audio)} сэмплов ({len(audio)/44100:.1f} сек)")

    # Сохраняем WAV
    save_wav(audio, output_wav, sample_rate=44100)

    # Декодирование SSTV
    if decode:
        success = try_decode_sstv(output_wav, output_image)
        return success
    else:
        print(f"\n💡 Для декодирования: python rtl_sdr_sstv_capture.py --decode-only {output_wav}")
        return True


def decode_only(wav_file: str, output_image: str = None):
    """Декодирование SSTV из существующего WAV."""
    if output_image is None:
        output_image = wav_file.replace('.wav', '.png')

    print(f"🔍 Декодирование: {wav_file}")
    success = try_decode_sstv(wav_file, output_image)
    return success


def main():
    parser = argparse.ArgumentParser(description='RTL-SDR SSTV Capture - Приём изображений с МКС')
    parser.add_argument('-f', '--frequency', type=float, default=145.800,
                        help='Частота MHz (default: 145.800 ISS)')
    parser.add_argument('-g', '--gain', type=float, default=20.0,
                        help='Усиление dB (default: 20)')
    parser.add_argument('-d', '--duration', type=float, default=120.0,
                        help='Длительность записи сек (default: 120)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Выходной WAV файл')
    parser.add_argument('--no-decode', action='store_true',
                        help='Только запись, без SSTV декодирования')
    parser.add_argument('--decode-only', type=str, metavar='WAV_FILE',
                        help='Декодировать существующий WAV файл')

    args = parser.parse_args()

    if args.decode_only:
        success = decode_only(args.decode_only)
        sys.exit(0 if success else 1)

    if not RTLSDR_AVAILABLE:
        print("❌ RTL-SDR не доступен")
        sys.exit(1)

    success = record_sstv(
        frequency=args.frequency,
        gain=args.gain,
        duration=args.duration,
        output_wav=args.output,
        decode=not args.no_decode,
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
