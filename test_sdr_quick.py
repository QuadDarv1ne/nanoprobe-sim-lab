"""
RTL-SDR V4 - Правильная FM демодуляция с ресемплингом
Записывает I/Q и конвертирует в аудио 44100 Hz для SSTV.
"""
import sys
import time
import os
import wave
import numpy as np
from rtlsdr import RtlSdr
from scipy.signal import resample_poly, firwin, lfilter

def fm_demodulate_with_resample(iq_samples, sample_rate, target_rate=44100):
    """
    Правильная FM демодуляция с ресемплингом.
    
    Args:
        iq_samples: Комплексные I/Q сэмплы
        sample_rate: Исходная частота (2.4 MHz)
        target_rate: Целевая частота аудио (44100 Hz)
    
    Returns:
        np.ndarray: Аудио float32
    """
    print('📻 FM демодуляция...', flush=True)
    
    # 1. Дифференцирование фазы
    phase = np.angle(iq_samples)
    audio = np.diff(np.unwrap(phase)).astype(np.float64)
    
    print(f'   Демодулировано: {len(audio)} сэмплов @ {sample_rate} Hz', flush=True)
    print(f'   Длительность: {len(audio)/sample_rate:.2f} сек', flush=True)
    
    # 2. Anti-aliasing фильтр перед ресемплингом
    nyquist = sample_rate / 2.0
    cutoff = min(target_rate / 2.0 * 0.9, nyquist * 0.8)
    fir = firwin(127, cutoff / nyquist)
    audio = lfilter(fir, 1.0, audio)
    
    # 3. Ресемплинг
    print(f'   Ресемплинг {sample_rate} → {target_rate} Hz...', flush=True)
    audio_resampled = resample_poly(audio, target_rate, sample_rate).astype(np.float32)
    
    # 4. Нормализация
    peak = np.max(np.abs(audio_resampled))
    if peak > 0:
        audio_resampled /= peak
    
    print(f'   ✓ Аудио: {len(audio_resampled)} сэмплов @ {target_rate} Hz', flush=True)
    print(f'   ✓ Длительность: {len(audio_resampled)/target_rate:.2f} сек', flush=True)
    print(f'   ✓ Пик: {np.max(np.abs(audio_resampled)):.3f}', flush=True)
    
    return audio_resampled


def main():
    print('📡 RTL-SDR V4 SSTV Capture с правильной FM демодуляцией', flush=True)
    print('='*60, flush=True)
    
    # Подключение
    print('\n🔌 Инициализация RTL-SDR...', flush=True)
    sdr = RtlSdr()
    sdr.rs = 2400000  # 2.4 MSPS
    sdr.fc = int(145.800 * 1e6)  # ISS
    sdr.gain = 20
    
    print(f'✅ Tuner: {sdr.get_tuner_type()}', flush=True)
    print(f'📊 Частота: 145.800 MHz', flush=True)
    
    # Запись 3 секунд
    duration = 3.0
    print(f'\n⏺️  Запись {duration} секунд...', flush=True)
    num_samples = int(2.4e6 * duration)
    chunks = []
    chunk_size = 256 * 1024
    start = time.time()
    
    while len(chunks) * chunk_size < num_samples:
        samples = sdr.read_samples(chunk_size)
        chunks.append(samples)
        elapsed = time.time() - start
        progress = min(100, (len(chunks) * chunk_size / num_samples) * 100)
        print(f'   {progress:.0f}% ({elapsed:.1f}с)', flush=True, end='\r')
    
    total_time = time.time() - start
    iq_all = np.concatenate(chunks)
    
    print(f'\n✅ Записано: {len(iq_all)/1e6:.1f}M I/Q сэмплов ({total_time:.1f}с)', flush=True)
    
    # FM демодуляция с ресемплингом
    audio = fm_demodulate_with_resample(iq_all, 2400000, 44100)
    
    # SSTV анализ
    print('\n📊 SSTV анализ...', flush=True)
    try:
        from scipy.signal import welch
        
        # Берём первые 2 секунды аудио
        audio_segment = audio[:min(len(audio), 44100*2)]
        f, pxx = welch(audio_segment, 44100, nperseg=2048)
        
        def band_power(f_low, f_high):
            mask = (f >= f_low) & (f <= f_high)
            return np.mean(pxx[mask]) if mask.any() else 0
        
        p_1900 = band_power(1850, 1950)
        p_1300 = band_power(1250, 1350)
        p_1200 = band_power(1150, 1250)
        p_total = np.mean(pxx) + 1e-12
        
        print(f'   1900 Hz (VIS): {p_1900/p_total:.4f}', flush=True)
        print(f'   1300 Hz (бит 0): {p_1300/p_total:.4f}', flush=True)
        print(f'   1200 Hz (старт): {p_1200/p_total:.4f}', flush=True)
        
        # SSTV определяется по наличию 1900 Hz лидера
        if p_1900 > p_total * 0.05:
            print('   ⚠️  Возможен SSTV сигнал!', flush=True)
        else:
            print('   ℹ️  SSTV не обнаружен', flush=True)
    except Exception as e:
        print(f'   ⚠️  Анализ: {e}', flush=True)
    
    # Сохранение WAV
    print('\n💾 Сохранение WAV...', flush=True)
    wav_file = 'data/sstv/iss_capture.wav'
    os.makedirs('data/sstv', exist_ok=True)
    
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(wav_file, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(pcm.tobytes())
    
    print(f'✅ WAV: {wav_file}', flush=True)
    print(f'   {len(audio)/44100:.1f} сек, {len(pcm)/1024:.1f} KB', flush=True)
    
    # Спектр
    print('\n📊 Спектр...', flush=True)
    fft_size = 4096
    hann = np.hanning(fft_size)
    fft_result = np.fft.fftshift(np.fft.fft(iq_all[:fft_size] * hann))
    power_db = 10 * np.log10(np.abs(fft_result) ** 2 + 1e-12)
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, 1.0/2400000)) / 1e6 + 145.800
    
    peak_idx = np.argmax(power_db)
    print(f'   Пик: {freqs[peak_idx]:.3f} MHz ({power_db[peak_idx]:.1f} dB)', flush=True)
    
    sdr.close()
    print('\n🔌 Готово!', flush=True)
    
    # Попробуем декодировать SSTV
    print('\n🔍 Попытка SSTV декодирования...', flush=True)
    try:
        from pysstv.sstv import SSTV
        
        audio_float = audio.astype(np.float32) / 32768.0
        sstv = SSTV(audio_float.tolist(), 44100, 16)
        
        print('   Декодирование (может занять время)...', flush=True)
        img = sstv.decode()
        
        if img and hasattr(img, 'size') and img.size[0] > 0:
            img_path = 'data/sstv/iss_decoded.png'
            img.save(img_path)
            print(f'✅ SSTV декодировано: {img.size[0]}x{img.size[1]}', flush=True)
            print(f'🖼️  {img_path}', flush=True)
        else:
            print('ℹ️  SSTV изображение не найдено', flush=True)
    except ImportError:
        print('⚠️  pysstv не установлен', flush=True)
    except Exception as e:
        print(f'ℹ️  SSTV не обнаружен: {e}', flush=True)


if __name__ == '__main__':
    main()
