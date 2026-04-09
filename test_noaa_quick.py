"""Быстрый тест NOAA 19 - 10 секунд."""
import time
import numpy as np
from rtlsdr import RtlSdr
from scipy.signal import resample_poly, firwin, lfilter

print('🛰️  RTL-SDR NOAA 19 Quick Test', flush=True)
print('='*60, flush=True)

# NOAA 19: 137.100 MHz
print('\n🔌 Подключение...', flush=True)
sdr = RtlSdr()
sdr.rs = 2400000
sdr.fc = int(137.100 * 1e6)
sdr.gain = 30

print(f'✅ Tuner: {sdr.get_tuner_type()}', flush=True)
print(f'📡 NOAA 19: 137.100 MHz', flush=True)

# Запись 10 секунд
duration = 10.0
print(f'\n⏺️  Запись {duration} сек...', flush=True)
num_samples = int(2.4e6 * duration)
chunks = []
chunk_size = 256 * 1024
start = time.time()
last_report = 0

while len(chunks) * chunk_size < num_samples:
    samples = sdr.read_samples(chunk_size)
    chunks.append(samples)
    
    elapsed = time.time() - start
    if int(elapsed) > last_report:
        progress = min(100, (len(chunks) * chunk_size / num_samples) * 100)
        print(f'   {progress:.0f}% ({elapsed:.0f}/{duration:.0f}с)', flush=True, end='\r')
        last_report = int(elapsed)

total_time = time.time() - start
iq_all = np.concatenate(chunks)
print(f'\n✅ Записано: {len(iq_all)/1e6:.1f}M I/Q ({total_time:.1f}с)', flush=True)

# FM демодуляция
print('\n📻 FM демодуляция...', flush=True)
phase = np.angle(iq_all)
audio = np.diff(np.unwrap(phase)).astype(np.float64)

# Ресемплинг до 11025 Hz
nyquist = 2400000 / 2.0
fir = firwin(127, (11025/2.0 * 0.9) / nyquist)
audio = lfilter(fir, 1.0, audio)
audio = resample_poly(audio, 11025, 2400000).astype(np.float32)

peak = np.max(np.abs(audio))
if peak > 0:
    audio /= peak

print(f'✅ Аудио: {len(audio)/11025:.1f} сек @ 11025 Hz', flush=True)

# Сохраняем WAV
import wave
import os
wav_file = 'data/noaa/noaa19_test.wav'
os.makedirs('data/noaa', exist_ok=True)

pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
with wave.open(wav_file, 'w') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(11025)
    wf.writeframes(pcm.tobytes())

print(f'💾 WAV: {wav_file}', flush=True)

# Анализ NOAA APT
print('\n📊 Анализ NOAA APT...', flush=True)
try:
    from scipy.signal import welch
    
    segment = audio[:min(len(audio), 11025*5)]
    f, pxx = welch(segment, 11025, nperseg=2048)
    
    p_2400 = np.mean(pxx[(f >= 2350) & (f <= 2450)])
    p_2000 = np.mean(pxx[(f >= 1950) & (f <= 2050)])
    p_1200 = np.mean(pxx[(f >= 1150) & (f <= 1250)])
    p_total = np.mean(pxx) + 1e-12
    
    print(f'   2400 Hz (APT): {p_2400/p_total:.4f}', flush=True)
    print(f'   2000 Hz (канал A): {p_2000/p_total:.4f}', flush=True)
    print(f'   1200 Hz (sync): {p_1200/p_total:.4f}', flush=True)
    
    if p_2400 > p_total * 0.1:
        print('   ✅ NOAA APT ОБНАРУЖЕН!', flush=True)
    else:
        print('   ℹ️  NOAA APT не обнаружен (нет пролёта сейчас)', flush=True)
except Exception as e:
    print(f'   ⚠️  {e}', flush=True)

# Спектр
print('\n📊 Спектр...', flush=True)
fft_size = 4096
hann = np.hanning(fft_size)
fft_result = np.fft.fftshift(np.fft.fft(iq_all[:fft_size] * hann))
power_db = 10 * np.log10(np.abs(fft_result) ** 2 + 1e-12)
freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, 1.0/2400000)) / 1e6 + 137.100

peak_idx = np.argmax(power_db)
print(f'   Пик: {freqs[peak_idx]:.3f} MHz ({power_db[peak_idx]:.1f} dB)', flush=True)
print(f'   Min: {np.min(power_db):.1f} dB, Max: {np.max(power_db):.1f} dB', flush=True)

sdr.close()
print('\n🔌 Готово!', flush=True)
print('\n💡 Для полного пролёта запусти:', flush=True)
print('   python rtl_sdr_noaa_capture.py -d 900  # 15 минут', flush=True)
