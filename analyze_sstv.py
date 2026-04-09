"""
Анализ и попытка декодирования SSTV из WAV файла.
"""
import wave
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Без GUI
import matplotlib.pyplot as plt

wav_file = 'data/sstv/iss_capture.wav'

print(f'📂 Анализ: {wav_file}', flush=True)

# Читаем WAV
with wave.open(wav_file, 'r') as wf:
    frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    sample_rate = wf.getframerate()
    duration = len(audio) / sample_rate

print(f'📊 Длительность: {duration:.2f} сек', flush=True)
print(f'📈 Sample rate: {sample_rate} Hz', flush=True)
print(f'🔊 Сэмплов: {len(audio)}', flush=True)
print(f'📊 Мин: {np.min(audio):.3f}, Макс: {np.max(audio):.3f}', flush=True)

# Спектральный анализ
print('\n📊 Спектральный анализ...', flush=True)
from scipy.signal import welch

f, pxx = welch(audio, sample_rate, nperseg=4096)

# Ищем SSTV частоты
def band_power(f_low, f_high):
    mask = (f >= f_low) & (f <= f_high)
    return np.mean(pxx[mask]) if mask.any() else 0

p_1900 = band_power(1850, 1950)
p_1300 = band_power(1250, 1350)
p_1200 = band_power(1150, 1250)
p_1500 = band_power(1450, 1550)
p_2300 = band_power(2250, 2350)
p_total = np.mean(pxx) + 1e-12

print(f'   1900 Hz (VIS лидер): {p_1900/p_total:.4f}', flush=True)
print(f'   1500 Hz (белый): {p_1500/p_total:.4f}', flush=True)
print(f'   1300 Hz (чёрный): {p_1300/p_total:.4f}', flush=True)
print(f'   1200 Hz (sync): {p_1200/p_total:.4f}', flush=True)
print(f'   2300 Hz (макс): {p_2300/p_total:.4f}', flush=True)

# Определяем есть ли SSTV
sstv_freqs = [p_1900, p_1500, p_1300, p_1200, p_2300]
sstv_ratio = np.mean([p/p_total for p in sstv_freqs])

print(f'\n📊 SSTV индекс: {sstv_ratio:.4f}', flush=True)

if sstv_ratio > 0.5:
    print('✅ SSTV сигнал ОБНАРУЖЕН!', flush=True)
else:
    print('ℹ️  SSTV сигнал не найден (фоновый шум)', flush=True)

# Визуализация спектра
print('\n📊 Создание спектрограммы...', flush=True)

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle(f'SSTV Analysis: {wav_file}', fontsize=14, fontweight='bold')

# 1. Waveform
axes[0].plot(audio[:min(10000, len(audio))], linewidth=0.5, color='#00ff88')
axes[0].set_title('Audio Waveform (first 0.23s)')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True, alpha=0.3)
axes[0].set_facecolor('#1a1a2e')

# 2. Power Spectrum
axes[1].plot(f/1000, 10*np.log10(pxx + 1e-12), linewidth=0.8, color='#00aaff')
axes[1].set_title('Power Spectrum')
axes[1].set_xlabel('Frequency (kHz)')
axes[1].set_ylabel('Power (dB)')
axes[1].grid(True, alpha=0.3)
axes[1].set_facecolor('#1a1a2e')

# Отмечаем SSTV частоты
for freq_hz, color in [(1200, 'red'), (1300, 'yellow'), (1500, 'green'), (1900, 'magenta'), (2300, 'cyan')]:
    axes[1].axvline(x=freq_hz/1000, color=color, alpha=0.5, linestyle='--', linewidth=0.8)
    axes[1].text(freq_hz/1000, 0, f'{freq_hz}', color=color, fontsize=8, rotation=90, va='bottom')

# 3. Spectrogram
nperseg = 1024
im = axes[2].specgram(audio, Fs=sample_rate, NFFT=nperseg, noverlap=nperseg//2,
                       cmap='viridis', vmin=-80, vmax=-20)[3]
axes[2].set_title('Spectrogram')
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Frequency (kHz)')
fig.colorbar(im, ax=axes[2], label='Power (dB)')

plt.tight_layout()
output_png = 'data/sstv/analysis.png'
plt.savefig(output_png, dpi=150, bbox_inches='tight', facecolor='#16213e')
print(f'✅ Спектрограмма сохранена: {output_png}', flush=True)

# Попробуем декодировать SSTV через другие методы
print('\n🔍 Поиск SSTV декодеров...', flush=True)

# Проверим есть ли другие SSTV библиотеки
try:
    import sstv as sstv_module
    print('✅ found: sstv module', flush=True)
except ImportError:
    print('ℹ️  sstv module не найден', flush=True)

try:
    from pillow_sstv import decode
    print('✅ pillow-sstv найден!', flush=True)
except ImportError:
    print('ℹ️  pillow-sstv не установлен', flush=True)

# Попробуем декодировать через pysstv (может есть другие методы)
print('\n🔍 Попытка pysstv декодирования...', flush=True)
try:
    from pysstv.color import MartinM1
    import inspect
    
    # Проверяем есть ли метод decode_img или похожий
    methods = [m for m in dir(MartinM1) if 'decode' in m.lower() or 'read' in m.lower()]
    print(f'   Методы: {methods}', flush=True)
    
    # Может можно использовать gen_image_tuples наоборот?
    if 'gen_image' in str(dir(MartinM1)):
        print('   ℹ️  pysstv только генератор, не декодер', flush=True)
except Exception as e:
    print(f'   ⚠️  {e}', flush=True)

print('\n💡 Рекомендации:', flush=True)
print('   Для SSTV декодирования установи:', flush=True)
print('   1. pillow-sstv: pip install pillow-sstv', flush=True)
print('   2. Или используй MMSSTV (Windows)', flush=True)
print('   3. Или QSSTV (Linux)', flush=True)
print('\n✅ Анализ завершён!', flush=True)
