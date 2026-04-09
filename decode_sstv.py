"""
Простой SSTV декодер для Martin/Scottie режимов
Декодирует SSTV из WAV файла используя частотное кодирование пикселей.

SSTV Martin M1:
- VIS: 1900Hz лидер, затем VIS код
- Sync pulse: 1200 Hz, 4.862ms
- Porch: 1500 Hz, 0.572ms  
- Пиксели: 1500-2300 Hz (чёрный-белый), 4.576ms на пиксель
- 320x256 разрешение, RGB по отдельности

Частоты:
- 1500 Hz = чёрный (0)
- 2300 Hz = белый (255)
"""

import wave
import numpy as np
from PIL import Image

def decode_sstv_simple(wav_file, output_file='data/sstv/decoded.png'):
    """Простой SSTV декодер."""
    
    print(f'📂 Чтение: {wav_file}', flush=True)
    with wave.open(wav_file, 'r') as wf:
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)
        sample_rate = wf.getframerate()
    
    print(f'📊 {len(audio)} сэмплов @ {sample_rate} Hz ({len(audio)/sample_rate:.1f}с)', flush=True)
    
    # Частота → яркость: 1500Hz=0, 2300Hz=255
    def freq_to_brightness(freq_hz):
        return int(np.clip((freq_hz - 1500) / 800 * 255, 0, 255))
    
    # Определяем частоту в окне через FFT
    def get_dominant_freq(samples, fs):
        if len(samples) < 64:
            return 0
        fft = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
        if np.max(fft) == 0:
            return 0
        freq_bin = np.argmax(fft)
        return freq_bin * fs / len(samples)
    
    # SSTV Martin M1 параметры
    SAMPLES_PER_PIXEL = int(sample_rate * 0.004576)  # 4.576ms на пиксель
    SYNC_PULSE_SAMPLES = int(sample_rate * 0.004862)  # 4.862ms sync
    PORCH_SAMPLES = int(sample_rate * 0.000572)       # 0.572ms porch
    
    print(f'\n📏 Параметры:', flush=True)
    print(f'   Сэмплов/пиксель: {SAMPLES_PER_PIXEL}', flush=True)
    print(f'   Частота пикселя: {sample_rate/SAMPLES_PER_PIXEL:.1f} Hz/bin', flush=True)
    
    # Сканируем в поисках sync pulse (1200 Hz)
    print(f'\n🔍 Поиск SSTV сигнала...', flush=True)
    
    # Берём первые 10 секунд максимум
    max_samples = min(len(audio), sample_rate * 10)
    audio_segment = audio[:max_samples]
    
    # Ищем паттерн: sync(1200Hz) → porch(1500Hz) → pixels
    found_sstv = False
    start_idx = 0
    
    # Простой поиск: ищем стабильный 1200 Hz
    window_size = SYNC_PULSE_SAMPLES
    for i in range(0, len(audio_segment) - window_size, window_size // 2):
        window = audio_segment[i:i+window_size]
        freq = get_dominant_freq(window, sample_rate)
        
        # Sync pulse ~1200 Hz (±100 Hz)
        if 1100 < freq < 1300:
            # Проверяем что за ним идёт porch ~1500 Hz
            porch_start = i + window_size
            if porch_start + PORCH_SAMPLES < len(audio_segment):
                porch = audio_segment[porch_start:porch_start+PORCH_SAMPLES]
                porch_freq = get_dominant_freq(porch, sample_rate)
                
                if 1400 < porch_freq < 1600:
                    start_idx = i
                    found_sstv = True
                    print(f'✅ SSTV найден на {i/sample_rate:.3f}с', flush=True)
                    break
    
    if not found_sstv:
        print('❌ SSTV сигнал не найден', flush=True)
        print('   Это может быть просто шум/фон', flush=True)
        return False
    
    # Декодируем изображение
    # Martin M1: 320x256, RGB отдельно, ~3 строки на цвет
    width = 320
    height = 256
    
    print(f'\n🖼️  Декодирование {width}x{height}...', flush=True)
    
    # Создаём изображение
    img = Image.new('RGB', (width, height), (0, 0, 0))
    
    # Позиция после sync+porch
    pos = start_idx + SYNC_PULSE_SAMPLES + PORCH_SAMPLES
    
    # Для каждой строки (RGB)
    total_pixels = width * height * 3  # RGB
    
    if pos + total_pixels * SAMPLES_PER_PIXEL > len(audio_segment):
        print(f'⚠️  Недостаточно данных ({total_pixels * SAMPLES_PER_PIXEL} > {len(audio_segment) - pos})', flush=True)
        print('   Возможно SSTV режим не Martin M1', flush=True)
    
    pixels_decoded = 0
    for row in range(height):
        for channel in range(3):  # RGB
            for col in range(width):
                if pos + SAMPLES_PER_PIXEL > len(audio_segment):
                    break
                
                pixel_samples = audio_segment[pos:pos+SAMPLES_PER_PIXEL]
                freq = get_dominant_freq(pixel_samples, sample_rate)
                brightness = freq_to_brightness(freq)
                
                # Определяем RGB значение
                if channel == 0:  # Red
                    r, g, b = brightness, 0, 0
                elif channel == 1:  # Green
                    r, g, b = 0, brightness, 0
                else:  # Blue
                    r, g, b = 0, 0, brightness
                
                # Читаем текущий пиксель и добавляем канал
                current = img.getpixel((col, row))
                if channel == 0:
                    img.putpixel((col, row), (brightness, current[1], current[2]))
                elif channel == 1:
                    img.putpixel((col, row), (current[0], brightness, current[2]))
                else:
                    img.putpixel((col, row), (current[0], current[1], brightness))
                
                pos += SAMPLES_PER_PIXEL
                pixels_decoded += 1
        
        if pos + SAMPLES_PER_PIXEL > len(audio_segment):
            break
        
        if row % 50 == 0:
            print(f'   Строка {row}/{height} ({pixels_decoded} пикселей)', flush=True, end='\r')
    
    print(f'\n✅ Декодировано {pixels_decoded} пикселей', flush=True)
    
    # Сохраняем
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    img.save(output_file)
    print(f'💾 Изображение: {output_file}', flush=True)
    print(f'   Размер: {img.size}', flush=True)
    
    return True


if __name__ == '__main__':
    import sys
    wav_file = sys.argv[1] if len(sys.argv) > 1 else 'data/sstv/iss_capture.wav'
    success = decode_sstv_simple(wav_file)
    sys.exit(0 if success else 1)
