#!/usr/bin/env python
"""
Тест RTL-SDR V4 + SSTV Receiver
Проверка всех возможностей
"""

import sys
import time
import numpy as np
from pathlib import Path

print("=" * 60)
print("  RTL-SDR V4 SSTV Receiver Test")
print("=" * 60)

# Проверка pyrtlsdr
print("\n[1/5] Проверка pyrtlsdr...")
try:
    from rtlsdr import RtlSdr
    print("✅ pyrtlsdr импортирован")
except ImportError as e:
    print(f"❌ pyrtlsdr не найден: {e}")
    sys.exit(1)

# Проверка pysstv
print("\n[2/5] Проверка pysstv...")
try:
    from pysstv.mode import PD120, MartinM1
    print("✅ pysstv импортирован")
    print(f"   Режимы: PD120, MartinM1 доступны")
except ImportError as e:
    print(f"❌ pysstv не найден: {e}")

# Инициализация RTL-SDR
print("\n[3/5] Инициализация RTL-SDR V4...")
try:
    from api.sstv.rtl_sstv_receiver import RTLSDRReceiver
    
    receiver = RTLSDRReceiver(
        frequency=145.800,  # МКС SSTV
        sample_rate=2.4e6,
        gain=49.6
    )
    
    if receiver.initialize():
        print("✅ RTL-SDR инициализирован")
        info = receiver.get_device_info()
        print(f"   Частота: {info.get('frequency_mhz')} МГц")
        print(f"   Усиление: {info.get('gain_db')} дБ")
        print(f"   Sample rate: {info.get('sample_rate_hz')/1e6:.2f} МГц")
    else:
        print("❌ Не удалось инициализировать RTL-SDR")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Тест чтения сэмплов
print("\n[4/5] Тест чтения сэмплов...")
try:
    samples = receiver.read_samples(1024)
    if samples is not None:
        print(f"✅ Прочитано {len(samples)} сэмплов")
        print(f"   Амплитуда: {np.mean(np.abs(samples)):.4f}")
        print(f"   Макс: {np.max(np.abs(samples)):.4f}")
    else:
        print("❌ Не удалось прочитать сэмплы")
except Exception as e:
    print(f"❌ Ошибка чтения: {e}")

# Тест спектра
print("\n[5/5] Тест спектра...")
try:
    freqs, power = receiver.get_spectrum(num_points=1024)
    if freqs is not None and power is not None:
        print(f"✅ Спектр получен: {len(freqs)} точек")
        print(f"   Диапазон: {freqs[0]/1e6:.3f} - {freqs[-1]/1e6:.3f} МГц")
        print(f"   Мощность: {np.min(power):.1f} - {np.max(power):.1f} дБ")
    else:
        print("❌ Не удалось получить спектр")
except Exception as e:
    print(f"❌ Ошибка спектра: {e}")

# Очистка
receiver.close()
print("\n" + "=" * 60)
print("  Тест завершен!")
print("=" * 60)
