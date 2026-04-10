#!/usr/bin/env python
"""
🚀 RTL-SDR V4 Control Panel
Интерактивная панель управления RTL-SDR
"""

import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

def print_header():
    print("=" * 70)
    print("  🚀 RTL-SDR V4 CONTROL PANEL")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

def print_menu():
    print("\n📋 МЕНЮ:")
    print("  [1] Проверка устройства")
    print("  [2] Спектральный анализ")
    print("  [3] Запись сигнала")
    print("  [4] Приём SSTV (МКС 145.800 МГц)")
    print("  [5] Сканирование диапазона")
    print("  [6] Информация об устройстве")
    print("  [7] Мониторинг сигнала")
    print("  [0] Выход")
    print()

def check_device():
    """Проверка устройства"""
    print("\n" + "=" * 70)
    print("  ПРОВЕРКА УСТРОЙСТВА")
    print("=" * 70)
    
    try:
        from rtlsdr import RtlSdr
        count = RtlSdr.get_device_count()
        print(f"\n✅ Найдено устройств: {count}")
        
        if count > 0:
            print("\nПопытка открытия устройства...")
            sdr = RtlSdr()
            print("✅ Устройство открыто успешно!")
            print(f"   Tuner: {sdr.get_tuner_type()}")
            sdr.close()
        else:
            print("\n❌ Устройства не найдены!")
            print("\n💡 Решение:")
            print("   1. Проверьте USB подключение")
            print("   2. Запустите Zadig и установите WinUSB драйвер")
            
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        print("\n💡 Запустите check_zadig_drivers.bat для диагностики")

def spectrum_analysis():
    """Спектральный анализ"""
    print("\n" + "=" * 70)
    print("  СПЕКТРАЛЬНЫЙ АНАЛИЗ")
    print("=" * 70)
    
    try:
        from api.sstv.rtl_sstv_receiver import RTLSDRReceiver
        
        freq = float(input("\nЧастота (МГц) [145.800]: ") or 145.800)
        points = int(input("Количество точек [4096]: ") or 4096)
        
        receiver = RTLSDRReceiver(frequency=freq, gain=49.6)
        if not receiver.initialize():
            print("❌ Не удалось инициализировать устройство")
            return
        
        print(f"\n📊 Анализ спектра на {freq} МГц...")
        freqs, power = receiver.get_spectrum(points)
        
        if freqs is not None:
            print(f"\n✅ Спектр получен:")
            print(f"   Диапазон: {freqs[0]/1e6:.3f} - {freqs[-1]/1e6:.3f} МГц")
            print(f"   Мощность: {np.min(power):.1f} - {np.max(power):.1f} дБ")
            print(f"   Средний SNR: {np.max(power) - np.mean(power):.1f} дБ")
            
            # Визуализация
            print("\n📈 Спектр (визуализация):")
            max_power = np.max(power)
            for i in range(0, len(power), len(power)//50):
                bar_len = int((power[i] - np.min(power)) / (max_power - np.min(power)) * 40)
                print(f"   {'█' * bar_len} {freqs[i]/1e6:.3f} МГц")
        else:
            print("❌ Не удалось получить спектр")
        
        receiver.close()
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

def record_signal():
    """Запись сигнала"""
    print("\n" + "=" * 70)
    print("  ЗАПИСЬ СИГНАЛА")
    print("=" * 70)
    
    try:
        from api.sstv.rtl_sstv_receiver import RTLSDRReceiver
        
        freq = float(input("\nЧастота (МГц) [145.800]: ") or 145.800)
        duration = float(input("Длительность (сек) [10]: ") or 10.0)
        
        receiver = RTLSDRReceiver(frequency=freq, gain=49.6)
        if not receiver.initialize():
            print("❌ Не удалось инициализировать устройство")
            return
        
        print(f"\n🎙️ Запись {duration} сек на {freq} МГц...")
        samples = receiver.record_audio(duration=duration, sample_rate=48000)
        
        if samples is not None:
            filename = f"recording_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.wav"
            receiver._save_wav(samples, filename, 48000)
            print(f"\n✅ Запись сохранена: {filename}")
            print(f"   Длительность: {len(samples)/48000:.2f} сек")
        else:
            print("❌ Ошибка записи")
        
        receiver.close()
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")

def receive_sstv():
    """Приём SSTV"""
    print("\n" + "=" * 70)
    print("  ПРИЁМ SSTV (МКС)")
    print("=" * 70)
    
    try:
        from api.sstv.rtl_sstv_receiver import RTLSDRReceiver, SSTVDecoder
        
        duration = float(input("\nДлительность приёма (сек) [60]: ") or 60.0)
        
        receiver = RTLSDRReceiver(frequency=145.800, gain=49.6)
        decoder = SSTVDecoder(mode='auto')
        
        if not receiver.initialize():
            print("❌ Не удалось инициализировать устройство")
            return
        
        print(f"\n🛰️ Приём SSTV {duration} сек...")
        print("   Частота: 145.800 МГц (МКС)")
        print("   Усиление: 49.6 dB")
        
        samples = receiver.record_audio(duration=duration, sample_rate=48000)
        
        if samples is not None:
            print("\n🔍 Декодирование SSTV...")
            image = decoder.decode_audio(samples, sample_rate=48000)
            
            if image:
                filename = f"sstv_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.png"
                image.save(filename)
                print(f"\n✅ SSTV изображение сохранено: {filename}")
                print(f"   Размер: {image.size}")
            else:
                print("\n❌ SSTV сигнал не обнаружен")
                print("   💡 Попробуйте увеличить время приёма")
        else:
            print("❌ Ошибка записи")
        
        receiver.close()
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

def scan_range():
    """Сканирование диапазона"""
    print("\n" + "=" * 70)
    print("  СКАНИРОВАНИЕ ДИАПАЗОНА")
    print("=" * 70)
    
    try:
        from api.sstv.rtl_sstv_receiver import RTLSDRReceiver
        
        start = float(input("\nНачальная частота (МГц) [145.0]: ") or 145.0)
        end = float(input("Конечная частота (МГц) [146.0]: ") or 146.0)
        step = float(input("Шаг (МГц) [0.1]: ") or 0.1)
        
        receiver = RTLSDRReceiver(frequency=start, gain=49.6)
        if not receiver.initialize():
            print("❌ Не удалось инициализировать устройство")
            return
        
        print(f"\n🔍 Сканирование {start}-{end} МГц (шаг {step} МГц)...")
        
        freq = start
        max_strength = 0
        max_freq = start
        
        while freq <= end:
            receiver.sdr.fc = freq * 1e6
            time.sleep(0.1)
            
            strength = receiver.get_signal_strength()
            bar_len = int(strength / 2)
            
            print(f"   {'█' * bar_len} {freq:.3f} МГц: {strength:.1f}%")
            
            if strength > max_strength:
                max_strength = strength
                max_freq = freq
            
            freq += step
        
        print(f"\n📊 Максимальная активность:")
        print(f"   Частота: {max_freq:.3f} МГц")
        print(f"   Сила: {max_strength:.1f}%")
        
        receiver.close()
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")

def device_info():
    """Информация об устройстве"""
    print("\n" + "=" * 70)
    print("  ИНФОРМАЦИЯ ОБ УСТРОЙСТВЕ")
    print("=" * 70)
    
    try:
        from api.sstv.rtl_sstv_receiver import RTLSDRReceiver
        
        receiver = RTLSDRReceiver()
        if not receiver.initialize():
            print("❌ Не удалось инициализировать устройство")
            return
        
        info = receiver.get_device_info()
        
        print("\n📡 Устройство:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        receiver.close()
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")

def signal_monitor():
    """Мониторинг сигнала"""
    print("\n" + "=" * 70)
    print("  МОНИТОРИНГ СИГНАЛА")
    print("=" * 70)
    
    try:
        from api.sstv.rtl_sstv_receiver import RTLSDRReceiver
        
        freq = float(input("\nЧастота мониторинга (МГц) [145.800]: ") or 145.800)
        duration = int(input("Длительность (сек) [30]: ") or 30)
        
        receiver = RTLSDRReceiver(frequency=freq, gain=49.6)
        if not receiver.initialize():
            print("❌ Не удалось инициализировать устройство")
            return
        
        print(f"\n📈 Мониторинг {duration} сек...")
        print("   Нажмите Ctrl+C для остановки\n")
        
        start_time = time.time()
        max_strength = 0
        
        try:
            while time.time() - start_time < duration:
                strength = receiver.get_signal_strength()
                bar_len = int(strength / 2)
                timestamp = datetime.now(timezone.utc).strftime('%H:%M:%S')
                
                print(f"   [{timestamp}] {'█' * bar_len} {strength:.1f}%")
                
                if strength > max_strength:
                    max_strength = strength
                
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n⏹️ Мониторинг остановлен")
        
        print(f"\n📊 Результат:")
        print(f"   Максимальная сила: {max_strength:.1f}%")
        
        receiver.close()
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")

def main():
    print_header()
    
    while True:
        print_menu()
        
        choice = input("Выберите действие [0-7]: ").strip()
        
        if choice == '0':
            print("\n👋 До свидания!")
            break
        elif choice == '1':
            check_device()
        elif choice == '2':
            spectrum_analysis()
        elif choice == '3':
            record_signal()
        elif choice == '4':
            receive_sstv()
        elif choice == '5':
            scan_range()
        elif choice == '6':
            device_info()
        elif choice == '7':
            signal_monitor()
        else:
            print("\n❌ Неверный выбор. Попробуйте снова.")
        
        input("\nНажмите Enter для продолжения...")

if __name__ == '__main__':
    main()
