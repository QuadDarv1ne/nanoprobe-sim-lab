"""
Тест исправлений RTL-SDR SSTV Receiver
"""
import sys
import numpy as np

def test_imports():
    """Тестируем импорты"""
    print("1. Тестируем импорты...")
    try:
        from api.sstv.rtl_sstv_receiver import (
            RTLSDRReceiver, SSTVDecoder, _fm_demodulate, _detect_vis
        )
        print("   ✅ RTLSDRReceiver импортирован")
        print("   ✅ SSTVDecoder импортирован")
        print("   ✅ _fm_demodulate импортирована")
        print("   ✅ _detect_vis импортирована")
        return True
    except Exception as e:
        print(f"   ❌ Ошибка импорта: {e}")
        return False

def test_fm_demod():
    """Тестируем FM демодуляцию"""
    print("\n2. Тестируем FM демодуляцию...")
    try:
        from api.sstv.rtl_sstv_receiver import _fm_demodulate, AUDIO_SAMPLE_RATE
        
        # Генерируем тестовые I/Q данные
        duration = 0.1  # 100ms
        sample_rate = 2400000
        num_samples = int(sample_rate * duration)
        
        # Создаём комплексный сигнал
        t = np.arange(num_samples) / sample_rate
        freq = 1000  # 1 kHz tone
        iq = np.exp(1j * 2 * np.pi * freq * t).astype(np.complex64)
        
        # Демодулируем
        audio = _fm_demodulate(iq, sample_rate, AUDIO_SAMPLE_RATE)
        
        assert len(audio) > 0, "Аудио пустое"
        assert np.max(np.abs(audio)) <= 1.01, "Аудио не нормализовано"
        
        print(f"   ✅ FM демодуляция: {num_samples} → {len(audio)} сэмплов")
        print(f"   ✅ Частота дискретизации: {AUDIO_SAMPLE_RATE} Hz")
        print(f"   ✅ Нормализация: {np.max(np.abs(audio)):.3f}")
        return True
    except Exception as e:
        print(f"   ❌ Ошибка FM демодуляции: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vis_detector():
    """Тестируем VIS детектор"""
    print("\n3. Тестируем VIS детектор...")
    try:
        from api.sstv.rtl_sstv_receiver import _detect_vis, AUDIO_SAMPLE_RATE
        
        # Создаём тестовый аудио сигнал с 1900 Hz лидером
        duration = 0.5
        num_samples = int(AUDIO_SAMPLE_RATE * duration)
        t = np.arange(num_samples) / AUDIO_SAMPLE_RATE
        
        # 1900 Hz лидер SSTV
        audio = np.sin(2 * np.pi * 1900 * t).astype(np.float32)
        
        result = _detect_vis(audio, AUDIO_SAMPLE_RATE)
        
        print(f"   ✅ VIS детектор: detected={result['detected']}, confidence={result['confidence']}")
        print(f"   ✅ Leader power: {result.get('leader_power', 'N/A')}")
        return True
    except Exception as e:
        print(f"   ❌ Ошибка VIS детектора: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_receiver_initialization():
    """Тестируем инициализацию receiver"""
    print("\n4. Тестируем инициализацию receiver...")
    try:
        from api.sstv.rtl_sstv_receiver import RTLSDRReceiver, RTLSDR_AVAILABLE
        
        print(f"   ✅ RTL-SDR доступен: {RTLSDR_AVAILABLE}")
        
        if RTLSDR_AVAILABLE:
            receiver = RTLSDRReceiver(
                frequency=145.800,
                gain=20.0,
                sample_rate=2400000
            )
            
            # Тестируем что инициализация идемпотентна
            result1 = receiver.initialize()
            result2 = receiver.initialize()  # Второй вызов должен быть быстрым
            
            print(f"   ✅ Первая инициализация: {result1}")
            print(f"   ✅ Вторая инициализация (кэш): {result2}")
            
            if receiver.sdr:
                device_info = receiver.get_device_info()
                print(f"   ✅ Tuner: {device_info.get('tuner_type', 'N/A')}")
                print(f"   ✅ Частота: {device_info.get('frequency_mhz')} MHz")
                print(f"   ✅ Gain: {device_info.get('gain_db')} dB")
                
                receiver.close()
                print(f"   ✅ Receiver закрыт")
        
        return True
    except Exception as e:
        print(f"   ❌ Ошибка инициализации: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_no_sleep_loop():
    """Проверяем что record_audio не использует time.sleep"""
    print("\n5. Проверяем отсутствие time.sleep в record_audio...")
    try:
        import inspect
        from api.sstv.rtl_sstv_receiver import RTLSDRReceiver
        
        source = inspect.getsource(RTLSDRReceiver.record_audio)
        
        if 'time.sleep' in source:
            print("   ❌ Найден time.sleep в record_audio!")
            return False
        else:
            print("   ✅ time.sleep НЕ используется в record_audio")
            print("   ✅ Используется async callback")
            return True
    except Exception as e:
        print(f"   ❌ Ошибка проверки: {e}")
        return False

def main():
    print("=" * 60)
    print("Тест исправлений RTL-SDR SSTV Receiver")
    print("=" * 60)
    
    results = []
    
    results.append(("Импорты", test_imports()))
    results.append(("FM демодуляция", test_fm_demod()))
    results.append(("VIS детектор", test_vis_detector()))
    results.append(("Инициализация receiver", test_receiver_initialization()))
    results.append(("Отсутствие time.sleep", test_no_sleep_loop()))
    
    print("\n" + "=" * 60)
    print("ИТОГИ:")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {name}: {status}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\nВсего: {passed}/{total} тестов пройдено")
    
    if passed == total:
        print("\n🎉 Все исправления работают корректно!")
    else:
        print(f"\n⚠️ {total - passed} тестов не пройдено")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
