# -*- coding: utf-8 -*-
"""
Тест поддержки RTL-SDR V4
Проверка перед использованием
"""

import sys
from pathlib import Path

# Добавляем путь к модулю
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sdr_interface import SDRInterface


def test_rtl_sdr_v4_support():
    """Тест поддержки RTL-SDR V4"""
    print("=" * 60)
    print("ТЕСТ ПОДДЕРЖКИ RTL-SDR V4")
    print("=" * 60)
    
    # 1. Проверка импорта
    print("\n1. Проверка импорта rtlsdr...")
    try:
        from rtlsdr import RtlSdr
        print("   ✓ rtlsdr импортирован")
    except ImportError as e:
        print(f"   ✗ rtlsdr не найден: {e}")
        print("   Установите: pip install rtlsdr pyrtlsdr")
        return False
    
    # 2. Проверка количества устройств
    print("\n2. Поиск RTL-SDR устройств...")
    try:
        num_devices = RtlSdr.get_device_count()
        print(f"   Найдено устройств: {num_devices}")
        if num_devices == 0:
            print("   ⚠ Устройства не найдены. Подключите RTL-SDR V4")
            return False
    except Exception as e:
        print(f"   ✗ Ошибка поиска: {e}")
        return False
    
    # 3. Инициализация интерфейса
    print("\n3. Инициализация SDRInterface...")
    sdr = SDRInterface(
        device_index=0,
        sample_rate=2400000,  # 2.4 MSPS для V4
        center_freq=145.800,
        gain=30,
        device_type='auto'
    )
    
    # 4. Проверка поддержки устройств
    print("\n4. Поддерживаемые устройства:")
    for dev_key, dev_name in sdr.SUPPORTED_DEVICES.items():
        marker = "✓" if 'v4' in dev_name.lower() or 'r828d' in dev_name.lower() else " "
        print(f"   {marker} {dev_key}: {dev_name}")
    
    # 5. Инициализация устройства
    print("\n5. Инициализация устройства...")
    success = sdr.initialize()
    if success:
        print(f"   ✓ Устройство: {sdr.device_name}")
        print(f"   ✓ Sample Rate: {sdr.sample_rate} sps")
        print(f"   ✓ Частота: {sdr.center_freq} МГц")
        print(f"   ✓ Gain: {sdr.gain} dB")
        
        # Проверка что это RTL-SDR V4
        if 'V4' in sdr.device_name or 'R828D' in sdr.device_name:
            print("\n   🎉 RTL-SDR V4 успешно обнаружен и настроен!")
        else:
            print(f"\n   ⚠ Обнаружено устройство: {sdr.device_name}")
            print("   RTL-SDR V4 также поддерживается")
    else:
        print("   ✗ Ошибка инициализации")
        print(f"   {sdr.metadata.get('error', 'Неизвестная ошибка')}")
        return False
    
    # 6. Проверка частот
    print("\n6. Проверка диапазона частот...")
    freq_range = sdr.get_frequency_range()
    print(f"   Диапазон: {freq_range[0]/1e6:.1f} - {freq_range[1]/1e6:.1f} МГц")
    
    # 7. Тестовая запись (короткая)
    print("\n7. Тестовая запись (1 секунда)...")
    try:
        samples = sdr.read_samples(1.0)
        if samples is not None:
            print(f"   ✓ Записано {len(samples)} отсчетов")
            print(f"   ✓ Размер: {samples.nbytes / 1024:.1f} KB")
        else:
            print("   ⚠ Нет данных")
    except Exception as e:
        print(f"   ⚠ Ошибка записи: {e}")
    
    print("\n" + "=" * 60)
    print("ТЕСТ ЗАВЕРШЕН")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_rtl_sdr_v4_support()
    sys.exit(0 if success else 1)
