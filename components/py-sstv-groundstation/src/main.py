# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Наземная станция SSTV
Скрипт для приема и декодирования SSTV-сигналов
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sstv_decoder import SSTVDecoder, detect_sstv_signal


def main():
    parser = argparse.ArgumentParser(description='Наземная станция SSTV')
    parser.add_argument('--audio', '-a', type=str, help='Путь к аудиофайлу')
    parser.add_argument('--device', '-d', type=str, default='default',
                        help='SDR устройство (по умолчанию: default)')
    parser.add_argument('--frequency', '-f', type=float, default=145.500,
                        help='Частота приема в МГц (по умолчанию: 145.500)')
    parser.add_argument('--output', '-o', type=str, help='Путь для сохранения изображения')
    parser.add_argument('--detect', action='store_true', help='Обнаружить SSTV сигнал')
    
    args = parser.parse_args()
    
    print("=== НАЗЕМНАЯ СТАНЦИЯ SSTV ===")
    decoder = SSTVDecoder()
    
    if args.detect:
        if not args.audio:
            print("Ошибка: Для обнаружения сигнала укажите --audio")
            sys.exit(1)
        
        audio_path = Path(args.audio)
        if not audio_path.exists():
            print(f"Ошибка: Файл '{audio_path}' не найден")
            sys.exit(1)
        
        print(f"Обнаружение SSTV сигнала в: {audio_path}")
        result = detect_sstv_signal(str(audio_path))
        if result:
            print(f"SSTV сигнал обнаружен: {result}")
        else:
            print("SSTV сигнал не обнаружен")
    
    elif args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            print(f"Ошибка: Файл '{audio_path}' не найден")
            sys.exit(1)
        
        print(f"Декодирование аудио: {audio_path}")
        image = decoder.decode_from_audio(str(audio_path))
        
        if image:
            output = args.output or "decoded_sstv_image.png"
            decoder.save_decoded_image(output)
            print(f"Изображение сохранено в: {output}")
        else:
            print("Не удалось декодировать изображение")
    
    else:
        print("Режим демонстрации:")
        print("- Декодировать аудио: --audio <файл>")
        print("- Обнаружить сигнал: --detect --audio <файл>")
        print("- Прием с SDR: --device <устройство> --frequency <МГц>")

if __name__ == "__main__":
    main()

