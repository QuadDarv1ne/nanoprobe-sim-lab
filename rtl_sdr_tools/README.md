# RTL-SDR Tools

Набор скриптов для работы с RTL-SDR V4 (RTLSDRBlog V4, R828D tuner)

## Скрипты

| Скрипт | Назначение | Статус |
|--------|------------|--------|
| `adsb_capture.py` | Захват ADS-B данных (1090 MHz) через Python API | ✅ |
| `fm_radio_capture.py` | Захват FM-радио (88-108 MHz) через Python API | ✅ |
| `fm_radio_scanner.py` | Сканирование FM-диапазона + захват станций | ✅ |
| `fm_capture_simple.py` | Простой захват FM-радио через rtl_fm | ✅ |
| `fm_multi_capture.py` | Многостанционный захват FM-радио | ✅ |
| `raw_to_wav.py` | Конвертация RAW аудио → WAV | ✅ |

## Инструменты

| Утилита | Путь | Назначение |
|---------|------|------------|
| rtl_adsb | `../tools/rtl-sdr-blog/x64/rtl_adsb.exe` | ADS-B декодер (1090 MHz) |
| rtl_fm | `../tools/rtl-sdr-blog/x64/rtl_fm.exe` | FM демодулятор |
| rtl_power | `../tools/rtl-sdr-blog/x64/rtl_power.exe` | Сканер спектра |
| rtl_test | `../tools/rtl-sdr-blog/x64/rtl_test.exe` | Тест устройства |

## Быстрый старт

### ADS-B (1090 MHz)
```bash
python adsb_capture.py
# Или напрямую:
../tools/rtl-sdr-blog/x64/rtl_adsb.exe -V > adsb.txt
```

### FM-радио
```bash
python fm_multi_capture.py
# Или напрямую:
../tools/rtl-sdr-blog/x64/rtl_fm.exe -f 106.0M -M wbfm -s 32k -g 40 -E deemp output.raw
```

### Конвертация RAW → WAV
```bash
python raw_to_wav.py input.raw
```

## Оборудование

- **RTL-SDR V4**: RTLSDRBlog V4
- **Тюнер**: Rafael Micro R828D
- **SN**: 00000001
- **Драйверы**: Zadig (WinUSB)

## Зависимости Python

- rtlsdr 0.2.93
- numpy 2.4.2
- wave (built-in)

## Данные

Все данные сохраняются в `../data/`:
- `data/adsb/` — ADS-B сообщения и I/Q сэмплы
- `data/fm_radio/` — FM-радио аудио (WAV/RAW)
- `data/sstv/` — SSTV захваты
- `data/noaa/` — NOAA APT захваты
