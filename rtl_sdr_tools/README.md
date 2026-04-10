# RTL-SDR Tools

Набор инструментов для работы с RTL-SDR V4 (RTLSDRBlog V4, R828D tuner)

## Оборудование
- **Устройство:** RTL-SDR Blog V4
- **Тюнер:** Rafael Micro R828D
- **Serial:** 00000001
- **Драйверы:** Zadig (WinUSB)

## Скрипты (20 файлов)

### ADS-B (1090 MHz) — отслеживание самолётов
| Скрипт | Описание |
|--------|----------|
| `adsb_capture.py` | Захват I/Q данных через Python API |
| `adsb_receiver.py` | ADS-B приёмник (из legacy) |

### FM-радио (88-108 MHz)
| Скрипт | Описание |
|--------|----------|
| `fm_radio.py` | Простой FM-приёмник (из legacy) |
| `fm_radio_capture.py` | Захват через Python API |
| `fm_radio_scanner.py` | Сканирование диапазона |
| `fm_capture_simple.py` | Простой захват через rtl_fm |
| `fm_multi_capture.py` | Многостанционный захват |

### Airband (118-137 MHz) — авиасвязь
| Скрипт | Описание |
|--------|----------|
| `am_airband.py` | AM-демодуляция авиасвязи |
| `listen_airband.py` | Прослушивание авиаканалов |
| `quick_scan_airband.py` | Быстрое сканирование |

### RTL_433 (433/868/915 MHz) — датчики
| Скрипт | Описание |
|--------|----------|
| `rtl433_scanner.py` | Сканер ISM-устройств |

### SSTV/NOAA (137/145 MHz) — спутники
| Скрипт | Описание |
|--------|----------|
| `capture_sstv_mmsstv.py` | SSTV capture + MMSSTV |
| `rtl_sdr_sstv_capture.py` | SSTV захват |
| `rtl_sdr_noaa_capture.py` | NOAA APT захват |
| `rtl_sdr_visualizer.py` | Спектр + waterfall |
| `sstv_ground_station.py` | Наземная станция |
| `iss_tracker.py` | Трекер МКС (SGP4) |

### Утилиты
| Скрипт | Описание |
|--------|----------|
| `raw_to_wav.py` | Конвертация RAW → WAV |
| `rtlsdr_control_panel.py` | Панель управления |
| `listen_fm_radio.py` | FM-радио плеер |

## Инструменты (`../tools/`)

| Утилита | Путь | Назначение |
|---------|------|------------|
| rtl_adsb | `../tools/rtl-sdr-blog/x64/` | ADS-B декодер |
| rtl_fm | `../tools/rtl-sdr-blog/x64/` | FM/AM демодулятор |
| rtl_power | `../tools/rtl-sdr-blog/x64/` | Сканер спектра |
| rtl_test | `../tools/rtl-sdr-blog/x64/` | Тест устройства |
| rtl_433 | `../tools/rtl_433/` | ISM декодер (253 протокола) |
| dump1090 | `../tools/dump1090/` | ADS-B (⚠️ не поддерживает V4) |

## Данные (`../data/`)

| Папка | Содержимое |
|-------|------------|
| `data/adsb/` | ADS-B сообщения и I/Q сэмплы |
| `data/airband/` | Авиасвязь данные |
| `data/fm_radio/` | FM-радио аудио (WAV/RAW) |
| `data/noaa/` | NOAA APT захваты |
| `data/rtl433/` | RTL_433 отчёты |
| `data/sstv/` | SSTV захваты |
| `data/spectrum/` | Данные спектра |

## Быстрый старт

### ADS-B (1090 MHz)
```bash
python adsb_capture.py
# Или:
../tools/rtl-sdr-blog/x64/rtl_adsb.exe -V
```

### FM-радио
```bash
python fm_multi_capture.py
# Или:
../tools/rtl-sdr-blog/x64/rtl_fm.exe -f 106.0M -M wbfm -s 32k -g 40 -E deemp
```

### RTL_433 (433 MHz)
```bash
python rtl433_scanner.py
# Или:
../tools/rtl_433/rtl_433-rtlsdr.exe -d 0:00000001 -f 433920000 -g 50 -T 60
```

### Конвертация RAW → WAV
```bash
python raw_to_wav.py input.raw
```

## Зависимости Python
- rtlsdr 0.2.93
- numpy 2.4.2
- scipy (для SSTV анализа)
- matplotlib (для визуализации)
- sounddevice (для воспроизведения)

## Ссылки
- [RTL-SDR Blog](https://www.rtl-sdr.com/)
- [rtl_433](https://github.com/merbanan/rtl_433)
- [dump1090](https://github.com/gvanem/Dump1090)
