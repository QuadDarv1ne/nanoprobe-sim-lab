# Руководство по устранению неполадок RTL-SDR V4

**Версия:** 1.0
**Последнее обновление:** 2026-04-18
**Устройство:** RTL-SDR Blog V4

---

## 📋 Содержание

1. [Первые шаги](#первые-шаги)
2. [Проблемы с установкой драйверов](#проблемы-с-установкой-драйверов)
3. [Blacklist DVB-T](#blacklist-dvb-t)
4. [udev правила (Linux)](#udev-правила-linux)
5. [PPM drift и калибровка](#ppm-drift-и-калибровка)
6. [Перегрев устройства](#перегрев-устройства)
7. [Отсутствие сигнала](#отсутствие-сигнала)
8. [Ошибки при захвате](#ошибки-при-захвате)
9. [Производительность и оптимизация](#производительность-и-оптимизация)
10. [Частые вопросы (FAQ)](#частые-вопросы-faq)

---

## Первые шаги

### Проверка подключения

#### Windows
```cmd
# Проверка в диспетчере устройств
devmgmt.msc

# Должно отображаться:
# - RTL2838UHIDIR SDR...
# - Или: Realtek RTL2832U
```

#### Linux
```bash
# Проверка подключения USB
lsusb

# Ожидаемый вывод:
# Bus 001 Device 005: ID 0bda:2838 Realtek RTL2838 DVB-T
```

#### Проверка в проекте
```bash
# Проверка доступности устройства
python -c "from utils.sdr.rtl_sdr_manager import RTLSDRManager; m = RTLSDRManager(); print(m.get_device_info())"

# Или через CLI
python rtl_sdr_tools/fm_radio_unified.py scan --gain 35
```

---

## Проблемы с установкой драйверов

### Windows: Устройство не определяется

#### Решение 1: Установка драйверов Zadig
1. Скачайте [Zadig](https://zadig.akeo.ie/)
2. Откройте Zadig от имени администратора
3. В меню `Options` → `List All Devices`
4. Выберите `RTL2838UHIDIR` или `Realtek RTL2832U`
5. Выберите драйвер `libusb-win32` или `libusbK`
6. Нажмите `Replace Driver`

#### Решение 2: Переустановка драйверов
```cmd
# Удаление устройства в диспетчере устройств
# Правой кнопкой → Удалить устройство → "Удалить драйверы"

# Переустановка:
# Скачайте Zadig и выполните шаги выше
```

### Linux: Нет прав доступа к устройству

#### Ошибка
```
libusb_open() failed with LIBUSB_ERROR_ACCESS
```

#### Решение: Добавление в группу plugdev
```bash
sudo usermod -aG plugdev $USER
# Перезайдите в систему или выполните:
newgrp plugdev
```

---

## Blacklist DVB-T

### Проблема
Система автоматически загружает драйвер DVB-T, который блокирует доступ к устройству для SDR-приложений.

### Симптомы
- `rtlsdr_open()` возвращает ошибку
- Устройство отображается как `dvb0` вместо `rtl0`
- Ошибки `Device busy` или `Permission denied`

### Решение: Blacklist драйверов DVB-T

#### Windows
Не требуется — Zadig заменяет драйвер.

#### Linux
```bash
# 1. Создайте файл blacklist
sudo nano /etc/modprobe.d/blacklist-rtl-sdr.conf

# 2. Добавьте следующие строки:
blacklist dvb_usb_rtl28xxu
blacklist dvb_usb_rtl2832
blacklist rtl2830
blacklist rtl2832
blacklist dvb_core

# 3. Обновите initramfs
sudo update-initramfs -u

# 4. Перезагрузитесь
sudo reboot

# 5. Проверьте, что модули не загружены
lsmod | grep rtl
# Должно быть пусто
```

#### Проверка
```bash
# Должно отображать устройство
lsusb | grep RTL

# Проверка доступности
rtl_test -t
```

---

## udev правила (Linux)

### Проблема
`Permission denied` при попытке открыть устройство.

### Решение

#### 1. Создайте правило udev
```bash
sudo nano /etc/udev/rules.d/81-rtl-sdr.rules
```

#### 2. Добавьте содержимое:
```bash
# RTL-SDR Blog V4
# ID производителя и устройства из lsusb
# ID 0bda:2838

# Для всех RTL-SDR устройств
SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", MODE="0666", GROUP="plugdev"

# Или для конкретного устройства
# SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2838", MODE="0666", GROUP="plugdev"
```

#### 3. Примените правила
```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

#### 4. Проверьте права
```bash
ls -l /dev/bus/usb/*/
# Должно быть: crw-rw-rw- 1 root plugdev ...
```

---

## PPM drift и калибровка

### Что такое PPM drift?

PPM (Parts Per Million) — ошибка частоты кварцевого генератора. Типичные значения:
- Дешёвые устройства: ±50-200 PPM
- RTL-SDR V4 с TCXO: ±0.5 PPM

### Симптомы
- Сигнал смещён по частоте
- SSTV изображения искажены
- Невозможно точно настроить частоту

### Калибровка PPM

#### Автоматическая калибровка (рекомендуется)
```bash
# Автоматическая калибровка через проект
python rtl_sdr_tools/rtl_sdr_auto_calibration.py --freq 106.0 --duration 10

# Или через API
curl -X POST "http://localhost:8000/api/v1/sstv/calibration/automated" \
  -H "Content-Type: application/json" \
  -d '{"freq_mhz": 106.0, "duration_sec": 10}'
```

#### Ручная калибровка
```bash
# 1. Найдите точную частоту известного сигнала (например, FM-станция)
# 2. Измерьте смещение
rtl_fm -f 106000000 -M wbfm -s 200000 -r 48000 - | aplay -r 48000

# 3. Если сигнал смещён на X Гц, вычислите PPM:
# PPM = (смещение_Гц / частота_Гц) * 1000000

# 4. Примените коррекцию:
rtl_fm -f 106000000 -p 25 -M wbfm ...
# -p 25 означает +25 PPM
```

#### Сохранение калибровки
```bash
# Сохраните в конфигурацию проекта
cat > ~/.config/nanoprobe-sim-lab/ppm_config.json << EOF
{
  "device_index": 0,
  "ppm_correction": 25,
  "calibrated_at": "2026-04-18T12:00:00Z"
}
EOF
```

---

## Перегрев устройства

### Симптомы
- Устройство отключается во время работы
- Ошибки `USB transfer error`
- Снижение производительности
- Горячий корпус (более 60°C)

### Решение

#### 1. Проверка температуры
```bash
# Через проект
python utils/sdr/hardware_health.py --check-temperature

# Или напрямую
rtl_test -t
```

#### 2. Улучшение охлаждения
- Добавьте небольшой вентилятор
- Используйте радиатор на корпус
- Не закрывайте вентиляционные отверстия

#### 3. Снижение нагрузки
```bash
# Уменьшите sample rate
rtl_fm -s 200000  # вместо 2400000

# Уменьшите gain
rtl_fm -g 30  # вместо 49.6
```

#### 4. Мониторинг в реальном времени
```bash
# Постоянный мониторинг
watch -n 1 'rtl_test -t'
```

---

## Отсутствие сигнала

### Диагностика

#### 1. Проверка подключения
```bash
# Windows
devmgmt.msc  # Диспетчер устройств

# Linux
lsusb
dmesg | grep RTL
```

#### 2. Проверка питания USB
```bash
# Linux: проверьте напряжение
cat /sys/bus/usb/devices/usb*/power/active_duration

# Решение: используйте активный USB-хаб
```

#### 3. Проверка антенны
- Убедитесь, что антенна подключена
- Проверьте целостность кабеля
- Используйте антенну, подходящую для диапазона

#### 4. Проверка частоты
```bash
# Убедитесь, что частота в диапазоне устройства
# RTL-SDR V4: 0.5-1864 MHz (с ограничениями по регионам)

# Пример проверки
python rtl_sdr_tools/fm_radio_unified.py scan --freq 88 --freq 108
```

### Частые причины

| Проблема | Решение |
|----------|---------|
| Антенна не подключена | Подключите антенну |
| Частота вне диапазона | Используйте частоту 88-108 MHz для FM |
| Слишком низкий gain | Увеличьте gain до 30-49.6 |
| Помехи от USB 3.0 | Используйте USB 2.0 или экранированный кабель |
| Blacklist не применён | Перезагрузитесь после blacklist |

---

## Ошибки при захвате

### Ошибка: `OverflowError` или `Buffer overflow`

```bash
# Уменьшите sample rate или увеличьте буфер
rtl_fm -s 200000 -b 16 ...

# Или в Python
from utils.sdr.rtl_sdr_manager import RTLSDRManager
manager = RTLSDRManager(buffer_size=2*1024*1024)  # 2MB буфер
```

### Ошибка: `Timeout` при чтении

```bash
# Увеличьте timeout
rtl_fm -t 5 ...

# Или в Python
manager.read_samples(timeout=5.0)
```

### Ошибка: `Device or resource busy`

```bash
# Проверьте, не используется ли устройство другим процессом
# Windows: Process Explorer
# Linux: lsof /dev/bus/usb/*/
sudo lsof | grep rtl

# Завершите процесс или отключите устройство
```

---

## Производительность и оптимизация

### Рекомендуемые настройки

#### Для FM радио
```bash
rtl_fm -f 106000000 -M wbfm -s 200000 -r 48000 -g 40
```

#### Для SSTV (МКС)
```bash
# Частота МКС: 145.800 MHz
rtl_fm -f 145800000 -M fm -s 24000 -r 24000 -g 49.6 -p 25
```

#### Для NOAA спутников
```bash
# NOAA 15: 137.62 MHz, NOAA 18: 137.9125 MHz, NOAA 19: 137.1 MHz
rtl_fm -f 137620000 -M wbfm -s 100000 -r 48000 -g 49.6 -p 25
```

### Оптимизация производительности

#### 1. Используйте буферы ring buffer
```python
from utils.sdr.ring_buffer import SharedRingBuffer
buffer = SharedRingBuffer(name="rtl_sdr", capacity=1024)
```

#### 2. Многопоточная обработка
```python
from utils.sdr.sdr_resource_manager import SDRResourceManager
manager = SDRResourceManager()
manager.allocate_resource(priority="high")
```

#### 3. Аппаратное ускорение
```bash
# Включите аппаратное декодирование (если поддерживается)
rtl_fm --agc-mode 1
```

---

## Частые вопросы (FAQ)

### Q: Почему устройство работает медленно?
**A:** Проверьте:
1. USB 2.0 вместо 3.0 (иногда 2.0 стабильнее)
2. Достаточно ли питания (используйте активный хаб)
3. Blacklist DVB-T драйверов применён

### Q: Как проверить, что устройство работает корректно?
**A:**
```bash
# Простой тест
rtl_test -t

# Тест с захватом
rtl_fm -f 106000000 -M wbfm -s 200000 -r 48000 - | aplay -r 48000
```

### Q: Можно ли использовать несколько RTL-SDR одновременно?
**A:** Да, укажите индекс устройства:
```bash
rtl_fm -d 0 -f 106000000 ...  # Устройство 0
rtl_fm -d 1 -f 145800000 ...  # Устройство 1
```

### Q: Как обновить прошивку?
**A:** RTL-SDR V4 не требует обновления прошивки. Драйверы обновляются через Zadig (Windows) или систему пакетов (Linux).

### Q: Почему SSTV изображения искажены?
**A:** Проверьте:
1. PPM калибровку
2. Стабильность частоты
3. Уровень сигнала (gain)
4. Правильность SSTV режима (Scottie, Martin, etc.)

---

## Полезные команды

### Диагностика
```bash
# Информация об устройстве
rtl_eeprom -s

# Тест устройства
rtl_test -t

# Просмотр спектра (требует gnuradio)
gr-osmocom viterbi

# Мониторинг температуры
watch -n 1 'rtl_test -t'
```

### Захват
```bash
# FM радио
rtl_fm -f 106000000 -M wbfm -s 200000 -r 48000 - | aplay -r 48000

# Запись в файл
rtl_fm -f 106000000 -M wbfm -s 200000 -r 48000 -w capture.wav

# SSTV
rtl_fm -f 145800000 -M fm -s 24000 -r 24000 -w sstv.wav
```

---

## Ссылки

- [Официальный сайт RTL-SDR Blog](https://www.rtl-sdr.com/)
- [Документация osmocom](https://osmocom.org/projects/rtl-sdr/wiki)
- [Celestrak TLE](https://celestrak.org/)
- [Satnobs](https://satnobs.io/)
- [Аматорская радиосвязь](https://www.ariss.org/)

---

**Владелец проекта:** Дуплей Максим Игоревич
**Лицензия:** Проприетарная (ограниченные права)
