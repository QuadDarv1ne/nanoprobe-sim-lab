# 🚀 RTL-SDR V4 - Запуск на полную мощность

## ✅ Текущий статус (2026-04-09)

### Установленное ПО:
- ✅ **pyrtlsdr 0.3.0** - Python библиотека для RTL-SDR
- ✅ **pysstv 0.5.7** - SSTV декодер
- ✅ **numpy 2.4.2** - Обработка сигналов
- ✅ **scipy 1.17.1** - Научные вычисления
- ✅ **setuptools 70.3.0** - Для pkg_resources

### Оборудование:
- ✅ **RTL-SDR V4** подключён (тюнер R828D)
- ⚠️ **Драйверы Zadig** - требуют проверки

---

## 🔧 Быстрый старт

### 1. Проверка драйверов

```powershell
# Проверка видимости устройства
python -c "from rtlsdr import RtlSdr; print(RtlSdr.get_device_count())"

# Если ошибка LIBUSB_ERROR_IO:
# 1. Скачайте Zadig: https://zadig.akeo.ie/
# 2. Запустите от администратора
# 3. Options → List All Devices
# 4. Выберите RTL2838UHIDIR
# 5. Установите драйвер WinUSB
```

### 2. Базовый тест

```powershell
python test_rtlsdr_sstv.py
```

**Ожидаемый результат:**
```
[1/5] ✅ pyrtlsdr импортирован
[2/5] ✅ pysstv импортирован
[3/5] ✅ RTL-SDR инициализирован
[4/5] ✅ Прочитано 1024 сэмплов
[5/5] ✅ Спектр получен: 1024 точек
```

### 3. Запуск SSTV станции (МКС 145.800 МГц)

```powershell
# Через API проекта
python main.py --realtime-sstv -f iss --duration 120 --gain 49.6

# Или напрямую
python sstv_ground_station.py --frequency 145.800 --duration 120 --gain 49.6
```

---

## 📡 Режимы работы

### 🛰️ Приём SSTV с МКС

**Частота:** 145.800 MHz  
**Режим:** PD120, Martin M1, Scottie S1  
**Усиление:** 49.6 dB (максимум для V4)  

```powershell
python -c "
from api.sstv.rtl_sstv_receiver import RTLSDRReceiver

receiver = RTLSDRReceiver(
    frequency=145.800,
    sample_rate=2.4e6,
    gain=49.6
)

if receiver.initialize():
    print('✅ RTL-SDR готов к приёму МКС!')
    info = receiver.get_device_info()
    print(f'Частота: {info[\"frequency_mhz\"]} МГц')
    print(f'Усиление: {info[\"gain_db\"]} дБ')
    receiver.close()
"
```

### 📊 Спектральный анализ

```powershell
python -c "
import numpy as np
from api.sstv.rtl_sstv_receiver import RTLSDRReceiver

receiver = RTLSDRReceiver(frequency=145.800, gain=49.6)
receiver.initialize()

freqs, power = receiver.get_spectrum(4096)
if freqs is not None:
    print(f'Спектр: {len(freqs)} точек')
    print(f'Диапазон: {freqs[0]/1e6:.3f} - {freqs[-1]/1e6:.3f} МГц')
    print(f'Мощность: {np.min(power):.1f} - {np.max(power):.1f} дБ')

receiver.close()
"
```

### 🎙️ Запись сигнала

```powershell
python -c "
from api.sstv.rtl_sstv_receiver import RTLSDRReceiver
import numpy as np

receiver = RTLSDRReceiver(frequency=145.800, gain=49.6)
receiver.initialize()

# Запись 10 секунд
samples = receiver.record_audio(duration=10.0, sample_rate=48000)
if samples is not None:
    print(f'✅ Записано {len(samples)} сэмплов')
    print(f'Длительность: {len(samples)/48000:.1f} сек')
    receiver._save_wav(samples, 'recording.wav', 48000)

receiver.close()
"
```

---

## 🔍 Расширенные возможности

### Сканирование диапазона

```powershell
python -c "
from components.py_sstv_groundstation.src.sdr_interface import SDRInterface
import time

sdr = SDRInterface(center_freq=145.800, gain=49.6)
sdr.initialize()

# Сканирование 145-146 МГц (диапазон МКС)
print('Сканирование 145-146 МГц...')
for freq in range(145000, 146000, 100):  # шаг 100 кГц
    sdr.set_frequency(freq / 1000)
    time.sleep(0.1)
    samples = sdr.read_samples(1024)
    if samples is not None:
        strength = np.mean(np.abs(samples)) * 100
        print(f'{freq/1000:.3f} МГц: {strength:.1f}%')

sdr.close()
"
```

### Автоматическое декодирование SSTV

```powershell
python -c "
from api.sstv.rtl_sstv_receiver import RTLSDRReceiver, SSTVDecoder
import numpy as np

receiver = RTLSDRReceiver(frequency=145.800, gain=49.6)
decoder = SSTVDecoder(mode='auto')

receiver.initialize()

# Запись 30 секунд
print('Запись 30 секунд...')
samples = receiver.record_audio(duration=30.0, sample_rate=48000)

if samples is not None:
    print('Декодирование SSTV...')
    image = decoder.decode_audio(samples, sample_rate=48000)
    if image:
        image.save('sstv_image.png')
        print('✅ SSTV изображение сохранено: sstv_image.png')
    else:
        print('❌ SSTV сигнал не обнаружен')

receiver.close()
"
```

---

## 📈 Мониторинг через API

После запуска проекта (`python main.py`):

### Swagger документация
```
http://localhost:8000/docs
```

### SSTV эндпоинты:
- `GET /api/sstv/status` - Статус приёмника
- `POST /api/sstv/start` - Начать запись
- `POST /api/sstv/stop` - Остановить запись
- `GET /api/sstv/spectrum` - Получить спектр
- `GET /api/sstv/image` - Последнее декодированное изображение

### Пример запроса:

```powershell
# Запуск SSTV приёмника
curl -X POST http://localhost:8000/api/sstv/start `
  -H "Content-Type: application/json" `
  -d '{\"duration\": 60, \"frequency\": 145.800, \"gain\": 49.6}'

# Проверка статуса
curl http://localhost:8000/api/sstv/status
```

---

## 🎯 Оптимизация для RTL-SDR V4

### Максимальная производительность:

| Параметр | Значение | Описание |
|----------|----------|----------|
| **Sample Rate** | 2.4 MSPS | Оптимально для V4 |
| **Gain** | 49.6 dB | Максимум для V4 |
| **Bias-Tee** | Вкл | Для активных антенн |
| **AGC** | Выкл | Ручной контроль |
| **Direct Sampling** | Авто | Для УКВ < 24 МГц |

### Команды для максимальной чувствительности:

```powershell
# Максимальное усиление
python -c "
from rtlsdr import RtlSdr
sdr = RtlSdr()
sdr.sample_rate = 2.4e6
sdr.center_freq = 145.800e6
sdr.gain = 49.6  # Максимум для V4
print('✅ Максимальная чувствительность установлена')
sdr.close()
"

# Bias-Tee для активной антенны
python -c "
from rtlsdr import RtlSdr
sdr = RtlSdr()
if hasattr(sdr, 'bias_tee'):
    sdr.bias_tee = True
    print('✅ Bias-Tee включён')
sdr.close()
"
```

---

## 📋 Чеклист готовности

- [x] pyrtlsdr установлен
- [x] pysstv установлен
- [x] numpy/scipy установлены
- [ ] Драйверы Zadig установлены (проверить)
- [ ] RTL-SDR открывается без ошибок
- [ ] Тест сэмплов проходит
- [ ] Спектральный анализ работает
- [ ] SSTV декодирование работает
- [ ] API эндпоинты доступны

---

## 🚨 Устранение проблем

### Ошибка: LIBUSB_ERROR_IO

**Решение:**
1. Запустите Zadig от администратора
2. Options → List All Devices ✓
3. Выберите "RTL2838UHIDIR" или "Bulk-In, Interface"
4. Драйвер: **WinUSB (v6.x.x.x)**
5. Нажмите "Replace Driver"
6. Переподключите RTL-SDR

### Ошибка: No module named 'pysstv.mode'

**Решение:** ✅ Исправлено! Теперь используется `pysstv.grayscale`

### Ошибка: No module named 'pkg_resources'

**Решение:** ✅ Исправлено! Установлен setuptools 70.3.0

---

## 📚 Дополнительные ресурсы

- **RTL-SDR Blog:** https://www.rtl-sdr.com/
- **ISS SSTV расписание:** https://www.ariss.org/
- **Heavens-Above (трекинг МКС):** https://www.heavens-above.com/
- **PySDR руководство:** https://pysdr.org/

---

## 🎉 Готово!

RTL-SDR V4 готов к работе на полную мощность! 🚀

**Следующие шаги:**
1. Проверить драйверы Zadig
2. Запустить `python test_rtlsdr_sstv.py`
3. Начать приём SSTV с МКС!

---

**Обновлено:** 2026-04-09  
**Статус:** ✅ Программное обеспечение готово
