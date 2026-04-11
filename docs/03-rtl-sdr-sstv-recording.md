# RTL-SDR SSTV Recording Guide

**Last Updated:** 2026-03-15
**Version:** 1.0.0

---

## 📋 Обзор

Этот документ описывает функционал записи RTL-SDR для приёма SSTV сигналов с МКС и других спутников.

---

## 🚀 Быстрый старт

### 1. Проверка оборудования

```bash
# Проверка наличия rtl_fm (часть rtl-sdr)
rtl_fm -h

# Если не найден - установите rtl-sdr:
# Windows: https://rtlsdr.org/software
# Linux: sudo apt-get install rtl-sdr
# macOS: brew install rtl-sdr
```

### 2. Запуск записи

```bash
# Запись на частоте МКС (145.800 MHz)
curl -X POST "http://localhost:8000/api/v1/sstv/record/start" \
  -H "Content-Type: application/json" \
  -d '{
    "frequency": 145.800,
    "duration": 600
  }'
```

### 3. Остановка записи

```bash
curl -X POST "http://localhost:8000/api/v1/sstv/record/stop"
```

### 4. Просмотр записей

```bash
curl "http://localhost:8000/api/v1/sstv/recordings"
```

---

## 📡 API Endpoints

### POST `/api/v1/sstv/record/start`

Запуск записи с RTL-SDR для приёма SSTV.

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `frequency` | float | 145.800 | Частота в MHz (145.800 для МКС) |
| `sample_rate` | int | 2048000 | Частота дискретизации (Hz) |
| `gain` | int | 496 | Усиление RTL-SDR (0-496) |
| `duration` | int | 600 | Длительность записи в секундах |

**Пример запроса:**

```json
{
  "frequency": 145.800,
  "sample_rate": 2048000,
  "gain": 496,
  "duration": 600
}
```

**Пример ответа:**

```json
{
  "status": "recording_started",
  "frequency_mhz": 145.800,
  "sample_rate": 2048000,
  "gain": 496,
  "output_file": "output/sstv/recordings/sstv_145.8MHz_20260315_143022.wav",
  "started_at": "2026-03-15T14:30:22.123456",
  "pid": 12345,
  "message": "Запись началась. Остановка через 600 секунд."
}
```

**Статусы:**

- `recording_started` - Запись началась (RTL-SDR найден)
- `recording_simulated` - Запись симулируется (RTL-SDR не найден)
- `already_recording` - Запись уже идёт

---

### POST `/api/v1/sstv/record/stop`

Остановка текущей записи SSTV.

**Пример ответа:**

```json
{
  "status": "recording_stopped",
  "duration_seconds": 120.45,
  "output_file": "output/sstv/recordings/sstv_145.8MHz_20260315_143022.wav",
  "message": "Запись остановлена"
}
```

**Статусы:**

- `recording_stopped` - Запись остановлена
- `recording_stopped_simulated` - Симуляция остановлена
- `not_recording` - Запись не шла

---

### GET `/api/v1/sstv/record/status`

Получение статуса текущей записи.

**Пример ответа (запись идёт):**

```json
{
  "status": "recording",
  "recording": true,
  "started_at": "2026-03-15T14:30:22.123456",
  "duration_seconds": 45.67,
  "metadata": {
    "frequency": 145.800,
    "sample_rate": 2048000,
    "gain": 496,
    "output_file": "output/sstv/recordings/sstv_145.8MHz_20260315_143022.wav",
    "simulated": false
  }
}
```

**Пример ответа (ожидание):**

```json
{
  "status": "idle",
  "recording": false,
  "message": "Запись не идёт"
}
```

---

### GET `/api/v1/sstv/recordings`

Получение списка всех записей SSTV.

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `limit` | int | 20 | Максимальное количество записей |

**Пример ответа:**

```json
{
  "recordings": [
    {
      "filename": "sstv_145.8MHz_20260315_143022.wav",
      "path": "output/sstv/recordings/sstv_145.8MHz_20260315_143022.wav",
      "size_bytes": 10485760,
      "created_at": "2026-03-15T14:30:22",
      "frequency": "145.8"
    },
    {
      "filename": "sstv_146.0MHz_20260315_120000.wav",
      "path": "output/sstv/recordings/sstv_146.0MHz_20260315_120000.wav",
      "size_bytes": 8388608,
      "created_at": "2026-03-15T12:00:00",
      "frequency": "146.0"
    }
  ]
}
```

---

## 🛰️ Частоты для приёма

### МКС (ISS)

| Тип | Частота (MHz) | Модуляция | Описание |
|-----|---------------|-----------|----------|
| SSTV | 145.800 | FM | Основное SSTV вещание |
| APRS | 145.825 | FM | Пакетная радиосвязь |
| Voice | 145.800 | FM | Голосовая связь (редко) |

### Другие спутники

| Спутник | Частота (MHz) | Модуляция | Описание |
|---------|---------------|-----------|----------|
| NO-84 | 435.350 | FM | SSTV эксперименты |
| EO-88 | 435.180 | FM | SSTV эксперименты |

---

## 🔧 Настройка RTL-SDR

### Параметры

**Gain (усиление):**
- 0-200: Низкое усиление (сильный сигнал)
- 200-400: Среднее усиление
- 400-496: Высокое усиление (слабый сигнал)

**Sample Rate:**
- 1024000 (1 MS/s): Узкая полоса
- 2048000 (2 MS/s): Стандартная полоса
- 2400000 (2.4 MS/s): Широкая полоса

### Командная строка (rtl_fm)

```bash
# Прямой вызов rtl_fm
rtl_fm -f 145800000 -s 2048000 -g 496 -F 9 -o 4 -M fm output.wav
```

**Параметры:**
- `-f`: Частота в Hz
- `-s`: Sample rate
- `-g`: Gain
- `-F 9`: Фильтр
- `-o 4**: Передискретизация
- `-M fm`: Модуляция FM

---

## 🧰 Тестирование

### Запуск тестов

```bash
python tests/test_rtl_sdr_recording.py
```

### Тесты включают:

- ✅ Инициализация переменных записи
- ✅ Статус записи (idle/recording)
- ✅ Структура ответа API
- ✅ Список записей
- ✅ Валидация параметров
- ✅ Метаданные записи
- ✅ Остановка записи
- ✅ Диапазон частот

---

## 📁 Структура файлов

```
output/
└── sstv/
    └── recordings/
        ├── sstv_145.8MHz_20260315_143022.wav
        ├── sstv_145.8MHz_20260315_150000.wav
        └── sstv_146.0MHz_20260315_120000.wav
```

### Формат имён файлов

```
sstv_{frequency}MHz_{timestamp}.wav
```

**Пример:**
- `sstv_145.8MHz_20260315_143022.wav`
- Частота: 145.8 MHz
- Дата: 2026-03-15
- Время: 14:30:22

---

## 🛠️ Режим симуляции

Если RTL-SDR не найден, система автоматически переходит в режим симуляции:

```json
{
  "status": "recording_simulated",
  "frequency_mhz": 145.800,
  "output_file": "output/sstv/recordings/sstv_145.8MHz_20260315_143022.wav",
  "message": "RTL-SDR не найден. Запись симулируется для тестирования."
}
```

**Преимущества:**
- Тестирование без оборудования
- Отладка frontend
- Демонстрация функционала

---

## 🔗 Связанные endpoints

### SSTV Health Check

```bash
curl "http://localhost:8000/api/v1/sstv/health"
```

**Ответ:**

```json
{
  "status": "healthy",
  "components": {
    "sstv_decoder": "available",
    "satellite_tracker": "available",
    "redis_cache": "available",
    "rtl_sdr": "ready",
    "timestamp": "2026-03-15T14:30:22"
  }
}
```

### ISS Tracking

```bash
# Следующий пролёт МКС
curl "http://localhost:8000/api/v1/sstv/iss/next-pass"

# Текущая позиция МКС
curl "http://localhost:8000/api/v1/sstv/iss/position"
```

---

## 📝 Заметки

### Автоматическая остановка

Запись автоматически останавливается через указанное время `duration`.

### Перезапуск сервера

При перезапуске сервера активная запись будет потеряна. Рекомендуется использовать `duration` параметр.

### Обработка файлов

После записи можно:
1. Декодировать SSTV через `pysstv`
2. Анализировать изображения
3. Сохранять в базу данных

---

## 🐛 Troubleshooting

### RTL-SDR не найден

**Проблема:** `rtl_fm: command not found`

**Решение:**
1. Установите rtl-sdr драйверы
2. Добавьте путь к PATH
3. Перезапустите сервер

### Нет записей в списке

**Проблема:** Пустой список записей

**Решение:**
1. Проверьте директорию `output/sstv/recordings/`
2. Убедитесь, что запись была запущена
3. Проверьте права доступа к файлам

### Ошибка записи

**Проблема:** `Failed to start recording`

**Решение:**
1. Проверьте, не запущена ли уже запись
2. Убедитесь, что частота корректна
3. Проверьте логи сервера

---

## 📚 Дополнительные ресурсы

- [RTL-SDR Official Site](https://www.rtl-sdr.com/)
- [pysstv Documentation](https://github.com/Florob/pysstv)
- [ISS SSTV Schedule](https://www.ariss.org/)
- [N2YO Satellite Tracking](https://www.n2yo.com/)

---

**Nanoprobe Sim Lab - SSTV Ground Station** 🛰️
