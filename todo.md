# Nanoprobe Sim Lab - TODO

## Реализовано ✅

### 1. RTL-SDR V4 Поддержка
- [x] Автоопределение устройства (R828D)
- [x] Sample rate 2.4 MSPS
- [x] Bias-T для питания антенны
- [x] AGC (автоматическая регулировка усиления)
- [x] Direct sampling для УКВ
- [x] Коррекция частоты (TCXO)
- [x] Пакетное чтение сэмплов
- [x] Сохранение WAV с метаданными
- [x] CLI опции (--bias-tee, --agc, --gain, --sample-rate)
- [x] Команда --check для проверки устройства

### 2. SSTV Декодирование
- [x] decode_from_samples() - декодирование из numpy массива
- [x] Real-time декодирование (decode_realtime_init/push/stop)
- [x] Автодетект SSTV сигнала
- [x] Сохранение изображений с timestamp
- [x] Пакетное декодирование (process_sstv_batch)

### 3. Спутниковое Отслеживание
- [x] SatelliteTracker класс с TLE данными
- [x] Предсказание пролётов (ISS, NOAA 15/18/19, Meteor-M2)
- [x] Частоты SSTV спутников
- [x] Команды --satellites и --schedule
- [x] Пакетный расчёт пролётов

### 4. Waterfall Дисплей
- [x] FFT анализ в реальном времени (512 bins)
- [x] Цветовая спектрограмма
- [x] WaterfallDisplay класс
- [x] WaterfallRecorder для записи
- [x] Сохранение PNG и GIF
- [x] Команда --waterfall

### 5. CI/CD Pipeline
- [x] tests.yml - тесты Python 3.9-3.12
- [x] lint.yml - flake8, black, mypy
- [x] build.yml - сборка при тегах
- [x] CODEOWNERS
- [x] PR и Issue шаблоны

### 6. Пакетная Обработка
- [x] callback поддержка в BatchJob
- [x] process_sstv_batch()
- [x] process_satellite_passes()
- [x] Real-time прогресс

### 7. Веб-интерфейс
- [x] API /api/components
- [x] API /api/processes
- [x] API /api/stats
- [x] API /api/health
- [x] API /api/logs/component/<name>
- [x] API /api/actions/start_component
- [x] API /api/actions/stop_component
- [x] API /api/actions/restart_component
- [x] API /api/actions/quick
- [x] WebSocket уведомления (component_status, stats_update)
- [x] Real-time обновление статистики
- [x] Кнопки управления компонентами
- [x] Просмотр логов компонентов

---

## В Ожидании ⏳

### Когда придёт RTL-SDR V4 (через 2 недели):
- [ ] Тестирование real-time SSTV с МКС
- [ ] Проверка waterfall дисплея
- [ ] Тестирование приёма NOAA/Meteor
- [ ] Калибровка частоты (TCXO)
- [ ] Запись и декодирование пролётов

---

## Будущие Улучшения 💡

### Приоритет 1
- [ ] Веб-интерфейс waterfall (WebSocket streaming)
- [ ] Автозапись при пролёте спутника
- [ ] QSL карточки для подтверждённых приёмов
- [ ] Интеграция с satnobs.io

### Приоритет 2
- [ ] TLE обновления автоматически (celestrak API)
- [ ] Улучшенный SSTV детектор (энергия сигнала)
- [ ] Экспорт в форматы радиолюбителей (ADIF)
- [ ] Мобильная версия веб-интерфейса

### Приоритет 3
- [ ] Docker контейнеризация
- [ ] Telegram бот для уведомлений
- [ ] База данных всех приёмов
- [ ] Статистика и графики

---

## Известные Проблемы 🐛

- [ ] test_metadata_structure требует mock SDR устройства
- [ ] Waterfall требует много памяти при длительной записи
- [ ] Real-time декодирование может пропускать сигналы при высокой нагрузке

---

## Команды для Использования

```bash
# Проверка RTL-SDR
python components/py-sstv-groundstation/src/main.py --check

# Real-time SSTV
python components/py-sstv-groundstation/src/main.py --realtime-sstv -f iss --duration 120

# Waterfall
python components/py-sstv-groundstation/src/main.py --waterfall -f 145.800 --save-waterfall

# Спутники
python components/py-sstv-groundstation/src/main.py --satellites
python components/py-sstv-groundstation/src/main.py --schedule --lat 55.75 --lon 37.61

# Запуск веб-интерфейса
python src/web/web_dashboard.py
```

---

**Последнее обновление:** 2026-03-12
**Статус:** 7/8 приоритетных улучшений реализовано
