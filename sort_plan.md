# План сортировки проекта

## 1. Корень проекта — убрать мусор

### Переместить в data/:
- airband_samples.dat → data/airband/
- sstv_test.raw → data/sstv/
- aviation_scan.csv → data/airband/
- api_startup.log → logs/
- backend.log → logs/

### Переместить в rtl_sdr_tools/:
- adsb_receiver.py (дубликат)
- am_airband.py (дубликат)
- fm_radio.py (дубликат)

### Переместить в scripts/:
- check_zadig_drivers.bat
- setup_rtlsdr.bat
- listen_airband.bat
- listen_fm.bat
- scan_airband.bat
- sort_project.py

### Удалить (дубликаты):
- rtl-sdr-bin/ (есть tools/rtl-sdr-blog/)
- zadig.exe, zadig-2.9.exe (есть в bin/)

### Оставить в корне (важные):
- main.py, run_api.py — точки входа
- admin_cli.py — CLI
- requirements*.txt — зависимости
- docker-compose, vercel, railway — деплой
- README, TODO, QUICKSTART — документация
- .env.example, .gitignore — конфиг

## 2. data/ — рассортировать

### data/airband/ (создать):
- airband_samples.dat
- aviation_scan.csv

### data/fm_radio/:
- Оставить wav/raw
- Удалить fm_stations (пустая папка?)
- Удалить дубликаты

### data/rtl433/:
- Оставить отчёты
- Удалить пустые jsonl

### data/sstv/:
- Оставить wav
- Удалить analysis.png если старый

## 3. tools/ — объединить

### tools/rtl-sdr-blog/ — оставить (актуальные v1.3.6)
### tools/dump1090/ — оставить (не работает с V4, но полезен)
### tools/rtl_433/ — оставить (актуальный v25.12)
### rtl-sdr-bin/ — УДАЛИТЬ (дубликат)

## 4. logs/

### Переместить логи из корня:
- api_startup.log → logs/
- backend.log → logs/
