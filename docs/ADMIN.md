# Документация для администратора Nanoprobe Sim Lab

**Версия:** 1.0.0  
**Дата обновления:** 2026-03-11  
**Для кого:** Системные администраторы, DevOps инженеры

---

## 📖 Содержание

1. [Быстрый старт](#быстрый-старт)
2. [Архитектура системы](#архитектура-системы)
3. [Установка и настройка](#установка-и-настройка)
4. [Управление базой данных](#управление-базой-данных)
5. [Мониторинг и логи](#мониторинг-и-логи)
6. [Резервное копирование](#резервное-копирование)
7. [Безопасность](#безопасность)
8. [Производительность](#производительность)
9. [Решение проблем](#решение-проблем)
10. [Чек-листы](#чек-листы)

---

## 🚀 Быстрый старт

### Команды для ежедневного использования

```bash
# Проверка статуса API
curl http://localhost:8000/health

# Запуск API
python run_api.py --reload

# Запуск веб-интерфейса
python start.py web

# Просмотр логов
tail -f logs/api.log
tail -f logs/flask.log

# Проверка БД
python -c "from utils.database import DatabaseManager; db = DatabaseManager(); print(db.get_statistics())"

# Очистка кэша
python start.py manager clean-cache
```

### Порты сервисов

| Сервис | Порт | Протокол |
|--------|------|----------|
| FastAPI API | 8000 | HTTP/WS |
| Flask Web | 5000 | HTTP |
| Redis (опционально) | 6379 | TCP |
| SQLite DB | - | File |

---

## 🏗️ Архитектура системы

### Компоненты

```
┌─────────────────────────────────────────────────────────┐
│                    Клиенты                              │
│  ┌───────────┐  ┌───────────┐  ┌──────────────┐       │
│  │  Browser  │  │  cURL     │  │  Python API  │       │
│  └─────┬─────┘  └─────┬─────┘  └──────┬───────┘       │
└────────┼──────────────┼────────────────┼──────────────┘
         │              │                │
    ┌────▼──────────────▼────┐      ┌───▼────────┐
    │   Flask Web (5000)     │      │ FastAPI    │
    │   HTML Templates       │      │ (8000)     │
    └────┬───────────────────┘      └─────┬──────┘
         │                                │
         └────────────┬───────────────────┘
                      │
                ┌─────▼──────┐
                │   SQLite   │
                │ nanoprobe  │
                │    .db     │
                └────────────┘
```

### Директории

```
nanoprobe-sim-lab/
├── api/                    # FastAPI REST API
├── src/                    # Исходный код
│   ├── cli/               # Консольные утилиты
│   └── web/               # Flask веб-интерфейс
├── utils/                  # Общие утилиты
├── data/                   # База данных и файлы
├── logs/                   # Логи
├── output/                 # Результаты сканирований
├── reports/                # PDF отчёты
├── config/                 # Конфигурация
└── tests/                  # Тесты
```

---

## ⚙️ Установка и настройка

### Системные требования

| Компонент | Минимум | Рекомендуется |
|-----------|---------|---------------|
| CPU | 2 ядра | 4+ ядра |
| RAM | 4 GB | 8+ GB |
| Disk | 10 GB | 50+ GB SSD |
| OS | Win10/Linux | Ubuntu 22.04 LTS |

### Установка на Linux (Ubuntu 22.04)

```bash
# Обновление системы
sudo apt update && sudo apt upgrade -y

# Установка Python и зависимостей
sudo apt install -y python3.11 python3.11-venv python3-pip
sudo apt install -y git cmake build-essential

# Создание пользователя
sudo useradd -r -m -s /bin/bash nanoprobe

# Клонирование репозитория
sudo -u nanoprobe -i
git clone <repository_url> /opt/nanoprobe-sim-lab
cd /opt/nanoprobe-sim-lab

# Создание виртуального окружения
python3 -m venv venv
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt
pip install -r requirements-api.txt

# Создание директорий
mkdir -p data logs output reports

# Настройка прав
chown -R nanoprobe:nanoprobe /opt/nanoprobe-sim-lab
```

### Systemd сервисы

**/etc/systemd/system/nanoprobe-api.service:**
```ini
[Unit]
Description=Nanoprobe Sim Lab FastAPI API
After=network.target

[Service]
Type=simple
User=nanoprobe
Group=nanoprobe
WorkingDirectory=/opt/nanoprobe-sim-lab
Environment="PATH=/opt/nanoprobe-sim-lab/venv/bin"
ExecStart=/opt/nanoprobe-sim-lab/venv/bin/python run_api.py --workers 4
Restart=always
RestartSec=10

# Security
NoNewPrivileges=true
PrivateTmp=true

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=nanoprobe-api

[Install]
WantedBy=multi-user.target
```

**/etc/systemd/system/nanoprobe-web.service:**
```ini
[Unit]
Description=Nanoprobe Sim Lab Flask Web
After=network.target nanoprobe-api.service

[Service]
Type=simple
User=nanoprobe
Group=nanoprobe
WorkingDirectory=/opt/nanoprobe-sim-lab
Environment="PATH=/opt/nanoprobe-sim-lab/venv/bin"
Environment="FLASK_ENV=production"
ExecStart=/opt/nanoprobe-sim-lab/venv/bin/python start.py web
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Управление сервисами:**
```bash
# Перезагрузка конфигурации
sudo systemctl daemon-reload

# Включение автозапуска
sudo systemctl enable nanoprobe-api nanoprobe-web

# Запуск
sudo systemctl start nanoprobe-api
sudo systemctl start nanoprobe-web

# Проверка статуса
sudo systemctl status nanoprobe-api
sudo systemctl status nanoprobe-web

# Просмотр логов
sudo journalctl -u nanoprobe-api -f
sudo journalctl -u nanoprobe-web -f
```

### Nginx конфигурация

**/etc/nginx/sites-available/nanoprobe:**
```nginx
server {
    listen 80;
    server_name nanoprobe.yourdomain.com;

    # Логи
    access_log /var/log/nginx/nanoprobe-access.log;
    error_log /var/log/nginx/nanoprobe-error.log;

    # Максимальный размер загрузки
    client_max_body_size 100M;

    # API (FastAPI)
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }

    # WebSocket
    location /ws/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
    }

    # Веб-интерфейс (Flask)
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

**Активация:**
```bash
sudo ln -s /etc/nginx/sites-available/nanoprobe /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## 🗄️ Управление базой данных

### SQLite оптимизация

**Файл БД:** `data/nanoprobe.db`

**Настройка производительности:**
```python
# В utils/database.py уже настроено:
PRAGMA journal_mode = WAL          # Write-Ahead Logging
PRAGMA synchronous = NORMAL        # Баланс скорость/надёжность
PRAGMA cache_size = -64000         # 64MB кэш
PRAGMA temp_store = MEMORY         # Временные данные в RAM
PRAGMA mmap_size = 268435456       # 256MB memory-mapped I/O
```

### Резервное копирование БД

**Скрипт backup_db.sh:**
```bash
#!/bin/bash
# Резервное копирование базы данных

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/backups/nanoprobe"
DB_FILE="/opt/nanoprobe-sim-lab/data/nanoprobe.db"

mkdir -p $BACKUP_DIR

# Копирование с блокировкой
sqlite3 $DB_FILE ".backup '$BACKUP_DIR/nanoprobe_$DATE.db'"

# Сжатие
gzip $BACKUP_DIR/nanoprobe_$DATE.db

# Удаление старых бэкапов (>30 дней)
find $BACKUP_DIR -name "nanoprobe_*.db.gz" -mtime +30 -delete

echo "Backup completed: nanoprobe_$DATE.db.gz"
```

**Cron задача (ежедневно в 3:00):**
```bash
# crontab -e
0 3 * * * /opt/nanoprobe-sim-lab/scripts/backup_db.sh
```

### Ваккумирование БД

```bash
# Очистка и оптимизация
sqlite3 data/nanoprobe.db "VACUUM;"

# Проверка целостности
sqlite3 data/nanoprobe.db "PRAGMA integrity_check;"

# Анализ таблиц
sqlite3 data/nanoprobe.db "ANALYZE;"
```

### Мониторинг БД

```bash
# Размер БД
du -h data/nanoprobe.db

# Количество записей
sqlite3 data/nanoprobe.db "SELECT 'scans': COUNT(*) FROM scan_results UNION ALL SELECT 'simulations': COUNT(*) FROM simulations;"

# Последние записи
sqlite3 data/nanoprobe.db "SELECT * FROM scan_results ORDER BY timestamp DESC LIMIT 5;"
```

---

## 📊 Мониторинг и логи

### Логи приложения

**Расположение:**
- `logs/api.log` - логи FastAPI
- `logs/flask.log` - логи Flask
- `logs/error.log` - общие ошибки

**Уровни логирования:**
- `DEBUG` - детальная отладка
- `INFO` - информационные сообщения
- `WARNING` - предупреждения
- `ERROR` - ошибки
- `CRITICAL` - критические ошибки

**Просмотр в реальном времени:**
```bash
# Все логи
tail -f logs/*.log

# Только ошибки
tail -f logs/error.log

# С фильтрацией
tail -f logs/api.log | grep ERROR
```

### Логирование в systemd journal

```bash
# API логи
journalctl -u nanoprobe-api -f

# Web логи
journalctl -u nanoprobe-web -f

# За последние 2 часа
journalctl -u nanoprobe-api --since "2 hours ago"

# Экспорт в файл
journalctl -u nanoprobe-api --since today > logs/api_today.log
```

### Метрики для мониторинга

**Health Check эндпоинты:**
```bash
# API health
curl http://localhost:8000/health

# Детальная статистика
curl http://localhost:8000/api/v1/statistics
```

**Ключевые метрики:**
- Время ответа API (< 200ms)
- Количество ошибок в секунду (< 1/min)
- Использование памяти (< 512MB)
- Размер БД (< 1GB)
- Количество активных подключений

### Prometheus + Grafana (опционально)

**docker-compose.monitoring.yml:**
```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

volumes:
  prometheus_data:
  grafana_data:
```

---

## 💾 Резервное копирование

### Стратегия 3-2-1

- **3** копии данных
- **2** разных типа носителей
- **1** копия вне площадки

### Скрипт полного бэкапа

**scripts/full_backup.sh:**
```bash
#!/bin/bash

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/backups/nanoprobe"
PROJECT_DIR="/opt/nanoprobe-sim-lab"

mkdir -p $BACKUP_DIR

# Бэкап БД
sqlite3 $PROJECT_DIR/data/nanoprobe.db ".backup '$BACKUP_DIR/db_$DATE.db'"

# Бэкап файлов
tar -czf $BACKUP_DIR/files_$DATE.tar.gz \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    -C $PROJECT_DIR .

# Сжатие БД
gzip $BACKUP_DIR/db_$DATE.db

# Чексумма
sha256sum $BACKUP_DIR/db_$DATE.db.gz > $BACKUP_DIR/db_$DATE.db.gz.sha256
sha256sum $BACKUP_DIR/files_$DATE.tar.gz > $BACKUP_DIR/files_$DATE.tar.gz.sha256

# Удаление старых бэкапов
find $BACKUP_DIR -mtime +30 -delete

echo "Backup completed: $DATE"
```

### Восстановление из бэкапа

```bash
# Восстановление БД
gunzip db_20260311_120000.db.gz
sqlite3 data/nanoprobe.db ".restore db_20260311_120000.db"

# Восстановление файлов
tar -xzf files_20260311_120000.tar.gz -C /opt/nanoprobe-sim-lab/
```

---

## 🔒 Безопасность

### JWT токены

**Настройка в .env:**
```bash
JWT_SECRET=your-super-secret-key-min-32-characters-random
JWT_EXPIRATION_MINUTES=60
JWT_REFRESH_EXPIRATION_DAYS=7
```

**Рекомендации:**
- Меняйте JWT_SECRET каждые 90 дней
- Используйте минимум 32 случайных символа
- Храните в secure vault (не в коде!)

### Firewall правила

**UFW (Ubuntu):**
```bash
# Базовая настройка
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Разрешить SSH
sudo ufw allow 22/tcp

# Разрешить HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Включить
sudo ufw enable
sudo ufw status
```

### SSL/TLS сертификат

**Let's Encrypt:**
```bash
# Установка certbot
sudo apt install certbot python3-certbot-nginx

# Получение сертификата
sudo certbot --nginx -d nanoprobe.yourdomain.com

# Автообновление
sudo certbot renew --dry-run
```

### Аудит безопасности

**Ежемесячный чек-лист:**
- [ ] Проверка логов на подозрительную активность
- [ ] Обновление зависимостей
- [ ] Проверка прав доступа к файлам
- [ ] Тестирование бэкапов
- [ ] Аудит пользователей

---

## ⚡ Производительность

### Оптимизация SQLite

```sql
-- Индексы (уже созданы в database.py)
CREATE INDEX idx_scan_timestamp ON scan_results(timestamp);
CREATE INDEX idx_scan_type ON scan_results(scan_type);
CREATE INDEX idx_scan_file_path ON scan_results(file_path);

-- Регулярная оптимизация
VACUUM;
ANALYZE;
```

### Кэширование

**Redis конфигурация:**
```bash
# Установка
sudo apt install redis-server

# Настройка
sudo nano /etc/redis/redis.conf
# maxmemory 256mb
# maxmemory-policy allkeys-lru

# Перезапуск
sudo systemctl restart redis
```

### Профилирование

```bash
# Профилирование API
python -m cProfile -o profile.stats run_api.py

# Анализ памяти
python -m memory_profiler utils/database.py

# Benchmark
ab -n 1000 -c 10 http://localhost:8000/health
```

---

## 🔧 Решение проблем

### API не запускается

```bash
# Проверка логов
journalctl -u nanoprobe-api -n 50

# Проверка порта
netstat -tlnp | grep 8000

# Проверка прав
ls -la /opt/nanoprobe-sim-lab/data/

# Тестовый запуск
sudo -u nanoprobe -i
cd /opt/nanoprobe-sim-lab
source venv/bin/activate
python run_api.py
```

### Ошибка "Database locked"

```bash
# Найти процессы
lsof data/nanoprobe.db

# Удалить WAL файлы
rm data/nanoprobe.db-wal data/nanoprobe.db-shm

# Перезапустить сервисы
sudo systemctl restart nanoprobe-api nanoprobe-web
```

### Утечка памяти

```bash
# Мониторинг
watch -n 1 'ps aux | grep nanoprobe'

# Перезапуск
sudo systemctl restart nanoprobe-api

# Анализ
python -m memory_profiler run_api.py
```

### Высокая загрузка CPU

```bash
# Топ процессов
top -p $(pgrep -d',' -f nanoprobe)

# Профилирование
py-spy top --pid <PID>

# Ограничение
systemctl set-property nanoprobe-api.service CPUQuota=50%
```

---

## ✅ Чек-листы

### Ежедневный чек-лист

- [ ] Проверка health endpoint
- [ ] Просмотр логов на ошибки
- [ ] Проверка свободного места
- [ ] Мониторинг использования RAM

### Еженедельный чек-лист

- [ ] Обновление зависимостей
- [ ] Анализ медленных запросов
- [ ] Проверка бэкапов
- [ ] Очистка старых логов

### Ежемесячный чек-лист

- [ ] Security updates
- [ ] Аудит пользователей
- [ ] Тестирование восстановления
- [ ] Review производительности
- [ ] Обновление документации

---

## 📞 Поддержка

**Контакты:**
- Email: maksimqwe42@mail.ru
- Сайт: https://school-maestro7it.ru/

**Время ответа:** 24-48 часов

---

*Документация для администратора v1.0.0*  
*Последнее обновление: 2026-03-11*
