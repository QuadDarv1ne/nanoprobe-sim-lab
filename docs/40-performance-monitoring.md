# Performance Monitoring Dashboard

## Мониторинг производительности Nanoprobe Sim Lab

**Статус:** ✅ Реализовано (2026-03-15)

---

## 🚀 Возможности

### System Metrics
- CPU usage (per core)
- Memory usage
- Disk usage (per partition)
- Network traffic
- Process info
- Uptime

### API Metrics
- Request count (total, by endpoint, by status)
- Request latency (histogram)
- Requests in progress

### Business Metrics
- SSTV recordings total
- SSTV decoded images
- Scans in database
- Active simulations
- Upload queue size

---

## 📊 Endpoints

### Prometheus Metrics

**URL:** `GET /api/v1/monitoring/metrics`

**Content-Type:** `text/plain; version=0.0.4`

**Пример:**
```
# HELP nanoprobe_cpu_percent CPU usage percentage
# TYPE nanoprobe_cpu_percent gauge
nanoprobe_cpu_percent{core="core_0"} 25.5
nanoprobe_cpu_percent{core="core_1"} 30.2
# HELP nanoprobe_api_requests_total Total API requests
# TYPE nanoprobe_api_requests_total counter
nanoprobe_api_requests_total{method="GET",endpoint="/api/v1/scans",status="200"} 142
```

**Использование с Prometheus:**
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'nanoprobe'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/api/v1/monitoring/metrics'
```

### Detailed Health Check

**URL:** `GET /api/v1/monitoring/health/detailed`

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-03-15T10:30:00Z",
  "system": {
    "cpu": {
      "percent": 25.5,
      "per_core": [25.5, 30.2, 28.1, 32.4],
      "frequency_mhz": 3200
    },
    "memory": {
      "percent": 65.2,
      "available_mb": 7168,
      "total_mb": 16384
    },
    "disk": {
      "percent": 45.8,
      "free_gb": 256,
      "total_gb": 512
    },
    "network": {
      "bytes_sent_mb": 1024,
      "bytes_recv_mb": 2048,
      "packets_sent": 15000,
      "packets_recv": 25000
    }
  },
  "process": {
    "pid": 12345,
    "cpu_percent": 15.2,
    "memory_percent": 8.5,
    "num_threads": 12,
    "status": "running",
    "uptime_seconds": 3600
  }
}
```

### Monitoring Stats

**URL:** `GET /api/v1/monitoring/stats`

**Response:**
```json
{
  "uptime": {
    "seconds": 3600,
    "formatted": "1:00:00",
    "boot_time": "2026-03-15T09:30:00Z"
  },
  "cpu": {
    "cores": 4,
    "logical_cores": 8
  },
  "memory": {
    "total_gb": 16
  },
  "disk": {
    "partitions": 3
  }
}
```

---

## 🔧 Настройка

### Переменные окружения

```bash
# Порт для Prometheus metrics server
METRICS_PORT=9090

# Опционально: Redis для кэширования
REDIS_HOST=localhost
REDIS_PORT=6379
```

### Запуск с мониторингом

```bash
# API запустится с metrics server на порту 9090
python main.py api --port 8000

# Metrics доступны на http://localhost:9090
curl http://localhost:8000/api/v1/monitoring/metrics
```

---

## 📈 Интеграция с Grafana

### Dashboard JSON

Импортируйте этот dashboard в Grafana:

```json
{
  "dashboard": {
    "title": "Nanoprobe Sim Lab Monitoring",
    "panels": [
      {
        "title": "CPU Usage",
        "targets": [
          {
            "expr": "nanoprobe_cpu_percent",
            "legendFormat": "{{core}}"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "targets": [
          {
            "expr": "nanoprobe_memory_percent"
          }
        ]
      },
      {
        "title": "API Request Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(nanoprobe_api_request_latency_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "SSTV Recordings",
        "targets": [
          {
            "expr": "rate(nanoprobe_sstv_recordings_total[5m])"
          }
        ]
      }
    ]
  }
}
```

---

## 🧪 Тестирование

### Проверка metrics

```bash
# Получить metrics
curl http://localhost:8000/api/v1/monitoring/metrics

# Проверить health
curl http://localhost:8000/api/v1/monitoring/health/detailed

# Проверить stats
curl http://localhost:8000/api/v1/monitoring/stats
```

### Python client

```python
from utils.monitoring.performance_monitor import (
    record_sstv_recording,
    record_sstv_decoded,
    update_scans_count,
    update_simulations_active
)

# Запись SSTV записи
record_sstv_recording()

# Запись успешного декодирования
record_sstv_decoded()

# Обновление счётчиков
update_scans_count(42)
update_simulations_active(3)
```

---

## 🎯 Метрики

### System Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `nanoprobe_cpu_percent` | Gauge | CPU usage per core |
| `nanoprobe_memory_percent` | Gauge | Memory usage % |
| `nanoprobe_memory_available_bytes` | Gauge | Available memory |
| `nanoprobe_disk_percent` | Gauge | Disk usage per mountpoint |
| `nanoprobe_disk_free_bytes` | Gauge | Free disk space |
| `nanoprobe_network_sent_bytes_total` | Counter | Total bytes sent |
| `nanoprobe_network_recv_bytes_total` | Counter | Total bytes received |

### API Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `nanoprobe_api_requests_total` | Counter | Total requests by method/endpoint/status |
| `nanoprobe_api_request_latency_seconds` | Histogram | Request latency distribution |
| `nanoprobe_api_requests_in_progress` | Gauge | Current in-progress requests |

### Business Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `nanoprobe_sstv_recordings_total` | Counter | Total SSTV recordings |
| `nanoprobe_sstv_decoded_images_total` | Counter | Successfully decoded SSTV images |
| `nanoprobe_scans_total` | Gauge | Scans in database |
| `nanoprobe_simulations_active` | Gauge | Active simulations |
| `nanoprobe_upload_queue_size` | Gauge | Upload queue size |

---

## 🚨 Alerts

### Prometheus alert rules

```yaml
groups:
  - name: nanoprobe
    rules:
      - alert: HighCPU
        expr: nanoprobe_cpu_percent > 80
        for: 5m
        annotations:
          summary: "High CPU usage detected"

      - alert: HighMemory
        expr: nanoprobe_memory_percent > 90
        for: 5m
        annotations:
          summary: "High memory usage detected"

      - alert: SSTVRecordingFailed
        expr: rate(nanoprobe_sstv_recordings_total[5m]) == 0
        for: 10m
        annotations:
          summary: "No SSTV recordings in last 10 minutes"
```

---

*Обновлено: 2026-03-15*
