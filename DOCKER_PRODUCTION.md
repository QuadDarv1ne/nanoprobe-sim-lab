# Docker Production Deployment Guide

## Production развёртывание Nanoprobe Sim Lab

**Статус:** ✅ Реализовано (2026-03-15)

---

## 📦 Архитектура

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Production Stack                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────┐                                                                 │
│  │ Nginx   │ :80, :443 (Reverse Proxy + SSL)                               │
│  └────┬────┘                                                                 │
│       │                                                                      │
│       ├──────────────┬───────────────┬──────────────┐                       │
│       │              │               │              │                       │
│  ┌────▼────┐    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐                 │
│  │FastAPI  │    │ Flask   │    │ Next.js │    │ Worker  │                 │
│  │:8000    │    │:5000    │    │:3000    │    │(bg)     │                 │
│  └────┬────┘    └─────────┘    └─────────┘    └─────────┘                 │
│       │                                                                   │
│       ├──────────────┬──────────────┐                                     │
│       │              │              │                                     │
│  ┌────▼────┐    ┌────▼────┐    ┌───▼────┐                               │
│  │PostgreSQL│    │ Redis   │    │Monitoring│                             │
│  │:5432    │    │:6379    │    │(Prom+Graf)│                            │
│  └─────────┘    └─────────┘    └──────────┘                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Быстрый старт

### 1. Подготовка

```bash
# Перейдите в директорию deployment
cd deployment

# Скопируйте .env.example в .env
cp .env.example .env

# Отредактируйте .env с вашими production значениями
nano .env
```

### 2. Развёртывание

```bash
# Автоматическое развёртывание
./deploy.sh up

# Или вручную через docker-compose
docker-compose -f docker-compose.prod.yml up -d
```

### 3. Проверка

```bash
# Статус сервисов
./deploy.sh status

# Логи
./deploy.sh logs

# Проверка health endpoints
curl http://localhost:80/health
curl http://localhost:80/api/v1/nasa/health
```

---

## 📁 Структура файлов

```
deployment/
├── docker-compose.prod.yml       # Production compose
├── docker-compose.yml            # Development compose
├── deploy.sh                     # Deployment script
├── .env.example                  # Environment template
├── docker/
│   ├── Dockerfile.fastapi       # FastAPI image
│   ├── Dockerfile.flask         # Flask image
│   └── Dockerfile.nextjs        # Next.js image
├── nginx/
│   ├── nginx.conf               # Nginx configuration
│   └── conf.d/
│       └── ssl.conf             # SSL settings
├── postgres/
│   └── init.sql                 # Database initialization
├── monitoring/
│   ├── prometheus.yml           # Prometheus config
│   └── grafana/
│       └── provisioning/        # Grafana dashboards
└── ssl/                         # SSL certificates
    ├── fullchain.pem
    └── privkey.pem
```

---

## 🔧 Сервисы

### Nginx (Reverse Proxy)

**Порт:** 80, 443

**Функции:**
- Reverse proxy для всех сервисов
- SSL termination
- Rate limiting
- Gzip compression
- Security headers

**Конфигурация:**
```nginx
location /api/ {
    limit_req zone=api_limit burst=20 nodelay;
    proxy_pass http://fastapi;
}

location /ws/ {
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_pass http://fastapi;
}
```

### FastAPI (Backend)

**Порт:** 8000

**Image:** Multi-stage build (~350MB)

**Features:**
- Gunicorn workers (4)
- Uvicorn worker class
- Health checks
- Resource limits

### Flask (Legacy Frontend)

**Порт:** 5000

**Image:** Multi-stage build (~200MB)

### Next.js (Modern Frontend)

**Порт:** 3000

**Image:** Multi-stage build (~150MB)

**Features:**
- SSR support
- Static optimization
- PWA ready

### PostgreSQL (Database)

**Порт:** 5432

**Image:** postgres:15-alpine

**Volume:** postgres-data:/var/lib/postgresql/data

### Redis (Cache)

**Порт:** 6379

**Image:** redis:7-alpine

**Config:**
```
--appendonly yes
--maxmemory 1gb
--maxmemory-policy allkeys-lru
```

### Prometheus (Monitoring)

**Порт:** 9090

**Metrics:**
- API requests
- Response times
- System metrics
- Cache hit rates

### Grafana (Visualization)

**Порт:** 3001

**Dashboards:**
- API Performance
- System Health
- Cache Statistics
- Business Metrics

---

## 🔐 Security

### Environment Variables

**Критичные переменные:**
```bash
# Security
JWT_SECRET=<32-char-random-string>
DB_PASSWORD=<secure-password>

# NASA API
NASA_API_KEY=<your-key>

# Monitoring
GRAFANA_ADMIN_PASSWORD=<secure-password>
```

### SSL/TLS

**Для production используйте HTTPS:**

1. **Let's Encrypt (рекомендуется):**
```bash
docker run -it --rm \
  -v ./ssl:/etc/letsencrypt \
  certbot/certbot certonly \
  --standalone -d your-domain.com
```

2. **Self-signed (для testing):**
```bash
openssl req -x509 -nodes -days 365 \
  -newkey rsa:2048 \
  -keyout ssl/privkey.pem \
  -out ssl/fullchain.pem
```

### Rate Limiting

**Nginx rate limits:**
- API: 10 requests/second
- Auth: 5 requests/minute
- WebSocket: no limit

---

## 📊 Monitoring

### Prometheus Metrics

**Endpoints:**
- `/metrics` — Prometheus metrics
- `/health` — Health check
- `/health/detailed` — Detailed health

**Key metrics:**
```prometheus
http_requests_total
http_request_duration_seconds
cache_hits_total
cache_misses_total
active_users
```

### Grafana Dashboards

**Импортируйте дашборды:**
1. API Performance (ID: 10566)
2. System Health (ID: 8919)
3. Redis Dashboard (ID: 763)
4. PostgreSQL Dashboard (ID: 9628)

---

## 🔄 Deployment

### Automated Deployment

```bash
# Production deploy
./deploy.sh up

# Restart services
./deploy.sh restart

# View logs
./deploy.sh logs

# Check status
./deploy.sh status
```

### Manual Deployment

```bash
# Build images
docker-compose -f docker-compose.prod.yml build

# Start services
docker-compose -f docker-compose.prod.yml up -d

# Check logs
docker-compose -f docker-compose.prod.yml logs -f

# Stop services
docker-compose -f docker-compose.prod.yml down
```

### Scaling

```bash
# Scale FastAPI workers
docker-compose -f docker-compose.prod.yml up -d --scale fastapi=3
```

---

## 🛠️ Troubleshooting

### Service not starting

```bash
# Check logs
docker-compose -f docker-compose.prod.yml logs <service-name>

# Restart service
docker-compose -f docker-compose.prod.yml restart <service-name>
```

### Database connection error

```bash
# Check PostgreSQL is running
docker-compose -f docker-compose.prod.yml ps postgres

# Check connection
docker-compose -f docker-compose.prod.yml exec postgres \
  psql -U nanoprobe -d nanoprobe -c "SELECT 1"
```

### High memory usage

```bash
# Check resource usage
docker stats

# Restart high-memory services
docker-compose -f docker-compose.prod.yml restart fastapi
```

---

## 📈 Performance

### Image Sizes

| Image | Size | Optimization |
|-------|------|-------------|
| FastAPI | ~350MB | Multi-stage build |
| Flask | ~200MB | Minimal dependencies |
| Next.js | ~150MB | Standalone output |
| Nginx | ~25MB | Alpine base |

### Resource Limits

```yaml
fastapi:
  deploy:
    resources:
      limits:
        cpus: '2'
        memory: 2G
      reservations:
        cpus: '0.5'
        memory: 512M

postgres:
  deploy:
    resources:
      limits:
        cpus: '2'
        memory: 4G
```

---

## ✅ Production Checklist

- [ ] .env configured with secure values
- [ ] SSL certificates installed
- [ ] Database backed up
- [ ] Monitoring configured
- [ ] Rate limiting enabled
- [ ] Health checks passing
- [ ] Logs rotating
- [ ] Backups scheduled

---

*Обновлено: 2026-03-15*
