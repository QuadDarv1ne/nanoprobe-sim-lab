# Deployment Guide

## Развёртывание Nanoprobe Sim Lab

**Last Updated:** 2026-03-15

---

## 🚀 Платформы для деплоя

### Поддерживаемые платформы:

1. **Vercel** - Frontend (Next.js)
2. **Railway** - Full stack (Backend + Frontend)
3. **Render** - Full stack с managed DB
4. **Docker** - Self-hosted
5. **Manual** - VPS/VDS

---

## 1️⃣ Vercel (Frontend Only)

### Быстрый старт

```bash
# Установить Vercel CLI
npm install -g vercel

# Войти в аккаунт
vercel login

# Деплой frontend
cd frontend
vercel --prod
```

### Конфигурация

**File:** `vercel.json`

```json
{
  "framework": "nextjs",
  "regions": ["fra1"],
  "env": {
    "NEXT_PUBLIC_API_URL": "https://api.your-domain.com",
    "NEXT_PUBLIC_WS_URL": "wss://api.your-domain.com"
  }
}
```

### Environment Variables

```bash
# В Vercel Dashboard
NEXT_PUBLIC_API_URL=https://your-api.com
NEXT_PUBLIC_WS_URL=wss://your-api.com
```

### Auto-deploy

```bash
# Connect GitHub repo
vercel link

# Enable auto-deploy on push
vercel --prod
```

---

## 2️⃣ Railway (Full Stack)

### Быстрый старт

1. Зайти на https://railway.app
2. "New Project" → "Deploy from GitHub"
3. Выбрать репозиторий `nanoprobe-sim-lab`
4. Railway автоматически обнаружит `railway.json`

### Конфигурация

**File:** `railway.json`

```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "python main.py api --port $PORT"
  }
}
```

### Environment Variables

```bash
# В Railway Dashboard
JWT_SECRET=your-secret-key
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
NASA_API_KEY=your-key
```

### Pricing

- Starter: $5/month
- Pro: $20/month
- Includes: Backend + Database + Redis

---

## 3️⃣ Render (Full Stack with Managed DB)

### Быстрый старт

1. Зайти на https://render.com
2. "New +" → "Blueprint"
3. Connect GitHub repo
4. Render автоматически применит `render.yaml`

### Конфигурация

**File:** `render.yaml`

```yaml
services:
  - type: web
    name: nanoprobe-sim-lab
    env: python
    buildCommand: |
      pip install -r requirements.txt
      cd frontend && npm install && npm run build
    startCommand: python main.py api
```

### Services

- **Web Service:** Backend + Frontend
- **PostgreSQL:** Managed database
- **Redis:** Managed cache

### Pricing

- Starter: $7/month (web) + $7/month (DB) + $7/month (Redis)
- Total: ~$21/month

---

## 4️⃣ Docker (Self-hosted)

### Production Deploy

```bash
# Build images
docker-compose -f deployment/docker-compose.prod.yml build

# Start services
docker-compose -f deployment/docker-compose.prod.yml up -d

# Check status
docker-compose ps
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| nginx | 80, 443 | Reverse proxy |
| fastapi | 8000 | Backend API |
| nextjs | 3000 | Frontend |
| postgres | 5432 | Database |
| redis | 6379 | Cache |
| prometheus | 9090 | Monitoring |
| grafana | 3001 | Visualization |

### SSL/TLS

```bash
# Let's Encrypt
docker run -it --rm \
  -v ./ssl:/etc/letsencrypt \
  certbot/certbot certonly \
  --standalone -d your-domain.com
```

### Update

```bash
# Pull latest images
docker-compose pull

# Restart services
docker-compose up -d
```

---

## 5️⃣ Manual (VPS/VDS)

### Requirements

- Ubuntu 20.04+ / Debian 11+
- Python 3.11+
- Node.js 20+
- PostgreSQL 15+
- Redis 7+
- Nginx

### Installation

```bash
# Clone repo
git clone https://github.com/QuadDarv1ne/nanoprobe-sim-lab.git
cd nanoprobe-sim-lab

# Install Python dependencies
pip install -r requirements.txt
pip install -r requirements-api.txt

# Install Node dependencies
cd frontend
npm install
npm run build

# Setup environment
cp deployment/.env.example .env
nano .env  # Edit with your values

# Start API
python main.py api --port 8000
```

### Nginx Configuration

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location /api/ {
        proxy_pass http://localhost:8000;
    }

    location / {
        proxy_pass http://localhost:3000;
    }
}
```

### Systemd Service

```ini
# /etc/systemd/system/nanoprobe.service
[Unit]
Description=Nanoprobe Sim Lab
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/nanoprobe-sim-lab
ExecStart=/usr/bin/python3 main.py api
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable service
sudo systemctl enable nanoprobe
sudo systemctl start nanoprobe
sudo systemctl status nanoprobe
```

---

## 🔐 Security

### Environment Variables

**Never commit secrets!**

```bash
# .env (add to .gitignore)
JWT_SECRET=your-secret-key
DATABASE_URL=postgresql://user:pass@localhost/db
REDIS_URL=redis://localhost:6379
NASA_API_KEY=your-key
```

### SSL/TLS

**Required for production!**

```bash
# Let's Encrypt (free)
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo systemctl enable certbot.timer
```

### Firewall

```bash
# UFW (Ubuntu)
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

---

## 📊 Monitoring

### Health Checks

```bash
# API health
curl https://your-domain.com/api/v1/health

# Detailed health
curl https://your-domain.com/api/v1/monitoring/health/detailed

# Prometheus metrics
curl https://your-domain.com/api/v1/monitoring/metrics
```

### Logs

```bash
# Docker
docker-compose logs -f fastapi
docker-compose logs -f nextjs

# Systemd
sudo journalctl -u nanoprobe -f

# Application
tail -f logs/api/nanoprobe_info.log
```

---

## 🔄 CI/CD

### GitHub Actions

**File:** `.github/workflows/deploy.yml`

```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy to Vercel
        uses: amondnet/vercel-action@v25
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.VERCEL_ORG_ID }}
          vercel-project-id: ${{ secrets.VERCEL_PROJECT_ID }}
          vercel-args: '--prod'
```

### Auto-deploy

| Platform | Trigger | Command |
|----------|---------|---------|
| Vercel | Push to main | `vercel --prod` |
| Railway | Push to main | Auto |
| Render | Push to main | Auto |
| Docker | Manual | `docker-compose up -d` |

---

## 📱 Mobile Deployment

### PWA Installation

**iOS (Safari):**
1. Open https://your-domain.com/mobile
2. Share → Add to Home Screen
3. Tap Add

**Android (Chrome):**
1. Open https://your-domain.com/mobile
2. Menu → Install app
3. Tap Install

### App Stores (Optional)

**Capacitor for native apps:**

```bash
# Install Capacitor
npm install @capacitor/core @capacitor/cli

# Initialize
npx cap init

# Build
npm run build
npx cap add ios
npx cap add android

# Deploy to stores
npx cap open ios  # Xcode
npx cap open android  # Android Studio
```

---

## 🎯 Recommended Setup

### For Development

- **Platform:** Vercel (frontend) + Railway (backend)
- **Cost:** Free (Vercel) + $5/month (Railway)
- **Setup Time:** ~15 minutes

### For Production

- **Platform:** Docker self-hosted or Render
- **Cost:** $20-30/month
- **Setup Time:** ~1 hour

### For Enterprise

- **Platform:** Kubernetes (GKE/AKS/EKS)
- **Cost:** $100+/month
- **Setup Time:** ~1 day

---

## 🔗 Links

- [Vercel Documentation](https://vercel.com/docs)
- [Railway Documentation](https://docs.railway.app)
- [Render Documentation](https://render.com/docs)
- [Docker Documentation](https://docs.docker.com)

---

*Updated: 2026-03-15*
