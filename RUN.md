# Nanoprobe Sim Lab - Launch Guide

## Quick Start

### Interactive mode (select version):
```bash
python start.py
```

### Specific version:
```bash
# Flask frontend (v1.0)
python start.py flask

# Next.js frontend (v2.0)
python start.py nextjs

# Only Backend API
python start.py api-only
```

---

## Versions

### Flask Dashboard (v1.0 - Legacy/Stable)
- **Port:** http://localhost:5000
- **Technologies:** Flask + Jinja2 + Socket.IO + Chart.js
- **Launch:** `python start.py flask`

### Next.js Dashboard (v2.0 - Modern/Production)
- **Port:** http://localhost:3000
- **Technologies:** Next.js 14 + TypeScript + Tailwind CSS + Zustand
- **Launch:** `python start.py nextjs`

### Backend API
- **Port:** http://localhost:8000
- **Swagger:** http://localhost:8000/docs
- **Launch:** `python start.py api-only`

---

## Features

- **Automatic port checking** - waits for services to start
- **Auto-open browser** - opens frontend/backend in browser
- **Process monitoring** - watches all services
- **Graceful shutdown** - Ctrl+C stops all services

---

## Documentation

- **Frontend comparison:** [`FRONTEND_VERSIONS.md`](FRONTEND_VERSIONS.md)
- **Main README:** [`README.md`](README.md)
