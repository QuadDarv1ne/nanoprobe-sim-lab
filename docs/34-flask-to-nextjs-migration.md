# План миграции Flask → Next.js

## Обзор

Этот документ описывает полный план миграции с Flask dashboard на Next.js, включая таймлайн, этапы и стратегию деприкации.

## Текущее состояние

### Flask Dashboard (v1.0 - Legacy)
- **Порт**: http://localhost:5000
- **Файл**: `templates/dashboard.html`
- **Технологии**: Flask + Jinja2 + Socket.IO + Chart.js
- **Статус**: ✅ Стабильная, проверенная версия

### Next.js Dashboard (v2.0 - Modern)
- **Порт**: http://localhost:3000
- **Папка**: `frontend/`
- **Технологии**: Next.js 14 + TypeScript + Tailwind CSS + Zustand
- **Статус**: ✅ Новая, современная версия

## Таймлайн миграции

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MIGRATION TIMELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Phase 1              Phase 2              Phase 3              Phase 4    │
│  Foundation           Feature Parity       Deprecation          Removal    │
│  (2 weeks)            (4 weeks)            (2 weeks)            (1 week)   │
│                                                                              │
│  ├──────────────────┤ ├──────────────────┤ ├──────────────────┤ ├─────────┤│
│  │                  │ │                  │ │                  │ │         ││
│  │  ✓ PWA setup     │ │  ✓ All features  │ │  ✓ Warnings      │ │  ✓ Remove││
│  │  ✓ Auth          │ │  ✓ Tests pass    │ │  ✓ Docs update   │ │  ✓ Final ││
│  │  ✓ Core API      │ │  ✓ Performance   │ │  ✓ Redirects     │ │  ✓ Deploy││
│  │  ✓ Basic pages   │ │  ✓ User testing  │ │  ✓ Monitoring    │ │         ││
│  │                  │ │                  │ │                  │ │         ││
│  └──────────────────┘ └──────────────────┘ └──────────────────┘ └─────────┘│
│                                                                              │
│  Week 1-2             Week 3-6             Week 7-8            Week 9      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Phase 1: Foundation (Week 1-2)

### Цели
- Настроить PWA в Next.js
- Реализовать аутентификацию
- Подключить все основные API endpoints
- Создать базовые страницы

### Задачи

#### Week 1: Setup

```markdown
- [ ] **PWA Configuration**
  - [ ] Создать manifest.json
  - [ ] Настроить Service Worker
  - [ ] Сгенерировать иконки (72x72 - 512x512)
  - [ ] Добавить offline страницу
  - [ ] Настроить next-pwa в next.config.js

- [ ] **Authentication**
  - [ ] Login page с формой
  - [ ] JWT token handling
  - [ ] Token refresh mechanism
  - [ ] Protected routes middleware
  - [ ] 2FA страница (TOTP)
  - [ ] Logout functionality

- [ ] **API Client Setup**
  - [ ] Axios instance с interceptors
  - [ ] Error handling
  - [ ] Rate limit handling
  - [ ] Request retry logic
```

#### Week 2: Core Pages

```markdown
- [ ] **Dashboard Page**
  - [ ] System metrics widget
  - [ ] Component status widget
  - [ ] Recent logs widget
  - [ ] Quick actions panel
  - [ ] Real-time updates (WebSocket)

- [ ] **SSTV Page**
  - [ ] ISS position tracker
  - [ ] Schedule display
  - [ ] SSTV decoder interface
  - [ ] Signal strength indicator

- [ ] **SPM Simulator Page**
  - [ ] Scan controls
  - [ ] Visualization canvas
  - [ ] Parameters panel
  - [ ] Results display

- [ ] **Layout Components**
  - [ ] Navigation sidebar
  - [ ] Header with user menu
  - [ ] Footer
  - [ ] Responsive design
```

### Код для Phase 1

```typescript
// frontend/src/app/layout.tsx
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { Providers } from '@/providers/Providers';
import { AuthProvider } from '@/providers/AuthProvider';

const inter = Inter({ subsets: ['latin', 'cyrillic'] });

export const metadata: Metadata = {
  title: 'Nanoprobe Sim Lab',
  description: 'Лаборатория моделирования нанозонда',
  manifest: '/manifest.json',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ru" suppressHydrationWarning>
      <head>
        <link rel="manifest" href="/manifest.json" />
        <link rel="apple-touch-icon" href="/icons/icon-192x192.png" />
        <meta name="theme-color" content="#0f172a" />
      </head>
      <body className={inter.className}>
        <AuthProvider>
          <Providers>{children}</Providers>
        </AuthProvider>
      </body>
    </html>
  );
}
```

```typescript
// frontend/src/providers/AuthProvider.tsx
'use client';

import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { useRouter } from 'next/navigation';
import { apiClient } from '@/lib/apiClient';

interface User {
  id: number;
  username: string;
  email: string;
  is_2fa_enabled: boolean;
}

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  verify2FA: (code: string) => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [pending2FA, setPending2FA] = useState(false);
  const router = useRouter();

  useEffect(() => {
    checkAuth();
  }, []);

  const checkAuth = async () => {
    try {
      const token = localStorage.getItem('access_token');
      if (token) {
        const userData = await apiClient.get<User>('/api/v1/auth/me');
        setUser(userData);
      }
    } catch (error) {
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
    } finally {
      setIsLoading(false);
    }
  };

  const login = async (username: string, password: string) => {
    const response = await apiClient.post<{
      access_token: string;
      refresh_token: string;
      requires_2fa: boolean;
    }>('/api/v1/auth/login', { username, password });

    if (response.requires_2fa) {
      setPending2FA(true);
      localStorage.setItem('temp_token', response.access_token);
      router.push('/auth/2fa');
    } else {
      localStorage.setItem('access_token', response.access_token);
      localStorage.setItem('refresh_token', response.refresh_token);
      await checkAuth();
      router.push('/dashboard');
    }
  };

  const verify2FA = async (code: string) => {
    const tempToken = localStorage.getItem('temp_token');
    const response = await apiClient.post<{
      access_token: string;
      refresh_token: string;
    }>('/api/v1/auth/2fa/verify-login', {
      code,
      temp_token: tempToken
    });

    localStorage.removeItem('temp_token');
    localStorage.setItem('access_token', response.access_token);
    localStorage.setItem('refresh_token', response.refresh_token);
    await checkAuth();
    router.push('/dashboard');
  };

  const logout = async () => {
    try {
      await apiClient.post('/api/v1/auth/logout');
    } finally {
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
      setUser(null);
      router.push('/auth/login');
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isAuthenticated: !!user,
        login,
        logout,
        verify2FA,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
```

## Phase 2: Feature Parity (Week 3-6)

### Сравнение функций

| Функция | Flask | Next.js | Статус |
|---------|-------|---------|--------|
| Dashboard Overview | ✅ | 🚧 | Week 3 |
| System Monitoring | ✅ | 🚧 | Week 3 |
| SSTV Station | ✅ | 🚧 | Week 4 |
| ISS Tracker | ✅ | 🚧 | Week 4 |
| SPM Simulator | ✅ | 🚧 | Week 5 |
| Image Analyzer | ✅ | 🚧 | Week 5 |
| NASA API Integration | ✅ | 🚧 | Week 5 |
| Settings Page | ✅ | 🚧 | Week 6 |
| User Management | ✅ | 🚧 | Week 6 |
| Real-time Updates | ✅ | 🚧 | Week 6 |

### Недостающие функции для имплементации

```typescript
// frontend/src/features/sstv/components/ISSTracker.tsx
'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useWebSocket } from '@/hooks/useWebSocket';

interface ISSPosition {
  latitude: number;
  longitude: number;
  altitude: number;
  velocity: number;
  timestamp: number;
}

export function ISSTracker() {
  const [position, setPosition] = useState<ISSPosition | null>(null);
  const { lastMessage, isConnected } = useWebSocket('/api/v1/sstv/ws/iss');

  useEffect(() => {
    if (lastMessage?.data) {
      const data = JSON.parse(lastMessage.data);
      setPosition(data.data);
    }
  }, [lastMessage]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          🛰️ ISS Position
          {isConnected && (
            <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {position ? (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-muted-foreground">Latitude</p>
                <p className="text-xl font-mono">{position.latitude.toFixed(4)}°</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Longitude</p>
                <p className="text-xl font-mono">{position.longitude.toFixed(4)}°</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Altitude</p>
                <p className="text-xl font-mono">{position.altitude.toFixed(1)} km</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Velocity</p>
                <p className="text-xl font-mono">{position.velocity.toFixed(0)} km/h</p>
              </div>
            </div>

            {/* Map component */}
            <div className="aspect-video bg-slate-100 dark:bg-slate-800 rounded-lg relative overflow-hidden">
              <ISSMap latitude={position.latitude} longitude={position.longitude} />
            </div>
          </div>
        ) : (
          <div className="text-center py-8 text-muted-foreground">
            Loading ISS position...
          </div>
        )}
      </CardContent>
    </Card>
  );
}
```

### WebSocket Hook

```typescript
// frontend/src/hooks/useWebSocket.ts
'use client';

import { useEffect, useRef, useState, useCallback } from 'react';

interface UseWebSocketOptions {
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (error: Event) => void;
  onMessage?: (data: any) => void;
  reconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

export function useWebSocket(
  url: string,
  options: UseWebSocketOptions = {}
) {
  const {
    onOpen,
    onClose,
    onError,
    onMessage,
    reconnect = true,
    reconnectInterval = 3000,
    maxReconnectAttempts = 5,
  } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<MessageEvent | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttempts = useRef(0);

  const connect = useCallback(() => {
    const wsUrl = `${process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'}${url}`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      reconnectAttempts.current = 0;
      onOpen?.();
    };

    ws.onclose = () => {
      setIsConnected(false);
      onClose?.();

      if (reconnect && reconnectAttempts.current < maxReconnectAttempts) {
        reconnectAttempts.current++;
        setTimeout(connect, reconnectInterval);
      }
    };

    ws.onerror = (error) => {
      onError?.(error);
    };

    ws.onmessage = (event) => {
      setLastMessage(event);
      onMessage?.(JSON.parse(event.data));
    };
  }, [url, reconnect, reconnectInterval, maxReconnectAttempts, onOpen, onClose, onError, onMessage]);

  const send = useCallback((data: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  useEffect(() => {
    connect();

    return () => {
      wsRef.current?.close();
    };
  }, [connect]);

  return {
    isConnected,
    lastMessage,
    send,
    reconnect: connect,
  };
}
```

## Phase 3: Deprecation (Week 7-8)

### Предупреждения в Flask

```python
# src/web/web_dashboard.py
from flask import Flask, render_template, jsonify
import logging

logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.before_request
def deprecation_warning():
    """Предупреждение о деприкации Flask версии"""
    logger.warning(
        "Flask dashboard is deprecated. "
        "Please use Next.js version at http://localhost:3000"
    )

@app.route('/')
def index():
    return render_template('dashboard.html',
        deprecation_notice=True,
        new_url="http://localhost:3000"
    )

@app.route('/api/deprecation-status')
def deprecation_status():
    """API endpoint для проверки статуса деприкации"""
    return jsonify({
        "deprecated": True,
        "removal_date": "2026-04-15",
        "new_url": "http://localhost:3000",
        "days_remaining": 30
    })
```

### Banner Component в Flask

```html
<!-- templates/dashboard.html -->
{% if deprecation_notice %}
<div class="deprecation-banner" style="
    background: linear-gradient(135deg, #f59e0b, #d97706);
    color: white;
    padding: 12px 24px;
    text-align: center;
    font-weight: 500;
    position: sticky;
    top: 0;
    z-index: 9999;
">
    ⚠️ Эта версия dashboard устарела и будет удалена 15 апреля 2026.
    <a href="{{ new_url }}" style="color: white; text-decoration: underline;">
        Перейти на новую версию →
    </a>
    <button onclick="this.parentElement.remove()" style="
        background: transparent;
        border: none;
        color: white;
        margin-left: 20px;
        cursor: pointer;
    ">✕</button>
</div>
{% endif %}
```

### Redirect Endpoint

```python
# api/routes/migration.py
from fastapi import APIRouter, RedirectResponse

router = APIRouter()

@router.get("/migrate")
async def migrate_to_nextjs():
    """Redirect to new Next.js dashboard"""
    return RedirectResponse(
        url="http://localhost:3000",
        status_code=301
    )
```

## Phase 4: Removal (Week 9)

### Checklist для удаления

```markdown
## Flask Removal Checklist

### Pre-removal
- [ ] Убедиться, что все функции работают в Next.js
- [ ] Все пользователи переведены на новую версию
- [ ] Документация обновлена
- [ ] Тесты проходят

### Files to Remove
- [ ] `src/web/web_dashboard.py`
- [ ] `templates/dashboard.html`
- [ ] `templates/components/` (Flask templates)
- [ ] Flask-specific requirements

### Code to Update
- [ ] Remove Flask from requirements.txt
- [ ] Remove Flask routes from main.py
- [ ] Update documentation
- [ ] Update docker-compose.yml

### Post-removal
- [ ] Update README.md
- [ ] Update FRONTEND_VERSIONS.md
- [ ] Announce removal
- [ ] Deploy new version
```

### Скрипт удаления

```bash
#!/bin/bash
# scripts/remove_flask.sh

echo "🚀 Removing Flask dashboard..."

# Backup
echo "📦 Creating backup..."
tar -czf flask_backup_$(date +%Y%m%d).tar.gz \
    src/web/ \
    templates/

# Remove Flask files
echo "🗑️ Removing Flask files..."
rm -rf src/web/
rm -rf templates/

# Update requirements
echo "📝 Updating requirements.txt..."
sed -i '/flask/d' requirements.txt
sed -i '/flask-socketio/d' requirements.txt
sed -i '/flask-login/d' requirements.txt

# Update main.py
echo "🔧 Updating main.py..."
sed -i '/Service.FLASK/d' main.py
sed -i '/_start_flask/d' main.py

# Update docs
echo "📚 Updating documentation..."
echo "Flask dashboard removed. Use Next.js version at http://localhost:3000" > FLASK_REMOVED.md

echo "✅ Flask dashboard removed successfully!"
echo "Next.js is now the primary frontend at http://localhost:3000"
```

## Deployment Strategy

### Docker Compose Update

```yaml
# docker-compose.yml (after migration)
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://...
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://api:8000
    depends_on:
      - api

  db:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### Nginx Configuration

```nginx
# nginx.conf
server {
    listen 80;
    server_name nanoprobe.example.com;

    # Next.js frontend
    location / {
        proxy_pass http://frontend:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # API backend
    location /api/ {
        proxy_pass http://api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket
    location /ws/ {
        proxy_pass http://api:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}
```

## Success Metrics

| Метрика | Цель | Измерение |
|---------|------|-----------|
| Feature Parity | 100% | Чек-лист функций |
| Test Coverage | > 80% | Jest/Vitest |
| Performance (LCP) | < 2.5s | Lighthouse |
| PWA Score | 100 | Lighthouse |
| User Satisfaction | > 4.0/5 | Survey |
| Migration Rate | 100% | Analytics |

## Rollback Plan

```markdown
## Rollback Procedure

Если миграция не удалась:

1. Восстановить Flask из бэкапа:
   ```bash
   tar -xzf flask_backup_YYYYMMDD.tar.gz
   ```

2. Переключить nginx на Flask:
   ```nginx
   location / {
       proxy_pass http://flask:5000;
   }
   ```

3. Восстановить зависимости:
   ```bash
   pip install flask flask-socketio flask-login
   ```

4. Перезапустить сервисы:
   ```bash
   docker-compose restart
   ```
```
