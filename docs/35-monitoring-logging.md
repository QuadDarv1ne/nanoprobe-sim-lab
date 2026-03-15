# Monitoring and Logging Setup

## Обзор

Комплексная система мониторинга и логирования для Nanoprobe Sim Lab с использованием Prometheus, Grafana, и ELK стека.

## Архитектура мониторинга

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MONITORING ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐                  │
│  │   API   │    │ Frontend│    │   DB    │    │  Redis  │                  │
│  │ :8000   │    │  :3000  │    │  :5432  │    │  :6379  │                  │
│  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘                  │
│       │              │              │              │                        │
│       ▼              ▼              ▼              ▼                        │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │                    Prometheus ( :9090 )                      │           │
│  │                   Metrics Collection                         │           │
│  └─────────────────────────┬───────────────────────────────────┘           │
│                            │                                                │
│                            ▼                                                │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │                    Grafana ( :3001 )                         │           │
│  │                   Visualization & Alerts                     │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │                    Loki ( :3100 )                            │           │
│  │                    Log Aggregation                           │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 1. Application Metrics

```python
# api/monitoring/metrics.py
"""
Prometheus Metrics for FastAPI

Metrics exposed at /metrics endpoint
"""

from prometheus_client import Counter, Histogram, Gauge, Info
from prometheus_fastapi_instrumentator import Instrumentator
import time
from functools import wraps

# ==========================================
# Counters
# ==========================================
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

ACTIVE_REQUESTS = Gauge(
    'http_requests_active',
    'Active HTTP requests',
    ['method']
)

# ==========================================
# Business Metrics
# ==========================================
SCANS_TOTAL = Counter(
    'nanoprobe_scans_total',
    'Total number of scans',
    ['scan_type', 'status']
)

SIMULATIONS_ACTIVE = Gauge(
    'nanoprobe_simulations_active',
    'Number of active simulations'
)

NASA_API_REQUESTS = Counter(
    'nasa_api_requests_total',
    'NASA API requests',
    ['endpoint', 'status']
)

NASA_API_LATENCY = Histogram(
    'nasa_api_request_duration_seconds',
    'NASA API request latency',
    ['endpoint']
)

# ==========================================
# System Metrics
# ==========================================
DB_CONNECTIONS = Gauge(
    'db_connections_active',
    'Active database connections'
)

DB_QUERY_LATENCY = Histogram(
    'db_query_duration_seconds',
    'Database query latency',
    ['query_type'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

REDIS_OPERATIONS = Counter(
    'redis_operations_total',
    'Redis operations',
    ['operation', 'status']
)

CACHE_HIT_RATE = Gauge(
    'cache_hit_rate',
    'Cache hit rate'
)

# ==========================================
# Security Metrics
# ==========================================
AUTH_ATTEMPTS = Counter(
    'auth_attempts_total',
    'Authentication attempts',
    ['type', 'status']  # type: login/refresh/2fa, status: success/failure
)

RATE_LIMIT_HITS = Counter(
    'rate_limit_hits_total',
    'Rate limit hits',
    ['endpoint', 'identifier_type']
)

BLOCKED_IPS = Gauge(
    'blocked_ips_total',
    'Number of blocked IPs'
)

# ==========================================
# Application Info
# ==========================================
APP_INFO = Info(
    'nanoprobe_app',
    'Application information'
)
APP_INFO.info({
    'version': '2.0.0',
    'environment': 'production'
})


# ==========================================
# Decorators for metrics
# ==========================================
def track_db_query(query_type: str):
    """Decorator to track database query latency"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                DB_QUERY_LATENCY.labels(query_type=query_type).observe(duration)
        return wrapper
    return decorator


def track_nasa_api(endpoint: str):
    """Decorator to track NASA API calls"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                NASA_API_REQUESTS.labels(endpoint=endpoint, status='success').inc()
                return result
            except Exception as e:
                NASA_API_REQUESTS.labels(endpoint=endpoint, status='error').inc()
                raise
            finally:
                duration = time.time() - start
                NASA_API_LATENCY.labels(endpoint=endpoint).observe(duration)
        return wrapper
    return decorator


# ==========================================
# FastAPI Instrumentator Setup
# ==========================================
def setup_metrics(app):
    """Setup Prometheus metrics for FastAPI"""
    
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/health", "/metrics"],
        env_var_name="ENABLE_METRICS",
        inprogress_name="http_requests_inprogress",
        inprogress_labels=True,
    )
    
    instrumentator.instrument(app).expose(app, endpoint="/metrics")
    
    return instrumentator
```

## 2. Structured Logging

```python
# api/monitoring/logging_config.py
"""
Structured JSON Logging Configuration

Outputs JSON formatted logs for:
- Elasticsearch/Loki ingestion
- Better searchability
- Structured data analysis
"""

import logging
import sys
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pydantic import BaseModel
import traceback


class LogContext(BaseModel):
    """Context for structured logging"""
    request_id: Optional[str] = None
    user_id: Optional[int] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        
        # Base log entry
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        if self.include_extra:
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in {
                    'name', 'msg', 'args', 'created', 'filename', 'funcName',
                    'levelname', 'levelno', 'lineno', 'module', 'msecs',
                    'pathname', 'process', 'processName', 'relativeCreated',
                    'stack_info', 'exc_info', 'exc_text', 'message', 'taskName'
                }:
                    try:
                        json.dumps(value)  # Check if serializable
                        extra_fields[key] = value
                    except (TypeError, ValueError):
                        extra_fields[key] = str(value)
            
            if extra_fields:
                log_entry["extra"] = extra_fields
        
        return json.dumps(log_entry)


class ContextFilter(logging.Filter):
    """Add context to log records"""
    
    def __init__(self):
        super().__init__()
        self._context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs):
        """Set context values"""
        self._context.update(kwargs)
    
    def clear_context(self):
        """Clear context"""
        self._context.clear()
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to record"""
        for key, value in self._context.items():
            setattr(record, key, value)
        return True


# Global context filter
context_filter = ContextFilter()


def setup_logging(
    level: str = "INFO",
    json_output: bool = True,
    include_context: bool = True
) -> None:
    """Setup structured logging"""
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    
    if json_output:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
    
    # Add context filter if needed
    if include_context:
        console_handler.addFilter(context_filter)
    
    root_logger.addHandler(console_handler)
    
    # Suppress noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


# Context manager for request logging
class RequestLoggingContext:
    """Context manager for request logging"""
    
    def __init__(
        self,
        request_id: str,
        user_id: Optional[int] = None,
        ip_address: Optional[str] = None,
        endpoint: Optional[str] = None,
        method: Optional[str] = None
    ):
        self.request_id = request_id
        self.user_id = user_id
        self.ip_address = ip_address
        self.endpoint = endpoint
        self.method = method
    
    def __enter__(self):
        context_filter.set_context(
            request_id=self.request_id,
            user_id=self.user_id,
            ip_address=self.ip_address,
            endpoint=self.endpoint,
            method=self.method
        )
        return self
    
    def __exit__(self, *args):
        context_filter.clear_context()
```

## 3. Health Check System

```python
# api/routes/health.py
"""
Health Check System

Provides multiple health check endpoints:
- /health - Basic liveness
- /health/ready - Readiness (all dependencies)
- /health/detailed - Detailed status with metrics
"""

from fastapi import APIRouter, Response, status
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import asyncpg
import redis.asyncio as redis
import aiohttp

router = APIRouter(tags=["Health"])


class ComponentHealth(BaseModel):
    """Health status of a component"""
    name: str
    status: str  # healthy, unhealthy, degraded
    latency_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Overall health response"""
    status: str  # healthy, unhealthy, degraded
    timestamp: datetime
    version: str
    uptime_seconds: float
    components: List[ComponentHealth]


# Track startup time
_startup_time = datetime.now()


async def check_database(pool: asyncpg.Pool) -> ComponentHealth:
    """Check database health"""
    start = datetime.now()
    
    try:
        async with pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            
        latency = (datetime.now() - start).total_seconds() * 1000
        
        return ComponentHealth(
            name="database",
            status="healthy" if latency < 100 else "degraded",
            latency_ms=round(latency, 2),
            details={"pool_size": pool.get_size()}
        )
    except Exception as e:
        return ComponentHealth(
            name="database",
            status="unhealthy",
            error=str(e)
        )


async def check_redis(redis_client: redis.Redis) -> ComponentHealth:
    """Check Redis health"""
    start = datetime.now()
    
    try:
        await redis_client.ping()
        latency = (datetime.now() - start).total_seconds() * 1000
        
        info = await redis_client.info()
        
        return ComponentHealth(
            name="redis",
            status="healthy" if latency < 50 else "degraded",
            latency_ms=round(latency, 2),
            details={
                "connected_clients": info.get("connected_clients"),
                "used_memory": info.get("used_memory_human"),
            }
        )
    except Exception as e:
        return ComponentHealth(
            name="redis",
            status="unhealthy",
            error=str(e)
        )


async def check_nasa_api() -> ComponentHealth:
    """Check NASA API connectivity"""
    start = datetime.now()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.nasa.gov/planetary/apod?api_key=DEMO_KEY",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                latency = (datetime.now() - start).total_seconds() * 1000
                
                return ComponentHealth(
                    name="nasa_api",
                    status="healthy" if resp.status == 200 else "degraded",
                    latency_ms=round(latency, 2),
                    details={"status_code": resp.status}
                )
    except Exception as e:
        return ComponentHealth(
            name="nasa_api",
            status="degraded",  # Non-critical dependency
            error=str(e)
        )


@router.get("/health")
async def health_basic():
    """Basic health check (liveness probe)"""
    return {"status": "ok"}


@router.get("/health/ready")
async def health_ready(response: Response):
    """Readiness check (all critical dependencies)"""
    
    from api.dependencies import get_db_pool, get_redis
    
    db_pool = await get_db_pool()
    redis_client = await get_redis()
    
    db_health = await check_database(db_pool)
    redis_health = await check_redis(redis_client)
    
    is_ready = (
        db_health.status == "healthy" and
        redis_health.status == "healthy"
    )
    
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    
    return {
        "ready": is_ready,
        "checks": {
            "database": db_health.status,
            "redis": redis_health.status
        }
    }


@router.get("/health/detailed", response_model=HealthResponse)
async def health_detailed(response: Response):
    """Detailed health check with all components"""
    
    from api.dependencies import get_db_pool, get_redis
    
    # Run all checks in parallel
    db_pool = await get_db_pool()
    redis_client = await get_redis()
    
    results = await asyncio.gather(
        check_database(db_pool),
        check_redis(redis_client),
        check_nasa_api(),
        return_exceptions=True
    )
    
    components = [
        r if isinstance(r, ComponentHealth) else ComponentHealth(
            name="unknown",
            status="unhealthy",
            error=str(r)
        )
        for r in results
    ]
    
    # Determine overall status
    statuses = [c.status for c in components]
    
    if "unhealthy" in statuses:
        overall_status = "unhealthy"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif "degraded" in statuses:
        overall_status = "degraded"
    else:
        overall_status = "healthy"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now(),
        version="2.0.0",
        uptime_seconds=(datetime.now() - _startup_time).total_seconds(),
        components=components
    )
```

## 4. Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    monitor: 'nanoprobe-monitor'

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

rule_files:
  - /etc/prometheus/alerts/*.yml

scrape_configs:
  # API metrics
  - job_name: 'api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: /metrics

  # Node exporter (system metrics)
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

  # PostgreSQL exporter
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  # Redis exporter
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  # Nginx exporter
  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx-exporter:9113']

  # Prometheus self-monitoring
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

## 5. Alert Rules

```yaml
# monitoring/alerts/rules.yml
groups:
  - name: api_alerts
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m])) 
          / sum(rate(http_requests_total[5m])) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"

      # High latency
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le)) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High request latency"
          description: "P95 latency is {{ $value }}s"

      # API down
      - alert: APIDown
        expr: up{job="api"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "API service is down"
          description: "API has been unreachable for more than 1 minute"

  - name: database_alerts
    rules:
      # Database connections
      - alert: HighDatabaseConnections
        expr: db_connections_active > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High database connection count"
          description: "{{ $value }} active connections"

      # Slow queries
      - alert: SlowDatabaseQueries
        expr: |
          histogram_quantile(0.95, sum(rate(db_query_duration_seconds_bucket[5m])) by (le)) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow database queries detected"

  - name: redis_alerts
    rules:
      # Redis memory
      - alert: RedisMemoryHigh
        expr: redis_memory_used_bytes / redis_memory_max_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Redis memory usage high"
          description: "Memory usage is {{ $value | humanizePercentage }}"

  - name: security_alerts
    rules:
      # Authentication failures
      - alert: HighAuthFailures
        expr: |
          sum(rate(auth_attempts_total{status="failure"}[5m])) > 10
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High authentication failure rate"
          description: "{{ $value }} failures per second"

      # Rate limit hits
      - alert: HighRateLimitHits
        expr: |
          sum(rate(rate_limit_hits_total[5m])) > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High rate limit hit rate"
          description: "Possible abuse or DDoS attack"
```

## 6. Grafana Dashboard

```json
// monitoring/grafana/dashboards/api-overview.json
{
  "dashboard": {
    "title": "Nanoprobe API Overview",
    "tags": ["api", "nanoprobe"],
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m])) by (endpoint)",
            "legendFormat": "{{endpoint}}"
          }
        ],
        "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8}
      },
      {
        "title": "Latency (P95)",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))",
            "legendFormat": "P95"
          }
        ],
        "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8}
      },
      {
        "title": "Error Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{status=~\"5..\"}[5m])) / sum(rate(http_requests_total[5m]))",
            "legendFormat": "Error Rate"
          }
        ],
        "gridPos": {"x": 0, "y": 8, "w": 6, "h": 4}
      },
      {
        "title": "Active Users",
        "type": "stat",
        "targets": [
          {
            "expr": "count(http_requests_active)",
            "legendFormat": "Active"
          }
        ],
        "gridPos": {"x": 6, "y": 8, "w": 6, "h": 4}
      },
      {
        "title": "Database Connections",
        "type": "graph",
        "targets": [
          {
            "expr": "db_connections_active",
            "legendFormat": "Connections"
          }
        ],
        "gridPos": {"x": 0, "y": 12, "w": 12, "h": 8}
      },
      {
        "title": "Cache Hit Rate",
        "type": "gauge",
        "targets": [
          {
            "expr": "cache_hit_rate",
            "legendFormat": "Hit Rate"
          }
        ],
        "gridPos": {"x": 12, "y": 12, "w": 12, "h": 8}
      }
    ]
  }
}
```

## 7. Alertmanager Configuration

```yaml
# monitoring/alertmanager.yml
global:
  resolve_timeout: 5m
  slack_api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'

route:
  group_by: ['alertname', 'severity']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  receiver: 'slack-notifications'
  routes:
    - match:
        severity: critical
      receiver: 'slack-critical'
      continue: true
    - match:
        severity: warning
      receiver: 'slack-warnings'

receivers:
  - name: 'slack-notifications'
    slack_configs:
      - channel: '#monitoring'
        send_resolved: true
        title: '{{ .Status | toUpper }}: {{ .CommonAnnotations.summary }}'
        text: '{{ .CommonAnnotations.description }}'

  - name: 'slack-critical'
    slack_configs:
      - channel: '#alerts-critical'
        send_resolved: true
        title: '🚨 CRITICAL: {{ .CommonAnnotations.summary }}'
        text: '{{ .CommonAnnotations.description }}'
        color: '{{ if eq .Status "firing" }}danger{{ else }}good{{ end }}'

  - name: 'slack-warnings'
    slack_configs:
      - channel: '#alerts-warnings'
        send_resolved: true
        title: '⚠️ WARNING: {{ .CommonAnnotations.summary }}'
        text: '{{ .CommonAnnotations.description }}'
        color: 'warning'

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname']
```

## 8. Log Aggregation (Loki)

```yaml
# monitoring/loki/loki-config.yml
auth_enabled: false

server:
  http_listen_port: 3100
  grpc_listen_port: 9096

common:
  instance_addr: 127.0.0.1
  path_prefix: /loki
  storage:
    filesystem:
      chunks_directory: /loki/chunks
      rules_directory: /loki/rules
  replication_factor: 1
  ring:
    kvstore:
      store: inmemory

schema_config:
  configs:
    - from: 2020-10-24
      store: tsdb
      object_store: filesystem
      schema: v13
      index:
        prefix: index_
        period: 24h

limits_config:
  reject_old_samples: true
  reject_old_samples_max_age: 168h

ruler:
  alertmanager_url: http://alertmanager:9093
```

## Quick Start Commands

```bash
# Start monitoring stack
docker-compose -f docker-compose.prod.yml up -d prometheus grafana

# View metrics
curl http://localhost:8000/metrics

# Health check
curl http://localhost:8000/health/detailed

# Grafana
open http://localhost:3001
# Default: admin/admin

# Prometheus
open http://localhost:9090

# Alerts
curl http://localhost:9093/api/v1/alerts
```
