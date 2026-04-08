import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import { API_BASE } from '@/lib/api-client';

export const WS_BASE = API_BASE.replace(/^http/, 'ws');

export interface DashboardStats {
  total_scans: number;
  total_simulations: number;
  total_analysis: number;
  total_comparisons: number;
  total_reports: number;
  total_items: number;
}

export interface SystemHealth {
  cpu_percent: number;
  memory_percent: number;
  disk_percent: number;
  status: 'healthy' | 'warning' | 'critical';
}

interface Alert {
  level: 'info' | 'warning' | 'critical';
  type: 'cpu' | 'memory' | 'disk';
  message: string;
  value: number;
  threshold: number;
}

interface DashboardState {
  stats: DashboardStats | null;
  systemHealth: SystemHealth | null;
  alerts: Alert[];
  isLoading: boolean;
  error: string | null;
  wsConnected: boolean;
  wsInstance: WebSocket | null;
  wsReconnectAttempts: number;
  isReconnecting: boolean;

  fetchDashboardData: () => Promise<void>;
  subscribeToRealtime: () => void;
  unsubscribeFromRealtime: () => void;
  checkAlerts: () => Promise<void>;
  _reconnectWS: () => void;
}

const MAX_RECONNECT_ATTEMPTS = 5;
const FETCH_TIMEOUT_MS = 10000;

async function fetchWithTimeout(url: string, timeoutMs: number = FETCH_TIMEOUT_MS): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { signal: controller.signal });
  } finally {
    clearTimeout(timeoutId);
  }
}

async function handleApiResponse(response: Response, context: string): Promise<any> {
  if (!response.ok) {
    let errorDetail = `${context}: HTTP ${response.status}`;
    try {
      const errorBody = await response.json();
      errorDetail += ` - ${errorBody.message || errorBody.detail || JSON.stringify(errorBody)}`;
    } catch {
      errorDetail += ` - ${response.statusText || 'Unknown error'}`;
    }
    throw new Error(errorDetail);
  }
  return response.json();
}

export const useDashboardStore = create<DashboardState>()(
  subscribeWithSelector((set, get) => ({
    stats: null,
    systemHealth: null,
    alerts: [],
    isLoading: false,
    error: null,
    wsConnected: false,
    wsInstance: null,
    wsReconnectAttempts: 0,
    isReconnecting: false,

    fetchDashboardData: async () => {
      set({ isLoading: true, error: null });
      try {
        const [statsRes, healthRes, alertsRes] = await Promise.all([
          fetchWithTimeout(`${API_BASE}/api/v1/dashboard/stats/detailed`),
          fetchWithTimeout(`${API_BASE}/health/detailed`),
          fetchWithTimeout(`${API_BASE}/api/v1/dashboard/alerts/check`),
        ]);

        const stats = await handleApiResponse(statsRes, 'Dashboard stats');
        const health = await handleApiResponse(healthRes, 'System health');
        const alertsData = await handleApiResponse(alertsRes, 'Alerts check');

        set({
          stats: stats.summary || stats,
          systemHealth: {
            cpu_percent: health.metrics?.cpu?.percent || 0,
            memory_percent: health.metrics?.memory?.percent || 0,
            disk_percent: health.metrics?.disk?.percent || 0,
            status: health.status as 'healthy' | 'warning' | 'critical',
          },
          alerts: alertsData.alerts || [],
          isLoading: false,
          error: null,
        });
      } catch (error) {
        set({
          isLoading: false,
          error: error instanceof Error ? error.message : 'Failed to fetch dashboard data',
        });
        console.error('Dashboard fetch error:', error);
      }
    },

    subscribeToRealtime: () => {
      const { wsInstance } = get();
      if (wsInstance || typeof WebSocket === 'undefined') return;

      try {
        const ws = new WebSocket(`${WS_BASE}/api/v1/dashboard/ws/metrics`);

        ws.onopen = () => {
          set({ wsConnected: true, error: null, wsReconnectAttempts: 0 });
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            set({
              systemHealth: {
                cpu_percent: data.cpu?.total || 0,
                memory_percent: data.memory?.percent || 0,
                disk_percent: data.disk?.percent || 0,
                status: 'healthy',
              },
            });
          } catch (e) {
            console.error('Failed to parse WebSocket message:', e);
          }
        };

        ws.onclose = () => {
          set({ wsConnected: false, wsInstance: null });
          get()._reconnectWS();
        };

        ws.onerror = () => {
          ws.close();
        };

        set({ wsInstance: ws });
      } catch (error) {
        set({
          error: error instanceof Error ? error.message : 'WebSocket not available',
          wsConnected: false,
          wsInstance: null,
        });
      }
    },

    _reconnectWS: () => {
      const { wsInstance, wsReconnectAttempts, isReconnecting } = get();
      if (isReconnecting || wsInstance) return;

      if (wsReconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
        set({ error: 'WebSocket reconnection failed after multiple attempts' });
        return;
      }

      const newAttempts = wsReconnectAttempts + 1;
      const delay = Math.min(1000 * Math.pow(2, newAttempts), 30000);

      set({ isReconnecting: true, wsReconnectAttempts: newAttempts });

      setTimeout(() => {
        set({ isReconnecting: false });
        if (!get().wsInstance) {
          get().subscribeToRealtime();
        }
      }, delay);
    },

    unsubscribeFromRealtime: () => {
      const { wsInstance } = get();
      if (wsInstance) {
        wsInstance.onclose = null;
        wsInstance.close();
      }
      set({ wsInstance: null, wsConnected: false, wsReconnectAttempts: 0 });
    },

    checkAlerts: async () => {
      try {
        const data = await handleApiResponse(
          await fetchWithTimeout(`${API_BASE}/api/v1/dashboard/alerts/check`),
          'Alerts check'
        );
        set({ alerts: data.alerts || [] });
      } catch (error) {
        console.error('Failed to check alerts:', error);
      }
    },
  }))
);
