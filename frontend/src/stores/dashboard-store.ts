import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

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

  fetchDashboardData: () => Promise<void>;
  subscribeToRealtime: () => void;
  unsubscribeFromRealtime: () => void;
  checkAlerts: () => Promise<void>;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

let ws: WebSocket | null = null;
let reconnectTimeout: NodeJS.Timeout | null = null;

export const useDashboardStore = create<DashboardState>()(
  subscribeWithSelector((set, get) => ({
    stats: null,
    systemHealth: null,
    alerts: [],
    isLoading: false,
    error: null,
    wsConnected: false,

    fetchDashboardData: async () => {
      set({ isLoading: true, error: null });
      try {
        const [statsRes, healthRes, alertsRes] = await Promise.all([
          fetch(`${API_BASE}/api/v1/dashboard/stats/detailed`),
          fetch(`${API_BASE}/health/detailed`),
          fetch(`${API_BASE}/api/v1/dashboard/alerts/check`),
        ]);

        const stats = await statsRes.json();
        const health = await healthRes.json();
        const alertsData = await alertsRes.json();

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
        });
      } catch (error) {
        set({
          isLoading: false,
          error: error instanceof Error ? error.message : 'Failed to fetch data',
        });
      }
    },

    subscribeToRealtime: () => {
      if (ws || typeof WebSocket === 'undefined') return;

      try {
        const wsUrl = `ws://${API_BASE.replace('http://', '').replace('/api', '')}/api/v1/dashboard/ws/metrics`;
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
          set({ wsConnected: true, error: null });
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
          set({ wsConnected: false });
          ws = null;
          // Reconnect after 5 seconds
          reconnectTimeout = setTimeout(() => {
            get().subscribeToRealtime();
          }, 5000);
        };

        ws.onerror = () => {
          set({ error: 'WebSocket connection failed' });
          ws?.close();
        };
      } catch (error) {
        set({ error: 'WebSocket not available', wsConnected: false });
      }
    },

    unsubscribeFromRealtime: () => {
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
        reconnectTimeout = null;
      }
      if (ws) {
        ws.close();
        ws = null;
      }
    },

    checkAlerts: async () => {
      try {
        const res = await fetch(`${API_BASE}/api/v1/dashboard/alerts/check`);
        const data = await res.json();
        set({ alerts: data.alerts || [] });
      } catch (error) {
        console.error('Failed to check alerts:', error);
      }
    },
  }))
);
