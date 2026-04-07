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
  // WebSocket state moved into store to avoid module-level mutable state
  wsInstance: WebSocket | null;
  wsReconnectAttempts: number;

  fetchDashboardData: () => Promise<void>;
  subscribeToRealtime: () => void;
  unsubscribeFromRealtime: () => void;
  checkAlerts: () => Promise<void>;
  _reconnectWS: () => void;
}

export const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
export const WS_BASE = API_BASE.replace('http', 'ws');

const MAX_RECONNECT_ATTEMPTS = 5;
const FETCH_TIMEOUT_MS = 10000;

/**
 * Helper to create fetch with timeout using AbortController
 */
async function fetchWithTimeout(url: string, timeoutMs: number = FETCH_TIMEOUT_MS): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  
  try {
    const response = await fetch(url, { signal: controller.signal });
    return response;
  } finally {
    clearTimeout(timeoutId);
  }
}

/**
 * Helper to handle API errors with detailed messages
 */
async function handleApiResponse(response: Response, context: string): Promise<any> {
  if (!response.ok) {
    let errorDetail = `${context}: HTTP ${response.status}`;
    try {
      const errorBody = await response.json();
      errorDetail += ` - ${errorBody.message || errorBody.detail || JSON.stringify(errorBody)}`;
    } catch {
      // If error body can't be parsed, use status text
      errorDetail += ` - ${response.statusText || 'Unknown error'}`;
    }
    throw new Error(errorDetail);
  }
  return await response.json();
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
        const errorMessage = error instanceof Error ? error.message : 'Failed to fetch dashboard data';
        set({
          isLoading: false,
          error: errorMessage,
        });
        console.error('Dashboard fetch error:', error);
      }
    },

    subscribeToRealtime: () => {
      const { wsInstance, wsReconnectAttempts } = get();
      
      // Don't create duplicate connections
      if (wsInstance || typeof WebSocket === 'undefined') return;

      try {
        const wsUrl = `${WS_BASE}/api/v1/dashboard/ws/metrics`;
        const ws = new WebSocket(wsUrl);

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
          // Trigger reconnection with backoff
          get()._reconnectWS();
        };

        ws.onerror = () => {
          // Error will be handled by onclose
          if (ws) {
            ws.close();
          }
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
      const { wsInstance, wsReconnectAttempts } = get();
      
      // Don't reconnect if there's an active connection
      if (wsInstance) return;

      if (wsReconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
        set({ error: 'WebSocket reconnection failed after multiple attempts' });
        return;
      }

      const newAttempts = wsReconnectAttempts + 1;
      const delay = Math.min(1000 * Math.pow(2, newAttempts), 30000);
      
      set({ wsReconnectAttempts: newAttempts });
      
      setTimeout(() => {
        const currentState = get();
        if (!currentState.wsInstance) {
          get().subscribeToRealtime();
        }
      }, delay);
    },

    unsubscribeFromRealtime: () => {
      const { wsInstance } = get();
      
      if (wsInstance) {
        wsInstance.onclose = null; // Prevent reconnection
        wsInstance.close();
      }
      
      set({ 
        wsInstance: null, 
        wsConnected: false, 
        wsReconnectAttempts: 0 
      });
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
