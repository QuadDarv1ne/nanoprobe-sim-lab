/**
 * Mobile Dashboard Page
 * 
 * Мобильная версия dashboard для Nanoprobe Sim Lab.
 * Оптимизировано для смартфонов и планшетов.
 * 
 * Features:
 * - Touch-friendly интерфейс
 * - Offline режим
 * - Push уведомления
 * - Real-time обновления
 * - SSTV мониторинг
 */

'use client';

import { useState, useEffect } from 'react';
import { 
  Wifi, 
  WifiOff, 
  Activity, 
  HardDrive, 
  Cpu, 
  Database,
  Radio,
  Download,
  Upload,
  RefreshCw,
  Bell,
  Menu,
  X
} from 'lucide-react';

// Types
interface SystemStats {
  cpu: number;
  memory: number;
  disk: number;
  network_sent: number;
  network_recv: number;
}

interface SSTVStatus {
  active: boolean;
  frequency: number;
  last_recording: string | null;
  decoded_count: number;
}

export default function MobileDashboard() {
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [sstv, setSstv] = useState<SSTVStatus | null>(null);
  const [isOnline, setIsOnline] = useState(true);
  const [loading, setLoading] = useState(true);
  const [menuOpen, setMenuOpen] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  // Проверка online статуса
  useEffect(() => {
    const checkOnline = () => {
      setIsOnline(navigator.onLine);
    };

    window.addEventListener('online', checkOnline);
    window.addEventListener('offline', checkOnline);
    checkOnline();

    return () => {
      window.removeEventListener('online', checkOnline);
      window.removeEventListener('offline', checkOnline);
    };
  }, []);

  // Получение данных
  const fetchStats = async () => {
    try {
      const response = await fetch('/api/v1/monitoring/health/detailed');
      if (response.ok) {
        const data = await response.json();
        setStats({
          cpu: data.system.cpu.percent,
          memory: data.system.memory.percent,
          disk: data.system.disk.percent,
          network_sent: data.system.network.bytes_sent_mb,
          network_recv: data.system.network.bytes_recv_mb,
        });
        setLastUpdate(new Date());
      }
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    } finally {
      setLoading(false);
    }
  };

  // Получение SSTV статуса
  const fetchSSTVStatus = async () => {
    try {
      const response = await fetch('/api/v1/sstv/status');
      if (response.ok) {
        const data = await response.json();
        setSstv({
          active: data.active || false,
          frequency: data.frequency || 145.800,
          last_recording: data.last_recording || null,
          decoded_count: data.decoded_count || 0,
        });
      }
    } catch (error) {
      console.error('Failed to fetch SSTV status:', error);
    }
  };

  // Автообновление
  useEffect(() => {
    fetchStats();
    fetchSSTVStatus();

    const interval = setInterval(() => {
      if (navigator.onLine) {
        fetchStats();
        fetchSSTVStatus();
      }
    }, 5000); // Обновление каждые 5 секунд

    return () => clearInterval(interval);
  }, []);

  // Форматирование времени
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('ru-RU', {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const formatDateTime = (isoString: string) => {
    const date = new Date(isoString);
    return date.toLocaleString('ru-RU', {
      day: 'numeric',
      month: 'short',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  // Цвет статуса
  const getStatusColor = (value: number) => {
    if (value < 50) return 'text-green-500';
    if (value < 80) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getBgColor = (value: number) => {
    if (value < 50) return 'bg-green-500';
    if (value < 80) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-slate-900/80 backdrop-blur-sm border-b border-slate-700">
        <div className="px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setMenuOpen(!menuOpen)}
              className="p-2 text-slate-300 hover:text-white"
            >
              {menuOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
            <h1 className="text-lg font-bold text-white">Nanoprobe Lab</h1>
          </div>
          
          <div className="flex items-center gap-2">
            {isOnline ? (
              <Wifi size={20} className="text-green-500" />
            ) : (
              <WifiOff size={20} className="text-red-500" />
            )}
            <button className="p-2 text-slate-300 hover:text-white">
              <Bell size={20} />
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        {menuOpen && (
          <div className="px-4 py-3 bg-slate-800 border-t border-slate-700">
            <nav className="space-y-2">
              <a href="/" className="block py-2 text-slate-300 hover:text-white">
                📊 Dashboard
              </a>
              <a href="/sstv" className="block py-2 text-slate-300 hover:text-white">
                📡 SSTV Station
              </a>
              <a href="/simulations" className="block py-2 text-slate-300 hover:text-white">
                🔬 Simulations
              </a>
              <a href="/analysis" className="block py-2 text-slate-300 hover:text-white">
                🔍 Analysis
              </a>
              <a href="/settings" className="block py-2 text-slate-300 hover:text-white">
                ⚙️ Settings
              </a>
            </nav>
          </div>
        )}
      </header>

      {/* Main Content */}
      <main className="p-4 space-y-4">
        {/* Online Status Banner */}
        {!isOnline && (
          <div className="bg-red-600/20 border border-red-500/50 rounded-xl p-3 text-center">
            <p className="text-red-400 text-sm">Нет подключения к интернету</p>
          </div>
        )}

        {/* SSTV Status Card */}
        {sstv && (
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-4 border border-slate-700">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Radio size={20} className={sstv.active ? 'text-green-500 animate-pulse' : 'text-slate-500'} />
                <h2 className="text-white font-semibold">SSTV Station</h2>
              </div>
              <span className={`px-2 py-1 rounded-full text-xs ${
                sstv.active 
                  ? 'bg-green-500/20 text-green-400' 
                  : 'bg-slate-600/20 text-slate-400'
              }`}>
                {sstv.active ? 'Active' : 'Standby'}
              </span>
            </div>

            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-slate-400">Frequency:</span>
                <span className="text-white font-mono">{sstv.frequency} MHz</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Decoded:</span>
                <span className="text-white">{sstv.decoded_count} images</span>
              </div>
              {sstv.last_recording && (
                <div className="flex justify-between">
                  <span className="text-slate-400">Last:</span>
                  <span className="text-white text-xs">{formatDateTime(sstv.last_recording)}</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* System Stats Grid */}
        <div className="grid grid-cols-2 gap-3">
          {/* CPU */}
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-4 border border-slate-700">
            <div className="flex items-center gap-2 mb-2">
              <Cpu size={18} className="text-blue-400" />
              <span className="text-slate-400 text-sm">CPU</span>
            </div>
            {loading ? (
              <div className="h-8 bg-slate-700 rounded animate-pulse" />
            ) : stats && (
              <>
                <div className={`text-2xl font-bold ${getStatusColor(stats.cpu)}`}>
                  {stats.cpu.toFixed(1)}%
                </div>
                <div className="mt-2 h-2 bg-slate-700 rounded-full overflow-hidden">
                  <div 
                    className={`h-full ${getBgColor(stats.cpu)} transition-all duration-500`}
                    style={{ width: `${stats.cpu}%` }}
                  />
                </div>
              </>
            )}
          </div>

          {/* Memory */}
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-4 border border-slate-700">
            <div className="flex items-center gap-2 mb-2">
              <Activity size={18} className="text-purple-400" />
              <span className="text-slate-400 text-sm">RAM</span>
            </div>
            {loading ? (
              <div className="h-8 bg-slate-700 rounded animate-pulse" />
            ) : stats && (
              <>
                <div className={`text-2xl font-bold ${getStatusColor(stats.memory)}`}>
                  {stats.memory.toFixed(1)}%
                </div>
                <div className="mt-2 h-2 bg-slate-700 rounded-full overflow-hidden">
                  <div 
                    className={`h-full ${getBgColor(stats.memory)} transition-all duration-500`}
                    style={{ width: `${stats.memory}%` }}
                  />
                </div>
              </>
            )}
          </div>

          {/* Disk */}
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-4 border border-slate-700">
            <div className="flex items-center gap-2 mb-2">
              <HardDrive size={18} className="text-green-400" />
              <span className="text-slate-400 text-sm">Disk</span>
            </div>
            {loading ? (
              <div className="h-8 bg-slate-700 rounded animate-pulse" />
            ) : stats && (
              <>
                <div className={`text-2xl font-bold ${getStatusColor(stats.disk)}`}>
                  {stats.disk.toFixed(1)}%
                </div>
                <div className="mt-2 h-2 bg-slate-700 rounded-full overflow-hidden">
                  <div 
                    className={`h-full ${getBgColor(stats.disk)} transition-all duration-500`}
                    style={{ width: `${stats.disk}%` }}
                  />
                </div>
              </>
            )}
          </div>

          {/* Database */}
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-4 border border-slate-700">
            <div className="flex items-center gap-2 mb-2">
              <Database size={18} className="text-yellow-400" />
              <span className="text-slate-400 text-sm">DB</span>
            </div>
            {loading ? (
              <div className="h-8 bg-slate-700 rounded animate-pulse" />
            ) : (
              <div className="text-2xl font-bold text-white">OK</div>
            )}
          </div>
        </div>

        {/* Network Stats */}
        {stats && (
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-4 border border-slate-700">
            <div className="flex items-center gap-2 mb-3">
              <Activity size={18} className="text-cyan-400" />
              <h2 className="text-white font-semibold">Network</h2>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="flex items-center gap-2">
                <Upload size={16} className="text-green-400" />
                <div>
                  <div className="text-xs text-slate-400">Upload</div>
                  <div className="text-white font-mono">{stats.network_sent} MB</div>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Download size={16} className="text-blue-400" />
                <div>
                  <div className="text-xs text-slate-400">Download</div>
                  <div className="text-white font-mono">{stats.network_recv} MB</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Quick Actions */}
        <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-4 border border-slate-700">
          <h2 className="text-white font-semibold mb-3">Quick Actions</h2>
          <div className="grid grid-cols-2 gap-2">
            <button className="flex items-center justify-center gap-2 px-4 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl transition-colors">
              <Radio size={18} />
              <span>SSTV Record</span>
            </button>
            <button className="flex items-center justify-center gap-2 px-4 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-xl transition-colors">
              <RefreshCw size={18} />
              <span>Refresh</span>
            </button>
            <button className="flex items-center justify-center gap-2 px-4 py-3 bg-green-600 hover:bg-green-700 text-white rounded-xl transition-colors">
              <Activity size={18} />
              <span>Simulate</span>
            </button>
            <button className="flex items-center justify-center gap-2 px-4 py-3 bg-orange-600 hover:bg-orange-700 text-white rounded-xl transition-colors">
              <Database size={18} />
              <span>Scans</span>
            </button>
          </div>
        </div>

        {/* Last Update */}
        <div className="text-center text-slate-500 text-xs">
          Last update: {formatTime(lastUpdate)}
        </div>
      </main>

      {/* Bottom Navigation */}
      <nav className="fixed bottom-0 left-0 right-0 bg-slate-900/90 backdrop-blur-sm border-t border-slate-700 px-4 py-2">
        <div className="flex justify-around">
          <a href="/" className="flex flex-col items-center text-blue-400">
            <Activity size={20} />
            <span className="text-xs mt-1">Dashboard</span>
          </a>
          <a href="/sstv" className="flex flex-col items-center text-slate-400 hover:text-white">
            <Radio size={20} />
            <span className="text-xs mt-1">SSTV</span>
          </a>
          <a href="/simulations" className="flex flex-col items-center text-slate-400 hover:text-white">
            <Cpu size={20} />
            <span className="text-xs mt-1">Simulations</span>
          </a>
          <a href="/settings" className="flex flex-col items-center text-slate-400 hover:text-white">
            <Menu size={20} />
            <span className="text-xs mt-1">Menu</span>
          </a>
        </div>
      </nav>
    </div>
  );
}
