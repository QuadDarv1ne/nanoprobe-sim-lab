"use client";

import { DashboardLayout } from "@/components/dashboard-layout";
import {
  Radio, Download, Trash2, Eye, Play, Square, Signal, Satellite,
  Clock, HardDrive, Settings, Activity, Waves, Volume2, Antenna
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useEffect, useState, useRef, useCallback } from "react";
import { API_BASE } from "@/lib/config";
import { format } from "date-fns";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { toast } from "@/components/ui/toaster";
import { apiClient } from "@/lib/api-client";

const SSTV_ISS_FREQUENCY_MHZ = 145.800;

interface SSTVRecording {
  filename: string;
  path: string;
  size_bytes: number;
  created_at: string;
  frequency: string;
}

interface ISSPass {
  aos: string;
  los: string;
  max_elevation: number;
  frequency_mhz: number;
  satellite: string;
  mode: string;
}

interface DeviceStatus {
  connected: boolean;
  device_index: number;
  device_name: string;
  serial: string;
  tuner: string;
}

interface SDRConfig {
  frequency: number;
  gain: number;
  sample_rate: number;
  bias_tee: boolean;
  agc: boolean;
  ppm: number;
  mode: string;
}

export default function SSTVPage() {
  const [recordings, setRecordings] = useState<SSTVRecording[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [deviceStatus, setDeviceStatus] = useState<DeviceStatus | null>(null);
  const [issPasses, setIssPasses] = useState<ISSPass[]>([]);
  const [sdrConfig, setSdrConfig] = useState<SDRConfig>({
    frequency: 145.800,
    gain: 49.6,
    sample_rate: 2400000,
    bias_tee: false,
    agc: false,
    ppm: 0,
    mode: 'auto',
  });
  const [showSettings, setShowSettings] = useState(false);

  // Spectrum visualization
  const [spectrumData, setSpectrumData] = useState<{ freqs: number[]; power: number[] } | null>(null);
  const [signalStrength, setSignalStrength] = useState<number>(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // ===== Device Check =====
  const checkDevice = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/sstv/device/check`);
      if (res.ok) {
        const data = await res.json();
        setDeviceStatus({
          connected: data.connected ?? data.devices?.length > 0,
          device_index: data.device_index ?? 0,
          device_name: data.device_name ?? 'RTL-SDR V4',
          serial: data.serial ?? '',
          tuner: data.tuner ?? 'R828D',
        });
      }
    } catch {
      setDeviceStatus(null);
    }
  }, []);

  // ===== Fetch Recordings =====
  const fetchRecordings = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/sstv/recordings`);
      if (res.ok) {
        const data = await res.json();
        const items = Array.isArray(data) ? data : (data.recordings ?? data.items ?? []);
        setRecordings(items);
      }
    } catch (e) {
      console.error('Failed to fetch recordings:', e);
    }
  }, []);

  // ===== Fetch ISS Passes =====
  const fetchISSPasses = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/sstv/iss/schedule`);
      if (res.ok) {
        const data = await res.json();
        setIssPasses(Array.isArray(data) ? data.slice(0, 5) : (data.passes ?? []));
      }
    } catch { /* pass */ }
  }, []);

  // ===== Recording Controls =====
  const startRecording = async () => {
    try {
      const params = new URLSearchParams({
        frequency: String(sdrConfig.frequency),
        duration: '120',
        gain: String(sdrConfig.gain),
      });
      const res = await fetch(`${API_BASE}/api/v1/sstv/record/start?${params}`, { method: 'POST' });
      if (res.ok) {
        setIsRecording(true);
        toast.success('Запись начата', { description: `Частота: ${sdrConfig.frequency} МГц` });
      } else {
        const err = await res.json().catch(() => null);
        toast.error('Ошибка записи', { description: err?.detail || `HTTP ${res.status}` });
      }
    } catch (e) {
      toast.error('Ошибка', { description: 'Не удалось начать запись' });
    }
  };

  const stopRecording = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/sstv/record/stop`, { method: 'POST' });
      if (res.ok) {
        setIsRecording(false);
        toast.success('Запись остановлена');
        fetchRecordings();
      }
    } catch {
      toast.error('Ошибка', { description: 'Не удалось остановить запись' });
    }
  };

  const downloadRecording = async (filename: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/sstv/recordings/${filename}`);
      if (res.ok) {
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        toast.success('Загрузка начата');
      }
    } catch {
      toast.error('Ошибка загрузки');
    }
  };

  const decodeRecording = async (filename: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/sstv/sstv/decode-recording/${filename}`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        toast.success('Декодирование завершено', {
          description: data.message || `Режим: ${data.mode || 'SSTV'}`,
        });
        fetchRecordings();
      } else {
        toast.error('Ошибка декодирования', { description: `HTTP ${res.status}` });
      }
    } catch {
      toast.error('Ошибка', { description: 'Не удалось декодировать запись' });
    }
  };

  const deleteRecording = async (filename: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/sstv/recordings/${filename}`, { method: 'DELETE' });
      if (res.ok || res.status === 204) {
        toast.success('Запись удалена');
        fetchRecordings();
      }
    } catch {
      toast.error('Ошибка удаления');
    }
  };

  // ===== Spectrum Visualization =====
  const drawSpectrum = useCallback((freqs: number[], power: number[]) => {
    const canvas = canvasRef.current;
    if (!canvas || freqs.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    const minP = Math.min(...power);
    const maxP = Math.max(...power);
    const range = maxP - minP || 1;

    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
      const y = (h / 5) * i;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    // Spectrum bars
    const barWidth = w / freqs.length;
    const gradient = ctx.createLinearGradient(0, h, 0, 0);
    gradient.addColorStop(0, '#22c55e');
    gradient.addColorStop(0.5, '#eab308');
    gradient.addColorStop(1, '#ef4444');

    for (let i = 0; i < freqs.length; i++) {
      const normalizedPower = (power[i] - minP) / range;
      const barH = normalizedPower * h * 0.9;
      ctx.fillStyle = gradient;
      ctx.fillRect(i * barWidth, h - barH, barWidth - 1, barH);
    }

    // Center frequency marker
    const centerIdx = Math.floor(freqs.length / 2);
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(centerIdx * barWidth, 0);
    ctx.lineTo(centerIdx * barWidth, h);
    ctx.stroke();
  }, []);

  // ===== WebSocket for real-time data =====
  const connectWebSocket = useCallback(() => {
    if (wsRef.current) return;

    const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const wsUrl = `${wsProtocol}://${window.location.host.replace(':3000', ':8000')}/api/v1/sstv/ws/stream`;

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'spectrum' && msg.frequencies && msg.power_db) {
            setSpectrumData({ freqs: msg.frequencies, power: msg.power_db });
            drawSpectrum(msg.frequencies, msg.power_db);
          }
          if (msg.type === 'signal_strength') {
            setSignalStrength(msg.strength ?? 0);
          }
        } catch { /* ignore */ }
      };

      ws.onclose = () => {
        wsRef.current = null;
        setTimeout(connectWebSocket, 5000); // Reconnect
      };

      ws.onerror = () => { ws.close(); };
    } catch { /* ignore */ }
  }, [drawSpectrum]);

  // ===== Init =====
  useEffect(() => {
    checkDevice();
    fetchRecordings();
    fetchISSPasses();
  }, [checkDevice, fetchRecordings, fetchISSPasses]);

  useEffect(() => {
    connectWebSocket();
    return () => { wsRef.current?.close(); wsRef.current = null; };
  }, [connectWebSocket]);

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">SSTV Ground Station</h1>
            <p className="text-muted-foreground mt-1">Приём SSTV через RTL-SDR V4</p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="icon" onClick={() => setShowSettings(!showSettings)}>
              <Settings className="h-4 w-4" />
            </Button>
            <Button onClick={isRecording ? stopRecording : startRecording}
              variant={isRecording ? "destructive" : "default"}>
              {isRecording ? <Square className="h-4 w-4 mr-2" /> : <Radio className="h-4 w-4 mr-2" />}
              {isRecording ? 'Остановить' : 'Начать запись'}
            </Button>
          </div>
        </div>

        {/* SDR Settings Panel */}
        {showSettings && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Settings className="h-5 w-5" /> Настройки SDR</CardTitle>
              <CardDescription>Параметры приёмника RTL-SDR V4</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <Label>Частота (МГц)</Label>
                  <Input type="number" step="0.001" value={sdrConfig.frequency}
                    onChange={e => setSdrConfig(p => ({ ...p, frequency: parseFloat(e.target.value) || 145.8 }))} />
                </div>
                <div>
                  <Label>Усиление (dB): {sdrConfig.gain}</Label>
                  <Slider min={0} max={49.6} step={0.2} value={[sdrConfig.gain]}
                    onValueChange={v => setSdrConfig(p => ({ ...p, gain: v[0] }))} />
                </div>
                <div>
                  <Label>Sample Rate (SPS)</Label>
                  <Input type="number" value={sdrConfig.sample_rate}
                    onChange={e => setSdrConfig(p => ({ ...p, sample_rate: parseInt(e.target.value) || 2400000 }))} />
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="flex items-center justify-between">
                  <Label>Bias-Tee</Label>
                  <Switch checked={sdrConfig.bias_tee}
                    onCheckedChange={v => setSdrConfig(p => ({ ...p, bias_tee: v }))} />
                </div>
                <div className="flex items-center justify-between">
                  <Label>AGC</Label>
                  <Switch checked={sdrConfig.agc}
                    onCheckedChange={v => setSdrConfig(p => ({ ...p, agc: v }))} />
                </div>
                <div>
                  <Label>PPM коррекция</Label>
                  <Input type="number" value={sdrConfig.ppm}
                    onChange={e => setSdrConfig(p => ({ ...p, ppm: parseInt(e.target.value) || 0 }))} />
                </div>
                <div>
                  <Label>Режим</Label>
                  <select className="w-full rounded-md border bg-background px-3 py-2 text-sm"
                    value={sdrConfig.mode}
                    onChange={e => setSdrConfig(p => ({ ...p, mode: e.target.value }))}>
                    <option value="auto">Auto</option>
                    <option value="manual">Manual</option>
                  </select>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Status Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-3">
                <Antenna className={`h-5 w-5 ${deviceStatus?.connected ? 'text-green-500' : 'text-red-500'}`} />
                <div>
                  <p className="text-sm text-muted-foreground">Устройство</p>
                  <p className="font-medium text-sm">
                    {deviceStatus?.connected ? deviceStatus.device_name : 'Не подключено'}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-3">
                <Waves className="h-5 w-5 text-blue-500" />
                <div>
                  <p className="text-sm text-muted-foreground">Частота</p>
                  <p className="font-medium text-sm">{sdrConfig.frequency} МГц</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-3">
                <Signal className="h-5 w-5 text-yellow-500" />
                <div>
                  <p className="text-sm text-muted-foreground">Сила сигнала</p>
                  <p className="font-medium text-sm">{signalStrength > 0 ? `${signalStrength.toFixed(0)}%` : '—'}</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-3">
                <HardDrive className="h-5 w-5 text-purple-500" />
                <div>
                  <p className="text-sm text-muted-foreground">Записей</p>
                  <p className="font-medium text-sm">{recordings.length}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Spectrum Visualization */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><Activity className="h-5 w-5" /> Спектр сигнала</CardTitle>
            <CardDescription>Real-time спектр через WebSocket</CardDescription>
          </CardHeader>
          <CardContent>
            <canvas ref={canvasRef} width={800} height={200}
              className="w-full rounded-lg border border-border bg-slate-900" />
            {!spectrumData && (
              <p className="text-center text-muted-foreground mt-2 text-sm">
                Подключение к WebSocket... Убедитесь что RTL-SDR подключен
              </p>
            )}
          </CardContent>
        </Card>

        {/* ISS Passes */}
        {issPasses.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Satellite className="h-5 w-5" /> Пролёты МКС</CardTitle>
              <CardDescription>Ближайшие пролёты для приёма SSTV</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {issPasses.map((pass, i) => (
                  <div key={i} className="flex items-center justify-between p-3 rounded-lg bg-secondary/30">
                    <div className="flex items-center gap-3">
                      <Clock className="h-4 w-4 text-muted-foreground" />
                      <div>
                        <p className="font-medium text-sm">{pass.satellite || 'МКС'}</p>
                        <p className="text-xs text-muted-foreground">
                          {format(new Date(pass.aos), 'HH:mm')} — {format(new Date(pass.los), 'HH:mm')}
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-medium">{pass.max_elevation.toFixed(0)}°</p>
                      <p className="text-xs text-muted-foreground">{pass.frequency_mhz} МГц</p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Recordings List */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><Download className="h-5 w-5" /> Записи</CardTitle>
            <CardDescription>Записанные SSTV сессии</CardDescription>
          </CardHeader>
          <CardContent>
            {recordings.length === 0 ? (
              <div className="text-center py-8">
                <Radio className="h-12 w-12 mx-auto mb-4 opacity-20" />
                <p className="text-muted-foreground">Нет записей</p>
                <p className="text-sm text-muted-foreground mt-1">Нажмите &quot;Начать запись&quot; для приёма SSTV</p>
              </div>
            ) : (
              <div className="space-y-2">
                {recordings.map((rec, i) => (
                  <div key={i} className="flex items-center justify-between p-3 rounded-lg bg-secondary/30 border border-border">
                    <div className="flex items-center gap-3">
                      <Volume2 className="h-4 w-4 text-blue-500" />
                      <div>
                        <p className="font-medium text-sm">{rec.filename}</p>
                        <p className="text-xs text-muted-foreground">
                          {formatSize(rec.size_bytes)} • {rec.frequency || '145.800'} МГц
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-1">
                      <Button variant="outline" size="icon" onClick={() => decodeRecording(rec.filename)} title="Декодировать">
                        <Eye className="h-4 w-4" />
                      </Button>
                      <Button variant="outline" size="icon" onClick={() => downloadRecording(rec.filename)} title="Скачать">
                        <Download className="h-4 w-4" />
                      </Button>
                      <Button variant="outline" size="icon" onClick={() => deleteRecording(rec.filename)} title="Удалить">
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
}
