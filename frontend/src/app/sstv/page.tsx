"use client";

import { DashboardLayout } from "@/components/dashboard-layout";
import { Radio, Download, Trash2, Eye, Play, Square, Signal, Satellite, Clock, HardDrive } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useEffect, useState, useRef } from "react";
import { API_BASE } from "@/lib/config";
import { format } from "date-fns";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { toast } from "@/components/ui/toaster";
import { apiClient } from "@/lib/api-client";

// SSTV ISS frequency - constant
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
  duration_minutes: number;
  time_until_aos: string;
}

interface ISSPosition {
  latitude: number;
  longitude: number;
  altitude_km: number;
  velocity_kmh: number;
  footprint_km: number;
  timestamp: string;
}

interface RecordingsResponse {
  recordings?: SSTVRecording[];
}

interface RecordingStatusResponse {
  recording?: boolean;
}

interface ISSResponse {
  status?: string;
  data?: ISSPass | ISSPosition;
}

export default function SSTVPage() {
  const [recordings, setRecordings] = useState<SSTVRecording[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [nextPass, setNextPass] = useState<ISSPass | null>(null);
  const [issPosition, setIssPosition] = useState<ISSPosition | null>(null);
  const [recordingDuration, setRecordingDuration] = useState(600);
  const [isFetching, setIsFetching] = useState(false);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    fetchData();
    
    // Start polling after initial fetch
    pollIntervalRef.current = setInterval(() => {
      fetchData();
    }, 10000);
    
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    };
  }, []);

  const fetchData = async () => {
    // Prevent overlapping requests
    if (isFetching) return;
    
    setIsFetching(true);
    try {
      const [recordingsData, statusData, passData, positionData] = await Promise.all([
        apiClient.get<RecordingsResponse>('/api/v1/sstv/recordings').catch(() => ({ recordings: [] })),
        apiClient.get<RecordingStatusResponse>('/api/v1/sstv/record/status').catch(() => ({ recording: false })),
        apiClient.get<ISSResponse>('/api/v1/sstv/iss/next-pass').catch(() => ({})),
        apiClient.get<ISSResponse>('/api/v1/sstv/iss/position').catch(() => ({})),
      ]);

      setRecordings(recordingsData.recordings || []);
      setIsRecording(statusData.recording || false);
      
      if (passData.status === "success" && passData.data) {
        setNextPass(passData.data as ISSPass);
      }
      
      if (positionData.status === "success" && positionData.data) {
        setIssPosition(positionData.data as ISSPosition);
      }
    } catch (error) {
      console.error('Failed to fetch SSTV data:', error);
    } finally {
      setIsFetching(false);
      setIsLoading(false);
    }
  };

  const toggleRecording = async () => {
    if (isRecording) {
      await stopRecording();
    } else {
      await startRecording();
    }
  };

  const startRecording = async () => {
    try {
      const data = await apiClient.post('/api/v1/sstv/record/start', {
        frequency: SSTV_ISS_FREQUENCY_MHZ,
        duration: recordingDuration
      });
      setIsRecording(true);
      toast.success('Запись началась', {
        description: data.message
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Не удалось подключиться к API';
      console.error('Failed to start recording:', error);
      toast.error('Ошибка запуска записи', {
        description: errorMessage
      });
    }
  };

  const stopRecording = async () => {
    try {
      const data = await apiClient.post('/api/v1/sstv/record/stop');
      setIsRecording(false);
      toast.success('Запись остановлена', {
        description: data.message
      });
      fetchData();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Не удалось подключиться к API';
      console.error('Failed to stop recording:', error);
      toast.error('Ошибка остановки записи', {
        description: errorMessage
      });
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">SSTV Станция</h1>
            <p className="text-muted-foreground mt-1">
              Наземная станция приёма изображений с МКС
            </p>
          </div>
          <div className="flex gap-2">
            <Button 
              variant={isRecording ? "destructive" : "default"} 
              onClick={toggleRecording}
              disabled={!nextPass}
            >
              {isRecording ? (
                <>
                  <Square className="h-4 w-4 mr-2" />
                  Остановить
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2" />
                  Записать
                </>
              )}
            </Button>
          </div>
        </div>

        {/* Alerts */}
        {nextPass && (
          <Alert className={nextPass.time_until_aos.includes('-') ? "destructive" : "default"}>
            <Satellite className="h-4 w-4" />
            <AlertDescription>
              {nextPass.time_until_aos.includes('-') 
                ? `МКС сейчас не видима. Следующий пролёт: ${format(new Date(nextPass.aos), 'HH:mm:ss')}`
                : `МКС видима! Максимальная высота: ${nextPass.max_elevation}°. Окончание: ${format(new Date(nextPass.los), 'HH:mm:ss')}`
              }
            </AlertDescription>
          </Alert>
        )}

        {/* Status Cards */}
        <div className="grid gap-4 md:grid-cols-3">
          {/* Recording Status */}
          <Card className="glass border-border">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Статус записи</CardTitle>
              <Radio className={`h-4 w-4 ${isRecording ? 'text-green-500 animate-pulse' : 'text-muted-foreground'}`} />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {isRecording ? 'Идёт запись' : 'Ожидание'}
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Частота: {SSTV_ISS_FREQUENCY_MHZ} MHz
              </p>
            </CardContent>
          </Card>

          {/* Next Pass */}
          <Card className="glass border-border">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Следующий пролёт</CardTitle>
              <Clock className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {nextPass ? format(new Date(nextPass.aos), 'HH:mm') : '--:--'}
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                {nextPass ? `${nextPass.duration_minutes} мин, макс. ${nextPass.max_elevation}°` : 'Нет данных'}
              </p>
            </CardContent>
          </Card>

          {/* ISS Position */}
          <Card className="glass border-border">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Позиция МКС</CardTitle>
              <Satellite className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {issPosition ? `${issPosition.altitude_km} км` : '-- км'}
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                {issPosition ? `${issPosition.latitude.toFixed(1)}°, ${issPosition.longitude.toFixed(1)}°` : '--'}
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Detailed ISS Info */}
        {nextPass && (
          <Card className="glass border-border">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Satellite className="h-5 w-5" />
                Информация о пролёте МКС
              </CardTitle>
              <CardDescription>
                Данные о следующем видимом пролёте Международной Космической Станции
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-4">
                <div>
                  <div className="text-sm text-muted-foreground">Начало (AOS)</div>
                  <div className="text-lg font-semibold">
                    {format(new Date(nextPass.aos), 'dd.MM.yyyy HH:mm:ss')}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground">Конец (LOS)</div>
                  <div className="text-lg font-semibold">
                    {format(new Date(nextPass.los), 'dd.MM.yyyy HH:mm:ss')}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground">Макс. высота</div>
                  <div className="text-lg font-semibold">{nextPass.max_elevation}°</div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground">Длительность</div>
                  <div className="text-lg font-semibold">{nextPass.duration_minutes} мин</div>
                </div>
              </div>
              
              <div className="mt-4 flex items-center gap-4">
                <div className="flex-1">
                  <div className="text-sm text-muted-foreground mb-2">Частота приёма</div>
                  <Badge variant="outline" className="text-lg px-4 py-2">
                    {nextPass.frequency_mhz} MHz
                  </Badge>
                </div>
                <div className="flex-1">
                  <div className="text-sm text-muted-foreground mb-2">Время до начала</div>
                  <Badge className="text-lg px-4 py-2 bg-blue-500">
                    {nextPass.time_until_aos.replace('-', '')}
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Recordings Table */}
        <Card className="glass border-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <HardDrive className="h-5 w-5" />
              Записи SSTV
            </CardTitle>
            <CardDescription>
              Список записанных аудиофайлов для декодирования
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="text-center py-12">
                <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-2" />
                <p className="text-muted-foreground">Загрузка записей...</p>
              </div>
            ) : recordings.length === 0 ? (
              <div className="text-center py-12">
                <Radio className="h-16 w-16 mx-auto mb-4 opacity-20" />
                <h3 className="text-lg font-semibold mb-2">Нет записей</h3>
                <p className="text-muted-foreground mb-4">
                  {isRecording ? 'Идёт запись...' : 'Запишите сигнал SSTV для декодирования'}
                </p>
                {!isRecording && (
                  <Button onClick={toggleRecording}>
                    <Play className="h-4 w-4 mr-2" />
                    Начать запись
                  </Button>
                )}
              </div>
            ) : (
              <div className="space-y-2">
                {recordings.map((recording) => (
                  <div
                    key={recording.filename}
                    className="flex items-center justify-between p-4 rounded-lg border border-border bg-secondary/30 hover:bg-secondary/50 transition-colors"
                  >
                    <div className="flex items-center gap-4">
                      <div className="p-2 rounded-full bg-blue-500/20">
                        <Radio className="h-5 w-5 text-blue-500" />
                      </div>
                      <div>
                        <div className="font-medium">{recording.filename}</div>
                        <div className="text-sm text-muted-foreground">
                          {recording.frequency} MHz • {format(new Date(recording.created_at), 'dd.MM.yyyy HH:mm')}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="text-sm text-muted-foreground">
                        {formatFileSize(recording.size_bytes)}
                      </div>
                      <div className="flex gap-2">
                        <Button variant="outline" size="icon">
                          <Eye className="h-4 w-4" />
                        </Button>
                        <Button variant="outline" size="icon">
                          <Download className="h-4 w-4" />
                        </Button>
                        <Button variant="outline" size="icon">
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
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
