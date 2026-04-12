"use client";

import { DashboardLayout } from "@/components/dashboard-layout";
import { ArrowLeft, Download, Calendar, Image, Tag, Ruler, Brain, Loader2, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/config";
import { format } from "date-fns";
import { toast } from "@/components/ui/toaster";
import { useParams, useRouter } from "next/navigation";

interface Scan {
  id: number;
  scan_type: string;
  surface_type?: string;
  width?: number;
  height?: number;
  created_at: string;
  file_path?: string;
  metadata?: Record<string, unknown>;
}

interface DefectInfo {
  type: string;
  x: number;
  y: number;
  width: number;
  height: number;
  area: number;
  confidence: number;
}

interface AnalysisResult {
  analysis_id: string;
  image_path: string;
  model_name: string;
  defects_count: number;
  defects: DefectInfo[];
  confidence_score: number;
  processing_time_ms: number;
  timestamp: string;
}

export default function ScanDetailPage() {
  const params = useParams();
  const router = useRouter();
  const scanId = Number(params.id);

  const [scan, setScan] = useState<Scan | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);

  useEffect(() => {
    fetchScan();
  }, [scanId]);

  const fetchScan = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/scans/${scanId}`);
      if (res.ok) {
        const data = await res.json();
        console.log('[ScanDetail] Received data:', JSON.stringify(data, null, 2));
        setScan(data);
      } else {
        const errorData = await res.json().catch(() => null);
        toast.error('Сканирование не найдено', {
          description: errorData?.detail || errorData?.message || `HTTP ${res.status}`
        });
        setScan(null);
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Не удалось загрузить сканирование';
      console.error('Failed to fetch scan:', error);
      toast.error('Ошибка загрузки', { description: errorMessage });
      setScan(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownload = () => {
    if (scan?.file_path) {
      window.open(`${API_BASE}${scan.file_path}`, '_blank');
    } else {
      toast.error('Файл недоступен');
    }
  };

  const handleAnalyze = async () => {
    if (!scan?.file_path) {
      toast.error('Нет изображения', {
        description: 'Невозможно запустить анализ — у сканирования нет сохранённого изображения'
      });
      return;
    }

    setIsAnalyzing(true);
    try {
      const res = await fetch(`${API_BASE}/api/v1/analysis/defects`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_path: scan.file_path,
          model_name: 'isolation_forest'
        })
      });

      if (res.ok) {
        const data: AnalysisResult = await res.json();
        setAnalysisResult(data);
        toast.success('Анализ завершён', {
          description: `Обнаружено дефектов: ${data.defects_count}`
        });
      } else {
        const errorData = await res.json().catch(() => null);
        toast.error('Ошибка анализа', {
          description: errorData?.detail || errorData?.message || `HTTP ${res.status}`
        });
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Ошибка запуска анализа';
      toast.error('Ошибка анализа', { description: errorMessage });
    } finally {
      setIsAnalyzing(false);
    }
  };

  if (isLoading) {
    return (
      <DashboardLayout>
        <div className="text-center py-12">
          <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-2" />
          <p className="text-muted-foreground">Загрузка сканирования...</p>
        </div>
      </DashboardLayout>
    );
  }

  if (!scan) {
    return (
      <DashboardLayout>
        <div className="text-center py-12">
          <h2 className="text-2xl font-bold mb-2">Сканирование не найдено</h2>
          <p className="text-muted-foreground mb-4">
            Сканирование с ID #{scanId} не существует или было удалено
          </p>
          <Button onClick={() => router.push('/scans')}>
            <ArrowLeft className="h-4 w-4 mr-2" />
            Вернуться к списку
          </Button>
        </div>
      </DashboardLayout>
    );
  }

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button variant="outline" size="icon" onClick={() => router.push('/scans')}>
              <ArrowLeft className="h-4 w-4" />
            </Button>
            <div>
              <h1 className="text-2xl font-bold">Сканирование #{scan.id}</h1>
              <p className="text-muted-foreground mt-1">
                Детали результата сканирования
              </p>
            </div>
          </div>
          <Button onClick={handleDownload} disabled={!scan.file_path}>
            <Download className="h-4 w-4 mr-2" />
            Скачать
          </Button>
        </div>

        {/* Info Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Основная информация */}
          <div className="glass rounded-xl border border-border p-6">
            <h2 className="text-lg font-semibold mb-4">Основная информация</h2>
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <Tag className="h-5 w-5 text-muted-foreground" />
                <div>
                  <p className="text-sm text-muted-foreground">Тип сканирования</p>
                  <p className="font-medium capitalize">{scan.scan_type}</p>
                </div>
              </div>
              {scan.surface_type && (
                <div className="flex items-center gap-3">
                  <Image className="h-5 w-5 text-muted-foreground" />
                  <div>
                    <p className="text-sm text-muted-foreground">Тип поверхности</p>
                    <p className="font-medium">{scan.surface_type}</p>
                  </div>
                </div>
              )}
              <div className="flex items-center gap-3">
                <Calendar className="h-5 w-5 text-muted-foreground" />
                <div>
                  <p className="text-sm text-muted-foreground">Дата создания</p>
                  <p className="font-medium">
                    {format(new Date(scan.created_at), 'dd.MM.yyyy HH:mm:ss')}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Разрешение */}
          <div className="glass rounded-xl border border-border p-6">
            <h2 className="text-lg font-semibold mb-4">Параметры</h2>
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <Ruler className="h-5 w-5 text-muted-foreground" />
                <div>
                  <p className="text-sm text-muted-foreground">Разрешение</p>
                  <p className="font-medium">
                    {scan.width && scan.height ? `${scan.width} × ${scan.height}` : '—'}
                  </p>
                </div>
              </div>
              {scan.metadata && Object.keys(scan.metadata).length > 0 && (
                <div>
                  <p className="text-sm text-muted-foreground mb-2">Метаданные</p>
                  <pre className="text-xs bg-secondary/50 rounded p-3 overflow-auto max-h-40">
                    {JSON.stringify(scan.metadata, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Image */}
        {scan.file_path ? (
          <div className="glass rounded-xl border border-border p-6">
            <h2 className="text-lg font-semibold mb-4">Изображение</h2>
            <div className="rounded-lg overflow-hidden bg-secondary/30 flex items-center justify-center">
              <img
                src={`${API_BASE}${scan.file_path}`}
                alt={`Scan #${scan.id}`}
                className="max-w-full h-auto"
                onError={(e) => {
                  console.error('[ScanDetail] Image load failed:', `${API_BASE}${scan.file_path}`);
                  (e.target as HTMLImageElement).style.display = 'none';
                }}
              />
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              Путь: {scan.file_path}
            </p>
          </div>
        ) : (
          <div className="glass rounded-xl border border-border p-12 text-center">
            <Image className="h-16 w-16 mx-auto mb-4 opacity-20" />
            <h3 className="text-lg font-semibold mb-2">Изображение отсутствует</h3>
            <p className="text-muted-foreground">
              Для этого сканирования нет сохранённого изображения
            </p>
          </div>
        )}

        {/* AI Analysis */}
        <div className="glass rounded-xl border border-border p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <Brain className="h-5 w-5" />
              AI/ML Анализ дефектов
            </h2>
            <Button
              onClick={handleAnalyze}
              disabled={isAnalyzing || !scan.file_path}
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Анализ...
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4 mr-2" />
                  Запустить анализ
                </>
              )}
            </Button>
          </div>

          {!scan.file_path && (
            <div className="text-center py-8">
              <Brain className="h-12 w-12 mx-auto mb-4 opacity-20" />
              <p className="text-muted-foreground">
                Нет анализов
              </p>
              <p className="text-sm text-muted-foreground mt-1">
                Запустите AI/ML анализ для обнаружения дефектов
              </p>
            </div>
          )}

          {analysisResult && (
            <div className="space-y-4">
              {/* Summary */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-secondary/30 rounded-lg p-4">
                  <p className="text-sm text-muted-foreground">Дефектов</p>
                  <p className="text-2xl font-bold">{analysisResult.defects_count}</p>
                </div>
                <div className="bg-secondary/30 rounded-lg p-4">
                  <p className="text-sm text-muted-foreground">Уверенность</p>
                  <p className="text-2xl font-bold">{(analysisResult.confidence_score * 100).toFixed(1)}%</p>
                </div>
                <div className="bg-secondary/30 rounded-lg p-4">
                  <p className="text-sm text-muted-foreground">Время</p>
                  <p className="text-2xl font-bold">{analysisResult.processing_time_ms}ms</p>
                </div>
                <div className="bg-secondary/30 rounded-lg p-4">
                  <p className="text-sm text-muted-foreground">Модель</p>
                  <p className="text-lg font-semibold capitalize">{analysisResult.model_name.replace('_', ' ')}</p>
                </div>
              </div>

              {/* Defects List */}
              {analysisResult.defects.length > 0 && (
                <div>
                  <h3 className="font-semibold mb-2">Обнаруженные дефекты</h3>
                  <div className="space-y-2 max-h-60 overflow-y-auto">
                    {analysisResult.defects.map((defect, idx) => (
                      <div key={idx} className="bg-secondary/30 rounded-lg p-3 flex items-center justify-between">
                        <div>
                          <p className="font-medium capitalize">{defect.type}</p>
                          <p className="text-sm text-muted-foreground">
                            Позиция: ({defect.x}, {defect.y}) • Площадь: {defect.area}px²
                          </p>
                        </div>
                        <span className="px-2 py-1 rounded-full bg-blue-500/10 text-blue-500 text-sm">
                          {(defect.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {analysisResult.defects.length === 0 && (
                <div className="text-center py-6">
                  <p className="text-green-600 font-medium">✓ Дефекты не обнаружены</p>
                  <p className="text-sm text-muted-foreground mt-1">Поверхность в хорошем состоянии</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </DashboardLayout>
  );
}
