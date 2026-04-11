"use client";

import { DashboardLayout } from "@/components/dashboard-layout";
import { Brain, FileText, Download, Trash2, Eye, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/config";
import { format } from "date-fns";
import { toast } from "@/components/ui/toaster";
import { apiClient } from "@/lib/api-client";

interface Analysis {
  id: number;
  analysis_id: string;
  image_path: string;
  model_name: string;
  defects_detected: number;
  confidence_score: number;
  processing_time_ms?: number;
  created_at: string;
}

export default function AnalysisPage() {
  const [analyses, setAnalyses] = useState<Analysis[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [deletingIds, setDeletingIds] = useState<Set<number>>(new Set());

  useEffect(() => {
    fetchAnalyses();
  }, []);

  const fetchAnalyses = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/analysis/defects/history`);
      if (res.ok) {
        const data = await res.json();
        setAnalyses(Array.isArray(data) ? data : (data.items ?? []));
      } else {
        toast.error('Ошибка загрузки анализов', { description: `HTTP ${res.status}` });
      }
    } catch (error) {
      console.error('Failed to fetch analyses:', error);
      toast.error('Ошибка загрузки анализов', {
        description: 'Не удалось получить список AI/ML анализов'
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleDelete = async (id: number) => {
    if (deletingIds.has(id)) return;

    setDeletingIds(prev => new Set(prev).add(id));
    try {
      await apiClient.delete(`/api/v1/analysis/defects/${id}`);
      toast.success('Анализ удалён');
      fetchAnalyses();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Ошибка удаления';
      toast.error('Ошибка удаления анализа', {
        description: errorMessage
      });
    } finally {
      setDeletingIds(prev => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
    }
  };

  const handleDownload = async (id: number) => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/analysis/defects/${id}/export`);
      if (res.ok) {
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `analysis_${id}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        toast.success('Анализ загружен');
      } else {
        toast.error('Ошибка загрузки');
      }
    } catch (error) {
      toast.error('Ошибка загрузки анализа');
    }
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">AI/ML Анализ</h1>
            <p className="text-muted-foreground mt-1">
              Анализ дефектов поверхностей с помощью машинного обучения
            </p>
          </div>
          <Button>
            <Brain className="h-4 w-4 mr-2" />
            Новый анализ
          </Button>
        </div>

        {/* AI Stats */}
        <div className="grid gap-4 md:grid-cols-3">
          <div className="glass rounded-xl border border-border p-6">
            <div className="flex items-center gap-3">
              <div className="p-3 rounded-full bg-purple-500/20">
                <Brain className="h-6 w-6 text-purple-500" />
              </div>
              <div>
                <div className="text-2xl font-bold">{analyses.length}</div>
                <div className="text-sm text-muted-foreground">Анализов</div>
              </div>
            </div>
          </div>
          <div className="glass rounded-xl border border-border p-6">
            <div className="flex items-center gap-3">
              <div className="p-3 rounded-full bg-green-500/20">
                <Sparkles className="h-6 w-6 text-green-500" />
              </div>
              <div>
                <div className="text-2xl font-bold">94.5%</div>
                <div className="text-sm text-muted-foreground">Точность модели</div>
              </div>
            </div>
          </div>
          <div className="glass rounded-xl border border-border p-6">
            <div className="flex items-center gap-3">
              <div className="p-3 rounded-full bg-blue-500/20">
                <FileText className="h-6 w-6 text-blue-500" />
              </div>
              <div>
                <div className="text-2xl font-bold">12</div>
                <div className="text-sm text-muted-foreground">Моделей</div>
              </div>
            </div>
          </div>
        </div>

        {/* Analyses Table */}
        {isLoading ? (
          <div className="text-center py-12">
            <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-2" />
            <p className="text-muted-foreground">Загрузка анализов...</p>
          </div>
        ) : analyses.length === 0 ? (
          <div className="text-center py-12 glass rounded-xl border border-border">
            <Brain className="h-16 w-16 mx-auto mb-4 opacity-20" />
            <h3 className="text-lg font-semibold mb-2">Нет анализов</h3>
            <p className="text-muted-foreground mb-4">
              Запустите AI/ML анализ для обнаружения дефектов
            </p>
            <Button>
              <Brain className="h-4 w-4 mr-2" />
              Запустить анализ
            </Button>
          </div>
        ) : (
          <div className="glass rounded-xl border border-border overflow-hidden">
            <table className="w-full">
              <thead className="bg-secondary/50 border-b border-border">
                <tr>
                  <th className="text-left p-4 font-medium text-muted-foreground">ID</th>
                  <th className="text-left p-4 font-medium text-muted-foreground">Изображение</th>
                  <th className="text-left p-4 font-medium text-muted-foreground">Модель</th>
                  <th className="text-left p-4 font-medium text-muted-foreground">Дефектов</th>
                  <th className="text-left p-4 font-medium text-muted-foreground">Уверенность</th>
                  <th className="text-left p-4 font-medium text-muted-foreground">Дата</th>
                  <th className="text-right p-4 font-medium text-muted-foreground">Действия</th>
                </tr>
              </thead>
              <tbody>
                {analyses.map((analysis) => (
                  <tr
                    key={analysis.id}
                    className="border-b border-border last:border-0 hover:bg-secondary/30 transition-colors"
                  >
                    <td className="p-4 font-medium">#{analysis.id}</td>
                    <td className="p-4">
                      <span className="px-2 py-1 rounded-full bg-purple-500/10 text-purple-500 text-sm truncate max-w-[120px] block">
                        {analysis.image_path.split('/').pop() || analysis.image_path}
                      </span>
                    </td>
                    <td className="p-4 text-muted-foreground">{analysis.model_name}</td>
                    <td className="p-4 text-muted-foreground">{analysis.defects_detected}</td>
                    <td className="p-4">
                      <div className="flex items-center gap-2">
                        <div className="flex-1 h-2 bg-gray-500/20 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-green-500 rounded-full"
                            style={{ width: `${(analysis.confidence_score * 100).toFixed(0)}%` }}
                          />
                        </div>
                        <span className="text-sm font-medium">{(analysis.confidence_score * 100).toFixed(1)}%</span>
                      </div>
                    </td>
                    <td className="p-4 text-muted-foreground">
                      {format(new Date(analysis.created_at), 'dd.MM.yyyy HH:mm')}
                    </td>
                    <td className="p-4">
                      <div className="flex items-center justify-end gap-2">
                        <Button variant="outline" size="icon" onClick={() => window.location.href = `/analysis/${analysis.id}`}>
                          <Eye className="h-4 w-4" />
                        </Button>
                        <Button variant="outline" size="icon" onClick={() => handleDownload(analysis.id)}>
                          <Download className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="outline"
                          size="icon"
                          onClick={() => handleDelete(analysis.id)}
                          disabled={deletingIds.has(analysis.id)}
                          aria-label={`Удалить анализ #${analysis.id}`}
                        >
                          <Trash2 className={`h-4 w-4 ${deletingIds.has(analysis.id) ? 'animate-pulse' : ''}`} />
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}
