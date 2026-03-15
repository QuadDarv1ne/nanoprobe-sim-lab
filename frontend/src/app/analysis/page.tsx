"use client";

import { DashboardLayout } from "@/components/dashboard-layout";
import { Brain, FileText, Download, Trash2, Eye, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/config";
import { format } from "date-fns";

interface Analysis {
  id: number;
  analysis_type: string;
  model: string;
  confidence: number;
  status: string;
  created_at: string;
}

export default function AnalysisPage() {
  const [analyses, setAnalyses] = useState<Analysis[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchAnalyses = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/v1/analysis`);
        if (res.ok) {
          const data = await res.json();
          setAnalyses(Array.isArray(data) ? data : []);
        }
      } catch (error) {
        console.error('Failed to fetch analyses:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchAnalyses();
  }, []);

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
                  <th className="text-left p-4 font-medium text-muted-foreground">Тип</th>
                  <th className="text-left p-4 font-medium text-muted-foreground">Модель</th>
                  <th className="text-left p-4 font-medium text-muted-foreground">Точность</th>
                  <th className="text-left p-4 font-medium text-muted-foreground">Статус</th>
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
                      <span className="px-2 py-1 rounded-full bg-purple-500/10 text-purple-500 text-sm">
                        {analysis.analysis_type}
                      </span>
                    </td>
                    <td className="p-4 text-muted-foreground">{analysis.model}</td>
                    <td className="p-4">
                      <div className="flex items-center gap-2">
                        <div className="flex-1 h-2 bg-gray-500/20 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-green-500 rounded-full"
                            style={{ width: `${analysis.confidence}%` }}
                          />
                        </div>
                        <span className="text-sm font-medium">{analysis.confidence}%</span>
                      </div>
                    </td>
                    <td className="p-4">
                      <span className={`px-2 py-1 rounded-full text-sm ${
                        analysis.status === 'completed'
                          ? 'bg-green-500/10 text-green-500'
                          : analysis.status === 'processing'
                          ? 'bg-blue-500/10 text-blue-500'
                          : 'bg-gray-500/10 text-gray-500'
                      }`}>
                        {analysis.status}
                      </span>
                    </td>
                    <td className="p-4 text-muted-foreground">
                      {format(new Date(analysis.created_at), 'dd.MM.yyyy HH:mm')}
                    </td>
                    <td className="p-4">
                      <div className="flex items-center justify-end gap-2">
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
