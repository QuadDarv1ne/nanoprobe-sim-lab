"use client";

import { DashboardLayout } from "@/components/dashboard-layout";
import { GitCompare, Download, Trash2, Eye, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/config";
import { format } from "date-fns";
import { toast } from "@/components/ui/toaster";
import { apiClient } from "@/lib/api-client";

interface Comparison {
  id: number;
  surface_a: string;
  surface_b: string;
  similarity: number;
  status: string;
  created_at: string;
}

export default function ComparisonPage() {
  const [comparisons, setComparisons] = useState<Comparison[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [deletingIds, setDeletingIds] = useState<Set<number>>(new Set());

  useEffect(() => {
    fetchComparisons();
  }, []);

  const fetchComparisons = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/comparison/history`);
      if (res.ok) {
        const data = await res.json();
        setComparisons(Array.isArray(data) ? data : (data.items ?? []));
      } else {
        toast.error('Ошибка загрузки сравнений', { description: `HTTP ${res.status}` });
      }
    } catch (error) {
      console.error('Failed to fetch comparisons:', error);
      toast.error('Ошибка загрузки сравнений', {
        description: 'Не удалось получить список сравнений поверхностей'
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleDelete = async (id: number) => {
    if (deletingIds.has(id)) return;
    
    setDeletingIds(prev => new Set(prev).add(id));
    try {
      await apiClient.delete(`/api/v1/comparison/${id}`);
      toast.success('Сравнение удалено');
      fetchComparisons();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Ошибка удаления';
      toast.error('Ошибка удаления сравнения', {
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
      const res = await fetch(`${API_BASE}/api/v1/comparison/${id}/export`);
      if (res.ok) {
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `comparison_${id}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        toast.success('Сравнение загружено');
      } else {
        toast.error('Ошибка загрузки');
      }
    } catch (error) {
      toast.error('Ошибка загрузки сравнения');
    }
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Сравнение поверхностей</h1>
            <p className="text-muted-foreground mt-1">
              Сравнение изображений поверхностей для анализа изменений
            </p>
          </div>
          <Button>
            <Plus className="h-4 w-4 mr-2" />
            Новое сравнение
          </Button>
        </div>

        {/* Comparison Stats */}
        <div className="grid gap-4 md:grid-cols-2">
          <div className="glass rounded-xl border border-border p-6">
            <div className="flex items-center gap-3">
              <div className="p-3 rounded-full bg-blue-500/20">
                <GitCompare className="h-6 w-6 text-blue-500" />
              </div>
              <div>
                <div className="text-2xl font-bold">{comparisons.length}</div>
                <div className="text-sm text-muted-foreground">Сравнений</div>
              </div>
            </div>
          </div>
          <div className="glass rounded-xl border border-border p-6">
            <div className="flex items-center gap-3">
              <div className="p-3 rounded-full bg-green-500/20">
                <GitCompare className="h-6 w-6 text-green-500" />
              </div>
              <div>
                <div className="text-2xl font-bold">
                  {comparisons.length > 0
                    ? Math.round(comparisons.reduce((acc, c) => acc + c.similarity, 0) / comparisons.length)
                    : 0}%
                </div>
                <div className="text-sm text-muted-foreground">Средняя схожесть</div>
              </div>
            </div>
          </div>
        </div>

        {/* Comparisons Table */}
        {isLoading ? (
          <div className="text-center py-12">
            <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-2" />
            <p className="text-muted-foreground">Загрузка сравнений...</p>
          </div>
        ) : comparisons.length === 0 ? (
          <div className="text-center py-12 glass rounded-xl border border-border">
            <GitCompare className="h-16 w-16 mx-auto mb-4 opacity-20" />
            <h3 className="text-lg font-semibold mb-2">Нет сравнений</h3>
            <p className="text-muted-foreground mb-4">
              Выберите два изображения для сравнения
            </p>
            <Button>
              <GitCompare className="h-4 w-4 mr-2" />
            Сравнить поверхности
            </Button>
          </div>
        ) : (
          <div className="glass rounded-xl border border-border overflow-hidden">
            <table className="w-full">
              <thead className="bg-secondary/50 border-b border-border">
                <tr>
                  <th className="text-left p-4 font-medium text-muted-foreground">ID</th>
                  <th className="text-left p-4 font-medium text-muted-foreground">Поверхность A</th>
                  <th className="text-left p-4 font-medium text-muted-foreground">Поверхность B</th>
                  <th className="text-left p-4 font-medium text-muted-foreground">Схожесть</th>
                  <th className="text-left p-4 font-medium text-muted-foreground">Статус</th>
                  <th className="text-left p-4 font-medium text-muted-foreground">Дата</th>
                  <th className="text-right p-4 font-medium text-muted-foreground">Действия</th>
                </tr>
              </thead>
              <tbody>
                {comparisons.map((comp) => (
                  <tr
                    key={comp.id}
                    className="border-b border-border last:border-0 hover:bg-secondary/30 transition-colors"
                  >
                    <td className="p-4 font-medium">#{comp.id}</td>
                    <td className="p-4 text-muted-foreground truncate max-w-[150px]">
                      {comp.surface_a}
                    </td>
                    <td className="p-4 text-muted-foreground truncate max-w-[150px]">
                      {comp.surface_b}
                    </td>
                    <td className="p-4">
                      <div className="flex items-center gap-2">
                        <div className="flex-1 h-2 bg-gray-500/20 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-blue-500 rounded-full"
                            style={{ width: `${comp.similarity}%` }}
                          />
                        </div>
                        <span className="text-sm font-medium">{comp.similarity}%</span>
                      </div>
                    </td>
                    <td className="p-4">
                      <span className={`px-2 py-1 rounded-full text-sm ${
                        comp.status === 'completed'
                          ? 'bg-green-500/10 text-green-500'
                          : comp.status === 'processing'
                          ? 'bg-blue-500/10 text-blue-500'
                          : 'bg-gray-500/10 text-gray-500'
                      }`}>
                        {comp.status}
                      </span>
                    </td>
                    <td className="p-4 text-muted-foreground">
                      {format(new Date(comp.created_at), 'dd.MM.yyyy HH:mm')}
                    </td>
                    <td className="p-4">
                      <div className="flex items-center justify-end gap-2">
                        <Button variant="outline" size="icon" onClick={() => window.location.href = `/comparison/${comp.id}`}>
                          <Eye className="h-4 w-4" />
                        </Button>
                        <Button variant="outline" size="icon" onClick={() => handleDownload(comp.id)}>
                          <Download className="h-4 w-4" />
                        </Button>
                        <Button 
                          variant="outline" 
                          size="icon" 
                          onClick={() => handleDelete(comp.id)}
                          disabled={deletingIds.has(comp.id)}
                          aria-label={`Удалить сравнение #${comp.id}`}
                        >
                          <Trash2 className={`h-4 w-4 ${deletingIds.has(comp.id) ? 'animate-pulse' : ''}`} />
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
