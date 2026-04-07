"use client";

import { DashboardLayout } from "@/components/dashboard-layout";
import { FileText, Download, Trash2, Eye, Printer, FilePlus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/config";
import { format } from "date-fns";
import { toast } from "@/components/ui/toaster";
import { apiClient } from "@/lib/api-client";

interface Report {
  id: number;
  report_type: string;
  format: string;
  status: string;
  created_at: string;
}

export default function ReportsPage() {
  const [reports, setReports] = useState<Report[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [deletingIds, setDeletingIds] = useState<Set<number>>(new Set());

  useEffect(() => {
    fetchReports();
  }, []);

  const fetchReports = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/reports`);
      if (res.ok) {
        const data = await res.json();
        setReports(Array.isArray(data) ? data : []);
      }
    } catch (error) {
      console.error('Failed to fetch reports:', error);
      toast.error('Ошибка загрузки отчётов', {
        description: 'Не удалось получить список отчётов'
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleDelete = async (id: number) => {
    if (deletingIds.has(id)) return;
    
    setDeletingIds(prev => new Set(prev).add(id));
    try {
      await apiClient.delete(`/api/v1/reports/${id}`);
      toast.success('Отчёт удалён');
      fetchReports();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Ошибка удаления';
      toast.error('Ошибка удаления отчёта', {
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
      const res = await fetch(`${API_BASE}/api/v1/reports/${id}/download`);
      if (res.ok) {
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `report_${id}.pdf`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        toast.success('Отчёт загружен');
      } else {
        toast.error('Ошибка загрузки');
      }
    } catch (error) {
      toast.error('Ошибка загрузки отчёта');
    }
  };

  const handlePrint = async (id: number) => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/reports/${id}/download`);
      if (res.ok) {
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const iframe = document.createElement('iframe');
        iframe.style.display = 'none';
        iframe.src = url;
        document.body.appendChild(iframe);
        
        let cleanedUp = false;
        const cleanup = () => {
          if (cleanedUp) return;
          cleanedUp = true;
          try {
            if (iframe.parentNode) {
              document.body.removeChild(iframe);
            }
            window.URL.revokeObjectURL(url);
          } catch {
            // Ignore cleanup errors
          }
        };
        
        // Timeout fallback - cleanup after 10s
        const timeoutId = setTimeout(() => {
          cleanup();
          toast.error('Печать не удалась (таймаут)');
        }, 10000);
        
        iframe.onload = () => {
          clearTimeout(timeoutId);
          try {
            iframe.contentWindow?.print();
            // Cleanup after print dialog closes
            setTimeout(cleanup, 1000);
          } catch {
            cleanup();
            toast.error('Ошибка печати');
          }
        };
        
        iframe.onerror = () => {
          clearTimeout(timeoutId);
          cleanup();
          toast.error('Ошибка загрузки для печати');
        };
      } else {
        toast.error('Ошибка печати');
      }
    } catch (error) {
      toast.error('Ошибка печати отчёта');
    }
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Отчёты</h1>
            <p className="text-muted-foreground mt-1">
              Генерация PDF отчётов для научных публикаций
            </p>
          </div>
          <Button>
            <FilePlus className="h-4 w-4 mr-2" />
            Новый отчёт
          </Button>
        </div>

        {/* Report Types */}
        <div className="grid gap-4 md:grid-cols-3">
          <div className="glass rounded-xl border border-border p-6 cursor-pointer hover:bg-secondary/50 transition-colors">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-3 rounded-full bg-blue-500/20">
                <FileText className="h-6 w-6 text-blue-500" />
              </div>
              <h3 className="font-semibold">Отчёт сканирования</h3>
            </div>
            <p className="text-sm text-muted-foreground">
              Детальный отчёт о результатах СЗМ сканирования
            </p>
          </div>
          <div className="glass rounded-xl border border-border p-6 cursor-pointer hover:bg-secondary/50 transition-colors">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-3 rounded-full bg-purple-500/20">
                <FileText className="h-6 w-6 text-purple-500" />
              </div>
              <h3 className="font-semibold">Отчёт анализа</h3>
            </div>
            <p className="text-sm text-muted-foreground">
              Отчёт AI/ML анализа дефектов поверхности
            </p>
          </div>
          <div className="glass rounded-xl border border-border p-6 cursor-pointer hover:bg-secondary/50 transition-colors">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-3 rounded-full bg-green-500/20">
                <FileText className="h-6 w-6 text-green-500" />
              </div>
              <h3 className="font-semibold">Сравнительный отчёт</h3>
            </div>
            <p className="text-sm text-muted-foreground">
              Отчёт сравнения поверхностей с метриками
            </p>
          </div>
        </div>

        {/* Reports Table */}
        {isLoading ? (
          <div className="text-center py-12">
            <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-2" />
            <p className="text-muted-foreground">Загрузка отчётов...</p>
          </div>
        ) : reports.length === 0 ? (
          <div className="text-center py-12 glass rounded-xl border border-border">
            <FileText className="h-16 w-16 mx-auto mb-4 opacity-20" />
            <h3 className="text-lg font-semibold mb-2">Нет отчётов</h3>
            <p className="text-muted-foreground mb-4">
              Создайте отчёт для экспорта результатов
            </p>
            <Button>
              <FilePlus className="h-4 w-4 mr-2" />
              Создать отчёт
            </Button>
          </div>
        ) : (
          <div className="glass rounded-xl border border-border overflow-hidden">
            <table className="w-full">
              <thead className="bg-secondary/50 border-b border-border">
                <tr>
                  <th className="text-left p-4 font-medium text-muted-foreground">ID</th>
                  <th className="text-left p-4 font-medium text-muted-foreground">Тип</th>
                  <th className="text-left p-4 font-medium text-muted-foreground">Формат</th>
                  <th className="text-left p-4 font-medium text-muted-foreground">Статус</th>
                  <th className="text-left p-4 font-medium text-muted-foreground">Дата</th>
                  <th className="text-right p-4 font-medium text-muted-foreground">Действия</th>
                </tr>
              </thead>
              <tbody>
                {reports.map((report) => (
                  <tr
                    key={report.id}
                    className="border-b border-border last:border-0 hover:bg-secondary/30 transition-colors"
                  >
                    <td className="p-4 font-medium">#{report.id}</td>
                    <td className="p-4">
                      <span className="px-2 py-1 rounded-full bg-blue-500/10 text-blue-500 text-sm">
                        {report.report_type}
                      </span>
                    </td>
                    <td className="p-4">
                      <span className="px-2 py-1 rounded-full bg-gray-500/10 text-gray-500 text-sm">
                        {report.format}
                      </span>
                    </td>
                    <td className="p-4">
                      <span className={`px-2 py-1 rounded-full text-sm ${
                        report.status === 'ready'
                          ? 'bg-green-500/10 text-green-500'
                          : report.status === 'generating'
                          ? 'bg-blue-500/10 text-blue-500'
                          : 'bg-gray-500/10 text-gray-500'
                      }`}>
                        {report.status}
                      </span>
                    </td>
                    <td className="p-4 text-muted-foreground">
                      {format(new Date(report.created_at), 'dd.MM.yyyy HH:mm')}
                    </td>
                    <td className="p-4">
                      <div className="flex items-center justify-end gap-2">
                        <Button variant="outline" size="icon" onClick={() => window.location.href = `/reports/${report.id}`}>
                          <Eye className="h-4 w-4" />
                        </Button>
                        <Button variant="outline" size="icon" onClick={() => handleDownload(report.id)} disabled={report.status !== 'ready'}>
                          <Download className="h-4 w-4" />
                        </Button>
                        <Button variant="outline" size="icon" onClick={() => handlePrint(report.id)} disabled={report.status !== 'ready'}>
                          <Printer className="h-4 w-4" />
                        </Button>
                        <Button 
                          variant="outline" 
                          size="icon" 
                          onClick={() => handleDelete(report.id)}
                          disabled={deletingIds.has(report.id)}
                          aria-label={`Удалить отчёт #${report.id}`}
                        >
                          <Trash2 className={`h-4 w-4 ${deletingIds.has(report.id) ? 'animate-pulse' : ''}`} />
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
