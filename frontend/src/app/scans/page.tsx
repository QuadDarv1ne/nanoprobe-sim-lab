"use client";

import { DashboardLayout } from "@/components/dashboard-layout";
import { FileText, Download, Trash2, Eye, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/config";
import { format } from "date-fns";
import { toast } from "@/components/ui/toaster";

interface Scan {
  id: number;
  scan_type: string;
  resolution: string;
  created_at: string;
  image_path?: string;
}

export default function ScansPage() {
  const [scans, setScans] = useState<Scan[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchScans();
  }, []);

  const fetchScans = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/scans`);
      const data = await res.json();
      setScans(Array.isArray(data) ? data : []);
    } catch (error) {
      console.error('Failed to fetch scans:', error);
      toast.error('Ошибка загрузки сканирований', {
        description: 'Не удалось получить список сканирований'
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleDelete = async (id: number) => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/scans/${id}`, {
        method: 'DELETE',
      });
      if (res.ok) {
        toast.success('Сканирование удалено');
        fetchScans();
      } else {
        toast.error('Ошибка удаления');
      }
    } catch (error) {
      toast.error('Ошибка удаления сканирования');
    }
  };

  const handleDownload = (scan: Scan) => {
    if (scan.image_path) {
      window.open(`${API_BASE}${scan.image_path}`, '_blank');
    } else {
      toast.error('Файл недоступен');
    }
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Сканирования</h1>
            <p className="text-muted-foreground mt-1">
              Управление результатами сканирований СЗМ
            </p>
          </div>
          <Button>
            <Plus className="h-4 w-4 mr-2" />
            Новое сканирование
          </Button>
        </div>

        {isLoading ? (
          <div className="text-center py-12">
            <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-2" />
            <p className="text-muted-foreground">Загрузка сканирований...</p>
          </div>
        ) : scans.length === 0 ? (
          <div className="text-center py-12 glass rounded-xl border border-border">
            <FileText className="h-16 w-16 mx-auto mb-4 opacity-20" />
            <h3 className="text-lg font-semibold mb-2">Нет сканирований</h3>
            <p className="text-muted-foreground mb-4">
              Создайте первое сканирование чтобы начать
            </p>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              Создать сканирование
            </Button>
          </div>
        ) : (
          <div className="glass rounded-xl border border-border overflow-hidden">
            <table className="w-full">
              <thead className="bg-secondary/50 border-b border-border">
                <tr>
                  <th className="text-left p-4 font-medium text-muted-foreground">ID</th>
                  <th className="text-left p-4 font-medium text-muted-foreground">Тип</th>
                  <th className="text-left p-4 font-medium text-muted-foreground">Разрешение</th>
                  <th className="text-left p-4 font-medium text-muted-foreground">Дата</th>
                  <th className="text-right p-4 font-medium text-muted-foreground">Действия</th>
                </tr>
              </thead>
              <tbody>
                {scans.map((scan) => (
                  <tr
                    key={scan.id}
                    className="border-b border-border last:border-0 hover:bg-secondary/30 transition-colors"
                  >
                    <td className="p-4 font-medium">#{scan.id}</td>
                    <td className="p-4">
                      <span className="px-2 py-1 rounded-full bg-blue-500/10 text-blue-500 text-sm">
                        {scan.scan_type}
                      </span>
                    </td>
                    <td className="p-4 text-muted-foreground">{scan.resolution}</td>
                    <td className="p-4 text-muted-foreground">
                      {format(new Date(scan.created_at), 'dd.MM.yyyy HH:mm')}
                    </td>
                    <td className="p-4">
                      <div className="flex items-center justify-end gap-2">
                        <Button variant="outline" size="icon" onClick={() => window.location.href = `/scans/${scan.id}`}>
                          <Eye className="h-4 w-4" />
                        </Button>
                        <Button variant="outline" size="icon" onClick={() => handleDownload(scan)}>
                          <Download className="h-4 w-4" />
                        </Button>
                        <Button variant="outline" size="icon" onClick={() => handleDelete(scan.id)}>
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
