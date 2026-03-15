"use client";

import { DashboardLayout } from "@/components/dashboard-layout";
import { Radio, Download, Trash2, Eye, Play, Square, Signal } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/config";
import { format } from "date-fns";

interface SSTVTransmission {
  id: number;
  frequency: number;
  mode: string;
  status: string;
  signal_strength: number;
  created_at: string;
}

export default function SSTVPage() {
  const [transmissions, setTransmissions] = useState<SSTVTransmission[]>([]);
  const [isListening, setIsListening] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchTransmissions = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/v1/sstv`);
        if (res.ok) {
          const data = await res.json();
          setTransmissions(Array.isArray(data) ? data : []);
        }
      } catch (error) {
        console.error('Failed to fetch SSTV transmissions:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchTransmissions();
  }, []);

  const toggleListening = () => {
    setIsListening(!isListening);
    // Здесь будет логика подключения к RTL-SDR
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
            <Button variant={isListening ? "destructive" : "default"} onClick={toggleListening}>
              {isListening ? (
                <>
                  <Square className="h-4 w-4 mr-2" />
                  Остановить
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2" />
                  Слушать
                </>
              )}
            </Button>
          </div>
        </div>

        {/* Status Card */}
        <div className="glass rounded-xl border border-border p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className={`p-3 rounded-full ${isListening ? 'bg-green-500/20' : 'bg-gray-500/20'}`}>
                <Signal className={`h-6 w-6 ${isListening ? 'text-green-500' : 'text-gray-500'}`} />
              </div>
              <div>
                <h3 className="font-semibold">Статус приёмника</h3>
                <p className="text-muted-foreground">
                  {isListening ? 'Активное сканирование частоты 145.800 MHz' : 'Ожидание запуска'}
                </p>
              </div>
            </div>
            {isListening && (
              <div className="text-right">
                <div className="text-2xl font-bold text-green-500">145.800</div>
                <div className="text-sm text-muted-foreground">MHz</div>
              </div>
            )}
          </div>
        </div>

        {/* Transmissions Table */}
        {isLoading ? (
          <div className="text-center py-12">
            <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-2" />
            <p className="text-muted-foreground">Загрузка передач...</p>
          </div>
        ) : transmissions.length === 0 ? (
          <div className="text-center py-12 glass rounded-xl border border-border">
            <Radio className="h-16 w-16 mx-auto mb-4 opacity-20" />
            <h3 className="text-lg font-semibold mb-2">Нет передач</h3>
            <p className="text-muted-foreground mb-4">
              {isListening ? 'Ожидание сигнала SSTV...' : 'Запустите прослушивание для приёма изображений'}
            </p>
            {!isListening && (
              <Button onClick={toggleListening}>
                <Play className="h-4 w-4 mr-2" />
                Начать прослушивание
              </Button>
            )}
          </div>
        ) : (
          <div className="glass rounded-xl border border-border overflow-hidden">
            <table className="w-full">
              <thead className="bg-secondary/50 border-b border-border">
                <tr>
                  <th className="text-left p-4 font-medium text-muted-foreground">ID</th>
                  <th className="text-left p-4 font-medium text-muted-foreground">Частота</th>
                  <th className="text-left p-4 font-medium text-muted-foreground">Режим</th>
                  <th className="text-left p-4 font-medium text-muted-foreground">Сигнал</th>
                  <th className="text-left p-4 font-medium text-muted-foreground">Статус</th>
                  <th className="text-left p-4 font-medium text-muted-foreground">Дата</th>
                  <th className="text-right p-4 font-medium text-muted-foreground">Действия</th>
                </tr>
              </thead>
              <tbody>
                {transmissions.map((tx) => (
                  <tr
                    key={tx.id}
                    className="border-b border-border last:border-0 hover:bg-secondary/30 transition-colors"
                  >
                    <td className="p-4 font-medium">#{tx.id}</td>
                    <td className="p-4">
                      <span className="px-2 py-1 rounded-full bg-blue-500/10 text-blue-500 text-sm">
                        {tx.frequency} MHz
                      </span>
                    </td>
                    <td className="p-4 text-muted-foreground">{tx.mode}</td>
                    <td className="p-4">
                      <div className="flex items-center gap-2">
                        <div className="flex gap-0.5">
                          {[...Array(5)].map((_, i) => (
                            <div
                              key={i}
                              className={`w-1 h-4 rounded-sm ${
                                i < Math.floor(tx.signal_strength / 20)
                                  ? 'bg-green-500'
                                  : 'bg-gray-500/30'
                              }`}
                            />
                          ))}
                        </div>
                        <span className="text-sm text-muted-foreground">{tx.signal_strength}%</span>
                      </div>
                    </td>
                    <td className="p-4">
                      <span className={`px-2 py-1 rounded-full text-sm ${
                        tx.status === 'received'
                          ? 'bg-green-500/10 text-green-500'
                          : tx.status === 'receiving'
                          ? 'bg-blue-500/10 text-blue-500'
                          : 'bg-gray-500/10 text-gray-500'
                      }`}>
                        {tx.status}
                      </span>
                    </td>
                    <td className="p-4 text-muted-foreground">
                      {format(new Date(tx.created_at), 'dd.MM.yyyy HH:mm')}
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
