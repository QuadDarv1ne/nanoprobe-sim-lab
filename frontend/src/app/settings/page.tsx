"use client";

import { DashboardLayout } from "@/components/dashboard-layout";
import { Settings as SettingsIcon, Database, Server, Bell, Monitor, Wifi, Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { useState, useEffect } from "react";
import { toast } from "@/components/ui/toaster";
import { API_BASE } from "@/lib/config";
import { useTheme } from "@/providers/theme-provider";

export default function SettingsPage() {
  const { theme, toggleTheme } = useTheme();
  const [notifications, setNotifications] = useState(true);
  const [autoSync, setAutoSync] = useState(true);

  useEffect(() => {
    const savedNotifications = localStorage.getItem('notifications');
    const savedAutoSync = localStorage.getItem('autoSync');

    if (savedNotifications !== null) setNotifications(savedNotifications === 'true');
    if (savedAutoSync !== null) setAutoSync(savedAutoSync === 'true');
  }, []);

  const handleNotificationsChange = (checked: boolean) => {
    setNotifications(checked);
    localStorage.setItem('notifications', String(checked));
    toast.success(checked ? 'Уведомления включены' : 'Уведомления выключены');
  };

  const handleAutoSyncChange = (checked: boolean) => {
    setAutoSync(checked);
    localStorage.setItem('autoSync', String(checked));
    toast.success(checked ? 'Автосинхронизация включена' : 'Автосинхронизация выключена');
  };

  const handleCheckDatabase = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/health/database`);
      if (res.ok) {
        toast.success('База данных работает корректно');
      } else {
        toast.error('Ошибка подключения к базе данных');
      }
    } catch (error) {
      toast.error('Не удалось проверить базу данных');
    }
  };

  const handleBackupDatabase = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/database/backup`, {
        method: 'POST',
      });
      if (res.ok) {
        toast.success('Бэкап базы данных создан');
      } else {
        toast.error('Ошибка создания бэкапа');
      }
    } catch (error) {
      toast.error('Не удалось создать бэкап');
    }
  };

  const handleCheckAPI = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/health`);
      if (res.ok) {
        const data = await res.json();
        toast.success(`API работает: ${data.status || 'OK'}`);
      } else {
        toast.error('API недоступен');
      }
    } catch (error) {
      toast.error('Не удалось подключиться к API');
    }
  };

  const handleOpenDocs = () => {
    window.open(`${API_BASE}/docs`, '_blank');
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold">Настройки</h1>
          <p className="text-muted-foreground mt-1">
            Управление конфигурацией системы
          </p>
        </div>

        {/* Appearance */}
        <Card className="glass border-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Monitor className="h-5 w-5" />
              Внешний вид
            </CardTitle>
            <CardDescription>Настройки темы и отображения</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                {theme === "dark" ? (
                  <Moon className="h-5 w-5 text-muted-foreground" />
                ) : (
                  <Sun className="h-5 w-5 text-muted-foreground" />
                )}
                <div>
                  <div className="font-medium">Тёмная тема</div>
                  <div className="text-sm text-muted-foreground">
                    {theme === "dark" ? 'Включена' : 'Выключена'}
                  </div>
                </div>
              </div>
              <Switch checked={theme === "dark"} onCheckedChange={toggleTheme} />
            </div>
          </CardContent>
        </Card>

        {/* Notifications */}
        <Card className="glass border-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Bell className="h-5 w-5" />
              Уведомления
            </CardTitle>
            <CardDescription>Настройки уведомлений системы</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Bell className="h-5 w-5 text-muted-foreground" />
                <div>
                  <div className="font-medium">Уведомления</div>
                  <div className="text-sm text-muted-foreground">
                    Показывать уведомления о событиях
                  </div>
                </div>
              </div>
              <Switch checked={notifications} onCheckedChange={handleNotificationsChange} />
            </div>
          </CardContent>
        </Card>

        {/* Database */}
        <Card className="glass border-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-5 w-5" />
              База данных
            </CardTitle>
            <CardDescription>Настройки подключения к базе данных</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="grid gap-2">
                <div className="text-sm font-medium">Connection String</div>
                <code className="bg-secondary px-3 py-2 rounded-lg text-sm">
                  sqlite+aiosqlite:///./nanoprobe.db
                </code>
              </div>
              <div className="flex gap-2">
                <Button variant="outline" onClick={handleCheckDatabase}>Проверить подключение</Button>
                <Button variant="outline" onClick={handleBackupDatabase}>Сделать бэкап</Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* API Settings */}
        <Card className="glass border-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Server className="h-5 w-5" />
              API Настройки
            </CardTitle>
            <CardDescription>Конфигурация Backend API</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Wifi className="h-5 w-5 text-muted-foreground" />
                  <div>
                    <div className="font-medium">Автосинхронизация</div>
                    <div className="text-sm text-muted-foreground">
                      Автоматическая синхронизация с Backend
                    </div>
                  </div>
                </div>
                <Switch checked={autoSync} onCheckedChange={handleAutoSyncChange} />
              </div>
              <div className="grid gap-2">
                <div className="text-sm font-medium">API Endpoint</div>
                <code className="bg-secondary px-3 py-2 rounded-lg text-sm">
                  {API_BASE}
                </code>
              </div>
              <div className="flex gap-2">
                <Button variant="outline" onClick={handleCheckAPI}>Проверить API</Button>
                <Button variant="outline" onClick={handleOpenDocs}>Документация</Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* System Info */}
        <Card className="glass border-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <SettingsIcon className="h-5 w-5" />
              Информация о системе
            </CardTitle>
            <CardDescription>Версии компонентов</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-1">
                <div className="text-sm text-muted-foreground">Frontend (Next.js)</div>
                <div className="font-medium">2.0.0</div>
              </div>
              <div className="space-y-1">
                <div className="text-sm text-muted-foreground">Backend (FastAPI)</div>
                <div className="font-medium">1.0.0</div>
              </div>
              <div className="space-y-1">
                <div className="text-sm text-muted-foreground">Python</div>
                <div className="font-medium">3.13</div>
              </div>
              <div className="space-y-1">
                <div className="text-sm text-muted-foreground">Database</div>
                <div className="font-medium">SQLite 3.x</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
}
