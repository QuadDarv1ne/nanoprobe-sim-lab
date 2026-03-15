"use client";

import { DashboardLayout } from "@/components/dashboard-layout";
import { Settings as SettingsIcon, Database, Server, Bell, Moon, Sun, Monitor, Wifi } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { useState } from "react";

export default function SettingsPage() {
  const [darkMode, setDarkMode] = useState(true);
  const [notifications, setNotifications] = useState(true);
  const [autoSync, setAutoSync] = useState(true);

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
                {darkMode ? (
                  <Moon className="h-5 w-5 text-muted-foreground" />
                ) : (
                  <Sun className="h-5 w-5 text-muted-foreground" />
                )}
                <div>
                  <div className="font-medium">Тёмная тема</div>
                  <div className="text-sm text-muted-foreground">
                    {darkMode ? 'Включена' : 'Выключена'}
                  </div>
                </div>
              </div>
              <Switch checked={darkMode} onCheckedChange={setDarkMode} />
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
              <Switch checked={notifications} onCheckedChange={setNotifications} />
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
                <Button variant="outline">Проверить подключение</Button>
                <Button variant="outline">Сделать бэкап</Button>
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
                <Switch checked={autoSync} onCheckedChange={setAutoSync} />
              </div>
              <div className="grid gap-2">
                <div className="text-sm font-medium">API Endpoint</div>
                <code className="bg-secondary px-3 py-2 rounded-lg text-sm">
                  http://localhost:8000
                </code>
              </div>
              <div className="flex gap-2">
                <Button variant="outline">Проверить API</Button>
                <Button variant="outline">Документация</Button>
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
