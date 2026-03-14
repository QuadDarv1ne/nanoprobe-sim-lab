"use client";

import { Bell, RefreshCw, Moon, Sun, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useTheme } from "@/providers/theme-provider";
import { useState } from "react";
import { useDashboardStore } from "@/stores/dashboard-store";
import { cn } from "@/lib/utils";

export function Header() {
  const { theme, toggleTheme } = useTheme();
  const [isRefreshing, setIsRefreshing] = useState(false);
  const { fetchDashboardData } = useDashboardStore();

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await fetchDashboardData();
    setIsRefreshing(false);
  };

  return (
    <header className="h-16 bg-card border-b border-border flex items-center justify-between px-6">
      <div className="flex items-center gap-4">
        <h2 className="text-lg font-semibold">Панель управления</h2>
      </div>

      <div className="flex items-center gap-3">
        {/* Refresh Button */}
        <Button
          variant="outline"
          size="icon"
          onClick={handleRefresh}
          disabled={isRefreshing}
          className="relative"
        >
          <RefreshCw className={cn("h-4 w-4", isRefreshing && "animate-spin")} />
        </Button>

        {/* Notifications */}
        <Button variant="outline" size="icon" className="relative">
          <Bell className="h-4 w-4" />
          <span className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 rounded-full text-xs flex items-center justify-center">
            3
          </span>
        </Button>

        {/* Theme Toggle */}
        <Button variant="outline" size="icon" onClick={toggleTheme}>
          {theme === "dark" ? (
            <Sun className="h-4 w-4" />
          ) : (
            <Moon className="h-4 w-4" />
          )}
        </Button>

        {/* User Menu */}
        <Button variant="ghost" className="gap-2">
          <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
            <User className="h-4 w-4 text-primary-foreground" />
          </div>
          <span className="text-sm font-medium hidden md:inline-block">Admin</span>
        </Button>
      </div>
    </header>
  );
}
