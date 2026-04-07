/**
 * ================================================================================
 * Nanoprobe Sim Lab - Next.js Frontend (VERSION 2.0 - Modern/Production)
 * ================================================================================
 * Порт: 3000
 * Технологии: Next.js 14 + TypeScript + Tailwind CSS + Zustand + WebSocket
 * Запуск: cd frontend && npm run dev
 * Доступ: http://localhost:3000
 *
 * Это НОВАЯ версия frontend.
 * Старая версия (Flask) доступна в templates/dashboard.html на порту 5000.
 * ================================================================================
 */

import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "@/providers/theme-provider";
import { Toaster } from "@/components/ui/toaster";
import { PWAProvider } from "@/components/pwa-provider";
import { ErrorBoundary } from "@/components/error-boundary";

const inter = Inter({ subsets: ["latin", "cyrillic"] });

export const metadata: Metadata = {
  title: "Nanoprobe Sim Lab - Панель управления",
  description: "Современный интерфейс для лаборатории моделирования нанозонда",
  manifest: "/manifest.json",
  themeColor: "#3b82f6",
  appleWebApp: {
    capable: true,
    statusBarStyle: "default",
    title: "Nanoprobe Lab",
  },
  viewport: {
    width: "device-width",
    initialScale: 1,
    maximumScale: 1,
  },
  icons: {
    icon: "/icons/icon-192x192.png",
    apple: "/icons/icon-512x512.png",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ru" suppressHydrationWarning>
      <body className={inter.className}>
        <ErrorBoundary>
          <ThemeProvider>
            <PWAProvider>
              {children}
              <Toaster />
            </PWAProvider>
          </ThemeProvider>
        </ErrorBoundary>
      </body>
    </html>
  );
}
