"use client";

import { useEffect, useState } from "react";
import { Toaster as SonnerToaster, toast } from "sonner";

interface ToasterProps {
  richColors?: boolean;
}

export function Toaster({ richColors = true }: ToasterProps) {
  return (
    <SonnerToaster
      className="toaster group"
      toastOptions={{
        classNames: {
          toast:
            "group toast group-[.toaster]:bg-card group-[.toaster]:text-foreground group-[.toaster]:border-border group-[.toaster]:shadow-lg",
          description: "group-[.toast]:text-muted-foreground",
          actionButton:
            "group-[.toast]:bg-primary group-[.toast]:text-primary-foreground",
          cancelButton:
            "group-[.toast]:bg-muted group-[.toast]:text-muted-foreground",
        },
      }}
      richColors={richColors}
    />
  );
}

export { toast };
