"use client";

import { cn } from "@/lib/utils";
import { ButtonHTMLAttributes, forwardRef } from "react";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "outline" | "ghost" | "destructive";
  size?: "default" | "sm" | "lg" | "icon";
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "default", size = "default", ...props }, ref) => {
    const baseStyles =
      "inline-flex items-center justify-center rounded-lg font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50";

    const variants = {
      default: "bg-primary text-primary-foreground hover:bg-primary/90 shadow-lg shadow-primary/20",
      outline: "border border-border bg-transparent hover:bg-secondary hover:text-foreground",
      ghost: "hover:bg-secondary hover:text-foreground",
      destructive: "bg-red-500 text-white hover:bg-red-600 shadow-lg shadow-red-500/20",
    };

    const sizes = {
      default: "h-10 px-4 py-2 text-sm",
      sm: "h-9 px-3 text-xs",
      lg: "h-11 px-8 text-base",
      icon: "h-10 w-10",
    };

    return (
      <button
        className={cn(baseStyles, variants[variant], sizes[size], className)}
        ref={ref}
        {...props}
      />
    );
  }
);

Button.displayName = "Button";

export { Button };
