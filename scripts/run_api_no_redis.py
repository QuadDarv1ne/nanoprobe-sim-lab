#!/usr/bin/env python
"""
Запуск FastAPI API без Redis (для тестирования)
"""

import os
import sys

import uvicorn

if sys.platform == "win32":
    os.system("chcp 65001 >nul")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Отключаем Redis перед импортом api.main
os.environ["REDIS_DISABLED"] = "1"

print("=" * 60)
print("🚀 Nanoprobe Sim Lab API (БЕЗ REDIS)")
print("=" * 60)
print()

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
