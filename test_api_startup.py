#!/usr/bin/env python
"""Диагностика запуска api.main"""
import sys
import traceback
import time

print(f"Python: {sys.executable}")
print()

# Пошаговая загрузка
steps = [
    "api.main",
]

for step in steps:
    print(f"\n[LOAD] {step}")
    print("-" * 50)
    
    start = time.time()
    try:
        # Таймаут через сигнал не работает на Windows, используем threading
        import threading
        
        result = [None]
        error = [None]
        
        def do_import():
            try:
                result[0] = __import__(step, fromlist=['app'])
            except Exception as e:
                error[0] = e
        
        t = threading.Thread(target=do_import)
        t.daemon = True
        t.start()
        t.join(timeout=30)
        
        elapsed = time.time() - start
        
        if t.is_alive():
            print(f"[TIMEOUT] {step} > 30s")
            print("Активные потоки:")
            for th in threading.enumerate():
                print(f"  - {th.name} (alive={th.is_alive()})")
        elif error[0]:
            print(f"[ERROR] {step} ({elapsed:.2f}s)")
            traceback.print_exc()
        else:
            print(f"[OK] {step} ({elapsed:.2f}s)")
            app = getattr(result[0], 'app', None)
            if app:
                print(f"  App: {app}")
                
    except Exception as e:
        elapsed = time.time() - start
        print(f"[EXCEPTION] {step} ({elapsed:.2f}s)")
        traceback.print_exc()

print("\n[DONE]")
