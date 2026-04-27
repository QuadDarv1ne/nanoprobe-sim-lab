#!/usr/bin/env python
"""
API Performance Benchmark
Быстрый бенчмарк основных endpoint'ов
"""
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.makedirs("data", exist_ok=True)

from fastapi.testclient import TestClient

import api.main
import api.state
from utils.database import DatabaseManager

db = DatabaseManager("data/nanoprobe.db")
api.state.set_db_manager(db)
api.main.db_manager = db

from api.main import app

client = TestClient(app)


def benchmark(endpoint: str, method: str = "GET", iterations: int = 50) -> dict:
    """Бенчмарк endpoint'а"""
    times = []
    status_codes = []
    errors = 0

    for _ in range(iterations):
        start = time.perf_counter()
        try:
            if method == "GET":
                resp = client.get(endpoint)
            elif method == "POST":
                resp = client.post(endpoint)
            status_codes.append(resp.status_code)
        except Exception:
            errors += 1
            continue
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    if not times:
        return {"status": "error", "errors": iterations}

    sorted_times = sorted(times)
    return {
        "avg": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "p95": sorted_times[int(len(sorted_times) * 0.95)],
        "p99": sorted_times[int(len(sorted_times) * 0.99)],
        "ok_rate": sum(1 for s in status_codes if s == 200) / len(status_codes) * 100,
        "errors": errors,
    }


def main():
    endpoints = [
        ("GET", "/health"),
        ("GET", "/api"),
        ("GET", "/api/v1/health"),
        ("GET", "/api/v1/scans"),
        ("GET", "/api/v1/simulations"),
        ("GET", "/api/v1/monitoring/health"),
        ("GET", "/metrics"),
    ]

    print("=" * 80)
    print("NANOPROBE SIM LAB — API PERFORMANCE BENCHMARK")
    print("=" * 80)
    print(
        f"{'Endpoint':<40} {'Avg':>8} {'Med':>8} {'P95':>8} {'P99':>8} {'Min':>8} {'Max':>8} {'OK%':>6}"
    )
    print("-" * 80)

    all_results = []
    for method, path in endpoints:
        result = benchmark(path, method, iterations=50)
        all_results.append(result)

        ok = f"{result['ok_rate']:.0f}%" if result["ok_rate"] < 100 else "100%"
        line = (
            f"{method:4s} {path:<35} "
            f"{result['avg']:6.1f} {result['median']:6.1f} "
            f"{result['p95']:6.1f} {result['p99']:6.1f} "
            f"{result['min']:6.1f} {result['max']:6.1f} {ok:>5}"
        )
        print(line)

    print("-" * 80)

    # Summary
    avg_all = statistics.mean(r["avg"] for r in all_results)
    p95_all = statistics.mean(r["p95"] for r in all_results)
    p99_all = statistics.mean(r["p99"] for r in all_results)

    print(f"OVERALL: avg={avg_all:.1f}ms p95={p95_all:.1f}ms p99={p99_all:.1f}ms")

    # Recommendations
    print("\nRecommendations:")
    if avg_all < 10:
        print("[OK] Excellent performance")
    elif avg_all < 50:
        print("[WARN] Good performance, room for improvement")
    else:
        print("[FAIL] Performance needs optimization")

    if any(r["p99"] > 100 for r in all_results):
        print("[WARN] High P99 - check slow endpoints")

    print("=" * 80)


if __name__ == "__main__":
    main()
