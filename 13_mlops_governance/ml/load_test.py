"""
Project 13 -- load-test the prediction API on this machine.

    python ml/load_test.py

Starts the API in its own process, warms it up, then sends real booking
requests at several concurrency levels and records latency percentiles,
throughput and errors in artifacts/load_test.json.

These are measurements of one uvicorn worker on the machine that ran them, and
are reported that way. They are not a production capacity claim.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
PORT = 8913
LEVELS = (1, 8, 32)
REQUESTS_PER_LEVEL = 600
WARMUP = 50

# Real, common routes: Midtown, the Upper East Side, JFK, LaGuardia, Penn Station.
ROUTES = [(161, 237), (237, 161), (132, 161), (138, 230), (186, 142), (236, 162), (48, 68), (170, 79)]


def request_body(i: int) -> bytes:
    pu, do = ROUTES[i % len(ROUTES)]
    hour = (7 + i * 5) % 24
    return json.dumps({"pu_location": pu, "do_location": do, "passenger_count": 1 + i % 3,
                       "when": f"2024-07-{1 + i % 28:02d}T{hour:02d}:15:00"}).encode()


def call(i: int) -> tuple[float, bool]:
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/api/predict", data=request_body(i),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            ok = resp.status == 200 and b"duration_minutes" in resp.read()
    except Exception:
        ok = False
    return (time.perf_counter() - t) * 1000, ok


def wait_ready(timeout: float = 120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/health", timeout=3):
                return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("API did not start")


def main() -> int:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "server"), str(ROOT / "ml")])}
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--app-dir", str(ROOT / "server"),
         "--port", str(PORT), "--log-level", "warning"],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        wait_ready()
        for i in range(WARMUP):              # load the model and fill caches
            call(i)
        levels = []
        for c in LEVELS:
            t = time.perf_counter()
            with ThreadPoolExecutor(max_workers=c) as pool:
                results = list(pool.map(call, range(REQUESTS_PER_LEVEL)))
            wall = time.perf_counter() - t
            lat = np.array([r[0] for r in results if r[1]])
            errors = sum(1 for r in results if not r[1])
            levels.append({
                "concurrency": c, "requests": REQUESTS_PER_LEVEL, "errors": errors,
                "throughput_rps": round((REQUESTS_PER_LEVEL - errors) / wall, 1),
                "p50_ms": round(float(np.percentile(lat, 50)), 2),
                "p95_ms": round(float(np.percentile(lat, 95)), 2),
                "p99_ms": round(float(np.percentile(lat, 99)), 2),
                "max_ms": round(float(lat.max()), 2),
            })
            print(f"concurrency {c:>3}: {levels[-1]['throughput_rps']:>7.1f} req/s  "
                  f"p50 {levels[-1]['p50_ms']:.1f} ms  p99 {levels[-1]['p99_ms']:.1f} ms  errors {errors}")
        out = {
            "machine": f"{platform.system()} {platform.release()}, {os.cpu_count()} logical CPUs",
            "server": "uvicorn, one worker", "warmup_requests": WARMUP, "levels": levels,
        }
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / "load_test.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        return 0
    finally:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(server.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        else:
            server.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
