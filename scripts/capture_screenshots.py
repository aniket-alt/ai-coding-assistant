"""
Capture a project's screenshot tour.

    python scripts/capture_screenshots.py 01_nyc_taxi_prediction

Starts the project's API and Vite dev server, walks the tabs, and writes PNGs
into <project>/screenshots/. Also asserts that each view actually rendered
content rather than an error state -- a screenshot of a blank page is worse
than no screenshot, because it looks like evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"

# Tab id -> (filename, a selector or text that proves the view rendered)
TOURS: dict[str, list[tuple[str, str, str]]] = {
    "01_nyc_taxi_prediction": [
        ("estimator", "01_trip_estimator", "Estimated duration"),
        ("crispdm", "02_crispdm_walkthrough", "Phase gate"),
        ("models", "03_model_tournament", "Model tournament"),
        ("audit", "04_leakage_audit", "Assertions executed during training"),
    ],
    "06_anomaly_detection": [
        ("scorer", "01_transaction_scorer", "Suspicion score"),
        ("budget", "02_review_budget", "Alerts reviewed per day"),
        ("crispdm", "03_crispdm_walkthrough", "Phase gate"),
        ("models", "04_detector_tournament", "Detector tournament"),
        ("audit", "05_leakage_audit", "Assertions executed during training"),
    ],
}

PORTS = {
    "01_nyc_taxi_prediction": (8001, 5101),
    "06_anomaly_detection": (8006, 5106),
}


def wait_for(url: str, timeout: int = 90) -> bool:
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status < 500:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("--api-port", type=int)
    ap.add_argument("--web-port", type=int)
    ap.add_argument("--keep-open", action="store_true")
    args = ap.parse_args()

    proj = ROOT / args.project
    if not proj.exists():
        print(f"no such project: {proj}", file=sys.stderr)
        return 2

    meta = json.loads((proj / "artifacts" / "provenance.json").read_text(encoding="utf-8")) \
        if (proj / "artifacts" / "provenance.json").exists() else {}
    default_api, default_web = PORTS.get(args.project, (8001, 5101))
    api_port = args.api_port or default_api
    web_port = args.web_port or default_web

    shots = proj / "screenshots"
    shots.mkdir(exist_ok=True)

    env_path = f"{proj / 'server'};{proj / 'ml'}"
    api = subprocess.Popen(
        [str(VENV_PY), "-m", "uvicorn", "main:app",
         "--app-dir", str(proj / "server"), "--port", str(api_port)],
        cwd=proj, env={**os.environ, "PYTHONPATH": env_path},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # Invoke vite directly rather than through npm: npm spawns vite as a
    # grandchild, and terminating npm on Windows leaves vite holding the port.
    # --host 127.0.0.1 forces IPv4; vite's default "localhost" binds ::1 only
    # on Windows, which the readiness poll below cannot reach.
    vite = ROOT / "node_modules" / "vite" / "bin" / "vite.js"
    web = subprocess.Popen(
        ["node", str(vite), "--port", str(web_port), "--strictPort",
         "--host", "127.0.0.1"],
        cwd=proj / "client",
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    failures: list[str] = []
    try:
        if not wait_for(f"http://127.0.0.1:{api_port}/api/health"):
            print("API did not start", file=sys.stderr)
            return 1
        if not wait_for(f"http://127.0.0.1:{web_port}/"):
            print("dev server did not start", file=sys.stderr)
            return 1
        print(f"api :{api_port}  web :{web_port}  -> capturing")

        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            for theme in ("light", "dark"):
                page = browser.new_page(
                    viewport={"width": 1560, "height": 1000},
                    device_scale_factor=2,
                    color_scheme=theme,
                )
                page.goto(f"http://127.0.0.1:{web_port}/", wait_until="networkidle")

                for tab_id, name, proof in TOURS[args.project]:
                    page.get_by_role("button", name=_tab_label(args.project, tab_id)).click()
                    page.wait_for_timeout(1400)

                    ok = page.get_by_text(proof, exact=False).count() > 0
                    if not ok:
                        failures.append(f"{theme}/{name}: expected text {proof!r} not found")

                    suffix = "" if theme == "light" else "_dark"
                    out = shots / f"{name}{suffix}.png"
                    page.screenshot(path=str(out), full_page=True)
                    mark = "ok " if ok else "!! "
                    print(f"  {mark}{out.relative_to(ROOT)}")
                page.close()
            browser.close()

        if failures:
            print("\nviews that did not render expected content:", file=sys.stderr)
            for f in failures:
                print(f"  - {f}", file=sys.stderr)
            return 1

        print(f"\nall views rendered. dataset: {meta.get('name', '—')}")
        return 0
    finally:
        if not args.keep_open:
            _kill_tree(web)
            _kill_tree(api)


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill a process and its descendants; terminate() alone orphans children."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _tab_label(project: str, tab_id: str) -> str:
    labels = {
        "estimator": "Trip estimator",
        "crispdm": "CRISP-DM",
        "models": "Model tournament" if project == "01_nyc_taxi_prediction"
                  else "Detector tournament",
        "audit": "Audit",
        "scorer": "Transaction scorer",
        "budget": "Review budget",
    }
    return labels[tab_id]


if __name__ == "__main__":
    raise SystemExit(main())
