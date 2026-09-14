"""
Build the taxi-zone SVG map from the official TLC shapefile.

    python ml/build_zone_map.py

The shapefile is published in NY State Plane (EPSG:2263, feet), which is
already a planar projection fitted to New York -- so the coordinates are used
directly rather than reprojected to lat/lon and re-flattened. Polygons are
simplified with Ramer-Douglas-Peucker and emitted as SVG paths in a normalised
viewBox, giving the client a ~200 KB map with no mapping library.

Output: client/src/zones.geo.json
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import shapefile  # pyshp

from dsfs import Provenance, fetch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "client" / "src" / "zones.geo.json"

VIEWBOX = 1000.0          # normalised coordinate space
RDP_TOLERANCE = 180.0     # feet; ~55 m, invisible at map scale

SHAPES = Provenance(
    slug="nyc-tlc-taxi-zones-shapefile",
    name="NYC TLC Taxi Zone Shapefile",
    source_url="https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip",
    license="NYC Taxi & Limousine Commission open data (public domain)",
    citation="NYC Taxi & Limousine Commission (2024). Taxi Zone Shapefile (EPSG:2263).",
    notes="Polygon boundaries for the 263 taxi zones, in NY State Plane feet.",
)


def rdp(points: list[tuple[float, float]], eps: float) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker polyline simplification."""
    if len(points) < 3:
        return points

    def dist(p, a, b) -> float:
        (px, py), (ax, ay), (bx, by) = p, a, b
        dx, dy = bx - ax, by - ay
        if dx == 0 and dy == 0:
            return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
        cx, cy = ax + t * dx, ay + t * dy
        return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5

    worst, idx = 0.0, 0
    for i in range(1, len(points) - 1):
        d = dist(points[i], points[0], points[-1])
        if d > worst:
            worst, idx = d, i

    if worst <= eps:
        return [points[0], points[-1]]
    left = rdp(points[: idx + 1], eps)
    right = rdp(points[idx:], eps)
    return left[:-1] + right


def main() -> None:
    # RDP recurses once per retained vertex on dense coastline rings.
    sys.setrecursionlimit(50_000)
    path, prov = fetch(SHAPES)
    raw = path.read_bytes()

    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        reader = shapefile.Reader(
            shp=io.BytesIO(z.read("taxi_zones/taxi_zones.shp")),
            dbf=io.BytesIO(z.read("taxi_zones/taxi_zones.dbf")),
            shx=io.BytesIO(z.read("taxi_zones/taxi_zones.shx")),
        )
        fields = [f[0] for f in reader.fields[1:]]
        records = [dict(zip(fields, r)) for r in reader.records()]
        shapes = reader.shapes()

    # Global bounds, so every zone shares one coordinate space.
    xs = [p[0] for s in shapes for p in s.points]
    ys = [p[1] for s in shapes for p in s.points]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    span = max(maxx - minx, maxy - miny)
    pad = (VIEWBOX - (maxx - minx) / span * VIEWBOX) / 2, \
          (VIEWBOX - (maxy - miny) / span * VIEWBOX) / 2

    def project(x: float, y: float) -> tuple[float, float]:
        # SVG y grows downward, so the vertical axis is flipped.
        px = (x - minx) / span * VIEWBOX + pad[0]
        py = VIEWBOX - ((y - miny) / span * VIEWBOX + pad[1])
        return round(px, 1), round(py, 1)

    zones = []
    kept = dropped = 0

    for rec, shp in zip(records, shapes):
        loc_id = int(rec.get("LocationID", 0))
        if not 1 <= loc_id <= 263:
            continue

        parts = list(shp.parts) + [len(shp.points)]
        rings: list[str] = []
        cx_acc = cy_acc = n_acc = 0.0

        for a, b in zip(parts[:-1], parts[1:]):
            ring = [tuple(p) for p in shp.points[a:b]]
            if len(ring) < 4:
                dropped += 1
                continue
            simple = rdp(ring, RDP_TOLERANCE)
            if len(simple) < 4:
                dropped += 1
                continue
            kept += 1
            pts = [project(x, y) for x, y in simple]
            rings.append(
                "M" + " L".join(f"{x},{y}" for x, y in pts) + " Z"
            )
            for x, y in pts:
                cx_acc += x
                cy_acc += y
                n_acc += 1

        if not rings:
            continue

        zones.append({
            "id": loc_id,
            "zone": str(rec.get("zone", "")).strip(),
            "borough": str(rec.get("borough", "")).strip(),
            "path": " ".join(rings),
            "cx": round(cx_acc / n_acc, 1),
            "cy": round(cy_acc / n_acc, 1),
        })

    zones.sort(key=lambda z: z["id"])
    payload = {
        "viewBox": f"0 0 {int(VIEWBOX)} {int(VIEWBOX)}",
        "projection": "EPSG:2263 (NY State Plane, feet), linearly normalised",
        "source": prov.source_url,
        "license": prov.license,
        "simplification": f"Ramer-Douglas-Peucker, tolerance {RDP_TOLERANCE} ft",
        "zones": zones,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    size_kb = OUT.stat().st_size / 1024
    print(f"zones written : {len(zones)}")
    print(f"rings kept    : {kept} (dropped {dropped} degenerate)")
    print(f"output        : {OUT}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
