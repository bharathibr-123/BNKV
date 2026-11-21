"""Build the simplified India state map used by the dashboard.

Source: DataMeet India state boundaries (CC BY 4.0),
https://github.com/datameet/maps (States/Admin2). Needs ``pip install pyshp``.
Run from the repository root:  python scripts/build_map.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import shapefile  # type: ignore[import-untyped]

SRC = Path("data_raw/dmaps/States/Admin2")
OUT = Path("app/static/data/india_states.json")

# Main official language of each state, simplified to the languages we measure.
STATE_LANGUAGE = {
    "Karnataka": "kan_Knda", "Tamil Nadu": "tam_Taml", "Puducherry": "tam_Taml",
    "Kerala": "mal_Mlym", "Lakshadweep": "mal_Mlym", "Andhra Pradesh": "tel_Telu",
    "Telangana": "tel_Telu", "Maharashtra": "mar_Deva", "Gujarat": "guj_Gujr",
    "Dadra and Nagar Haveli and Daman and Diu": "guj_Gujr", "Punjab": "pan_Guru",
    "Odisha": "ory_Orya", "West Bengal": "ben_Beng", "Tripura": "ben_Beng",
    "Assam": "asm_Beng", "Sikkim": "npi_Deva", "Jammu & Kashmir": "urd_Arab",
    **{s: "hin_Deva" for s in (
        "Uttar Pradesh", "Madhya Pradesh", "Rajasthan", "Bihar", "Haryana", "Delhi",
        "Chandigarh", "Himachal Pradesh", "Uttarakhand", "Jharkhand", "Chhattisgarh",
        "Andaman & Nicobar",
    )},
}


def _simplify(points: list[tuple[float, float]], tol: float) -> list[tuple[float, float]]:
    """Douglas-Peucker line simplification."""
    if len(points) < 3:
        return points
    (x1, y1), (x2, y2) = points[0], points[-1]
    dx, dy = x2 - x1, y2 - y1
    norm = (dx * dx + dy * dy) ** 0.5 or 1e-12
    best, idx = 0.0, 0
    for i in range(1, len(points) - 1):
        px, py = points[i]
        d = abs(dy * px - dx * py + x2 * y1 - y2 * x1) / norm
        if d > best:
            best, idx = d, i
    if best <= tol:
        return [points[0], points[-1]]
    return _simplify(points[: idx + 1], tol)[:-1] + _simplify(points[idx:], tol)


def _simplify_ring(points: list[tuple[float, float]], tol: float) -> list[tuple[float, float]]:
    """Simplify a closed ring by splitting it at the point farthest from its start."""
    x0, y0 = points[0]
    far = max(range(len(points)), key=lambda i: (points[i][0] - x0) ** 2 + (points[i][1] - y0) ** 2)
    if far in (0, len(points) - 1):
        return points
    return _simplify(points[: far + 1], tol)[:-1] + _simplify(points[far:], tol)


def build(tol: float = 0.02) -> int:
    reader = shapefile.Reader(str(SRC))
    states = []
    for shape_rec in reader.iterShapeRecords():
        name = str(shape_rec.record[0])
        shp = shape_rec.shape
        parts = [*shp.parts, len(shp.points)]
        rings = []
        for a, b in itertools.pairwise(parts):
            ring = _simplify_ring([(float(x), float(y)) for x, y in shp.points[a:b]], tol)
            if len(ring) >= 4:
                rings.append([[round(x, 3), round(y, 3)] for x, y in ring])
        states.append({"name": name, "language": STATE_LANGUAGE.get(name), "rings": rings})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "attribution": "India state boundaries: DataMeet India community, CC BY 4.0",
        "states": states,
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return OUT.stat().st_size


if __name__ == "__main__":
    import sys

    sys.setrecursionlimit(200_000)
    print(build(), "bytes")
