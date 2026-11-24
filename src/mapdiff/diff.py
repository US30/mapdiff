"""Map change detection: detected objects vs OpenStreetMap, aggregated on H3 cells.

Per class:
  matched  - detection within `match_m` of an OSM feature of the same class
  new      - detection with no OSM feature nearby  -> candidate map addition
  missing  - OSM feature with good image coverage but never detected
             -> candidate removal / wrong position in the map
  unseen   - OSM feature without enough image coverage (no evidence either way)
"""

from __future__ import annotations

import h3
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .config import CLASS_BY_NAME
from .geo import LocalFrame, coverage

H3_RES = 11  # ~25 m hexagon edge: about one street segment / intersection


def _match(det_xy, osm_xy, max_d):
    """Optimal 1:1 matching (Hungarian) with a distance gate."""
    if len(det_xy) == 0 or len(osm_xy) == 0:
        return []
    D = cdist(det_xy, osm_xy)
    big = 1e6
    r, c = linear_sum_assignment(np.where(D <= max_d, D, big))
    return [(i, j, D[i, j]) for i, j in zip(r, c) if D[i, j] <= max_d]


def diff_map(objects: pd.DataFrame, osm: pd.DataFrame, images: pd.DataFrame,
             match_m: float = 12.0, min_coverage: int = 3) -> pd.DataFrame:
    """One row per map change candidate or confirmed feature."""
    frame = LocalFrame(images.lon.mean(), images.lat.mean())
    osm = osm[osm.cls.isin(CLASS_BY_NAME)].copy()
    osm["coverage"] = coverage(osm, images)
    rows = []
    for cls in sorted(set(objects.cls) | set(osm.cls)):
        det = objects[objects.cls == cls].reset_index(drop=True)
        ref = osm[osm.cls == cls].reset_index(drop=True)
        dxy = frame.to_xy(det.lon, det.lat) if len(det) else np.zeros((0, 2))
        oxy = frame.to_xy(ref.lon, ref.lat) if len(ref) else np.zeros((0, 2))
        pairs = _match(dxy, oxy, match_m)
        md, mo = {i for i, _, _ in pairs}, {j for _, j, _ in pairs}
        for i, j, d in pairs:
            rows.append({"status": "matched", "cls": cls, "lon": det.lon[i], "lat": det.lat[i],
                         "obj_id": det.obj_id[i], "osm_id": ref.osm_id[j], "dist_m": d,
                         "n_images": det.n_images[i], "coverage": ref.coverage[j]})
        for i in set(range(len(det))) - md:
            rows.append({"status": "new", "cls": cls, "lon": det.lon[i], "lat": det.lat[i],
                         "obj_id": det.obj_id[i], "osm_id": None, "dist_m": np.nan,
                         "n_images": det.n_images[i], "coverage": np.nan})
        for j in set(range(len(ref))) - mo:
            status = "missing" if ref.coverage[j] >= min_coverage else "unseen"
            rows.append({"status": status, "cls": cls, "lon": ref.lon[j], "lat": ref.lat[j],
                         "obj_id": None, "osm_id": ref.osm_id[j], "dist_m": np.nan,
                         "n_images": 0, "coverage": ref.coverage[j]})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["h3"] = [h3.latlng_to_cell(la, lo, H3_RES) for la, lo in zip(out.lat, out.lon)]
    return out


def cell_summary(changes: pd.DataFrame) -> pd.DataFrame:
    """Per-H3-cell change counts and an anomaly score in [0, 1]."""
    if changes.empty:
        return pd.DataFrame()
    t = changes.pivot_table(index="h3", columns="status", values="cls", aggfunc="count",
                            fill_value=0)
    for s in ("matched", "new", "missing", "unseen"):
        if s not in t:
            t[s] = 0
    evidence = t.matched + t.new + t.missing
    t["anomaly"] = (t.new + t.missing) / (evidence + 1)  # +1: shrink tiny cells toward 0
    t = t.reset_index()
    t["boundary"] = [[[lo, la] for la, lo in h3.cell_to_boundary(c)] for c in t.h3]
    return t.sort_values("anomaly", ascending=False)
