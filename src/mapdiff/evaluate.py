"""Evaluation.

1. Against Mapillary's own map features (an independent detector + SfM pipeline):
   precision / recall at a distance gate and localisation error, per class.
2. Against manual labels written by the Streamlit labelling page
   (labels.csv: obj_id, verdict in {correct, wrong, unsure}).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CLASSES
from .diff import _match
from .geo import LocalFrame, coverage


def _mly_class(value: str) -> str | None:
    for c in CLASSES:
        if any(value.startswith(p) for p in c.mly_prefixes):
            return c.name
    return None


def vs_mapillary(objects: pd.DataFrame, mly: pd.DataFrame, images: pd.DataFrame,
                 gate_m: float = 10.0, min_coverage: int = 3) -> pd.DataFrame:
    """Recall only counts reference features our image subset actually saw
    (>= min_coverage images with the feature in field of view)."""
    frame = LocalFrame(images.lon.mean(), images.lat.mean())
    mly = mly.assign(cls=mly.object_value.map(_mly_class)).dropna(subset=["cls"])
    mly = mly[coverage(mly, images) >= min_coverage]
    rows = []
    for c in CLASSES:
        if not c.mly_prefixes:
            continue
        det = objects[objects.cls == c.name]
        ref = mly[mly.cls == c.name]
        if det.empty and ref.empty:
            continue
        dxy = frame.to_xy(det.lon, det.lat) if len(det) else np.zeros((0, 2))
        rxy = frame.to_xy(ref.lon, ref.lat) if len(ref) else np.zeros((0, 2))
        m = _match(dxy, rxy, gate_m)
        errs = [d for _, _, d in m]
        rows.append({
            "cls": c.name, "ours": len(det), "mapillary": len(ref), "matched": len(m),
            "precision": len(m) / len(det) if len(det) else np.nan,
            "recall": len(m) / len(ref) if len(ref) else np.nan,
            "median_err_m": float(np.median(errs)) if errs else np.nan,
            "p90_err_m": float(np.percentile(errs, 90)) if errs else np.nan,
        })
    return pd.DataFrame(rows)


def vs_labels(changes: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Precision of change flags from manual labels (unsure labels ignored)."""
    lab = labels[labels.verdict.isin(["correct", "wrong"])]
    j = changes.dropna(subset=["obj_id"]).astype({"obj_id": int}).merge(
        lab.astype({"obj_id": int}), on="obj_id")
    if j.empty:
        return pd.DataFrame()
    g = j.groupby(["status", "cls"]).verdict
    return pd.DataFrame({"labelled": g.size(),
                         "precision": g.apply(lambda v: (v == "correct").mean())}).reset_index()
