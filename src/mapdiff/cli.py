"""mapdiff CLI. Each stage caches its output under data/<aoi>/ so stages can rerun alone.

  mapdiff fetch  --aoi bandra [--max-images 3000]
  mapdiff detect --aoi bandra
  mapdiff locate --aoi bandra
  mapdiff diff   --aoi bandra
  mapdiff eval   --aoi bandra
  mapdiff all    --aoi bandra
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from .config import AOIS


def _p(aoi, name):
    return aoi.dir / name


def fetch(aoi, max_images=None):
    from .mapillary import download_thumbs, fetch_images, fetch_map_features
    from .osm import fetch_osm

    imgs = fetch_images(aoi, max_images)
    imgs = download_thumbs(imgs, aoi.dir / "thumbs")
    imgs.to_parquet(_p(aoi, "images.parquet"))
    fetch_osm(aoi).to_parquet(_p(aoi, "osm.parquet"))
    fetch_map_features(aoi).to_parquet(_p(aoi, "mly_features.parquet"))
    print(f"images: {len(imgs)}  sequences: {imgs.sequence.nunique()}")


def detect(aoi, conf=0.25):
    from .detect import detect_and_track

    imgs = pd.read_parquet(_p(aoi, "images.parquet"))
    dets = detect_and_track(imgs, conf=conf)
    dets.to_parquet(_p(aoi, "detections.parquet"))
    print(dets.cls.value_counts().to_string() if len(dets) else "no detections")


def locate(aoi):
    from .geo import locate_objects

    imgs = pd.read_parquet(_p(aoi, "images.parquet"))
    dets = pd.read_parquet(_p(aoi, "detections.parquet"))
    objs = locate_objects(dets, imgs)
    objs.to_parquet(_p(aoi, "objects.parquet"))
    print(objs.cls.value_counts().to_string() if len(objs) else "no objects")


def diff(aoi):
    from .diff import cell_summary, diff_map

    imgs = pd.read_parquet(_p(aoi, "images.parquet"))
    objs = pd.read_parquet(_p(aoi, "objects.parquet"))
    osm = pd.read_parquet(_p(aoi, "osm.parquet"))
    ch = diff_map(objs, osm, imgs)
    ch.to_parquet(_p(aoi, "changes.parquet"))
    cells = cell_summary(ch)
    cells.to_parquet(_p(aoi, "cells.parquet"))
    print(ch.groupby(["cls", "status"]).size().unstack(fill_value=0).to_string())


def evaluate(aoi):
    from .evaluate import vs_labels, vs_mapillary

    imgs = pd.read_parquet(_p(aoi, "images.parquet"))
    objs = pd.read_parquet(_p(aoi, "objects.parquet"))
    mly = pd.read_parquet(_p(aoi, "mly_features.parquet"))
    out = {"vs_mapillary": vs_mapillary(objs, mly, imgs).to_dict("records")}
    labels = _p(aoi, "labels.csv")
    if labels.exists():
        ch = pd.read_parquet(_p(aoi, "changes.parquet"))
        out["vs_labels"] = vs_labels(ch, pd.read_csv(labels)).to_dict("records")
    _p(aoi, "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    for k, v in out.items():
        print(f"\n== {k}\n{pd.DataFrame(v).to_string(index=False)}")


def main():
    ap = argparse.ArgumentParser(prog="mapdiff")
    ap.add_argument("stage", choices=["fetch", "detect", "locate", "diff", "eval", "all"])
    ap.add_argument("--aoi", default="bandra", choices=sorted(AOIS))
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--conf", type=float, default=0.1)
    a = ap.parse_args()
    aoi = AOIS[a.aoi]
    stages = ["fetch", "detect", "locate", "diff", "eval"] if a.stage == "all" else [a.stage]
    for s in stages:
        print(f"--- {s} ---")
        {"fetch": lambda: fetch(aoi, a.max_images), "detect": lambda: detect(aoi, a.conf),
         "locate": lambda: locate(aoi), "diff": lambda: diff(aoi),
         "eval": lambda: evaluate(aoi)}[s]()
