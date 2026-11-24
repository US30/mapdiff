"""Mapillary Graph API v4 client: image metadata, thumbnails, reference map features."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from .config import AOI, mapillary_token

API = "https://graph.mapillary.com"
IMAGE_FIELDS = ",".join(
    [
        "id",
        "geometry",
        "computed_geometry",
        "compass_angle",
        "computed_compass_angle",
        "captured_at",
        "camera_type",
        "camera_parameters",
        "width",
        "height",
        "sequence",
        "thumb_1024_url",
    ]
)
TILE_DEG = 0.004  # Mapillary times out on large bboxes; query in small tiles


def _tiles(bbox, step=TILE_DEG):
    x0, y0, x1, y1 = bbox
    for x in np.arange(x0, x1, step):
        for y in np.arange(y0, y1, step):
            yield (x, y, min(x + step, x1), min(y + step, y1))


def _get(url: str, params: dict, retries: int = 4) -> dict:
    for i in range(retries):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2**i)
            continue
        r.raise_for_status()
    r.raise_for_status()
    return {}


def _paged(endpoint: str, params: dict) -> list[dict]:
    out, url = [], f"{API}/{endpoint}"
    while url:
        js = _get(url, params)
        out.extend(js.get("data", []))
        url = js.get("paging", {}).get("next")
        params = {}  # `next` URL already carries all params
    return out


def fetch_images(aoi: AOI, max_images: int | None = None) -> pd.DataFrame:
    """Image metadata for every Mapillary image inside the AOI."""
    tok = mapillary_token()
    rows = []
    for t in tqdm(list(_tiles(aoi.bbox)), desc="image tiles"):
        rows += _paged(
            "images",
            {
                "access_token": tok,
                "fields": IMAGE_FIELDS,
                "bbox": ",".join(f"{v:.6f}" for v in t),
                "limit": 2000,
            },
        )
    df = pd.json_normalize(rows)
    if df.empty:
        raise SystemExit("No Mapillary images in this AOI.")
    df = df.drop_duplicates("id")
    geom = df["computed_geometry.coordinates"].where(
        df["computed_geometry.coordinates"].notna(), df["geometry.coordinates"]
    )
    df["lon"] = geom.str[0]
    df["lat"] = geom.str[1]
    df["bearing"] = df["computed_compass_angle"].fillna(df["compass_angle"])
    df["focal"] = df["camera_parameters"].apply(
        lambda p: p[0] if isinstance(p, list) and p else np.nan
    )
    df = df.dropna(subset=["lon", "lat", "bearing", "thumb_1024_url"])
    df = df.sort_values(["sequence", "captured_at"])
    if max_images:
        df = df.head(max_images)
    keep = ["id", "sequence", "captured_at", "lon", "lat", "bearing", "camera_type",
            "focal", "width", "height", "thumb_1024_url"]
    return df[keep].reset_index(drop=True)


def download_thumbs(images: pd.DataFrame, out: Path, workers: int = 8) -> pd.DataFrame:
    """Download 1024px thumbnails; returns images with a local `path` column."""
    out.mkdir(parents=True, exist_ok=True)

    def one(row):
        p = out / f"{row.id}.jpg"
        if not p.exists():
            r = requests.get(row.thumb_1024_url, timeout=60)
            if r.status_code != 200:
                return None
            p.write_bytes(r.content)
        return str(p)

    with ThreadPoolExecutor(workers) as ex:
        paths = list(tqdm(ex.map(one, images.itertuples()), total=len(images), desc="thumbs"))
    images = images.assign(path=paths)
    return images.dropna(subset=["path"]).reset_index(drop=True)


def fetch_map_features(aoi: AOI) -> pd.DataFrame:
    """Mapillary's own detected map features: an independent reference for evaluation."""
    tok = mapillary_token()
    rows = []
    for t in tqdm(list(_tiles(aoi.bbox)), desc="feature tiles"):
        rows += _paged(
            "map_features",
            {
                "access_token": tok,
                "fields": "id,object_value,geometry",
                "bbox": ",".join(f"{v:.6f}" for v in t),
                "limit": 2000,
            },
        )
    df = pd.json_normalize(rows)
    if df.empty:
        return pd.DataFrame(columns=["id", "object_value", "lon", "lat"])
    df = df.drop_duplicates("id")
    df["lon"] = df["geometry.coordinates"].str[0]
    df["lat"] = df["geometry.coordinates"].str[1]
    return df[["id", "object_value", "lon", "lat"]].reset_index(drop=True)
