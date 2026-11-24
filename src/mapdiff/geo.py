"""Geometry: pixel -> bearing rays, multi-view triangulation, object clustering.

All metric work happens in the local UTM zone (metres). Each detection becomes a
2D ray from the camera position along (camera heading + pixel angle offset).
  * Tracked objects (ByteTrack, same sequence): least-squares intersection of all
    rays of the track (multi-view triangulation).
  * Untracked / single-view rays: pairwise intersections across images.
Candidate points are merged with DBSCAN; a cluster becomes an object only if it
is supported by enough distinct images, which suppresses ghost intersections.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

DEFAULT_FOCAL = 0.85  # normalised focal (f / max(w,h)) when Mapillary has none
SPHERICAL = {"spherical", "equirectangular"}


class LocalFrame:
    """WGS84 <-> UTM (metres) for the zone containing (lon0, lat0)."""

    def __init__(self, lon0: float, lat0: float):
        zone = int((lon0 + 180) // 6) + 1
        epsg = (32600 if lat0 >= 0 else 32700) + zone
        self.fwd = Transformer.from_crs(4326, epsg, always_xy=True)
        self.inv = Transformer.from_crs(epsg, 4326, always_xy=True)

    def to_xy(self, lon, lat):
        return np.column_stack(self.fwd.transform(np.asarray(lon), np.asarray(lat)))

    def to_lonlat(self, xy):
        xy = np.atleast_2d(xy)
        return np.column_stack(self.inv.transform(xy[:, 0], xy[:, 1]))


def pixel_bearing(u, img_w, img_h, cam_bearing, camera_type, focal):
    """Absolute compass bearing (deg) of image column u."""
    u, img_w, img_h = map(np.asarray, (u, img_w, img_h))
    cam_bearing = np.asarray(cam_bearing, float)
    sph = np.isin(np.asarray(camera_type, dtype=object), list(SPHERICAL))
    f = np.where(np.isnan(np.asarray(focal, float)), DEFAULT_FOCAL, focal)
    f_px = f * np.maximum(img_w, img_h)
    persp = np.degrees(np.arctan((u - img_w / 2) / f_px))
    equi = (u / img_w - 0.5) * 360.0
    return (cam_bearing + np.where(sph, equi, persp)) % 360.0


def half_fov(img_w, img_h, camera_type, focal):
    if camera_type in SPHERICAL:
        return 180.0
    f = DEFAULT_FOCAL if focal is None or np.isnan(focal) else focal
    return float(np.degrees(np.arctan((img_w / 2) / (f * max(img_w, img_h)))))


def unit(bearing_deg):
    b = np.radians(bearing_deg)
    return np.column_stack([np.sin(b), np.cos(b)])  # (east, north)


def triangulate(origins: np.ndarray, dirs: np.ndarray, min_depth=1.0, max_depth=50.0,
                min_spread_deg=4.0, max_rms=3.0):
    """Least-squares point closest to all rays. Returns (point, rms) or None."""
    if len(origins) < 2:
        return None
    ang = np.degrees(np.arctan2(dirs[:, 0], dirs[:, 1]))
    spread = np.ptp(np.unwrap(np.radians(ang)))
    if np.degrees(spread) < min_spread_deg:
        return None  # rays near-parallel: depth unobservable
    A = np.zeros((2, 2))
    b = np.zeros(2)
    for o, d in zip(origins, dirs):
        P = np.eye(2) - np.outer(d, d)
        A += P
        b += P @ o
    try:
        p = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    depth = np.einsum("ij,ij->i", p - origins, dirs)
    if (depth < min_depth).any() or (depth > max_depth).any():
        return None  # behind a camera or implausibly far
    perp = (p - origins) - depth[:, None] * dirs
    rms = float(np.sqrt((perp**2).sum(1).mean()))
    if rms > max_rms:
        return None
    return p, rms


def build_rays(dets: pd.DataFrame, images: pd.DataFrame, frame: LocalFrame) -> pd.DataFrame:
    im = images.set_index("id")
    d = dets.join(im[["lon", "lat", "bearing", "camera_type", "focal"]], on="image_id")
    u = (d.x1 + d.x2) / 2
    d["ray_bearing"] = pixel_bearing(u, d.img_w, d.img_h, d.bearing, d.camera_type, d.focal)
    xy = frame.to_xy(d.lon.values, d.lat.values)
    d["ox"], d["oy"] = xy[:, 0], xy[:, 1]
    return d.reset_index(drop=True)


def _candidates_from_tracks(rays: pd.DataFrame):
    out, used = [], set()
    for (cls, tr), g in rays.dropna(subset=["track"]).groupby(["cls", "track"]):
        g = g.drop_duplicates("image_id")
        if len(g) < 2:
            continue
        res = triangulate(g[["ox", "oy"]].values, unit(g.ray_bearing.values))
        if res is None:
            continue
        p, rms = res
        out.append({"cls": cls, "x": p[0], "y": p[1], "rms": rms,
                    "images": frozenset(g.image_id), "score": g.score.mean(), "src": "track"})
        used.update(g.index)
    return out, used


def _candidates_from_pairs(rays: pd.DataFrame, max_baseline=40.0, min_angle=6.0):
    out = []
    for cls, g in rays.groupby("cls"):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue
        O = g[["ox", "oy"]].values
        D = unit(g.ray_bearing.values)
        img = g.image_id.values
        for i, j in cKDTree(O).query_pairs(max_baseline):
            if img[i] == img[j]:
                continue
            cross = D[i, 0] * D[j, 1] - D[i, 1] * D[j, 0]
            if abs(np.degrees(np.arcsin(np.clip(cross, -1, 1)))) < min_angle:
                continue
            res = triangulate(O[[i, j]], D[[i, j]], max_rms=0.5)
            if res is None:
                continue
            p, rms = res
            out.append({"cls": cls, "x": p[0], "y": p[1], "rms": rms,
                        "images": frozenset((img[i], img[j])),
                        "score": (g.score[i] + g.score[j]) / 2, "src": "pair"})
    return out


def locate_objects(dets: pd.DataFrame, images: pd.DataFrame, eps=3.0, min_images=3) -> pd.DataFrame:
    """Detections -> georeferenced object list."""
    cols = ["obj_id", "cls", "lon", "lat", "n_images", "n_track_pts", "rms", "score", "image_ids"]
    if dets.empty:
        return pd.DataFrame(columns=cols)
    frame = LocalFrame(images.lon.mean(), images.lat.mean())
    rays = build_rays(dets, images, frame)
    track_c, used = _candidates_from_tracks(rays)
    pair_c = _candidates_from_pairs(rays.drop(index=list(used)))
    cand = pd.DataFrame(track_c + pair_c)
    if cand.empty:
        return pd.DataFrame(columns=cols)
    objs = []
    for cls, g in cand.groupby("cls"):
        labels = DBSCAN(eps=eps, min_samples=1).fit_predict(g[["x", "y"]].values)
        for lab in np.unique(labels):
            c = g[labels == lab]
            ims = frozenset().union(*c.images)
            if len(ims) < min_images:
                continue
            w = 1.0 / (c.rms.values + 0.5)  # trust tight intersections more
            xy = np.average(c[["x", "y"]].values, axis=0, weights=w)
            lon, lat = frame.to_lonlat(xy)[0]
            objs.append({"cls": cls, "lon": lon, "lat": lat, "n_images": len(ims),
                         "n_track_pts": int((c.src == "track").sum()),
                         "rms": float(np.average(c.rms, weights=w)),
                         "score": float(c.score.mean()), "image_ids": sorted(ims)})
    df = pd.DataFrame(objs)
    if df.empty:
        return pd.DataFrame(columns=cols)
    df.insert(0, "obj_id", range(len(df)))
    return df[cols]


def coverage(points: pd.DataFrame, images: pd.DataFrame, radius=25.0) -> np.ndarray:
    """Number of images that should have seen each point (within radius and in FOV)."""
    if points.empty:
        return np.zeros(0, int)
    frame = LocalFrame(images.lon.mean(), images.lat.mean())
    P = frame.to_xy(points.lon.values, points.lat.values)
    C = frame.to_xy(images.lon.values, images.lat.values)
    hf = np.array([
        half_fov(r.width if pd.notna(r.width) else 1024,
                 r.height if pd.notna(r.height) else 768, r.camera_type, r.focal)
        for r in images.itertuples()
    ])
    tree = cKDTree(C)
    counts = np.zeros(len(P), int)
    for k, idx in enumerate(tree.query_ball_point(P, radius)):
        if not idx:
            continue
        v = P[k] - C[idx]
        brg = np.degrees(np.arctan2(v[:, 0], v[:, 1])) % 360
        diff = np.abs((brg - images.bearing.values[idx] + 180) % 360 - 180)
        counts[k] = int((diff <= hf[idx]).sum())
    return counts
