"""Synthetic-scene tests: known objects, simulated cameras, exact detections."""

import numpy as np
import pandas as pd

from mapdiff.geo import LocalFrame, coverage, locate_objects, pixel_bearing, triangulate, unit

LON0, LAT0 = 72.833, 19.059
W, H, F = 1024, 768, 0.85


def _scene(objects_xy, noise_px=0.0, seed=0):
    """Cameras drive north along x=0 every 3 m, facing north-east (30 deg)."""
    rng = np.random.default_rng(seed)
    fr = LocalFrame(LON0, LAT0)
    base = fr.to_xy([LON0], [LAT0])[0]
    cams = np.array([[0.0, y] for y in np.arange(0, 60, 3.0)])
    ll = fr.to_lonlat(cams + base)
    images = pd.DataFrame({
        "id": [f"im{i}" for i in range(len(cams))], "sequence": "s1",
        "captured_at": np.arange(len(cams)) * 1000, "lon": ll[:, 0], "lat": ll[:, 1],
        "bearing": 30.0, "camera_type": "perspective", "focal": F, "width": W, "height": H,
    })
    f_px = F * max(W, H)
    dets = []
    for oi, o in enumerate(objects_xy):
        for ci, c in enumerate(cams):
            v = np.asarray(o) - c
            dist = np.hypot(*v)
            rel = (np.degrees(np.arctan2(v[0], v[1])) - 30 + 180) % 360 - 180
            if dist > 40 or abs(rel) > 30:
                continue
            u = W / 2 + f_px * np.tan(np.radians(rel)) + rng.normal(0, noise_px)
            dets.append({"image_id": f"im{ci}", "sequence": "s1", "cls": "traffic_light",
                         "score": 0.9, "track": f"0:{oi}", "x1": u - 5, "y1": 100,
                         "x2": u + 5, "y2": 140, "img_w": W, "img_h": H})
    truth = fr.to_lonlat(np.asarray(objects_xy, float).reshape(-1, 2) + base)
    return images, pd.DataFrame(dets), truth, fr


def test_pixel_bearing_center_and_edge():
    assert np.isclose(pixel_bearing(W / 2, W, H, 90, "perspective", F), 90)
    assert np.isclose(pixel_bearing(0, 1000, 500, 0, "spherical", np.nan), 180)


def test_triangulate_two_rays():
    o = np.array([[0, 0], [10, 0]], float)
    d = np.vstack([unit(45), unit(-45)])
    p, rms = triangulate(o, d)
    assert np.allclose(p, [5, 5]) and rms < 1e-9


def test_triangulate_rejects_behind_camera():
    o = np.array([[0, 0], [10, 0]], float)
    d = np.vstack([unit(225), unit(135)])  # lines cross at (5,5), behind both cameras
    assert triangulate(o, d) is None


def _err_m(fr, a, b):
    return np.hypot(*(fr.to_xy([a[0]], [a[1]])[0] - fr.to_xy([b[0]], [b[1]])[0]))


def test_locate_objects_exact():
    images, dets, truth, fr = _scene([[12, 30], [20, 45]])
    objs = locate_objects(dets, images)
    assert len(objs) == 2
    for t in truth:
        e = min(_err_m(fr, t, (r.lon, r.lat)) for r in objs.itertuples())
        assert e < 0.5


def test_locate_objects_noisy_untracked():
    images, dets, truth, fr = _scene([[12, 30]], noise_px=2.0)
    dets["track"] = None  # force pairwise path
    objs = locate_objects(dets, images)
    assert len(objs) >= 1
    e = min(_err_m(fr, truth[0], (r.lon, r.lat)) for r in objs.itertuples())
    assert e < 3.0


def test_coverage_counts_only_in_fov():
    images, _, _, fr = _scene([])
    pts = pd.DataFrame({"lon": [images.lon[5]], "lat": [images.lat[5]]})
    base = fr.to_xy(pts.lon, pts.lat)[0]
    ahead = fr.to_lonlat(base + [5, 10])[0]   # north-east of camera 5: in view
    behind = fr.to_lonlat(base + [0, -20])[0]  # south: only seen from cameras south of it
    c = coverage(pd.DataFrame({"lon": [ahead[0], behind[0]], "lat": [ahead[1], behind[1]]}), images)
    assert c[0] > 0
    assert c[0] > c[1]


def test_diff_map_statuses():
    from mapdiff.diff import cell_summary, diff_map

    images, dets, truth, fr = _scene([[12, 30], [20, 45]])
    objs = locate_objects(dets, images)
    base = fr.to_xy([LON0], [LAT0])[0]
    in_view = fr.to_lonlat(base + [8, 20])[0]  # in view of many cameras, never detected
    osm = pd.DataFrame({
        "osm_id": ["n/1", "n/2"], "cls": ["traffic_light"] * 2,
        "lon": [truth[0][0] + 3e-5, in_view[0]], "lat": [truth[0][1], in_view[1]],
    })
    ch = diff_map(objs, osm, images)
    assert sorted(ch.status) == ["matched", "missing", "new"]
    cells = cell_summary(ch)
    assert cells.anomaly.between(0, 1).all()
