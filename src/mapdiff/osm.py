"""Reference map: point features from OpenStreetMap for each object class.

Uses a raw Overpass query (one request for all classes) with mirror rotation,
because the main public Overpass endpoint is frequently at capacity.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

from .config import AOI

ENDPOINTS = [
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
HEADERS = {"User-Agent": "mapdiff/0.1 (research prototype)"}
HIGHWAY_TO_CLASS = {
    "traffic_signals": "traffic_light",
    "street_lamp": "street_light",
    "stop": "traffic_sign",
    "give_way": "traffic_sign",
}


def _ql(bbox) -> str:
    w, s, e, n = bbox
    b = f"({s},{w},{n},{e})"
    hw = "|".join(HIGHWAY_TO_CLASS)
    return (
        "[out:json][timeout:90];("
        f'nwr["highway"~"^({hw})$"]{b};'
        f'nwr["traffic_sign"]{b};'
        ");out center tags;"
    )


def _query(bbox) -> list[dict]:
    last = None
    for attempt in range(6):
        url = ENDPOINTS[attempt % len(ENDPOINTS)]
        try:
            r = requests.post(url, data={"data": _ql(bbox)}, headers=HEADERS, timeout=120)
            if r.status_code == 200:
                return r.json()["elements"]
            last = f"HTTP {r.status_code} from {url}"
        except (requests.RequestException, ValueError) as e:
            last = f"{type(e).__name__} from {url}"
        time.sleep(min(30, 3 * 2**attempt))
    raise RuntimeError(f"Overpass failed after retries: {last}")


def _classify(tags: dict) -> str | None:
    hw = tags.get("highway")
    if hw in HIGHWAY_TO_CLASS:
        return HIGHWAY_TO_CLASS[hw]
    if tags.get("traffic_sign"):
        return "traffic_sign"
    return None


def fetch_osm(aoi: AOI) -> pd.DataFrame:
    rows = []
    for el in _query(aoi.bbox):
        cls = _classify(el.get("tags", {}))
        lat = el.get("lat", el.get("center", {}).get("lat"))
        lon = el.get("lon", el.get("center", {}).get("lon"))
        if cls and lat is not None:
            rows.append({"osm_id": f"{el['type']}/{el['id']}", "cls": cls, "lon": lon, "lat": lat})
    return pd.DataFrame(rows, columns=["osm_id", "cls", "lon", "lat"]).drop_duplicates(
        ["osm_id", "cls"]
    )
