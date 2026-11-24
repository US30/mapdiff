"""Project-wide settings: area of interest, object classes, paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


@dataclass(frozen=True)
class AOI:
    name: str
    # (min_lon, min_lat, max_lon, max_lat)
    bbox: tuple[float, float, float, float]

    @property
    def dir(self) -> Path:
        d = DATA / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d


AOIS = {
    # ~1.5 km x 1.5 km around Hill Road / Linking Road, Bandra West, Mumbai
    "bandra": AOI("bandra", (72.826, 19.052, 72.840, 19.066)),
    # ~1.2 km x 1.2 km around Hiranandani Gardens, Powai, Mumbai
    "powai": AOI("powai", (72.900, 19.113, 72.912, 19.124)),
}


@dataclass(frozen=True)
class ObjClass:
    name: str  # canonical class name used everywhere in the pipeline
    prompt: str  # open-vocabulary text prompt for YOLO-World
    osm_tags: dict  # osmnx tag filter selecting this class in OpenStreetMap
    mly_prefixes: tuple[str, ...]  # Mapillary map_feature object_value prefixes


CLASSES = [
    ObjClass(
        "traffic_light",
        "traffic light",
        {"highway": "traffic_signals"},
        ("object--traffic-light",),
    ),
    ObjClass(
        "street_light",
        "street light",
        {"highway": "street_lamp"},
        ("object--street-light",),
    ),
    ObjClass(
        "traffic_sign",
        "traffic sign",
        {"traffic_sign": True, "highway": ["stop", "give_way"]},
        ("object--traffic-sign", "regulatory--", "warning--", "information--", "complementary--"),
    ),
]
CLASS_BY_NAME = {c.name: c for c in CLASSES}


def mapillary_token() -> str:
    tok = os.environ.get("MAPILLARY_TOKEN", "").strip()
    if not tok:
        raise SystemExit(
            "MAPILLARY_TOKEN missing. Create one at "
            "https://www.mapillary.com/dashboard/developers and put it in .env"
        )
    return tok
