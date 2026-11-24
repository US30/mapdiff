"""Streamlit viewer: 3D H3 change map, per-object evidence, manual labelling.

Run:  uv run streamlit run app.py
"""

import json

import pandas as pd
import pydeck as pdk
import streamlit as st
from PIL import Image, ImageDraw

from mapdiff.config import AOIS

st.set_page_config(page_title="MapDiff", layout="wide")
aoi = AOIS[st.sidebar.selectbox("Area", sorted(AOIS))]
d = aoi.dir


@st.cache_data
def load(name):
    p = d / name
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


changes, cells, images, dets = (load(n) for n in
                                ("changes.parquet", "cells.parquet", "images.parquet",
                                 "detections.parquet"))
if changes.empty:
    st.warning("No results yet. Run `uv run mapdiff all --aoi %s` first." % aoi.name)
    st.stop()

COLORS = {"matched": [46, 160, 67], "new": [31, 119, 255], "missing": [220, 50, 47],
          "unseen": [150, 150, 150]}
tab_map, tab_label, tab_metrics = st.tabs(["Map", "Label", "Metrics"])

with tab_map:
    classes = st.multiselect("Classes", sorted(changes.cls.unique()), sorted(changes.cls.unique()))
    statuses = st.multiselect("Status", list(COLORS), ["matched", "new", "missing"])
    ch = changes[changes.cls.isin(classes) & changes.status.isin(statuses)].copy()
    ch["color"] = ch.status.map(COLORS)
    c = cells.copy()
    c["elev"] = c.anomaly * 120
    c["fill"] = [[255, int(220 * (1 - a)), 0, 90] for a in c.anomaly]
    layers = [
        pdk.Layer("ScatterplotLayer", images, get_position=["lon", "lat"], get_radius=1,
                  get_fill_color=[120, 120, 120, 60]),
        pdk.Layer("PolygonLayer", c, get_polygon="boundary", get_elevation="elev",
                  extruded=True, get_fill_color="fill", pickable=True, auto_highlight=True),
        pdk.Layer("ScatterplotLayer", ch, get_position=["lon", "lat"], get_radius=4,
                  get_fill_color="color", pickable=True),
    ]
    view = pdk.ViewState(latitude=images.lat.mean(), longitude=images.lon.mean(), zoom=15.5,
                         pitch=50)
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, map_style="light",
                             tooltip={"text": "{cls} {status}\nH3 {h3}\nanomaly {anomaly}"}))
    st.caption("Points: green matched, blue new (not in OSM), red missing (in OSM, visible in "
               "imagery, not detected). Hex height = anomaly score.")
    st.dataframe(cells.drop(columns="boundary").head(20), hide_index=True)

with tab_label:
    lab_path = d / "labels.csv"
    labels = pd.read_csv(lab_path) if lab_path.exists() else pd.DataFrame(
        columns=["obj_id", "verdict"])
    objs = load("objects.parquet")
    status = st.selectbox("Label objects flagged as", ["new", "matched"])
    pool = changes[(changes.status == status)].dropna(subset=["obj_id"])
    todo = pool[~pool.obj_id.astype(int).isin(labels.obj_id.astype(int))]
    # random, class-mixed order so labels are an unbiased sample (fixed seed: stable queue)
    todo = todo.sample(frac=1, random_state=0)
    st.write(f"{len(labels)} labelled, {len(todo)} remaining in '{status}'")
    if not todo.empty:
        row = todo.iloc[0]
        oid = int(row.obj_id)
        st.subheader(f"Object {oid}: {row.cls} ({row.status})")
        ids = list(objs.loc[objs.obj_id == oid, "image_ids"].iloc[0])[:4]
        cols = st.columns(len(ids))
        im = images.set_index("id")
        for col, iid in zip(cols, ids):
            img = Image.open(im.loc[iid, "path"]).convert("RGB")
            dr = ImageDraw.Draw(img)
            for b in dets[(dets.image_id == iid) & (dets.cls == row.cls)].itertuples():
                dr.rectangle([b.x1, b.y1, b.x2, b.y2], outline=(255, 0, 0), width=4)
            col.image(img, width="stretch")
        b1, b2, b3 = st.columns(3)
        for btn, verdict in ((b1, "correct"), (b2, "wrong"), (b3, "unsure")):
            if btn.button(verdict, width="stretch"):
                labels = pd.concat([labels, pd.DataFrame([{"obj_id": oid, "verdict": verdict}])])
                labels.to_csv(lab_path, index=False)
                st.rerun()

with tab_metrics:
    m = d / "metrics.json"
    if m.exists():
        for k, v in json.loads(m.read_text()).items():
            st.subheader(k)
            st.dataframe(pd.DataFrame(v), hide_index=True)
    else:
        st.info("Run `uv run mapdiff eval` to compute metrics.")
