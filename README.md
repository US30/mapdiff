# MapDiff: Street-Level Map Change Detection + 3D Reconstruction

Finds map features that OpenStreetMap is missing (or that no longer exist) from
crowd-sourced street-level imagery, and reconstructs a street scene in 3D.

```
Mapillary images ─► YOLO-World (open-vocab) + ByteTrack ─► pixel→bearing rays
   (geotagged)        detection / tracking                    │
                                                              ▼
OpenStreetMap ─► Hungarian matching ◄─ DBSCAN merge ◄─ multi-view triangulation (UTM)
                     │
                     ▼
        matched / new / missing per class ─► H3 cells ─► anomaly score ─► 3D map (pydeck)
```

## Pipeline

| Stage | What it does | Code |
|---|---|---|
| fetch | Mapillary Graph API v4 image metadata + thumbnails (tiled bbox queries), OSM features via Overpass, Mapillary's own map features for evaluation | `mapillary.py`, `osm.py` |
| detect | YOLO-World v2-L (CLIP text encoder) with text prompts: traffic light, street light, traffic sign, plus 11 distractor prompts (car, bus, shop sign, ...) whose boxes are discarded. Every box is kept; ByteTrack only attaches ids. Runs on Apple MPS (~0.17 s/img). | `detect.py` |
| locate | Each box centre becomes a compass bearing (camera heading + pixel angle; perspective and 360° cameras). Tracks are triangulated by least-squares ray intersection; leftover rays by pairwise intersection. DBSCAN merges candidates; an object needs ≥3 distinct images. | `geo.py` |
| diff | Optimal 1:1 matching (Hungarian, 12 m gate) against OSM. Unmatched detections are **new**. Unmatched OSM features are **missing** only when ≥3 images had them in field of view (coverage check), otherwise **unseen**. | `diff.py` |
| index | Changes aggregated on H3 res-11 hexagons with an anomaly score `(new+missing)/(evidence+1)` | `diff.py` |
| eval | Precision/recall and localisation error against Mapillary's independent map features; precision of change flags from manual labels | `evaluate.py` |
| 3D | Gaussian Splatting (Brush, Metal backend) on Tanks&Temples *Truck*, splat → cleaned point cloud → Poisson mesh (Open3D) | `recon/` |

## Run

```bash
uv sync
cp .env.example .env            # add your Mapillary token
uv run mapdiff all --aoi bandra --max-images 3000
uv run streamlit run app.py     # 3D map, labelling, metrics
uv run pytest -p no:dash        # synthetic-scene geometry tests
```

Area presets live in `src/mapdiff/config.py` (Bandra West and Powai, Mumbai).

## 3D reconstruction

```bash
bash recon/run_splat.sh          # trains a Gaussian Splat on the Truck scene, exports .ply
uv run python recon/splat_to_mesh.py recon/out/truck.ply --out recon/out
```

## Results

### Map features: Bandra West, Mumbai (~2.2 km², 3,000 Mapillary images, 188 sequences)

5,068 detections became 729 georeferenced objects. They are scored against Mapillary's own
map features (an independent detector and SfM pipeline). Only reference features that our
images actually saw count toward recall (at least 3 images with the feature in view).
Match gate: 10 m.

Locate parameters were tuned on the **west half** of the area. Held-out **east half**:

| class | ours | reference | precision | recall | median error |
|---|---|---|---|---|---|
| traffic light | 27 | 21 | 0.30 | 0.38 | 4.2 m |
| street light | 222 | 78 | 0.17 | 0.49 | 6.3 m |
| traffic sign | 130 | 62 | 0.26 | 0.55 | 3.6 m |

Full area: recall 0.32 / 0.55 / 0.44, median localisation error 4.1 / 5.1 / 3.4 m.

Precision here is a lower bound, because Mapillary's feature layer is incomplete.
True precision comes from manual labels (Streamlit *Label* tab, then `mapdiff eval`).

**Manual labels:** 109 of 119 "new" street-light objects were correct, so **precision is 0.92**
(95% Wilson CI about 0.85–0.95). Each object was judged on its own, so this does not
measure double counting. Traffic lights and signs are not yet labelled.

**Map diff vs OSM:** 13 OSM traffic signals confirmed. 48 traffic lights, 429 street
lights and 239 signs are candidate OSM additions (OSM has almost no street lamps or signs
mapped here). 7 traffic signals were in view of imagery but never detected.

### Lessons learned
- `model.track()` returns only *confirmed* tracks. With street-level frames about 3 m apart,
  this silently dropped 95% of detections (3 vs 62 on one sequence). The fix was to
  decouple detection from tracking.
- At about 3 m frame spacing, boxes rarely overlap between frames, so ByteTrack tracks
  last about 1 frame. Matching sightings of the same object comes from geometric
  (multi-view ray) association, not from tracking.
- Open-vocabulary detectors need distractor prompts. With only target prompts, buses and
  shop signs were labelled "bus stop" 2,329 times.

### 3D reconstruction (Tanks&Temples Truck, Apple M3)

| metric | value |
|---|---|
| training | 7,000 steps, 5 min 59 s, 20.4 steps/s, 661 MB GPU |
| held-out views (every 8th) | **26.27 PSNR / 0.912 SSIM** at 640 px |
| splats | 400k → 145k confident (opacity > 0.2) → 49k cleaned points |
| mesh (scene / truck crop) | 432k / 186k triangles, screened Poisson depth 10 / 9 |

PSNR and SSIM are at 640 px, so they are not comparable to full-resolution 3DGS paper numbers.
