"""Open-vocabulary detection + multi-frame tracking on Mapillary sequences.

YOLO-World (CLIP text encoder, transformer-based vision-language fusion) detects
classes from text prompts, so no fine-tuning is needed. ByteTrack links the same
physical object across consecutive frames of one capture sequence; each track is
later triangulated from all of its views.
"""

from __future__ import annotations

import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLOWorld
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

from .config import CLASSES


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


# Extra prompts that absorb common street clutter. Without them YOLO-World assigns
# cars, buses and shop signs to the nearest target prompt. Their boxes are dropped.
DISTRACTORS = ["car", "bus", "truck", "auto rickshaw", "motorcycle", "person", "tree",
               "shop sign", "billboard", "building window", "electric pole"]


def load_model(weights: str = "yolov8l-worldv2.pt") -> YOLOWorld:
    m = YOLOWorld(weights)
    m.set_classes([c.prompt for c in CLASSES] + DISTRACTORS)
    return m


def _new_tracker(conf: float) -> BYTETracker:
    cfg = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
    # street furniture is detected at low scores; let low-score boxes start tracks
    cfg.track_high_thresh = max(conf, 0.15)
    cfg.track_low_thresh = conf / 2
    cfg.new_track_thresh = max(conf, 0.15)
    cfg.track_buffer = 3  # sparse frames: a lost object rarely reappears >3 frames later
    return BYTETracker(cfg)


def detect_and_track(images: pd.DataFrame, conf: float = 0.1, weights: str = "yolov8l-worldv2.pt",
                     max_gap_s: float = 10.0) -> pd.DataFrame:
    """Detect every box, then attach ByteTrack ids where a track exists.

    `model.track()` only returns *confirmed* tracks, which silently drops most
    detections when frames are ~3 m apart. So detection and tracking are separate:
    all boxes are kept; the tracker only adds a `track` id (None if untracked).
    The tracker resets on a new sequence or a capture-time gap.
    """
    model = load_model(weights)
    dev = device()
    rows = []
    prev_seq, prev_t, segment, tracker = None, None, 0, None
    for img in tqdm(images.itertuples(), total=len(images), desc="detect+track"):
        if (img.sequence != prev_seq or prev_t is None
                or (img.captured_at - prev_t) / 1000 > max_gap_s):
            segment += 1
            tracker = _new_tracker(conf)
        prev_seq, prev_t = img.sequence, img.captured_at
        res = model.predict(img.path, conf=conf, imgsz=1024, device=dev, verbose=False)[0]
        boxes = res.boxes[res.boxes.cls < len(CLASSES)]  # drop distractor prompts
        h, w = res.orig_shape
        if len(boxes) == 0:
            tracker.update(boxes.cpu().numpy(), res.orig_img)  # age existing tracks
            continue
        tracked = tracker.update(boxes.cpu().numpy(), res.orig_img)
        tid = {int(t[-1]): int(t[4]) for t in tracked}  # detection idx -> track id
        for k, ((x1, y1, x2, y2), c, s) in enumerate(
                zip(boxes.xyxy.tolist(), boxes.cls.int().tolist(), boxes.conf.tolist())):
            rows.append({
                "image_id": img.id,
                "sequence": img.sequence,
                "cls": CLASSES[c].name,
                "score": s,
                "track": f"{segment}:{tid[k]}" if k in tid else None,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "img_w": w, "img_h": h,
            })
    return pd.DataFrame(rows)
