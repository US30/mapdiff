#!/usr/bin/env bash
# Train a 3D Gaussian Splat of the Tanks&Temples "Truck" street scene with Brush
# (Rust + WebGPU, runs on Apple Metal; no CUDA needed).
#
# Fast preset for an 8 GB M-series Mac (~15-25 min): fewer steps, 640 px images,
# capped splat count, densification stops at half-way. Opens the live viewer
# (training curves, current render vs. ground truth). Close the viewer window
# after the final export to finish.
#   bash recon/run_splat.sh            # fast preset with live viewer
#   bash recon/run_splat.sh --help     # all Brush options
set -euo pipefail
cd "$(dirname "$0")"

BRUSH=tools/brush-app-aarch64-apple-darwin/brush_app
SCENE=data/tandt/truck          # COLMAP layout: images/ + sparse/0/
OUT=out
STEPS=7000
mkdir -p "$OUT"

if [[ "${1:-}" == "--help" ]]; then "$BRUSH" --help; exit; fi

"$BRUSH" "$SCENE" \
  --with-viewer \
  --total-steps $STEPS \
  --max-resolution 640 \
  --max-splats 400000 \
  --refine-every 100 \
  --growth-stop-iter 3500 \
  --eval-split-every 8 \
  --eval-every 3500 \
  --export-every $STEPS \
  --export-path "$OUT" \
  --export-name "truck.ply" \
  2>&1 | tee "$OUT/train.log"
