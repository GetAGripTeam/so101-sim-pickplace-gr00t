"""Grab one clean 640x480 frame per camera index (after a warm-up) and report the real fps.

lerobot-find-cameras saves the very first frame, which is often partial/black on macOS. Run this in
a terminal app that already has camera permission:

    .venv/bin/python scripts/camera_check.py            # indices 0 and 1
    .venv/bin/python scripts/camera_check.py 1          # just index 1

Images go to outputs/captured_images/camera_<index>.jpg.
"""
import sys
import time
from pathlib import Path

import cv2

WIDTH, HEIGHT, FPS = 640, 480, 30
WARMUP_S = 2.0
MEASURE_FRAMES = 60

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "captured_images"
out_dir.mkdir(parents=True, exist_ok=True)
indices = [int(a) for a in sys.argv[1:]] or [0, 1]

for idx in indices:
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print(f"camera {idx}: cannot open")
        continue
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    t0 = time.time()
    while time.time() - t0 < WARMUP_S:  # discard frames while exposure settles
        cap.read()
    ok, frame = cap.read()
    if not ok:
        print(f"camera {idx}: no frame after {WARMUP_S:.0f}s warm-up")
        cap.release()
        continue
    t0 = time.time()
    n = sum(cap.read()[0] for _ in range(MEASURE_FRAMES))
    fps = n / (time.time() - t0)
    path = out_dir / f"camera_{idx}.jpg"
    cv2.imwrite(str(path), frame)
    print(f"camera {idx}: {frame.shape[1]}x{frame.shape[0]} @ {fps:.1f} fps -> {path}")
    cap.release()
