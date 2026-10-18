#!/usr/bin/env python3
"""Convert a single-camera LeRobot v3.0 dataset (e.g. from record_gui.sh) to the v2.1 layout GR00T reads.

    python scripts/v30_to_v21.py ~/so101_real_data/<session> datasets/real_stack --camera front \
        --task "stack the medium, then the big, then the small blue block on the pink pad" --note flip_h=true

Uses the same writer as the sim collector, so sim and real datasets have an identical layout. Frames are
decoded sequentially from the (concatenated) v3.0 video; episode boundaries come from ``meta/episodes``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from so101_pickplace.lerobot_writer import LeRobotV21Writer  # noqa: E402
from so101_pickplace.so101_units import FEATURE_NAMES  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("src", type=Path)
p.add_argument("dst", type=Path)
p.add_argument("--camera", default="front")
p.add_argument("--task", default=None, help="override the task text (default: the one in the dataset)")
p.add_argument("--note", action="append", default=[], help="key=value provenance notes, e.g. flip_h=true")
a = p.parse_args()

info = json.loads((a.src / "meta/info.json").read_text())
assert info["codebase_version"].startswith("v3"), info["codebase_version"]
fps = int(info["fps"])
tasks = pd.read_parquet(a.src / "meta/tasks.parquet")
task = a.task or tasks.index.tolist()[0]
eps = pd.read_parquet(sorted((a.src / "meta/episodes").rglob("*.parquet"))[0]).sort_values("episode_index")
data = pd.concat([pd.read_parquet(f) for f in sorted((a.src / "data").rglob("*.parquet"))]).sort_values("index").reset_index(drop=True)
vids = sorted((a.src / "videos" / f"observation.images.{a.camera}").rglob("*.mp4"))
assert len(vids) == 1, f"expected one concatenated video file, found {len(vids)}"
cap = cv2.VideoCapture(str(vids[0]))
h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(f"{len(eps)} episodes, {len(data)} frames, video {w}x{h} ({int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames), task: {task!r}")
assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(data), "video and data frame counts differ"

writer = LeRobotV21Writer(a.dst, task=task, fps=fps, video_keys=(a.camera,), image_hw=(h, w), feature_names=FEATURE_NAMES)
for _, ep in eps.iterrows():
    lo, hi = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
    writer.start_episode()
    for i in range(lo, hi):
        ok, bgr = cap.read()
        if not ok:
            raise SystemExit(f"video ended early at frame {i}")
        row = data.iloc[i]
        writer.add_frame(np.asarray(row["observation.state"], np.float32), np.asarray(row["action"], np.float32),
                         {a.camera: cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)})
    writer.end_episode()
    print(f"  episode {int(ep['episode_index'])}: {hi - lo} frames", flush=True)
writer.finalize()
notes = dict(n.split("=", 1) for n in a.note)
(a.dst / "meta" / "provenance.json").write_text(json.dumps({"source": str(a.src), "source_format": info["codebase_version"], "camera": a.camera, "task": task, **notes}, indent=2) + "\n")
print("done ->", a.dst)
