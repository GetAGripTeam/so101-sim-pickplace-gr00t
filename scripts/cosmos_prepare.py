#!/usr/bin/env python3
"""Prepare an immutable 93-frame Cosmos smoke clip with explicit action alignment."""
import argparse
import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('source', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--start-frame', type=int, default=90)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
meta = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,r_frame_rate,nb_frames', '-of', 'json', str(a.source)]))['streams'][0]
fps = Fraction(meta['r_frame_rate'])
indices = list(range(a.start_frame, a.start_frame + 93))
if indices[-1] >= int(meta['nb_frames']):
    raise SystemExit('Source does not contain requested 93-frame range')
clip = a.output / 'input.mp4'
subprocess.run(['ffmpeg', '-v', 'error', '-n', '-i', str(a.source), '-vf', f"select=between(n\\,{indices[0]}\\,{indices[-1]}),setpts=N/({fps}*TB)", '-frames:v', '93', '-an', '-c:v', 'libx264', '-crf', '15', '-pix_fmt', 'yuv420p', str(clip)], check=True)
prompt = ('A photorealistic fixed camera recording of a real SO-101 desktop robot arm picking up a small 30 mm cube and placing it on a green pad. '
          'Real laboratory tabletop, natural surface texture, soft indoor lamp illumination and realistic shadows. '
          'Preserve the exact robot geometry, joint positions, gripper, cube and pad locations and all motion in every frame. No camera movement, no extra objects.')
spec = {'name': 'so101_episode000000_front_clip090', 'prompt': prompt, 'video_path': str(clip.resolve()), 'max_frames': 93, 'num_steps': 4, 'seed': 42, 'guidance': 3, 'edge': {'control_weight': 1.0}}
(a.output / 'spec.json').write_text(json.dumps(spec, indent=2) + '\n')
provenance = {'source_video': str(a.source.resolve()), 'source_sha256': hashlib.sha256(a.source.read_bytes()).hexdigest(), 'source_metadata': meta, 'source_frame_indices': indices, 'source_timestamps_seconds': [float(i / fps) for i in indices], 'model': 'nvidia/Cosmos-Transfer2.5-2B edge/distilled', 'checkpoint_revision': 'bbaeedb2b57cc8b14a44653099e2551adb69dcc7', 'seed': 42, 'prompt': prompt, 'input_clip_sha256': hashlib.sha256(clip.read_bytes()).hexdigest(), 'actions_modified': False, 'dataset_integrated': False, 'note': 'Smoke augmentation only. Requires visual verification and paired camera/action mapping before training.'}
(a.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
print(a.output)
