#!/usr/bin/env python3
"""Inventory LeRobot v2.1 sources and enforce the hackathon episode caps.

Example: python scripts/dataset_manifest.py --source sim=datasets/local/so101_sim_pickplace \
    --output logs/dataset-manifest.json
This is an inventory, not proof that Cosmos augmentation or curation occurred.
"""
import argparse
import hashlib
import json
from pathlib import Path


def inventory(sources):
    counts = dict(sim=0, cosmos=0, real=0)
    episodes = []
    roots = set()
    for source in sources:
        kind, location = source.split('=', 1)
        if kind not in counts:
            raise ValueError(f'Unknown source {kind}; use sim, cosmos, or real')
        root = Path(location).resolve()
        if root in roots:
            raise ValueError(f'Duplicate dataset: {root}')
        roots.add(root)
        info = json.loads((root / 'meta/info.json').read_text())
        if info['codebase_version'] != 'v2.1':
            raise ValueError(f'Expected LeRobot v2.1 at {root}')
        rows = [json.loads(line) for line in (root / 'meta/episodes.jsonl').read_text().splitlines() if line.strip()]
        if len(rows) != info['total_episodes']:
            raise ValueError(f'Inconsistent episode count at {root}; retry after collection finishes')
        indices = [row['episode_index'] for row in rows]
        if len(indices) != len(set(indices)):
            raise ValueError(f'Duplicate episode indices at {root}')
        for row in rows:
            idx = row['episode_index']
            chunk = idx // info['chunks_size']
            data = root / info['data_path'].format(episode_chunk=chunk, episode_index=idx)
            videos = [root / info['video_path'].format(episode_chunk=chunk, episode_index=idx, video_key=key)
                      for key, feature in info['features'].items() if feature['dtype'] == 'video']
            for file in [data, *videos]:
                if not file.is_file():
                    raise ValueError(f'Missing episode file: {file}')
            episodes.append(dict(source=kind, dataset=str(root), episode_index=idx,
                                 frames=row['length'], fps=info['fps'], data=str(data),
                                 data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
                                 videos=[str(v) for v in videos]))
        counts[kind] += len(rows)
    if counts['real'] > 50 or sum(counts.values()) > 300:
        raise ValueError(f'Challenge cap exceeded: {counts}; real <= 50, total <= 300')
    return dict(schema_version=1, purpose='source_inventory', counts=counts,
                total_episodes=sum(counts.values()), contains_sim_derived=bool(counts['sim'] + counts['cosmos']),
                episodes=episodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='append', required=True, help='sim=PATH, cosmos=PATH, or real=PATH')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = inventory(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(counts=manifest['counts'], total_episodes=manifest['total_episodes'])))


if __name__ == '__main__':
    main()
