#!/usr/bin/env python3
"""Summarize local MoE development JSONL without changing formal reports."""
import argparse
import collections
import json
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=Path)
args = parser.parse_args()
summary = {}
for name in ['projections-2', 'projections-kstep', 'fusion-1', 'fusion-kstep', 'routing-1', 'model-10k', 'model-30k', 'model-final-10k']:
    path = args.directory / (name + '.jsonl')
    if not path.exists():
        continue
    groups = collections.defaultdict(list)
    checks = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        kind = row.get('type', '')
        if kind == 'prefill_variant':
            if row['run'] == 0:
                checks.append({k: row.get(k) for k in ['variant', 'answer_correct', 'text', 'first_token']})
                continue
            key = row['variant']
        elif kind == 'fusion_benchmark':
            key = f"{row['variant']}/tokens={row['m']}/skew={row['skew']}"
        elif kind == 'expert_benchmark' and row['run'] > 0:
            key = f"{row['name']}/K={row['k']}/N={row['n']}/skew={row['skew']}"
        elif kind == 'routing_benchmark' and row['run'] > 0:
            key = f"parallel={row['parallel']}/tokens={row['m']}/tile={row['tileRows']}/skew={row['skew']}"
        else:
            if kind.endswith('_check'):
                checks.append(row)
            continue
        groups[key].append(row)
    timings = {}
    for key, rows in groups.items():
        wall = [x['wall_ms'] for x in rows]
        gpu = [x['timing']['gpu_ms'] for x in rows]
        timings[key] = dict(runs=len(rows), wall_ms_median=statistics.median(wall),
                            wall_ms_min=min(wall), wall_ms_max=max(wall), gpu_ms_median=statistics.median(gpu))
        if 'prompt_tokens' in rows[0]:
            timings[key]['tokens_per_second'] = rows[0]['prompt_tokens'] / (statistics.median(wall) / 1000)
    summary[name] = dict(timings=timings, checks=checks)
print(json.dumps(summary, indent=2))
