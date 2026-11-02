#!/usr/bin/env python3
"""Summarize the two warmed, reversed-order kernel benchmark passes (b and c)."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=Path)
args = parser.parse_args()
groups = collections.defaultdict(list)
passes = collections.defaultdict(lambda: collections.defaultdict(list))
files = []
for engine in ['ours', 'mlx']:
    for pass_id in ['b', 'c']:
        path = args.directory / f'{engine}-{pass_id}.jsonl'
        files.append(path)
        for line in path.read_text().splitlines():
            d = json.loads(line)
            kind = d['type']
            if kind == 'dense_benchmark':
                key = ('dense', f"{d['m']}x{d['k']}x{d['n']}", 'packed' if d['include_packing'] else 'core')
            elif kind == 'expert_benchmark' and d['run'] > 0:
                key = ('experts', f"{d['k']}x{d['n']}-{'skew' if d['skew'] else 'uniform'}", d['name'])
            elif kind == 'fast_attention_microbenchmark':
                key = ('attention', f"{d['history']}+{d['rows']}", 'sdpa')
            elif kind == 'mlx_benchmark':
                family = d['family']
                shape = (f"{d['m']}x{d['k']}x{d['n']}" if family == 'dense' else
                         f"{d['k']}x{d['n']}-{'skew' if d['skew'] else 'uniform'}" if family == 'experts' else
                         f"{d['history']}+{d['rows']}")
                key = (family, shape, d['name'])
            else:
                continue
            groups[(engine, *key)].append(d['wall_ms'])
            passes[(engine, *key)][pass_id].append(d['wall_ms'])
summary = []
for key, values in groups.items():
    if set(passes[key]) != {'b', 'c'}:
        raise ValueError(f'Missing a comparison pass: {key}')
    row = dict(zip(['engine', 'family', 'shape', 'variant'], key))
    row.update(samples=len(values), median_ms=statistics.median(values), min_ms=min(values), max_ms=max(values),
               pass_medians_ms={p: statistics.median(v) for p, v in passes[key].items()})
    summary.append(row)
    print(f"{' / '.join(key)}: {row['median_ms']:.3f} ms ({len(values)} samples)")
output = args.directory / 'summary.json'
output.write_text(json.dumps(summary, indent=2) + '\n')
files.append(output)
(args.directory / 'results-sha256.json').write_text(json.dumps(
    {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, indent=2) + '\n')
