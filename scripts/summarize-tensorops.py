#!/usr/bin/env python3
"""Summarize paired, uncached development measurements (not leaderboard results)."""
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
records = [json.loads(line) for line in (root / 'compare.jsonl').read_text().splitlines() if line.strip()]
fixture = next(r for r in records if r['type'] == 'fixture')
correctness = next(r for r in records if r['type'] == 'model_correctness')
summary = {'kind': 'paired development diagnostic; same tiles, strict FP32', 'fixture': fixture, 'correctness': correctness}
for enabled, label in [(False, 'legacy'), (True, 'tensorops')]:
    warm = [r for r in records if r.get('mode') == 'warm_uncached' and r['fp32_tensorops'] == enabled]
    sampled = next(r for r in records if r.get('mode') == 'all_kernel_encoder_counters' and r['fp32_tensorops'] == enabled)
    medians = {key: statistics.median(r[key] for r in warm) for key in ('wall_ms', 'tokens_per_second', 'process_cpu_ms')}
    for key, value in warm[0]['timing'].items():
        if isinstance(value, (int, float)):
            medians[key] = statistics.median(r['timing'][key] for r in warm)
    kernel_times = {r['kernel']: r['gpu_ms'] for r in sampled['timing']['kernels']}
    summary[label] = {'warm_runs': len(warm), 'median': medians,
        'wall_range_ms': [min(r['wall_ms'] for r in warm), max(r['wall_ms'] for r in warm)],
        'sampled_wall_ms': sampled['wall_ms'], 'sampled_gpu_ms': sampled['timing']['gpu_ms'],
        'sampled_overhead_pct': (sampled['wall_ms'] / medians['wall_ms'] - 1) * 100,
        'sampled_kernel_ms': kernel_times}
summary['wall_speedup'] = summary['legacy']['median']['wall_ms'] / summary['tensorops']['median']['wall_ms']
summary['wall_reduction_pct'] = (1 - 1 / summary['wall_speedup']) * 100
summary['paired_speedups'] = [
    next(r['wall_ms'] for r in records if r.get('mode') == 'warm_uncached' and r['run'] == run and not r['fp32_tensorops']) /
    next(r['wall_ms'] for r in records if r.get('mode') == 'warm_uncached' and r['run'] == run and r['fp32_tensorops'])
    for run in sorted({r['run'] for r in records if r.get('mode') == 'warm_uncached'})]
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
