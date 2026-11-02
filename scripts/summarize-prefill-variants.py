#!/usr/bin/env python3
"""Summarize precision/attention diagnostics without mixing quality/cold timings."""
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
records = [json.loads(s) for s in (root / 'compare.jsonl').read_text().splitlines() if s.strip()]
fixture = next(r for r in records if r['type'] == 'fixture')
result = {'kind': 'development experiment, not an official evaluation', 'fixture': fixture, 'variants': {}}
for name in fixture['variants']:
    warm = [r for r in records if r.get('variant') == name and r.get('mode') == 'warm_uncached']
    if len(warm) != fixture['runs']:
        raise ValueError(f'Incomplete warm measurements for {name}: {len(warm)}')
    quality = next(r for r in records if r.get('variant') == name and r.get('mode') == 'cold_full_answer')
    smoke = next(r for r in records if r.get('variant') == name and r['type'] == 'model_smoke')
    medians = {key: statistics.median(r[key] for r in warm) for key in ('wall_ms', 'process_cpu_ms', 'prefill_tokens_per_second')}
    medians.update({key: statistics.median(r['timing'][key] for r in warm)
        for key in ('gpu_ms', 'host_encode_ms', 'host_commit_ms', 'host_wait_ms', 'command_buffers', 'dispatches')})
    result['variants'][name] = {'warm_runs': len(warm), 'median': medians,
        'wall_range_ms': [min(r['wall_ms'] for r in warm), max(r['wall_ms'] for r in warm)],
        'smoke': smoke, 'long_answer': quality['text'], 'long_answer_correct': quality['answer_correct'],
        'first_tokens_match_baseline': all(r['first_token_matches_baseline'] for r in warm)}
base = result['variants']['legacy']['median']['wall_ms']
for value in result['variants'].values():
    value['speedup_vs_legacy'] = base / value['median']['wall_ms']
result['attention_microbenchmarks'] = []
a = [json.loads(s) for s in (root / 'attention-legacy.jsonl').read_text().splitlines()]
b = [json.loads(s) for s in (root / 'attention-tiled.jsonl').read_text().splitlines()]
for history in sorted({r['history'] for r in a}):
    for phase in ('combined', 'qk', 'softmax', 'av'):
        legacy = statistics.median(r['timing']['gpu_ms'] for r in a if r['history'] == history and r['phase'] == phase)
        tiled = statistics.median(r['timing']['gpu_ms'] for r in b if r['history'] == history and r['phase'] == phase)
        result['attention_microbenchmarks'].append({'history': history, 'phase': phase,
            'legacy_gpu_ms': legacy, 'tiled_gpu_ms': tiled, 'speedup': legacy / tiled})
result['limitations'] = [
    'Precision tests compare kernels to CPU references after explicitly rounding operands. They do not establish full model quality.',
    'Model quality checks cover one 626-token arithmetic prompt and the full answer to one 10520-token retrieval prompt.',
    'Attention remains FP32 with a materialized score buffer; this is tiled QK/AV, not fused FlashAttention.',
    'Attention microbenchmarks use resident synthetic buffers. End-to-end warm measurements determine application speedup.',
    'Model weights/KV storage, tiny projection kernels, single-token decode and draft projections retain existing formats/implementations. Tiled attention also applies to target batched verification.',
    'No all-kernel sampling in throughput measurements. No hardware Neural Accelerator utilization measurements.']
verification = root / 'verification.jsonl'
if verification.exists() and verification.read_text().strip():
    result['verification'] = json.loads(verification.read_text().splitlines()[-1])
(root / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
for name, value in result['variants'].items():
    print(name, json.dumps({'wall_ms': value['median']['wall_ms'], 'tok_s': value['median']['prefill_tokens_per_second'],
        'speedup': value['speedup_vs_legacy'], 'answer_correct': value['long_answer_correct']}))
