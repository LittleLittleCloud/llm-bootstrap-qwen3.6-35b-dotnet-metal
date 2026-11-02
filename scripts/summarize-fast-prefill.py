#!/usr/bin/env python3
"""Report paired fast-prefill experiments; cold full-answer runs are excluded."""
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
records = [json.loads(line) for line in (root / 'compare.jsonl').read_text().splitlines() if line.strip()]
fixture = next(r for r in records if r['type'] == 'fixture')
summary = {'kind': 'development diagnostic, not official evaluation', 'fixture': fixture, 'variants': {}}
for name in fixture['variants']:
    warm = [r for r in records if r.get('variant') == name and r.get('mode') == 'warm_uncached']
    if len(warm) != fixture['runs']:
        raise ValueError(f'Incomplete warm runs: {name}')
    cold = next(r for r in records if r.get('variant') == name and r.get('mode') == 'cold_full_answer')
    smoke = next(r for r in records if r.get('variant') == name and r['type'] == 'model_smoke')
    median = {k: statistics.median(r[k] for r in warm) for k in ('wall_ms', 'process_cpu_ms', 'prefill_tokens_per_second')}
    median.update({k: statistics.median(r['timing'][k] for r in warm) for k in ('gpu_ms', 'host_encode_ms', 'host_wait_ms', 'command_buffers', 'dispatches')})
    summary['variants'][name] = {'warm_runs': len(warm), 'median': median,
        'wall_range_ms': [min(r['wall_ms'] for r in warm), max(r['wall_ms'] for r in warm)],
        'model_smoke': smoke, 'long_answer': cold['text'], 'long_answer_correct': cold['answer_correct'],
        'first_tokens_match_control': all(r['first_token_matches_baseline'] for r in warm)}
control = summary['variants'][fixture['variants'][0]]['median']['wall_ms']
for v in summary['variants'].values():
    v['speedup_vs_paired_previous_best'] = control / v['median']['wall_ms']
historical = root.parent / 'lowp-attention' / 'summary.json'
if historical.exists():
    original = json.loads(historical.read_text())['variants']['legacy']['median']['wall_ms']
    summary['historical_original_fp32_wall_ms'] = original
    for v in summary['variants'].values():
        v['speedup_vs_historical_original_fp32'] = original / v['median']['wall_ms']
rs = [json.loads(line) for line in (root / 'attention-sliced.jsonl').read_text().splitlines()]
summary['attention_microbenchmarks'] = []
for history, rows, mode in sorted({(r['history'], r['rows'], r['mode']) for r in rs}):
    group = [r for r in rs if (r['history'], r['rows'], r['mode']) == (history, rows, mode)]
    summary['attention_microbenchmarks'].append({'history': history, 'rows': rows, 'mode': mode,
        'median_gpu_ms': statistics.median(r['timing']['gpu_ms'] for r in group),
        'max_abs_error_vs_cpu': max(r['max_abs_error_vs_cpu'] for r in group), 'alias_checks_passed': all(r['alias_check'] == 'passed' for r in group)})
verification = root / 'verification.jsonl'
if verification.exists() and verification.read_text().strip():
    summary['verification'] = json.loads(verification.read_text().splitlines()[-1])
summary['limitations'] = [
    'The paired control is FP16 projections plus FP32 tiled attention (previous best), not the original FP32 engine.',
    'Original FP32 speedup uses the preceding experiment as a historical reference.',
    'Full model checks cover two fixed prompts. Lower-precision computation is explicitly accepted for this development work.',
    'The fused kernels are evaluated prototypes; the selected fast mode uses sliced FP16 attention, not a fused kernel.',
    'Checkpoint formats and KV storage remain unchanged. No all-kernel counter instrumentation is used for throughput.']
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
for n, v in summary['variants'].items():
    print(n, json.dumps({'wall_ms': v['median']['wall_ms'], 'tok_s': v['median']['prefill_tokens_per_second'], 'speedup': v['speedup_vs_paired_previous_best'], 'answer_correct': v['long_answer_correct']}))
