#!/usr/bin/env python3
"""Summarize paired development measurements; never emits an evaluation report."""
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
records = [json.loads(line) for line in (root / 'final-comparison.jsonl').read_text().splitlines() if line.strip()]
fixture = next(r for r in records if r.get('type') == 'fixture')
result = {'kind': 'development diagnostics, not an endpoint-bench report', 'fixture': fixture, 'prefill': {}, 'decode': {}}
for name in fixture['variants']:
    warm = [r for r in records if r.get('variant') == name and r.get('mode') == 'warm_uncached']
    if len(warm) != fixture['runs']:
        raise ValueError(f'Incomplete prefill measurements: {name}')
    answer = next(r for r in records if r.get('variant') == name and r.get('mode') == 'cold_full_answer')
    result['prefill'][name] = {'median_tok_s': statistics.median(r['prefill_tokens_per_second'] for r in warm),
        'median_wall_ms': statistics.median(r['wall_ms'] for r in warm),
        'wall_range_ms': [min(r['wall_ms'] for r in warm), max(r['wall_ms'] for r in warm)],
        'answer_correct': answer['answer_correct'], 'answer': answer['text'], 'warm_runs': len(warm)}
result['prefill_speedup'] = result['prefill']['tuned_prefill']['median_tok_s'] / result['prefill']['previous_fast_prefill']['median_tok_s']
for name in ['previous_fast_prefill', 'optimized_attention_small_verifier']:
    warm = [r for r in records if r.get('type') == 'decode_variant' and r['variant'] == name and r['warm']]
    if len(warm) != fixture['runs']:
        raise ValueError(f'Incomplete decode measurements: {name}')
    result['decode'][name] = {'median_tok_s': statistics.median(r['decode_tokens_per_second'] for r in warm),
        'median_wall_ms': statistics.median(r['wall_ms'] for r in warm), 'warm_runs': len(warm),
        'speculation': [r['speculation'] for r in warm]}
result['verification'] = next(r for r in records if 'verifier_checks_passed' in r)
result['cache'] = next(r for r in records if 'cases' in r and 'snapshot_bytes' in r)
result['matrix_checks'] = json.loads((root / 'final-matrix-test.jsonl').read_text())
result['kernel_checks'] = json.loads((root / 'final-kernel-test.jsonl').read_text())
attention = [json.loads(line) for line in (root / 'final-attention-test.jsonl').read_text().splitlines()]
result['attention_checks'] = {'all_alias_checks_passed': all(r['alias_check'] == 'passed' for r in attention),
    'max_abs_error_vs_cpu': max(r['max_abs_error_vs_cpu'] for r in attention), 'shapes': sorted(set((r['history'], r['rows']) for r in attention))}
result['limitations'] = ['One fixed public 10K prompt and arithmetic smoke for full-model correctness.',
    'Decode is a local 1000-token diagnostic with a public endpoint-bench prompt; not the browser evaluation.',
    'Both paired variants reserve the largest tested workspace; timing excludes model load and tokenization.',
    'The target of another 5x prefill improvement is not achieved.',
    'No new formal quality evaluation or leaderboard result is claimed.']
(root / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ['prefill', 'prefill_speedup', 'decode']}, indent=2))
