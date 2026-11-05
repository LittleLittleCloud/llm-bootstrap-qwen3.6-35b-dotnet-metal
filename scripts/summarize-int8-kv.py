#!/usr/bin/env python3
"""Summarize paired native timings and saved first-token probability distributions."""
import argparse
import array
from collections import defaultdict
import importlib.util
import json
import math
from pathlib import Path
import statistics

spec = importlib.util.spec_from_file_location('groups', Path(__file__).with_name('summarize-current-gap.py'))
groups = importlib.util.module_from_spec(spec)
spec.loader.exec_module(groups)

def probability_comparison(a, b):
    if len(a) != 248320 or len(b) != len(a) or not all(math.isfinite(x) for x in (*a, *b)):
        raise ValueError('Unexpected/malformed full-vocabulary logits')
    ma, mb = max(a), max(b)
    la = ma + math.log(sum(math.exp(x - ma) for x in a))
    lb = mb + math.log(sum(math.exp(x - mb) for x in b))
    mean_a, mean_b = statistics.fmean(a), statistics.fmean(b)
    kl = tv = err = ref = 0
    for x, y in zip(a, b):
        lp, lq = x - la, y - lb
        p, q = math.exp(lp), math.exp(lq)
        kl += p * (lp - lq)
        tv += abs(p - q) / 2
        err += ((x - mean_a) - (y - mean_b)) ** 2
        ref += (x - mean_a) ** 2
    top_a = sorted(range(len(a)), key=a.__getitem__, reverse=True)[:10]
    top_b = sorted(range(len(b)), key=b.__getitem__, reverse=True)[:10]
    return {'kl_nats': max(0, kl), 'total_variation': tv, 'centered_relative_l2': math.sqrt(err / ref),
            'top1_equal': top_a[0] == top_b[0], 'top10_overlap': len(set(top_a) & set(top_b))}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    a = p.parse_args()
    result = {'scope': 'Native diagnostics; counter intervals may overlap and are not additive wall time.', 'profile': {}, 'quality': []}
    for directory in sorted(a.root.iterdir()):
        if not directory.is_dir() or not (directory / 'exit.txt').exists():
            continue
        if int((directory / 'exit.txt').read_text()) != 0:
            raise RuntimeError(f'Failed run: {directory}')
        rows = groups.read(directory / 'stdout.jsonl')
        prefills = [r for r in rows if r.get('type') == 'prefill']
        for tokens in sorted({r['prompt_tokens'] for r in prefills}):
            matching = [r for r in prefills if r['prompt_tokens'] == tokens]
            warm = [r for r in matching if r['mode'] == 'warm_uncached']
            sampled = next(r for r in matching if r['mode'] == 'all_kernel_encoder_counters')
            if len({r['first_token'] for r in matching}) != 1:
                raise ValueError('Non-repeatable first token')
            wall = statistics.median(r['wall_ms'] for r in warm)
            totals = defaultdict(float)
            for kernel in sampled['timing']['kernels']:
                totals[groups.ours_group(kernel['kernel'])] += kernel['gpu_ms']
            result['profile'][f'{directory.name}-{tokens}'] = {
                'tokens': tokens, 'warm_ms': [r['wall_ms'] for r in warm], 'median_ms': wall,
                'tok_s': tokens * 1000 / wall, 'dispatches': sampled['timing']['dispatches'],
                'sampled_groups_ms': totals, 'counter_overhead_pct': (sampled['wall_ms'] / wall - 1) * 100,
                'memory': groups.memory(directory / 'memory.jsonl'),
                'warm_memory': groups.previous.memory(groups.read(directory / 'memory.jsonl'), warm[0]['started_utc'], warm[-1]['completed_utc'])}
    if (a.root / 'quality-fp16/exit.txt').exists() and (a.root / 'quality-int8/exit.txt').exists():
        ref_rows = groups.read(a.root / 'quality-fp16/stdout.jsonl')
        act_rows = groups.read(a.root / 'quality-int8/stdout.jsonl')
        ref_by_id = {(r['fixture'], r['speculative']): r for r in ref_rows if r.get('type') == 'kv_quality'}
        for r in act_rows:
            if r.get('type') != 'kv_quality':
                continue
            ref = ref_by_id[(r['fixture'], r['speculative'])]
            if ref['prompt_tokens'] != r['prompt_tokens'] or not ref['passed'] or not r['passed']:
                raise ValueError('Unmatched or failed quality fixture')
            aa, bb = array.array('f'), array.array('f')
            aa.frombytes((a.root / 'quality-fp16/logits' / ref['logits_file']).read_bytes())
            bb.frombytes((a.root / 'quality-int8/logits' / r['logits_file']).read_bytes())
            result['quality'].append(dict(fixture=r['fixture'], speculative=r['speculative'], prompt_tokens=r['prompt_tokens'], passed=True,
                tokens_equal=ref['tokens'] == r['tokens'], target_kv_bytes=r['target_kv_bytes'], **probability_comparison(aa, bb)))
    (a.root / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    for name, row in result['profile'].items():
        print(name, round(row['median_ms'], 3), 'ms', round(row['tok_s'], 2), 'tok/s', row['dispatches'], 'dispatches')
    if result['quality']:
        print('Quality:', len(result['quality']), 'passed; max KL', max(r['kl_nats'] for r in result['quality']))

if __name__ == '__main__':
    main()
