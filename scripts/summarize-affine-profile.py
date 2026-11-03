#!/usr/bin/env python3
"""Summarize development measurements, keeping wall and sampled intervals separate."""
import argparse
import collections
import importlib.util
import json
import pathlib
import statistics

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('packed_summary', HERE / 'summarize-packed-profile.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def kernels(root, schedules=False):
    timing = collections.defaultdict(list)
    checks = []
    identities = []
    for label in ['validation', 'forward', 'reverse']:
        identity = json.loads((root / (label + '-identity.json')).read_text())
        identities.append(identity)
        assert (identity['shader_validation'] == '1') == (label == 'validation')
        rows = read_rows(root / (label + '.jsonl'))
        cc = [r for r in rows if r['type'] == 'check']
        tt = [r for r in rows if r['type'] == 'timing']
        assert len(cc) == (36 if schedules else 54), (label, len(cc))
        assert len(tt) == (0 if label == 'validation' else 240 if schedules else 390), (label, len(tt))
        checks.extend(cc)
        for r in tt:
            timing[r['family'], r['m'], r['k'], r['n'], r['routing'], r['variant']].append(r['gpu_ms'])
    for key in identities[0]['hashes']:
        if key.endswith(('.metal', '.mm')):
            assert len({i['hashes'][key] for i in identities}) == 1, key
    assert all(len(v) == 10 for v in timing.values())
    assert all(r['guards'] and r['all_live_outputs_finite'] for r in checks)
    return {'timed_samples': sum(map(len, timing.values())), 'check_cases': len(checks),
            'cpu_reference_points_per_implementation': sum(r['cpu_reference_points'] for r in checks),
            'max_cpu_reference_error': max(max(r['ours_max_abs'], r['splash_max_abs']) for r in checks),
            'samples': [{'family': k[0], 'm': k[1], 'k': k[2], 'n': k[3], 'routing': k[4], 'variant': k[5],
                         'median_ms': statistics.median(v), 'range_ms': [min(v), max(v)],
                         'forward_median_ms': statistics.median(v[:5]), 'reverse_median_ms': statistics.median(v[5:])}
                        for k, v in sorted(timing.items())]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=pathlib.Path)
    a = p.parse_args()
    rows = read_rows(a.root / 'affine.jsonl')
    memory = read_rows(a.root / 'affine-memory.jsonl')
    result = {'kind': 'development profile, not official evaluation', 'model': {},
              'warnings': ['Encoder intervals overlap; shares divide their sum, not exclusive wall time.',
                           'CPU, GPU and host waiting overlap and must not be added.',
                           'Synthetic kernel fixtures preserve decoded weights; production checkpoints differ from Splash.',
                           'The fixed-row diagnostic requires allocation padding; fast-math is diagnostic only.']}
    for fixture in [r for r in rows if r['type'] == 'fixture']:
        tokens = fixture['prompt_tokens']
        runs = [r for r in rows if r.get('type') == 'prefill' and r['prompt_tokens'] == tokens]
        warm = [r for r in runs if r['mode'] == 'warm_uncached']
        sampled = [r for r in runs if r['mode'] == 'all_kernel_encoder_counters']
        assert fixture['affine8'] and len(warm) == 3 and len(sampled) == 1 and len(runs) == 5
        assert len({r['first_token'] for r in runs}) == 1
        sample = sampled[0]
        groups = collections.defaultdict(float)
        for k in sample['timing']['kernels']:
            groups[previous.category(k['kernel'])] += k['gpu_ms']
        total = sum(groups.values())
        wall = statistics.median(r['wall_ms'] for r in warm)
        result['model'][tokens] = {
            'fixture': fixture, 'warm_wall_ms': wall,
            'first_token': runs[0]['first_token'],
            'warm_range_ms': [min(r['wall_ms'] for r in warm), max(r['wall_ms'] for r in warm)],
            'warm_gpu_ms': statistics.median(r['timing']['gpu_ms'] for r in warm),
            'warm_cpu_ms': statistics.median(r['process_cpu_ms'] for r in warm),
            'tok_s': tokens / wall * 1000, 'counter_wall_ms': sample['wall_ms'],
            'instrumentation_overhead_pct': (sample['wall_ms'] / wall - 1) * 100,
            'sampled_interval_sum_ms': total,
            'groups': {k: {'ms': v, 'share_pct': v / total * 100} for k, v in sorted(groups.items(), key=lambda x: -x[1])},
            'kernels': sorted(sample['timing']['kernels'], key=lambda x: -x['gpu_ms']),
            'warm_memory': previous.memory(memory, warm[0]['started_utc'], warm[-1]['completed_utc']),
            'cold_wall_ms': runs[0]['wall_ms'], 'cold_gpu_ms': runs[0]['timing']['gpu_ms']}
    result['matched_kernels'] = kernels(a.root / 'kernels')
    if (a.root / 'schedules/done.txt').exists():
        result['schedule_sensitivity'] = kernels(a.root / 'schedules', schedules=True)
    for name in ['load-first', 'reference-after']:
        directory = a.root / name
        if not (directory / 'done.json').exists():
            continue
        rr = read_rows(directory / 'affine.jsonl')
        mm = read_rows(directory / 'affine-memory.jsonl')
        compared = {}
        for fixture in [r for r in rr if r['type'] == 'fixture']:
            tokens = fixture['prompt_tokens']
            original = result['model'][tokens]
            assert fixture['sha256'] == original['fixture']['sha256']
            runs = [r for r in rr if r.get('type') == 'prefill' and r['prompt_tokens'] == tokens]
            warm = [r for r in runs if r['mode'] == 'warm_uncached']
            counters = [r for r in runs if r['mode'] == 'all_kernel_encoder_counters']
            assert len(runs) == 5 and len(warm) == 3 and len(counters) == 1
            assert {r['first_token'] for r in runs} == {original['first_token']}
            wall = statistics.median(r['wall_ms'] for r in warm)
            sampled = counters[0]
            compared[tokens] = {
                'warm_wall_ms': wall, 'warm_gpu_ms': statistics.median(r['timing']['gpu_ms'] for r in warm),
                'warm_range_ms': [min(r['wall_ms'] for r in warm), max(r['wall_ms'] for r in warm)],
                'tok_s': tokens / wall * 1000, 'throughput_gain_vs_initial_pct': (original['warm_wall_ms'] / wall - 1) * 100,
                'counter_wall_ms': sampled['wall_ms'],
                'counter_overhead_pct': (sampled['wall_ms'] / wall - 1) * 100,
                'kernels': sorted(sampled['timing']['kernels'], key=lambda k: -k['gpu_ms']),
                'warm_memory': previous.memory(mm, warm[0]['started_utc'], warm[-1]['completed_utc'])}
        result[name] = compared
    if 'load-first' in result and 'reference-after' in result:
        result['paired_followup'] = {
            t: {'candidate_gain_vs_reference_after_pct': (result['reference-after'][t]['warm_wall_ms'] / c['warm_wall_ms'] - 1) * 100,
                'reference_wall_drift_pct': (result['reference-after'][t]['warm_wall_ms'] / result['model'][t]['warm_wall_ms'] - 1) * 100}
            for t, c in result['load-first'].items()}
    (a.root / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'model_tokens': list(result['model']), 'matched_samples': result['matched_kernels']['timed_samples'],
                      'schedules_complete': 'schedule_sensitivity' in result}))


if __name__ == '__main__':
    main()
