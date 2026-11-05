#!/usr/bin/env python3
"""Summarize native-system diagnostics; encoder intervals are not wall costs."""
import argparse
from collections import defaultdict
import csv
import importlib.util
import json
from pathlib import Path
import re
import statistics as stats

spec = importlib.util.spec_from_file_location('packed_summary', Path(__file__).with_name('summarize-packed-profile.py'))
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)

def read(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.startswith('{')]

def ours_group(label):
    name, _, shape = label.partition('|')
    p = {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', shape)}
    if name.startswith('draft_') or (name.startswith(('matrix_product', 'projection_')) and p.get('K') == 16384):
        return 'Draft context'
    if name.startswith(('attention_', 'batch_attention_', 'batch_rope')) or name == 'batch_split_q':
        return 'Attention + preparation'
    if name.startswith(('delta_', 'batch_conv')) or name in ('batch_delta', 'batch_l2') or (name == 'rmsnorm_pack' and p.get('Flags', 0) & 2):
        return 'DeltaNet + preparation'
    if name.startswith(('expert_', 'batch_mix')) or name in ('batch_router', 'shared_gate_up_pack'):
        return 'MoE incl shared / routing / combine'
    if p and (name.startswith(('projection_', 'matrix_product'))):
        if (p.get('N') in (1, 256) or (p.get('N') == 2048 and p.get('K') == 512)
                or (p.get('A') == 512 and p.get('B') == 512)):
            return 'MoE incl shared / routing / combine'
    if name.startswith(('matrix_product', 'projection_')) or name in ('rmsnorm', 'rmsnorm_pack', 'add', 'silu_product'):
        return 'Dense + norm / residual'
    if name in ('batch_embedding_affine', 'gemv', 'argmax'):
        return 'Embedding / head'
    raise ValueError(f'Unclassified ours kernel: {label}')

def splash_group(name):
    if name.startswith('moe_') or name.startswith('gguf_float'):
        return 'MoE incl shared / routing / combine'
    if name.startswith('prefill_attention') or name == 'rope_build_tables':
        return 'Attention + preparation'
    if name.startswith('prefill_gdn'):
        return 'DeltaNet + preparation'
    if name.startswith(('gguf_prefill_af4', 'prefill_draft')) or name == 'copy_rows_bf16':
        return 'Draft context'
    if name.startswith(('gguf_prefill_af8', 'norm_rms')):
        return 'Dense + norm / residual'
    if name.startswith(('gguf_embed', 'gguf_decode', 'decode_sample')):
        return 'Embedding / head'
    raise ValueError(f'Unclassified Splash kernel: {name}')

def memory(path):
    rows = read(path)
    def swapouts(r):
        page = int(re.search(r'page size of (\d+) bytes', r['vm'])[1])
        count = int(re.search(r'Swapouts:\s+(\d+)', r['vm'])[1])
        return count * page / 2**20
    return {'new_swapouts_mib': swapouts(rows[-1])-swapouts(rows[0]),
            'thermal_warning_samples': sum('No thermal warning' not in r['thermal'] for r in rows),
            'scope': 'Whole machine, including loading; not attributed to the engine'}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    args = ap.parse_args()
    result = {'kind': 'development profile, not endpoint evaluation', 'cases': {},
              'limitations': ['Native checkpoints and activation/draft precision differ.',
                             'Counter encoder intervals may overlap and are not additive wall costs.',
                             'Ours creates prefix snapshots; all reuse is cleared before each request.',
                             'Splash includes shared experts inside MoE kernels; ours separates them.',
                             'Shape-tagged groups improve attribution; remaining norm/epilogue fusion boundaries differ.']}
    kernel_rows = []
    for directory in sorted(args.root.iterdir()):
        if not directory.is_dir() or not (directory/'exit.txt').exists():
            continue
        assert (directory/'exit.txt').read_text() == '0', directory
        rows = read(directory/'stdout.jsonl')
        prefills = [r for r in rows if r.get('type') == 'prefill']
        if not prefills:
            continue
        is_ours = directory.name.startswith('ours')
        for tokens in sorted({r.get('prompt_tokens', r.get('tokens')) for r in prefills}):
            rr = [r for r in prefills if r.get('prompt_tokens', r.get('tokens')) == tokens]
            warm = [r for r in rr if r.get('mode') == 'warm_uncached' or (not is_ours and r['run'] in (1, 2, 3))]
            sample = next(r for r in rr if r.get('sampled') or r.get('mode') == 'all_kernel_encoder_counters')
            keys = defaultdict(lambda: [0., 0])
            if is_ours:
                assert len({r['first_token'] for r in rr}) == 1
                for k in sample['timing']['kernels']:
                    keys[k['kernel']] = [k['gpu_ms'], k['count']]
                shape_tagged = any('|' in k for k in keys)
                classify = ours_group
            else:
                checks = [r for r in rows if r.get('type') == 'check']
                assert len(checks) == 5 and all(r['repeat_equal'] for r in checks)
                assert len({r['first_token'] for r in checks}) == 1
                for command in read(directory/'intervals.jsonl'):
                    events = command['events']
                    span = (max(e['start_ns']+e['duration_ns'] for e in events)-min(e['start_ns'] for e in events))/1e6
                    assert command['command_buffers'] == 1 and span <= command['gpu_ms']*1.05+.01
                    for e in events:
                        keys[e['kernel']][0] += e['duration_ns']/1e6
                        keys[e['kernel']][1] += 1
                shape_tagged = False
                classify = splash_group
            groups = defaultdict(float)
            for name, (ms, count) in keys.items():
                group = classify(name)
                # Without shapes these cannot separate shared/draft from dense.
                if is_ours and not shape_tagged and group in ('MoE incl shared / routing / combine', 'Dense + norm / residual', 'Draft context'):
                    group = 'Projection / MoE / norm / draft (unseparated)'
                groups[group] += ms
                kernel_rows.append([directory.name, tokens, name, ms, count, group])
            wall = stats.median(r['wall_ms'] for r in warm)
            gpu = stats.median((r['timing'] if is_ours else r)['gpu_ms'] for r in warm)
            result['cases'][f'{directory.name}-{tokens}'] = {
                'tokens': tokens, 'warm_wall_ms': wall, 'warm_gpu_ms': gpu,
                'warm_wall_samples_ms': [r['wall_ms'] for r in warm], 'tokens_per_second': tokens*1000/wall,
                'sampled_wall_ms': sample['wall_ms'], 'sampled_gpu_ms': (sample['timing'] if is_ours else sample)['gpu_ms'],
                'counter_wall_overhead_pct': (sample['wall_ms']/wall-1)*100,
                'dispatches': sum(v[1] for v in keys.values()), 'shape_tagged': shape_tagged,
                'interval_sum_ms': sum(groups.values()), 'groups_ms': dict(groups),
                'memory': memory(directory/'memory.jsonl'),
            }
            if is_ours:
                result['cases'][f'{directory.name}-{tokens}']['warm_memory'] = previous.memory(
                    read(directory/'memory.jsonl'), warm[0]['started_utc'], warm[-1]['completed_utc'])
                result['cases'][f'{directory.name}-{tokens}']['warm_cpu_ms'] = stats.median(r['process_cpu_ms'] for r in warm)
    (args.root/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    with (args.root/'kernels.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['case', 'tokens', 'kernel', 'interval_ms', 'count', 'group'])
        writer.writerows(kernel_rows)
    micro = args.root/'kernels'
    if all((micro/f'{label}.jsonl').exists() for label in ('forward', 'reverse')):
        measured = defaultdict(list)
        checks = []
        for label in ('forward', 'reverse'):
            rows = read(micro/f'{label}.jsonl')
            cc = [r for r in rows if r['type'] == 'check']
            tt = [r for r in rows if r['type'] == 'timing']
            assert len(cc) == 30 and len(tt) == 162, (label, len(cc), len(tt))
            checks.extend(cc)
            for r in tt:
                measured[(r['family'], r['m'], r['k'], r['n'], r['routing'], r['variant'])].append(r['gpu_ms'])
        assert all(len(v) == 6 for v in measured.values())
        assert all(r['guards'] and r['all_live_outputs_finite'] for r in checks)
        result['matched_kernels'] = {
            'check_cases': len(checks), 'timed_observations': sum(map(len, measured.values())),
            'cpu_reference_points_per_side': sum(r['cpu_reference_points'] for r in checks),
            'max_cpu_error_ours': max(r['ours_max_abs'] for r in checks),
            'max_cpu_error_splash': max(r['splash_max_abs'] for r in checks),
            'compact_stage_full_output_bitwise_in_normal_passes': True,
            'compact_validation': {'exit_code': int((micro/'validation-exit.txt').read_text()) if (micro/'validation-exit.txt').exists() else None,
                                   'stderr': (micro/'validation.stderr').read_text() if (micro/'validation.stderr').exists() else None},
            'samples': [dict(zip(('family', 'm', 'k', 'n', 'routing', 'variant'), key),
                             median_ms=stats.median(values), samples_ms=values)
                        for key, values in sorted(measured.items())],
        }
        if (micro/'validation-current-identity.json').exists():
            identity = json.loads((micro/'validation-current-identity.json').read_text())
            assert identity['exit_code'] == 0
            current_checks = [r for r in read(micro/'validation-current.jsonl') if r['type'] == 'check']
            assert len(current_checks) == 18 and all(r['guards'] and r['all_live_outputs_finite'] for r in current_checks)
            result['matched_kernels']['production_validation'] = {'check_cases': len(current_checks), 'exit_code': 0}
        (args.root/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    for name, row in result['cases'].items():
        print(name, round(row['warm_wall_ms'], 2), round(row['tokens_per_second'], 1), 'counter overhead', round(row['counter_wall_overhead_pct'], 1))

if __name__ == '__main__':
    main()
