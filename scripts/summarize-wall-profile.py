#!/usr/bin/env python3
"""Summarize development wall diagnostics; never produces EvaluationReport data."""
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])

def read(name):
    return [json.loads(line) for line in (root / name).read_text().splitlines() if line.strip()]

records = read('prefill.jsonl')
fixture = next(row for row in records if row['type'] == 'fixture')
warm = [row for row in records if row.get('mode') == 'warm_uncached']
sampled = next(row for row in records if row.get('mode') in ('attention_encoder_counters', 'all_kernel_encoder_counters'))

def median_field(rows, key, nested=False):
    return statistics.median((row['timing'] if nested else row)[key] for row in rows)

warm_summary = {key: median_field(warm, key) for key in ['wall_ms', 'process_cpu_ms', 'tokens_per_second']}
warm_summary.update({key: median_field(warm, key, True) for key, value in warm[0]['timing'].items() if isinstance(value, (int, float))})
warm_summary['host_nonwait_wall_ms'] = statistics.median(row['wall_ms'] - row['timing']['host_wait_ms'] for row in warm)
timing = sampled['timing']
attention_ms = sum(timing[key] for key in ['attention_qk_ms', 'attention_softmax_ms', 'attention_av_ms'])
summary = {
    'kind': 'development diagnostic, not endpoint-bench',
    'fixture': fixture,
    'warm_run_count': len(warm),
    'warm_medians': warm_summary,
    'warm_wall_range_ms': [min(row['wall_ms'] for row in warm), max(row['wall_ms'] for row in warm)],
    'sampled_run': sampled,
    'attention_total_ms': attention_ms,
    'attention_fraction_sampled_gpu': attention_ms / timing['gpu_ms'],
    'sampled_wall_change_vs_warm_median_pct': (sampled['wall_ms'] / warm_summary['wall_ms'] - 1) * 100,
    'interpretation': 'Host wait overlaps GPU execution. Process CPU time is not wall time. Encoder counter mode changes encoder boundaries; isolated kernels use hot synthetic buffers.',
}
if timing.get('kernels'):
    kernels = sorted(timing['kernels'], key=lambda item: item['gpu_ms'], reverse=True)
    categories = {}
    for item in kernels:
        name = item['kernel']
        if name in ('batch_attention_scores', 'attention_scores', 'attention_softmax', 'batch_attention_values', 'attention_values'):
            category = 'full_attention_qk_softmax_av'
        elif name in ('expert_matrix', 'expert_matrix_mpp', 'expert_matrix_tiny', 'expert_gate_up', 'expert_down'):
            category = 'routed_expert_projections'
        elif name.startswith('matrix_product') or name == 'gemv':
            category = 'other_matrix_projections'
        elif name in ('batch_delta', 'delta'):
            category = 'deltanet_recurrence'
        else:
            category = 'other_kernels'
        categories[category] = categories.get(category, 0) + item['gpu_ms']
    summary['kernel_breakdown'] = kernels
    summary['category_gpu_ms'] = categories
    summary['counter_coverage_fraction'] = sum(item['gpu_ms'] for item in kernels) / timing['gpu_ms']
    summary['counter_dispatch_count'] = sum(item['count'] for item in kernels)
if (root / 'attention.jsonl').exists():
    micro = read('attention.jsonl')
    summary['attention_microbenchmark_medians'] = []
    for history in sorted({row['history'] for row in micro}):
        phases = {}
        rows = None
        for phase in ['combined', 'combined_counters', 'qk', 'softmax', 'av']:
            values = [row for row in micro if row['history'] == history and row['phase'] == phase]
            if values:
                rows = values[0]['rows']
                phases[phase] = {'runs': len(values), 'wall_ms': median_field(values, 'wall_ms'), 'gpu_ms': median_field(values, 'gpu_ms', True)}
                if phase == 'combined_counters':
                    phases[phase]['counter_sum_ms'] = statistics.median(sum(row['timing'][key] for key in ['attention_qk_ms', 'attention_softmax_ms', 'attention_av_ms']) for row in values)
        summary['attention_microbenchmark_medians'].append({'history': history, 'rows': rows, 'phases': phases})
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
