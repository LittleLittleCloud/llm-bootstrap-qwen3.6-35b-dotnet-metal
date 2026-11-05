#!/usr/bin/env python3
"""Summarize BF16 expert-output diagnostics; warm uninstrumented timings only."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
from statistics import median

def read(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s.startswith('{')]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);root=ap.parse_args().root
    result={'checks':{},'micro':[],'memory':{}}
    for name in ['micro','validation','api-validation']:
        assert (root/name/'exit.txt').read_text().strip()=='0'
        rows=read(root/name/'stdout.jsonl')
        down=[r for r in rows if r['type']=='expert_output_down_check'];mix=[r for r in rows if r['type']=='expert_output_mix_check']
        assert len(down)==6 and all(r['rounded_bitwise'] and r['guards'] for r in down)
        assert len(mix)==12 and all(r['rounded_reference_bitwise'] and r['guards'] for r in mix)
        repeats=[r for r in rows if r['type']=='expert_output_repeat']
        assert len(repeats)==6 and all(r['inconsistent']==0 for r in repeats)
        result['checks'][name]={'down':down,'mix':mix,'reference_repeats':repeats}
        if name=='micro':
            groups=defaultdict(lambda:defaultdict(list))
            for r in rows:
                if r['type']=='expert_output_bench' and r['run']>0:groups[(r['rows'],r['layout'],r['scope'])][r['variant']].append(r['timing']['gpu_ms'])
            for (rows,layout,scope),v in groups.items():
                a,b=v['fp32'],v['bf16'];assert len(a)==len(b)==7
                result['micro'].append({'rows':rows,'layout':layout,'scope':scope,'fp32_ms':median(a),'bf16_ms':median(b),'latency_change_percent':(median(b)/median(a)-1)*100,'fp32_observations_ms':a,'bf16_observations_ms':b})
    for case in ['paired-30k','paired-30k-repeat']:
        if not (root/case/'exit.txt').exists():continue
        assert (root/case/'exit.txt').read_text().strip()=='0'
        allrows=read(root/case/'stdout.jsonl');rows=[r for r in allrows if r['type']=='expert_output_pair']
        assert len(rows)==12 and all(r['prompt_tokens']==29939 and r['repeatable'] for r in rows)
        model={'first_tokens':sorted(set(r['first_token'] for r in rows)),'all_first_tokens_match':all(r['first_token_matches'] for r in rows)}
        warm=[r for r in rows if r['run']>0]
        for variant in ['fp32','bf16']:
            v=[r for r in warm if r['variant']==variant];assert len(v)==5
            assert len(set(r['timing']['dispatches'] for r in v))==1
            model[variant]={'median_wall_ms':median(r['wall_ms'] for r in v),'wall_ms':[r['wall_ms'] for r in v],
                'dispatches':v[0]['timing']['dispatches'],'median_gpu_ms':median(r['timing']['gpu_ms'] for r in v),
                'median_host_encode_ms':median(r['timing']['host_encode_ms'] for r in v),
                'metal_buffer_bytes':sorted(set(r['metal_buffer_bytes'] for r in v)),
                'logit_max_abs_error':max(r['logit_max_abs_error'] for r in v),'logit_relative_l2_error':max(r['logit_relative_l2_error'] for r in v),'logit_cosine':min(r['logit_cosine'] for r in v)}
            model[variant]['tokens_per_second']=29939/(model[variant]['median_wall_ms']/1000)
        differences=[]
        for i in range(1,6):
            v={r['variant']:r['wall_ms'] for r in warm if r['run']==i};differences.append(v['bf16']-v['fp32'])
        model['paired_extra_ms']=differences;model['median_paired_extra_ms']=median(differences)
        model['latency_change_percent']=(model['bf16']['median_wall_ms']/model['fp32']['median_wall_ms']-1)*100
        model['throughput_change_percent']=(model['bf16']['tokens_per_second']/model['fp32']['tokens_per_second']-1)*100
        result[case.replace('-', '_')]=model
        result['selected_pair']=case.replace('-', '_')
        generation=[r for r in allrows if r['type']=='expert_output_generation'];assert len(generation)==4
        result['generation']=generation
        result['generation_checks']={'all_tokens_match':all(r['tokens_match_baseline'] for r in generation),'arithmetic_passed':all(r['arithmetic_passed'] for r in generation if r['fixture']=='arithmetic'),'needle_passed':all(r['text'].strip()=='73DY-DRV5-B8FJ' for r in generation if r['fixture']=='full_prompt')}
    if (root/'quality/exit.txt').exists():
        assert (root/'quality/exit.txt').read_text().strip()=='0'
        quality=[r for r in read(root/'quality/stdout.jsonl') if r['type']=='expert_output_quality'];assert len(quality)==12
        result['quality']=quality
        result['quality_checks']={'passed':sum(r['passed'] for r in quality),'total':len(quality),'all_tokens_match':all(r['tokens_match_baseline'] for r in quality),'all_top1_match':all(r['top1_matches'] for r in quality),'max_softmax_kl_nats':max(r['softmax_kl_nats'] for r in quality),'max_softmax_total_variation':max(r['softmax_total_variation'] for r in quality)}
    for p in root.glob('*/memory.jsonl'):
        rows=read(p)
        def swap(r):
            page=int(re.search(r'page size of (\d+) bytes',r['vm'])[1]);n=int(re.search(r'Swapouts:\s+(\d+)',r['vm'])[1]);return page*n/2**20
        result['memory'][p.parent.name]={'sampled_new_swapouts_mib':swap(rows[-1])-swap(rows[0]),'thermal_warning_samples':sum('No thermal warning' not in r['thermal'] for r in rows),'scope':'whole machine, loading and generation included'}
        if p.parent.name.startswith('paired-30k'):
            starts=[r for r in rows if r.get('latest_observation',{}).get('type')=='expert_output_pair' and r['latest_observation']['run']==0]
            if starts:result['memory'][p.parent.name]['after_last_run0_sample_new_swapouts_mib']=swap(rows[-1])-swap(starts[-1])
            warm_start=[r for r in starts if r['latest_observation']['variant']=='bf16']
            warm_end=[r for r in rows if r.get('latest_observation',{}).get('type')=='expert_output_pair' and r['latest_observation']['run']==5 and r['latest_observation']['variant']=='fp32']
            if warm_start and warm_end:result['memory'][p.parent.name]['timed_warm_new_swapouts_mib']=swap(warm_end[0])-swap(warm_start[0])
            timing_samples=[r for r in rows if r.get('latest_observation',{}).get('type')=='expert_output_pair']
            if timing_samples:result['memory'][p.parent.name]['through_last_timing_observation_new_swapouts_mib']=swap(timing_samples[-1])-swap(rows[0])
            result['memory'][p.parent.name]['sampling_note']='Machine counters sampled about every two seconds; transition intervals cannot be attributed exactly to a request.'
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (root/'micro.csv').open('w') as f:
        keys=['rows','layout','scope','fp32_ms','bf16_ms','latency_change_percent'];writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(result['micro'])
    manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and p.name!='evidence-manifest.json'}
    (root/'evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
