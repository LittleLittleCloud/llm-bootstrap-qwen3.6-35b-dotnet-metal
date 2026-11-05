#!/usr/bin/env python3
"""Summarize shared-gate diagnostics, excluding warmup and validation timings."""
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
    for name in ['micro','validation']:
        assert (root/name/'exit.txt').read_text().strip()=='0'
        rows=read(root/name/'stdout.jsonl');checks=[r for r in rows if r['type']=='shared_gate_check']
        assert len(checks)==16 and all(r['bitwise'] and r['guards'] for r in checks)
        result['checks'][name]={'count':len(checks),'bitwise':True,'guards':True,'cpu_max_abs_error':max(r['cpu_max_abs_error'] for r in checks)}
        if name=='micro':
            groups=defaultdict(lambda:defaultdict(list))
            for r in rows:
                if r['type']=='shared_gate_bench' and r['run']>0:groups[(r['rows'],r['scope'])][r['variant']].append(r['timing']['gpu_ms'])
            for (rows,scope),v in groups.items():
                a,b=v['batch_mix'],v['batch_mix_precomputed_gate'];assert len(a)==len(b)==9
                result['micro'].append({'rows':rows,'scope':scope,'baseline_ms':median(a),'precomputed_ms':median(b),'latency_change_percent':(median(b)/median(a)-1)*100,'baseline_observations_ms':a,'precomputed_observations_ms':b})
    assert (root/'paired-30k/exit.txt').read_text().strip()=='0'
    rows=[r for r in read(root/'paired-30k/stdout.jsonl') if r['type']=='shared_gate_pair']
    assert len(rows)==12 and all(r['first_token']==22 and r['prompt_tokens']==29939 and r['logits_bitwise'] for r in rows)
    model={'all_final_logits_bitwise':True};warm=[r for r in rows if r['run']>0]
    for variant in ['baseline','precomputed']:
        v=[r for r in warm if r['variant']==variant];assert len(v)==5
        assert len(set(r['timing']['dispatches'] for r in v))==1
        model[variant]={'median_wall_ms':median(r['wall_ms'] for r in v),'wall_ms':[r['wall_ms'] for r in v],
            'dispatches':v[0]['timing']['dispatches'],'median_gpu_ms':median(r['timing']['gpu_ms'] for r in v),
            'median_host_encode_ms':median(r['timing']['host_encode_ms'] for r in v),
            'metal_buffer_bytes':sorted(set(r['metal_buffer_bytes'] for r in v))}
        model[variant]['tokens_per_second']=29939/(model[variant]['median_wall_ms']/1000)
    differences=[]
    for i in range(1,6):
        v={r['variant']:r['wall_ms'] for r in warm if r['run']==i};differences.append(v['precomputed']-v['baseline'])
    model['paired_extra_ms']=differences;model['median_paired_extra_ms']=median(differences)
    model['latency_change_percent']=(model['precomputed']['median_wall_ms']/model['baseline']['median_wall_ms']-1)*100
    result['paired_30k']=model
    for p in root.glob('*/memory.jsonl'):
        rows=read(p)
        def swap(r):
            page=int(re.search(r'page size of (\d+) bytes',r['vm'])[1]);n=int(re.search(r'Swapouts:\s+(\d+)',r['vm'])[1]);return page*n/2**20
        result['memory'][p.parent.name]={'sampled_new_swapouts_mib':swap(rows[-1])-swap(rows[0]),'thermal_warning_samples':sum('No thermal warning' not in r['thermal'] for r in rows),'scope':'whole machine, loading included'}
        if p.parent.name=='paired-30k':
            starts=[r for r in rows if r.get('latest_observation',{}).get('type')=='shared_gate_pair' and r['latest_observation']['run']==0]
            if starts:result['memory'][p.parent.name]['after_last_run0_sample_new_swapouts_mib']=swap(rows[-1])-swap(starts[-1])
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (root/'micro.csv').open('w') as f:
        keys=['rows','scope','baseline_ms','precomputed_ms','latency_change_percent'];writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(result['micro'])
    manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and p.name!='evidence-manifest.json'}
    (root/'evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
