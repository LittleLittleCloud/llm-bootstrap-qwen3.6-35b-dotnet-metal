#!/usr/bin/env python3
"""Summarize fused matmul diagnostics; exclude warmup and validation timings."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics as stats

def read(p):
    result=[]
    for s in p.read_text().splitlines():
        if s.startswith('{'):result.append(json.loads(s))
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);args=ap.parse_args();root=args.root
    result={'scope':'Current K32/K64 implementations, not all possible fused matmul designs','micro':{},'checks':{},'memory':{}}
    csvrows=[]
    for kernel in ['dense_affine_pipeline','dense_affine_pipeline64']:
        for mode in ['micro','validation']:
            name=f'{kernel}-{mode}'
            assert (root/name/'exit.txt').read_text().strip()=='0'
            rows=read(root/name/'stdout.jsonl');checks=[r for r in rows if r['type']=='check']
            result['checks'][name]={'count':len(checks),'max_abs_error':max(r.get('max_abs_error',0) for r in checks),
                'max_relative_l2':max(r.get('relative_l2',0) for r in checks)}
            assert len(checks)==48
            if mode!='micro':continue
            groups=defaultdict(lambda:defaultdict(list))
            for r in rows:
                if r['type']=='bench' and r['run']>0 and r['name'].startswith('staged-'):
                    groups[r['name']][r['candidate']].append(r['timing']['gpu_ms'])
            result['micro'][kernel]={}
            for name,vals in groups.items():
                assert len(vals[False])==len(vals[True])==5
                a,b=stats.median(vals[False]),stats.median(vals[True])
                row={'baseline_ms':a,'fused_ms':b,'latency_change_percent':(b/a-1)*100,'baseline_observations_ms':vals[False],'fused_observations_ms':vals[True]}
                result['micro'][kernel][name]=row
                csvrows.append({'kernel':kernel,'shape':name,'baseline_ms':a,'fused_ms':b,'latency_change_percent':row['latency_change_percent']})
    paired=root/'paired-30k'
    if (paired/'exit.txt').exists():
        assert (paired/'exit.txt').read_text().strip()=='0'
        rows=[r for r in read(paired/'stdout.jsonl') if r['type']=='dispatch_pair']
        assert len(rows)==12 and all(r['first_token']==22 and r['prompt_tokens']==29939 for r in rows)
        model={};warm=[r for r in rows if r['run']>0]
        for variant in [29,31]:
            v=[r for r in warm if r['variant']==variant];assert len(v)==5
            assert len(set(r['timing']['dispatches'] for r in v))==1
            model[str(variant)]={'median_wall_ms':stats.median(r['wall_ms'] for r in v),'wall_ms':[r['wall_ms'] for r in v],
                'dispatches':v[0]['timing']['dispatches'],'median_gpu_ms':stats.median(r['timing']['gpu_ms'] for r in v),
                'median_host_encode_ms':stats.median(r['timing']['host_encode_ms'] for r in v),
                'metal_buffer_bytes':list(sorted(set(r['metal_buffer_bytes'] for r in v)))}
            model[str(variant)]['tokens_per_second']=29939/(model[str(variant)]['median_wall_ms']/1000)
        differences=[]
        for i in range(1,6):
            v={r['variant']:r['wall_ms'] for r in warm if r['run']==i};differences.append(v[31]-v[29])
        model['paired_extra_ms']=differences;model['median_paired_extra_ms']=stats.median(differences)
        model['latency_change_percent']=(model['31']['median_wall_ms']/model['29']['median_wall_ms']-1)*100
        model['dispatch_reduction']=model['29']['dispatches']-model['31']['dispatches']
        model['dispatch_reduction_percent']=model['dispatch_reduction']/model['29']['dispatches']*100
        result['paired_30k']=model
    for p in root.glob('*/memory.jsonl'):
        rows=read(p)
        def swap(r):
            page=int(re.search(r'page size of (\d+) bytes',r['vm'])[1]);n=int(re.search(r'Swapouts:\s+(\d+)',r['vm'])[1]);return page*n/2**20
        result['memory'][p.parent.name]={'sampled_new_swapouts_mib':swap(rows[-1])-swap(rows[0]),'thermal_warning_samples':sum('No thermal warning' not in r['thermal'] for r in rows),'scope':'whole machine, loading included'}
        if p.parent.name=='paired-30k':
            # Bracket the first warm run by the final sample with a run-0 observation.
            starts=[r for r in rows if r.get('latest_observation',{}).get('type')=='dispatch_pair' and r['latest_observation']['run']==0]
            if starts:result['memory'][p.parent.name]['after_last_run0_sample_new_swapouts_mib']=swap(rows[-1])-swap(starts[-1])
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (root/'micro.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=csvrows[0]);writer.writeheader();writer.writerows(csvrows)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
