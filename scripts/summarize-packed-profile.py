#!/usr/bin/env python3
"""Summarize paired development profiles without treating overlapping samples as exclusive costs."""
import argparse, collections, datetime, json, pathlib, re, statistics

def category(n):
    if n.startswith(('expert_gate_up','ablation_gate_up')):return 'expert_gate_up'
    if n.startswith(('expert_matrix','ablation_matrix')):return 'expert_down_and_tiny'
    if n.startswith('matrix_product') or n=='gemv':return 'dense_and_draft_matmul'
    if n=='projection_unpack_half':return 'dense_weight_decode'
    if n=='projection_input_half':return 'dense_input_conversion'
    if n.startswith(('attention_','batch_rope','batch_split_q')):return 'attention_layout_rope'
    if n.startswith(('delta_','batch_delta','batch_conv','batch_l2')):return 'deltanet_conv_l2'
    if n in ['batch_router','expert_group_parallel','expert_tiles','expert_counts','expert_group']:return 'expert_routing'
    if n=='expert_input_half':return 'expert_input_conversion'
    if n=='batch_mix':return 'expert_combine'
    return 'other'

def parse_time(value):
    # .NET emits variable fractional precision; Python 3.9 accepts only 3/6 digits.
    value=re.sub(r'\.(\d+)(?=[+-]|Z)',lambda m:'.'+(m[1]+'000000')[:6],value)
    return datetime.datetime.fromisoformat(value.replace('Z','+00:00'))

def memory(samples,start,end):
    start,end=map(parse_time,(start,end))
    chosen=[r for r in samples if start<=parse_time(r['utc'])<=end]
    if not chosen:return {'samples':0}
    vm=[];swap=[]
    for r in chosen:
        page=int(re.search(r'page size of (\d+)',r['vm']).group(1))
        counters={k:int(v)*page/2**30 for k,v in re.findall(r'^([^:\n]+):\s+(\d+)\.',r['vm'],re.M)}
        vm.append(counters);swap.append(float(re.search(r'used = ([\d.]+)M',r['swap']).group(1)))
    fields=['Pages wired down','Pages occupied by compressor','Swapins','Swapouts']
    return {'samples':len(chosen),'whole_machine':True,'swap_used_mib_range':[min(swap),max(swap)],'gib_ranges':{k:[min(v[k] for v in vm),max(v[k] for v in vm)] for k in fields},'swapin_delta_gib':vm[-1]['Swapins']-vm[0]['Swapins'],'swapout_delta_gib':vm[-1]['Swapouts']-vm[0]['Swapouts'],'thermal_messages':sorted(set(r['thermal'].strip() for r in chosen))}

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=pathlib.Path);a=p.parse_args();root=a.root
    result={'kind':'development profile, not official evaluation','warnings':['GPU encoder intervals can overlap: shares below divide their sum, not exclusive wall time.','Separate processes run original then packed; order and whole-machine residency can confound comparisons.','CPU execution overlaps GPU work and host waiting. Do not sum CPU/wait/GPU.'],'models':{}}
    for mode in ['original','packed']:
        rows=[json.loads(s) for s in (root/(mode+'.jsonl')).read_text().splitlines()]
        samples=[json.loads(s) for s in (root/(mode+'-memory.jsonl')).read_text().splitlines()]
        profiles={};result['models'][mode]=profiles
        for fixture in [r for r in rows if r['type']=='fixture']:
            tokens=fixture['prompt_tokens'];runs=[r for r in rows if r.get('type')=='prefill' and r['prompt_tokens']==tokens]
            warm=[r for r in runs if r['mode']=='warm_uncached'];sample=next(r for r in runs if r['mode']=='all_kernel_encoder_counters')
            assert len(warm)==3 and len(runs)==5, (mode,tokens,'incomplete model profile')
            assert fixture['packed_experts']==(mode=='packed')
            wall=statistics.median(r['wall_ms'] for r in warm);groups=collections.defaultdict(float)
            for k in sample['timing']['kernels']:groups[category(k['kernel'])]+=k['gpu_ms']
            total=sum(groups.values())
            profiles[str(tokens)]={'fixture':fixture,'warm_count':len(warm),'wall_ms_median':wall,'wall_ms_range':[min(r['wall_ms'] for r in warm),max(r['wall_ms'] for r in warm)],'tok_s':tokens/wall*1000,'cpu_ms':statistics.median(r['process_cpu_ms'] for r in warm),'gpu_ms':statistics.median(r['timing']['gpu_ms'] for r in warm),'encode_ms':statistics.median(r['timing']['host_encode_ms'] for r in warm),'host_wait_ms':statistics.median(r['timing']['host_wait_ms'] for r in warm),'counter_wall_ms':sample['wall_ms'],'counter_overhead_pct':(sample['wall_ms']/wall-1)*100,'counter_sum_ms':total,'groups':{k:{'interval_ms':v,'share_of_interval_sum_pct':v/total*100} for k,v in sorted(groups.items(),key=lambda i:-i[1])},'kernels':sorted(sample['timing']['kernels'],key=lambda k:-k['gpu_ms']),'first_tokens':sorted(set(r['first_token'] for r in runs)),'metal_buffer_bytes':sample['metal_buffer_bytes'],'cold':{'wall_ms':runs[0]['wall_ms'],'gpu_ms':runs[0]['timing']['gpu_ms']},'warm_memory':memory(samples,warm[0]['started_utc'],warm[-1]['completed_utc']),'sample_memory':memory(samples,sample['started_utc'],sample['completed_utc'])}
    result['comparisons']={}
    for tokens,x in result['models']['original'].items():
        y=result['models']['packed'][tokens]
        assert x['fixture']['sha256']==y['fixture']['sha256']
        assert x['first_tokens']==y['first_tokens'] and len(x['first_tokens'])==1
        result['comparisons'][tokens]={'throughput_gain_pct':(y['tok_s']/x['tok_s']-1)*100,'wall_saved_ms':x['wall_ms_median']-y['wall_ms_median'],'counter_interval_saved_ms':{k:v['interval_ms']-y['groups'][k]['interval_ms'] for k,v in x['groups'].items()}}
    stages=root/'stages'
    if (root/'stages-done.txt').exists():
        timings=[];checks=[]
        for name in ['exact-forward','exact-reverse','broad-forward']:
            rows=[json.loads(s) for s in (stages/(name+'.jsonl')).read_text().splitlines()]
            timings += [r for r in rows if r['type']=='timing'];checks += [r for r in rows if r['type']=='check']
        assert len(timings)==840 and len(checks)==252, 'incomplete stage profile'
        grouped=collections.defaultdict(list)
        for r in timings:grouped[(r['family'],r['m'],r['routing'],r['variant'])].append(r['gpu_ms'])
        assert all(len(vals)==10 for vals in grouped.values()), 'unbalanced stage samples'
        result['stage_sensitivity']=[{'family':key[0],'m':key[1],'routing':key[2],'variant':key[3],'runs':len(vals),'gpu_ms_median':statistics.median(vals),'gpu_ms_range':[min(vals),max(vals)]} for key,vals in sorted(grouped.items())]
        result['stage_checks']={'cases':len(checks),'cpu_reference_points':sum(r['cpu_reference_points'] for r in checks),'max_relative_l2':max(r['relative_l2'] for r in checks),'guards_and_finite':all(r['guards'] and r['all_live_outputs_finite'] for r in checks)}
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['comparisons'],indent=2))
if __name__=='__main__':main()
