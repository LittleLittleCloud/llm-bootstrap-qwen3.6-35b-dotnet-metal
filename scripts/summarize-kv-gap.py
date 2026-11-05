#!/usr/bin/env python3
"""Derive KV-only controls and native-system comparisons from recorded profiles."""
import argparse,json,subprocess,sys
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);a=ap.parse_args();root=a.root.resolve()
 assert (root/'done.txt').read_text().strip()=='complete'
 subprocess.run([sys.executable,str(Path(__file__).with_name('summarize-current-gap.py')),str(root)],check=True)
 data=json.loads((root/'summary.json').read_text());cases=data['cases'];out={'kind':'development diagnostics; not formal evaluation or a measured ours INT8 implementation','controls':{},'native_comparisons':{},'target_kv_capacity':{'tokens':32768,'fp16_bytes':32768*10*2*2*256*2,'int8_with_fp32_scales_bytes':32768*10*2*2*(256+4)}}
 for tag,n in [('10k',10520),('30k',29939)]:
  i=cases[f'splash-{tag}-int8-{n}'];b=cases[f'splash-{tag}-bf16-{n}'];o=cases[f'ours-after-{n}'];ob=cases[f'ours-before-{n}']
  groups=set(i['groups_ms'])|set(b['groups_ms'])
  out['controls'][tag]={'tokens':n,'bf16_wall_ms':b['warm_wall_ms'],'int8_wall_ms':i['warm_wall_ms'],'saved_ms':b['warm_wall_ms']-i['warm_wall_ms'],'latency_reduction_pct':(1-i['warm_wall_ms']/b['warm_wall_ms'])*100,'throughput_increase_pct':(b['warm_wall_ms']/i['warm_wall_ms']-1)*100,'sampled_group_reduction_ms':{g:b['groups_ms'].get(g,0)-i['groups_ms'].get(g,0) for g in sorted(groups)},'dispatches_bf16':b['dispatches'],'dispatches_int8':i['dispatches']}
  out['native_comparisons'][tag]={'ours_before_ms':ob['warm_wall_ms'],'ours_after_ms':o['warm_wall_ms'],'splash_int8_ms':i['warm_wall_ms'],'ours_extra_ms':o['warm_wall_ms']-i['warm_wall_ms'],'ours_extra_latency_pct':(o['warm_wall_ms']/i['warm_wall_ms']-1)*100,'ours_throughput_shortfall_pct':(1-i['warm_wall_ms']/o['warm_wall_ms'])*100,'ours_vs_splash_bf16_latency_pct':(o['warm_wall_ms']/b['warm_wall_ms']-1)*100,'sampled_attention_share_pct':o['groups_ms']['Attention + preparation']/o['interval_sum_ms']*100,'ours_sampled_groups_ms':o['groups_ms'],'splash_int8_sampled_groups_ms':i['groups_ms'],'ours_after_warm_swapout_mib':o['warm_memory']['swapout_delta_gib']*1024,'ours_before_warm_swapout_mib':ob['warm_memory']['swapout_delta_gib']*1024,'splash_int8_total_swapout_mib':i['memory']['new_swapouts_mib'],'splash_bf16_total_swapout_mib':b['memory']['new_swapouts_mib']}
 (root/'kv-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
