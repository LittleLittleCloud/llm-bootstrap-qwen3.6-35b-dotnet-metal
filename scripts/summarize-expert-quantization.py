#!/usr/bin/env python3
"""Summarize controlled expert format timings, preserving all variants and runs."""
import argparse,collections,json,pathlib,statistics

def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=pathlib.Path);a=ap.parse_args();root=a.root
    passes={label:[json.loads(line) for line in (root/(label+'.jsonl')).read_text().splitlines()] for label in ['validation','forward','reverse']}
    identities={label:json.loads((root/(label+'-identity.json')).read_text()) for label in passes}
    for name in ['compare.metal','native/kernels.metal','scripts/splash-kernel-bench.mm']:
        values=[next(v for k,v in i['sha256'].items() if k.endswith('/'+name)) for i in identities.values()]
        assert len(set(values))==1, ('different shader or timer',name)
    for name in ['bench.mm','scripts/compare-expert-quantization.py']:
        values=[next(v for k,v in identities[label]['sha256'].items() if k.endswith('/'+name)) for label in ['forward','reverse']]
        assert len(set(values))==1, ('different timing harness',name)
    assert identities['validation']['shader_validation']=='1'
    assert all(identities[label]['shader_validation']=='0' for label in ['forward','reverse'])
    allchecks=[];weights=[];codechecks=[];timings=[];grouped=collections.defaultdict(list)
    for label,rows in passes.items():
        check=[r for r in rows if r['type']=='check']
        assert len(check) in ([72,84] if label=='validation' else [84])
        assert {c['variant'] for c in check}==({'fp8_native','fp8_native_lut','fp8_g64','fp8_g64_lut','affine_g64_cast','affine_g64_bits'}|({'packed'} if len(check)==84 else set()))
        assert all(r['max_abs_vs_baseline']==0 and r['relative_l2']==0 and r['guards'] and r['all_live_outputs_finite'] for r in check)
        wc=[r for r in rows if r['type']=='weight_check'];assert len(wc)==2
        assert all(r['exactly_equal_before_operand_rounding'] and r['operand_values_equal'] for r in wc)
        cc=[r for r in rows if r['type']=='affine_code_check'];assert len(cc)==1 and cc[0]['reference_points']==4608
        allchecks+=check;weights+=wc;codechecks+=cc
        ts=[dict(r,pass_name=label) for r in rows if r['type']=='timing'];assert len(ts)==(0 if label=='validation' else 420)
        timings+=ts
        for r in ts:grouped[(r['family'],r['m'],r['routing'],r['variant'])].append(r)
    summary={'kind':'controlled diagnostic, not endpoint evaluation','shader_mode':'Metal 4.0 safe, every measured variant','source_commit':identities['forward']['source_commit'],'validation':{'matrix_cases':len(allchecks),'matrix_cpu_reference_points':sum(r['cpu_reference_points'] for r in allchecks),'max_relative_l2':max(r['relative_l2'] for r in allchecks),'scalar_weight_checks_including_repeats':sum(r['values'] for r in weights),'all_code_reference_points_including_repeats':sum(r['reference_points'] for r in codechecks),'time_samples':len(timings)},'medians':[],'comparisons':[]}
    for (family,m,routing,variant),rows in sorted(grouped.items()):
        assert len(rows)==10
        summary['medians'].append({'family':family,'m':m,'routing':routing,'variant':variant,'gpu_ms':statistics.median(r['gpu_ms'] for r in rows),'range_ms':[min(r['gpu_ms'] for r in rows),max(r['gpu_ms'] for r in rows)],'pass_medians':{label:statistics.median(r['gpu_ms'] for r in rows if r['pass_name']==label) for label in ['forward','reverse']}})
    lookup={(r['family'],r['m'],r['routing'],r['variant']):r['gpu_ms'] for r in summary['medians']}
    for family in ['gate_up','down']:
        for m in [32,512,2048]:
            for routing in ['uniform','skewed']:
                get=lambda v:lookup[family,m,routing,v]
                relative=lambda a,b:(1-get(b)/get(a))*100
                summary['comparisons'].append({'family':family,'m':m,'routing':routing,'generated_fp8_vs_unmodified_change_pct':(get('fp8_native')/get('packed')-1)*100,'same_metadata_affine_cast_time_saving_pct':relative('fp8_g64','affine_g64_cast'),'same_metadata_affine_bits_time_saving_pct':relative('fp8_g64','affine_g64_bits'),'same_metadata_affine_bits_vs_fp8_lut_saving_pct':relative('fp8_g64_lut','affine_g64_bits'),'affine_bits_vs_current_fp8_saving_pct':relative('packed','affine_g64_bits'),'affine_bits_vs_native_fp8_lut_saving_pct':relative('fp8_native_lut','affine_g64_bits'),'metadata_control_vs_native_fp8_change_pct':(get('fp8_g64')/get('fp8_native')-1)*100})
    summary['limitations']=['Shared 32-level exactly representable synthetic weights; not real-model requantization or quality evidence.','Group-64 FP8 with a runtime zero bias is a decoder control, not the production FP8 format.','Payload byte permutation, metadata layout within group-64, grid, activations, tiles, epilogue and math mode are held fixed.','Native FP8 and group-64 comparisons also change metadata footprint, access frequency, scale/bias loading and multiply/FMA.','This ports affine conversion into our kernel. It does not measure all of Splash or establish a whole-model speedup.']
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['validation'],indent=2))
    for c in summary['comparisons']:
        if c['m']==2048:print(json.dumps(c))
if __name__=='__main__':main()
