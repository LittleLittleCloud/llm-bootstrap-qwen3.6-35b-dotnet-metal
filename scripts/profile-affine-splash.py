#!/usr/bin/env python3
"""Profile current affine kernels against pinned Splash; generated interventions only.

The fixed-row diagnostic keeps compact live rows and allocates 32 trailing rows.
It permits unused tensor lanes to read adjacent rows but stores only live outputs.
It is not safe to port without proving the production allocation bound.
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
PIN = '1482f6bbc2abf12375332725302019594150b6cb'


def once(text, old, new):
    assert text.count(old) == 1, (old[:100], text.count(old))
    return text.replace(old, new)


def prepare(out, splash):
    shader = (ROOT / 'native/kernels.metal').read_text()
    start = shader.index('template<int Rows, class Acc>\ninline void expert_affine_accumulate')
    end = shader.index('template [[host_name("ablation_gate_up_predecoded")]]', start)
    extra = shader[start:end]
    for name in ['expert_affine_accumulate', 'expert_ablation_accumulate', 'expert_ablation_run', 'expert_ablation(']:
        extra = extra.replace(name, 'profile_' + name)
    extra = extra.replace('int(k),int(live)', 'int(k),Rows').replace('int(p.k),int(tile.count)', 'int(p.k),Rows')
    for family, fused in [('gate_up', 'true'), ('matrix', 'false')]:
        template = f'profile_expert_ablation<false,{fused},32,4,0>'
        extra += f'\ntemplate [[host_name("profile_{family}_fixed")]] kernel decltype({template}) {template};\n'
    # Name replacement above only changes the declaration, not template references.
    extra = extra.replace('kernel void profile_expert_ablation(', 'kernel void profile_expert_ablation(')
    (out / 'profile.metal').write_text(shader + '\n' + extra)

    harness = (ROOT / 'scripts/splash-kernel-bench.mm').read_text()
    harness = once(harness, 'for(int i=1;i<=3;i++)', 'for(int i=1;i<=4;i++)')
    harness = once(harness, '@(argv[i]) encoding:', '@(argv[i==4?1:i]) encoding:')
    harness = once(harness, 'i==1?"ours":i==2?"moe":"dense"', 'i==1?"ours":i==2?"moe":i==3?"dense":"ours_fast"')
    harness = once(harness, 'Buffer raw,desc,plane,meta,decoded;', 'Buffer raw,desc,plane,meta,decoded,affine,affineMeta,affineDesc;')
    allocation = '''
  size_t affineSlot=nk+size_t(n)*(k/64)*4;
  affine=g.alloc(size_t(experts)*choices*affineSlot);
  affineMeta=g.alloc(size_t(experts)*choices*n*(k/64)*4);
  affineDesc=g.alloc(size_t(experts)*48);
'''
    harness = once(harness, '  if(dense)decoded=g.alloc(nk*2);', allocation + '  if(dense)decoded=g.alloc(nk*2);')
    fill = '''
   size_t ao=(size_t(e)*choices+h)*affineSlot;
   for(int d=0;d<3;d++)if(choices==1||d==h){((uint64_t*)affineDesc.contents)[e*6+d*2]=ao;((uint64_t*)affineDesc.contents)[e*6+d*2+1]=ao+nk;}
   for(int c=0;c<n;c++)for(int j=0;j<k;j++){
    size_t at=dense?size_t(c)*k+j:(size_t(c)/32)*k*32+(j/32)*1024+((j%32)/4)*128+(c%32)*4+j%4;
    ((uint8_t*)affine.contents)[ao+at]=uint8_t((code(e,c,j,h)+16)*8);
   }
   auto am=(uint16_t*)((uint8_t*)affine.contents+ao+nk);
   for(int c=0;c<n;c++)for(int j=0;j<k;j+=64){
    size_t at=(dense?size_t(c)*(k/64)+j/64:(size_t(c)/32)*(k/64)*32+(j/64)*32+c%32)*2;
    am[at]=bf(scale(e,c,j,h)/8);am[at+1]=bf(-16*scale(e,c,j,h));
   }
   for(int c=0;c<n;c++)for(int j=0;j<k;j++){
    size_t at=dense?size_t(c)*k+j:(size_t(c)/32)*k*32+(j/32)*1024+((j%32)/4)*128+(c%32)*4+j%4;
    size_t mt=(dense?size_t(c)*(k/64)+j/64:(size_t(c)/32)*(k/64)*32+(j/64)*32+c%32)*2;
    float v=std::fma(float(((uint8_t*)affine.contents)[ao+at]),unbf(am[mt]),unbf(am[mt+1]));
    if(v!=weight(e,c,j,h))throw std::runtime_error("affine fixture mismatch");
   }
   if(dense)memcpy(affineMeta.contents,am,size_t(n)*(k/64)*4);
'''
    harness = once(harness, '\n  }\n }\n};\nstruct Variant', fill + '\n  }\n }\n};\nstruct Variant')
    # Original FP8 inputs remain only for independent fixture construction.
    harness = once(harness, 'for(int m:{512,2048})', 'for(int m:{32,512,2048})')
    harness = once(harness, 'auto ours=[&](id<MTLComputeCommandEncoder> enc){g.dispatch(enc,"ours",gate?"expert_gate_up_staged":"expert_matrix_staged",{ox,w.raw,w.desc,ot,tc,oy}',
                   'auto oursBase=[&](id<MTLComputeCommandEncoder> enc){g.dispatch(enc,"ours",gate?"expert_gate_up_affine":"expert_matrix_affine",{ox,w.affine,w.affineDesc,ot,tc,oy}')
    variants = '''
   auto oursFixed=[&](id<MTLComputeCommandEncoder> enc){g.dispatch(enc,"ours",gate?"profile_gate_up_fixed":"profile_matrix_fixed",{ox,w.affine,w.affineDesc,ot,tc,oy},&op,sizeof(op),15,n/(gate?32:64),std::min(rows,(rows+31)/32+256),64);};
   auto oursFast=[&](id<MTLComputeCommandEncoder> enc){g.dispatch(enc,"ours_fast",gate?"expert_gate_up_affine":"expert_matrix_affine",{ox,w.affine,w.affineDesc,ot,tc,oy},&op,sizeof(op),15,n/(gate?32:64),std::min(rows,(rows+31)/32+256),64);};
   auto oursFixedFast=[&](id<MTLComputeCommandEncoder> enc){g.dispatch(enc,"ours_fast",gate?"profile_gate_up_fixed":"profile_matrix_fixed",{ox,w.affine,w.affineDesc,ot,tc,oy},&op,sizeof(op),15,n/(gate?32:64),std::min(rows,(rows+31)/32+256),64);};
   std::vector<Variant> variants={{"affine",oursBase},{"fixed_rows",oursFixed},{"fast_math",oursFast},{"fixed_fast",oursFixedFast}};
   for(auto cv:variants){auto ours=cv.encode;
'''
    harness = once(harness, '   initOutput(oy,size_t(rows)*n*(gate?2:4));', variants + '   initOutput(oy,size_t(rows)*n*(gate?2:4));')
    harness = once(harness, '@"cpu_reference_points":@(checks),@"ours_max_abs"', '@"variant":@(cv.name.c_str()),@"cpu_reference_points":@(checks),@"ours_max_abs"')
    harness = once(harness, '   measure(g,{{"ours",ours},{"splash",splash}},', '   }\n   variants.push_back({"splash",splash});\n   measure(g,variants,')
    # Dense compares the actual affine decode + existing FP16 TensorOps path.
    harness = once(harness, 'Args op;op.N=n;op.M=m;op.K=k;', 'Args op;op.N=n;op.M=m;op.K=k;op.Kind=3;')
    harness = once(harness, '{w.raw,scales,decoded}', '{w.affine,w.affineMeta,decoded}')
    (out / 'bench.mm').write_text(harness)
    source_hashes = {}
    for name, rel in [('moe', 'shared/moe_gguf.metal'), ('dense', 'shared/gguf_linear.metal')]:
        seen = set()

        def expand(p):
            p = p.resolve()
            if p in seen:
                return ''
            seen.add(p)
            source_hashes[str(p.relative_to(splash))] = hashlib.sha256(p.read_bytes()).hexdigest()
            text = re.sub(r'^\s*#pragma once\s*$', '', p.read_text(), flags=re.M)
            return re.sub(r'^\s*#include "([^"]+)"\s*$', lambda m: expand(splash / 'runtime' / m[1]), text, flags=re.M)

        (out / (name + '.metal')).write_text('#include <metal_stdlib>\nusing namespace metal;\n' + expand(splash / 'runtime/metal/kernels' / rel))
    return source_hashes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--splash', type=pathlib.Path, required=True)
    p.add_argument('--out', type=pathlib.Path, required=True)
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--label', required=True)
    p.add_argument('--family', choices=['all', 'experts', 'dense'], default='all')
    p.add_argument('--reverse', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    if a.runs < 0:
        p.error('runs must be nonnegative')
    splash = a.splash.resolve()
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=splash, text=True).strip() == PIN
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=splash, text=True).strip()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / (a.label + '-identity.json')).exists(), 'Do not overwrite observations'
    sources = prepare(out, splash)
    build = ['xcrun', 'clang++', '-std=c++20', '-O3', '-fobjc-arc', '-mmacosx-version-min=26.4',
             '-framework', 'Foundation', '-framework', 'Metal', '-I' + str(splash / 'runtime'),
             str(out / 'bench.mm'), '-o', str(out / 'bench')]
    subprocess.run(build, check=True)
    if a.prepare_only:
        return
    cmd = [str(out / 'bench'), str(out / 'profile.metal'), str(out / 'moe.metal'), str(out / 'dense.metal'),
           str(a.runs), a.family, 'reverse' if a.reverse else 'forward']
    identity = {'command': cmd, 'build': build, 'splash_commit': PIN, 'splash_hashes': sources,
                'engine_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'git_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
                'shader_validation': os.getenv('MTL_SHADER_VALIDATION', '0'),
                'hashes': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                           for f in [pathlib.Path(__file__).resolve(), ROOT / 'native/kernels.metal',
                                     ROOT / 'scripts/splash-kernel-bench.mm', out / 'bench', out / 'bench.mm', out / 'profile.metal']}}
    (out / (a.label + '-identity.json')).write_text(json.dumps(identity, indent=2) + '\n')
    with (out / (a.label + '.jsonl')).open('w') as stdout, (out / (a.label + '.stderr')).open('w') as stderr:
        subprocess.run(cmd, stdout=stdout, stderr=stderr, check=True)


if __name__ == '__main__':
    main()
