#!/usr/bin/env python3
"""Isolate affine expert load/stage scheduling; never changes serving shaders."""
import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('affine_profile', ROOT / 'scripts/profile-affine-splash.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

ACCUMULATE = r'''
template<int Rows,bool Prefetch,class Acc>
inline void schedule_affine_accumulate(device half*x,device const uchar*w,device const ushort*s,
    uint k,uint n,uint firstColumn,uint live,uint lane,threadgroup half*stage,thread Acc &acc) {
    constexpr auto desc=matmul2d_descriptor(Rows,32,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<1>> op;
    auto full=tensor(x,dextents<int,2>{int(k),int(live)},array<int,2>{1,int(k)});
    auto b0=tensor(stage,extents<int,32,32>{},array<int,2>{1,32});
    auto b1=tensor(stage+1024,extents<int,32,32>{},array<int,2>{1,32});
    uint row=firstColumn+lane;float scale=0,bias=0;uchar4 bytes[8];
    if constexpr(Prefetch)if(row<n){
        for(uint j=0;j<8;j++)bytes[j]=*((device const uchar4*)(w+expert_byte_offset<2>(row,j*4,k)));
        ushort2 sb=((device const ushort2*)s)[(row/32)*(k/64)*32+row%32];scale=bf(sb.x);bias=bf(sb.y);
    }
    for(uint block=0;block<k;block+=32){
        uint buffer=Prefetch?((block/32)&1):0;
        if constexpr(!Prefetch)if(row<n){
            for(uint j=0;j<8;j++)bytes[j]=*((device const uchar4*)(w+expert_byte_offset<2>(row,block+j*4,k)));
            if(block%64==0){ushort2 sb=((device const ushort2*)s)[(row/32)*(k/64)*32+(block/64)*32+row%32];scale=bf(sb.x);bias=bf(sb.y);}
        }
        for(uint j=0;j<8;j++)*((threadgroup half4*)(stage+buffer*1024+lane*32+j*4))=
            row<n?half4(fma(float4(bytes[j]),float4(scale),float4(bias))):half4(0);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if constexpr(Prefetch)if(block+32<k&&row<n){
            for(uint j=0;j<8;j++)bytes[j]=*((device const uchar4*)(w+expert_byte_offset<2>(row,block+32+j*4,k)));
            if((block+32)%64==0){ushort2 sb=((device const ushort2*)s)[(row/32)*(k/64)*32+((block+32)/64)*32+row%32];scale=bf(sb.x);bias=bf(sb.y);}
        }
        auto a=full.slice<32,Rows>(int(block),0);
        if(buffer)op.run(a,b1,acc);else op.run(a,b0,acc);
        if constexpr(!Prefetch)simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
}
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--splash', type=pathlib.Path, required=True)
    p.add_argument('--out', type=pathlib.Path, required=True)
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--label', required=True)
    p.add_argument('--reverse', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    assert a.runs >= 0
    splash = a.splash.resolve()
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=splash, text=True).strip() == base.PIN
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=splash, text=True).strip()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / (a.label + '-identity.json')).exists()
    sources = base.prepare(out, splash)
    shader = (ROOT / 'native/kernels.metal').read_text()
    start = shader.index('template<int Rows, int KS, bool Prefetch, bool Fused, int Layout, int Options>\ninline void expert_ablation_run')
    end = shader.index('template [[host_name("ablation_gate_up_predecoded")]]', start)
    body = shader[start:end].replace('expert_ablation_run', 'schedule_expert_run').replace('kernel void expert_ablation(', 'kernel void schedule_expert(')
    body = body.replace('expert_affine_accumulate<Rows>', 'schedule_affine_accumulate<Rows,Prefetch>')
    for name, prefetch in [('load_first', 'false'), ('prefetch', 'true')]:
        for family, fused in [('gate_up', 'true'), ('matrix', 'false')]:
            template = f'schedule_expert<{prefetch},{fused},32,4,0>'
            body += f'\ntemplate [[host_name("schedule_{family}_{name}")]] kernel decltype({template}) {template};\n'
    # No unrelated fixed-row/fast-math changes enter these schedule variants.
    (out / 'profile.metal').write_text(shader + '\n' + ACCUMULATE + body)
    harness = (out / 'bench.mm').read_text()
    # Remove unused fixed-row/fast-math lambda definitions and substitute scheduling only.
    start = harness.index('   auto oursFixed=')
    end = harness.index('   for(auto cv:variants)', start)
    declarations = ''
    for name in ['load_first', 'prefetch']:
        declarations += f'''   auto {name}=[&](id<MTLComputeCommandEncoder> enc){{g.dispatch(enc,"ours",gate?"schedule_gate_up_{name}":"schedule_matrix_{name}",{{ox,w.affine,w.affineDesc,ot,tc,oy}},&op,sizeof(op),15,n/(gate?32:64),std::min(rows,(rows+31)/32+256),64);}};\n'''
    declarations += '   std::vector<Variant> variants={{"affine",oursBase},{"load_first",load_first},{"prefetch",prefetch}};\n   std::vector<uint8_t> baseline;\n'
    harness = harness[:start] + declarations + harness[end:]
    harness = base.once(harness, 'g.run(ours);g.run(splash);\n   double worstO', '''g.run(ours);g.run(splash);
   size_t used=size_t(rows)*n*(gate?2:4);
   if(baseline.empty())baseline.assign((uint8_t*)oy.contents,(uint8_t*)oy.contents+used);
   else if(memcmp(baseline.data(),oy.contents,used))throw std::runtime_error("schedule changes output: "+cv.name);
   double worstO''')
    (out / 'bench.mm').write_text(harness)
    build = ['xcrun', 'clang++', '-std=c++20', '-O3', '-fobjc-arc', '-mmacosx-version-min=26.4',
             '-framework', 'Foundation', '-framework', 'Metal', '-I' + str(splash / 'runtime'), str(out / 'bench.mm'), '-o', str(out / 'bench')]
    subprocess.run(build, check=True)
    if a.prepare_only:
        return
    cmd = [str(out / 'bench'), str(out / 'profile.metal'), str(out / 'moe.metal'), str(out / 'dense.metal'), str(a.runs), 'experts', 'reverse' if a.reverse else 'forward']
    identity = {'command': cmd, 'build': build, 'splash_commit': base.PIN, 'splash_hashes': sources,
                'engine_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'shader_validation': os.getenv('MTL_SHADER_VALIDATION', '0'),
                'hashes': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                           for f in [pathlib.Path(__file__).resolve(), ROOT / 'scripts/profile-affine-splash.py', ROOT / 'native/kernels.metal',
                                     ROOT / 'scripts/splash-kernel-bench.mm', out / 'bench', out / 'bench.mm', out / 'profile.metal']}}
    (out / (a.label + '-identity.json')).write_text(json.dumps(identity, indent=2) + '\n')
    with (out / (a.label + '.jsonl')).open('w') as stdout, (out / (a.label + '.stderr')).open('w') as stderr:
        subprocess.run(cmd, stdout=stdout, stderr=stderr, check=True)


if __name__ == '__main__':
    main()
