#!/usr/bin/env python3
"""Generate isolated, correctness-checked expert cost-sensitivity diagnostics.

Uses the existing ablation harness and its independent CPU reference. Generated
shader/harness files remain under ignored artifacts; production kernels unchanged.
Interventions do not provide additive instruction-level GPU cost attribution.
"""
import argparse, hashlib, json, os, pathlib, re, subprocess
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=pathlib.Path,required=True)
    ap.add_argument('--splash',type=pathlib.Path,required=True);ap.add_argument('--runs',type=int,default=5)
    ap.add_argument('--fixture',choices=['exact','broad'],default='exact');ap.add_argument('--reverse',action='store_true')
    ap.add_argument('--prepare-only',action='store_true');a=ap.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    shader=(ROOT/'native/kernels.metal').read_text()
    start=shader.index('template<int Rows, int KS, bool Prefetch, int Layout, int Options, class Acc>')
    end=shader.index('template [[host_name("ablation_gate_up_predecoded")]]',start)
    extra=shader[start:end].replace('expert_ablation_accumulate','profile_expert_accumulate').replace('expert_ablation_run','profile_expert_run').replace('kernel void expert_ablation(','kernel void profile_expert(')
    extra=extra.replace('half4(f8x4(bytes[j])*scale)','half4(profile_fp8<Options>(bytes[j])*scale)')
    # Both alternate decoders retain zero, subnormal, signed and NaN semantics.
    # The lookup is float32 so multiplication/FP16 rounding remain unchanged.
    values=[]
    for v in range(256):
        e=(v>>3)&15;m=v&7
        value=(1+m/8)*2**(e-7) if e else m/512
        if v&128:value=-value
        values.append('NAN' if e==15 and m==7 else float(value).hex()+'f')
    helpers='constant float profile_fp8_table[256]={'+','.join(values)+'};\n'+r'''
    template<int Options> inline float4 profile_fp8(uchar4 bytes) {
        if constexpr(Options&16) {
            ushort4 v=ushort4(bytes),e=(v>>3)&15,m=v&7;
            half4 normal=as_type<half4>(ushort4(((v&128)<<8)|((e+8)<<10)|(m<<7)));
            half4 sub=select(half4(1),half4(-1),(v&128)!=0)*half4(m)*half(1.0/512.0);
            return float4(select(select(normal,sub,e==0),half4(NAN),(e==15)&(m==7)));
        } else if constexpr(Options&32) {
            return float4(profile_fp8_table[bytes.x],profile_fp8_table[bytes.y],profile_fp8_table[bytes.z],profile_fp8_table[bytes.w]);
        } else return f8x4(bytes);
    }
    '''
    specs=[('half_bits',False,32,24),('lookup',False,32,40),('prefetch',True,32,8),('k64',False,64,8),('k128',False,128,8)]
    exports=''
    for name,prefetch,ks,opts in specs:
        for family,fused in [('gate_up',True),('matrix',False)]:
            template=f'profile_expert<{str(prefetch).lower()},{str(fused).lower()},{ks},2,{opts}>'
            exports+=f'template [[host_name("ablation_{family}_{name}")]] kernel decltype({template}) {template};\n'
    helpers += r'''
    kernel void profile_decode_values(device uchar *x [[buffer(0)]],device float *y [[buffer(1)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
        if(i>=p.n)return;
        y[i]=p.kind?profile_fp8<32>(uchar4(x[i])).x:profile_fp8<16>(uchar4(x[i])).x;
    }
    '''
    (out/'profile.metal').write_text(shader+'\n'+helpers+extra+exports)
    harness=(ROOT/'scripts/expert-ablation-bench.mm').read_text()
    # Preserve M=32 tiny-row and M=512 controls as well as M=2048 production chunks.
    begin=harness.index('   std::vector<Spec> specs=');finish=harness.index('   std::vector<Variant>',begin)
    entries=['{"staged",w.raw,w.desc,false,false,false,false}','{"packed",packed,w.desc,false,false,false,false}']
    entries += ['{"'+name+'",packed,w.desc,false,false,false,false}' for name,_,_,_ in specs]
    harness=harness[:begin]+'   std::vector<Spec> specs={'+','.join(entries)+'};\n'+harness[finish:]
    harness=harness.replace('int checks=0;double cpuMax=0;', 'if((spec.name=="half_bits"||spec.name=="lookup")&&maxerr!=0)throw std::runtime_error("exact decoder output mismatch");int checks=0;double cpuMax=0;')
    decoder_check=r'''
    static void checkDecoders(GPU&g) {
        Buffer x=g.alloc(256),y=g.alloc(256*4);for(int i=0;i<256;i++)((uint8_t*)x.contents)[i]=i;
        for(int variant=0;variant<2;variant++) {
            Args p;p.N=256;p.Kind=variant;
            g.run([&](id<MTLComputeCommandEncoder>e){g.dispatch(e,"ours","profile_decode_values",{x,y},&p,sizeof(p),15,2,1,128);});
            for(int i=0;i<256;i++) {
                int ex=(i>>3)&15,m=i&7;float actual=((float*)y.contents)[i];
                if(ex==15&&m==7){if(!std::isnan(actual))throw std::runtime_error("NaN decoder mismatch");continue;}
                float expected=ex?std::ldexp(1.f+m/8.f,ex-7):std::ldexp(float(m),-9);if(i&128)expected=-expected;
                if(memcmp(&actual,&expected,4))throw std::runtime_error("FP8 bitwise decoder mismatch");
            }
        }
        emit(@{@"type":@"decoder_check",@"codes_per_decoder":@256,@"decoders":@2,@"bitwise_finite_and_signed_zero":@YES,@"nan_preserved":@YES});
    }
    '''
    harness=harness.replace('int main(int argc,char**argv)',decoder_check+'\nint main(int argc,char**argv)')
    harness=harness.replace('GPU g(argv);ablations(', 'GPU g(argv);checkDecoders(g);ablations(')
    (out/'bench.mm').write_text(harness)
    source=a.splash.resolve()
    pin=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    assert pin=='1482f6bbc2abf12375332725302019594150b6cb'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip()
    for name,rel in [('moe','shared/moe_gguf.metal'),('dense','shared/gguf_linear.metal')]:
        seen=set()
        def expand(p):
            p=p.resolve()
            if p in seen:return ''
            seen.add(p);text=re.sub(r'^\s*#pragma once\s*$','',p.read_text(),flags=re.M)
            return re.sub(r'^\s*#include "([^"]+)"\s*$',lambda m:expand(source/'runtime'/m[1]),text,flags=re.M)
        (out/(name+'.metal')).write_text('#include <metal_stdlib>\nusing namespace metal;\n'+expand(source/'runtime/metal/kernels'/rel))
    if a.prepare_only:return
    build=['xcrun','clang++','-std=c++20','-O3','-fobjc-arc','-mmacosx-version-min=26.4','-framework','Foundation','-framework','Metal','-I'+str(source/'runtime'),'-I'+str(ROOT/'scripts'),str(out/'bench.mm'),'-o',str(out/'bench')]
    subprocess.run(build,check=True)
    label=a.fixture+('-reverse' if a.reverse else '-forward')
    command=[str(out/'bench'),str(out/'profile.metal'),str(out/'moe.metal'),str(out/'dense.metal'),str(a.runs),a.fixture,'reverse' if a.reverse else 'forward']
    paths=[out/'profile.metal',out/'bench.mm',out/'bench',out/'moe.metal',out/'dense.metal',ROOT/'scripts/splash-kernel-bench.mm',pathlib.Path(__file__).resolve()]
    (out/(label+'-identity.json')).write_text(json.dumps({'command':command,'build':build,'shader_validation':os.getenv('MTL_SHADER_VALIDATION','0'),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'git_status':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}},indent=2)+'\n')
    with (out/(label+'.jsonl')).open('w') as stdout,(out/(label+'.stderr')).open('w') as stderr:
        subprocess.run(command,stdout=stdout,stderr=stderr,check=True)
if __name__=='__main__':main()
