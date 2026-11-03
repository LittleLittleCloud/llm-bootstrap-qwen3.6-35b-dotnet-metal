#!/usr/bin/env python3
"""Controlled FP8/affine expert comparison within one TensorOps kernel skeleton.

Only generated diagnostic shaders/harnesses are changed. Fixture values are
exactly representable in both formats; no whole-model requantization is done.
"""
import argparse, datetime, hashlib, json, os, pathlib, re, subprocess
ROOT=pathlib.Path(__file__).resolve().parents[1]
PIN='1482f6bbc2abf12375332725302019594150b6cb'

def replace_once(s,old,new):
    assert s.count(old)==1,old[:90]
    return s.replace(old,new)

def generate(out,source):
    shader=(ROOT/'native/kernels.metal').read_text()
    start=shader.index('template<int Rows, int KS, bool Prefetch, int Layout, int Options, class Acc>')
    end=shader.index('template [[host_name("ablation_gate_up_predecoded")]]',start)
    body=shader[start:end].replace('expert_ablation_accumulate','quant_compare_accumulate').replace('expert_ablation_run','quant_compare_run').replace('kernel void expert_ablation(','kernel void quant_compare(')
    body=replace_once(body,'uchar4 bytes[KS/4];float scale=0;', 'uchar4 bytes[KS/4];float scale=0,bias=0;')
    old='if(!Prefetch&&Layout!=-1&&row<n){for(uint j=0;j<KS/4;j++)bytes[j]=*((device const uchar4*)(w+expert_byte_offset<(Layout==3?2:Layout)>(row,block+j*4,k)));if(!(Options&8)||block%128==0)scale=bf(s[(row/(Layout==3?32:128))*(k/128)+block/128]);}'
    new='''if(!Prefetch&&row<n){
        for(uint j=0;j<KS/4;j++)bytes[j]=*((device const uchar4*)(w+expert_byte_offset<2>(row,block+j*4,k)));
        if constexpr(Options&64){
            if(block%64==0){uint at=(row/32)*(k/64)*32+(block/64)*32+row%32;
                ushort2 sb=((device const ushort2*)s)[at];scale=bf(sb.x);bias=bf(sb.y);}
        }else if(block%128==0)scale=bf(s[(row/128)*(k/128)+block/128]);
    }'''
    body=replace_once(body,old,new)
    body=replace_once(body,'half4(f8x4(bytes[j])*scale)','quant_compare_decode<Options>(bytes[j],scale,bias)')
    values=[]
    for v in range(256):
        e=(v>>3)&15;m=v&7;value=(1+m/8)*2**(e-7) if e else m/512
        if v&128:value=-value
        values.append('NAN' if e==15 and m==7 else float(value).hex()+'f')
    helpers='constant float quant_compare_table[256]={'+','.join(values)+'};\n'+r'''
    template<int Options> inline half4 quant_compare_decode(uchar4 bytes,float scale,float bias){
        float4 value;
        if constexpr(Options&128){
            if constexpr(Options&256){
                // Same 1024+code exact-half technique as Splash staged_linear,
                // adapted to the common four-consecutive-byte input layout.
                uint word=as_type<uint>(bytes);
                half2 even=as_type<half2>((word&0x00ff00ffu)|0x64006400u)-half2(1024);
                half2 odd=as_type<half2>(((word>>8)&0x00ff00ffu)|0x64006400u)-half2(1024);
                value=float4(even.x,odd.x,even.y,odd.y);
            }else value=float4(bytes);
        }else if constexpr(Options&16){
            value=float4(quant_compare_table[bytes.x],quant_compare_table[bytes.y],quant_compare_table[bytes.z],quant_compare_table[bytes.w]);
        }else value=f8x4(bytes);
        if constexpr(Options&64)return half4(fma(value,float4(scale),float4(bias)));
        else return half4(value*scale);
    }
    kernel void quant_compare_codes(device uchar*x [[buffer(0)]],device float*y [[buffer(1)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]){
        if(i>=p.n)return;
        uchar4 b=uchar4(x[i]);
        y[i]=p.kind==0?float(quant_compare_decode<200>(b,p.scale,p.value).x):float(quant_compare_decode<456>(b,p.scale,p.value).x);
    }
    '''
    specs=[('fp8_native',8),('fp8_native_lut',24),('fp8_g64',72),('fp8_g64_lut',88),('affine_g64_cast',200),('affine_g64_bits',456)]
    exports=''
    for name,opts in specs:
        for family,fused in [('gate_up',True),('matrix',False)]:
            specialization=f'quant_compare<false,{str(fused).lower()},32,2,{opts}>'
            exports+=f'template [[host_name("ablation_{family}_{name}")]] kernel decltype({specialization}) {specialization};\n'
    (out/'compare.metal').write_text(shader+'\n'+helpers+body+exports)
    harness=(ROOT/'scripts/expert-ablation-bench.mm').read_text()
    # Reuse allocation/reference/routing/timer logic; replace only fixture formats.
    allocation='''
  size_t groupSlot=size_t(n)*k+size_t(n)*(k/64)*4;
  Buffer fp64=g.alloc(groupSlot*256*choices),af64=g.alloc(groupSlot*256*choices),gd=g.alloc(w.desc.length);
  memcpy(gd.contents,w.desc.contents,w.desc.length);
'''
    harness=replace_once(harness,'  for(int e=0;e<256;e++)for(int h=0;h<choices;h++) {',allocation+'  for(int e=0;e<256;e++)for(int h=0;h<choices;h++) {')
    a=harness.index('   if(broad){');b=harness.index('   memcpy((uint8_t*)packed.contents',a)
    harness=harness[:a]+'''   if(broad)for(int c=0;c<n;c+=128)for(int j=0;j<k;j+=128)
    scales[c/128*(k/128)+j/128]=bf(scale(e,c,j,h)*(1.f+float((e+h+c/128+j/128)%23)/128.f));
'''+harness[b:]
    fixture=r'''
   size_t go=(size_t(e)*choices+h)*groupSlot;
   for(int d=0;d<3;d++)if(choices==1||d==h){((uint64_t*)gd.contents)[e*6+2*d]=go;((uint64_t*)gd.contents)[e*6+2*d+1]=go+size_t(n)*k;}
   for(int c=0;c<n;c++)for(int j=0;j<k;j++){
    size_t at=(size_t(c)/32)*k*32+(j/32)*1024+((j%32)/4)*128+(c%32)*4+j%4;
    ((uint8_t*)fp64.contents)[go+at]=bytes[size_t(c)*k+j];
    ((uint8_t*)af64.contents)[go+at]=uint8_t((code(e,c,j,h)+16)*8);
   }
   for(int c=0;c<n;c++)for(int j=0;j<k;j+=64){
    size_t at=((size_t(c)/32)*(k/64)*32+(j/64)*32+c%32)*2;
    auto fs=(uint16_t*)((uint8_t*)fp64.contents+go+size_t(n)*k),as=(uint16_t*)((uint8_t*)af64.contents+go+size_t(n)*k);
    float sc=unbf(scales[c/128*(k/128)+j/128]);
    fs[at]=bf(sc);fs[at+1]=bf(0.f);as[at]=bf(sc/8);as[at+1]=bf(-16*sc);
   }
   // Exhaustively verify every scalar weight, including descriptor/metadata addressing.
   for(int c=0;c<n;c++)for(int j=0;j<k;j++){
    size_t at=(size_t(c)/32)*k*32+(j/32)*1024+((j%32)/4)*128+(c%32)*4+j%4;
    size_t mt=((size_t(c)/32)*(k/64)*32+(j/64)*32+c%32)*2;
    auto fs=(uint16_t*)((uint8_t*)fp64.contents+go+size_t(n)*k),as=(uint16_t*)((uint8_t*)af64.contents+go+size_t(n)*k);
    uint8_t v=((uint8_t*)fp64.contents)[go+at];int ex=(v>>3)&15,mant=v&7;
    float f=ex?std::ldexp(1.f+mant/8.f,ex-7):std::ldexp(float(mant),-9);if(v&128)f=-f;
    float a=float(((uint8_t*)af64.contents)[go+at])*unbf(as[mt])+unbf(as[mt+1]);
    float ref=code(e,c,j,h)*unbf(scales[c/128*(k/128)+j/128]);
    if(f*unbf(fs[mt])+unbf(fs[mt+1])!=ref||a!=ref)throw std::runtime_error("decoded formats differ");
    _Float16 hf=(_Float16)a,original=((_Float16*)((uint8_t*)decoded.contents+off*2))[size_t(c)*k+j];
    if(hf!=original)throw std::runtime_error("operand rounding differs");
   }
'''
    marker='  for(int m:{32,512,2048})for(bool skew:{false,true}) {'
    pos=harness.index(marker)
    closing=harness.rfind('  }',0,pos)
    harness=harness[:closing]+fixture+harness[closing:]
    harness=replace_once(harness,marker,'''  emit(@{@"type":@"weight_check",@"family":fused?@"gate_up":@"down",@"values":@(size_t(256)*choices*n*k),@"exactly_equal_before_operand_rounding":@YES,@"operand_values_equal":@YES});
'''+marker)
    begin=harness.index('   std::vector<Spec> specs=');finish=harness.index('   std::vector<Variant>',begin)
    entries=['{"packed",packed,w.desc,false,false,false,false}']
    for name,_ in specs:
        raw,desc=('af64','gd') if name.startswith('affine') else ('fp64','gd') if '_g64' in name else ('packed','w.desc')
        entries.append('{"'+name+'",'+raw+','+desc+',false,false,false,false}')
    harness=harness[:begin]+'   std::vector<Spec> specs={'+','.join(entries)+'};\n'+harness[finish:]
    harness=replace_once(harness,'if(spec.name=="staged")baseline.push_back(v);','if(spec.name=="packed")baseline.push_back(v);')
    harness=replace_once(harness,'int checks=0;double cpuMax=0;','if(maxerr!=0)throw std::runtime_error("common kernel outputs differ: "+spec.name);int checks=0;double cpuMax=0;')
    harness=harness.replace('@"broad_fp8"', '@"varied_scales"')
    # Independent all-code check for the two affine conversion implementations.
    check_codes=r'''
    static void checkAffine(GPU&g){
      Buffer x=g.alloc(256),y=g.alloc(256*4);for(int i=0;i<256;i++)((uint8_t*)x.contents)[i]=i;
      int count=0;
      for(int variant=0;variant<2;variant++)for(float sc:{0.0078125f,0.0091552734375f,-0.015625f})for(float bias:{-1.0f,0.0f,0.3125f}){
       Args p;p.N=256;p.Kind=variant;p.Scale=sc;p.Value=bias;
       g.run([&](id<MTLComputeCommandEncoder>e){g.dispatch(e,"ours","quant_compare_codes",{x,y},&p,sizeof(p),15,2,1,128);});
       for(int i=0;i<256;i++){float expected=float((_Float16)(double(i)*sc+bias));if(((float*)y.contents)[i]!=expected)throw std::runtime_error("affine code check");count++;}
      }
      emit(@{@"type":@"affine_code_check",@"reference_points":@(count),@"all_codes":@YES});
    }
    '''
    harness=replace_once(harness,'int main(int argc,char**argv)',check_codes+'\nint main(int argc,char**argv)')
    harness=replace_once(harness,'GPU g(argv);ablations(','GPU g(argv);checkAffine(g);ablations(')
    (out/'bench.mm').write_text(harness)
    for name,rel in [('moe','shared/moe_gguf.metal'),('dense','shared/gguf_linear.metal')]:
        seen=set()
        def expand(p):
            p=p.resolve()
            if p in seen:return ''
            seen.add(p);text=re.sub(r'^\s*#pragma once\s*$','',p.read_text(),flags=re.M)
            return re.sub(r'^\s*#include "([^"]+)"\s*$',lambda m:expand(source/'runtime'/m[1]),text,flags=re.M)
        (out/(name+'.metal')).write_text('#include <metal_stdlib>\nusing namespace metal;\n'+expand(source/'runtime/metal/kernels'/rel))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=pathlib.Path,required=True);ap.add_argument('--splash',type=pathlib.Path,required=True)
    ap.add_argument('--runs',type=int,default=5);ap.add_argument('--label',required=True);ap.add_argument('--varied-scales',action='store_true');ap.add_argument('--reverse',action='store_true');a=ap.parse_args()
    if a.runs<0:ap.error('runs must be nonnegative')
    out=a.out.resolve();source=a.splash.resolve();out.mkdir(parents=True,exist_ok=True)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()==PIN
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip()
    if (out/(a.label+'.jsonl')).exists():raise SystemExit('Refusing to overwrite existing evidence; choose a new label/directory')
    generate(out,source)
    build=['xcrun','clang++','-std=c++20','-O3','-fobjc-arc','-mmacosx-version-min=26.4','-framework','Foundation','-framework','Metal','-I'+str(source/'runtime'),'-I'+str(ROOT/'scripts'),str(out/'bench.mm'),'-o',str(out/'bench')]
    subprocess.run(build,check=True)
    cmd=[str(out/'bench'),str(out/'compare.metal'),str(out/'moe.metal'),str(out/'dense.metal'),str(a.runs),'broad' if a.varied_scales else 'exact','reverse' if a.reverse else 'forward']
    paths=[out/n for n in ['compare.metal','bench.mm','bench','moe.metal','dense.metal']]+[ROOT/'native/kernels.metal',ROOT/'scripts/expert-ablation-bench.mm',ROOT/'scripts/splash-kernel-bench.mm',pathlib.Path(__file__).resolve()]
    identity={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':cmd,'build':build,'math_mode':'safe, all six variants','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'git_status':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),'shader_validation':os.getenv('MTL_SHADER_VALIDATION','0'),'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (out/(a.label+'-identity.json')).write_text(json.dumps(identity,indent=2)+'\n')
    with (out/(a.label+'.jsonl')).open('w') as stdout,(out/(a.label+'.stderr')).open('w') as stderr:subprocess.run(cmd,stdout=stdout,stderr=stderr,check=True)
    (out/(a.label+'-after.json')).write_text(json.dumps({key:subprocess.check_output(cmd,text=True) for key,cmd in [('thermal',['pmset','-g','therm']),('swap',['sysctl','vm.swapusage'])]},indent=2)+'\n')
if __name__=='__main__':main()
