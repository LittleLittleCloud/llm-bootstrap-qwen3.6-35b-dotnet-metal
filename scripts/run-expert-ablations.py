#!/usr/bin/env python3
"""Reproduce expert cost ablations; raw development evidence stays ignored."""
import argparse, hashlib, json, os, pathlib, subprocess
from importlib.machinery import SourceFileLoader
ROOT = pathlib.Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--splash',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,default=ROOT/'artifacts/expert-ablations')
    ap.add_argument('--runs',type=int,default=7)
    ap.add_argument('--label',default='forward')
    ap.add_argument('--fixture',choices=['exact','broad'],default='exact')
    ap.add_argument('--reverse',action='store_true')
    a=ap.parse_args();assert a.runs>=0
    src=a.splash.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    helper=SourceFileLoader('matched',str(ROOT/'scripts/compare-splash-kernels.py')).load_module()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip()==helper.PIN
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=src,text=True).strip()
    # The shared harness compiles the original Splash libraries even though the
    # ablations run Bootstrap only. Flatten their includes exactly as the matched runner.
    import re
    for name,rel in [('moe','shared/moe_gguf.metal'),('dense','shared/gguf_linear.metal')]:
        seen=set()
        def expand(p):
            p=p.resolve()
            if p in seen:return ''
            seen.add(p);s=re.sub(r'^\s*#pragma once\s*$','',p.read_text(),flags=re.M)
            return re.sub(r'^\s*#include "([^"]+)"\s*$',lambda m:expand(src/'runtime'/m[1]),s,flags=re.M)
        (out/(name+'.metal')).write_text('#include <metal_stdlib>\nusing namespace metal;\n'+expand(src/'runtime/metal/kernels'/rel))
    build=['xcrun','clang++','-std=c++20','-O3','-fobjc-arc','-mmacosx-version-min=26.4','-framework','Foundation','-framework','Metal','-I'+str(src/'runtime'),str(ROOT/'scripts/expert-ablation-bench.mm'),'-o',str(out/'bench')]
    subprocess.run(build,check=True)
    cmd=[str(out/'bench'),str(ROOT/'native/kernels.metal'),str(out/'moe.metal'),str(out/'dense.metal'),str(a.runs),a.fixture,'reverse' if a.reverse else 'forward']
    identity={'command':cmd,'build':build,'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'git_status':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),'shader_validation':os.getenv('MTL_SHADER_VALIDATION','0'),'hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'native/kernels.metal',ROOT/'scripts/expert-ablation-bench.mm',ROOT/'scripts/splash-kernel-bench.mm',out/'bench']}}
    (out/(a.label+'-identity.json')).write_text(json.dumps(identity,indent=2)+'\n')
    with (out/(a.label+'.jsonl')).open('w') as stdout,(out/(a.label+'.stderr')).open('w') as stderr:subprocess.run(cmd,stdout=stdout,stderr=stderr,check=True)
    (out/(a.label+'-after.json')).write_text(json.dumps({x:subprocess.check_output(c,text=True) for x,c in [('thermal',['pmset','-g','therm']),('swap',['sysctl','vm.swapusage'])]},indent=2)+'\n')
if __name__=='__main__':main()
