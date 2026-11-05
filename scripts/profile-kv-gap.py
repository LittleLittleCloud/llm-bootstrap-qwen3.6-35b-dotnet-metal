#!/usr/bin/env python3
"""Refresh native prefill profiles; change only Splash KV format in its controls."""
import argparse,datetime,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prior',type=Path,required=True);ap.add_argument('--dll',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);dll=a.dll.resolve();prior=json.loads((a.prior/'plan.json').read_text())
 pin=subprocess.check_output(['git','rev-parse','HEAD'],cwd=dll.parent,text=True).strip()
 assert pin=='41f8e470efc1e4628a467936eddcbdd618be1d3f'
 assert sha(dll.parent/'kernels.metal')==sha(ROOT/'native/kernels.metal')
 assert sha(dll)=='4fbfe7799c8f07fa8a35182301974bc700b19a166e0173ba90eb8c55223d2778'
 plan=[];checks={}
 base=prior[0][1];base[1]=str(dll);base+=['--bf16-expert-output']
 for tag,flag in [('10k','--prompt-file'),('30k','--long-prompt-file')]:
  prompt=Path(base[base.index(flag)+1]);splash=next(cmd for name,cmd in prior if name==f'splash-{tag}-int8');ids=Path(splash[splash.index('--tokens-file')+1]);expected=list(map(int,ids.read_text().split()))
  actual=json.loads(subprocess.check_output([base[0],str(dll),'tokenize','--model-dir',base[base.index('--model-dir')+1],'--text',prompt.read_text()],text=True))['chat']
  assert actual==expected
  checks[tag]={'tokens':len(actual),'all_ids_equal':True,'prompt_sha256':sha(prompt),'ids_sha256':sha(ids)}
 for name,cmd in prior:
  if name.startswith('ours'):cmd=list(base)
  else:
   cmd=list(cmd);cmd[cmd.index('--trace-out')+1]=str(out/name/'intervals.jsonl')
   assert sha(cmd[0])=='def70bddd367b94a9ab7aab520f60fc4ddf7d14c12638c0c3c3c2308cab8cbce'
   assert sha(cmd[1])=='e30124e8bceba1a9de7afe75cf1a10439bcfa5c1b777490c829ba4da0d0cd1cd'
  plan.append((name,cmd))
 (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 (out/'provenance.json').write_text(json.dumps({'engine_commit':pin,'splash_commit':'1482f6bbc2abf12375332725302019594150b6cb','token_checks':checks,'hashes':{str(p):sha(p) for p in [dll,dll.parent/'kernels.metal',dll.parent/'libbootstrap_metal.dylib',Path(plan[1][1][0]),Path(plan[1][1][1])]},'scope':'Native model/activation/draft formats differ. First token and draft setup included; no prefix reuse. Splash has all experts resident; this is not the streaming endpoint configuration. Ours enables BF16 expert output with FP16 KV.'},indent=2)+'\n')
 for name,cmd in plan:
  p=out/name;p.mkdir();env={k:v for k,v in os.environ.items() if not k.startswith(('BOOTSTRAP_PROFILE','BOOTSTRAP_TIMELINE','SPLASH_COUNTER','MTL_'))}
  if name.startswith('ours'):env['BOOTSTRAP_PROFILE_SHAPES']='1'
  (p/'identity.json').write_text(json.dumps({'command':cmd,'started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'shape_labels':name.startswith('ours')},indent=2)+'\n')
  print(name,'start',flush=True)
  with (p/'stdout.jsonl').open('w') as stdout,(p/'stderr.txt').open('w') as stderr,(p/'memory.jsonl').open('w') as memory:
   proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
   (p/'pid.txt').write_text(str(proc.pid))
   while proc.poll() is None:
    row={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':proc.pid}
    for key,c in [('vm',['vm_stat']),('swap',['sysctl','vm.swapusage']),('thermal',['pmset','-g','therm'])]:row[key]=subprocess.check_output(c,text=True)
    memory.write(json.dumps(row)+'\n');memory.flush();time.sleep(3)
   (p/'exit.txt').write_text(str(proc.returncode))
   if proc.returncode:raise RuntimeError(f'{name}: exit {proc.returncode}')
  print(name,'complete',flush=True)
 (out/'done.txt').write_text('complete\n')
if __name__=='__main__':main()
