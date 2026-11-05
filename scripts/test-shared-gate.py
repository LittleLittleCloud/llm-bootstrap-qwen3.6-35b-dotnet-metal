#!/usr/bin/env python3
"""Shared-gate sigmoid epilogue diagnostics and alternating 30K comparison."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/shared-gate'
DLL=ROOT/'artifacts/shared-gate-bin/Bootstrap.Engine.dll'
DOTNET='/Users/littlecloud/.dotnet/dotnet'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(name,cmd,extra=None):
    case=OUT/name;case.mkdir(exist_ok=False)
    env=os.environ.copy()
    for k in ['BOOTSTRAP_PROFILE','BOOTSTRAP_PROFILE_SHAPES','BOOTSTRAP_TIMELINE_FILE','MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION']:env.pop(k,None)
    if extra:env.update(extra)
    (case/'identity.json').write_text(json.dumps({'command':cmd,'environment':extra or {},'started':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
    print(name,'start',flush=True)
    with (case/'stdout.jsonl').open('w') as so,(case/'stderr.txt').open('w') as se,(case/'memory.jsonl').open('w') as mem:
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=so,stderr=se)
        while p.poll() is None:
            row={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
            for key,command in [('vm',['vm_stat']),('swap',['sysctl','vm.swapusage']),('thermal',['pmset','-g','therm'])]:row[key]=subprocess.check_output(command,text=True)
            # Associate each machine sample with the latest completed observation.
            so.flush()
            lines=(case/'stdout.jsonl').read_text().splitlines()
            for line in reversed(lines):
                try:row['latest_observation']=json.loads(line);break
                except ValueError:pass
            mem.write(json.dumps(row)+'\n');mem.flush();time.sleep(2)
        (case/'exit.txt').write_text(str(p.returncode)+'\n')
        if p.returncode:raise RuntimeError(f'{name}: exit {p.returncode}; see {case}')
    print(name,'complete',flush=True)
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'provenance.json').exists(),'Do not overwrite prior observations'
    provenance={'engine_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'scope':'Precompute shared-gate sigmoid in existing projection epilogue; FP32 storage and ordered combine; random microfixtures and same-process 30K comparison',
      'hashes':{str(p):sha(p) for p in [DLL,DLL.parent/'libbootstrap_metal.dylib',DLL.parent/'kernels.metal',ROOT/'src/Bootstrap.Engine/SharedGateTests.cs',ROOT/'src/Bootstrap.Engine/Prefill.cs',ROOT/'src/Bootstrap.Engine/WallProfile.cs']}}
    provenance['fixture']=json.loads((ROOT/'artifacts/current-gap/provenance.json').read_text())['token_checks']['30k']
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    base=[DOTNET,str(DLL),'test-shared-gate']
    run('micro',base+['--runs','9'])
    run('validation',base+['--runs','0'],{'MTL_DEBUG_LAYER':'1','MTL_SHADER_VALIDATION':'1'})
    cmd=json.loads((ROOT/'artifacts/current-gap/ours-shapes/identity.json').read_text())['command']
    cmd[1]=str(DLL);cmd[2]='compare-shared-gate'
    longer=cmd[cmd.index('--long-prompt-file')+1]
    at=cmd.index('--long-prompt-file');del cmd[at:at+2]
    assert sha(longer)==provenance['fixture']['prompt_sha256']
    cmd[cmd.index('--prompt-file')+1]=longer
    cmd[cmd.index('--runs')+1]='5'
    cmd.remove('--all-kernels')
    run('paired-30k',cmd)
    (OUT/'done.txt').write_text('complete\n')
if __name__=='__main__':main()
