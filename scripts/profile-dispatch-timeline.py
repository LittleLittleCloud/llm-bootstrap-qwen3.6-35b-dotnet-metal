#!/usr/bin/env python3
"""Run native 10K prefill diagnostics serially; no formal evaluation artifacts."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/dispatch-timeline'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ['ours-fp16', 'splash-int8', 'splash-bf16']:
        if (out/name).exists():
            raise FileExistsError(f'{out/name}: choose a new --out directory')
    old = ROOT / 'artifacts/current-gap'
    ours = json.loads((old/'ours-shapes/identity.json').read_text())['command']
    ours[1] = str(ROOT/'artifacts/timeline-bin/Bootstrap.Engine.dll')
    i = ours.index('--long-prompt-file')
    del ours[i:i+2]
    plan = [('ours-fp16', ours)]
    for kv in ['int8', 'bf16']:
        cmd = json.loads((old/f'splash-10k-{kv}/identity.json').read_text())['command']
        cmd[0] = str(OUT/'splash-profile')
        cmd[cmd.index('--trace-out')+1] = str(out/f'splash-{kv}/intervals.jsonl')
        plan.append((f'splash-{kv}', cmd))
    env = os.environ.copy()
    for key in ['BOOTSTRAP_PROFILE', 'BOOTSTRAP_PROFILE_SHAPES', 'BOOTSTRAP_TIMELINE_FILE',
                'SPLASH_TIMELINE_FILE', 'SPLASH_COUNTER_TRACE_FILE', 'MTL_DEBUG_LAYER', 'MTL_SHADER_VALIDATION']:
        env.pop(key, None)
    provenance = {'scope': '10520 identical token IDs; native weights/precision differ; cold + 3 untraced warm + CPU trace + GPU counters',
                  'engine_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  'splash_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd='/tmp/splash-timeline-1482',text=True).strip(),
                  'hashes': {str(p):sha(p) for p in [Path(ours[1]), Path(ours[1]).parent/'libbootstrap_metal.dylib', Path(ours[1]).parent/'kernels.metal', OUT/'splash-profile', OUT/'MetalBackend.o', Path('/tmp/splash-profile-1482/build/engine/libsplash.a'), Path(plan[1][1][1])]},
                  'os': subprocess.check_output(['sw_vers'],text=True), 'plan':plan}
    token_path = Path(plan[1][1][plan[1][1].index('--tokens-file')+1])
    prompt_path = Path(ours[ours.index('--prompt-file')+1])
    provenance['fixture'] = {'token_count':len(token_path.read_text().split()), 'tokens_sha256':sha(token_path), 'prompt_sha256':sha(prompt_path)}
    prior_fixture = json.loads((old/'provenance.json').read_text())['token_checks']['10k']
    assert provenance['fixture']['token_count'] == prior_fixture['count'] == 10520
    assert sha(token_path) == prior_fixture['ids_sha256'] and sha(prompt_path) == prior_fixture['prompt_sha256']
    provenance['fixture']['prior_tokenizer_equality_check'] = str(old/'provenance.json')
    (out/'ours-timeline.patch').write_bytes(subprocess.check_output(['git','diff','HEAD','--','native/bridge.mm','src/Bootstrap.Engine/WallProfile.cs'],cwd=ROOT))
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (out/'splash-timeline.patch').write_bytes(subprocess.check_output(['git','diff'],cwd='/tmp/splash-timeline-1482'))
    for name, cmd in plan:
        case = out/name
        case.mkdir(exist_ok=False)
        caseenv = env.copy()
        if name.startswith('ours'):
            caseenv.update(BOOTSTRAP_TIMELINE_FILE=str(case/'timeline.jsonl'), BOOTSTRAP_PROFILE_SHAPES='1')
        (case/'identity.json').write_text(json.dumps({'command':cmd, 'started':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'environment':{k:v for k,v in caseenv.items() if k.startswith(('BOOTSTRAP_','SPLASH_'))}},indent=2)+'\n')
        print(name, 'start',flush=True)
        with (case/'stdout.jsonl').open('w') as stdout, (case/'stderr.txt').open('w') as stderr, (case/'memory.jsonl').open('w') as mem:
            p = subprocess.Popen(cmd, cwd=ROOT, env=caseenv, stdout=stdout, stderr=stderr)
            while p.poll() is None:
                row = {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
                for key, command in [('vm',['vm_stat']),('swap',['sysctl','vm.swapusage']),('thermal',['pmset','-g','therm'])]:
                    row[key] = subprocess.check_output(command,text=True)
                mem.write(json.dumps(row)+'\n'); mem.flush()
                time.sleep(3)
            (case/'exit.txt').write_text(str(p.returncode)+'\n')
            if p.returncode:
                raise RuntimeError(f'{name} exit {p.returncode}; see {case}')
        print(name, 'complete',flush=True)
    (out/'done.txt').write_text('complete\n')

if __name__ == '__main__':
    main()
