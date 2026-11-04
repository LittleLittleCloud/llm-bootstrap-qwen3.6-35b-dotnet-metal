#!/usr/bin/env python3
"""Serial development profiles. Does not create official evaluation reports."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prior', type=Path, required=True)
    ap.add_argument('--splash-source', type=Path, required=True)
    ap.add_argument('--splash-model', type=Path, required=True)
    ap.add_argument('--metallib', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    prior = args.prior.resolve()
    old = prior / 'artifacts/splash-prefill-profile'
    dll = ROOT / 'src/Bootstrap.Engine/bin/Release/net10.0/Bootstrap.Engine.dll'
    base = json.loads((prior / 'artifacts/prefill-next-profile/command.json').read_text())
    base[1] = str(dll)
    base += ['--dense-prefill', 'all', '--projection-preparation', 'fused', '--dispatch-fusion', '29']
    # Current tokenizer must reproduce every historical ID, not just the length.
    token_checks = {}
    for tag, flag in [('10k', '--prompt-file'), ('30k', '--long-prompt-file')]:
        prompt = Path(base[base.index(flag) + 1])
        tokens = json.loads(subprocess.check_output([
            base[0], str(dll), 'tokenize', '--model-dir', base[base.index('--model-dir')+1],
            '--text', prompt.read_text()], text=True))['chat']
        expected = [int(x) for x in (old / f'tokens-{tag}.txt').read_text().split()]
        assert tokens == expected, f'{tag}: token ID mismatch'
        token_checks[tag] = {'count': len(tokens), 'ids_equal': True,
                             'prompt_sha256': sha(prompt), 'ids_sha256': sha(old / f'tokens-{tag}.txt')}
    provenance = {
        'engine_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'splash_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.splash_source, text=True).strip(),
        'token_checks': token_checks,
        'hashes': {str(p): sha(p) for p in [dll, dll.parent/'kernels.metal', dll.parent/'libbootstrap_metal.dylib', old/'profile-checked-v2', args.metallib]},
        'scope': 'Same input IDs; native model/activation/draft formats differ. Timings include first token. No prefix reuse; allocations warm.',
    }
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    patch = subprocess.check_output(['git', 'diff'], cwd=args.splash_source)
    (out / 'splash-instrumentation.patch').write_bytes(patch)
    plan = [('ours-before', base)]
    for tag, kv in [('10k', 'int8'), ('10k', 'bf16'), ('30k', 'bf16'), ('30k', 'int8')]:
        name = f'splash-{tag}-{kv}'
        cmd = [str(old/'profile-checked-v2'), str(args.metallib), str(args.splash_model),
               '--tokens-file', str(old/f'tokens-{tag}.txt'), '--trace-out', str(out/name/'intervals.jsonl'),
               '--kv-format', kv, '--state-format', 'fp32']
        plan.append((name, cmd))
    plan.append(('ours-after', base))
    (out / 'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    env = os.environ.copy()
    for key in ['BOOTSTRAP_PROFILE', 'SPLASH_COUNTER_TRACE_FILE', 'MTL_DEBUG_LAYER', 'MTL_SHADER_VALIDATION']:
        env.pop(key, None)
    for name, cmd in plan:
        p = out / name
        p.mkdir()
        (p/'identity.json').write_text(json.dumps({'command': cmd, 'started': datetime.datetime.now(datetime.timezone.utc).isoformat()}, indent=2)+'\n')
        print(name, 'start', flush=True)
        with (p/'stdout.jsonl').open('w') as stdout, (p/'stderr.txt').open('w') as stderr, (p/'memory.jsonl').open('w') as memory:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            while proc.poll() is None:
                row = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'pid': proc.pid}
                for key, command in [('vm', ['vm_stat']), ('swap', ['sysctl', 'vm.swapusage']), ('thermal', ['pmset', '-g', 'therm'])]:
                    row[key] = subprocess.check_output(command, text=True)
                memory.write(json.dumps(row)+'\n')
                memory.flush()
                time.sleep(3)
            (p/'exit.txt').write_text(str(proc.returncode))
            if proc.returncode:
                raise RuntimeError(f'{name}: exit {proc.returncode}; see {p}')
        print(name, 'complete', flush=True)
    (out/'done.txt').write_text('complete\n')

if __name__ == '__main__':
    main()
