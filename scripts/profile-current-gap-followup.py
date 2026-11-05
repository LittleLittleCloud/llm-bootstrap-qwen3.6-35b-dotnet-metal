#!/usr/bin/env python3
"""Run shape attribution and a prefix-snapshot control after the primary campaign."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    ap.add_argument('--dll', type=Path, required=True)
    args = ap.parse_args()
    root, dll = args.root.resolve(), args.dll.resolve()
    assert (root/'done.txt').read_text().strip() == 'complete'
    base = json.loads((root/'plan.json').read_text())[0][1]
    base[1] = str(dll)
    base[base.index('--runs')+1] = '3'
    env = os.environ.copy()
    for name in ['BOOTSTRAP_PROFILE', 'SPLASH_COUNTER_TRACE_FILE', 'MTL_DEBUG_LAYER', 'MTL_SHADER_VALIDATION']:
        env.pop(name, None)
    env['BOOTSTRAP_PROFILE_SHAPES'] = '1'
    for name, extra in [('ours-shapes', []), ('ours-no-prefix', ['--no-prefix-cache'])]:
        directory = root/name
        directory.mkdir()
        cmd = base+extra
        (directory/'identity.json').write_text(json.dumps({
            'command': cmd, 'environment': {'BOOTSTRAP_PROFILE_SHAPES': '1'},
            'started': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'engine_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'hashes': {name: hashlib.sha256((dll.parent/name).read_bytes()).hexdigest()
                       for name in ['Bootstrap.Engine.dll', 'kernels.metal', 'libbootstrap_metal.dylib']},
        }, indent=2)+'\n')
        (directory/'instrumentation.patch').write_bytes(subprocess.check_output(['git', 'diff', '--', 'native/bridge.mm'], cwd=ROOT))
        print(name, 'start', flush=True)
        with (directory/'stdout.jsonl').open('w') as stdout, (directory/'stderr.txt').open('w') as stderr, (directory/'memory.jsonl').open('w') as memory:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            while proc.poll() is None:
                row = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
                for key, command in [('vm', ['vm_stat']), ('swap', ['sysctl', 'vm.swapusage']), ('thermal', ['pmset', '-g', 'therm'])]:
                    row[key] = subprocess.check_output(command, text=True)
                memory.write(json.dumps(row)+'\n'); memory.flush()
                time.sleep(3)
            (directory/'exit.txt').write_text(str(proc.returncode))
            if proc.returncode:
                raise RuntimeError(f'{name}: exit {proc.returncode}')
        print(name, 'complete', flush=True)

if __name__ == '__main__':
    main()
