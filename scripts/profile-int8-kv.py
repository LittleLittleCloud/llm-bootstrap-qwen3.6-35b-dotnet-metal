#!/usr/bin/env python3
"""Serial FP16/INT8 native prefill controls and local quality/cache diagnostics."""
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
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('dll', 'model-dir', 'draft-dir', 'affine-cache', 'prompt-file', 'long-prompt-file', 'out'):
        p.add_argument('--' + flag, type=Path, required=True)
    p.add_argument('--dotnet', default='dotnet')
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--phase', choices=['all', 'profile', 'quality', 'cache'], default='all')
    a = p.parse_args()
    if a.runs < 1:
        p.error('--runs must be positive')
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    launcher = [a.dotnet, str(a.dll.resolve())]
    common = ['--model-dir', str(a.model_dir.resolve()), '--draft-dir', str(a.draft_dir.resolve()),
              '--affine-cache', str(a.affine_cache.resolve()), '--tuned-prefill', '--bf16-expert-output',
              '--context-window', '32768', '--delta-mode', 'staged8', '--dispatch-fusion', '29']
    prompts = ['--prompt-file', str(a.prompt_file.resolve()), '--long-prompt-file', str(a.long_prompt_file.resolve())]
    plan = []
    if a.phase in ('all', 'profile'):
        for tag, kv in [('fp16-before', 'fp16'), ('int8', 'int8'), ('fp16-after', 'fp16')]:
            plan.append((tag, launcher + ['profile-prefill'] + common + prompts + ['--kv-cache-precision', kv, '--runs', str(a.runs), '--all-kernels']))
    if a.phase in ('all', 'quality'):
        for kv in ('fp16', 'int8'):
            plan.append(('quality-' + kv, launcher + ['check-kv-quality'] + common + prompts + ['--kv-cache-precision', kv, '--output-dir', str(out / ('quality-' + kv) / 'logits')]))
    if a.phase in ('all', 'cache'):
        plan.append(('cache-int8', launcher + ['kv-cache-self-test'] + common + ['--kv-cache-precision', 'int8']))
    (out / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    files = [a.dll.resolve(), a.dll.resolve().parent / 'kernels.metal', a.dll.resolve().parent / 'libbootstrap_metal.dylib', ROOT / 'native/kernels.metal', a.prompt_file.resolve(), a.long_prompt_file.resolve()]
    hashes = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    if hashes[str(files[1])] != hashes[str(files[3])]:
        raise RuntimeError('Rebuild first: runtime and workspace Metal sources differ')
    (out / 'provenance.json').write_text(json.dumps({'hashes': hashes, 'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), 'scope': 'Native diagnostics; not the repository-owned endpoint evaluation.'}, indent=2) + '\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('BOOTSTRAP_PROFILE', 'BOOTSTRAP_TIMELINE', 'MTL_'))}
    env['BOOTSTRAP_PROFILE_SHAPES'] = '1'
    for name, command in plan:
        directory = out / name
        directory.mkdir()
        print(name, 'start', flush=True)
        with (directory / 'stdout.jsonl').open('w') as stdout, (directory / 'stderr.txt').open('w') as stderr, (directory / 'memory.jsonl').open('w') as memory:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            while process.poll() is None:
                sample = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'pid': process.pid}
                for key, cmd in [('vm', ['vm_stat']), ('swap', ['sysctl', 'vm.swapusage']), ('thermal', ['pmset', '-g', 'therm'])]:
                    sample[key] = subprocess.check_output(cmd, text=True)
                memory.write(json.dumps(sample) + '\n')
                memory.flush()
                time.sleep(3)
            (directory / 'exit.txt').write_text(str(process.returncode) + '\n')
            if process.returncode:
                raise RuntimeError(f'{name} failed: {process.returncode}; see {directory}')
        print(name, 'complete', flush=True)
    (out / 'done.txt').write_text('complete\n')

if __name__ == '__main__':
    main()
