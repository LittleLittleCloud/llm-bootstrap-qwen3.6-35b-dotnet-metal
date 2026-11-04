#!/usr/bin/env python3
"""Current production affine kernels vs pinned Splash, without unrelated variants."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('affine_profile', ROOT/'scripts/profile-affine-splash.py')
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)

def validate_current(out, splash):
    bench = out/'current-validation.mm'
    assert not bench.exists(), 'Do not overwrite validation evidence'
    text = profile.once((out/'bench.mm').read_text(),
        'std::vector<Variant> variants={{"affine",oursBase},{"compact_stage",oursCompact}};',
        'std::vector<Variant> variants={{"affine",oursBase}};')
    bench.write_text(text)
    build = ['xcrun', 'clang++', '-std=c++20', '-O3', '-fobjc-arc', '-mmacosx-version-min=26.4',
             '-framework', 'Foundation', '-framework', 'Metal', '-I'+str(splash/'runtime'), str(bench), '-o', str(out/'current-validation')]
    subprocess.run(build, check=True)
    cmd = [str(out/'current-validation'), str(out/'profile.metal'), str(out/'moe.metal'), str(out/'dense.metal'), '0', 'all', 'forward']
    env = os.environ.copy(); env['MTL_SHADER_VALIDATION'] = '1'
    with (out/'validation-current.jsonl').open('w') as stdout, (out/'validation-current.stderr').open('w') as stderr:
        result = subprocess.run(cmd, env=env, stdout=stdout, stderr=stderr)
    (out/'validation-current-identity.json').write_text(json.dumps({'command': cmd, 'build': build,
        'environment': {'MTL_SHADER_VALIDATION': '1'}, 'exit_code': result.returncode,
        'source_sha256': hashlib.sha256(bench.read_bytes()).hexdigest(),
        'binary_sha256': hashlib.sha256((out/'current-validation').read_bytes()).hexdigest()}, indent=2)+'\n')
    result.check_returncode()
    print('production kernels validation complete', flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splash', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--validate-current-only', action='store_true')
    args = ap.parse_args()
    splash, out = args.splash.resolve(), args.out.resolve()
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=splash, text=True).strip() == profile.PIN
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=splash, text=True).strip()
    if args.validate_current_only:
        validate_current(out, splash)
        return
    out.mkdir(parents=True, exist_ok=False)
    hashes = profile.prepare(out, splash)
    shader = (ROOT/'native/kernels.metal').read_text()
    extra = shader[shader.index('template<int Rows, class Acc>\ninline void expert_affine_accumulate'):shader.index('template [[host_name("ablation_gate_up_predecoded")]]')]
    for name in ['expert_affine_accumulate', 'expert_ablation_accumulate', 'expert_ablation_run', 'expert_ablation(']:
        extra = extra.replace(name, 'compact_'+name)
    extra = profile.once(extra, 'threadgroup half stage[128*KS];', 'threadgroup half stage[(Layout==4?64:128)*KS];')
    assert extra.count('lane,stage+sid*64*KS,acc);') == 2
    extra = extra.replace('lane,stage+sid*64*KS,acc);', 'lane,stage+sid*(Layout==4?32:64)*KS,acc);')
    # FP32 gate scratch spans both compact SIMD stages: both matmuls must finish
    # before either stage is reused. Retain FP32 gate, live extents and epilogue.
    extra = profile.once(extra, 'if constexpr(Fused){\n        using Gate=', 'if constexpr(Fused){\n        if constexpr(Layout==4)threadgroup_barrier(mem_flags::mem_threadgroup);\n        using Gate=')
    for family, fused in [('gate_up', 'true'), ('matrix', 'false')]:
        template = f'compact_expert_ablation<false,{fused},32,4,0>'
        extra += f'\ntemplate [[host_name("compact_{family}")]] kernel decltype({template}) {template};\n'
    with (out/'profile.metal').open('a') as f:
        f.write('\n'+extra)
    bench = out/'bench.mm'
    text = bench.read_text()
    old = 'std::vector<Variant> variants={{"affine",oursBase},{"fixed_rows",oursFixed},{"fast_math",oursFast},{"fixed_fast",oursFixedFast}};'
    assert text.count(old) == 1
    compact = '''auto oursCompact=[&](id<MTLComputeCommandEncoder> enc){g.dispatch(enc,"ours",gate?"compact_gate_up":"compact_matrix",{ox,w.affine,w.affineDesc,ot,tc,oy},&op,sizeof(op),15,n/(gate?32:64),std::min(rows,(rows+31)/32+256),64);};
   std::vector<uint8_t> baseline;
   std::vector<Variant> variants={{"affine",oursBase},{"compact_stage",oursCompact}};'''
    text = text.replace(old, compact)
    text = profile.once(text, 'g.run(ours);g.run(splash);\n   double worstO=', '''g.run(ours);g.run(splash);
   size_t used=size_t(rows)*n*(gate?2:4);
   if(baseline.empty())baseline.assign((uint8_t*)oy.contents,(uint8_t*)oy.contents+used);
   else if(memcmp(baseline.data(),oy.contents,used))throw std::runtime_error("compact stage full-output bitwise mismatch");
   double worstO=''')
    bench.write_text(text)
    command = ['xcrun', 'clang++', '-std=c++20', '-O3', '-fobjc-arc', '-mmacosx-version-min=26.4',
               '-framework', 'Foundation', '-framework', 'Metal', '-I'+str(splash/'runtime'), str(bench), '-o', str(out/'bench')]
    subprocess.run(command, check=True)
    identity = {'splash_commit': profile.PIN, 'splash_source_hashes': hashes, 'build': command,
                'engine_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                'hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'native/kernels.metal', bench, out/'profile.metal', out/'bench']},
                'scope': 'Identical decoded dyadic weights and activations; original implementation-specific layouts, rounding and outputs retained.'}
    (out/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    for label, runs, order in [('forward', 3, 'forward'), ('reverse', 3, 'reverse'), ('validation', 0, 'forward')]:
        cmd = [str(out/'bench'), str(out/'profile.metal'), str(out/'moe.metal'), str(out/'dense.metal'), str(runs), 'all', order]
        with (out/f'{label}.jsonl').open('w') as stdout, (out/f'{label}.stderr').open('w') as stderr:
            env = os.environ.copy()
            env.pop('MTL_SHADER_VALIDATION', None)
            env.pop('MTL_DEBUG_LAYER', None)
            if label == 'validation':
                env['MTL_SHADER_VALIDATION'] = '1'
            result = subprocess.run(cmd, env=env, stdout=stdout, stderr=stderr)
        (out/f'{label}-exit.txt').write_text(str(result.returncode))
        if result.returncode:
            if label != 'validation':
                result.check_returncode()
            print('compact diagnostic validation failed; retaining evidence and validating production kernels separately', flush=True)
            validate_current(out, splash)
            continue
        print(label, 'complete', flush=True)

if __name__ == '__main__':
    main()
