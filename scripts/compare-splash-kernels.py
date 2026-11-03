#!/usr/bin/env python3
"""Build and run isolated, original Splash and Bootstrap Metal kernels.

Development measurements only; never writes official evaluator reports.
Supply an unmodified Splash checkout at the revision below. No third-party
source is vendored: quoted includes are expanded into ignored build artifacts.
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

PIN = "1482f6bbc2abf12375332725302019594150b6cb"
ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splash", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "artifacts/splash-kernel-comparison")
    ap.add_argument("--runs", type=int, default=9)
    ap.add_argument("--family", choices=["all", "experts", "dense"], default="all")
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--label", default="forward")
    args = ap.parse_args()
    assert args.runs > 0
    src = args.splash.resolve()
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=src, text=True).strip() == PIN
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=src, text=True).strip()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, rel in [("moe", "shared/moe_gguf.metal"), ("dense", "shared/gguf_linear.metal")]:
        seen = set()

        def expand(file):
            file = file.resolve()
            if file in seen:
                return ""
            seen.add(file)
            data = file.read_text()
            hashes[str(file.relative_to(src))] = hashlib.sha256(file.read_bytes()).hexdigest()
            data = re.sub(r'^\s*#pragma once\s*$', '', data, flags=re.M)
            return re.sub(r'^\s*#include "([^"]+)"\s*$',
                          lambda m: expand(src / "runtime" / m[1]), data, flags=re.M)

        (out / (name + ".metal")).write_text("#include <metal_stdlib>\nusing namespace metal;\n" + expand(src / "runtime/metal/kernels" / rel))
    build = ["xcrun", "clang++", "-std=c++20", "-O3", "-fobjc-arc", "-mmacosx-version-min=26.4",
             "-framework", "Foundation", "-framework", "Metal", "-I" + str(src / "runtime"),
             str(ROOT / "scripts/splash-kernel-bench.mm"), "-o", str(out / "bench")]
    subprocess.run(build, check=True)
    identity = {"splash_commit": PIN, "bootstrap_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "source_hashes": hashes, "build": build, "runs": args.runs,
                "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
                "shader_validation": os.environ.get("MTL_SHADER_VALIDATION", "0"),
                "os": subprocess.check_output(["sw_vers"], text=True),
                "power": subprocess.check_output(["pmset", "-g", "custom"], text=True),
                "compiler": "Runtime Metal 4.0: Bootstrap safe math, Splash fast math with original reassociate(off) pragmas; compilation excluded",
                "timing": "MTLCommandBuffer GPUStartTime to GPUEndTime; one operation per command; warmed; no concurrent benchmark"}
    for path in [ROOT / "native/kernels.metal", ROOT / "scripts/splash-kernel-bench.mm", out / "bench"]:
        identity.setdefault("harness_hashes", {})[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / (args.label + "-identity.json")).write_text(json.dumps(identity, indent=2) + "\n")
    with (out / (args.label + ".jsonl")).open("w") as stdout, (out / (args.label + ".stderr")).open("w") as stderr:
        subprocess.run([str(out / "bench"), str(ROOT / "native/kernels.metal"), str(out / "moe.metal"),
                        str(out / "dense.metal"), str(args.runs), args.family, "reverse" if args.reverse else "forward"],
                       stdout=stdout, stderr=stderr, check=True)
    (out / (args.label + "-after.json")).write_text(json.dumps({
        "thermal": subprocess.check_output(["pmset", "-g", "therm"], text=True),
        "swap": subprocess.check_output(["sysctl", "vm.swapusage"], text=True)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
