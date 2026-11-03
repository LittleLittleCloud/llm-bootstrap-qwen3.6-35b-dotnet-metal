# Fused prefill projection preparation

This experiment starts from main `3851e4fcf7e883da0d040223e82019b82e0746df`,
after merging the GDN and dense-prefill improvements. It tests whether removing
intermediate reads and dispatches around dense projections reduces wall time.
The new preset selects `--projection-preparation fused`; `baseline` restores
the original add, RMSNorm and input-conversion dispatches. Changing the mode
invalidates reusable prefix snapshots.

## Implementation

A new `rmsnorm_pack` kernel retains the original FP32 normalization and writes
both the FP32 activation and its existing FP16 projection operands. The caller
explicitly reuses that packed input for the immediately following projection.
It applies to target input-layer norms, post-attention norms and gated DeltaNet
output norms. At post-attention norms it also performs the original residual
addition, preserving the updated hidden state. The existing FP16 scratch is
reused; no extra buffer is allocated.

The 2048-wide norm retains 256 threads and its original reduction order. The
128-wide gated norm uses 128 threads, omitting the original four SIMD groups
whose input sums were all zero. Its active contributions and expression order
are unchanged. FP32 output and state, FP16 operands/KV, affine weights, draft FC,
attention, expert matmuls, chunk sizes and prefix checkpoints are unchanged.
This does not introduce an additional precision conversion. Tiny rows (<=8)
and speculative verification keep the old paths.

## Rejected combine variant

The fresh main profile also found substantial `batch_mix` time. A four-component
vectorized implementation preserved full-output bitwise equality, aliasing and
guards in 16 checks. Seven alternating warm observations at 2048 rows measured
0.6431 ms for the original versus 0.6489 ms for the candidate. It was not enabled
in model execution. The diagnostic kernel and `profile-mix` retain this negative
result rather than presenting vectorization as a gain.

## Validation and reproduction

```sh
dotnet build src/Bootstrap.Engine -c Release
dotnet src/Bootstrap.Engine/bin/Release/net10.0/Bootstrap.Engine.dll \
  profile-projection-preparation --runs 7
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
  dotnet src/Bootstrap.Engine/bin/Release/net10.0/Bootstrap.Engine.dll \
  profile-projection-preparation --runs 0
```

The 24 fixtures cover 1, 8, 9, 17, 127, 272, 1259 and 2048 tokens for ordinary,
residual and gated normalization. Every FP32 output, FP16 bit pattern and updated
residual matches the old dispatch sequence bitwise. Tests include zero rows,
small magnitudes, signed BF16 weights, in-place gated output, guards and sampled
independent double references. The full suite passes Metal API/GPU validation.
Microbenchmarks include the complete old and new preparation sequences and
alternate order; CPU fixture resets precede each dispatch, so only GPU time is
used for the microbenchmark comparison.

Full-model checks use the unchanged fixed 10K/30K prompts and `moe-self-tests
--packed-model-check --profile-after-checks`, with the existing affine-8 group-64
v2 cache, DFlash2, FP16 KV, staged8, 2048-token chunks and a 32768-token context.
All GPU workloads run serially. The original main is profiled before and after
the candidate; all samples and rejected attempts are retained locally under
`artifacts/prefill-next/`. Formal results use the separate pinned Full browser
evaluation. Local profile intervals can overlap and are not additive wall times.

## Measurements

Fresh main encoder intervals (ms) identify the preparation opportunity:

| Kernel | 10,520 tokens | 29,939 tokens |
|---|---:|---:|
| Dense predecoded core | 1034.25 | 2927.73 |
| Expert gate/up | 831.54 | 2272.58 |
| Expert down | 461.33 | 1275.78 |
| Online attention | 433.46 | 3363.90 |
| RMSNorm | 124.27 | 353.42 |
| Projection input packing | 94.02 | 266.10 |
| Residual add | 30.60 | 87.96 |

These are overlapping diagnostic intervals, not exclusive wall components.
The largest matmuls and long-context attention are unchanged in this experiment.

Final preparation microbenchmarks use seven warm observations per variant:

| 2048-token preparation | Original GPU ms | Fused GPU ms |
|---|---:|---:|
| Input norm + packing | 0.20079 | 0.15038 |
| Residual add + norm + packing | 0.35325 | 0.26438 |
| Gated norm + packing | 0.67108 | 0.43008 |

The primary whole-model comparison alternates baseline/fused order inside one
process, excludes one warmup per variant, and retains all five warm observations
per length. Both variants use the same retained allocations. Timings exclude
loading/tokenization and clear prefix reuse before every request.

| Input tokens | Baseline median ms | Fused median ms | Throughput gain | Wall saved ms |
|---|---:|---:|---:|---:|
| 10520 | 3901.47 | 3831.23 | 1.83% | 70.24 |
| 29939 | 12879.86 | 12693.34 | 1.47% | 186.52 |

At 10520 tokens, baseline ranges 3895.02–3902.75 ms and fused ranges 3830.48–3833.83 ms. Dispatches fall from 10338 to 9438; allocated bytes are identical at 39585497088.

At 29939 tokens, baseline ranges 12871.28–12886.82 ms and fused ranges 12673.12–12699.86 ms. Dispatches fall from 24144 to 21894; allocated bytes are identical at 39927054336.

Separate-process diagnostics were noisier and remain part of the evidence:

| Process / stage | 10K median ms | 30K median ms |
|---|---:|---:|
| Initial main profile | 3920.92 | 12894.91 |
| Candidate full-answer checks | 3848.17 | 12788.23 |
| Candidate profile after state/cache checks | 3902.73 | 12807.43 |
| Later main profile | 3920.11 | 12845.18 |

The candidate's later profile retains verification/cache buffers that the plain
main-profile processes did not allocate. Those cross-process numbers are not the
primary speedup estimate. Kernel counter intervals also shift with surrounding
scheduling; do not assign every observed difference to its named kernel.

Full 10K/30K answers, three 256-token decode runs, 45 verifier checks and 48
prefix-cache checks pass. Target smoke logits remain bitwise identical to main,
SHA-256 `5f85879f85492a1f7aa729069bd49b5df7ca8646cd4ddbfda2fdda56d4c720e2`.
This is finite-fixture evidence, not a proof of universal numerical equivalence.

baseline-profile: whole-machine new swap-outs 0.000 MiB.

candidate-checks: whole-machine new swap-outs 0.000 MiB.

baseline-after: whole-machine new swap-outs 0.000 MiB.

Alternating comparison: whole-machine new swap-outs 0.000 MiB. These are system-wide counters. All development processes recorded no thermal warnings. No memory or power settings changed.

To reproduce the paired comparison, use `compare-projection-preparation --runs 5`
with the same model/draft/affine-cache paths, `--tuned-prefill --delta-mode staged8
--context-window 32768`, and both `--prompt-file` / `--long-prompt-file` fixtures.
It switches only projection preparation; this diagnostic is not an evaluator.
The remaining long-context attention and expert matmuls need separate measured
experiments. The inherited formal prefill target is not assumed met by these
local improvements.
