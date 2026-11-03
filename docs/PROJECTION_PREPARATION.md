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
