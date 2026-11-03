# Parallel FP32 DeltaNet preparation — 2026-11-03

The implementation starts from merged main
`dad9fd0d4474b0809e2a05c490fd400a02dcfcde` (PRs #13–#15).
Its hypothesis is that token-parallel convolution and fused preparation reduce
prefill latency without changing weights, arithmetic precision, recurrent scan,
attention, draft capture, or prefix-cache policy.

## Implementation

The original convolution assigns a channel to each thread and loops over tokens.
The new convolution reads the current and preceding three input rows directly,
using incoming convolution carry only for the first three tokens. Tokens can
therefore execute independently. The `parallel` control still launches the
original Q/K normalization and gate preparation separately.

The `fused` path assigns a 128-thread group to each token/key-head pair and
combines convolution, SiLU, Q/K normalization and the two corresponding
value-head gates. It writes Q/K, gates and V directly in FP32. It preserves the
original reduction and expression order. The Q/K portion of the convolution
scratch is not written or consumed in this path.

A separate carry dispatch updates the final three input rows only after all
preparation readers have completed. Updating carry from the last token inside
the preparation dispatch would race the first three tokens. The carry update
also supports one- and two-token blocks in development fixtures.

`--tuned-prefill` selects `--delta-preparation fused`. It applies only when the
DeltaNet mode is optimized (for example `--delta-mode staged8`), the prefill
block has more than eight rows, and execution is not speculative verification.
Baseline recurrence, tiny blocks, decode and verification retain the original
convolution/tape path. Changing preparation mode clears prefix cache.
`--delta-preparation baseline` and `parallel` remain available as controls.

## Reproduction

```sh
dotnet build src/Bootstrap.Engine/Bootstrap.Engine.csproj -c Release
ENGINE=src/Bootstrap.Engine/bin/Release/net10.0/Bootstrap.Engine.dll
dotnet "$ENGINE" profile-delta-prepare --runs 5
MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 dotnet "$ENGINE" profile-delta-prepare --runs 1

# Paired model checks with identical prompts, weights and settings:
dotnet "$ENGINE" moe-self-tests \
  --model-dir "$TARGET" --draft-dir "$DRAFT" --affine-cache "$AFFINE_DIR" \
  --tuned-prefill --delta-mode staged8 --delta-preparation fused \
  --context-window 32768 --packed-model-check \
  --prompt-file "$PROMPT10K" --long-prompt-file "$PROMPT30K" \
  --expected-answer "$EXPECTED" --runs 3 --output-dir "$OUTPUT"
```

All GPU work is serialized. Shader-validation timings are not performance data.
The microbenchmark alternates variant order, warms each configuration and
includes the carry update in GPU time. It retains run zero but excludes it from
reported medians. Whole-model observations clear prefix reuse and exclude model
loading/tokenization. The cold complete-answer run is separate from three warm
single-token observations; decode, verifier and prefix-cache checks follow.

## Preparation microbenchmark

Apple M5 Pro, 20 GPU cores, 48 GiB; safe math, FP32 intermediate values/state,
BF16 convolution weights. Five GPU-time observations per configuration after
warmup, reverse order on alternating iterations. These measure the complete
preparation sequence, not the scan or the full model.

| Tokens | Original ms | Parallel convolution ms | Fused preparation ms |
| ---: | ---: | ---: | ---: |
| 17 | 0.01050 | 0.00979 | 0.00687 |
| 128 | 0.04062 | 0.02825 | 0.02121 |
| 2048 | 1.01396 | 0.79429 | 0.55817 |

At 2048 rows, parallel-only lowers preparation GPU time by 21.7%; fusion lowers
it by 45.0%. Small blocks do not justify an extra carry dispatch on their own;
production blocks of at most eight rows retain the original path.

The fixture checks 39 variant/shape/split combinations over 1, 2, 3, 7, 17,
128 and 2048 tokens, including nonzero incoming carry and recurrent state, zero
Q/K heads, signed weights, guard values and an independent double convolution/
normalization reference. Both new paths match the baseline Q/K/V, gates,
convolution carry, recurrent output and final recurrent state bitwise, including
split blocks. This is evidence for the tested fixtures, not a proof for all inputs.

## Whole-model development comparison

All four processes use the same affine-8 group-64 v2 cache, BF16 scale/bias and
draft weights, FP16 KV/operands, FP32 accumulation/state/output, staged8 scan,
2048-token chunks, prefix checkpoints, DFlash2 and explicit 32768-token context.
The first and last processes use the unmodified merged-main engine/shader.
The parallel and fused candidates run in between. These are runtime diagnostics,
not endpoint scores or a randomized crossover.

| Input tokens | Main before ms | Parallel ms | Fused ms | Main after ms | Fused throughput gain vs later main |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10520 | 4155.49 | 4105.21 | 4068.64 | 4145.94 | 1.90% |
| 29939 | 13614.94 | 13441.94 | 13424.04 | 13613.06 | 1.41% |

All figures are medians of three warm uncached observations; complete-answer
cold runs are retained separately. Fused ranges: 10520 tokens 4063.04–4086.00 ms; 29939 tokens 13351.16–13567.86 ms; 
the later-main ranges are 4142.79–4169.30 and 13570.26–13687.13 ms.
The extra benefit of fusion over parallel-only is small at 30K and is not a
precise isolated speedup estimate. The preparation microbenchmark explains only
part of whole-model latency; attention and matrix work are unchanged.

Each process passed the complete 10K/30K answer checks, first-token consistency,
45 verifier checks and 48 prefix-cache checks (including exact-state hashes,
branching, shortening and KV growth). Smoke logits are bitwise identical across
all four processes, SHA-256
`5f85879f85492a1f7aa729069bd49b5df7ca8646cd4ddbfda2fdda56d4c720e2`.
A separate run with Metal API and GPU validation enabled passed all 39 preparation
cases. No precision relaxation was needed.

Whole-machine lifecycle monitors, including loading: initial main incurred
49.9375 MiB of new swap-outs; parallel, fused and later main incurred zero.
All four had zero thermal-warning samples. These counters are not model-attributed.
Peak allocated model buffers in the measured long fixture are unchanged at
40,327,692,288 bytes. No new scratch buffer is introduced by the preparation path.

## Fresh full-model GPU profile

A separate counter pass on each engine measures per-encoder GPU intervals. It
keeps the original command-buffer boundaries; interval sums may overlap and
must not be treated as exclusive wall-time components. Performance claims above
use uninstrumented observations, not these counter passes.

| Input tokens | Main preparation ms | Fused preparation including carry/tiny fallback ms | Main dispatches | Fused dispatches |
| ---: | ---: | ---: | ---: | ---: |
| 10520 | 154.00 | 76.32 | 9906 | 9726 |
| 29939 | 443.31 | 216.76 | 23064 | 22614 |

Preparation intervals fall by 50.4% / 51.1% at 10K / 30K. At 30K, the new
fused kernel accounts for 215.50 ms and carry updates for 1.02 ms; tiny-block
fallback adds 0.24 ms. This supports the measured bottleneck hypothesis while
also showing why total prefill throughput improves by only about 1–2%.
The recurrent scan is unchanged; its interval is 984.55 versus 961.86 ms at
30K, and that difference is not attributed to a scan implementation change.

Local-only evidence is under `artifacts/gdn-prepare/`: all timings, both main
controls, the parallel-only candidate, shader validation, counter profiles,
command/artifact identities, memory/thermal logs and summaries. Model prompts,
answers, binaries and logs are not committed. The official Full-profile browser
evaluation is frozen and submitted separately; this document does not substitute
for it. Further precision changes, INT8 KV and draft-window optimizations are
outside this experiment.
