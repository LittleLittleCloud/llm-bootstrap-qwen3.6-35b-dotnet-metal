# Prefill research after fused MoE

These are development measurements on the same Apple M5 Pro (20 GPU cores,
48 GiB), official Qwen3.6-35B-A3B FP8 checkpoint and DFlash2 draft used by the
previous experiment. They are **not a new endpoint evaluation or leaderboard
submission**. Serving defaults remain unchanged; the new recurrence is opt-in
with `--tuned-prefill --delta-mode staged8`.

## Updated profile

The inference baseline is `59209d213c645130c6555a8a299f7db4ebd8c8dd`.
Counter-accounting commit `c0911d7729fa4c4650174e70fe9f0ccf3729b963` changes
instrumentation only. It validates the span and calibration bounds of timestamp
samples, allowing overlapping encoder intervals instead of treating their sum
as command-buffer elapsed time.

Same fixed public fixtures as the previous profile, prefix cache cleared before
each request, 2048-token blocks, FP16 KV/operands and FP32 accumulation/state.
Loading and tokenization excluded. One cold run, three warm uncached requests,
then one sampled request per size. GPU workloads run serially. No OS or power
settings changed.

| Normal median | 10,520 tokens | 29,939 tokens |
| --- | ---: | ---: |
| Wall time | 5844.5 ms | 18679.5 ms |
| GPU time | 5813.2 ms | 18639.9 ms |
| Process CPU time | 39.2 ms | 46.4 ms |
| Throughput | 1800.0 tok/s | 1602.8 tok/s |
| Counter wall overhead | 1.3% | 4.4% |

| Sampled group | 10K | 30K |
| --- | ---: | ---: |
| Expert matmul including FP8 unpack | 39.5% | 34.1% |
| Dense matmul, weight/input conversion | 24.0% | 21.5% |
| DeltaNet recurrence, convolution/L2 | 16.7% | 14.9% |
| Attention core, layout/RoPE | 10.1% | 21.1% |
| Expert routing/grouping | 0.1% | 0.1% |

Shares normalize **overlapping sampled intervals**, not exclusive wall time.
They rank optimization targets but cannot predict exact end-to-end speedups.
Routing is no longer an important target. Expert matmul remains the largest
category, and attention remains important specifically at long context.

## FP8 weight-layout experiment

Two layouts reorder the original bytes into 32-column by 32-input tiles:
contiguous per-lane payload and interleaved four-byte lane loads. BF16 scales,
FP8 values, FP16 operands and FP32 accumulation stay unchanged. This does not
re-quantize to affine integers. Scale metadata is not repacked in this experiment.
Repacking and checks happen outside measured kernel time; these measurements do
not include a production loading/cache implementation.

Every repacked byte is checked against its source. Complete output arrays match
exactly across projection layouts. Fused gate/up uses distinct weights and
varying scales, independent CPU references, complete output comparisons and
output guards. Uniform and skewed expert populations are covered. The projection
sweep was repeated in reverse variant order after explicit warming.

| 2048-token workload, GPU median | Original layout | Repacked layout 1 |
| --- | ---: | ---: |
| Projection K=2048, N=512, uniform | 3.116 ms | 2.803 ms |
| Projection K=512, N=2048, uniform | 3.062 ms | 2.735 ms |
| Fused gate/up, uniform | 5.256 ms | 4.679 ms |
| Fused gate/up, skewed | 5.012 ms | 4.631 ms |

Projection timing includes input gathering. Fused timing includes activation.
The two layouts perform similarly. Software prefetch/double buffering remains
slower even after repacking. These results support a byte-preserving load-time
layout change, but do not establish its end-to-end gain. Avoid retaining a second
full copy of the expert weights on a 48 GiB machine. Production integration must
also teach decode/verification to read the chosen layout and preserve checkpoint
hash verification; merely repacking before every prefill block may erase the gain.

## FP32 DeltaNet experiment

The original recurrence computes decay and beta repeatedly inside every state-row
loop. Variants isolate gate precomputation, eight-lane row reductions, and staging
8 or 16 tokens in threadgroup memory. The selected development variant uses an
8-token block and carries sixteen value rows per threadgroup. All Q/K/V, outputs
and recurrent state remain FP32; only summation order changes. Verification tape
and small-block paths retain the original kernel.

| 2048-token recurrence, including gate preparation | GPU median |
| --- | ---: |
| Original | 5.588 ms |
| Precomputed gates only | 4.870 ms |
| Eight-lane rows, direct input reads | 2.822 ms |
| Eight-token staging | 2.403 ms |
| Sixteen-token staging | 2.972 ms |

Independent double-precision reference checks, complete output/state comparisons,
nonzero initial states, guards and split-chunk continuation cover 1, 7, 17, 128
and 2048 tokens. Five timed repeats use alternating order and explicit warming.

Paired model runs alternate configurations in one process. Each configuration
first generates and checks the full public needle answer; each of three timed
warm runs clears prefix reuse and generates one token. All configurations answer
both the arithmetic smoke and the needle correctly, and logits remain finite.

| Fixed fixture | Original wall | Staged8 wall | Original tok/s | Staged8 tok/s | Throughput gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10,520 tokens | 5810.4 ms | 5330.2 ms | 1810.5 | 1973.7 | 9.0% |
| 29,939 tokens | 18605.4 ms | 17248.8 ms | 1609.2 | 1735.7 | 7.9% |

The whole-model smoke logits are **not bitwise identical**: the staged variants
have about 0.0711 relative L2 difference from the baseline despite agreeing on
the answer. A separate real-model diagnostic compares every recurrence output
and final state against the original kernel, using identical per-layer inputs;
its purpose is to distinguish local kernel error from downstream amplification.
Do not infer broad quality equivalence from the two public fixtures. A complete
official evaluation is still required before promoting the new default.

## Model validation and post-change profile

The separate 10K diagnostic passed all 360 complete output/state checks
(30 recurrent layers across six blocks), with maximum absolute error
1.91e-05 and maximum relative L2 error
3.78e-07. It returned the expected full needle
answer. The composed regression command passed 45 verification checks and 48
prefix-cache checks. The earlier regression run also passed while checking
4800 layer outputs/states, including cache continuation cases.

The per-layer CPU diagnostic observed elevated system compression/swap and was
slow; its time is not performance evidence. The following profiles explicitly
disable that diagnostic before their cold/warm runs. The independently paired
model measurements above remain the speedup evidence.

The post-regression profile warm medians are 5466.2 ms / 1924.6 tok/s at 10K
and 17771.2 ms / 1684.7 tok/s at 30K, about 2.5–3% slower than the paired
candidate runs. Counter wall overhead is 7.3% / 12.5%. These separate-run
observations are retained rather than replacing the paired results with the
fastest available measurement. System-wide memory snapshots during CPU-heavy
diagnostics do not attribute compression/swap to the inference kernel.

| Post-change sampled group | 10K | 30K |
| --- | ---: | ---: |
| expert_matmul_including_fp8_decode | 43.8% | 36.6% |
| dense_matmul | 20.9% | 17.6% |
| dense_weight_decode | 2.3% | 2.4% |
| dense_input_conversion | 2.8% | 2.4% |
| deltanet_recurrence | 6.9% | 6.1% |
| deltanet_conv_l2 | 2.7% | 2.4% |
| attention_core_layout_rope | 10.5% | 23.5% |
| expert_routing_grouping | 0.1% | 0.1% |

The diagnostic runner initially had a C# local-name collision. That failed build
was retained; the existing binary completed the original combined state/cache
checks with layer diagnostics enabled. The runner was corrected and rebuilt,
then its full-answer/check/profile sequence was rerun successfully. No inference
kernel changed in the correction. The initial run did not produce performance
profiles and was not used for speed comparisons.

## Dense tile sweep

The direct FP16-input, FP32-output TensorOps matmul was swept over 32x64, 64x64,
128x64, 64x128 and 128x128 tiles and 1/2/4 SIMD groups where supported. Five
geometries include a 127-row tail and the actual large/shared projection shapes.
All 35 full-output checks pass, with independent CPU reference points and guards.
Five repeats alternate variant order after warming.

At M=2048,N=8192,K=2048, the existing kernel takes 2.119 ms and the best smaller
variants remain about 2.10 ms. The 128x128 tile takes 2.680 ms. Other shapes show
only small improvements or regressions. Larger tiles alone do not explain or
close the remaining gap. Prioritize avoiding conversions/materialization and
studying the weight format path over another blind tile sweep.

## Comparison with Splash and next experiments

The historical reference is Splash commit
`1482f6bbc2abf12375332725302019594150b6cb`, with MLX affine 8-bit group-64 weights.
Its 2897 tok/s endpoint report uses a different target checkpoint/quantization
and an earlier evaluation pin. It is a best-permitted-system reference, **not a
matched speedup baseline for these local fixed-fixture timings**. Splash is a
custom Metal runtime; MLX-format checkpoint loading does not imply MLX runtime
execution. No new Splash whole-model run was performed here.

The reference report also has a correctness caveat: the [full report producing
2897.3 tok/s](https://github.com/LittleLittleCloud/LLM-Bootstrap/blob/main/experiments/2026-11-01-splash-qwen36-8bit-001/evaluation/20261101-194357.json)
records an empty answer, `correct: false` and `compliant: false` for that 10K
prefill request (10,434 actual prompt tokens). A [later conform/needle report](https://github.com/LittleLittleCloud/LLM-Bootstrap/blob/main/experiments/2026-11-01-splash-qwen36-8bit-001/evaluation/20261101-214244.json)
passes needle 10/10 at about 70K context, but does not remeasure prefill. This does
not establish that Splash's current engine is incorrect or that its speed number
is invalid; it means equal-quality superiority needs a fresh controlled run.
Our previous official prefill report used 10,440 actual prompt tokens and returned
the correct visible answer. The old/new prefill plan sizes match, but the evaluator
revision, parsing/generation behavior, quantization and resource ceilings differ.

Source references at that fixed revision:

- [Staged quantized matrix loop](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/metal/kernels/common/gguf_staged_tile.h)
- [DeltaNet prefill](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/metal/kernels/prefill/gdn.metal)
- [Dense execution geometry](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/metal/abi/Gguf.h)

The next useful steps are:

1. Evaluate the opt-in staged recurrence with the full pinned endpoint evaluation
   before making it the serving default. Retain original verification semantics.
2. Integrate byte-preserving expert repacking with bounded resident memory, then
   measure loading cost, decode/verification and full prefill together.
3. Test FP16/BF16 intermediate storage and fused preparation, separately from
   accumulation precision. Splash stages narrower Q/K/V; our current experiment
   deliberately retains FP32, so its faster 8-token block does not establish that
   Splash's 16-token choice is wrong.
4. Target the long-context attention path with the existing 8192/2048 and
   30000/2048 tests, while checking actual end-to-end benefit. It is now a larger
   share after DeltaNet gets faster.
5. Re-run the reference and candidate under the same released evaluation and
   resource settings. Report quantization differences explicitly. Reaching or
   exceeding Splash has not yet been demonstrated.

## Reproduction and local evidence

Build with `dotnet build src/Bootstrap.Engine -c Release`. Kernel-only commands:

```sh
dotnet src/Bootstrap.Engine/bin/Release/net10.0/Bootstrap.Engine.dll profile-experts --layouts --runs 7
dotnet src/Bootstrap.Engine/bin/Release/net10.0/Bootstrap.Engine.dll profile-experts --layouts --reverse --runs 7
dotnet src/Bootstrap.Engine/bin/Release/net10.0/Bootstrap.Engine.dll profile-expert-fusion --layouts --runs 7
dotnet src/Bootstrap.Engine/bin/Release/net10.0/Bootstrap.Engine.dll profile-delta --runs 5
dotnet src/Bootstrap.Engine/bin/Release/net10.0/Bootstrap.Engine.dll profile-dense-tiles --runs 5
```

Model comparisons use `compare-prefill-variants --tuned-prefill --context-window
32768 --delta-variants` for all variants or `--delta-final-variants` for the
baseline/staged8 pair, with `--model-dir`, `--draft-dir`, `--prompt-file`,
`--expected-answer` and `--runs 3`. For real-layer checks plus state/cache
regressions and optional post-change profiles:

```sh
dotnet src/Bootstrap.Engine/bin/Release/net10.0/Bootstrap.Engine.dll moe-self-tests \
  --model-dir "$TARGET" --draft-dir "$DRAFT" --tuned-prefill --context-window 32768 \
  --delta-mode staged8 --validate-delta --prompt-file "$PROMPT10K" \
  --expected-answer "$NEEDLE" --profile-after-checks --long-prompt-file "$PROMPT30K" --runs 3
```

Raw evidence is local-only under `artifacts/prefill-profile-fused/`,
`artifacts/prefill-layout-research/`, `artifacts/prefill-delta-research/`, and
`artifacts/prefill-dense-research/`. `scripts/summarize-prefill-research.py` reads
these development JSONL files only. It never modifies official evaluator reports
or the leaderboard index. Failed/negative attempts are retained alongside
successful runs.
