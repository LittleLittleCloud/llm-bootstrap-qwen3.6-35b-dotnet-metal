# Fused online attention on M5 Pro

The `--tuned-prefill` preset now selects `fp16-online`: QK, online softmax and
AV execute in one threadgroup loop. At 8192 history + 2048 new queries,
synchronized attention wall time drops from 37.12 to 16.45 ms; at
30000 + 2048 it drops from 169.30 to 54.13 ms. A paired full-model check
improves 10,520-token prefill from 1548.73 to 1678.54 tok/s (+8.4%).
These are development measurements, not a new formal Bootstrap evaluation.

## Selected implementation

- Stage FP32 queries into grouped FP16 rows, then run a fused TensorOps kernel.
  QK and AV accumulate in FP32. Running maxima, denominators and output
  accumulators stay FP32; block-local exponentials are rounded to FP16 for AV.
  This changes rounding relative to the previous normalized-FP16 probability
  path. Gate application and final output remain FP32.
- Use 32 logical query rows by 256 KV rows, eight SIMD groups (256 threads),
  and 32 KiB of shared memory. A logical row combines token and query head:
  with G=8 this tile contains four tokens across eight query heads, **not**
  32 tokens per head. Each group shares one of the two KV heads.
- Reuse FP32 score scratch for half probabilities after each row reduction.
  Its unused second half holds the output-rescaling factor and denominator.
  No full device-memory score/probability matrix is produced by this kernel.
- Dynamic tensor slices protect query/KV tails. Static slices are used only
  when a full tile fits the allocated tensor extent. The causal loop ends
  at the last real query in the tile; each row masks future keys separately.
- Use the existing causal device path below 32 new queries. At 32768+9,
  the exploratory fused path took about 1.98 ms versus 1.14 ms for that path.
  Small speculative-verification batches also retain their existing behavior.

With resident FP16 KV the long path needs two dispatches (query staging and
fused attention), versus 97 at 30000+2048 in the previous implementation.
FP32 KV remains supported through two additional FP16-conversion dispatches.
The general batch workspace still reserves its existing 128 MiB score buffer
for fallback/verification. This change avoids allocating a long-path half
probability buffer; it does not remove every attention workspace allocation.

## Tuning and GQA control

The sweep included 16/32/64/128-row query tiles, 32/64/128/256/512-row KV tiles,
four/eight SIMD groups, and G=1/2/8 grouping (selected combinations, not a full
Cartesian product). A separate 128+128 head-dimension split measured about
40–42 ms at 8192+2048, so it was not retained. The direct D=256 TensorOps path
with larger KV blocks and reused scratch performed better.

At 30000+2048, the final sweep measured G=1/G=2/G=8 at approximately
54.15/54.58/54.00 ms. GQA sharing is implemented, but its independent speedup
is within run-to-run variation; the large improvement comes from the fused
loop, tile selection and avoiding device intermediates. The G=1 control
remains available in the microbenchmark. Rejected kernels are retained only
in local tuning artifacts, not the production shader.

## Attention measurements

Same M5 Pro, 48 GiB, macOS 26.6.2 as the
[earlier MLX comparison](MLX_KERNEL_COMPARISON.md). Seven measured calls per
configuration after 250 ms GPU warmup; medians below include query conversion,
output gating, host submission and synchronized completion. Fixtures have
16 query heads, two KV heads and D=256, with identical FP16 KV input. No other
GPU benchmark ran concurrently. Short-call differences include host noise.

| History + queries | Previous ms | Selected ms | Speedup |
| --- | ---: | ---: | ---: |
| 0 + 512 | 0.485 | 0.444 | 1.09x |
| 0 + 2048 | 3.875 | 2.518 | 1.54x |
| 8192 + 2048 | 37.122 | 16.454 | 2.26x |
| 10520 + 190 | 4.245 | 2.226 | 1.91x |
| 32768 + 63 | 5.060 | 2.180 | 2.32x |
| 30000 + 2048 | 169.300 | 54.129 | 3.13x |

At 30000+2048 GPU time is 53.926 ms of 54.129 ms wall. With FP32 KV,
including conversions, wall time is 54.715 ms versus 171.788 ms previously.
Historical MLX medians were 15.184 ms at 8192+2048 and 55.392 ms at
30000+2048; the selected path is now close to those numbers. MLX was not
remeasured in this sweep, and these numbers do not establish a full-model
SOTA comparison or identical numerical behavior.

## Full-model measurement

Target checkpoint: official Qwen3.6-35B-A3B-FP8 revision
`95a723d08a9490559dae23d0cff1d9466213d989`, with the existing DFlash2 draft,
FP16 KV, and unchanged FP8 expert weights. Both variants use gathered expert
inputs, adaptive expert tiles, predecoded dense weights and 2048-token batches.
The only variant difference is attention mode. Three warm uncached runs
alternate order; model loading and tokenization are excluded. Ordinary
checkpoint writes and draft feature capture remain enabled.

| Variant | Median wall | Prefill tok/s |
| --- | ---: | ---: |
| Previous causal device attention | 6.793 s | 1548.73 |
| Online grouped attention | 6.267 s | 1678.54 |

The selected samples span 1678.50–1682.20 tok/s. Both variants return the full
public needle answer `73DY-DRV5-B8FJ` and arithmetic answer `4`. The prompt has
10,520 tokens; SHA256:
`d3367dad7b2bb2004af3ab0e54fe2d2d76944cf58c7ba36d18673b62a950b6ee`.
The arithmetic fixture produces the same token, but its full logits differ:
maximum absolute difference 0.71558 and relative L2 difference 0.06854. These
finite answer checks do not establish unchanged model quality.

Attention acceleration yields an 8.4% full-model improvement for this prompt;
other model work still dominates. No decode-throughput improvement is claimed.

## Correctness and memory regression

The dedicated benchmark checks all outputs for finiteness, compares selected
coordinates against independent FP64 attention with FP16 Q/K/V, verifies
in-place output equality, and protects KV tails/output bounds with NaN guards.
Small fixtures check every query row and all 16 heads; large fixtures sample
first/middle/last queries and heads spanning both KV groups. The standard
selected-path sweep has maximum sampled absolute error 6.13e-5, including
small-batch fallback; fused cases are at most 2.62e-5. A multi-block sharp-logit
fixture (Q and K scaled by 16) has error 1.18e-5. The independent softmax
reference remains unrounded, with a 5e-4 absolute-error bound.

The existing rounded CPU oracle also passes nine attention paths × nine
shapes with FP16 KV and five paths × nine shapes with FP32 KV; maximum
sampled error across these checks is 7.70e-7. Its online reference reproduces
block-local half exponentials, while the separate benchmark above uses an
unrounded FP64 softmax. The 31/32/33-query selection boundaries and 32768+9
fallback pass, as do 253,280 causal-softmax checks and 1,659 kernel checks.
Release build has zero warnings/errors. At 32768+9 the selected fallback
measures 1.137 ms versus 1.142 ms for the same baseline path.

The selected full-model preset passes 45 verifier checks and 48 prefix-cache
checks. Both 30,044-token retrievals (needle depths 5% and 90%) return the
complete answer, with two exact cached repeats each and a short cached request
after growth. TTFT is 19.128/18.768 s. There is no OOM at 32K; target KV is
671,088,640 bytes, total tracked Metal buffers 40,076,771,328 bytes, and CPU
prefix snapshots 334,536,704 bytes. System swap was active; these allocation
counts are not a residency or peak process-memory measurement.

## Reproduction

Run from the repository root, with .NET 10 and the external model snapshots:

```sh
dotnet build src/Bootstrap.Engine -c Release
ENGINE=src/Bootstrap.Engine/bin/Release/net10.0/Bootstrap.Engine.dll
# Dedicated correctness + performance cases; no model weights required.
dotnet "$ENGINE" profile-online-attention --case 8192,2048 --runs 7
dotnet "$ENGINE" profile-online-attention --case 30000,2048 --runs 7
# All shapes, threshold boundaries, production fallback and GQA control.
dotnet "$ENGINE" profile-online-attention --runs 3
# Multi-block sharp logits and the FP32 KV compatibility path.
dotnet "$ENGINE" profile-online-attention --case 1024,63 --stress --runs 3
dotnet "$ENGINE" profile-online-attention --case 30000,2048 \
  --kv-cache-precision fp32 --runs 3
# Paired full-model comparison; set TARGET, DRAFT and PROMPT to local files.
dotnet "$ENGINE" compare-prefill-variants --model-dir "$TARGET" \
  --draft-dir "$DRAFT" --prompt-file "$PROMPT" --expected-answer 73DY-DRV5-B8FJ \
  --context-window 32768 --kv-cache-precision fp16 --online-variants --runs 3
dotnet "$ENGINE" kv-cache-self-test --model-dir "$TARGET" \
  --draft-dir "$DRAFT" --tuned-prefill
```

`--variants baseline,selected` compares the previous and selected production
paths; `q32_k256_s8_g1_r1,q32_k256_s8_g8_r1` isolates GQA grouping with FP16 KV.
JSON output includes synchronized wall/GPU time and correctness checks.
Local ignored evidence is under `artifacts/attention-online/`: `selected-all`,
`selected-stress`, `selected-fp32-kv`, `model-paired`, `model-kv-cache`, and
exploratory sweeps including failures. Frozen experiment records and generated
leaderboard indexes are unchanged.
