# Prefill after BF16 expert output: KV precision controls

Follow-up: the proposed ours INT8 path is now implemented and tested in [the implementation report](INT8_KV_20261104.md). The measurements below remain the original pre-implementation profile.

INT8 KV is the next high-value **long-context experiment**, especially around 30K tokens. It is not yet implemented or timed in our engine. A fresh Splash-only KV control saves 1.274 s at 30K, versus 0.129 s at 10K. Our remaining 10K gap is distributed across dense/norm preparation, MoE, DeltaNet and draft work; KV alone is unlikely to close it.

## Configuration and method

Measured on November 4, 2026 (America/Los_Angeles), Apple M5 Pro, 20 GPU cores, 48 GiB. GPU workloads ran serially. No engine, power or OS settings were changed.

- Ours: merged commit `41f8e470efc1e4628a467936eddcbdd618be1d3f`, using the exact Release DLL from the full BF16-output evaluation. `--bf16-expert-output`, FP16 KV and operands, affine-8 group-64 v2 target weights, BF16 DFlash2 draft weights, FP32 accumulation/state, dispatch mask 29, staged8 recurrence, 2048-token chunks and 32768 context.
- Splash 1.4.0: source `1482f6bbc2abf12375332725302019594150b6cb`, previously checked host-only profiling harness, unchanged release Metal library. Native MLX affine-8 target, BF16 activations and affine-4 draft projections; all experts resident. This direct harness is different from the endpoint's automatic expert-streaming configuration.
- Same **10,520 / 29,939 input token IDs**, regenerated with the current tokenizer and checked element by element against the Splash fixture. Prompt and token-file hashes are in local provenance.
- Loading/tokenization excluded; first-token production and draft context setup included. Prefix reuse is cleared before each request. Ours still creates prefix snapshots; Splash's direct harness captures its final draft context without equivalent checkpoints.
- Each process: one first-use request, three warm requests, then one separate counter request. Order: ours before; Splash 10K INT8, 10K BF16, 30K BF16, 30K INT8; ours after. This reverses KV-format order between lengths, not an interleaved same-process AB experiment.
- Wall medians use only uninstrumented warm requests. Shapes label the separate sampled kernel intervals. Encoder intervals can overlap and have different fusion boundaries; do not add group differences to predict whole-request speedups.

## Wall results

| Input tokens | Ours before, ms | Ours after, ms | Splash INT8 KV, ms | Splash BF16 KV, ms |
| ---: | ---: | ---: | ---: | ---: |
| 10,520 | 3894.569 | 3916.333 | 3507.925 | 3636.830 |
| 29,939 | 12982.903 | 12973.493 | 12305.965 | 13579.887 |

The after process is the primary ours reference because both of its sampled warm windows have zero new swap-outs. The before process has 164.438 MiB of whole-machine swap-outs during its 10K warm window and zero at 30K; it is retained rather than discarded. All four Splash processes have zero new swap-outs across loading and measurement. No thermal warnings were recorded. Zero new swap-outs does not mean zero existing swap usage or identical memory conditions.

Our primary latency is 408.408 ms / 11.64% above Splash INT8 at 10K, and 667.529 ms / 5.42% above at 30K. Throughput is respectively 10.43% and 5.15% lower. Against Splash BF16 KV, ours is 7.69% slower at 10K and **4.47% faster at 30K**. FP16 and BF16 have the same storage width but different numerical behavior; the engines also differ in other precision/layout/epilogue choices.

These numbers are native diagnostics, not a replacement for the earlier endpoint comparison of 2706.6 versus 2889.9 tok/s. The prompt tokens, timing boundary, prefix handling and Splash expert-residency configuration differ.

## What changing Splash KV actually saves

| Input tokens | BF16 → INT8 saving | Request latency reduction | Throughput increase | Attention/preparation interval reduction |
| ---: | ---: | ---: | ---: | ---: |
| 10,520 | 128.906 ms | 3.54% | 3.67% | 127.297 ms |
| 29,939 | 1273.922 ms | 9.38% | 10.35% | 1296.103 ms |

Other sampled Splash groups differ only slightly. The control therefore supports a KV/attention effect in Splash, not a generic dispatch reduction: both formats use **3543 dispatches at 10K and 8781 at 30K**. Ours still uses 7218 / 16344.

The 30K saving exceeds our measured 667.5 ms native-system deficit, but transferring that saving to ours would be an unsupported prediction. The engines use different attention tile shapes, layouts and precision. An ours INT8 kernel must be measured before claiming a gain or that it catches Splash.

## Remaining sampled work

Primary ours-after intervals versus Splash INT8, milliseconds:

| Group | Ours 10K | Splash 10K | Ours 30K | Splash 30K |
| --- | ---: | ---: | ---: | ---: |
| Attention + preparation | 482.4 | 497.9 | 3665.3 | 3856.8 |
| MoE, including shared/routing/combine | 1699.6 | 1577.4 | 4609.5 | 4356.6 |
| Dense + remaining norm/residual | 1136.8 | 985.8 | 3135.4 | 2798.5 |
| DeltaNet + preparation | 489.8 | 414.9 | 1392.3 | 1180.3 |
| Draft context | 61.0 | 10.3 | 298.1 | 10.7 |

Attention is **12.43% of our sampled interval sum at 10K and 27.95% at 30K**. This is not an exclusive wall-time share. Our FP16 attention interval is already slightly smaller than Splash INT8's: attention being expensive does not mean our existing implementation is inferior. Narrower KV is an optimization hypothesis for our own path, not proof that attention explains the entire cross-engine deficit.

BF16 expert output has already reduced the combine opportunity. Current combine intervals are 95.7 versus Splash's 78.0 ms at 10K, and 274.4 versus 222.2 ms at 30K. These are much smaller differences than the former FP32-output comparison, although other MoE and dense preparation work remains.

Sampling overhead is material: ours-after counter request is 11.33% / 6.09% slower than its 10K / 30K warm median; ours-before is 17.77% / 9.41%. Splash sampled wall time differs by -0.15% to +0.46%. Draft attribution is particularly variable (ours 30K before 193.2 ms, after 298.1 ms); do not treat its isolated interval difference as a reliable standalone gain. Warm GPU time accounts for about 98.0% / 99.3% of ours-after wall time, again pointing away from CPU dispatch encoding as the main opportunity.

## Recommended next experiment

Implement an optional INT8 target-KV path, starting with full-attention prefill and retaining FP32 accumulation, softmax statistics and recurrent state. The storage format should have separate symmetric INT8 K/V with one FP32 scale per token and KV head over its 256 dimensions. Keep the current FP16 path as a control.

Splash's implementation provides a concrete starting point: `runtime/metal/kernels/common/paged_store_row.h` computes the per-token/head maximum and rounds/clamps to [-127,127]. In `paged_attention_tile.h`, INT8 tensors feed TensorOps directly. K scales multiply QK scores before softmax, and V scales multiply probabilities before PV. This avoids expanding the full history back into a 16-bit scratch buffer. Our FP16-query/INT8-operand variant still requires compilation, numerical and performance verification; Splash's BF16-query result is not proof for that exact combination.

At 32K capacity, our ten full-attention layers, two KV heads, dimension 256 and separate K/V require 640 MiB in FP16. INT8 plus separate FP32 K/V scales would require **325 MiB**, saving 315 MiB, excluding draft caches, snapshots, alignment and scratch. The main prefill motivation is repeatedly reading less KV and potentially cheaper mixed-type matmul, not halving total model memory.

Validation should include zero vectors, outliers and rounding boundaries; QK/softmax/PV output errors; causal masks and chunk tails; prefix-cache restore, append/decode and speculative verification compatibility; then same-prompt logits/probability and retrieval/task comparisons. The previous acceptance of bounded numerical differences does not establish the safety of this additional quantization. Benchmark alternating FP16/INT8 runs at 10K and 30K with identical cache behavior before selecting a default.

For 10K-specific improvement, keep dense/norm preparation and remaining MoE/DeltaNet work on the list. Avoid another generic dispatch-count exercise: the KV control changes performance without changing dispatch count.

## Artifacts and reproduction

Local-only evidence is in `artifacts/kv-gap-refresh/`: all commands, input checks/hashes, raw warm/counter results, interval traces, memory samples, captured runner, `summary.json`, `kernels.csv`, `kv-analysis.json` and an evidence manifest. The first-token repeat check passes in all cases, but this profile does not establish full-answer or numerical equivalence across KV formats.

Runtime SHA-256:

- Ours DLL: `4fbfe7799c8f07fa8a35182301974bc700b19a166e0173ba90eb8c55223d2778`.
- Ours native library: `e2f679c286fb76be45c32a62a6ba611869cb317a1a3dcb648281986447f363eb`.
- Ours Metal source: `41b5e30ed76023bf4fa4d66eee4ccd8fbfdbc3cb53f00eefc4b3cc8cb8b28c9e`.
- Splash checked profiling executable: `def70bddd367b94a9ab7aab520f60fc4ddf7d14c12638c0c3c3c2308cab8cbce`.
- Splash release Metal library: `e30124e8bceba1a9de7afe75cf1a10439bcfa5c1b777490c829ba4da0d0cd1cd`.

```sh
python3 scripts/profile-kv-gap.py \
  --prior /path/to/previous/artifacts/current-gap \
  --dll /path/to/clean/merged-runtime/artifacts/bf16-full/bin/Bootstrap.Engine.dll \
  --out artifacts/kv-gap-new-run
python3 scripts/summarize-kv-gap.py artifacts/kv-gap-new-run
```

Use a fresh output directory. The prior plan supplies the already verified model and fixture paths, which must exist locally. The runtime checkout, binary and source hashes are checked. No production implementation is changed by this profile.
