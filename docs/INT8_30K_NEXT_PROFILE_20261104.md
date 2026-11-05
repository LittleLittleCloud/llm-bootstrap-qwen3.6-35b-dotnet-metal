# Fresh INT8 KV + 30K profile: next optimization

Prioritize the routed MoE affine matmuls, starting with a K64 tile and then a
separate loading/compute pipeline experiment. This is a hypothesis supported by
kernel timings and source inspection, not a measured speedup from those changes.

## Run

The optimized K256 INT8 build was profiled again on M5 Pro using the same pinned
models and 29,939-token retrieval prompt as the [matched tuning controls](INT8_30K_PREFILL_20261104.md).
The run completed at 2026-11-05 05:30 UTC (November 4 local time). It used one
first-use request, three warm uncached requests and one independent all-kernel
counter request, with `BOOTSTRAP_PROFILE_SHAPES=1`.

Warm wall times: **12,325.917 / 12,375.151 / 12,578.341 ms**. Median throughput:
**2419.28 tok/s**. Median command-buffer GPU time: 12,349.628 ms; median CPU
encoding time: 9.576 ms. There are **16,324 dispatches in 17 command buffers**.
All five requests produce first token 22. The faster median than the prior
12,642 ms run reflects a separate run of the same binary; it is not another
code improvement or a matched old/new comparison.

No new whole-machine swap-outs or thermal warnings were recorded. Warm samples
still include about 37 MiB of swap-ins and existing swap usage, so these results
do not establish absence of memory pressure.

## Hotspots

The counter request takes 12,721.243 ms, 2.80% above the warm median. The table
uses summed encoder intervals (total 12,586.650 ms); these may overlap. Percentages
are shares of that sum, not exclusive wall-time attribution.

| Group | Interval sum, ms | Share | Dispatches |
| --- | ---: | ---: | ---: |
| MoE, shared expert, routing and combine | 4524.05 | 35.94% | 8240 |
| Attention and preparation | 3299.14 | 26.21% | 340 |
| Dense, norm and residual | 3102.58 | 24.65% | 5640 |
| DeltaNet and preparation | 1397.70 | 11.10% | 1890 |
| Draft context | 249.22 | 1.98% | 190 |
| Embedding and head | 13.96 | 0.11% | 24 |

Largest individual kernels:

| Kernel | Interval sum, ms | Calls |
| --- | ---: | ---: |
| `attention_online_int8_q32_k256_r2` | 3228.26 | 160 |
| `matrix_product_predecoded` (multiple shapes/groups) | 2445.04 | 3315 |
| `expert_gate_up_affine` | 2293.34 | 600 |
| `expert_matrix_affine_bf16_output` | 1275.63 | 600 |
| `delta_staged8` | 986.21 | 450 |

The previous candidate counter run has the same group ordering: MoE 4626 ms,
attention 3371 ms, dense 3119 ms, DeltaNet 1394 ms. Thus the prioritization is
stable across the two profiles despite wall-time variation.

## Concrete next experiment

The two routed expert matmuls account for **3568.97 ms**, about 28.36% of summed
intervals. Main gate/up shapes have 16,384 routed rows (2048 tokens × top-8),
K=2048 and N=512 per projection. Down uses K=512 and N=2048. Both run 600 times
across 40 layers and 15 chunks.

`expert_affine_accumulate` in `native/kernels.metal` currently:

1. Reads a K32 payload into registers and converts affine bytes to FP16 in
   threadgroup storage.
2. Synchronizes, runs one TensorOps matmul, then synchronizes before reuse.
3. Repeats 64 times for gate/up (128 SIMD-group barriers per SIMD group), or
   16 times for down (32 barriers).

The scale/bias pair is **already reused across two K32 steps**, so redundant
metadata reloads are not the proposed fix. A K64 tile could halve the number
of staging/matmul iterations and barrier sites while matching the quantization
group size. It increases shared storage/register requirements, so it may lose
occupancy; only measurement can decide. Parameterize the affine helper's
descriptor, slice and staging stride together: changing only the outer `KS`
template leaves the affine helper hard-coded to K32.

Test K64 alone first, then a separately controlled double-buffer/prefetch
variant. Preserve affine weight bytes, FP16 dequantization rounding, FP32
accumulation, fused gate/up activation and BF16 down output. Validate actual
affine weights and partial expert tiles; the older FP8 expert benchmark is not
an adequate substitute for this production path. Follow with bracketing native
30K runs and numerical checks before selecting a kernel.

For sizing only, reducing these two kernels' intervals by 10–20% corresponds
to roughly 0.36–0.71 seconds, or about 3–6% of current wall time **if the savings
transfer**. This is an opportunity estimate, not a performance forecast.

## Lower priorities

- INT8 KV preparation is only 70.77 ms, about 0.56% of summed intervals.
  Narrower storage or quantization-write fusion alone has little remaining
  headroom here. Attention computation is still a large second target at
  3228 ms; this profile does not prove which internal phase dominates it.
- Router plus grouping is 18.62 ms. Combine is 275.20 ms and expert-input gather
  is 243.35 ms, both much smaller than the routed matmuls. Existing gate/up
  fusion and BF16 combine are already enabled.
- CPU encoding is under 0.1% of median wall time, and GPU command-buffer time
  is close to wall time. This supports prioritizing GPU work; it does not prove
  GPU dispatch/scheduling overhead is zero. Dispatch count alone is a poor
  ranking criterion.
- Dense predecoded matmul is the next independent candidate. Weight unpack
  (`projection_unpack_half`) totals 314.02 ms across 2715 calls; removing these
  calls must be compared against any cost of decoding inside the matmul.

## Reproduction and saved evidence

`artifacts/int8-30k-next-profile/command.json` records the exact command and
working directory. The DLL directory is also the working directory, so the
copied Metal source is used. `hashes.json` pins DLL, native library, Metal source
and prompt. `stdout.jsonl`, `stderr.txt`, `exit.txt`, and `memory.jsonl` are raw
evidence; `summary.json` and `kernels.csv` contain grouped and shape-level results.
The source hash matches the K256 tuning report. No production code was changed
during this fresh profiling run.
