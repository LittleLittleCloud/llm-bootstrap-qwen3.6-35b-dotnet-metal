# Packed expert prefill experiment — 2026-11-03

The four proposed ablations are complete. The selected runtime change is an
opt-in, byte-preserving layout for expert weights, prepared once at load time.
The experiments do not establish parity with Splash.

## Matched kernel evidence

Reuse the shapes, weights, routing and GPU timing boundary in
[SPLASH_KERNEL_COMPARISON.md](SPLASH_KERNEL_COMPARISON.md). All ablations use the
same SIMD ownership and 8/16/32-row TensorOps path. Each result below is the median
of ten samples, in two five-sample passes with reversed starting order. Every
sample follows at least 250 ms of warm execution. No other model used the GPU.
Original and reversed runs include 1,680 timing samples, 336 case checks and
18,816 independent double-precision CPU reference points. Compilation is excluded.

| Operation | Tokens | Routes | Baseline ms | Packed ms | Reduction |
|---|---:|---|---:|---:|---:|
| gate_up | 32 | uniform | 2.633 | 2.129 | 19.1% |
| gate_up | 32 | skewed | 1.988 | 1.608 | 19.1% |
| gate_up | 512 | uniform | 2.893 | 2.355 | 18.6% |
| gate_up | 512 | skewed | 3.059 | 2.579 | 15.7% |
| gate_up | 2048 | uniform | 5.364 | 4.833 | 9.9% |
| gate_up | 2048 | skewed | 6.468 | 5.973 | 7.7% |
| down | 32 | uniform | 1.312 | 1.071 | 18.3% |
| down | 32 | skewed | 1.002 | 0.810 | 19.2% |
| down | 512 | uniform | 1.573 | 1.284 | 18.4% |
| down | 512 | skewed | 1.630 | 1.401 | 14.1% |
| down | 2048 | uniform | 2.929 | 2.593 | 11.5% |
| down | 2048 | skewed | 3.460 | 3.169 | 8.4% |

## Four ablations and the selected combination

The following GPU milliseconds are for 2048 source tokens, 256 experts, top-8,
and uniform routes. FP16 predecoded weights retain the exact baseline staging and
matrix tile; this is a diagnostic, not a proposal to expand the resident model.

| Variant | Gate/up ms | Down ms |
|---|---:|---:|
| Original | 5.364 | 2.929 |
| Same tile, resident predecoded FP16 | 6.869 | 3.616 |
| FP16 gate scratch / FP16 down output | 5.381 | 2.801 |
| Scale conversion once per 128 K | 5.370 | 2.949 |
| Packed bytes + scale reuse | 4.833 | 2.593 |
| Packed bytes + replicated tile-order scale metadata | 4.800 | 2.585 |
| 1–2-row column-dot fallback | 5.394 | 2.936 |
| Packed + FP16 epilogue + tiny fallback | 4.866 | 2.511 |
| Live indirect grid (arguments resident) | 5.396 | 2.913 |
| Live indirect grid including argument preparation | 5.385 | 2.906 |
| Packed including per-request repack + dispatch preparation | 18.766 | 9.514 |

1. **Weight decoding control:** predecoded FP16 is slower despite removing FP8
   conversion. This does not isolate arithmetic from bandwidth: the control
   doubles weight traffic. Expanding expert weights is not selected.
2. **Layout and scales:** packed four-byte loads improve every tested shape.
   Replicated scale metadata adds no consistent benefit beyond byte packing.
   Repacking on each request is prohibitively expensive, so the runtime packs
   once while loading verified weights.
3. **Precision:** FP16 gate scratch has no useful gate/up improvement. FP16 down
   output saves roughly 1–5% on the 512/2048-token cases, but is not part of this
   selected, precision-preserving runtime change. It remains a measured candidate;
   a whole-model gain from combining it with the selected layout is not claimed.
4. **Small experts and dispatch:** the tested column-dot fallback is slower on
   small experts. Its combination with packing loses performance on skewed
   routes. Indirect dispatch with argument preparation has only marginal effects.
   The runtime retains the original maximum grid and early-return policy.

The selected combination is packed FP8 bytes with original BF16 scales and scale
reuse. Accumulation, gate scratch and down output remain FP32; the existing fused
activation and matrix operands remain FP16. No requantization occurs.

## Correctness and memory contract

A separate broad fixture covers all finite E4M3 codes with non-power-of-two BF16
scales. All 168 cases and 9,408 sampled CPU references pass. All live outputs are
finite and output guards remain intact. Selected packed outputs are byte-identical
to the baseline across this fixture; reduced-precision variants have maximum
relative L2 error 0.000382. Experimental variants are not assumed equivalent to
production simply because their sampled references pass.

`packed-experts-self-test` checks every packed byte, unchanged scales, decode,
verification, tails 1/2/7/8/9/16/17/31/32, direct-versus-indirect results, guards,
and the private-map contract under Metal shader validation. The original on-disk
checkpoint must remain unchanged, and the GPU must observe the private mapped
writes. The runtime requires fused gathered FP16 prefill when packing is enabled.
Sequential decode and the small verification path retain their FP32 arithmetic.

Packing uses the checkpoint's existing MAP_PRIVATE mapping and a matrix-sized
CPU scratch buffer. It does not keep a second full model allocation. Private pages
become dirty and loading is more expensive; warm kernel timings do not measure
those costs. Full-model startup, memory, decode and evaluation are reported below.

## Reproduction and evidence

Use `scripts/run-expert-ablations.py` as documented in the README. Original
JSONL, identity files, broad fixture results, compiler failures during development,
and memory observations remain in ignored `artifacts/expert-ablations/`. The
initial screen contained three samples per case and is retained separately;
final tables use only the predeclared forward/reverse confirmation passes.
Source and binary hashes identify each pass. Results are specific to Apple M5 Pro
20 GPU cores / 48 GiB, runtime Metal 4.0 safe math, and these synthetic routes.
They do not measure real model route distributions or establish system SOTA.

## Whole-model development checks

Both configurations use the same engine build, official FP8 target, DFlash2,
FP16 KV, 2048-token prefill batches, staged8 DeltaNet and 32768-token context.
Baseline and packed models run in separate processes, serially in that order.
Each row below is the median of three uncached warm requests; an earlier full-answer
request warms the model. This process order is not a randomized crossover.

| Prompt tokens | Original ms | Packed ms | Throughput gain |
|---:|---:|---:|---:|
| 10520 | 5458.0 | 5277.1 | 3.4% |
| 29939 | 17713.5 | 17288.8 | 2.5% |

Both long-prompt full answers are correct. The arithmetic smoke's complete logits
are byte-identical. Both models pass 45 verifier and 48 prefix-cache checks.
Two warm 256-token decode requests give medians 18.47 and 19.05 tok/s respectively;
this short development probe is not the official 1000-token decode headline.
The final Metal shader-validation run passes 46 packed-loader/kernel checks.

Metal allocations match at 40,077,262,848 bytes after the long-context runs.
Private-page packing does not double that allocation, but it does not eliminate
system memory pressure: the packed process starts with more whole-machine swap
than the baseline. Startup/COW effects and ambient processes confound attribution
of the swap difference. No thermal or performance warning is recorded.
The full browser evaluation is a separate frozen experiment and reports all seven
tests; use its report and results PR for official performance and goal status.
