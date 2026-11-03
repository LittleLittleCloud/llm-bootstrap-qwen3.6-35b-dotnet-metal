# Affine-8 prefill profile — 2026-11-03

The next measured deficit is the routed expert kernel implementation, even
after both sides use affine-8 group-64 weights. At 2048 source tokens, our
gate/up takes 29–35% longer than the pinned Splash kernel and down takes
35–39% longer. Dense projections do not show that deficit. At 30K input,
attention becomes the largest individual sampled group and merits a separate
long-context optimization effort.

These are development measurements, not new endpoint evaluation results.
The engine is `9c5889413defeeb5eb30f202fb14d2d6d8a6e721`; serving shaders and
defaults are unchanged. Splash is the historical comparison revision
`1482f6bbc2abf12375332725302019594150b6cb`, not a claim about its latest release.

## Whole-model measurements

Apple M5 Pro, 20 GPU cores, 48 GiB; affine cache v2, DFlash2, FP16 KV and
operands, FP32 accumulation/state, 2048-token chunks, staged8 DeltaNet,
32768-token context. Cache-manifest SHA-256:
`7106f2aaaa4ada10f4f3a8512d1b5ecef83f8e271b542ad6549d56ff582b4974`.

The same local synthetic 10K/30K prompts used by the previous profile run
serially in one process. Each has one first-use run, three warm uncached
runs, and one all-kernel counter run. Prefix reuse is cleared each time;
tokenization/loading are outside the timed generation. Each request generates
one token; first-token identity is stable within each prompt across all five
runs. This is not a new full-answer quality test.

| Input tokens | Warm wall median ms | Warm GPU median ms | Process CPU median ms | Development tok/s |
|---:|---:|---:|---:|---:|
| 10,520 | 4,714.44 | 4,690.91 | 29.98 | 2,231.44 |
| 29,939 | 15,804.08 | 15,680.69 | 103.13 | 1,894.38 |

Warm wall ranges are 4713.73–4773.91 ms and 15764.22–15805.75 ms.
CPU execution, GPU work and host waiting overlap; do not add them. GPU time
is about 99.5%/99.2% of warm wall time, so host dispatch is not the leading
optimization target. First 10K generation is 11.93 s wall / 4.71 s GPU;
cold residency/pipeline costs remain distinct from warm throughput.

System-wide memory sampling records no additional swap-outs in either warm
window and less than 0.4 MiB swap-in per window. Swap usage stays at
4364.94 MiB; wired memory spans approximately 39.78–39.89 GiB. No thermal
or performance warning is recorded. This rules out sustained swap-out as
the observed warm bottleneck, not every possible residency cost.

## GPU interval breakdown

Percentages divide the **sum of sampled encoder intervals**, not exclusive
wall time. Intervals overlap, and per-command span/union checks are preserved
in raw stderr. Instrumentation raises wall time by 12.39% at 10K and 6.97%
at 30K; use these samples to prioritize work, not as an exact Amdahl model.

| Group | 10K sampled ms | Share | 30K sampled ms | Share |
|---|---:|---:|---:|---:|
| Dense and draft matmul | 1288.8 | 26.5% | 3594.3 | 22.2% |
| Expert gate/up | 1101.1 | 22.7% | 2954.8 | 18.3% |
| Expert down and tiny paths | 667.3 | 13.7% | 1742.1 | 10.8% |
| Attention/layout/RoPE | 556.7 | 11.5% | 4261.1 | 26.3% |
| DeltaNet/conv/L2 | 537.5 | 11.1% | 1535.8 | 9.5% |
| Expert combine | 165.2 | 3.4% | 398.5 | 2.5% |
| Dense input conversion | 152.7 | 3.1% | 435.6 | 2.7% |
| Dense weight decoding | 115.8 | 2.4% | 480.5 | 3.0% |
| Expert input conversion | 81.9 | 1.7% | 236.1 | 1.5% |
| Routing | 7.9 | 0.16% | 19.5 | 0.12% |

Other kernels account for the remainder. The core predecoded dense kernel
alone is 983.2 ms / 20.2% at 10K. A high share alone is not evidence that
this core is slower than Splash; the matched measurements below test that.

## Current affine kernels versus Splash

Same 256 experts, top-8 routing, logical activation values, decoded weight
values and GPU timing boundary. Both weights use unsigned affine-8 group-64
and BF16 scale/bias. Every scalar of our packed fixture is independently
checked against the source value. Physical layouts and original epilogues
remain implementation-specific: ours has compact FP16 activation rows,
FP32 gate scratch/down/dense outputs; Splash uses padded BF16 rows and BF16
gate/up rounding and outputs. Thus this is not a single-variable precision
or layout comparison. The production checkpoints also differ.

Two five-sample passes reverse initial order and alternate order within each
pass. Each measured operation follows at least 250 ms warm execution;
compilation, packing and CPU references are excluded. No model or other
owned GPU benchmark runs concurrently. Numbers are GPU command-buffer
elapsed milliseconds, median of ten observations.

| M=2048 operation | Routing | Ours affine ms | Splash ms | Ours / Splash |
|---|---|---:|---:|---:|
| Gate/up, K=2048 N=512 | Uniform | 4.306 | 3.330 | 1.29× |
| Gate/up, K=2048 N=512 | Skewed | 4.759 | 3.513 | 1.35× |
| Down, K=512 N=2048 | Uniform | 2.581 | 1.916 | 1.35× |
| Down, K=512 N=2048 | Skewed | 2.742 | 1.977 | 1.39× |

Uniform has 64 rows/expert; skewed alternates 1/127. M=32 and M=512 are
additional controls in the raw results. These are synthetic route populations,
not a claim that the real model has either distribution.

| Dense M=2048, K → N | Ours including affine decode ms | Splash ms |
|---|---:|---:|
| 2048 → 512 | 0.149 | 0.187 |
| 2048 → 2048 | 0.575 | 0.620 |
| 2048 → 8192 | 2.313 | 2.348 |
| 4096 → 2048 | 1.196 | 1.231 |
| 512 → 2048 | 0.144 | 0.159 |

Dense input conversion is excluded on both sides; ours includes weight decode
and FP16 TensorOps in the same command, Splash stages affine inside its matmul.
This does not exclude fusion opportunities around dense operations.

## Controlled hypotheses

Changing only our expert activation tensor extent from live rows to the
selected 8/16/32 rows changes the M=2048 medians by less than 0.3% in either
direction. Switching only our compilation from safe to fast math saves less
than 0.5%. Combining both stays below 0.5%. These do not explain the remaining
29–39% expert-kernel time deficit. Fixed-row diagnostics allocate 32 extra
activation rows and mask output stores; a production implementation would
need its own allocation proof. No fast-math or extent change was adopted.

The complete matched study contains 780 timed observations and 162 checked
cases including a separate Metal shader-validation pass. Each side passes
8718 sampled CPU reference points across these passes, all live outputs are
finite, and output guards pass. Sampled errors are zero for this deliberately
dyadic fixture; this is not a broad precision or whole-model equivalence claim.

Both libraries are runtime-compiled with Metal 4.0. Ours uses safe math by
default; Splash uses fast math with its original reassociation pragmas. The
absence of the offline Metal compiler remains a difference from Splash's
normal offline build. Original third-party kernel sources are not modified.

## Positive scheduling intervention

The current affine loop loads a four-byte vector, decodes it with FP32 FMA,
converts it to FP16 and stores it in threadgroup memory, repeated eight times
per lane per K32 step. The `load_first` diagnostic loads all eight byte vectors
into registers before decoding/storing them. The format, metadata, compact
activations, live-row tensor extents, TensorOps geometry, arithmetic precision,
epilogue and dispatch grid remain unchanged. A second diagnostic, `prefetch`,
uses alternating threadgroup stages and loads the following K32 step before
running the current matmul. Production sources remain unchanged.

| M=2048 operation | Routing | Current ms | Load first ms | Time saved | Double-buffer prefetch ms |
|---|---|---:|---:|---:|---:|
| Gate/up | Uniform | 4.285 | 3.655 | 14.7% | 3.786 |
| Gate/up | Skewed | 4.766 | 3.870 | 18.8% | 4.085 |
| Down | Uniform | 2.557 | 2.289 | 10.5% | 2.314 |
| Down | Skewed | 2.730 | 2.347 | 14.0% | 2.411 |

These medians come from a separate two-pass, five-sample-per-pass comparison
with its own interleaved current-kernel control. Double buffering improves
over the current loop but loses to the simpler load-first schedule. This
supports changing load/decode scheduling as the next experiment; it does not
measure an additive "memory stall" fraction or prove a specific compiler
instruction sequence. The uniform gate/up load-first median varies from
3.761 ms in the first pass to 3.646 ms in the second; all observations,
including isolated slower samples, are retained.

The schedule study has 480 timed observations, 108 checked cases including
Metal shader validation, and 6336 sampled CPU reference points per side.
Every live output of both schedule variants is also compared **bitwise** to
the current kernel; all match. No lower precision or requantization is used.
Validation covers the measured dyadic fixtures, not arbitrary model quality.

Even the faster load-first experts still trail Splash: approximately 9–10%
for gate/up and 21–22% for down in this second study. Its improvement cannot
by itself be treated as a 33% whole-model throughput gain. The remaining
down-path differences include layout, output precision/stores and dispatch;
they require separate interventions before assigning causality.

## Whole-model diagnostic confirmation

A generated shader replaces only `expert_affine_accumulate` with the
load-first schedule. The original executable and all runtime flags, weights,
prompts and allocations stay the same. It runs from an isolated diagnostic
working directory, followed by a fresh process using the original shader.
No serving source is replaced. Runtime shader SHA-256:
`3374d260a815e91cb3f3f6fbd22d2993ff3ab652020c53762f864ed0307494d0`.

| Tokens | Initial reference ms | Later reference ms | Candidate ms | Later reference tok/s | Candidate tok/s | Gain versus later reference |
|---:|---:|---:|---:|---:|---:|---:|
| 10,520 | 4714.44 | 4621.43 | 4311.93 | 2276.35 | 2439.74 | 7.18% |
| 29,939 | 15804.08 | 15253.41 | 14354.88 | 1962.77 | 2085.63 | 6.26% |

Each cell is the median of three warm uncached requests. Candidate ranges
are 4301.87–4344.55 ms / 14337.89–14429.61 ms. Later-reference ranges are
4618.10–4673.24 ms / 15248.79–15313.44 ms. GPU medians likewise fall from
4602.05 to 4280.01 ms and 15225.48 to 14326.88 ms. First tokens match across
every run and instrumentation mode for both prompts. Warm windows have no
new swap-outs or thermal warnings.

The initial reference would suggest 9.33%/10.10% higher throughput. However,
the reference itself speeds up on repetition (wall time -1.97%/-3.48%), so
report the smaller 7.18%/6.26% comparison as the follow-up observation. This
reference/candidate/reference sequence reduces order concerns but is not a
randomized repeated crossover or a confidence bound.

Later-reference versus candidate counter intervals provide a useful cross-check:

| Kernel | 10K reference → candidate ms | 30K reference → candidate ms |
|---|---:|---:|
| Expert gate/up | 1050.25 → 833.98 | 2892.78 → 2264.84 |
| Expert down | 594.60 → 490.53 | 1651.40 → 1350.27 |
| Dense core | 1003.14 → 1009.49 | 2844.38 → 2804.73 |
| Online attention | 479.28 → 481.34 | 3858.91 → 3888.52 |

The sampled expert savings are about 320/929 ms, close in scale to the
309/899 ms ordinary wall savings, while dense and attention stay close.
Counter overhead is 0.99%/2.59% for the later reference and 1.48%/3.18% for
the candidate. Overlap and instrumentation still prevent exact attribution.

The generated whole-model shader also passes the complete existing affine
suite under Metal shader validation: **187,194 checks**, maximum normalized
CPU-reference error `1.0058284e-7`. This includes general scalar codes,
metadata, row tails, fused experts, dense, embedding and cache validation;
the schedule microbench additionally checks all live outputs bitwise.
The model runs remain one-token development profiles, not a substitute for
full-answer checks and the complete endpoint evaluation before adoption.

## Next optimization order

1. Promote the load-first affine expert schedule into a separately reviewed
   implementation experiment, then run full-answer/cache/speculation checks
   and the pinned full evaluation. It has both isolated and whole-model
   evidence and does not change weights or arithmetic precision.
2. Isolate the remaining down-path differences: compact versus tiled output,
   FP32 versus narrower output storage, and fusion with weighted combine.
   A smaller output type also changes numerical behavior and must be tested
   independently. Do not claim that stores explain the whole remaining gap.
3. For 30K throughput, profile and tune the online attention loop separately;
   its current 26% sampled share makes it a larger long-context target than
   further routing work. No fresh matched Splash attention measurement was
   made here, so that share alone does not establish an attention deficit.

Fixed-row extents, global fast math and additional routing work are not
supported as the next large gains. Dense surrounding conversions/fusion can
still matter, but its matched matmul itself is already competitive.

## Reproduction and evidence

Use `profile-prefill` with `--affine-cache`, the preset above, both prompt-file
options, `--runs 3 --all-kernels`. Exact commands, prompt hashes, source/binary
hashes, raw JSONL, memory samples and counter diagnostics are local-only under
`artifacts/affine8-prefill-profile-20261103/`.

```sh
python3 scripts/profile-affine-splash.py --splash /path/to/pinned/splash \
  --out artifacts/affine8-prefill-profile-20261103/kernels --label forward --runs 5
python3 scripts/profile-affine-splash.py --splash /path/to/pinned/splash \
  --out artifacts/affine8-prefill-profile-20261103/kernels --label reverse --runs 5 --reverse
MTL_SHADER_VALIDATION=1 python3 scripts/profile-affine-splash.py \
  --splash /path/to/pinned/splash --out artifacts/affine8-prefill-profile-20261103/kernels \
  --label validation --runs 0
python3 scripts/summarize-affine-profile.py artifacts/affine8-prefill-profile-20261103
```

For the scheduling study use `scripts/profile-affine-schedules.py` with the
same `--splash`, `--label`, `--runs` and `--reverse` options, writing to the
`schedules` subdirectory. Keep shader validation in its separate zero-timing
`validation` pass. The summarizer includes this study when its serial driver
has written `schedules/done.txt`.

The historical headline comparison is 2171.1 versus 2897 tok/s: matching that
rate requires 33.4% more throughput, equivalently 25.1% less time at fixed work.
The development rates above use different timing boundaries and prompts and
must not replace either official score. No evaluation report or index changed.
