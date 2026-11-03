# Packed prefill profile — 2026-11-03 UTC

The current packed layout improves warm prefill, but experts remain the largest
combined kernel group. Long-context attention dilutes the end-to-end benefit.
This is a development profile and cost-sensitivity study, not an official
evaluation or a new runtime optimization. Production shaders/defaults are unchanged.

## Whole-model timing

Base engine: `2caf5890c3977b18a5aa137b6f3821a6966a8097`, with diagnostic-only
changes to record UTC run boundaries/layout selection and accept a second prompt.
Apple M5 Pro, 20 GPU cores, 48 GiB; official FP8 target and DFlash2; FP16 KV;
`--tuned-prefill --delta-mode staged8 --context-window 32768`; prefix reuse is
cleared before each request. Both processes produce the same first token on both
prompts. These one-token profiles do not establish full-answer quality.

Original and packed run serially in separate processes, original first. Each
prompt has one cold/first-use run, three warm runs and one all-kernel counter run.
The 30K prompt follows 10K in the same process. Checkpoint loading/tokenization
are outside generation timing, but first-use residency, allocations and
pipeline creation remain in the cold run. No other owned model was running.

| Prompt tokens | Original warm ms | Packed warm ms | Throughput gain | Packed GPU ms | Packed process CPU ms |
|---:|---:|---:|---:|---:|---:|
| 10520 | 5424.8 | 5171.9 | 4.89% | 5152.0 | 28.2 |
| 29939 | 17246.9 | 16804.2 | 2.63% | 16776.6 | 28.0 |

Medians above exclude instrumentation. CPU, GPU and host waiting overlap; do
not add them. Counter instrumentation changes encoder boundaries and adds
0.75–3.29% wall time in these four cases. Single-order measurements are not a
randomized repeated crossover and do not provide confidence intervals.

## Where GPU intervals go

Each percentage divides the **sum of encoder intervals**, not exclusive wall
duration. Intervals can overlap. The raw stderr preserves span/union/overlap
validation for every sampled command. These shares identify priorities; they
cannot be used as exact Amdahl fractions.

| Group, packed model | 10K sampled ms | Share | 30K sampled ms | Share |
|---|---:|---:|---:|---:|
| expert_gate_up | 1452.7 | 27.0% | 4030.3 | 22.8% |
| expert_down_and_tiny | 769.7 | 14.3% | 2099.2 | 11.9% |
| dense_and_draft_matmul | 1313.5 | 24.4% | 3668.9 | 20.8% |
| attention_layout_rope | 529.4 | 9.8% | 4063.9 | 23.0% |
| deltanet_conv_l2 | 532.3 | 9.9% | 1505.6 | 8.5% |
| expert_routing | 7.8 | 0.1% | 19.0 | 0.1% |

The two expert groups save about 211 ms of sampled intervals at 10K and 415 ms
at 30K. Ordinary wall time falls by 253 and 443 ms respectively. Their similar
scale supports an expert-local improvement, but overlapping/instrumented
interval differences are not an exact decomposition of the wall-time change.
Attention becomes as large as gate/up at 30K. Routing is only about 0.1% of
sampled intervals and is not a promising large-throughput target.

## Residency and cold-start cost

First 10K generation: original **14.45 s wall / 5.42 s GPU**, packed
**30.49 s wall / 5.16 s GPU**. Packed loading/private-map writes and first use
coincide with about **9.93 GiB of additional whole-machine swap-out pages**;
original has no new swap-out in its process window. These are uncompressed
page-count equivalents from `vm_stat`, not physical bytes written to disk.
Packed warm requests have **zero additional swap-out** within sampled windows,
but tens of MiB of swap-in remain. There are no reported thermal/performance
warnings. Whole-machine counters cannot identify the owner of individual pages.

Packed warm swap usage is about 6.7–6.9 GiB versus 4.5–4.7 GiB for original.
The fixed process order and ambient processes limit attribution. This confirms
a substantial first-use/residency concern; it does **not** establish ongoing
swap-out as the explanation for the modest warm prefill speedup.

## Expert loop sensitivity

The same native FP8 weights, BF16 scales, activations, routes, output contract
and GPU timing boundary are used for every variant. Compilation is warmed;
each timing follows at least 250 ms of execution. Two five-sample passes reverse
the starting order and alternate order within passes. The full model has exited.
All final cells below are medians of ten GPU samples.

- `half_bits`: decode E4M3 through exact FP16 bit patterns, then convert to FP32
  before the existing scale multiply/FP16 operand rounding.
- `lookup`: a 256-entry constant FP32 lookup replaces E4M3 arithmetic; scale
  multiplication, operand rounding, staging and TensorOps are retained.
- `prefetch`: the existing double-buffered load/compute schedule on packed bytes.
- `k64` / `k128`: larger K tiles, reducing loop/barrier counts while changing
  TensorOps shape and threadgroup-memory requirements.

| 2048 source tokens | Routes | Packed K32 | Half bits | Lookup | Prefetch | K64 | K128 |
|---|---|---:|---:|---:|---:|---:|---:|
| gate_up | uniform | 4.743 | 4.706 | 4.501 | 5.136 | 6.837 | 21.609 |
| gate_up | skewed | 5.986 | 5.968 | 5.421 | 7.832 | 10.956 | 26.187 |
| down | uniform | 2.574 | 2.572 | 2.299 | 2.771 | 3.318 | 10.787 |
| down | skewed | 3.165 | 3.183 | 2.527 | 4.341 | 5.502 | 13.349 |

The lookup reduces GPU time by **5.1–9.4% for gate/up** and **10.7–20.2% for
down** on the 2048-token cases. Half-bit conversion is essentially neutral.
No whole-model gain for lookup has been measured.

Validation: 840 timed samples; 252 matrix cases and 14,112
independent double-precision CPU reference points across exact and broad finite
E4M3 fixtures. Output guards and finiteness checks pass. The two alternate
decoders separately pass every one of the 256 input codes, including signed
zero and NaNs; their finite outputs are checked bitwise. Their full matrix
outputs have zero numerical difference from the baseline in these fixtures.
Maximum matrix relative L2 across all variants: 0.

These are sensitivity measurements, **not** additive conversion, memory, barrier
and TensorOps time components. Faster lookup does not tell us the total fraction
previously spent decoding; modified scheduling and compiler decisions can overlap
costs. Larger K and this prefetch implementation regress; fewer barriers alone
are not sufficient. The result does not rule out a different pipelining design.

## Next experiments

1. Carry the exact lookup decoder into a separately controlled whole-model
   candidate, including full-answer and sustained decode checks. Its kernel
   speedup will be diluted by the rest of prefill; do not promise a similar
   end-to-end percentage.
2. Compare fixed/padded activation tensor extents and compiler/math settings
   against the compact dynamic extents in this engine, holding decoded weights
   and routes constant. The pinned Splash staged loop uses fixed row extents
   and double buffering; merely copying double buffering was not beneficial here.
   No causal result for those remaining differences has been measured yet.
3. Treat long-context attention as a separate large target, now comparable to
   gate/up. Treat read-only packed backing storage as a cold-start/residency fix;
   the warm profile does not justify promising a large steady-state gain from it.

## Reproduction and raw evidence

`profile-prefill --model-dir <target> --draft-dir <draft> --tuned-prefill
--delta-mode staged8 --context-window 32768 --prompt-file <10k>
--long-prompt-file <30k> --runs 3 --all-kernels`, with/without `--packed-experts`.
Exact prompts are local synthetic fixtures; their SHA256 hashes and exact
commands are in `artifacts/packed-prefill-profile/`. Do not use these diagnostics
as endpoint-bench reports.

Run `scripts/profile-expert-stages.py --splash /path/to/pinned/splash --out
artifacts/packed-prefill-profile/stages --runs 5`, repeat with `--reverse`,
and run `--fixture broad --runs 0`. Splash is pinned to
`1482f6bbc2abf12375332725302019594150b6cb`; its shader includes are flattened
without source changes. The diagnostic adds generated shaders in artifacts only.
`scripts/summarize-packed-profile.py artifacts/packed-prefill-profile` combines
paired model JSONL, memory JSONL and the completed stage results.

Raw files, generated source/binaries, hashes, commands, memory samples, cold
and instrumented runs remain in ignored `artifacts/packed-prefill-profile/`.
This study neither re-runs the official evaluation nor updates frozen reports.
