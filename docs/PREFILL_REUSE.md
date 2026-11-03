# Prefill reuse and scheduling — 2026-11-03

The next implementation after load-first affine experts improves warm development
throughput by **3.84% at 10K and 5.58% at 30K** against a later original-engine
control. It does not close the remaining historical gap to Splash. Weights,
quantization, arithmetic/storage precision, context limit and 2048-token chunks
are unchanged. These are development measurements; formal endpoint results are
submitted separately.

## Changes

- Online attention keeps the existing 32 x 256 tile and eight SIMD groups. When
  a complete history block is causally visible to every query, it skips per-key
  masks. At the causal edge and all tails it retains the original checks. Each
  SIMD group checks all row scale factors; if all are exactly one, it skips the
  output rescaling loop. This needs no storage beyond the existing 32 KiB tile.
  The original r1 kernel remains as a development control; the preset selects r2.
- Dense projections explicitly reuse FP16 input scratch for consecutive
  projections of the same unchanged normalized input. Routed experts have their
  own scratch, so shared-expert gate/up can reuse the router's conversion.
  The caller establishes this lifetime; this is not an implicit persistent cache.
  Small verification batches and fallback projection paths ignore the reuse hint.
- Affine expert down uses TensorOps cooperative FP32 stores into dynamic live-row
  extents. Gate/up and FP16-output diagnostic variants retain their old epilogues.
- DeltaNet staged8 fetches the following eight-token block into FP32 registers
  while its recurrence consumes the current shared-memory block. Block size,
  state layout, recurrence order and single-stage barriers are unchanged. The
  original staged8 kernel remains available to the development benchmark.

## Controlled kernel measurements

Apple M5 Pro, 20 GPU cores, 48 GiB. GPU command-buffer time, after warmup;
validation runs are separate from performance measurements. No owned GPU
workloads overlap. These are local fixtures, not whole-model rates.

| Operation | Reference ms | Candidate ms | Reduction |
| --- | ---: | ---: | ---: |
| Attention, 8192 history + 2048 queries | 16.230 | 14.463 | 10.9% |
| Attention, 30000 history + 2048 queries | 54.506 | 44.439 | 18.5% |
| Affine down, 2048 tokens, uniform routing | 2.276 | 2.159 | 5.1% |
| Affine down, 2048 tokens, skewed routing | 2.324 | 2.215 | 4.7% |
| DeltaNet staged8, 2048 tokens | 2.410 | 2.193 | 9.0% |

Attention uses five observations per configuration in sequential processes;
DeltaNet's final five observations alternate variant order inside one process.
Expert measurements use three alternating-order observations and the same
256-expert/top-8 dyadic fixture as the preceding affine profile. Kernel error
checks and timing evidence are retained, including rejected attempts. Precision
is unchanged, but these timings do not justify assigning an exact percentage of
whole-model improvement to each component.

## Whole-model checks and profile

Same affine-8 group-64 v2 cache and DFlash2 draft as the load-first experiment,
FP16 KV/operands, FP32 state/accumulation, staged8, 2048-token chunks, explicit
32768-token context. Cache manifest SHA-256:
`7106f2aaaa4ada10f4f3a8512d1b5ecef83f8e271b542ad6549d56ff582b4974`.
Each ordinary measurement generates one token after clearing prefix reuse;
loading and tokenization are excluded. The candidate's first run checks the
complete expected answer before three warm uncached observations.

| Input tokens | Initial reference ms | Later reference ms | Candidate ms | Later reference tok/s | Candidate tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10,520 | 4314.72 | 4308.47 | 4148.97 | 2441.70 | 2535.57 |
| 29,939 | 14398.59 | 14362.74 | 13603.20 | 2084.49 | 2200.88 |

Candidate ranges are 4148.17–4186.47 ms and 13585.23–13667.56 ms. The reference
improves slightly on repetition, so gains above use the later control. This is
not a randomized crossover or a confidence interval. The later control binary
has only a diagnostic attention-variant whitelist added to the load-first
source; its inference paths and original shader are unchanged. Binary and
shader hashes are in local identities.

A separate candidate counter pass and the later reference show:

| Sampled encoder interval | 10K reference → candidate ms | 30K reference → candidate ms |
| --- | ---: | ---: |
| Dense predecoded core | 1023.2 → 964.8 | 2837.8 → 2750.8 |
| Expert gate/up | 829.3 → 957.1 | 2272.3 → 2550.5 |
| Expert down | 487.4 → 463.0 | 1341.7 → 1272.8 |
| Online attention | 478.0 → 435.2 | 3874.7 → 3392.4 |
| DeltaNet | 371.7 → 337.3 | 1057.6 → 978.7 |

These intervals overlap and are affected by instrumentation and surrounding
work; they are not exclusive latency components. Gate/up's increase is preserved
rather than attributed to its unchanged arithmetic. Removing only cooperative
down stores measured 4170.55 / 13649.64 ms (three warm observations), slightly
slower than the complete candidate. This does not support reverting those stores,
but the small difference is not a stable isolated whole-model speedup estimate.

Ordinary dispatch counts fall from 10686 to 9906 at 10K and 25014 to 23064 at
30K. Separate sampled profiles are diagnostic only. System-wide memory monitors
covering candidate full-model checks, candidate profiling and the later control
recorded no new swap-outs and no thermal warnings; counters are not model-attributed.

Validation includes the 12-case online attention suite under Metal shader
validation, causal/query/KV tails, aliased output, poisoned KV padding and output
guards, plus a scaled-logit stress case. Selected outputs match r1 bitwise, with
maximum sampled CPU-reference error about 1.18e-5 including stress. DeltaNet
checks every output and state element against the existing recurrence, a double
reference at selected coordinates, split-state carry and guards; prefetched
staged8 also matches the original staged8 bitwise. The 187,194 affine checks
pass (maximum normalized error 1.01e-7). Full-model 10K/30K answers, three
256-token decode runs, 45 speculative-verifier checks and 48 cache checks pass.
Smoke logits are bitwise unchanged, SHA-256:
`5f85879f85492a1f7aa729069bd49b5df7ca8646cd4ddbfda2fdda56d4c720e2`.
These checks do not establish broad output-quality equivalence.

## Rejected directions

- K64 affine TensorOps steps worsen gate/up by approximately 20–30% on the main
  2048-token fixtures; increasing a tile dimension is not automatically faster.
- 64-row expert tiles initially failed output checks because FP32 gate scratch
  overlapped the other SIMD group's live weight stage. Adding a group barrier
  fixes correctness. Corrected M64/K32 is only 2.6% faster for uniform gate/up
  and about 1.5% slower for skewed routing; M64/K64 is much slower. Not adopted.
- A first cooperative gate/up store prototype fails compilation because FP32
  cooperative tensors cannot directly store FP16 tensors. Only the tested FP32
  down store is retained; no rounding change was introduced to bypass this.
- An additional atomic rescale flag makes the 32 KiB attention tile exceed its
  shared-memory budget. The retained SIMD-wide check needs no extra storage.
- Fast exponentials show essentially no gain and are not retained. Explicit
  softmax unrolling gives no convincing gain. Retesting alternative Q/K tile
  sizes after the attention changes still favors 32 x 256 / eight SIMD groups.
- DeltaNet prefetch16 measures 2.163 ms versus prefetch8 at 2.196 ms in the
  exploratory pass. The small difference does not justify changing the block
  size; staged8 remains selected.

## Reproduction and remaining work

```sh
dotnet "$ENGINE" profile-online-attention --case 8192,2048 \
  --variants q32_k256_s8_g8_r1,q32_k256_s8_g8_r2 --runs 5
dotnet "$ENGINE" profile-online-attention --case 30000,2048 \
  --variants q32_k256_s8_g8_r2,q32_k256_s8_g8_r1 --runs 5
MTL_SHADER_VALIDATION=1 dotnet "$ENGINE" profile-online-attention --variants selected --runs 1
MTL_SHADER_VALIDATION=1 dotnet "$ENGINE" profile-delta --runs 1
dotnet "$ENGINE" profile-delta --runs 5
dotnet "$ENGINE" affine8-self-test
```

The full-model commands are `moe-self-tests --packed-model-check` and
`profile-prefill --all-kernels`, with the same fixed 10K/30K prompt files and
runtime flags as the affine load-first experiment. Local ignored evidence is
under `artifacts/affine8-k64/` (including negative attempts) and
`artifacts/prefill-reuse-final/`; no raw traces or weights are published.

Dense core and expert gate/up remain the leading short-context sampled groups;
attention still leads at 30K. Further work should isolate those remaining costs
with matched workload and memory conditions. Historical Splash headline results
use a different checkpoint and complete-system configuration; these development
rates are not a replacement for either engine's official score.
