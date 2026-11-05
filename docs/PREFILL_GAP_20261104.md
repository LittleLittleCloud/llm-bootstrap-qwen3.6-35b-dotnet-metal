# Current prefill gap to Splash — 2026-11-04

The next target is the **MoE output/combine path**, starting with a precision-preserving combine-to-next-normalization fusion and a separate investigation of expert-down output traffic. Current dense TensorOps is competitive. The native INT8-KV Splash reference is faster overall, but its BF16-KV control is slightly slower than ours at 30K. A smaller expert threadgroup-memory diagnostic did not improve performance and is not adopted.

This is a development profile, not a new endpoint evaluation, implementation speedup, or quality-equivalence result. Production kernels and serving defaults remain unchanged.

## Identity and timing boundary

- Machine: Apple M5 Pro, 20 GPU cores, 48 GiB.
- Ours: merged main `c3de96555f1097f6a21a9172c4ad06b011d2c2a2`, affine-8 group-64 v2 derived from official Qwen3.6 FP8, BF16 DFlash2 weights, FP16 KV/operands, FP32 accumulation/state/output, dispatch mask 29, staged8 recurrence, fused preparation, 2048-token chunks, 32768 context.
- Splash host source: `1482f6bbc2abf12375332725302019594150b6cb`, with the previously reviewed host-only counter/fixture patch; original installed 1.4.0 metallib SHA-256 `e30124e8bceba1a9de7afe75cf1a10439bcfa5c1b777490c829ba4da0d0cd1cd`. Its target is the independently quantized MLX affine-8 checkpoint; native activations/output are BF16, draft projections affine-4, recurrent state FP32. All experts are resident.
- Same 10,520 and 29,939 input token IDs, freshly regenerated with the current tokenizer and checked against every supplied Splash ID. Loading and tokenization are excluded. First-token production and draft context setup are included.
- Each case has one first-use run, three warm uncached requests, and a separate counter request. Ours runs before and after all four Splash controls. GPU workloads are serial; no OS/power settings are changed.
- Ours clears prefix reuse but still creates its ordinary prefix checkpoints. Splash's direct runtime harness does not materialize those checkpoints and captures only its final draft context window. These are native-system comparisons, not identical-weight/precision causal comparisons.

The chat's configured `llm-bootstrap-qwen3.8-35b` path did not exist. The actual current experiment and runtime are Qwen3.6; diagnostics use a new `codex/prefill-gap-profile-20261104` worktree without changing the existing serving checkout.

## Fresh uninstrumented results

| Input tokens | Ours before, ms / tok/s | Ours after, ms / tok/s | Splash INT8 KV, ms / tok/s | Splash BF16 KV, ms / tok/s |
|---:|---:|---:|---:|---:|
| 10,520 | 3869.64 / 2718.6 | 3892.21 / 2702.8 | 3469.34 / 3032.3 | 3597.60 / 2924.2 |
| 29,939 | 13024.28 / 2298.7 | 13233.60 / 2262.3 | 12096.72 / 2475.0 | 13407.11 / 2233.1 |

Against native INT8 KV, ours takes 11.5–12.2% more time at 10K and 7.7–9.4% more at 30K. These ranges are the two process medians, not confidence intervals. Against Splash BF16 KV, ours takes 7.6–8.2% more time at 10K but 1.3–2.9% less at 30K. FP16 and BF16 are both two-byte formats, but they do not have identical numerical behavior.

Within Splash, changing only KV from INT8 to BF16 adds 128.3 ms at 10K and 1310.4 ms at 30K. Its sampled attention/preparation group accounts for approximately the same difference. A large long-context attention share does not by itself demonstrate that our attention implementation is inferior.

Warm GPU time is 98.6–99.3% of ours' wall time; warm process CPU medians are 38.7–105.7 ms. Host work and GPU time overlap. This does not support CPU launch overhead as the principal remaining bottleneck.

## Shape-attributed model profile

The extra diagnostic build changes host labels only; the Metal shader hash matches the production baseline. Its warm medians are 3853.20 ms / 2730.2 tok/s at 10K and 13092.69 ms / 2286.7 tok/s at 30K, consistent with the earlier process range. Sampling overhead is 15.0% / 5.8%.

| Sampled encoder intervals, ms | Ours 10K | Splash INT8 10K | Ours 30K | Splash INT8 30K |
|---|---:|---:|---:|---:|
| MoE including shared expert, routing, gather and combine | 1743.6 | 1561.1 | 4729.4 | 4296.3 |
| Dense plus remaining norm/residual preparation | 1109.3 | 979.2 | 3091.8 | 2775.3 |
| DeltaNet including preparation and gated normalization | 485.4 | 411.0 | 1380.5 | 1166.1 |
| Attention plus preparation | 474.3 | 486.7 | 3559.4 | 3760.3 |
| Draft context work, excluding final normalization | 64.8 | 10.2 | 206.9 | 10.8 |
| Embedding/output head | 10.1 | 2.6 | 13.0 | 4.5 |

These are diagnostic interval sums, not mutually exclusive latency components. Remaining fused-epilogue and normalization boundaries differ; a row difference is not an isolated causal speedup estimate. Nevertheless, the MoE deficit appears in both model profiles and matched expert-down kernels. The isolated dense measurements distinguish a competitive matmul core from additional norm/packing/conversion work around it.

Current `batch_mix` sums to 144.3 / 412.1 ms at 10K / 30K; Splash `moe_combine` sums to 78.6 / 221.0 ms. Ours reads FP32 expert results and separately applies the shared gate; Splash consumes BF16 expert results and has already prepared shared routing weights. This motivates a down-plus-combine experiment, not a claim that either precision or fusion alone explains the difference.

Disabling only our prefix caching measures 3842.21 / 13021.15 ms versus the shape build's 3853.20 / 13092.69 ms: 0.29% / 0.55% apparent throughput improvement. Both controls have zero new warm-window swap-outs and the same 7218 / 16344 dispatches. This separate-process difference is small compared with observed timing drift and does not establish a reliable implementation gain. Source inspection confirms `PrefixBoundaries` still selects the same chunks when caching is disabled. Prefix snapshot copying does not explain most of the remaining gap; disabling useful cache behavior is not the recommended optimization.

## Matched current-kernel measurements

The kernel harness uses identical logical activations and exactly checked decoded dyadic weights, 256 experts, top-8 routing, and the original physical layouts/epilogues. Ours retains FP16 activations and FP32 down output; Splash retains BF16 activations/output and its gate/up rounding. Native target checkpoints are not involved in this isolated test. Every measured operation receives at least 250 ms of warming; two three-observation passes reverse initial order and alternate order internally.

| 2048-token operation | Routing | Ours, ms | Splash, ms | Ours additional time |
|---|---|---:|---:|---:|
| Expert gate/up, K=2048 N=512 | uniform | 3.658 | 3.508 | 4.3% |
| Expert gate/up, K=2048 N=512 | skewed | 3.875 | 3.531 | 9.8% |
| Expert down, K=512 N=2048 | uniform | 2.172 | 1.876 | 15.8% |
| Expert down, K=512 N=2048 | skewed | 2.234 | 1.955 | 14.3% |

Uniform routing has 64 rows/expert; skewed routing alternates 1/127. These are synthetic controls, not measured route distributions of the real model.

| Dense M=2048, K → N | Ours incl affine unpack, ms | Splash, ms |
|---|---:|---:|
| 2048 → 512 | 0.149 | 0.186 |
| 2048 → 2048 | 0.578 | 0.616 |
| 2048 → 8192 | 2.314 | 2.346 |
| 4096 → 2048 | 1.244 | 1.228 |
| 512 → 2048 | 0.144 | 0.159 |

The one 1.3% slower dense shape is not evidence of a broad dense-core deficit. Input packing is excluded from both sides of these kernel measurements. Weight unpack is included for ours; Splash unpacks inside its matmul.

The two normal passes retain 324 timing observations and 60 checked cases (2996 CPU reference points per side, zero measured error on the deliberately dyadic inputs). A separate production-only Metal shader-validation pass completes all 18 cases with finite outputs and intact guards. This validates the measured current-kernel fixtures, not the rejected compact-stage variant or general whole-model numerical equivalence.

## Rejected compact-stage diagnostic

The current affine expert loop uses one staging buffer per SIMD group but inherits an 8 KiB allocation from generic double-buffer variants. A generated diagnostic reduces it to 4 KiB. For gate/up, an additional threadgroup barrier is required before FP32 gate scratch reuses the up group's stage. The intended control preserves values, rounding, live-row extents, arithmetic order and dispatch grid; the validation failure below prevents treating that intention as established correctness. The production shader is unchanged.

| 2048-token operation | Original ms | Compact-stage ms | Outcome |
|---|---:|---:|---|
| Gate/up uniform | 3.658 | 3.777 | 3.3% slower |
| Gate/up skewed | 3.875 | 3.953 | 2.0% slower |
| Down uniform | 2.172 | 2.170 | effectively unchanged |
| Down skewed | 2.234 | 2.247 | 0.6% slower |

Smaller threadgroup allocation alone is not the next optimization. It also **fails full-output bitwise comparison with Metal shader validation enabled**, at the down/uniform/M=512 case; validation stops there with exit 1. Both ordinary timing passes passed their complete-output bitwise comparisons, CPU samples, finite-output and guard checks, but that does not override the validation failure. The compact diagnostic is rejected; its timings are retained only as negative research evidence, never as a valid candidate implementation. The unmodified production kernels are validated separately.

## Next implementation experiment

1. **Fuse MoE combine with the following layer's input RMSNorm/FP16 packing.** Retain the FP32 residual output for draft capture and caching, preserve the ordered eight-expert sum, and reuse each combined value for the RMS reduction and packed projection input. This targets repeated reads, scalar-gate work and materialization surrounding MoE. Compare the whole `combine → norm → pack` sequence, then alternating whole-model runs. The existing vector-only `batch_mix_vec4` diagnostic was already ineffective; do not repeat it as the main hypothesis.
2. **Isolate the expert-down output representation.** The current 2048-token routed output contains 8 × 2048 × 2048 FP32 values: 128 MiB written and subsequently consumed by combine per layer/chunk. A two-byte temporary would halve that payload, but it changes numerical behavior. First separate storage/layout effects from matmul by measuring down plus combine together; if narrower storage is tried, retain FP32 accumulation and perform complete-output, cache/speculation and task-quality checks before any promotion. The measured 14–16% down deficit is not proof that output stores explain all of it.
3. **Consider bounded draft feature work after the MoE experiment.** Our draft context still processes every prompt chunk whereas Splash captures the required final window. Preserve the context needed by every saved prefix checkpoint; simply skipping early draft work would break cache-restoration behavior.

Do not prioritize another blind dense tile sweep, generic dispatch-count reduction, or blanket attention rewrite on the basis of these results. INT8 KV is a separate numerical/storage experiment, not a precision-preserving kernel optimization. No whole-model speedup is predicted by summing sampled encoder intervals.

## Instrumentation and evidence

Counter mode changes encoder boundaries and can perturb memory residency. Initial ours counter overhead was 27.1% at 10K and 63.7% at 30K; the later process observed 11.1% and 6.1%. Splash overhead was approximately 0–0.6%. Counter wall time is never used as throughput evidence.

The first ours lifecycle observed 2227.9 MiB of system-wide new swap-outs, including loading/first-use/counters; the warm 10K window had none, and the first warm 30K window had 56.6 MiB. The later ours warm windows had zero new swap-outs. No thermal warnings were recorded. These are whole-machine observations, not allocations or swap attributed solely to the engine.

Shape attribution is opt-in with `BOOTSTRAP_PROFILE_SHAPES=1`; it appends KernelArgs dimensions to sampled kernel labels without changing any dispatched parameters, pipelines or command boundaries. This is necessary because the same dense kernel implements target, shared-expert and draft FC work. Splash already includes shared-expert projections inside its MoE kernels; comparing its MoE group directly with only our routed experts would be misleading. Some norm/epilogue fusion boundaries still differ.

Local raw evidence is under `artifacts/current-gap/`: commands and binary/source hashes in `provenance.json` and per-case `identity.json`, full stdout/stderr, memory samples, Splash interval traces, `summary.json`, `kernels.csv`, and generated isolated-kernel source/results. All first-token repeat checks are retained; they are not full-answer or quality-equivalence tests. Official reports and leaderboard files are untouched.

All 60 model requests returned the same first token (22); this is a repeatability check, not a complete-answer score. Production shader files in both executable packages match the pinned baseline byte-for-byte. The 72-file local evidence manifest is `artifacts/current-gap/evidence-manifest.json`, SHA-256 `65c7195faf712c2e3d78656eeb6278eb45804b9a6e6bf23ce605efd7c018e130`.

Reproduction:

```sh
dotnet build src/Bootstrap.Engine -c Release
python3 scripts/profile-current-gap.py --help
python3 scripts/profile-current-gap-kernels.py --splash /path/to/clean/pinned/splash --out artifacts/current-gap/kernels
dotnet build src/Bootstrap.Engine -c Release -o artifacts/shape-bin
python3 scripts/profile-current-gap-followup.py artifacts/current-gap --dll artifacts/shape-bin/Bootstrap.Engine.dll
python3 scripts/summarize-current-gap.py artifacts/current-gap
```

Use the exact prior fixture/model/metallib locations recorded by the primary driver's arguments. Output directories must be new so observations cannot be silently overwritten.

## Follow-up: dispatch and 10K prefill timelines

Fresh serial captures use the same 10,520 input IDs and model configurations as above. Each process performs one cold request, three warm uncached requests without timeline collection, one CPU-only timeline request, then one CPU + GPU-counter request. Loading and tokenization are excluded; reset, prefill and first-token selection are included. These are direct-runtime diagnostics, not endpoint TTFT measurements.

| Configuration | Warm median, ms | Tokens/s | Compute dispatches | Command buffers | CPU encode, ms | CPU dispatch calls, ms |
|---|---:|---:|---:|---:|---:|---:|
| Ours FP16 KV | 3807.254 | 2763.1 | 7218 | 8 | 4.537 | 1.878 |
| Splash INT8 KV | 3438.091 | 3059.8 | 3543 | 6 | 0.983 | 0.874 |
| Splash BF16 KV | 3569.519 | 2947.2 | 3543 | 6 | 0.993 | 0.878 |

The CPU columns come from the CPU-only trace, not the warm median. Dispatch call time covers the native dispatch wrapper's pipeline/buffer binding and encoding; CPU encode covers the enclosing command construction and includes those calls. They are nested, not additive. Ours' reset-only blit command is included in the eight command buffers but has zero compute dispatches. Splash's CPU span starts at its backend submission encoding; runtime command-graph preparation before that point is outside this span. For ours, native encoding time also includes managed work between `Begin` and `End`. These are observed submission spans, not matched-language microbenchmarks.

Ours has 2.04 times the dispatch count, but the additional CPU encoding is only about 3.55 ms. In the CPU-only captures, GPU command execution covers 3785.536 / 3431.146 / 3553.518 ms of 3810.561 / 3439.908 / 3562.351 ms request wall time. CPU API dispatch overhead does not explain the roughly 369 ms warm latency difference against Splash INT8. Dispatch count can still matter through GPU-side launch/scheduling, fusion, intermediate traffic and repeated work; these measurements do not isolate those effects.

### A small tail that the previous aggregate profile concealed

Shape-labelled GPU traces show ours executes chunks `2048 × 5, 272, 8`, whereas Splash's prefill budget produces `2048 × 5, 280`. `PrefixBoundaries` retains the last message end at token 10512, in addition to the full prompt end at 10520. `Generate` honors that checkpoint before the final eight chat-header tokens. This splits the last 280-token block and selects the tiny-row kernel path for the final eight rows.

| Ours command, CPU-only trace | Rows | Dispatches | GPU time, ms |
|---|---:|---:|---:|
| Reset / clear | — | 0 | 1.564 |
| Prefill 1 | 2048 | 1014 | 641.840 |
| Prefill 2 | 2048 | 1017 | 672.706 |
| Prefill 3 | 2048 | 1014 | 704.747 |
| Prefill 4 | 2048 | 1017 | 735.321 |
| Prefill 5 | 2048 | 1014 | 765.613 |
| Prefill 6 | 272 | 1017 | 204.495 |
| Prefill 7 | 8 | 1125 | 59.250 |

The tail is a concrete follow-up experiment: measure a policy that avoids splitting a tiny final suffix at a prefix checkpoint, or retains the checkpoint using an implementation that does not run a separate full-network tiny block. Prefix reuse and draft-context restoration semantics must be checked. The 59.25 ms is an observed command cost, not a promised saving: merged rows still require computation, and changing checkpoint policy changes reusable prefixes. No checkpoint or kernel optimization was applied in this profiling task.

### Sampling overhead and timeline interpretation

| Configuration | CPU-only trace, ms | Difference vs warm median | GPU-counter trace, ms | Difference vs warm median |
|---|---:|---:|---:|---:|
| Ours | 3810.561 | +0.087% | 4029.623 | +5.841% |
| Splash INT8 | 3439.908 | +0.053% | 3461.232 | +0.673% |
| Splash BF16 | 3562.351 | −0.201% | 3581.779 | +0.343% |

These are single diagnostic repeats relative to three warm repeats; signed differences include timing drift, not just instrumentation cost. CPU-only mode retains ordinary compute encoders and command boundaries. GPU sampling uses one encoder per dispatch inside the original command buffers. Ours' eight-row final command grows from 59.250 to 214.054 ms: this accounts for 81.3% of the observed increase in total command GPU duration. The sampled tail's gaps must not be interpreted as normal production idle time.

Host events use `mach_absolute_time` converted through the Mach timebase. GPU command bounds use Metal's CPU-clock timestamps; GPU encoder counters are calibrated using `sampleTimestamps`. All sampled intervals were checked against their corresponding GPU command bounds, allowing 100 µs calibration tolerance. Encoder intervals can overlap; their sum is not wall time or exclusively occupied GPU time. Every independently captured process starts at zero in the charts. CPU burst-start ticks in the overview mark positions; their widths do not represent CPU durations, which are mostly subpixel at this scale.

Ours serializes trace JSON after the timed request. Splash's diagnostic completion callback resolves/writes its per-command trace before completion is consumed; that cost remains in its diagnostic wall measurement and is reflected in the overhead table. Normal medians have neither trace collection nor trace output.

All 18 requests returned first token 22; CPU and GPU capture dispatch counts match exactly for each configuration. The Release build passed with zero warnings/errors, and production `kernels.metal` is byte-identical to the earlier baseline. This is a first-token repeat check, not a full generation-quality evaluation. No thermal warning was sampled. Whole-process ours showed 702.625 MiB of new system swap-outs during loading/cold/early warm activity; the 3-second samples bracketing its first warm request include 346 MiB, so that request is not a clean memory-pressure control. Later sampled warm/trace windows show no further increments; the two later warm measurements are 3806.052 and 3807.254 ms. Both Splash processes showed zero new swap-outs. These machine-wide samples are not precise per-request attribution.

### Artifacts and reproduction

All local outputs are in `artifacts/dispatch-timeline/` (ignored generated evidence):

- `prefill-10k-overview.png` / `.svg`: normal-boundary CPU/GPU command timeline.
- `dispatch-and-kernels-zoom.png` / `.svg`: CPU dispatch encoding and the first 45 ms of sampled kernels, with separate clocks/captures explicitly labelled.
- `prefill-10k-kernels.png` / `.svg`: full sampled kernel timeline.
- `{ours-fp16,splash-int8,splash-bf16}-{cpu,sampled}.trace.json`: six Chrome trace-event files, importable into Perfetto or a compatible viewer for individual dispatch inspection.
- `commands.csv`, `summary.json`, per-case stdout/stderr/memory/raw JSONL, binary/source identities, source patches and `evidence-manifest.json`.

The source patch `scripts/splash-dispatch-timeline.patch` applies to Splash `1482f6bbc2abf12375332725302019594150b6cb`; it includes the earlier token-fixture/counter harness and the new host timeline hooks. The captured build uses a separate checkout `/tmp/splash-timeline-1482`. The standalone instrumented backend object is linked ahead of the existing pinned baseline `libsplash.a`, leaving the original checkout, profiling executable and installed metallib intact. Both objects/archive and resulting executable hashes are retained in provenance.

```sh
# From the engine worktree. Preserve artifacts/current-gap, which supplies
# the already-verified fixtures, commands and model paths.
/Users/littlecloud/.dotnet/dotnet build src/Bootstrap.Engine -c Release -o artifacts/timeline-bin

# In a clean checkout of the pinned Splash source, apply the saved patch.
# Build/link flags used for this capture (paths below are local prerequisites):
xcrun clang++ -std=c++20 -O3 -fobjc-arc -mmacosx-version-min=26.4 \
  -I/tmp/splash-timeline-1482/runtime -DSPLASH_BACKEND_INSTRUMENTATION=1 \
  -c /tmp/splash-timeline-1482/runtime/metal/MetalBackend.mm \
  -o artifacts/dispatch-timeline/MetalBackend.o
xcrun clang++ -std=c++20 -O3 -fobjc-arc -mmacosx-version-min=26.4 \
  -I/tmp/splash-timeline-1482/runtime -I/tmp/splash-timeline-1482/dev \
  /tmp/splash-timeline-1482/dev/benchmarks/decode_profile.mm \
  artifacts/dispatch-timeline/MetalBackend.o \
  /tmp/splash-profile-1482/build/engine/libsplash.a \
  -framework Foundation -framework Metal -framework IOKit -framework IOSurface \
  -o artifacts/dispatch-timeline/splash-profile

# Use a fresh result directory; capture is fixed to one 10K fixture,
# --runs 3, and --all-kernels. Do not enable BOOTSTRAP_PROFILE.
python3 scripts/profile-dispatch-timeline.py --out artifacts/dispatch-timeline-repeat
/tmp/qwen-prefill-plot-venv/bin/python scripts/render-dispatch-timeline.py artifacts/dispatch-timeline-repeat
```


## Follow-up: fused affine matmul retest

Retested the existing `dense_affine_pipeline` (K32) and `dense_affine_pipeline64` implementations at base `205c81e`, with additional diagnostic shapes covering the 30K fixture's 1259-row partial block and the GDN Z projection. No production shader, inference default, checkpoint or prefix policy changed. This tests the current TensorOps-based implementations; it is not a port of Splash's dense kernel and does not establish a limit on other fusion designs.

Both paths consume the same affine-8 group-64 weights, FP16 activations and FP32 outputs. The baseline timing includes weight expansion plus matmul. The fused path expands weights inside each matmul tile. Kernel tests use deterministic random inputs and non-dyadic affine scale/bias values, compare every live output, verify guards, and sample independent CPU dot products using the baseline's rounded FP16 weights. Both K32 and K64 pass all 48 diagnostic checks in ordinary mode and again with Metal API/GPU validation; all reported full-output errors are zero. These checks also include the existing grouped/shared/attention fixtures. Build: zero warnings/errors.

Five warm GPU-command measurements per path alternate order, excluding run zero:

| M / N / K | Unpack + matmul, ms | Fused K32, ms | K32 change | Fused K64, ms | K64 change vs its own baseline |
|---|---:|---:|---:|---:|---:|
| 2048 / 8192 / 2048 | 2.3128 | 3.0264 | +30.9% | 3.1373 | +35.1% |
| 2048 / 4096 / 2048 | 1.1570 | 1.5445 | +33.5% | 1.5786 | +36.6% |
| 2048 / 2048 / 4096 | 1.2306 | 1.5979 | +29.8% | 1.6163 | +32.7% |
| 2048 / 2048 / 512 | 0.1426 | 0.2428 | +70.2% | 0.2526 | +75.8% |
| 1259 / 8192 / 2048 | 1.5179 | 1.9024 | +25.3% | 1.9669 | +30.4% |
| 1259 / 2048 / 4096 | 0.7907 | 0.9480 | +19.9% | 1.0760 | +38.3% |
| 1259 / 2048 / 512 | 0.0941 | 0.1509 | +60.3% | 0.1636 | +73.8% |

K64 was run separately with its own alternating baseline; all baseline observations are retained in `micro.csv` and `summary.json`. The table's baseline column is the K32 process. Additional N=256/512 microbenchmarks also regress, but these are generic affine fixtures rather than an assertion that all such projections use affine weights or this kernel in the model.

### Same-process 30K comparison

The 29,939-token fixture is unchanged. Mask 29 is the current default. Mask 31 adds only bit 2, selecting K32 fused affine matmul for the single-projection path. Grouped projections, BF16 weight conversions, routed experts, draft behavior and the eight-token suffix remain unchanged. Both paths warm once, then run five times each in alternating order, clearing reusable prefixes before every request; no encoder counters or timeline collection is enabled.

| Variant | Dispatches | Median wall, ms | Tokens/s | Median GPU command time, ms |
|---|---:|---:|---:|---:|
| Current unpack + matmul (29) | 16344 | 12679.002 | 2361.3 | 12607.407 |
| Fused K32 single affine projections (31) | 14244 | 13403.450 | 2233.7 | 13334.857 |

Fusing removes **2100 dispatches (12.849%)**, but median request latency increases **5.714%**. Every pair regresses: 611.847–765.740 ms extra, median paired increase 721.451 ms. The median CPU encoding time is 8.463 ms versus 8.163 ms; the regression is in GPU work. Each path retains exactly 39,939,244,032 bytes of Metal buffers, because other operations still need the existing scratch storage. All 12 requests, including warmups, return first token 22. This is a repeatability check, not a full generation-quality evaluation.

This switch covers the **2100 individual affine expansions**, not the additional **750 grouped expansions** in the earlier 2850-dispatch estimate. No grouped fused-matmul implementation was added. The remaining 615 BF16-to-FP16 expansions are also untouched.

Code inspection explains a likely source of the tradeoff: pre-expansion decodes each weight once per projection/chunk and feeds the existing device-resident TensorOps core. The current fused kernels decode again for each 128-row output tile (16 tiles at M=2048), with threadgroup staging and barriers. That avoids a scratch write and one dispatch, but changes both decoding reuse and the matmul schedule. Hardware counters were not collected to separate those costs, so this is an implementation-based explanation rather than an isolated causal measurement.

Whole-machine new swap-outs total 641.5 MiB over model loading and initial warmup; no additional swap-outs appear after the last run-zero sample. All four micro/validation processes show zero new swap-outs. No thermal warnings were sampled. The first baseline warmup had 16444 dispatches because it also initialized 100 resident small weights; the reported warm runs have exactly 16344 versus 14244 dispatches.

**Decision:** retain mask 29. The tested fused implementations are correctness-valid on these fixtures but slower, so dispatch reduction alone does not justify enabling bit 2. K64 was rejected on isolated measurements; only K32 was measured end to end.

Local evidence: `artifacts/fused-dense-retest/` contains commands, binary/source hashes, logs, all observations, memory samples, `micro.csv`, `summary.json` and `evidence-manifest.json`. Production shader SHA-256 remains `e53803cf622ed53924948abfdb312547be985cc2d2da94e25027d49e77771812`. Reproduce with `dotnet build src/Bootstrap.Engine -c Release -o artifacts/fused-dense-bin`, then `python3 scripts/test-fused-dense.py` using a fresh output location (the script refuses to overwrite existing provenance). Summarize with `python3 scripts/summarize-fused-dense.py artifacts/fused-dense-retest`. The runner reuses the pinned model/fixture commands under `artifacts/current-gap/`; these local artifacts are prerequisites.


## Follow-up: precompute the shared-gate sigmoid

Tested at base `8fa82a7`, with an opt-in `--precompute-shared-gate` experiment. The sigmoid moves from `batch_mix` into the existing scalar shared-gate projection epilogue. Tiny projections and the predecoded TensorOps projection each have a separate candidate kernel. `batch_mix_precomputed_gate` reads the resulting FP32 scalar. Expert/shared outputs, residuals and gate storage remain FP32; accumulation remains residual + shared, then the eight routed experts in the original order. No extra dispatch or model buffer is introduced. Decode and speculative verification retain their existing behavior; unsupported large-prefill projection configurations use the original path. Default remains off.

### Correctness and isolated timing

The diagnostic covers rows 1, 8, 9, 17, 127, 272, 1259 and 2048, random full-permutation expert routing, saturated and ordinary gates, separate output and in-place residual output. All live outputs match the original projection + combine bitwise, guard regions survive, and sampled independent CPU combine checks have maximum absolute error 1.711e-7. All 16 checks pass normally and again under Metal API/GPU validation. Build has zero warnings/errors.

Nine warm observations per variant, alternating order after one excluded warmup. Values below are median GPU command-buffer times, in milliseconds. Projection + combine includes both dispatches with packed operands already resident; common input/weight preparation is excluded. The two-dispatch microsequence does not model the intervening shared down projection in the full engine.

| Rows | Combine baseline | Combine precomputed | Projection + combine baseline | Projection + combine precomputed |
|---|---:|---:|---:|---:|
| 127 | 0.022708 | 0.022500 | 0.049833 | 0.051833 |
| 272 | 0.052125 | 0.052125 | 0.088750 | 0.089750 |
| 1259 | 0.395750 | 0.395333 | 0.440875 | 0.441625 |
| 2048 | 0.643125 | 0.642875 | 0.691625 | 0.692458 |

At 2048 rows, combine changes by -0.039%; including the scalar projection changes by +0.120%. Thus eliminating repeated sigmoid arithmetic does not materially accelerate this combine implementation on the tested hardware. Memory traffic is a plausible remaining bottleneck, but this experiment does not measure hardware bandwidth or prove the cause of Splash's advantage.

### Same-process 30K comparison

The same 29,939-token fixture and tuned mask 29 are used, with prefixes cleared on every request. Each variant warms once, then five warm pairs alternate order. No encoder counters or timeline collection are enabled. All 12 requests, including warmups, have bitwise-identical full final-logit vectors and first token 22. This is one fixture's numerical check, not a generation-quality evaluation.

| Variant | Median wall, ms | Tokens/s | Median GPU command time, ms | Dispatches |
|---|---:|---:|---:|---:|
| Current baseline | 12689.005 | 2359.4 | 12628.808 | 16344 |
| Precomputed sigmoid | 12679.179 | 2361.3 | 12627.147 | 16344 |

Median wall latency decreases by **0.077%** (9.827 ms); median GPU time decreases only 1.661 ms out of approximately 12.63 seconds. All five paired wall differences favor the candidate by 4.876–46.769 ms, with a median paired saving of 27.263 ms. The difference of variant medians and median paired difference are distinct statistics; neither supports a substantial speedup claim. Both variants retain 39,939,244,032 bytes of Metal buffers. The first baseline warmup includes 100 resident-weight initialization dispatches (16444); subsequent requests use 16344. No new whole-machine swap-outs or thermal warnings were sampled, including loading.

**Decision:** keep this experiment opt-in. It preserves this fixture's output but does not explain the large previously observed Splash combine gap. The next more promising isolated experiment is BF16 expert-output storage, retaining FP32 accumulation: at 2048 tokens, the routed output footprint would fall from 128 MiB to 64 MiB. That precision change has not been implemented or validated by this experiment.

Local evidence is in `artifacts/shared-gate/`: command/environment identities, binary/source hashes, raw logs, machine samples, `micro.csv`, `summary.json`, and `evidence-manifest.json`. Reproduce with `dotnet build src/Bootstrap.Engine -c Release -o artifacts/shared-gate-bin`, then `python3 scripts/test-shared-gate.py` using a fresh output location, and summarize with `python3 scripts/summarize-shared-gate.py artifacts/shared-gate`. The runner reuses the pinned model and prompt configuration under `artifacts/current-gap/` and refuses to overwrite existing provenance.


## Follow-up: BF16 routed expert-output storage

Tested at base `e694176`. `--bf16-expert-output` selects a BF16 write epilogue for the existing affine expert-down matmul and a BF16-reading combine kernel. The affine-8 group-64 weights, FP16 operands, FP32 matmul accumulation, FP32 combine accumulation, and shared/residual storage stay unchanged. Combine still adds shared first, then the eight routed experts in their existing order. The sigmoid experiment remains disabled. Only non-verification prefill blocks with the fused affine path and more than eight tokens use BF16; decode, speculative verification, and the eight-token suffix keep their original FP32 output path.

The BF16 epilogue uses explicit valid-element stores because the installed Metal cooperative-tensor `store` requires its destination type to match the FP32 accumulator type. The production FP32 TensorOps store remains unchanged. A separate scalar FP32-store kernel is included solely as a diagnostic control.

At 2048 tokens, the active routed-output range falls from 128 MiB to 64 MiB. The benchmark reuses the original allocation for both modes, so **total allocated Metal bytes remain unchanged**. There is no conversion pass and no dispatch reduction. This experiment does not change the default; the measured optimization is available through the opt-in flag.

### Kernel correctness and timing

Six fixtures cover 32/127/272/1259/2048 tokens, 256 affine experts, uniform and skewed distributions, and explicit live-tile counts 1/2/7/8/9/16/17/31/32. Every BF16 output equals the CPU round-to-nearest-even conversion of the corresponding FP32 output bit for bit. Combine matches the original FP32 combine supplied with these rounded values bit for bit, for separate and in-place outputs. Guard regions, sampled independent CPU dots, repeated FP32 outputs, and equality between the scalar and bulk FP32 stores are checked. Local expert-output relative L2 rounding error is approximately 0.166%; combine error is approximately 0.035% on the large fixtures.

Seven warm observations per variant alternate order after one excluded warmup. Median GPU command-buffer times (ms), with input/weights already resident:

| Tokens / distribution | Down FP32 | Down BF16 | Combine FP32 | Combine BF16 | Down + combine FP32 | Down + combine BF16 |
|---|---:|---:|---:|---:|---:|---:|
| 127 / uniform | 1.0858 | 1.0512 | 0.0229 | 0.0147 | 1.1264 | 1.0689 |
| 272 / uniform | 1.1426 | 1.1233 | 0.0631 | 0.0301 | 1.2376 | 1.1659 |
| 1259 / uniform | 1.7493 | 1.6856 | 0.3995 | 0.2618 | 2.1331 | 1.9148 |
| 2048 / uniform | 2.2811 | 2.0556 | 0.6517 | 0.4312 | 2.8651 | 2.5065 |
| 2048 / skew | 2.2748 | 2.1324 | 0.6537 | 0.4300 | 2.9181 | 2.6232 |

At 2048 uniform tokens, combine latency falls 33.84%, while down + combine falls 12.52%. This isolates a useful output-storage optimization; the down epilogue also changes its store implementation, so its gain is not exclusively a measurement of bit width. These synthetic timings are not a new direct Splash benchmark.

### Validation qualification

Ordinary execution and Metal API-only validation pass all six down and twelve combine checks. With Metal GPU validation enabled, the **original FP32 bulk-store reference itself was nonrepeatable**: a diagnostic rerun found 476 differing elements on the tails fixture before invoking the BF16 kernel. Two initial failures and the repeated-baseline diagnostic are preserved under `validation-diagnostics/`. No confirmed driver/compiler root cause is claimed.

The BF16 candidate passes all six down and twelve combine checks with API + GPU validation against the scalar FP32-store reference. The scalar reference matches the original bulk-store output bitwise under ordinary/API-only execution. Thus candidate validation succeeded through an independently checked reference; the unmodified baseline bulk-store path did not pass the GPU-instrumented repeatability check. All builds complete with zero C# warnings/errors.

### Same-process 30K performance

The unchanged 29,939-token fixture, tuned mask 29, FP16 KV, and draft configuration are used. Both variants warm once, then run five pairs in alternating order with prefixes cleared. No kernel counters or timeline collection is enabled. The first batch showed a 1.689% median wall reduction, but machine monitoring recorded 91.313 MiB of new swap-outs in the sampled warm timing window, so a second batch was taken. The repeat is the primary result:

| Variant | Median wall, ms | Tokens/s | Median GPU command time, ms | Dispatches |
|---|---:|---:|---:|---:|
| FP32 output | 13154.760 | 2275.9 | 13054.126 | 16344 |
| BF16 output | 12978.803 | 2306.8 | 12846.640 | 16344 |

Repeat median wall latency improves **1.338%**, and throughput improves **1.356%**. All five pairs favor BF16 by 173.689–200.057 ms (median paired saving 188.742 ms). Both variants retain 39,939,244,032 bytes of Metal buffers. Across both batches, all ten warm pairs favor BF16, and all requests return first token 22; each variant's final logits repeat bitwise across its own requests.

In the repeat, machine samples show zero new swap-outs through the last timing observation. An 83.063 MiB increase is first recorded in samples showing subsequent untimed generation checks. Sampling is approximately every two seconds, so boundary intervals cannot be assigned exactly to individual requests. No thermal warnings were sampled. Whole-process memory observations, including loading and untimed checks, are preserved rather than discarded. Use the within-batch comparisons; absolute times should not be compared with earlier experiments run under different machine conditions.

### Accepting numerical differences

The user clarified that bounded numerical differences are acceptable; bitwise equivalence with FP32 is not a model-quality requirement. The kernel tests above enforce correct BF16 rounding, while model acceptance considers outputs and normalized probability distributions.

On the 30K fixture, BF16 changes raw final logits by relative L2 **8.002%**, maximum absolute error **1.06951**, cosine **0.996805**. This metric alone does not indicate a wrong answer. The full access code `73DY-DRV5-B8FJ` and the arithmetic smoke answer are identical, including generated token sequences, in both timing batches.

Six additional deterministic acceptance probes cover the 30K needle, multiplication, sorting, extraction, counting, and a corrected-record instruction. Both modes pass all six; complete generated token sequences and first-token argmax agree. First-token logits are captured before decode in a non-timed generation callback. Across these probes, maximum softmax KL is **8.252e-5 nats**, and maximum total variation distance is **0.0003274**. These are local acceptance probes, not a broad task-accuracy evaluation.

For 30K specifically, softmax KL is **0.00002982 nats**, total variation is **0.00008743**, and the baseline-selected first-token probability changes from **99.97426% to 99.98281%**. The observed numerical difference is acceptable on these tested tasks, and the candidate provides a small, consistent end-to-end speed gain.

Local evidence: `artifacts/expert-output/` contains both timing batches, ordinary/API/GPU checks, the qualification diagnostics, six-task quality outputs, source/binary hashes, memory observations, `micro.csv`, `summary.json`, and `evidence-manifest.json`. The extra quality diagnostics were built separately under `artifacts/expert-output-quality-bin`; both builds use identical inference shaders. Reproduce with `dotnet build src/Bootstrap.Engine -c Release -o artifacts/expert-output-bin`, then `python3 scripts/test-expert-output.py` in a fresh result location, and `python3 scripts/summarize-expert-output.py artifacts/expert-output`. The runner uses the pinned model/prompt configuration from `artifacts/current-gap/` and refuses to overwrite existing provenance.
